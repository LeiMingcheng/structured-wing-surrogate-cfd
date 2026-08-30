import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest

from surrogate3d.evaluation.residual_eval import (
    ResidualChannelMetricsConfig,
    compare_exported_residual_channels,
    run_restart_residual_evaluation,
)
from surrogate3d.evaluation.adflow_restart import ADflowRestartEvalConfig
from parametric_swept_wing_cfd.export.cgns_state import VOLUME_STATE_FIELDS


def _write_text_node(parent, name: str, text: str):
    group = parent.create_group(name)
    group.create_dataset(" data", data=np.frombuffer(text.encode("ascii"), dtype=np.int8))
    return group


def _write_data_node(parent, name: str, data):
    group = parent.create_group(name)
    group.create_dataset(" data", data=data)
    return group


def _write_coordinates(parent, coords_ijk: np.ndarray):
    coords = parent.create_group("GridCoordinates")
    raw = np.transpose(coords_ijk, tuple(reversed(range(coords_ijk.ndim - 1))) + (coords_ijk.ndim - 1,))
    _write_data_node(coords, "CoordinateX", raw[..., 0])
    _write_data_node(coords, "CoordinateY", raw[..., 1])
    _write_data_node(coords, "CoordinateZ", raw[..., 2])


def _make_node_coords(shape):
    grids = np.meshgrid(*[np.arange(n, dtype=np.float64) for n in shape], indexing="ij")
    return np.stack(grids, axis=-1)


def _raw_with_rind(cell_values: np.ndarray) -> np.ndarray:
    raw = np.zeros((3, 3, 4), dtype=np.float32)
    raw[1:-1, 1:-1, 1:-1] = np.transpose(cell_values.astype(np.float32), (2, 1, 0))
    return raw


def _make_volume_cgns(
    path: Path,
    *,
    res_density: tuple[float, float],
    res_turb: tuple[float, float],
):
    with h5py.File(path, "w") as h5file:
        base = h5file.create_group("BASE#1")
        zone = base.create_group("domain.00001")
        zone.create_dataset(" data", data=np.array([[3, 2, 2], [2, 1, 1], [0, 0, 0]], dtype=np.int64))
        _write_text_node(zone, "ZoneType", "Structured")
        _write_coordinates(zone, _make_node_coords((3, 2, 2)))

        solution = zone.create_group("Flow solution")
        _write_text_node(solution, "GridLocation", "CellCenter")
        _write_data_node(solution, "Rind", np.array([1, 1, 1, 1, 1, 1], dtype=np.int32))

        base_values = np.array([[[10.0]], [[20.0]]], dtype=np.float32)
        for idx, field in enumerate(VOLUME_STATE_FIELDS):
            _write_data_node(solution, field, _raw_with_rind(base_values + idx))
        _write_data_node(
            solution,
            "ResDensity",
            _raw_with_rind(np.array([[[res_density[0]]], [[res_density[1]]]], dtype=np.float32)),
        )
        _write_data_node(
            solution,
            "ResTurbulentSANuTilde",
            _raw_with_rind(np.array([[[res_turb[0]]], [[res_turb[1]]]], dtype=np.float32)),
        )


def _write_source_case(root: Path) -> Path:
    source_case_dir = root / "sample0" / "cfd" / "cond_0"
    source_case_dir.mkdir(parents=True, exist_ok=True)
    _make_volume_cgns(
        source_case_dir / "wing_000_vol.cgns",
        res_density=(10.0, 20.0),
        res_turb=(3.0, 7.0),
    )
    (source_case_dir.parent / "input.json").write_text("{}", encoding="utf-8")
    (source_case_dir / "conds.json").write_text("{}", encoding="utf-8")
    (source_case_dir / "output.json").write_text(json.dumps({"results": {}}), encoding="utf-8")
    return source_case_dir


def test_compare_exported_residual_channels_reports_metrics(tmp_path):
    reference_cgns = tmp_path / "reference.cgns"
    candidate_cgns = tmp_path / "candidate.cgns"
    _make_volume_cgns(reference_cgns, res_density=(10.0, 20.0), res_turb=(3.0, 7.0))
    _make_volume_cgns(candidate_cgns, res_density=(12.0, 18.0), res_turb=(1.0, 9.0))

    summary = compare_exported_residual_channels(
        reference_volume_cgns=reference_cgns,
        candidate_volume_cgns=candidate_cgns,
        output_dir=tmp_path / "compare",
        metrics_config=ResidualChannelMetricsConfig(near_wall_layers=1),
        metadata={"case_index": 0},
    )

    density_metrics = summary["global_metrics"]["fields"]["ResDensity"]
    turb_metrics = summary["global_metrics"]["fields"]["ResTurbulentSANuTilde"]
    assert summary["residual_fields"] == ["ResDensity", "ResTurbulentSANuTilde"]
    assert density_metrics["mae"] == pytest.approx(2.0)
    assert density_metrics["rmse"] == pytest.approx(2.0)
    assert density_metrics["volume_weighted_rmse"] == pytest.approx(2.0)
    assert turb_metrics["mae"] == pytest.approx(2.0)
    assert summary["near_wall_n_cells"] == 2
    assert summary["blocks"][0]["block_name"] == "domain.00001"
    assert Path(summary["summary_path"]).exists()


def test_run_restart_residual_evaluation_writes_summary_with_fake_adflow(tmp_path, monkeypatch):
    source_case_dir = _write_source_case(tmp_path)
    restart_cgns = tmp_path / "surrogate_restart_vol.cgns"
    restart_cgns.write_text("restart", encoding="utf-8")
    run_dir = tmp_path / "residual_eval_run"

    def _fake_run_subprocess(cmd, cwd, env, log_file=None, timeout=None):
        assert "--restart-file" in cmd
        assert str(restart_cgns.resolve()) in cmd
        assert "--no-volume-solution" not in cmd
        assert cmd[cmd.index("--solution-precision") + 1] == "double"
        assert "--refresh-residual-before-write" in cmd
        assert "--write-full-flow-residual" in cmd
        _make_volume_cgns(
            Path(cwd) / "wing_000_vol.cgns",
            res_density=(12.0, 18.0),
            res_turb=(1.0, 9.0),
        )
        (Path(cwd) / "output.json").write_text(
            json.dumps(
                {
                    "results": {"wing_cl": 0.2, "wing_cd": 0.01, "wing_cmz": 0.03},
                    "working_conditions": {"M3": 0.7, "AA": 2.0},
                    "solver_call_protocol": {
                        "cycle_protocol": "resume",
                        "requested_cycles": 1,
                        "solver_calls": 1,
                        "nCycles_per_call": 1,
                    },
                }
            ),
            encoding="utf-8",
        )
        if log_file is not None:
            Path(log_file).write_text("fake driver log", encoding="utf-8")
        (Path(cwd) / "adflow.log").write_text("fake adflow log", encoding="utf-8")
        return True, "", 0

    monkeypatch.setattr("surrogate3d.evaluation.residual_eval.run_subprocess", _fake_run_subprocess)
    summary = run_restart_residual_evaluation(
        restart_cgns=restart_cgns,
        source_case_dir=source_case_dir,
        output_dir=run_dir,
        config=ADflowRestartEvalConfig(
            mpi_ranks=4,
            python_executable=sys.executable,
            cycle_protocol="resume",
            cycles=1,
            inner_cycles=1,
            write_volume_solution=False,
            solution_precision="double",
            refresh_residual_before_write=True,
            write_full_flow_residual=True,
        ),
        metrics_config=ResidualChannelMetricsConfig(near_wall_layers=1),
        metadata={"case_index": 0},
    )

    candidate_key = "surrogate_one_step_exported_residual"
    assert summary["residual_eval_method"] == "one_step_exported_residual_proxy"
    assert summary["residual_eval_candidate_key"] == candidate_key
    assert summary["adflow_run"]["success"] is True
    assert summary["residual_channel_global_rmse_mean"] == pytest.approx(2.0)
    assert Path(summary["residual_channel_summary_json"]).exists()
    assert summary[candidate_key]["global_metrics"]["fields"]["ResDensity"]["rmse"] == pytest.approx(2.0)
    on_disk = json.loads(Path(summary["summary_path"]).read_text(encoding="utf-8"))
    assert on_disk["metadata"]["case_index"] == 0
