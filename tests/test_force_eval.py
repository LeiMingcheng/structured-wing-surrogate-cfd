import json
from pathlib import Path
import sys

import pytest

from surrogate3d.evaluation.force_eval import (
    ADflowForceEvalConfig,
    build_adflow_force_eval_command,
    compute_force_error_metrics,
    run_restart_force_evaluation,
)


def _write_source_case(root: Path) -> Path:
    source_case_dir = root / "sample0" / "cfd" / "cond_0"
    source_case_dir.mkdir(parents=True, exist_ok=True)
    (source_case_dir / "wing_000_vol.cgns").write_text("grid", encoding="utf-8")
    (source_case_dir.parent / "input.json").write_text("{}", encoding="utf-8")
    (source_case_dir / "conds.json").write_text("{}", encoding="utf-8")
    (source_case_dir / "output.json").write_text(
        json.dumps(
            {
                "results": {
                    "wing_cl": 0.2,
                    "wing_cd": 0.01,
                    "wing_cmz": 0.03,
                }
            }
        ),
        encoding="utf-8",
    )
    return source_case_dir


def test_build_adflow_force_eval_command_resume1_defaults(tmp_path):
    source_case_dir = _write_source_case(tmp_path)
    restart_cgns = tmp_path / "restart.cgns"
    restart_cgns.write_text("restart", encoding="utf-8")
    config = ADflowForceEvalConfig(
        mpi_ranks=8,
        python_executable=sys.executable,
        cycle_protocol="resume",
        cycles=1,
        inner_cycles=1,
        solver_preset="nk",
    )

    command = build_adflow_force_eval_command(
        source_case_dir=source_case_dir,
        restart_cgns=restart_cgns,
        config=config,
    )

    assert command[0:3] == ["mpirun", "-np", "8"]
    assert command[3] == str(Path(sys.executable).resolve())
    assert command[4].endswith("surrogate3d/evaluation/force_eval.py")
    assert "--restart" in command
    assert str(restart_cgns.resolve()) in command


def test_compute_force_error_metrics_matches_expected_values():
    metrics = compute_force_error_metrics(
        {"CL": 0.2, "CD": 0.01, "CMz": 0.03},
        {"CL": 0.21, "CD": 0.012, "CMz": 0.028},
    )

    assert metrics["dCL"] == pytest.approx(0.01)
    assert metrics["dCD"] == pytest.approx(0.002)
    assert metrics["dCMz"] == pytest.approx(-0.002)
    assert metrics["abs_dCL"] == pytest.approx(0.01)
    assert metrics["rel_CL"] == pytest.approx(0.05)
    assert metrics["force_l1_cl_cd"] == pytest.approx(0.012)


def test_run_restart_force_evaluation_writes_summary_with_fake_adflow(tmp_path, monkeypatch):
    source_case_dir = _write_source_case(tmp_path)
    restart_cgns = tmp_path / "surrogate_restart_vol.cgns"
    restart_cgns.write_text("restart", encoding="utf-8")
    run_dir = tmp_path / "force_eval_run"

    def _fake_run_subprocess(cmd, cwd, env, log_file=None, timeout=None):
        assert "--restart" in cmd
        assert str(restart_cgns.resolve()) in cmd
        output_json = Path(cwd) / "output.json"
        output_json.write_text(
            json.dumps(
                {
                    "results": {
                        "wing_cl": 0.202,
                        "wing_cd": 0.011,
                        "wing_cmz": 0.029,
                        "err_cl": 0.001,
                    },
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

    monkeypatch.setattr("surrogate3d.evaluation.force_eval.run_subprocess", _fake_run_subprocess)
    summary = run_restart_force_evaluation(
        restart_cgns=restart_cgns,
        source_case_dir=source_case_dir,
        output_dir=run_dir,
        config=ADflowForceEvalConfig(
            mpi_ranks=4,
            python_executable=sys.executable,
            cycle_protocol="resume",
            cycles=1,
            inner_cycles=1,
        ),
        metadata={"case_index": 0},
    )

    assert summary["force_eval_method"] == "adflow_pure_force_eval0"
    assert summary["force_eval_candidate_key"] == "surrogate_adflow_force_eval"
    assert summary["reference"] == {"CL": 0.2, "CD": 0.01, "CMz": 0.03}
    assert summary["surrogate_adflow_force_eval"] == {
        "CL": 0.202,
        "CD": 0.011,
        "CMz": 0.029,
    }
    assert summary["abs_dCL"] == pytest.approx(0.002)
    assert summary["adflow_run"]["success"] is True
    assert Path(summary["summary_path"]).exists()
    on_disk = json.loads(Path(summary["summary_path"]).read_text(encoding="utf-8"))
    assert on_disk["force_eval_candidate_key"] == "surrogate_adflow_force_eval"
    assert on_disk["metadata"]["case_index"] == 0
