import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from surrogate3d.evaluation.inference import WholeCasePrediction
from surrogate3d.evaluation.field_force_eval import (
    ForceEvalConfig,
    run_force_evaluation,
)
from surrogate3d.physics.field_force import (
    ForceEvaluatorConfig,
    compute_force_coefficients_torch,
)
from surrogate3d.physics.wall_geometry import BlockWallGeometry


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


def _make_wall_mesh_coords() -> np.ndarray:
    x = np.arange(3, dtype=np.float64)
    z = np.arange(2, dtype=np.float64)
    y = np.array([0.0, -1.0], dtype=np.float64)
    xx, zz, yy = np.meshgrid(x, z, y, indexing="ij")
    return np.stack([xx, yy, zz], axis=-1)


def _make_volume_cgns(path: Path):
    with h5py.File(path, "w") as h5file:
        base = h5file.create_group("BASE#1")
        zone = base.create_group("domain.00001")
        zone.create_dataset(" data", data=np.array([[3, 2, 2], [2, 1, 1], [0, 0, 0]], dtype=np.int64))
        _write_text_node(zone, "ZoneType", "Structured")
        _write_coordinates(zone, _make_wall_mesh_coords())

        solution = zone.create_group("Flow solution")
        _write_text_node(solution, "GridLocation", "CellCenter")
        _write_data_node(solution, "Rind", np.array([1, 1, 1, 1, 1, 1], dtype=np.int32))
        raw = np.zeros((3, 3, 4), dtype=np.float32)
        for field in (
            "Density",
            "VelocityX",
            "VelocityY",
            "VelocityZ",
            "Pressure",
            "TurbulentSANuTilde",
        ):
            _write_data_node(solution, field, raw)


def _make_block_geometry() -> dict[int, BlockWallGeometry]:
    return {
        0: BlockWallGeometry(
            block_index=0,
            zone_name="domain.00001",
            face_centers=torch.tensor(
                [
                    [[0.5, 0.0, 0.5]],
                    [[1.5, 0.0, 0.5]],
                ],
                dtype=torch.float64,
            ),
            area_vectors=torch.tensor(
                [
                    [[0.0, -1.0, 0.0]],
                    [[0.0, -1.0, 0.0]],
                ],
                dtype=torch.float64,
            ),
            cell_areas=torch.ones((2, 1), dtype=torch.float64),
            cell_centers=torch.tensor(
                [
                    [[0.5, -0.5, 0.5]],
                    [[1.5, -0.5, 0.5]],
                ],
                dtype=torch.float64,
            ),
            wall_distance=torch.full((2, 1), 0.5, dtype=torch.float64),
            normal=torch.tensor(
                [
                    [[0.0, -1.0, 0.0]],
                    [[0.0, -1.0, 0.0]],
                ],
                dtype=torch.float64,
            ),
            tangent_i=torch.tensor(
                [
                    [[1.0, 0.0, 0.0]],
                    [[1.0, 0.0, 0.0]],
                ],
                dtype=torch.float64,
            ),
            tangent_j=torch.tensor(
                [
                    [[0.0, 0.0, 1.0]],
                    [[0.0, 0.0, 1.0]],
                ],
                dtype=torch.float64,
            ),
        )
    }


def _make_block_fields(*, pressure: float, velocity_x: float = 0.0) -> dict[int, torch.Tensor]:
    fields = torch.zeros((1, 6, 2, 1, 1), dtype=torch.float64)
    fields[:, 0, :, :, :] = 1.0
    fields[:, 1, :, :, :] = velocity_x
    fields[:, 4, :, :, :] = pressure
    return {0: fields}


def test_compute_force_coefficients_torch_matches_pressure_only_expectation():
    mach = 1.0
    gamma = 1.4
    q = 0.5 * gamma * mach * mach
    coeffs = compute_force_coefficients_torch(
        _make_block_fields(pressure=1.0 + q),
        _make_block_geometry(),
        torch.tensor([[mach, 0.0, 1.0e6]], dtype=torch.float64),
        area_ref=2.0,
        chord_ref=1.0,
        moment_center=(0.25, 0.0, 0.0),
        config=ForceEvaluatorConfig(mode="pressure_only", gamma=gamma),
    )

    assert coeffs["CL"].item() == pytest.approx(1.0)
    assert coeffs["CD"].item() == pytest.approx(0.0, abs=1.0e-12)
    assert coeffs["CMz"].item() == pytest.approx(0.75)
    np.testing.assert_allclose(
        coeffs["pressure_force_coeff"][0].detach().cpu().numpy(),
        np.array([0.0, 1.0, 0.0]),
    )


def test_compute_force_coefficients_torch_first_cell_viscous_proxy_produces_drag():
    mach = 1.0
    gamma = 1.4
    q = 0.5 * gamma * mach * mach
    mu_inf = mach * np.sqrt(gamma)
    expected_cf = (mu_inf / 0.5) / q
    coeffs = compute_force_coefficients_torch(
        _make_block_fields(pressure=1.0, velocity_x=1.0),
        _make_block_geometry(),
        torch.tensor([[mach, 0.0, 1.0]], dtype=torch.float64),
        area_ref=2.0,
        chord_ref=1.0,
        moment_center=(0.25, 0.0, 0.0),
        config=ForceEvaluatorConfig(mode="full", gamma=gamma, use_turbulent_viscosity=False),
    )

    assert coeffs["CL"].item() == pytest.approx(0.0, abs=1.0e-12)
    assert coeffs["CD"].item() == pytest.approx(expected_cf)
    np.testing.assert_allclose(
        coeffs["viscous_force_coeff"][0].detach().cpu().numpy(),
        np.array([expected_cf, 0.0, 0.0]),
    )


def test_run_force_evaluation_matches_reference_on_synthetic_prediction(tmp_path):
    source_case_dir = tmp_path / "sample0" / "cfd" / "cond_0"
    source_case_dir.mkdir(parents=True, exist_ok=True)
    volume_cgns = source_case_dir / "wing_000_vol.cgns"
    _make_volume_cgns(volume_cgns)
    (source_case_dir.parent / "input.json").write_text(
        json.dumps({"surface_area": 2.0, "x_ref": 0.25, "y_ref": 0.0, "z_ref": 0.0}),
        encoding="utf-8",
    )
    (source_case_dir / "conds.json").write_text(json.dumps({"AA": 0.0}), encoding="utf-8")
    (source_case_dir / "output.json").write_text(
        json.dumps({"results": {"wing_cl": 1.0, "wing_cd": 0.0, "wing_cmz": 0.75}}),
        encoding="utf-8",
    )

    block_fields = _make_block_fields(pressure=1.7)
    prediction = WholeCasePrediction(
        batch={
            "sample_id": ["sample0"],
            "cond_idx": torch.tensor([0], dtype=torch.long),
            "source_case_dir": [str(source_case_dir)],
            "cgns_basename": ["wing_000_vol.cgns"],
            "flow_conditions": torch.tensor([[1.0, 0.0, 1.0e6]], dtype=torch.float32),
            "blocks": {0: {"block_name": "domain.00001"}},
        },
        pred_model={},
        pred_physical=block_fields,
        target_model={},
        target_physical=block_fields,
        config={},
        checkpoint_payload={},
        case_index=0,
    )

    summary = run_force_evaluation(
        prediction,
        output_dir=tmp_path / "field_force_eval",
        config=ForceEvalConfig(mode="pressure_only"),
    )

    assert summary["force_eval_method"] == "force_torch"
    assert summary["force_torch"]["CL"] == pytest.approx(1.0)
    assert summary["force_torch"]["CD"] == pytest.approx(0.0, abs=1.0e-12)
    assert summary["force_torch"]["CMz"] == pytest.approx(0.75)
    assert summary["force_mode"] == "pressure_only"
    assert summary["is_complete_force"] is False
    assert summary["force_l1_cl_cd"] == pytest.approx(0.0, abs=1.0e-8)
    assert summary["wall_field_metrics"]["cp_mae"] == pytest.approx(0.0, abs=1.0e-12)
    assert summary["n_surface_cells"] == 2
    assert Path(summary["summary_path"]).exists()
