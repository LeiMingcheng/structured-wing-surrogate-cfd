import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from surrogate3d.evaluation.surface_force_eval import run_surface_force_evaluation
from surrogate3d.physics.surface_forces import (
    compute_surface_field_metrics,
    compute_surface_force_coefficients_torch,
    prepare_surface_force_geometry,
)


def _write_text_node(parent, name: str, text: str):
    group = parent.create_group(name)
    group.create_dataset(" data", data=np.frombuffer(text.encode("ascii"), dtype=np.int8))
    return group


def _write_data_node(parent, name: str, data):
    group = parent.create_group(name)
    group.create_dataset(" data", data=data)
    return group


def _write_coordinates(parent, coords_ij: np.ndarray):
    coords = parent.create_group("GridCoordinates")
    raw = np.transpose(coords_ij, tuple(reversed(range(coords_ij.ndim - 1))) + (coords_ij.ndim - 1,))
    _write_data_node(coords, "CoordinateX", raw[..., 0])
    _write_data_node(coords, "CoordinateY", raw[..., 1])
    _write_data_node(coords, "CoordinateZ", raw[..., 2])


def _raw_with_rind(cell_values: np.ndarray) -> np.ndarray:
    raw = np.zeros((3, 4), dtype=np.float32)
    raw[1:-1, 1:-1] = np.transpose(cell_values.astype(np.float32), (1, 0))
    return raw


def _make_surface_case(root: Path) -> Path:
    source_case_dir = root / "sample0" / "cfd" / "cond_0"
    source_case_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(source_case_dir / "wing_000_surf.cgns", "w") as h5file:
        base = h5file.create_group("BaseSurfaceSol")
        zone = base.create_group("NSWallAdiabaticBCZone1")
        zone.create_dataset(" data", data=np.array([[3, 2], [2, 1], [0, 0]], dtype=np.int64))
        _write_text_node(zone, "ZoneType", "Structured")
        coords_ij = np.array(
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                [[1.0, 0.0, 0.0], [1.0, 0.0, 1.0]],
                [[2.0, 0.0, 0.0], [2.0, 0.0, 1.0]],
            ],
            dtype=np.float64,
        )
        _write_coordinates(zone, coords_ij)

        solution = zone.create_group("Flow solution")
        _write_text_node(solution, "GridLocation", "CellCenter")
        _write_data_node(solution, "Rind", np.array([1, 1, 1, 1], dtype=np.int32))
        cp = np.array([[1.0], [1.0]], dtype=np.float32)
        cfx = np.array([[0.2], [0.2]], dtype=np.float32)
        zeros = np.zeros((2, 1), dtype=np.float32)
        _write_data_node(solution, "CoefPressure", _raw_with_rind(cp))
        _write_data_node(solution, "SkinFrictionX", _raw_with_rind(cfx))
        _write_data_node(solution, "SkinFrictionY", _raw_with_rind(zeros))
        _write_data_node(solution, "SkinFrictionZ", _raw_with_rind(zeros))

    (source_case_dir.parent / "input.json").write_text(
        json.dumps({"surface_area": 2.0, "x_ref": 0.25, "y_ref": 0.0, "z_ref": 0.0}),
        encoding="utf-8",
    )
    (source_case_dir / "conds.json").write_text(json.dumps({"AA": 0.0}), encoding="utf-8")
    (source_case_dir / "output.json").write_text(
        json.dumps({"results": {"wing_cl": 1.0, "wing_cd": 0.2, "wing_cmz": 0.75}}),
        encoding="utf-8",
    )
    return source_case_dir


def test_compute_surface_force_coefficients_torch_matches_expected_values():
    geometry = prepare_surface_force_geometry(
        centers=torch.tensor([[0.5, 0.0, 0.5], [1.5, 0.0, 0.5]], dtype=torch.float32),
        area_vectors=torch.tensor([[0.0, -1.0, 0.0], [0.0, -1.0, 0.0]], dtype=torch.float32),
        cell_areas=torch.tensor([1.0, 1.0], dtype=torch.float32),
    )
    surface_fields = torch.tensor(
        [
            [1.0, 0.2, 0.0, 0.0],
            [1.0, 0.2, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    coeffs = compute_surface_force_coefficients_torch(
        surface_fields,
        geometry,
        aoa_deg=0.0,
        area_ref=2.0,
        chord_ref=1.0,
        moment_center=(0.25, 0.0, 0.0),
        cp_index=0,
        cf_indices=(1, 2, 3),
        compute_viscous=True,
    )

    assert coeffs["CL"].item() == pytest.approx(1.0)
    assert coeffs["CD"].item() == pytest.approx(0.2)
    assert coeffs["CMz"].item() == pytest.approx(0.75)


def test_compute_surface_field_metrics_zero_for_identical_inputs():
    fields = torch.tensor([[1.0, 0.2, 0.0, 0.0], [1.0, 0.2, 0.0, 0.0]], dtype=torch.float32)
    metrics = compute_surface_field_metrics(fields, fields, cp_index=0, cf_indices=(1, 2, 3))
    assert metrics["cp_mae"] == pytest.approx(0.0)
    assert metrics["cfx_rmse"] == pytest.approx(0.0)
    assert metrics["cfy_mae"] == pytest.approx(0.0)
    assert metrics["cfz_rmse"] == pytest.approx(0.0)


def test_run_surface_force_evaluation_matches_reference_on_synthetic_case(tmp_path):
    source_case_dir = _make_surface_case(tmp_path)
    summary = run_surface_force_evaluation(
        source_case_dir=source_case_dir,
        output_dir=tmp_path / "surface_force_eval",
    )

    assert summary["force_eval_method"] == "surface_cp_cf_torch"
    assert summary["surface_cp_cf_torch"]["CL"] == pytest.approx(1.0)
    assert summary["surface_cp_cf_torch"]["CD"] == pytest.approx(0.2)
    assert summary["surface_cp_cf_torch"]["CMz"] == pytest.approx(0.75)
    assert summary["force_l1_cl_cd"] == pytest.approx(0.0, abs=1.0e-8)
    assert Path(summary["summary_path"]).exists()
