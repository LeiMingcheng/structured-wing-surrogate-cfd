from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from surrogate3d.physics.block_geometry import (
    extract_block_geometry,
    extract_case_block_geometry_from_volume_cgns,
)
from surrogate3d.physics.residual_sample import WholeCaseResidualSampleBuilder


def _orthogonal_nodes(shape: tuple[int, int, int]) -> np.ndarray:
    i, j, k = np.meshgrid(
        np.arange(shape[0], dtype=np.float64),
        np.arange(shape[1], dtype=np.float64),
        np.arange(shape[2], dtype=np.float64),
        indexing="ij",
    )
    return np.stack((i, k, j), axis=-1)


def _write_data_node(parent, name: str, data: np.ndarray) -> None:
    group = parent.create_group(name)
    group.create_dataset(" data", data=data)


def _write_mesh_cgns(
    path: Path,
    nodes: np.ndarray,
    *,
    turbulent_distance: np.ndarray | None = None,
) -> None:
    with h5py.File(path, "w") as h5file:
        base = h5file.create_group("BASE#1")
        zone = base.create_group("domain.00001")
        coords = zone.create_group("GridCoordinates")
        raw = np.transpose(nodes, (2, 1, 0, 3))
        _write_data_node(coords, "CoordinateX", raw[..., 0])
        _write_data_node(coords, "CoordinateY", raw[..., 1])
        _write_data_node(coords, "CoordinateZ", raw[..., 2])
        if turbulent_distance is not None:
            solution = zone.create_group("Flow solution")
            raw_distance = np.transpose(turbulent_distance, (2, 1, 0))
            _write_data_node(
                solution,
                "TurbulentDistance",
                raw_distance,
            )


def _cell_area_closure(geometry) -> torch.Tensor:
    return (
        geometry.i_face_area_vectors[1:] - geometry.i_face_area_vectors[:-1]
        + geometry.j_face_area_vectors[:, 1:] - geometry.j_face_area_vectors[:, :-1]
        + geometry.k_face_area_vectors[:, :, 1:] - geometry.k_face_area_vectors[:, :, :-1]
    )


def test_block_geometry_has_oriented_closed_faces_and_wall_distance():
    geometry = extract_block_geometry(
        _orthogonal_nodes((4, 4, 4)),
        block_index=0,
        zone_name="domain.00001",
    )

    assert geometry.cell_volumes.shape == (3, 3, 3)
    assert geometry.i_face_area_vectors.shape == (4, 3, 3, 3)
    assert geometry.j_face_area_vectors.shape == (3, 4, 3, 3)
    assert geometry.k_face_area_vectors.shape == (3, 3, 4, 3)
    torch.testing.assert_close(geometry.cell_volumes, torch.ones_like(geometry.cell_volumes))
    torch.testing.assert_close(
        _cell_area_closure(geometry),
        torch.zeros((3, 3, 3, 3), dtype=torch.float64),
    )
    torch.testing.assert_close(
        geometry.wall_distance[0, 0],
        torch.tensor([0.5, 1.5, 2.5], dtype=torch.float64),
    )
    assert geometry.wall_distance_source == "kmin_projection"


def test_affine_block_geometry_preserves_volume_and_face_closure():
    nodes = _orthogonal_nodes((4, 4, 4))
    transform = np.array(
        [
            [1.0, 0.2, 0.0],
            [0.1, 1.0, 0.15],
            [0.0, 0.1, 1.0],
        ],
        dtype=np.float64,
    )
    geometry = extract_block_geometry(
        nodes @ transform.T,
        block_index=0,
        zone_name="domain.00001",
    )

    expected_volume = abs(float(np.linalg.det(transform)))
    torch.testing.assert_close(
        geometry.cell_volumes,
        torch.full_like(geometry.cell_volumes, expected_volume),
    )
    torch.testing.assert_close(
        _cell_area_closure(geometry),
        torch.zeros((3, 3, 3, 3), dtype=torch.float64),
        atol=1.0e-14,
        rtol=1.0e-14,
    )


def test_residual_sample_builder_maps_whole_case_batch(tmp_path):
    mesh_path = tmp_path / "wing_000_vol.cgns"
    _write_mesh_cgns(mesh_path, _orthogonal_nodes((4, 4, 4)))
    geometry = extract_case_block_geometry_from_volume_cgns(mesh_path)[0]
    coords_pde = geometry.cell_centers.permute(3, 0, 1, 2).unsqueeze(0).float()
    fields = {0: torch.zeros((1, 6, 3, 3, 3), dtype=torch.float32)}
    batch = {
        "sample_id": ["sample0"],
        "cond_idx": torch.tensor([0]),
        "source_case_dir": [str(tmp_path)],
        "cgns_basename": [mesh_path.name],
        "flow_conditions": torch.tensor([[0.7, 2.0, 1.0]], dtype=torch.float32),
        "blocks": {
            0: {
                "block_name": "domain.00001",
                "coords_pde": coords_pde,
                "cell_volumes": geometry.cell_volumes.unsqueeze(0).float(),
            }
        },
    }

    sample = WholeCaseResidualSampleBuilder().build_case_sample(
        fields,
        batch,
        batch_index=0,
    )

    block = sample.blocks[0]
    assert block.fields.shape == (1, 6, 3, 3, 3)
    assert block.coords_center.shape == (1, 3, 3, 3, 3)
    assert block.coords_vertex.shape == (1, 3, 4, 4, 4)
    assert block.cell_volumes.shape == (1, 3, 3, 3)
    assert block.wall_distance.shape == (1, 3, 3, 3)
    assert block.wall_distance[0, 0, 0, 0].item() == pytest.approx(0.5)
    assert sample.metadata["volume_cgns"] == str(mesh_path)


def test_cgns_turbulent_distance_replaces_kmin_projection(tmp_path):
    mesh_path = tmp_path / "wing_000_vol.cgns"
    exact_distance = np.full((3, 3, 3), 7.5, dtype=np.float64)
    _write_mesh_cgns(
        mesh_path,
        _orthogonal_nodes((4, 4, 4)),
        turbulent_distance=exact_distance,
    )

    geometry = extract_case_block_geometry_from_volume_cgns(mesh_path)[0]

    torch.testing.assert_close(
        geometry.wall_distance,
        torch.full((3, 3, 3), 7.5, dtype=torch.float64),
    )
    assert geometry.wall_distance_source == "adflow_turbulent_distance"
