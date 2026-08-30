"""Structured 3D block geometry extraction for surrogate residual operators."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import torch

from parametric_swept_wing_cfd.export.cgns_state import (
    CGNS_DATA,
    _compute_cell_volumes_adflow,
    _find_solution_group,
    _rind,
    _to_ijk,
    _trim_rind,
)


def _node_data(group) -> np.ndarray:
    if isinstance(group, h5py.Dataset):
        return np.asarray(group)
    if CGNS_DATA not in group:
        raise KeyError(f"CGNS node has no '{CGNS_DATA}' dataset: {group.name}")
    return np.asarray(group[CGNS_DATA])


def _iter_volume_zones(h5file: h5py.File) -> list[tuple[str, Any]]:
    zones: list[tuple[str, Any]] = []
    for base_group in h5file.values():
        if not hasattr(base_group, "items"):
            continue
        for zone_name, zone_group in base_group.items():
            if not hasattr(zone_group, "keys") or "GridCoordinates" not in zone_group:
                continue
            zones.append((str(zone_name), zone_group))
    return zones


def _read_coordinates(zone_group, *, dtype: np.dtype) -> np.ndarray:
    coords = zone_group["GridCoordinates"]
    raw = np.stack(
        [
            np.asarray(_node_data(coords["CoordinateX"])),
            np.asarray(_node_data(coords["CoordinateY"])),
            np.asarray(_node_data(coords["CoordinateZ"])),
        ],
        axis=-1,
    )
    if raw.ndim != 4:
        raise ValueError(f"Expected 3D node coordinates, got shape {raw.shape} in {zone_group.name}")
    axes = (2, 1, 0, 3)
    return np.ascontiguousarray(np.transpose(raw, axes)).astype(dtype, copy=False)


def _cell_centers(node_coords: np.ndarray) -> np.ndarray:
    return 0.125 * (
        node_coords[:-1, :-1, :-1, :]
        + node_coords[1:, :-1, :-1, :]
        + node_coords[:-1, 1:, :-1, :]
        + node_coords[1:, 1:, :-1, :]
        + node_coords[:-1, :-1, 1:, :]
        + node_coords[1:, :-1, 1:, :]
        + node_coords[:-1, 1:, 1:, :]
        + node_coords[1:, 1:, 1:, :]
    )


def _normalize(vectors: np.ndarray, *, eps: float = 1.0e-30) -> np.ndarray:
    norm = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norm, eps)


def _quad_area_vectors(
    p00: np.ndarray,
    p10: np.ndarray,
    p01: np.ndarray,
    p11: np.ndarray,
) -> np.ndarray:
    tri0 = np.cross(p10 - p00, p01 - p00)
    tri1 = np.cross(p11 - p01, p11 - p10)
    return 0.5 * (tri0 + tri1)


def _face_metrics(node_coords: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    if axis == 0:
        p00 = node_coords[:, :-1, :-1, :]
        p10 = node_coords[:, 1:, :-1, :]
        p01 = node_coords[:, :-1, 1:, :]
        p11 = node_coords[:, 1:, 1:, :]
    elif axis == 1:
        p00 = node_coords[:-1, :, :-1, :]
        p10 = node_coords[:-1, :, 1:, :]
        p01 = node_coords[1:, :, :-1, :]
        p11 = node_coords[1:, :, 1:, :]
    elif axis == 2:
        p00 = node_coords[:-1, :-1, :, :]
        p10 = node_coords[1:, :-1, :, :]
        p01 = node_coords[:-1, 1:, :, :]
        p11 = node_coords[1:, 1:, :, :]
    else:
        raise ValueError(f"axis must be 0, 1, or 2, got {axis}")
    face_centers = 0.25 * (p00 + p10 + p01 + p11)
    area_vectors = _quad_area_vectors(p00, p10, p01, p11)
    return face_centers, area_vectors


def _orient_face_area_vectors(
    face_centers: np.ndarray,
    area_vectors: np.ndarray,
    cell_centers: np.ndarray,
    axis: int,
) -> np.ndarray:
    oriented = np.array(area_vectors, copy=True)
    directions = np.zeros_like(oriented)
    ni, nj, nk = cell_centers.shape[:3]

    if axis == 0:
        directions[1:ni, :, :, :] = cell_centers[1:, :, :, :] - cell_centers[:-1, :, :, :]
        directions[0, :, :, :] = cell_centers[0, :, :, :] - face_centers[0, :, :, :]
        directions[ni, :, :, :] = face_centers[ni, :, :, :] - cell_centers[ni - 1, :, :, :]
    elif axis == 1:
        directions[:, 1:nj, :, :] = cell_centers[:, 1:, :, :] - cell_centers[:, :-1, :, :]
        directions[:, 0, :, :] = cell_centers[:, 0, :, :] - face_centers[:, 0, :, :]
        directions[:, nj, :, :] = face_centers[:, nj, :, :] - cell_centers[:, nj - 1, :, :]
    elif axis == 2:
        directions[:, :, 1:nk, :] = cell_centers[:, :, 1:, :] - cell_centers[:, :, :-1, :]
        directions[:, :, 0, :] = cell_centers[:, :, 0, :] - face_centers[:, :, 0, :]
        directions[:, :, nk, :] = face_centers[:, :, nk, :] - cell_centers[:, :, nk - 1, :]
    else:
        raise ValueError(f"axis must be 0, 1, or 2, got {axis}")

    flip_mask = np.sum(oriented * directions, axis=-1, dtype=np.float64) < 0.0
    oriented = np.where(flip_mask[..., None], -oriented, oriented)
    return oriented


def _wall_tangents(
    node_coords: np.ndarray,
    wall_normals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    face_p00 = node_coords[:-1, :-1, 0, :]
    face_p10 = node_coords[1:, :-1, 0, :]
    face_p01 = node_coords[:-1, 1:, 0, :]
    face_p11 = node_coords[1:, 1:, 0, :]

    tangent_i = 0.5 * ((face_p10 - face_p00) + (face_p11 - face_p01))
    tangent_i = _normalize(tangent_i)

    tangent_j_raw = 0.5 * ((face_p01 - face_p00) + (face_p11 - face_p10))
    tangent_j = tangent_j_raw - np.sum(tangent_j_raw * wall_normals, axis=-1, keepdims=True) * wall_normals
    tangent_j = tangent_j - np.sum(tangent_j * tangent_i, axis=-1, keepdims=True) * tangent_i
    tangent_j = _normalize(tangent_j)
    return tangent_i, tangent_j


def _wall_distance_from_kmin(
    cell_centers: np.ndarray,
    wall_face_centers: np.ndarray,
    wall_normals: np.ndarray,
) -> np.ndarray:
    delta = cell_centers - wall_face_centers[:, :, None, :]
    return np.abs(np.sum(delta * wall_normals[:, :, None, :], axis=-1, dtype=np.float64))


@dataclass(frozen=True)
class StructuredBlockGeometry:
    """Structured per-block geometry tensors aligned to surrogate3d block order."""

    block_index: int
    zone_name: str
    vertex_coords: torch.Tensor
    cell_centers: torch.Tensor
    cell_volumes: torch.Tensor
    i_face_centers: torch.Tensor
    j_face_centers: torch.Tensor
    k_face_centers: torch.Tensor
    i_face_area_vectors: torch.Tensor
    j_face_area_vectors: torch.Tensor
    k_face_area_vectors: torch.Tensor
    wall_face_centers: torch.Tensor
    wall_face_area_vectors: torch.Tensor
    wall_face_normals: torch.Tensor
    wall_tangent_i: torch.Tensor
    wall_tangent_j: torch.Tensor
    wall_distance: torch.Tensor
    wall_distance_source: str

    def to(
        self,
        device: str | torch.device,
        *,
        dtype: torch.dtype | None = None,
    ) -> "StructuredBlockGeometry":
        torch_device = torch.device(device)

        def _move(value: torch.Tensor) -> torch.Tensor:
            return value.to(device=torch_device, dtype=dtype or value.dtype)

        return StructuredBlockGeometry(
            block_index=int(self.block_index),
            zone_name=str(self.zone_name),
            vertex_coords=_move(self.vertex_coords),
            cell_centers=_move(self.cell_centers),
            cell_volumes=_move(self.cell_volumes),
            i_face_centers=_move(self.i_face_centers),
            j_face_centers=_move(self.j_face_centers),
            k_face_centers=_move(self.k_face_centers),
            i_face_area_vectors=_move(self.i_face_area_vectors),
            j_face_area_vectors=_move(self.j_face_area_vectors),
            k_face_area_vectors=_move(self.k_face_area_vectors),
            wall_face_centers=_move(self.wall_face_centers),
            wall_face_area_vectors=_move(self.wall_face_area_vectors),
            wall_face_normals=_move(self.wall_face_normals),
            wall_tangent_i=_move(self.wall_tangent_i),
            wall_tangent_j=_move(self.wall_tangent_j),
            wall_distance=_move(self.wall_distance),
            wall_distance_source=str(self.wall_distance_source),
        )


def extract_block_geometry(
    node_coords: np.ndarray,
    *,
    block_index: int,
    zone_name: str,
    dtype: torch.dtype = torch.float64,
) -> StructuredBlockGeometry:
    """Extract structured 3D control-volume geometry from one CGNS block."""
    if node_coords.ndim != 4 or node_coords.shape[-1] != 3:
        raise ValueError(f"Expected node coords (I+1,J+1,K+1,3), got {node_coords.shape}")

    nodes = np.asarray(node_coords, dtype=np.float64)
    cell_centers = _cell_centers(nodes)
    cell_volumes = _compute_cell_volumes_adflow(nodes, np.float64)

    i_face_centers, i_face_area_vectors = _face_metrics(nodes, 0)
    j_face_centers, j_face_area_vectors = _face_metrics(nodes, 1)
    k_face_centers, k_face_area_vectors = _face_metrics(nodes, 2)

    i_face_area_vectors = _orient_face_area_vectors(i_face_centers, i_face_area_vectors, cell_centers, 0)
    j_face_area_vectors = _orient_face_area_vectors(j_face_centers, j_face_area_vectors, cell_centers, 1)
    k_face_area_vectors = _orient_face_area_vectors(k_face_centers, k_face_area_vectors, cell_centers, 2)

    wall_face_centers = k_face_centers[:, :, 0, :]
    wall_face_area_vectors = k_face_area_vectors[:, :, 0, :]
    wall_face_normals = _normalize(wall_face_area_vectors)
    wall_tangent_i, wall_tangent_j = _wall_tangents(nodes, wall_face_normals)
    wall_distance = _wall_distance_from_kmin(cell_centers, wall_face_centers, wall_face_normals)

    return StructuredBlockGeometry(
        block_index=int(block_index),
        zone_name=str(zone_name),
        vertex_coords=torch.as_tensor(nodes, dtype=dtype),
        cell_centers=torch.as_tensor(cell_centers, dtype=dtype),
        cell_volumes=torch.as_tensor(np.maximum(cell_volumes, 1.0e-30), dtype=dtype),
        i_face_centers=torch.as_tensor(i_face_centers, dtype=dtype),
        j_face_centers=torch.as_tensor(j_face_centers, dtype=dtype),
        k_face_centers=torch.as_tensor(k_face_centers, dtype=dtype),
        i_face_area_vectors=torch.as_tensor(i_face_area_vectors, dtype=dtype),
        j_face_area_vectors=torch.as_tensor(j_face_area_vectors, dtype=dtype),
        k_face_area_vectors=torch.as_tensor(k_face_area_vectors, dtype=dtype),
        wall_face_centers=torch.as_tensor(wall_face_centers, dtype=dtype),
        wall_face_area_vectors=torch.as_tensor(wall_face_area_vectors, dtype=dtype),
        wall_face_normals=torch.as_tensor(wall_face_normals, dtype=dtype),
        wall_tangent_i=torch.as_tensor(wall_tangent_i, dtype=dtype),
        wall_tangent_j=torch.as_tensor(wall_tangent_j, dtype=dtype),
        wall_distance=torch.as_tensor(np.maximum(wall_distance, 1.0e-30), dtype=dtype),
        wall_distance_source="kmin_projection",
    )


def extract_case_block_geometry_from_volume_cgns(
    volume_cgns: str | Path,
    *,
    zone_to_block_index: Mapping[str, int] | None = None,
    dtype: torch.dtype = torch.float64,
) -> dict[int, StructuredBlockGeometry]:
    """Extract block-indexed structured geometry for all volume zones in one mesh CGNS."""
    volume_cgns = Path(volume_cgns).expanduser().resolve()
    geometry_by_block: dict[int, StructuredBlockGeometry] = {}
    with h5py.File(volume_cgns, "r") as h5file:
        for zone_name, zone_group in _iter_volume_zones(h5file):
            if zone_to_block_index is not None and zone_name not in zone_to_block_index:
                continue
            block_index = int(zone_to_block_index[zone_name]) if zone_to_block_index is not None else len(geometry_by_block)
            node_coords = _read_coordinates(zone_group, dtype=np.float64)
            geometry_by_block[block_index] = extract_block_geometry(
                node_coords,
                block_index=block_index,
                zone_name=zone_name,
                dtype=dtype,
            )
            solution = _find_solution_group(zone_group)
            if solution is not None and "TurbulentDistance" in solution:
                raw_distance = np.asarray(_node_data(solution["TurbulentDistance"]))
                distance = _to_ijk(
                    _trim_rind(raw_distance, _rind(solution, ndim=3), ndim=3)
                )
                geometry_by_block[block_index] = replace(
                    geometry_by_block[block_index],
                    wall_distance=torch.as_tensor(
                        np.maximum(distance, 1.0e-30),
                        dtype=dtype,
                    ),
                    wall_distance_source="adflow_turbulent_distance",
                )
    return geometry_by_block


def block_geometry_from_prediction_batch(
    volume_cgns: str | Path,
    batch: Mapping[str, Any],
    *,
    dtype: torch.dtype = torch.float64,
) -> dict[int, StructuredBlockGeometry]:
    """Extract block geometry aligned with a whole-case prediction batch."""
    zone_to_block = {
        str(block["block_name"]): int(block_index)
        for block_index, block in batch["blocks"].items()
        if str(block.get("block_name", ""))
    }
    geometry = extract_case_block_geometry_from_volume_cgns(
        volume_cgns,
        zone_to_block_index=zone_to_block,
        dtype=dtype,
    )
    expected = {int(block_index) for block_index in batch["blocks"].keys()}
    missing = sorted(expected.difference(geometry.keys()))
    if missing:
        raise RuntimeError(f"Failed to extract block geometry for block(s) {missing} from {volume_cgns}")
    return geometry


__all__ = [
    "StructuredBlockGeometry",
    "block_geometry_from_prediction_batch",
    "extract_block_geometry",
    "extract_case_block_geometry_from_volume_cgns",
]
