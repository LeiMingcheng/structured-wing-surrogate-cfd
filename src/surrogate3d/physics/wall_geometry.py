"""Wall-surface geometry views derived from structured 3D block geometry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from surrogate3d.physics.block_geometry import (
    StructuredBlockGeometry,
    block_geometry_from_prediction_batch,
    extract_block_geometry,
    extract_case_block_geometry_from_volume_cgns,
)


@dataclass(frozen=True)
class BlockWallGeometry:
    """Wall-face geometry for one structured volume block."""

    block_index: int
    zone_name: str
    face_centers: torch.Tensor
    area_vectors: torch.Tensor
    cell_areas: torch.Tensor
    cell_centers: torch.Tensor
    wall_distance: torch.Tensor
    normal: torch.Tensor
    tangent_i: torch.Tensor
    tangent_j: torch.Tensor

    def to(self, device: str | torch.device) -> "BlockWallGeometry":
        torch_device = torch.device(device)
        return BlockWallGeometry(
            block_index=int(self.block_index),
            zone_name=str(self.zone_name),
            face_centers=self.face_centers.to(torch_device),
            area_vectors=self.area_vectors.to(torch_device),
            cell_areas=self.cell_areas.to(torch_device),
            cell_centers=self.cell_centers.to(torch_device),
            wall_distance=self.wall_distance.to(torch_device),
            normal=self.normal.to(torch_device),
            tangent_i=self.tangent_i.to(torch_device),
            tangent_j=self.tangent_j.to(torch_device),
        )


def _wall_view(block_geometry: StructuredBlockGeometry) -> BlockWallGeometry:
    area_vectors = block_geometry.wall_face_area_vectors
    return BlockWallGeometry(
        block_index=int(block_geometry.block_index),
        zone_name=str(block_geometry.zone_name),
        face_centers=block_geometry.wall_face_centers,
        area_vectors=area_vectors,
        cell_areas=torch.linalg.vector_norm(area_vectors, dim=-1),
        cell_centers=block_geometry.cell_centers[:, :, 0, :],
        wall_distance=block_geometry.wall_distance[:, :, 0],
        normal=block_geometry.wall_face_normals,
        tangent_i=block_geometry.wall_tangent_i,
        tangent_j=block_geometry.wall_tangent_j,
    )


def extract_block_wall_geometry(
    node_coords: np.ndarray,
    *,
    block_index: int,
    zone_name: str,
    dtype: torch.dtype = torch.float64,
) -> BlockWallGeometry:
    """Extract k=0 viscous-wall geometry from one structured 3D block."""
    return _wall_view(
        extract_block_geometry(
            node_coords,
            block_index=block_index,
            zone_name=zone_name,
            dtype=dtype,
        )
    )


def extract_case_wall_geometry_from_volume_cgns(
    volume_cgns: str | Path,
    *,
    zone_to_block_index: Mapping[str, int] | None = None,
    dtype: torch.dtype = torch.float64,
) -> dict[int, BlockWallGeometry]:
    """Extract wall geometry for all structured volume blocks in one mesh CGNS."""
    block_geometry = extract_case_block_geometry_from_volume_cgns(
        volume_cgns,
        zone_to_block_index=zone_to_block_index,
        dtype=dtype,
    )
    return {int(block_index): _wall_view(geometry) for block_index, geometry in block_geometry.items()}


def wall_geometry_from_prediction_batch(
    volume_cgns: str | Path,
    batch: Mapping[str, Any],
    *,
    dtype: torch.dtype = torch.float64,
) -> dict[int, BlockWallGeometry]:
    """Extract block-indexed wall geometry aligned with a whole-case prediction batch."""
    block_geometry = block_geometry_from_prediction_batch(
        volume_cgns,
        batch,
        dtype=dtype,
    )
    return {int(block_index): _wall_view(geometry) for block_index, geometry in block_geometry.items()}


__all__ = [
    "BlockWallGeometry",
    "extract_block_wall_geometry",
    "extract_case_wall_geometry_from_volume_cgns",
    "wall_geometry_from_prediction_batch",
]
