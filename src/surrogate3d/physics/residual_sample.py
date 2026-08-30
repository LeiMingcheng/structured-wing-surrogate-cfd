"""Whole-case residual sample adapters for structured 3D surrogate batches."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from surrogate3d.physics.block_geometry import (
    StructuredBlockGeometry,
    block_geometry_from_prediction_batch,
)


def _batch_case_key(batch: Mapping[str, Any], batch_index: int) -> str:
    sample_ids = batch.get("sample_id")
    cond_idx = batch.get("cond_idx")
    source_case_dir = batch.get("source_case_dir")
    basenames = batch.get("cgns_basename")

    sample_text = str(sample_ids[batch_index]) if isinstance(sample_ids, Sequence) else f"sample{batch_index}"
    cond_text = (
        str(int(cond_idx[batch_index].item()))
        if isinstance(cond_idx, torch.Tensor)
        else str(cond_idx[batch_index])
        if isinstance(cond_idx, Sequence)
        else "0"
    )
    source_text = str(source_case_dir[batch_index]) if isinstance(source_case_dir, Sequence) else ""
    basename_text = str(basenames[batch_index]) if isinstance(basenames, Sequence) else ""
    return f"{source_text}::{basename_text}::{sample_text}::cond{cond_text}"


def _case_volume_cgns_path(batch: Mapping[str, Any], batch_index: int) -> Path:
    source_case_dir = batch.get("source_case_dir")
    basenames = batch.get("cgns_basename")
    if not isinstance(source_case_dir, Sequence) or not str(source_case_dir[batch_index]).strip():
        raise ValueError("Whole-case batch requires source_case_dir for residual sample building")
    case_dir = Path(str(source_case_dir[batch_index])).expanduser().resolve()
    basename = (
        str(basenames[batch_index]).strip()
        if isinstance(basenames, Sequence) and str(basenames[batch_index]).strip()
        else "wing_000_vol.cgns"
    )
    return (case_dir / basename).expanduser().resolve()


def _metadata_value(batch: Mapping[str, Any], key: str, batch_index: int) -> Any:
    value = batch.get(key)
    if isinstance(value, torch.Tensor):
        item = value[batch_index]
        return item.detach().cpu().item() if item.numel() == 1 else item.detach().cpu().tolist()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value[batch_index]
    return value


def _block_sample_tensor(
    value: torch.Tensor,
    *,
    batch_index: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return value[batch_index : batch_index + 1].to(device=device, dtype=dtype)


@dataclass(frozen=True)
class ResidualBlockSample:
    """Per-block residual inputs aligned to one whole-case sample."""

    block_index: int
    block_name: str
    fields: torch.Tensor
    coords_center: torch.Tensor
    coords_vertex: torch.Tensor
    cell_volumes: torch.Tensor
    wall_distance: torch.Tensor
    geometry: StructuredBlockGeometry

    def to_backend_input(self) -> dict[str, Any]:
        return {
            "block_index": int(self.block_index),
            "block_name": str(self.block_name),
            "fields": self.fields,
            "coords": {
                "center": self.coords_center,
                "vertex": self.coords_vertex,
            },
            "cell_volumes": self.cell_volumes,
            "wall_distance": self.wall_distance,
            "geometry": self.geometry,
        }


@dataclass(frozen=True)
class ResidualCaseSample:
    """One whole-case residual sample with per-block geometry-aligned tensors."""

    flow_conditions: torch.Tensor
    blocks: dict[int, ResidualBlockSample]
    metadata: dict[str, Any]

    def to_backend_input(self) -> dict[str, Any]:
        return {
            "flow_conditions": self.flow_conditions,
            "blocks": {
                int(block_index): block.to_backend_input()
                for block_index, block in self.blocks.items()
            },
            "metadata": dict(self.metadata),
        }


class WholeCaseResidualSampleBuilder:
    """Cache mesh-derived block geometry and build residual-ready case samples."""

    def __init__(self, *, geometry_dtype: torch.dtype = torch.float64) -> None:
        self.geometry_dtype = geometry_dtype
        self._case_cache: dict[str, dict[int, StructuredBlockGeometry]] = {}

    def _resolve_case_geometry(
        self,
        batch: Mapping[str, Any],
        *,
        batch_index: int,
    ) -> dict[int, StructuredBlockGeometry]:
        key = _batch_case_key(batch, batch_index)
        cached = self._case_cache.get(key)
        if cached is None:
            volume_cgns = _case_volume_cgns_path(batch, batch_index)
            batch_stub = {
                "blocks": {
                    int(block_index): {"block_name": block.get("block_name", "")}
                    for block_index, block in batch["blocks"].items()
                }
            }
            cached = block_geometry_from_prediction_batch(
                volume_cgns,
                batch_stub,
                dtype=self.geometry_dtype,
            )
            self._case_cache[key] = cached
        return cached

    def build_case_sample(
        self,
        fields: Mapping[int, torch.Tensor],
        batch: Mapping[str, Any],
        *,
        batch_index: int,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> ResidualCaseSample:
        if not fields:
            raise ValueError("fields must contain at least one block tensor")

        sample_device = torch.device(device) if device is not None else next(iter(fields.values())).device
        sample_dtype = dtype or next(iter(fields.values())).dtype
        geometry_by_block = self._resolve_case_geometry(batch, batch_index=batch_index)
        block_samples: dict[int, ResidualBlockSample] = {}

        for block_index, block_fields in fields.items():
            block_key = int(block_index)
            if block_key not in batch["blocks"]:
                raise KeyError(f"Batch does not include block {block_key}")
            if block_key not in geometry_by_block:
                raise KeyError(f"Geometry cache does not include block {block_key}")
            batch_block = batch["blocks"][block_key]
            if "coords_pde" not in batch_block:
                raise KeyError(f"Batch block {block_key} requires 'coords_pde' for residual samples")

            geometry = geometry_by_block[block_key].to(sample_device, dtype=self.geometry_dtype)
            coords_center = _block_sample_tensor(
                batch_block["coords_pde"],
                batch_index=batch_index,
                device=sample_device,
                dtype=self.geometry_dtype,
            )
            if "cell_volumes" in batch_block:
                cell_volumes = _block_sample_tensor(
                    batch_block["cell_volumes"],
                    batch_index=batch_index,
                    device=sample_device,
                    dtype=self.geometry_dtype,
                )
            else:
                cell_volumes = geometry.cell_volumes.unsqueeze(0)

            block_samples[block_key] = ResidualBlockSample(
                block_index=block_key,
                block_name=str(batch_block.get("block_name", geometry.zone_name)),
                fields=_block_sample_tensor(
                    block_fields,
                    batch_index=batch_index,
                    device=sample_device,
                    dtype=sample_dtype,
                ),
                coords_center=coords_center,
                coords_vertex=geometry.vertex_coords.permute(3, 0, 1, 2).unsqueeze(0),
                cell_volumes=cell_volumes,
                wall_distance=geometry.wall_distance.unsqueeze(0),
                geometry=geometry,
            )

        metadata = {
            "sample_id": _metadata_value(batch, "sample_id", batch_index),
            "cond_idx": _metadata_value(batch, "cond_idx", batch_index),
            "source_case_dir": _metadata_value(batch, "source_case_dir", batch_index),
            "cgns_basename": _metadata_value(batch, "cgns_basename", batch_index),
            "case_key": _batch_case_key(batch, batch_index),
            "volume_cgns": str(_case_volume_cgns_path(batch, batch_index)),
        }
        return ResidualCaseSample(
            flow_conditions=_block_sample_tensor(
                batch["flow_conditions"],
                batch_index=batch_index,
                device=sample_device,
                dtype=self.geometry_dtype,
            ),
            blocks=block_samples,
            metadata=metadata,
        )

    def build_batch_samples(
        self,
        fields: Mapping[int, torch.Tensor],
        batch: Mapping[str, Any],
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> list[ResidualCaseSample]:
        if not fields:
            return []
        batch_size = int(next(iter(fields.values())).shape[0])
        return [
            self.build_case_sample(
                fields,
                batch,
                batch_index=batch_index,
                device=device,
                dtype=dtype,
            )
            for batch_index in range(batch_size)
        ]


__all__ = [
    "ResidualBlockSample",
    "ResidualCaseSample",
    "WholeCaseResidualSampleBuilder",
]
