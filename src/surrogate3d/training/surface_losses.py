"""Surface-aware training helpers for whole-case 3D surrogates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from surrogate3d.physics.field_force import (
    ForceEvaluatorConfig,
    compute_volume_surface_fields_torch,
    force_evaluator_config_from_dict,
    prepare_volume_force_geometry,
)
from surrogate3d.physics.wall_geometry import BlockWallGeometry, wall_geometry_from_prediction_batch


def _block_batch_to_cpu(batch_blocks: Mapping[int, Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    copied: dict[int, dict[str, Any]] = {}
    for block_index, block in batch_blocks.items():
        copied[int(block_index)] = {
            key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
            for key, value in block.items()
        }
    return copied


def _surface_loss_reduction(
    pred_surface: torch.Tensor,
    target_surface: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cp_error = pred_surface[..., 0] - target_surface[..., 0]
    cf_error = pred_surface[..., 1:4] - target_surface[..., 1:4]
    cp_mse = torch.mean(cp_error.pow(2))
    cf_mse = torch.mean(cf_error.pow(2))
    return cp_mse, cf_mse


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
        raise ValueError("Whole-case batch requires source_case_dir for surface-aware training losses")
    case_dir = Path(str(source_case_dir[batch_index])).expanduser().resolve()
    basename = (
        str(basenames[batch_index]).strip()
        if isinstance(basenames, Sequence) and str(basenames[batch_index]).strip()
        else "wing_000_vol.cgns"
    )
    return (case_dir / basename).expanduser().resolve()


@dataclass
class CachedWallGeometry:
    """Per-case cached wall geometry and prepared flat integration tensors."""

    block_geometry: dict[int, BlockWallGeometry]
    prepared_geometry: dict[str, Any]

    def to(self, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
        geometry = {int(k): v.to(device) for k, v in self.block_geometry.items()}
        prepared = prepare_volume_force_geometry(geometry, device=device, dtype=dtype)
        return {
            "block_geometry": geometry,
            "prepared_geometry": prepared,
        }


class WholeCaseSurfaceLossHelper:
    """Cache wall geometry and compute differentiable Cp/Cf losses."""

    def __init__(
        self,
        *,
        force_config: ForceEvaluatorConfig | Mapping[str, Any] | None = None,
    ) -> None:
        self.force_config = (
            force_config
            if isinstance(force_config, ForceEvaluatorConfig)
            else force_evaluator_config_from_dict(force_config)
        )
        self._case_cache: dict[str, CachedWallGeometry] = {}

    def _resolve_case_geometry(
        self,
        batch: Mapping[str, Any],
        *,
        batch_index: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, Any]:
        key = _batch_case_key(batch, batch_index)
        cached = self._case_cache.get(key)
        if cached is None:
            volume_cgns = _case_volume_cgns_path(batch, batch_index)
            batch_blocks_cpu = _block_batch_to_cpu(batch["blocks"])
            batch_stub = {"blocks": batch_blocks_cpu}
            block_geometry = wall_geometry_from_prediction_batch(
                volume_cgns,
                batch_stub,
                dtype=dtype,
            )
            prepared = prepare_volume_force_geometry(block_geometry, device=device, dtype=dtype)
            cached = CachedWallGeometry(
                block_geometry=block_geometry,
                prepared_geometry=prepared,
            )
            self._case_cache[key] = cached
        return cached.to(device, dtype)

    def compute_surface_losses(
        self,
        *,
        pred_phys: Mapping[int, torch.Tensor],
        target_phys: Mapping[int, torch.Tensor],
        batch: Mapping[str, Any],
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cp_losses: list[torch.Tensor] = []
        cf_losses: list[torch.Tensor] = []
        batch_size = int(batch["geometry"].shape[0])
        for batch_index in range(batch_size):
            case_geometry = self._resolve_case_geometry(
                batch,
                batch_index=batch_index,
                device=device,
                dtype=dtype,
            )
            pred_case = {
                int(block_index): block[batch_index : batch_index + 1].to(device=device, dtype=dtype)
                for block_index, block in pred_phys.items()
            }
            target_case = {
                int(block_index): block[batch_index : batch_index + 1].to(device=device, dtype=dtype)
                for block_index, block in target_phys.items()
            }
            flow_case = batch["flow_conditions"][batch_index : batch_index + 1].to(device=device, dtype=dtype)
            pred_surface = compute_volume_surface_fields_torch(
                pred_case,
                case_geometry["block_geometry"],
                flow_case,
                config=self.force_config,
                prepared_geometry=case_geometry["prepared_geometry"],
                device=device,
                dtype=dtype,
            )["surface_fields_batched"]
            target_surface = compute_volume_surface_fields_torch(
                target_case,
                case_geometry["block_geometry"],
                flow_case,
                config=self.force_config,
                prepared_geometry=case_geometry["prepared_geometry"],
                device=device,
                dtype=dtype,
            )["surface_fields_batched"]
            cp_mse, cf_mse = _surface_loss_reduction(pred_surface, target_surface)
            cp_losses.append(cp_mse)
            cf_losses.append(cf_mse)
        if not cp_losses:
            zero = torch.zeros((), device=device, dtype=dtype)
            return zero, zero
        return torch.stack(cp_losses).mean(), torch.stack(cf_losses).mean()


__all__ = [
    "WholeCaseSurfaceLossHelper",
]
