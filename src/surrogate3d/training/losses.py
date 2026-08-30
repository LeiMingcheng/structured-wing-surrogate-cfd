"""Whole-case loss helpers for multi-block 3D fields."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch
import torch.nn.functional as F


def whole_case_mse(
    pred: Mapping[int, torch.Tensor],
    target: Mapping[int, torch.Tensor],
    *,
    block_indices: Optional[list[int]] = None,
    per_block_equal_weight: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute MSE over all predicted blocks."""
    indices = list(block_indices) if block_indices is not None else sorted(target.keys())
    if not indices:
        raise ValueError("whole_case_mse requires at least one block")

    losses = []
    metrics: dict[str, torch.Tensor] = {}
    sq_sum = None
    count_sum = 0
    for block_index in indices:
        if block_index not in pred:
            raise KeyError(f"Prediction missing block {block_index}")
        block_loss = F.mse_loss(pred[block_index], target[block_index], reduction="mean")
        metrics[f"block_{block_index}_mse"] = block_loss.detach()
        losses.append(block_loss)
        if not per_block_equal_weight:
            err = (pred[block_index] - target[block_index]).pow(2)
            value = err.sum()
            sq_sum = value if sq_sum is None else sq_sum + value
            count_sum += err.numel()

    if per_block_equal_weight:
        mse = torch.stack(losses).mean()
    else:
        assert sq_sum is not None
        mse = sq_sum / max(1, count_sum)
    metrics["mse"] = mse.detach()
    return mse, metrics


def whole_case_volume_weighted_mse(
    pred: Mapping[int, torch.Tensor],
    target: Mapping[int, torch.Tensor],
    volumes: Mapping[int, torch.Tensor],
    *,
    block_indices: Optional[list[int]] = None,
    per_block_equal_weight: bool = True,
    alpha: float = 0.5,
    w_max: float = 1000.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute PIR-DM volume-weighted MSE independently on each block."""
    indices = list(block_indices) if block_indices is not None else sorted(target.keys())
    if not indices:
        raise ValueError("whole_case_volume_weighted_mse requires at least one block")

    losses = []
    weighted_sum = None
    count_sum = 0
    metrics: dict[str, torch.Tensor] = {}
    for block_index in indices:
        if block_index not in volumes:
            raise KeyError(f"Volumes missing block {block_index}")
        volume = volumes[block_index].to(device=pred[block_index].device, dtype=pred[block_index].dtype)
        if volume.ndim == pred[block_index].ndim:
            if volume.shape[1] != 1:
                raise ValueError("3D volume weights with channel dimension must have one channel")
            volume = volume[:, 0]
        if volume.ndim == pred[block_index].ndim - 2:
            volume = volume.unsqueeze(0).expand(pred[block_index].shape[0], *volume.shape)
        if volume.ndim != pred[block_index].ndim - 1:
            raise ValueError(
                f"Volume tensor for block {block_index} must be (I,J,K), (B,I,J,K), "
                f"or (B,1,I,J,K), got {tuple(volume.shape)}"
            )
        eps = torch.tensor(1.0e-12, dtype=volume.dtype, device=volume.device)
        weights = torch.pow(volume.mean() / (volume + eps), alpha)
        weights = torch.clamp(weights, max=w_max)
        weights = weights / weights.mean().clamp_min(1.0e-12)
        block_loss = (weights.unsqueeze(1) * (pred[block_index] - target[block_index]).pow(2)).mean()
        losses.append(block_loss)
        metrics[f"block_{block_index}_volume_weighted_mse"] = block_loss.detach()
        metrics[f"block_{block_index}_vw_w_min"] = weights.min().detach()
        metrics[f"block_{block_index}_vw_w_max"] = weights.max().detach()
        metrics[f"block_{block_index}_vw_w_mean"] = weights.mean().detach()
        metrics[f"block_{block_index}_vw_w_std"] = weights.std().detach()
        if not per_block_equal_weight:
            n = pred[block_index].numel()
            value = block_loss * n
            weighted_sum = value if weighted_sum is None else weighted_sum + value
            count_sum += n

    if per_block_equal_weight:
        loss = torch.stack(losses).mean()
    else:
        assert weighted_sum is not None
        loss = weighted_sum / max(1, count_sum)
    metrics["volume_weighted_mse"] = loss.detach()
    return loss, metrics


def move_whole_case_batch(batch: dict, device: torch.device) -> dict:
    """Move tensor values in a whole-case batch to a device."""
    out = {}
    for key, value in batch.items():
        if key == "blocks":
            moved_blocks = {}
            for block_index, block in value.items():
                moved_blocks[int(block_index)] = {
                    inner_key: inner_value.to(device) if isinstance(inner_value, torch.Tensor) else inner_value
                    for inner_key, inner_value in block.items()
                }
            out[key] = moved_blocks
        elif isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def transform_target_blocks(
    batch: Mapping[str, Any],
    normalizer: Optional[torch.nn.Module],
) -> dict[int, torch.Tensor]:
    targets = {int(k): v["target"] for k, v in batch["blocks"].items()}
    if normalizer is None:
        return targets
    if hasattr(normalizer, "transform_blocks"):
        return normalizer.transform_blocks(targets)
    return {int(k): normalizer.transform(v) for k, v in targets.items()}


def inverse_transform_blocks(
    blocks: Mapping[int, torch.Tensor],
    normalizer: Optional[torch.nn.Module],
) -> dict[int, torch.Tensor]:
    if normalizer is None:
        return {int(k): v for k, v in blocks.items()}
    if hasattr(normalizer, "inverse_transform_blocks"):
        return normalizer.inverse_transform_blocks(blocks)
    return {int(k): normalizer.inverse_transform(v) for k, v in blocks.items()}


__all__ = [
    "inverse_transform_blocks",
    "move_whole_case_batch",
    "transform_target_blocks",
    "whole_case_mse",
    "whole_case_volume_weighted_mse",
]
