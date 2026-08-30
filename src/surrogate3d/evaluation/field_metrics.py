"""Whole-case field error metrics for surrogate3d predictions."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch

from surrogate3d.evaluation.inference import WholeCasePrediction


def _summary(squared_error: float, count: float) -> dict[str, float]:
    mse = squared_error / count
    return {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "l2": math.sqrt(squared_error),
    }


def compute_block_field_metrics(
    candidate_blocks: Mapping[int, torch.Tensor],
    reference_blocks: Mapping[int, torch.Tensor],
    *,
    channel_names: Sequence[str],
    volume_blocks: Mapping[int, torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Aggregate MSE/L2 metrics over all blocks, optionally weighted by cell volume."""
    block_indices = sorted(int(key) for key in reference_blocks)
    if block_indices != sorted(int(key) for key in candidate_blocks):
        raise ValueError("Candidate and reference block sets differ")

    n_channels = int(reference_blocks[block_indices[0]].shape[1])
    if len(channel_names) != n_channels:
        raise ValueError(f"Expected {n_channels} channel names, got {len(channel_names)}")

    squared_error = torch.zeros(n_channels, dtype=torch.float64)
    counts = torch.zeros(n_channels, dtype=torch.float64)

    for block_index in block_indices:
        candidate = candidate_blocks[block_index].detach().to(dtype=torch.float64)
        reference = reference_blocks[block_index].detach().to(dtype=torch.float64)
        if candidate.shape != reference.shape:
            raise ValueError(
                f"Block {block_index} shape mismatch: {tuple(candidate.shape)} != {tuple(reference.shape)}"
            )
        error_squared = (candidate - reference) ** 2

        if volume_blocks is None:
            reduce_dims = tuple(dim for dim in range(error_squared.ndim) if dim != 1)
            squared_error += error_squared.sum(dim=reduce_dims).cpu()
            counts += error_squared.numel() / n_channels
            continue

        volumes = volume_blocks[block_index].detach().to(device=error_squared.device, dtype=torch.float64)
        if volumes.ndim == error_squared.ndim - 1:
            volumes = volumes.unsqueeze(1)
        if volumes.ndim != error_squared.ndim or volumes.shape[1] != 1:
            raise ValueError(
                f"Block {block_index} cell volume shape {tuple(volumes.shape)} is incompatible with "
                f"state shape {tuple(error_squared.shape)}"
            )
        reduce_dims = tuple(dim for dim in range(error_squared.ndim) if dim != 1)
        squared_error += (volumes * error_squared).sum(dim=reduce_dims).cpu()
        counts += volumes.sum().cpu()

    channels = {
        str(name): _summary(float(squared_error[idx]), float(counts[idx]))
        for idx, name in enumerate(channel_names)
    }
    return {
        "summary": _summary(
            float(squared_error.sum()),
            float(counts.sum()),
        ),
        "channels": channels,
    }


def compute_prediction_field_metrics(prediction: WholeCasePrediction) -> dict[str, Any]:
    """Compute physical, normalized, and cell-volume-weighted whole-case metrics."""
    channel_names = prediction.checkpoint_payload.get("state_fields")
    if not channel_names:
        channel_names = prediction.config.get("data", {}).get("state_fields")
    if not channel_names:
        channel_names = [
            "Density",
            "VelocityX",
            "VelocityY",
            "VelocityZ",
            "Pressure",
            "TurbulentSANuTilde",
        ]
    volume_blocks = {
        int(block_index): block["cell_volumes"]
        for block_index, block in prediction.batch["blocks"].items()
    }
    return {
        "physical": compute_block_field_metrics(
            prediction.pred_physical,
            prediction.target_physical,
            channel_names=channel_names,
        ),
        "normalized": compute_block_field_metrics(
            prediction.pred_model,
            prediction.target_model,
            channel_names=channel_names,
        ),
        "physical_cell_volume_weighted": compute_block_field_metrics(
            prediction.pred_physical,
            prediction.target_physical,
            channel_names=channel_names,
            volume_blocks=volume_blocks,
        ),
        "normalized_cell_volume_weighted": compute_block_field_metrics(
            prediction.pred_model,
            prediction.target_model,
            channel_names=channel_names,
            volume_blocks=volume_blocks,
        ),
    }


__all__ = ["compute_block_field_metrics", "compute_prediction_field_metrics"]
