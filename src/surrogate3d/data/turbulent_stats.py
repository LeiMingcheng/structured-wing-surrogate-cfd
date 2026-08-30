"""Turbulent-channel log-scaling statistics for surrogate3d datasets."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd


def _resolve_shard_path(root: Path, shard_path: str) -> Path:
    path = Path(shard_path)
    if not path.is_absolute():
        path = root / path
    return path.expanduser().resolve()


def _state_field_names(metadata: dict[str, Any]) -> list[str]:
    fields = metadata.get("state_fields") or metadata.get("export_state_fields")
    if not fields:
        raise ValueError("metadata.json is missing state_fields/export_state_fields")
    return [str(field) for field in fields]


def _other_channel_stats(
    metadata: dict[str, Any],
    turbulent_channel_idx: int,
) -> tuple[list[float], list[float], float, float]:
    norm = metadata.get("norm") or {}
    means = norm.get("state_mean")
    stds = norm.get("state_std")
    if means is None or stds is None:
        raise ValueError("metadata.json is missing norm.state_mean/state_std")
    if len(means) <= turbulent_channel_idx or len(stds) <= turbulent_channel_idx:
        raise ValueError(
            f"metadata has {len(means)} state channels, cannot read turbulent channel {turbulent_channel_idx}"
        )
    return (
        [float(v) for v in means[:turbulent_channel_idx]],
        [float(v) for v in stds[:turbulent_channel_idx]],
        float(means[turbulent_channel_idx]),
        float(stds[turbulent_channel_idx]),
    )


def _shard_turbulent_log_stats(
    shard_path: str,
    local_indices: list[int],
    turbulent_channel_idx: int,
    scale_factor: float,
    state_dataset_name: str,
) -> tuple[float, float, int, int, int]:
    with h5py.File(shard_path, "r") as h5file:
        dataset = h5file[state_dataset_name]
        shard_values = np.asarray(dataset[local_indices, turbulent_channel_idx], dtype=np.float64)
    n_negative = int(np.sum(shard_values < 0.0))
    n_zero = int(np.sum(shard_values == 0.0))
    shard_values = np.maximum(shard_values, 0.0)
    log_values = np.log1p(scale_factor * shard_values)
    return (
        float(np.sum(log_values)),
        float(np.sum(np.square(log_values))),
        int(log_values.size),
        n_negative,
        n_zero,
    )


def compute_turbulent_scale_stats(
    index_path: str | Path,
    *,
    turbulent_channel_idx: int = 5,
    state_dataset_name: str = "state",
    num_workers: int = 1,
) -> dict[str, Any]:
    """Compute PIR-DM-compatible ``log1p(k * nu_tilde)`` statistics.

    The scale factor follows PIR-DM: ``k = 1 / mean(nu_tilde)``. Non-positive
    values are clipped to zero only for log-stat computation, matching the
    physical non-negative model-space assumption without modifying source data.
    """
    index_path = Path(index_path).expanduser().resolve()
    root = index_path.parent
    metadata_path = root / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found next to {index_path}")
    with metadata_path.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)

    field_names = _state_field_names(metadata)
    if turbulent_channel_idx < 0 or turbulent_channel_idx >= len(field_names):
        raise ValueError(f"Invalid turbulent_channel_idx={turbulent_channel_idx} for {len(field_names)} fields")

    other_means, other_stds, original_mean, original_std = _other_channel_stats(metadata, turbulent_channel_idx)
    if original_mean <= 0.0:
        raise RuntimeError(f"Non-positive turbulent mean encountered: {original_mean}")
    scale_factor = 1.0 / original_mean

    index_df = pd.read_csv(index_path, low_memory=False)
    log_sum = 0.0
    log_sumsq = 0.0
    total_cells = 0
    n_negative = 0
    n_zero = 0

    shard_tasks = [
        (
            str(_resolve_shard_path(root, str(shard_rel))),
            sorted(int(v) for v in frame["local_index"].tolist()),
            int(turbulent_channel_idx),
            float(scale_factor),
            str(state_dataset_name),
        )
        for shard_rel, frame in index_df.groupby("shard_path", sort=True)
    ]
    num_workers = max(1, int(num_workers))
    if num_workers == 1:
        results = (
            _shard_turbulent_log_stats(
                shard_path,
                local_indices,
                channel_idx,
                shard_scale_factor,
                dataset_name,
            )
            for shard_path, local_indices, channel_idx, shard_scale_factor, dataset_name in shard_tasks
        )
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = executor.map(_shard_turbulent_log_stats, *zip(*shard_tasks), chunksize=1)
            for shard_log_sum, shard_log_sumsq, shard_total_cells, shard_negative, shard_zero in results:
                log_sum += float(shard_log_sum)
                log_sumsq += float(shard_log_sumsq)
                total_cells += int(shard_total_cells)
                n_negative += int(shard_negative)
                n_zero += int(shard_zero)
            results = None
    if results is not None:
        for shard_log_sum, shard_log_sumsq, shard_total_cells, shard_negative, shard_zero in results:
            log_sum += float(shard_log_sum)
            log_sumsq += float(shard_log_sumsq)
            total_cells += int(shard_total_cells)
            n_negative += int(shard_negative)
            n_zero += int(shard_zero)

    if total_cells <= 0:
        raise RuntimeError(f"No turbulent-channel cells found in {index_path}")
    log_mean = log_sum / total_cells
    log_var = max(log_sumsq / total_cells - log_mean * log_mean, 0.0)
    log_std = math.sqrt(log_var)

    return {
        "version": "turbulent-scale-surrogate3d-v1",
        "turbulent_channel": {
            "index": int(turbulent_channel_idx),
            "name": field_names[turbulent_channel_idx],
            "scale_factor": float(scale_factor),
            "original_mean": float(original_mean),
            "original_std": float(original_std),
            "log_mean": float(log_mean),
            "log_std": float(log_std),
        },
        "other_channels": {
            "means": other_means,
            "stds": other_stds,
        },
        "statistics": {
            "n_samples": int(len(index_df)),
            "n_cells": int(total_cells),
            "n_negative_source_values": int(n_negative),
            "n_zero_source_values": int(n_zero),
            "source_index": str(index_path),
            "source_metadata": str(metadata_path),
        },
    }


def write_turbulent_scale_stats(
    index_path: str | Path,
    output_path: str | Path | None = None,
    *,
    turbulent_channel_idx: int = 5,
    num_workers: int = 1,
) -> Path:
    """Compute and write ``turbulent_scale_stats.json`` next to an index."""
    index_path = Path(index_path).expanduser().resolve()
    if output_path is None:
        output_path = index_path.parent / "turbulent_scale_stats.json"
    output_path = Path(output_path).expanduser().resolve()
    stats = compute_turbulent_scale_stats(
        index_path,
        turbulent_channel_idx=turbulent_channel_idx,
        num_workers=num_workers,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(stats, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return output_path


__all__ = ["compute_turbulent_scale_stats", "write_turbulent_scale_stats"]
