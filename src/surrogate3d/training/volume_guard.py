"""Validation helpers for volume-weighted whole-case training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional

import torch


def metadata_path_from_index(index_path: str | Path) -> Path:
    index_path = Path(index_path).expanduser().resolve()
    return index_path.parent / "metadata.json"


def require_dataset_cell_volumes(
    index_path: str | Path,
    *,
    use_volume_weighted_mse: bool,
) -> None:
    if not use_volume_weighted_mse:
        return
    metadata_path = metadata_path_from_index(index_path)
    with metadata_path.open("r", encoding="utf-8") as stream:
        metadata = dict(json.load(stream) or {})
    if bool(metadata.get("cell_volumes")):
        return
    raise RuntimeError(
        "training.loss.use_volume_weighted_mse=true requires dataset metadata to declare "
        f"`cell_volumes: true`, but {metadata_path} does not."
    )


def volume_blocks_from_batch(batch: Mapping[str, Any]) -> Optional[dict[int, torch.Tensor]]:
    volumes: dict[int, torch.Tensor] = {}
    for block_index, block in batch["blocks"].items():
        volume = block.get("cell_volumes")
        if volume is None:
            volume = block.get("volumes")
        if not isinstance(volume, torch.Tensor):
            return None
        volumes[int(block_index)] = volume
    return volumes


def case_label_from_batch(batch: Mapping[str, Any]) -> str:
    sample_ids = batch.get("sample_id")
    cond_idx = batch.get("cond_idx")
    sample_label = "unknown_sample"
    if isinstance(sample_ids, list) and sample_ids:
        sample_label = str(sample_ids[0])
        if len(sample_ids) > 1:
            sample_label = f"{sample_label}+{len(sample_ids) - 1}"
    cond_label = "unknown_cond"
    if isinstance(cond_idx, torch.Tensor) and cond_idx.numel() > 0:
        cond_label = f"cond_{int(cond_idx.reshape(-1)[0].item())}"
    return f"{sample_label}/{cond_label}"


def require_volume_blocks(batch: Mapping[str, Any]) -> dict[int, torch.Tensor]:
    volumes = volume_blocks_from_batch(batch)
    if volumes is not None:
        return volumes

    missing_blocks = []
    for block_index, block in batch["blocks"].items():
        if "cell_volumes" not in block and "volumes" not in block:
            missing_blocks.append(
                f"block {int(block_index)} ({block.get('block_name', 'unknown_block')})"
            )
    raise RuntimeError(
        "training.loss.use_volume_weighted_mse=true requires per-block cell volumes, "
        f"but batch {case_label_from_batch(batch)} is missing volume tensors for "
        f"{', '.join(missing_blocks) or 'all blocks'}"
    )


__all__ = [
    "case_label_from_batch",
    "metadata_path_from_index",
    "require_dataset_cell_volumes",
    "require_volume_blocks",
    "volume_blocks_from_batch",
]
