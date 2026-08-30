"""DataLoader helpers for whole-case 3D structured-block datasets."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from surrogate3d.data.case_dataset import WholeCaseBlockDataset, whole_case_collate


@dataclass(frozen=True)
class WholeCaseSplit:
    train_indices: list[int]
    val_indices: list[int]
    split_key: str


@dataclass(frozen=True)
class WholeCaseLoaders:
    dataset: WholeCaseBlockDataset
    train_loader: DataLoader
    val_loader: Optional[DataLoader]
    split: WholeCaseSplit


def split_whole_cases(
    dataset: WholeCaseBlockDataset,
    *,
    val_fraction: float = 0.2,
    seed: int = 0,
    split_by_sample_id: bool = True,
) -> WholeCaseSplit:
    """Split whole-case items, grouping by ``sample_id`` by default."""
    n_cases = len(dataset)
    if n_cases <= 0:
        raise ValueError("Cannot split an empty dataset")
    if n_cases == 1 or float(val_fraction) <= 0.0:
        return WholeCaseSplit(
            train_indices=list(range(n_cases)),
            val_indices=[],
            split_key="none",
        )

    rng = random.Random(int(seed))
    if split_by_sample_id:
        groups: dict[str, list[int]] = {}
        split_field = (
            "geometry_id"
            if all("geometry_id" in case for case in dataset.cases)
            else "sample_id"
        )
        for idx, case in enumerate(dataset.cases):
            groups.setdefault(str(case[split_field]), []).append(idx)
        keys = sorted(groups)
        rng.shuffle(keys)
        n_val_groups = int(round(len(keys) * float(val_fraction)))
        n_val_groups = min(max(1, n_val_groups), max(1, len(keys) - 1))
        val_keys = set(keys[:n_val_groups])
        train_indices = [
            idx for key in keys for idx in groups[key] if key not in val_keys
        ]
        val_indices = [
            idx for key in keys for idx in groups[key] if key in val_keys
        ]
        if not train_indices:
            train_indices, val_indices = val_indices[:-1], val_indices[-1:]
        return WholeCaseSplit(
            train_indices=sorted(train_indices),
            val_indices=sorted(val_indices),
            split_key=split_field,
        )

    indices = list(range(n_cases))
    rng.shuffle(indices)
    n_val = int(round(n_cases * float(val_fraction)))
    n_val = min(max(1, n_val), n_cases - 1)
    val_indices = sorted(indices[:n_val])
    train_indices = sorted(indices[n_val:])
    return WholeCaseSplit(
        train_indices=train_indices,
        val_indices=val_indices,
        split_key="case",
    )


def create_whole_case_loaders(
    index_path: str | Path,
    *,
    batch_size: int = 1,
    val_fraction: float = 0.2,
    seed: int = 0,
    num_cases: Optional[int] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    shuffle_train: bool = True,
    split_by_sample_id: bool = True,
    cache_size_mb: int = 256,
    use_ddp: bool = False,
) -> WholeCaseLoaders:
    dataset = WholeCaseBlockDataset(
        index_path,
        num_cases=num_cases,
        cache_size_mb=cache_size_mb,
    )
    split = split_whole_cases(
        dataset,
        val_fraction=val_fraction,
        seed=seed,
        split_by_sample_id=split_by_sample_id,
    )
    train_dataset = Subset(dataset, split.train_indices)
    val_dataset = Subset(dataset, split.val_indices) if split.val_indices else None
    train_kwargs = {
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "collate_fn": whole_case_collate,
    }
    val_kwargs = dict(train_kwargs)
    if use_ddp and dist.is_initialized():
        train_kwargs["sampler"] = DistributedSampler(
            train_dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=bool(shuffle_train),
            drop_last=False,
            seed=int(seed),
        )
    else:
        train_kwargs["shuffle"] = bool(shuffle_train)
    train_loader = DataLoader(train_dataset, **train_kwargs)
    val_loader = None
    if val_dataset is not None:
        if use_ddp and dist.is_initialized():
            val_kwargs["sampler"] = DistributedSampler(
                val_dataset,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=False,
                drop_last=False,
                seed=int(seed),
            )
        else:
            val_kwargs["shuffle"] = False
        val_loader = DataLoader(val_dataset, **val_kwargs)
    return WholeCaseLoaders(
        dataset=dataset,
        train_loader=train_loader,
        val_loader=val_loader,
        split=split,
    )


__all__ = [
    "WholeCaseLoaders",
    "WholeCaseSplit",
    "create_whole_case_loaders",
    "split_whole_cases",
]
