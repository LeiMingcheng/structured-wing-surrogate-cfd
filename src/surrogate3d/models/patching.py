"""3D patch helpers for structured blocks."""

from __future__ import annotations

from typing import Tuple

import torch


PatchSize3D = Tuple[int, int, int]


def validate_patch_size(shape: tuple[int, int, int], patch_size: PatchSize3D) -> None:
    for axis, (extent, patch) in enumerate(zip(shape, patch_size)):
        if int(patch) <= 0:
            raise ValueError(f"patch_size values must be positive, got {patch_size}")
        if int(extent) % int(patch) != 0:
            raise ValueError(
                f"Block shape {shape} is not divisible by patch_size {patch_size} on axis {axis}"
            )


def patchify3d(x: torch.Tensor, patch_size: PatchSize3D) -> torch.Tensor:
    """Convert ``(B, C, I, J, K)`` into patch vectors.

    Returns ``(B, n_i, n_j, n_k, C * p_i * p_j * p_k)``.
    """
    if x.ndim != 5:
        raise ValueError(f"patchify3d expects 5D tensor, got shape {tuple(x.shape)}")
    bsz, channels, ni, nj, nk = x.shape
    pi, pj, pk = (int(v) for v in patch_size)
    validate_patch_size((ni, nj, nk), (pi, pj, pk))
    x = x.reshape(bsz, channels, ni // pi, pi, nj // pj, pj, nk // pk, pk)
    x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
    return x.reshape(bsz, ni // pi, nj // pj, nk // pk, channels * pi * pj * pk)


def unpatchify3d(
    patches: torch.Tensor,
    *,
    channels: int,
    patch_size: PatchSize3D,
) -> torch.Tensor:
    """Convert patch vectors back into ``(B, C, I, J, K)``."""
    if patches.ndim != 5:
        raise ValueError(f"unpatchify3d expects 5D patch tensor, got shape {tuple(patches.shape)}")
    bsz, n_i, n_j, n_k, patch_dim = patches.shape
    pi, pj, pk = (int(v) for v in patch_size)
    expected = int(channels) * pi * pj * pk
    if int(patch_dim) != expected:
        raise ValueError(f"patch_dim={patch_dim} does not match channels*patch volume={expected}")
    x = patches.reshape(bsz, n_i, n_j, n_k, channels, pi, pj, pk)
    x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
    return x.reshape(bsz, int(channels), n_i * pi, n_j * pj, n_k * pk)


__all__ = ["PatchSize3D", "patchify3d", "unpatchify3d", "validate_patch_size"]
