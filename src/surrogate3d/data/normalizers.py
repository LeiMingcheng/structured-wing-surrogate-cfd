"""Field normalization helpers for whole-case 3D surrogate training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
import torch.nn as nn


def _channel_view(values: torch.Tensor, ndim: int) -> torch.Tensor:
    return values.view(1, -1, *([1] * (int(ndim) - 2)))


class WholeCaseFieldNormalizer(nn.Module):
    """PIR-DM-style field normalizer generalized to 3D block tensors.

    Supports both structured-wing export ``metadata.json`` raw state
    statistics and PIR-DM ``turbulent_scale_stats.json`` files. The model
    space can be inverted directly before NK resume payload writing.
    """

    def __init__(
        self,
        *,
        means: torch.Tensor,
        stds: torch.Tensor,
        scale_turbulent: bool = False,
        normalize: bool = True,
        turbulent_channel_idx: int = 5,
        turbulent_scale_factor: Optional[float] = None,
        eps: float = 1.0e-12,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.normalize = bool(normalize)
        self.scale_turbulent = bool(scale_turbulent) or self.normalize
        self.turbulent_channel_idx = int(turbulent_channel_idx)
        self.turbulent_scale_factor = (
            None if turbulent_scale_factor is None else float(turbulent_scale_factor)
        )
        self.eps = float(eps)
        self.metadata = dict(metadata or {})

        means = torch.as_tensor(means, dtype=torch.float32)
        stds = torch.as_tensor(stds, dtype=torch.float32).clamp_min(self.eps)
        if means.ndim != 1 or stds.ndim != 1 or means.shape != stds.shape:
            raise ValueError("means/stds must be 1D tensors with identical shape")
        if self.turbulent_channel_idx < 0 or self.turbulent_channel_idx >= int(means.numel()):
            raise ValueError(
                f"turbulent_channel_idx={self.turbulent_channel_idx} is invalid for {means.numel()} channels"
            )
        if self.scale_turbulent and self.turbulent_scale_factor is None:
            raise ValueError("turbulent_scale_factor is required when scale_turbulent=True")

        self.register_buffer("means", means)
        self.register_buffer("stds", stds)
        if self.turbulent_scale_factor is not None:
            self.register_buffer("turbulent_k", torch.tensor(self.turbulent_scale_factor, dtype=torch.float32))

    @property
    def n_channels(self) -> int:
        return int(self.means.numel())

    def transform(self, fields: torch.Tensor) -> torch.Tensor:
        """Transform physical fields to model space."""
        if fields.ndim < 3:
            raise ValueError(f"Expected field tensor with channel dimension, got {tuple(fields.shape)}")
        if fields.shape[1] != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, got {fields.shape[1]}")

        out = fields.clone()
        if self.scale_turbulent:
            channel = self.turbulent_channel_idx
            k = self.turbulent_k.to(device=out.device, dtype=out.dtype)
            nu_tilde = out[:, channel : channel + 1].clamp_min(0.0)
            out[:, channel : channel + 1] = torch.log1p(k * nu_tilde)

        if self.normalize:
            means = _channel_view(self.means.to(device=out.device, dtype=out.dtype), out.ndim)
            stds = _channel_view(self.stds.to(device=out.device, dtype=out.dtype), out.ndim)
            out = (out - means) / stds
        return out

    def inverse_transform(self, fields: torch.Tensor) -> torch.Tensor:
        """Transform model-space fields back to physical fields."""
        if fields.ndim < 3:
            raise ValueError(f"Expected field tensor with channel dimension, got {tuple(fields.shape)}")
        if fields.shape[1] != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, got {fields.shape[1]}")

        out = fields.clone()
        if self.normalize:
            means = _channel_view(self.means.to(device=out.device, dtype=out.dtype), out.ndim)
            stds = _channel_view(self.stds.to(device=out.device, dtype=out.dtype), out.ndim)
            out = out * stds + means

        if self.scale_turbulent:
            channel = self.turbulent_channel_idx
            k = self.turbulent_k.to(device=out.device, dtype=out.dtype)
            log_nu = out[:, channel : channel + 1]
            out[:, channel : channel + 1] = torch.expm1(log_nu).clamp_min(0.0) / k
        return out

    def transform_blocks(self, blocks: Mapping[int, torch.Tensor]) -> dict[int, torch.Tensor]:
        return {int(k): self.transform(v) for k, v in blocks.items()}

    def inverse_transform_blocks(self, blocks: Mapping[int, torch.Tensor]) -> dict[int, torch.Tensor]:
        return {int(k): self.inverse_transform(v) for k, v in blocks.items()}

    def forward(self, fields: torch.Tensor) -> torch.Tensor:
        return self.transform(fields)

    def get_config(self) -> dict[str, Any]:
        return {
            "normalize": self.normalize,
            "scale_turbulent": self.scale_turbulent,
            "turbulent_channel_idx": self.turbulent_channel_idx,
            "turbulent_scale_factor": self.turbulent_scale_factor,
            "means": self.means.detach().cpu().tolist(),
            "stds": self.stds.detach().cpu().tolist(),
            "metadata": dict(self.metadata),
        }


def _stats_from_structured_metadata(metadata: Mapping[str, Any]) -> tuple[list[float], list[float]]:
    norm = metadata.get("norm") or {}
    means = norm.get("state_mean")
    stds = norm.get("state_std")
    if means is None or stds is None:
        raise ValueError("metadata.json is missing norm.state_mean/state_std")
    return list(means), list(stds)


def _stats_from_turbulent_scale(path: Path, turbulent_channel_idx: int) -> tuple[list[float], list[float], float]:
    with path.open("r", encoding="utf-8") as stream:
        stats = json.load(stream)
    version = str(stats.get("version", ""))
    if not version.startswith("turbulent-scale"):
        raise ValueError(f"Unsupported turbulent stats version: {version}")
    turb = stats["turbulent_channel"]
    if int(turb["index"]) != int(turbulent_channel_idx):
        raise ValueError(
            f"Turbulent-channel index mismatch: stats={turb['index']}, requested={turbulent_channel_idx}"
        )
    other_means = list(stats["other_channels"]["means"])
    other_stds = list(stats["other_channels"]["stds"])
    if len(other_means) != int(turbulent_channel_idx):
        raise ValueError(
            "Expected other_channels to contain all non-turbulent channels before "
            f"index {turbulent_channel_idx}, got {len(other_means)}"
        )
    means = other_means + [float(turb["log_mean"])]
    stds = other_stds + [float(turb["log_std"])]
    return means, stds, float(turb["scale_factor"])


def create_normalizer_from_metadata(
    metadata_path: str | Path,
    *,
    normalize: bool = True,
    scale_turbulent: bool = False,
    turbulent_stats_path: str | Path | None = None,
    turbulent_channel_idx: int = 5,
) -> Optional[WholeCaseFieldNormalizer]:
    """Create a normalizer from structured export metadata or turbulent stats."""
    effective_scale_turbulent = bool(scale_turbulent) or bool(normalize)
    if not normalize and not effective_scale_turbulent:
        return None

    metadata_path = Path(metadata_path).expanduser().resolve()
    with metadata_path.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)

    if effective_scale_turbulent:
        if turbulent_stats_path is None:
            candidate = metadata_path.parent / "turbulent_scale_stats.json"
            if not candidate.exists():
                raise FileNotFoundError(
                    "normalize=True or scale_turbulent=True requires turbulent_stats_path or "
                    f"{candidate} to exist"
                )
            turbulent_stats_path = candidate
        means, stds, scale_factor = _stats_from_turbulent_scale(
            Path(turbulent_stats_path).expanduser().resolve(),
            turbulent_channel_idx,
        )
    else:
        means, stds = _stats_from_structured_metadata(metadata)
        scale_factor = None

    return WholeCaseFieldNormalizer(
        means=torch.tensor(means, dtype=torch.float32),
        stds=torch.tensor(stds, dtype=torch.float32),
        scale_turbulent=effective_scale_turbulent,
        normalize=normalize,
        turbulent_channel_idx=turbulent_channel_idx,
        turbulent_scale_factor=scale_factor,
        metadata={
            "source": str(metadata_path),
            "version": metadata.get("version"),
            "state_fields": metadata.get("state_fields", metadata.get("export_state_fields")),
        },
    )


__all__ = [
    "WholeCaseFieldNormalizer",
    "create_normalizer_from_metadata",
]
