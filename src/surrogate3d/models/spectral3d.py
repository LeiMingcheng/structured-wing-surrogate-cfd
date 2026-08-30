"""Minimal 3D spectral layers for surrogate3d."""

from __future__ import annotations

import torch
import torch.nn as nn


class FourierSpectralConv3d(nn.Module):
    """FNO-style spectral convolution for ``(B, C, I, J, K)`` tensors."""

    _N_CORNERS = 4

    def __init__(self, in_channels: int, out_channels: int, modes: tuple[int, int, int]) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes = tuple(int(v) for v in modes)
        scale = 1.0 / max(1, self.in_channels * self.out_channels)
        self.weights = nn.Parameter(
            scale
            * torch.randn(
                self._N_CORNERS,
                self.in_channels,
                self.out_channels,
                self.modes[0],
                self.modes[1],
                self.modes[2],
                dtype=torch.cfloat,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"FourierSpectralConv3d expects 5D tensor, got {tuple(x.shape)}")
        batch, _, ni, nj, nk = x.shape
        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1))
        out_ft = torch.zeros(
            batch,
            self.out_channels,
            ni,
            nj,
            nk // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )
        # rfftn keeps full signed spectra in the first two transformed
        # dimensions and only truncates the last dimension. Use all four
        # low-frequency corners for a proper 3D FNO update.
        mi = min(self.modes[0], max(1, ni // 2)) if ni > 1 else 1
        mj = min(self.modes[1], max(1, nj // 2)) if nj > 1 else 1
        mk = min(self.modes[2], nk // 2 + 1)

        i_slices = [slice(0, mi)]
        j_slices = [slice(0, mj)]
        if ni > 1:
            i_slices.append(slice(-mi, None))
        if nj > 1:
            j_slices.append(slice(-mj, None))

        corner = 0
        for i_slice in i_slices:
            for j_slice in j_slices:
                out_ft[:, :, i_slice, j_slice, :mk] = torch.einsum(
                    "bcijk,coijk->boijk",
                    x_ft[:, :, i_slice, j_slice, :mk],
                    self.weights[corner, :, :, :mi, :mj, :mk],
                )
                corner += 1
        return torch.fft.irfftn(out_ft, s=(ni, nj, nk), dim=(-3, -2, -1))


class SpectralBlock3d(nn.Module):
    """Spectral update plus local and channel-mixing residual paths."""

    def __init__(self, width: int, modes: tuple[int, int, int], condition_dim: int = 0) -> None:
        super().__init__()
        self.spectral = FourierSpectralConv3d(width, width, modes)
        self.skip = nn.Conv3d(width, width, kernel_size=1)
        self.local = nn.Conv3d(width, width, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(1, width)
        self.mlp = nn.Sequential(
            nn.Conv3d(width, width * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(width * 2, width, kernel_size=1),
        )
        self.condition_proj = (
            nn.Sequential(nn.SiLU(), nn.Linear(condition_dim, width * 2))
            if int(condition_dim) > 0
            else None
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        residual = self.spectral(x) + self.skip(x) + self.local(x)
        if self.condition_proj is not None and condition is not None:
            scale_shift = self.condition_proj(condition)
            scale, shift = scale_shift.chunk(2, dim=1)
            residual = residual * (1.0 + scale[..., None, None, None]) + shift[..., None, None, None]
        x = x + residual
        x = x + self.mlp(self.norm(x))
        return x


__all__ = ["FourierSpectralConv3d", "SpectralBlock3d"]
