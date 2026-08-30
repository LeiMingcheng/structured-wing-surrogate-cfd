"""Base classes for 3D whole-case surrogate models."""

from __future__ import annotations

from typing import Any

from surrogate.common import BaseDirectModel


class BaseWholeCaseDirectModel(BaseDirectModel):
    """Base direct model with PIR-DM-compatible metadata and 3D field names."""

    def __init__(
        self,
        geometry_dim: int = 46,
        flow_condition_dim: int = 3,
        coord_channels: int = 6,
        out_channels: int = 6,
        **params: Any,
    ) -> None:
        super().__init__(
            geometry_dim=geometry_dim,
            flow_condition_dim=flow_condition_dim,
            coord_channels=coord_channels,
            out_channels=out_channels,
            **params,
        )
        self.field_names = [
            "Density",
            "VelocityX",
            "VelocityY",
            "VelocityZ",
            "Pressure",
            "TurbulentSANuTilde",
        ][: self.out_channels]


__all__ = ["BaseWholeCaseDirectModel"]
