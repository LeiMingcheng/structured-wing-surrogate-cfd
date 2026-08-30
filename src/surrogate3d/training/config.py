"""Training config extensions for whole-case 3D direct models."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import fields
from typing import Any
from typing import Optional

from surrogate.direct.training import DirectTrainerConfig


@dataclass
class WholeCaseDirectTrainerConfig(DirectTrainerConfig):
    """PIR-DM direct trainer config plus whole-case loss controls."""

    per_block_equal_weight: bool = True
    loss_block_indices: Optional[list[int]] = None
    volume_weighted_mse_weight: float = 0.0
    surface_cp_weight: float = 0.0
    surface_cf_weight: float = 0.0
    surface_loss_dtype: str = "float32"

    def validate(self) -> None:
        validate = getattr(super(), "validate", None)
        if callable(validate):
            validate()
        self.per_block_equal_weight = bool(self.per_block_equal_weight)
        self.volume_weighted_mse_weight = float(self.volume_weighted_mse_weight)
        self.surface_cp_weight = float(self.surface_cp_weight)
        self.surface_cf_weight = float(self.surface_cf_weight)
        self.surface_loss_dtype = str(self.surface_loss_dtype).lower()
        if self.loss_block_indices is not None:
            self.loss_block_indices = [int(value) for value in self.loss_block_indices]
        if self.volume_weighted_mse_weight < 0.0:
            raise ValueError("training.loss.volume_weighted_mse_weight must be non-negative")
        if self.surface_cp_weight < 0.0:
            raise ValueError("training.loss.surface_cp_weight must be non-negative")
        if self.surface_cf_weight < 0.0:
            raise ValueError("training.loss.surface_cf_weight must be non-negative")
        if self.surface_loss_dtype not in {"float32", "float64"}:
            raise ValueError("training.loss.surface_loss_dtype must be float32 or float64")


def whole_case_trainer_config_from_dict(values: dict[str, Any] | None) -> WholeCaseDirectTrainerConfig:
    """Build trainer config while ignoring unrelated YAML sections."""
    values = dict(values or {})
    loss_values = values.pop("loss", None)
    if isinstance(loss_values, dict):
        for key, value in loss_values.items():
            values.setdefault(key, value)
    allowed = {field.name for field in fields(WholeCaseDirectTrainerConfig)}
    data = {key: value for key, value in values.items() if key in allowed}
    config = WholeCaseDirectTrainerConfig(**data)
    validate = getattr(config, "validate", None)
    if callable(validate):
        validate()
    return config


__all__ = ["WholeCaseDirectTrainerConfig", "whole_case_trainer_config_from_dict"]
