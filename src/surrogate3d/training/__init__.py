"""Training helpers for surrogate3d."""

__all__ = ["WholeCaseDirectTrainer", "WholeCaseDirectTrainerConfig", "whole_case_mse"]


def __getattr__(name: str):
    if name == "WholeCaseDirectTrainer":
        from surrogate3d.training.trainer import WholeCaseDirectTrainer

        return WholeCaseDirectTrainer
    if name == "WholeCaseDirectTrainerConfig":
        from surrogate3d.training.config import WholeCaseDirectTrainerConfig

        return WholeCaseDirectTrainerConfig
    if name == "whole_case_mse":
        from surrogate3d.training.losses import whole_case_mse

        return whole_case_mse
    raise AttributeError(f"module 'surrogate3d.training' has no attribute {name!r}")
