"""Public 3D residual calculator interface."""

from surrogate3d.physics.residual.backend_torch3d import (
    BlockResidualResult,
    CaseResidualResult,
    TorchResidualBackend,
)
from surrogate3d.physics.residual.calculator import PDEResidualCalculator

__all__ = [
    "BlockResidualResult",
    "CaseResidualResult",
    "PDEResidualCalculator",
    "TorchResidualBackend",
]
