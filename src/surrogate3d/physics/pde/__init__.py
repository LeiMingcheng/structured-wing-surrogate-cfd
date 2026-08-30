"""Differentiable 3D finite-volume operators."""

from surrogate3d.physics.pde.residual3d import (
    FLOW_RESIDUAL_NAMES,
    FULL_RESIDUAL_NAMES,
    compute_interior_residual,
)

__all__ = [
    "FLOW_RESIDUAL_NAMES",
    "FULL_RESIDUAL_NAMES",
    "compute_interior_residual",
]
