"""Shared schema constants for whole-case structured wing surrogate data."""

STATE_FIELDS = (
    "Density",
    "VelocityX",
    "VelocityY",
    "VelocityZ",
    "Pressure",
    "TurbulentSANuTilde",
)

FLOW_NAMES = ("mach", "aoa", "reynolds")
FORCE_NAMES = ("wing_cl", "wing_cd", "wing_cmz")
MODEL_COORD_FIELDS = ("x", "y", "z", "i_norm", "j_norm", "k_norm")
PDE_COORD_FIELDS = ("x", "y", "z")

__all__ = ["STATE_FIELDS", "FLOW_NAMES", "FORCE_NAMES", "MODEL_COORD_FIELDS", "PDE_COORD_FIELDS"]
