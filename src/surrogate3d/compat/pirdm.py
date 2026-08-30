"""Import checks for the PIR-DM surrogate package."""

from __future__ import annotations

from importlib import import_module


def require_pirdm_surrogate() -> None:
    """Raise a clear error when PIR-DM's ``surrogate`` package is unavailable."""
    try:
        import_module("surrogate.common")
        import_module("surrogate.direct.training")
        import_module("surrogate.training.loss_config")
    except Exception as exc:  # pragma: no cover - import environment dependent
        raise ImportError(
            "surrogate3d requires the top-level 'surrogate' package from "
            "surrogate-newton-cfd. Install the locked foundation revision "
            "before using model or training modules."
        ) from exc


__all__ = ["require_pirdm_surrogate"]
