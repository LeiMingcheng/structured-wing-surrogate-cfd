"""PIR-DM-style facade for structured 3D residual backends."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from surrogate3d.physics.residual.backend_torch3d import (
    CaseResidualResult,
    TorchResidualBackend,
)
from surrogate3d.physics.residual_sample import ResidualCaseSample


class PDEResidualCalculator:
    """Dispatch differentiable residual evaluation through the Torch backend."""

    def __init__(
        self,
        backend: str = "torch",
        *,
        weights: Mapping[str, float] | None = None,
        backend_config: Mapping[str, Any] | None = None,
    ) -> None:
        backend_name = str(backend).lower()
        if backend_name == "pytorch":
            backend_name = "torch"
        if backend_name != "torch":
            raise NotImplementedError("surrogate3d residual evaluation currently supports backend='torch' only")
        self.backend = backend_name
        self._calculator = TorchResidualBackend(
            weights=weights,
            **dict(backend_config or {}),
        )

    def compute_residual(
        self,
        sample: ResidualCaseSample,
    ) -> tuple[torch.Tensor, CaseResidualResult]:
        return self._calculator.compute_residual(sample)

    def compute_batch_residuals(
        self,
        samples: list[ResidualCaseSample],
    ) -> tuple[list[float], list[CaseResidualResult]]:
        scores: list[float] = []
        results: list[CaseResidualResult] = []
        for sample in samples:
            score, result = self.compute_residual(sample)
            scores.append(float(score.detach().cpu().item()))
            results.append(result)
        return scores, results

    def __call__(self, sample: ResidualCaseSample) -> tuple[torch.Tensor, CaseResidualResult]:
        return self.compute_residual(sample)


__all__ = ["PDEResidualCalculator"]
