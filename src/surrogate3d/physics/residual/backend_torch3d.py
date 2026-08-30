"""Torch backend for structured 3D surrogate residual evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from surrogate3d.physics.pde.residual3d import (
    FLOW_RESIDUAL_NAMES,
    FULL_RESIDUAL_NAMES,
    compute_interior_residual,
)
from surrogate3d.physics.residual_sample import ResidualCaseSample


@dataclass(frozen=True)
class BlockResidualResult:
    """Residual maps and norms for one structured block."""

    block_index: int
    block_name: str
    residual: torch.Tensor
    valid_mask: torch.Tensor
    component_rms: dict[str, torch.Tensor]


@dataclass(frozen=True)
class CaseResidualResult:
    """Whole-case residual result aggregated over valid cells in all blocks."""

    blocks: dict[int, BlockResidualResult]
    component_rms: dict[str, torch.Tensor]
    score: torch.Tensor
    metadata: dict[str, Any]
    formulation: str


class TorchResidualBackend:
    """Differentiable 3D residual backend for continuity, momentum, and energy."""

    def __init__(
        self,
        *,
        weights: Mapping[str, float] | None = None,
        include_viscous: bool = True,
        include_sa: bool = True,
        dissipation_mode: str = "jameson",
        gamma: float = 1.4,
        adis: float = 2.0 / 3.0,
        acoustic_scale_factor: float = 1.0,
        vis2: float = 0.25,
        vis4: float = 0.0156,
        p_inf_corr: float = 1.0,
        rho_inf: float = 1.0,
        reference_temperature_k: float = 300.0,
        use_turbulent_viscosity: bool = True,
        nu_tilde_index: int = 5,
        cv1: float = 7.1,
        residual_filter: float = 1.0,
        prandtl: float = 0.72,
        turbulent_prandtl: float = 0.90,
        sa_use_ft2: bool = True,
        sa_approx: bool = False,
        require_adflow_wall_distance: bool = True,
    ) -> None:
        supplied = dict(weights or {})
        self.include_sa = bool(include_sa)
        self.residual_names = (
            FULL_RESIDUAL_NAMES if self.include_sa else FLOW_RESIDUAL_NAMES
        )
        self.weights = {
            name: float(supplied.get(name, 1.0))
            for name in self.residual_names
        }
        self.include_viscous = bool(include_viscous)
        self.operator_config = {
            "include_viscous": self.include_viscous,
            "include_sa": self.include_sa,
            "dissipation_mode": str(dissipation_mode),
            "gamma": float(gamma),
            "adis": float(adis),
            "acoustic_scale_factor": float(acoustic_scale_factor),
            "vis2": float(vis2),
            "vis4": float(vis4),
            "p_inf_corr": float(p_inf_corr),
            "rho_inf": float(rho_inf),
            "reference_temperature_k": float(reference_temperature_k),
            "use_turbulent_viscosity": bool(use_turbulent_viscosity),
            "nu_tilde_index": int(nu_tilde_index),
            "cv1": float(cv1),
            "residual_filter": float(residual_filter),
            "prandtl": float(prandtl),
            "turbulent_prandtl": float(turbulent_prandtl),
            "sa_use_ft2": bool(sa_use_ft2),
            "sa_approx": bool(sa_approx),
            "require_adflow_wall_distance": bool(require_adflow_wall_distance),
        }

    def _component_rms(
        self,
        residual: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        values = residual[:, :, valid_mask]
        return {
            name: torch.sqrt(torch.mean(values[:, index] ** 2))
            for index, name in enumerate(self.residual_names)
        }

    def _formulation(self) -> str:
        terms = [
            "interior",
            str(self.operator_config["dissipation_mode"]),
            "continuity",
            "momentum",
            "energy",
        ]
        terms.append("viscous" if self.include_viscous else "inviscid_diagnostic")
        if self.include_sa:
            terms.append("sa")
        return "_".join(terms)

    def compute_residual(
        self,
        sample: ResidualCaseSample,
    ) -> tuple[torch.Tensor, CaseResidualResult]:
        block_results: dict[int, BlockResidualResult] = {}
        case_values: dict[str, list[torch.Tensor]] = {
            name: [] for name in self.residual_names
        }

        for block_index, block in sample.blocks.items():
            residual, valid_mask = compute_interior_residual(
                block.fields,
                block.geometry,
                flow_conditions=sample.flow_conditions,
                **self.operator_config,
            )
            component_rms = self._component_rms(residual, valid_mask)
            block_results[int(block_index)] = BlockResidualResult(
                block_index=int(block_index),
                block_name=str(block.block_name),
                residual=residual,
                valid_mask=valid_mask,
                component_rms=component_rms,
            )
            valid_values = residual[:, :, valid_mask]
            for component_index, name in enumerate(self.residual_names):
                case_values[name].append(valid_values[:, component_index].reshape(-1))

        component_rms = {
            name: torch.sqrt(torch.mean(torch.cat(values) ** 2))
            for name, values in case_values.items()
        }
        score = -sum(
            self.weights[name] * component_rms[name]
            for name in self.residual_names
        )
        result = CaseResidualResult(
            blocks=block_results,
            component_rms=component_rms,
            score=score,
            metadata=dict(sample.metadata),
            formulation=self._formulation(),
        )
        return score, result


__all__ = [
    "BlockResidualResult",
    "CaseResidualResult",
    "TorchResidualBackend",
]
