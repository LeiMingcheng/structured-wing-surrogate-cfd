"""Field-based wall-force recovery from surrogate3d volume predictions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from surrogate3d.data.schema import STATE_FIELDS
from surrogate3d.physics.surface_forces import (
    compute_surface_coefficient_metrics,
    compute_surface_field_metrics,
    compute_surface_force_coefficients_torch,
    prepare_surface_force_geometry,
)
from surrogate3d.physics.wall_geometry import BlockWallGeometry


def _as_tensor(
    value: Any,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value
        if device is not None or dtype is not None:
            tensor = tensor.to(device=device or tensor.device, dtype=dtype or tensor.dtype)
        return tensor
    return torch.as_tensor(value, device=device, dtype=dtype)


def _ensure_block_batch(fields: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if fields.ndim == 4:
        return fields.unsqueeze(0), True
    if fields.ndim == 5:
        return fields, False
    raise ValueError(f"Expected block fields with shape (C,I,J,K) or (B,C,I,J,K), got {tuple(fields.shape)}")


def _ensure_flow_batch(
    flow_conditions: torch.Tensor | Any,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    flow = _as_tensor(flow_conditions, device=device, dtype=dtype)
    if flow.ndim == 1:
        if int(flow.shape[0]) < 2:
            raise ValueError(f"Expected at least [mach, aoa] in flow_conditions, got {tuple(flow.shape)}")
        flow = flow.unsqueeze(0).expand(batch_size, -1)
    elif flow.ndim == 2 and int(flow.shape[0]) == batch_size:
        pass
    else:
        raise ValueError(
            f"Expected flow_conditions with shape (3,) or (B,3), got {tuple(flow.shape)} for batch_size={batch_size}"
        )
    return flow


def _resolve_state_index(name: str, explicit: int | None) -> int:
    if explicit is not None:
        return int(explicit)
    return int(STATE_FIELDS.index(name))


def _compute_sa_eddy_viscosity(
    *,
    rho: torch.Tensor,
    nu_tilde: torch.Tensor,
    mu_lam: torch.Tensor,
    cv1: float = 7.1,
) -> torch.Tensor:
    chi = rho * torch.clamp(nu_tilde, min=0.0) / (mu_lam + 1.0e-14)
    chi3 = chi ** 3
    cv13 = float(cv1) ** 3
    fv1 = chi3 / (chi3 + cv13)
    return fv1 * rho * torch.clamp(nu_tilde, min=0.0)


@dataclass(frozen=True)
class ForceEvaluatorConfig:
    """Controls for field-based 3D force evaluation from volume predictions."""

    mode: str = "full"
    gamma: float = 1.4
    pressure_index: int | None = None
    density_index: int | None = None
    velocity_indices: tuple[int, int, int] = (1, 2, 3)
    nu_tilde_index: int | None = None
    use_turbulent_viscosity: bool = True
    cv1: float = 7.1
    reference_temperature_k: float = 300.0

    def __post_init__(self) -> None:
        if self.mode not in {"full", "pressure_only"}:
            raise ValueError(f"Unsupported force-evaluation mode '{self.mode}'")

    def pressure_channel(self) -> int:
        return _resolve_state_index("Pressure", self.pressure_index)

    def density_channel(self) -> int:
        return _resolve_state_index("Density", self.density_index)

    def nu_tilde_channel(self) -> int:
        return _resolve_state_index("TurbulentSANuTilde", self.nu_tilde_index)

    def includes_viscous_force(self) -> bool:
        return self.mode == "full"

    def is_complete_force(self) -> bool:
        return self.includes_viscous_force()

    def viscous_model(self) -> str | None:
        return "first_cell_proxy" if self.includes_viscous_force() else None


def force_evaluator_config_from_dict(config: Mapping[str, Any] | None = None) -> ForceEvaluatorConfig:
    payload = dict(config or {})
    if "velocity_indices" in payload:
        payload["velocity_indices"] = tuple(int(value) for value in payload["velocity_indices"])
    return ForceEvaluatorConfig(**payload)


def prepare_volume_force_geometry(
    block_geometry: Mapping[int, BlockWallGeometry],
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> dict[str, Any]:
    """Flatten per-block wall geometry for shared force integration."""
    if not block_geometry:
        raise ValueError("prepare_volume_force_geometry requires at least one wall-geometry block")

    torch_device = torch.device(device) if device is not None else None
    block_slices: dict[int, slice] = {}
    centers_parts: list[torch.Tensor] = []
    area_parts: list[torch.Tensor] = []
    cell_area_parts: list[torch.Tensor] = []
    block_cell_counts: dict[int, int] = {}
    cursor = 0
    for block_index in sorted(int(key) for key in block_geometry.keys()):
        geometry = block_geometry[int(block_index)]
        centers = _as_tensor(geometry.face_centers, device=torch_device, dtype=dtype or geometry.face_centers.dtype)
        area_vectors = _as_tensor(
            geometry.area_vectors,
            device=centers.device,
            dtype=centers.dtype,
        )
        cell_areas = _as_tensor(
            geometry.cell_areas,
            device=centers.device,
            dtype=centers.dtype,
        )
        n_cells = int(cell_areas.numel())
        block_slices[int(block_index)] = slice(cursor, cursor + n_cells)
        block_cell_counts[int(block_index)] = n_cells
        cursor += n_cells
        centers_parts.append(centers.reshape(-1, 3))
        area_parts.append(area_vectors.reshape(-1, 3))
        cell_area_parts.append(cell_areas.reshape(-1))

    flat_centers = torch.cat(centers_parts, dim=0)
    flat_area_vectors = torch.cat(area_parts, dim=0)
    flat_cell_areas = torch.cat(cell_area_parts, dim=0)
    return {
        "geometry": prepare_surface_force_geometry(
            centers=flat_centers,
            area_vectors=flat_area_vectors,
            cell_areas=flat_cell_areas,
            device=flat_centers.device,
            dtype=flat_centers.dtype,
        ),
        "block_slices": block_slices,
        "block_cell_counts": block_cell_counts,
        "n_surface_cells": int(flat_cell_areas.numel()),
    }


def compute_volume_surface_fields_torch(
    block_fields: Mapping[int, torch.Tensor | Any],
    block_geometry: Mapping[int, BlockWallGeometry],
    flow_conditions: torch.Tensor | Any,
    *,
    config: ForceEvaluatorConfig | Mapping[str, Any] | None = None,
    prepared_geometry: Mapping[str, Any] | None = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> dict[str, Any]:
    """Recover wall Cp/Cf-like fields from 3D volume predictions."""
    cfg = config if isinstance(config, ForceEvaluatorConfig) else force_evaluator_config_from_dict(config)
    if not block_fields:
        raise ValueError("compute_volume_surface_fields_torch requires at least one block field")

    first_block_index = sorted(int(key) for key in block_fields.keys())[0]
    first_block = _as_tensor(block_fields[first_block_index], device=None, dtype=None)
    first_block, single_sample = _ensure_block_batch(first_block)
    batch_size = int(first_block.shape[0])
    torch_device = torch.device(device) if device is not None else first_block.device
    torch_dtype = dtype or first_block.dtype
    flow = _ensure_flow_batch(
        flow_conditions,
        batch_size=batch_size,
        device=torch_device,
        dtype=torch_dtype,
    )
    prepared = (
        dict(prepared_geometry)
        if prepared_geometry is not None
        else prepare_volume_force_geometry(block_geometry, device=torch_device, dtype=torch_dtype)
    )

    q_nondim = 0.5 * float(cfg.gamma) * flow[:, 0].clamp_min(1.0e-6) ** 2
    q_nondim_view = q_nondim.view(batch_size, 1, 1)
    reynolds = (
        flow[:, 2]
        if int(flow.shape[1]) > 2
        else torch.full((batch_size,), 1.0e6, device=torch_device, dtype=torch_dtype)
    )
    mu_inf = flow[:, 0] * math.sqrt(float(cfg.gamma)) / (reynolds + 1.0e-30)
    sutherland_ratio = 110.55 / float(cfg.reference_temperature_k)

    pressure_index = cfg.pressure_channel()
    density_index = cfg.density_channel()
    nu_tilde_index = cfg.nu_tilde_channel()
    velocity_indices = tuple(int(value) for value in cfg.velocity_indices)
    if len(velocity_indices) != 3:
        raise ValueError(f"velocity_indices must contain 3 channels, got {velocity_indices}")

    cp_parts: list[torch.Tensor] = []
    cf_parts: list[torch.Tensor] = []
    details_by_block: dict[int, dict[str, torch.Tensor]] = {}

    for block_index in sorted(int(key) for key in block_fields.keys()):
        if int(block_index) not in block_geometry:
            raise KeyError(f"Missing wall geometry for block {block_index}")

        fields = _as_tensor(block_fields[int(block_index)], device=torch_device, dtype=torch_dtype)
        fields, block_single = _ensure_block_batch(fields)
        if int(fields.shape[0]) != batch_size:
            raise ValueError(
                f"Block {block_index} batch size {int(fields.shape[0])} does not match expected {batch_size}"
            )
        if block_single != single_sample:
            single_sample = False

        geometry = block_geometry[int(block_index)]
        normal = _as_tensor(geometry.normal, device=torch_device, dtype=torch_dtype).reshape(1, *geometry.normal.shape)
        wall_distance = _as_tensor(
            geometry.wall_distance,
            device=torch_device,
            dtype=torch_dtype,
        ).reshape(1, *geometry.wall_distance.shape)

        rho = torch.clamp(fields[:, density_index, :, :, 0], min=1.0e-12)
        pressure = fields[:, pressure_index, :, :, 0]
        velocity = torch.stack([fields[:, index, :, :, 0] for index in velocity_indices], dim=-1)

        cp = (pressure - 1.0) / (q_nondim_view + 1.0e-12)
        cp_parts.append(cp.reshape(batch_size, -1))

        cf = torch.zeros(*cp.shape, 3, device=torch_device, dtype=torch_dtype)
        if cfg.includes_viscous_force():
            vel_normal = torch.sum(velocity * normal, dim=-1, keepdim=True)
            vel_tangent = velocity - vel_normal * normal
            temperature_ratio = torch.clamp(pressure / rho, min=1.0e-12)
            mu_lam = (
                mu_inf.view(batch_size, 1, 1)
                * torch.pow(temperature_ratio, 1.5)
                * (1.0 + sutherland_ratio)
                / (temperature_ratio + sutherland_ratio)
            )
            mu_eff = mu_lam
            if cfg.use_turbulent_viscosity and int(fields.shape[1]) > nu_tilde_index:
                nu_tilde = torch.clamp(fields[:, nu_tilde_index, :, :, 0], min=0.0)
                mu_eff = mu_eff + _compute_sa_eddy_viscosity(
                    rho=rho,
                    nu_tilde=nu_tilde,
                    mu_lam=mu_lam,
                    cv1=float(cfg.cv1),
                )
            tau_wall = mu_eff.unsqueeze(-1) * vel_tangent / wall_distance.unsqueeze(-1)
            cf = tau_wall / (q_nondim_view.unsqueeze(-1) + 1.0e-12)

        cf_parts.append(cf.reshape(batch_size, -1, 3))
        details_by_block[int(block_index)] = {
            "cp": cp,
            "cf": cf,
            "pressure": pressure,
            "velocity_tangent": velocity - torch.sum(velocity * normal, dim=-1, keepdim=True) * normal,
        }

    cp_flat = torch.cat(cp_parts, dim=1)
    cf_flat = torch.cat(cf_parts, dim=1)
    surface_fields = torch.cat([cp_flat.unsqueeze(-1), cf_flat], dim=-1)
    if single_sample:
        surface_fields_out = surface_fields[0]
    else:
        surface_fields_out = surface_fields

    return {
        "surface_fields": surface_fields_out,
        "surface_fields_batched": surface_fields,
        "prepared_geometry": prepared,
        "details_by_block": details_by_block,
        "single_sample": bool(single_sample),
    }


def compute_force_coefficients_torch(
    block_fields: Mapping[int, torch.Tensor | Any],
    block_geometry: Mapping[int, BlockWallGeometry],
    flow_conditions: torch.Tensor | Any,
    *,
    area_ref: torch.Tensor | float,
    chord_ref: torch.Tensor | float = 1.0,
    moment_center: Sequence[float] = (0.25, 0.0, 0.0),
    config: ForceEvaluatorConfig | Mapping[str, Any] | None = None,
    prepared_geometry: Mapping[str, Any] | None = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
    return_surface_fields: bool = False,
) -> dict[str, Any]:
    """Compute CL/CD/CMz directly from volume-predicted wall-adjacent cells."""
    cfg = config if isinstance(config, ForceEvaluatorConfig) else force_evaluator_config_from_dict(config)
    volume_surface = compute_volume_surface_fields_torch(
        block_fields,
        block_geometry,
        flow_conditions,
        config=cfg,
        prepared_geometry=prepared_geometry,
        device=device,
        dtype=dtype,
    )
    coeffs = compute_surface_force_coefficients_torch(
        volume_surface["surface_fields"],
        volume_surface["prepared_geometry"]["geometry"],
        aoa_deg=_ensure_flow_batch(
            flow_conditions,
            batch_size=volume_surface["surface_fields_batched"].shape[0],
            device=volume_surface["surface_fields_batched"].device,
            dtype=volume_surface["surface_fields_batched"].dtype,
        )[:, 1],
        area_ref=area_ref,
        chord_ref=chord_ref,
        moment_center=moment_center,
        cp_index=0,
        cf_indices=(1, 2, 3),
        compute_viscous=cfg.includes_viscous_force(),
    )
    result: dict[str, Any] = dict(coeffs)
    if return_surface_fields:
        result["surface_fields"] = volume_surface["surface_fields"]
        result["prepared_geometry"] = volume_surface["prepared_geometry"]
        result["details_by_block"] = volume_surface["details_by_block"]
    return result


def compute_force_metric_bundle(
    pred_block_fields: Mapping[int, torch.Tensor | Any],
    target_block_fields: Mapping[int, torch.Tensor | Any],
    block_geometry: Mapping[int, BlockWallGeometry],
    flow_conditions: torch.Tensor | Any,
    *,
    area_ref: torch.Tensor | float,
    chord_ref: torch.Tensor | float = 1.0,
    moment_center: Sequence[float] = (0.25, 0.0, 0.0),
    config: ForceEvaluatorConfig | Mapping[str, Any] | None = None,
    prepared_geometry: Mapping[str, Any] | None = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> dict[str, Any]:
    """Compare predicted and target volume fields in Cp/Cf and force coefficients."""
    cfg = config if isinstance(config, ForceEvaluatorConfig) else force_evaluator_config_from_dict(config)
    prepared = (
        dict(prepared_geometry)
        if prepared_geometry is not None
        else prepare_volume_force_geometry(block_geometry, device=device, dtype=dtype)
    )
    pred_surface = compute_volume_surface_fields_torch(
        pred_block_fields,
        block_geometry,
        flow_conditions,
        config=cfg,
        prepared_geometry=prepared,
        device=device,
        dtype=dtype,
    )
    target_surface = compute_volume_surface_fields_torch(
        target_block_fields,
        block_geometry,
        flow_conditions,
        config=cfg,
        prepared_geometry=prepared,
        device=device,
        dtype=dtype,
    )
    field_metrics = compute_surface_field_metrics(
        pred_surface["surface_fields"],
        target_surface["surface_fields"],
        cp_index=0,
        cf_indices=(1, 2, 3),
    )
    coeff_metrics = compute_surface_coefficient_metrics(
        pred_surface["surface_fields"],
        target_surface["surface_fields"],
        prepared["geometry"],
        aoa_deg=_ensure_flow_batch(
            flow_conditions,
            batch_size=pred_surface["surface_fields_batched"].shape[0],
            device=pred_surface["surface_fields_batched"].device,
            dtype=pred_surface["surface_fields_batched"].dtype,
        )[:, 1],
        area_ref=area_ref,
        chord_ref=chord_ref,
        moment_center=moment_center,
        cp_index=0,
        cf_indices=(1, 2, 3),
        compute_viscous=cfg.includes_viscous_force(),
    )
    return {
        "field_metrics": field_metrics,
        "coefficient_metrics": coeff_metrics,
        "prepared_geometry": prepared,
        "pred_surface_fields": pred_surface["surface_fields"],
        "target_surface_fields": target_surface["surface_fields"],
    }


__all__ = [
    "ForceEvaluatorConfig",
    "compute_force_coefficients_torch",
    "compute_force_metric_bundle",
    "compute_volume_surface_fields_torch",
    "prepare_volume_force_geometry",
    "force_evaluator_config_from_dict",
]
