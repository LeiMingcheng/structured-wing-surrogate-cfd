"""Torch surface-force utilities for structured 3D wing wall data."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch


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


def _ensure_batch(tensor: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if tensor.ndim == 2:
        return tensor.unsqueeze(0), True
    if tensor.ndim == 3:
        return tensor, False
    raise ValueError(f"Expected tensor with ndim 2 or 3, got shape {tuple(tensor.shape)}")


def _ensure_batch_areas(tensor: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if tensor.ndim == 1:
        return tensor.unsqueeze(0), True
    if tensor.ndim == 2:
        return tensor, False
    raise ValueError(f"Expected area tensor with ndim 1 or 2, got shape {tuple(tensor.shape)}")


def _ensure_batch_scalar(
    value: torch.Tensor | float,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    tensor = _as_tensor(value, device=device, dtype=dtype)
    if tensor.ndim == 0:
        tensor = tensor.expand(batch_size)
    elif tensor.ndim == 1 and int(tensor.shape[0]) == batch_size:
        pass
    else:
        raise ValueError(f"Expected scalar or (B,) tensor, got shape {tuple(tensor.shape)}")
    return tensor


def prepare_surface_force_geometry(
    *,
    centers: torch.Tensor | Any,
    area_vectors: torch.Tensor | Any,
    cell_areas: torch.Tensor | Any | None = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> dict[str, torch.Tensor]:
    """Prepare flat surface geometry tensors for force integration."""
    torch_device = torch.device(device) if device is not None else None
    centers_tensor, centers_single = _ensure_batch(
        _as_tensor(centers, device=torch_device, dtype=dtype or torch.float32)
    )
    area_vectors_tensor, area_single = _ensure_batch(
        _as_tensor(area_vectors, device=centers_tensor.device, dtype=centers_tensor.dtype)
    )
    if centers_single != area_single:
        raise ValueError("centers and area_vectors must either both share or both include batch dimensions")
    if centers_tensor.shape != area_vectors_tensor.shape:
        raise ValueError(
            f"centers and area_vectors must share shape, got {tuple(centers_tensor.shape)} "
            f"vs {tuple(area_vectors_tensor.shape)}"
        )
    if centers_tensor.shape[-1] != 3:
        raise ValueError(f"Expected last geometry dim=3, got {tuple(centers_tensor.shape)}")
    if cell_areas is None:
        cell_areas_tensor = torch.linalg.norm(area_vectors_tensor, dim=-1)
    else:
        cell_areas_tensor, areas_single = _ensure_batch_areas(
            _as_tensor(cell_areas, device=centers_tensor.device, dtype=centers_tensor.dtype)
        )
        if areas_single != centers_single:
            raise ValueError("cell_areas batch layout must match centers/area_vectors")
        if cell_areas_tensor.shape != centers_tensor.shape[:-1]:
            raise ValueError(
                f"cell_areas must have shape {tuple(centers_tensor.shape[:-1])}, "
                f"got {tuple(cell_areas_tensor.shape)}"
            )
    return {
        "centers": centers_tensor,
        "area_vectors": area_vectors_tensor,
        "cell_areas": cell_areas_tensor,
        "shared_geometry": torch.tensor(bool(centers_single), device=centers_tensor.device),
    }


def compute_surface_force_coefficients_torch(
    surface_fields: torch.Tensor | Any,
    geometry: Mapping[str, torch.Tensor],
    *,
    aoa_deg: torch.Tensor | float,
    area_ref: torch.Tensor | float,
    chord_ref: torch.Tensor | float = 1.0,
    moment_center: Sequence[float] = (0.25, 0.0, 0.0),
    cp_index: int,
    cf_indices: Sequence[int] = (11, 12, 13),
    compute_viscous: bool = True,
) -> dict[str, torch.Tensor]:
    """Integrate Cp/Cf surface fields into CL/CD/CMz coefficients."""
    fields_tensor, single_sample = _ensure_batch(
        _as_tensor(surface_fields, device=geometry["centers"].device, dtype=geometry["centers"].dtype)
    )
    centers = geometry["centers"]
    area_vectors = geometry["area_vectors"]
    cell_areas = geometry["cell_areas"]
    if int(centers.shape[0]) == 1 and int(fields_tensor.shape[0]) > 1:
        centers = centers.expand(fields_tensor.shape[0], -1, -1)
        area_vectors = area_vectors.expand(fields_tensor.shape[0], -1, -1)
        cell_areas = cell_areas.expand(fields_tensor.shape[0], -1)
    if centers.shape[0] != fields_tensor.shape[0]:
        raise ValueError(
            f"Geometry batch {tuple(centers.shape)} does not match fields {tuple(fields_tensor.shape)}"
        )
    if centers.shape[1] != fields_tensor.shape[1]:
        raise ValueError(
            f"Geometry cell count {tuple(centers.shape)} does not match fields {tuple(fields_tensor.shape)}"
        )

    batch_size = int(fields_tensor.shape[0])
    device = fields_tensor.device
    dtype = fields_tensor.dtype
    aoa = _ensure_batch_scalar(aoa_deg, batch_size=batch_size, device=device, dtype=dtype)
    area_ref_tensor = _ensure_batch_scalar(area_ref, batch_size=batch_size, device=device, dtype=dtype)
    chord_ref_tensor = _ensure_batch_scalar(chord_ref, batch_size=batch_size, device=device, dtype=dtype)
    moment_center_tensor = _as_tensor(moment_center, device=device, dtype=dtype).view(1, 1, 3)

    cp = fields_tensor[:, :, int(cp_index)]
    pressure_force_coeff = -cp.unsqueeze(-1) * area_vectors / area_ref_tensor.view(batch_size, 1, 1)

    viscous_force_coeff = torch.zeros_like(pressure_force_coeff)
    if compute_viscous:
        if len(cf_indices) != 3:
            raise ValueError(f"cf_indices must contain 3 channel indices, got {tuple(cf_indices)}")
        cf = torch.stack(
            [fields_tensor[:, :, int(index)] for index in cf_indices],
            dim=-1,
        )
        viscous_force_coeff = cf * cell_areas.unsqueeze(-1) / area_ref_tensor.view(batch_size, 1, 1)

    total_force_coeff = pressure_force_coeff + viscous_force_coeff
    moment_arm = centers - moment_center_tensor
    moment_coeff = torch.cross(moment_arm, total_force_coeff, dim=-1) / chord_ref_tensor.view(batch_size, 1, 1)

    total_force = total_force_coeff.sum(dim=1)
    pressure_force = pressure_force_coeff.sum(dim=1)
    viscous_force = viscous_force_coeff.sum(dim=1)
    total_moment = moment_coeff.sum(dim=1)

    aoa_rad = aoa * (math.pi / 180.0)
    cos_a = torch.cos(aoa_rad)
    sin_a = torch.sin(aoa_rad)
    cx = total_force[:, 0]
    cy = total_force[:, 1]
    cl = cy * cos_a - cx * sin_a
    cd = cx * cos_a + cy * sin_a
    cmz = total_moment[:, 2]

    result = {
        "CL": cl,
        "CD": cd,
        "CMz": cmz,
        "Cx": cx,
        "Cy": cy,
        "Cz": total_force[:, 2],
        "CMx": total_moment[:, 0],
        "CMy": total_moment[:, 1],
        "pressure_force_coeff": pressure_force,
        "viscous_force_coeff": viscous_force,
    }
    if single_sample:
        return {key: value[0] for key, value in result.items()}
    return result


def compute_surface_field_metrics(
    pred_surface_fields: torch.Tensor | Any,
    target_surface_fields: torch.Tensor | Any,
    *,
    cp_index: int,
    cf_indices: Sequence[int] = (11, 12, 13),
) -> dict[str, float]:
    """Compute Cp/Cf channel errors for surface batches."""
    pred_tensor = _as_tensor(pred_surface_fields, dtype=torch.float32)
    target_tensor = _as_tensor(target_surface_fields, device=pred_tensor.device, dtype=pred_tensor.dtype)
    if pred_tensor.shape != target_tensor.shape:
        raise ValueError(f"pred/target surface fields must share shape, got {pred_tensor.shape} vs {target_tensor.shape}")

    def _metric_for_channel(index: int, name: str) -> dict[str, float]:
        error = pred_tensor[..., index] - target_tensor[..., index]
        return {
            f"{name}_mae": float(torch.mean(torch.abs(error)).detach().cpu().item()),
            f"{name}_rmse": float(torch.sqrt(torch.mean(error ** 2)).detach().cpu().item()),
        }

    metrics = {}
    metrics.update(_metric_for_channel(int(cp_index), "cp"))
    for axis_name, index in zip(("cfx", "cfy", "cfz"), cf_indices):
        metrics.update(_metric_for_channel(int(index), axis_name))
    return metrics


def compute_surface_coefficient_metrics(
    pred_surface_fields: torch.Tensor | Any,
    target_surface_fields: torch.Tensor | Any,
    geometry: Mapping[str, torch.Tensor],
    *,
    aoa_deg: torch.Tensor | float,
    area_ref: torch.Tensor | float,
    chord_ref: torch.Tensor | float = 1.0,
    moment_center: Sequence[float] = (0.25, 0.0, 0.0),
    cp_index: int,
    cf_indices: Sequence[int] = (11, 12, 13),
    compute_viscous: bool = True,
) -> dict[str, float]:
    """Compute coefficient-level errors from predicted and target surface fields."""
    pred = compute_surface_force_coefficients_torch(
        pred_surface_fields,
        geometry,
        aoa_deg=aoa_deg,
        area_ref=area_ref,
        chord_ref=chord_ref,
        moment_center=moment_center,
        cp_index=cp_index,
        cf_indices=cf_indices,
        compute_viscous=compute_viscous,
    )
    target = compute_surface_force_coefficients_torch(
        target_surface_fields,
        geometry,
        aoa_deg=aoa_deg,
        area_ref=area_ref,
        chord_ref=chord_ref,
        moment_center=moment_center,
        cp_index=cp_index,
        cf_indices=cf_indices,
        compute_viscous=compute_viscous,
    )

    metrics: dict[str, float] = {}
    for name in ("CL", "CD", "CMz"):
        pred_value = _as_tensor(pred[name], dtype=torch.float32)
        target_value = _as_tensor(target[name], device=pred_value.device, dtype=pred_value.dtype)
        error = pred_value - target_value
        metrics[f"force_pred_{name}_mean"] = float(pred_value.mean().detach().cpu().item())
        metrics[f"force_target_{name}_mean"] = float(target_value.mean().detach().cpu().item())
        metrics[f"force_mae_{name}"] = float(torch.mean(torch.abs(error)).detach().cpu().item())
        metrics[f"force_rmse_{name}"] = float(torch.sqrt(torch.mean(error ** 2)).detach().cpu().item())
    return metrics


__all__ = [
    "compute_surface_coefficient_metrics",
    "compute_surface_field_metrics",
    "compute_surface_force_coefficients_torch",
    "prepare_surface_force_geometry",
]
