"""ADFLOW-style interior viscous momentum fluxes for structured 3D blocks."""

from __future__ import annotations

import math

import torch

from surrogate3d.physics.block_geometry import StructuredBlockGeometry


def _transverse_cell_average(
    values: torch.Tensor,
    *,
    normal_axis: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if normal_axis == 0:
        left = values[:, :, :-1]
        right = values[:, :, 1:]
        average = lambda side: 0.25 * (
            side[:, :, :, :-1, :-1]
            + side[:, :, :, 1:, :-1]
            + side[:, :, :, :-1, 1:]
            + side[:, :, :, 1:, 1:]
        )
    elif normal_axis == 1:
        left = values[:, :, :, :-1]
        right = values[:, :, :, 1:]
        average = lambda side: 0.25 * (
            side[:, :, :-1, :, :-1]
            + side[:, :, 1:, :, :-1]
            + side[:, :, :-1, :, 1:]
            + side[:, :, 1:, :, 1:]
        )
    elif normal_axis == 2:
        left = values[:, :, :, :, :-1]
        right = values[:, :, :, :, 1:]
        average = lambda side: 0.25 * (
            side[:, :, :-1, :-1, :]
            + side[:, :, 1:, :-1, :]
            + side[:, :, :-1, 1:, :]
            + side[:, :, 1:, 1:, :]
        )
    else:
        raise ValueError(f"normal_axis must be 0, 1, or 2, got {normal_axis}")
    return average(left), average(right)


def _transverse_metric_sum(
    area_vectors: torch.Tensor,
    *,
    normal_axis: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if normal_axis == 0:
        cell_metrics = area_vectors[:-1] + area_vectors[1:]
        left = cell_metrics[:-1]
        right = cell_metrics[1:]
        metric_sum = lambda side: (
            side[:, :-1, :-1]
            + side[:, 1:, :-1]
            + side[:, :-1, 1:]
            + side[:, 1:, 1:]
        )
    elif normal_axis == 1:
        cell_metrics = area_vectors[:, :-1] + area_vectors[:, 1:]
        left = cell_metrics[:, :-1]
        right = cell_metrics[:, 1:]
        metric_sum = lambda side: (
            side[:-1, :, :-1]
            + side[1:, :, :-1]
            + side[:-1, :, 1:]
            + side[1:, :, 1:]
        )
    elif normal_axis == 2:
        cell_metrics = area_vectors[:, :, :-1] + area_vectors[:, :, 1:]
        left = cell_metrics[:, :, :-1]
        right = cell_metrics[:, :, 1:]
        metric_sum = lambda side: (
            side[:-1, :-1, :]
            + side[1:, :-1, :]
            + side[:-1, 1:, :]
            + side[1:, 1:, :]
        )
    else:
        raise ValueError(f"normal_axis must be 0, 1, or 2, got {normal_axis}")
    return metric_sum(left), metric_sum(right)


def _node_control_volumes(cell_volumes: torch.Tensor) -> torch.Tensor:
    return (
        cell_volumes[:-1, :-1, :-1]
        + cell_volumes[1:, :-1, :-1]
        + cell_volumes[:-1, 1:, :-1]
        + cell_volumes[1:, 1:, :-1]
        + cell_volumes[:-1, :-1, 1:]
        + cell_volumes[1:, :-1, 1:]
        + cell_volumes[:-1, 1:, 1:]
        + cell_volumes[1:, 1:, 1:]
    )


def compute_interior_nodal_gradients(
    values: torch.Tensor,
    geometry: StructuredBlockGeometry,
) -> torch.Tensor:
    """Compute gradients at internal vertices using ADFLOW's dual-volume stencil.

    Args:
        values: Cell-centered values with shape ``(B,N,I,J,K)``.

    Returns:
        Tensor with shape ``(B,N,3,I-1,J-1,K-1)``. The final three
        dimensions correspond to physical vertices ``1:I, 1:J, 1:K``.
    """
    if values.ndim != 5:
        raise ValueError(f"Expected values with shape (B,N,I,J,K), got {tuple(values.shape)}")

    geometry = geometry.to(values.device, dtype=values.dtype)
    numerator = torch.zeros(
        (
            values.shape[0],
            values.shape[1],
            3,
            values.shape[2] - 1,
            values.shape[3] - 1,
            values.shape[4] - 1,
        ),
        device=values.device,
        dtype=values.dtype,
    )
    area_vectors = (
        geometry.i_face_area_vectors,
        geometry.j_face_area_vectors,
        geometry.k_face_area_vectors,
    )
    for axis, face_areas in enumerate(area_vectors):
        value_left, value_right = _transverse_cell_average(values, normal_axis=axis)
        metric_left, metric_right = _transverse_metric_sum(face_areas, normal_axis=axis)
        metric_left = metric_left.movedim(-1, 0).unsqueeze(0).unsqueeze(0)
        metric_right = metric_right.movedim(-1, 0).unsqueeze(0).unsqueeze(0)
        numerator = numerator + (
            value_right.unsqueeze(2) * metric_right
            - value_left.unsqueeze(2) * metric_left
        )

    node_volumes = _node_control_volumes(geometry.cell_volumes)
    return numerator / node_volumes.unsqueeze(0).unsqueeze(0).unsqueeze(0)


def _face_gradients(
    nodal_gradients: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    i_face = 0.25 * (
        nodal_gradients[:, :, :, 1:-1, 1:-2, 1:-2]
        + nodal_gradients[:, :, :, 1:-1, 2:-1, 1:-2]
        + nodal_gradients[:, :, :, 1:-1, 1:-2, 2:-1]
        + nodal_gradients[:, :, :, 1:-1, 2:-1, 2:-1]
    )
    j_face = 0.25 * (
        nodal_gradients[:, :, :, 1:-2, 1:-1, 1:-2]
        + nodal_gradients[:, :, :, 2:-1, 1:-1, 1:-2]
        + nodal_gradients[:, :, :, 1:-2, 1:-1, 2:-1]
        + nodal_gradients[:, :, :, 2:-1, 1:-1, 2:-1]
    )
    k_face = 0.25 * (
        nodal_gradients[:, :, :, 1:-2, 1:-2, 1:-1]
        + nodal_gradients[:, :, :, 2:-1, 1:-2, 1:-1]
        + nodal_gradients[:, :, :, 1:-2, 2:-1, 1:-1]
        + nodal_gradients[:, :, :, 2:-1, 2:-1, 1:-1]
    )
    return i_face, j_face, k_face


def _correct_face_normal_gradient(
    gradient: torch.Tensor,
    value_left: torch.Tensor,
    value_right: torch.Tensor,
    center_delta: torch.Tensor,
) -> torch.Tensor:
    distance = torch.linalg.vector_norm(center_delta, dim=-1)
    direction = center_delta.movedim(-1, 0).unsqueeze(0) / distance.unsqueeze(0).unsqueeze(0)
    averaged_normal_derivative = torch.sum(gradient * direction.unsqueeze(1), dim=2)
    direct_normal_derivative = (value_right - value_left) / distance.unsqueeze(0).unsqueeze(0)
    correction = averaged_normal_derivative - direct_normal_derivative
    return gradient - correction.unsqueeze(2) * direction.unsqueeze(1)


def _correct_face_gradients(
    face_gradients: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    values: torch.Tensor,
    cell_centers: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    value_pairs = (
        (
            values[:, :, 1:-2, 2:-2, 2:-2],
            values[:, :, 2:-1, 2:-2, 2:-2],
        ),
        (
            values[:, :, 2:-2, 1:-2, 2:-2],
            values[:, :, 2:-2, 2:-1, 2:-2],
        ),
        (
            values[:, :, 2:-2, 2:-2, 1:-2],
            values[:, :, 2:-2, 2:-2, 2:-1],
        ),
    )
    center_deltas = (
        cell_centers[2:-1, 2:-2, 2:-2] - cell_centers[1:-2, 2:-2, 2:-2],
        cell_centers[2:-2, 2:-1, 2:-2] - cell_centers[2:-2, 1:-2, 2:-2],
        cell_centers[2:-2, 2:-2, 2:-1] - cell_centers[2:-2, 2:-2, 1:-2],
    )
    return tuple(
        _correct_face_normal_gradient(gradient, left, right, delta)
        for gradient, (left, right), delta in zip(
            face_gradients,
            value_pairs,
            center_deltas,
        )
    )


def compute_cell_viscosity(
    fields: torch.Tensor,
    flow_conditions: torch.Tensor,
    *,
    gamma: float = 1.4,
    reference_temperature_k: float = 300.0,
    use_turbulent_viscosity: bool = True,
    nu_tilde_index: int = 5,
    cv1: float = 7.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover ADFLOW's nondimensional laminar and SA eddy viscosities."""
    if flow_conditions.ndim == 1:
        flow_conditions = flow_conditions.unsqueeze(0)
    if flow_conditions.ndim != 2 or flow_conditions.shape[1] < 3:
        raise ValueError(
            "Viscous residual requires flow_conditions with [Mach, AoA, Reynolds]"
        )
    if flow_conditions.shape[0] != fields.shape[0]:
        raise ValueError(
            f"Flow/field batch mismatch: {flow_conditions.shape[0]} != {fields.shape[0]}"
        )

    flow = flow_conditions.to(device=fields.device, dtype=fields.dtype)
    mach = flow[:, 0]
    reynolds = flow[:, 2]
    mu_inf = mach * math.sqrt(float(gamma)) / reynolds
    temperature_ratio = fields[:, 4] / fields[:, 0]
    sutherland_ratio = 110.55 / float(reference_temperature_k)
    mu_laminar = (
        mu_inf[:, None, None, None]
        * temperature_ratio**1.5
        * (1.0 + sutherland_ratio)
        / (temperature_ratio + sutherland_ratio)
    )

    mu_eddy = torch.zeros_like(mu_laminar)
    if use_turbulent_viscosity:
        if fields.shape[1] <= nu_tilde_index:
            raise ValueError(
                f"SA eddy viscosity requires field channel {nu_tilde_index}, "
                f"got {fields.shape[1]} channels"
            )
        rho_nu_tilde = fields[:, 0] * fields[:, nu_tilde_index]
        chi = rho_nu_tilde / mu_laminar
        chi3 = chi**3
        fv1 = chi3 / (chi3 + float(cv1) ** 3)
        mu_eddy = fv1 * rho_nu_tilde
    return mu_laminar, mu_eddy


def _face_viscosities(
    cell_viscosity: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        0.5
        * (
            cell_viscosity[:, 1:-2, 2:-2, 2:-2]
            + cell_viscosity[:, 2:-1, 2:-2, 2:-2]
        ),
        0.5
        * (
            cell_viscosity[:, 2:-2, 1:-2, 2:-2]
            + cell_viscosity[:, 2:-2, 2:-1, 2:-2]
        ),
        0.5
        * (
            cell_viscosity[:, 2:-2, 2:-2, 1:-2]
            + cell_viscosity[:, 2:-2, 2:-2, 2:-1]
        ),
    )


def _viscous_stress(
    velocity_gradient: torch.Tensor,
    viscosity: torch.Tensor,
) -> torch.Tensor:
    divergence = (
        velocity_gradient[:, 0, 0]
        + velocity_gradient[:, 1, 1]
        + velocity_gradient[:, 2, 2]
    )
    strain = velocity_gradient + velocity_gradient.transpose(1, 2)
    identity = torch.eye(
        3,
        device=velocity_gradient.device,
        dtype=velocity_gradient.dtype,
    ).view(1, 3, 3, 1, 1, 1)
    return viscosity[:, None, None] * (
        strain - (2.0 / 3.0) * divergence[:, None, None] * identity
    )


def _viscous_momentum_flux(
    stress: torch.Tensor,
    area_vectors: torch.Tensor,
) -> torch.Tensor:
    area = area_vectors.movedim(-1, 0).unsqueeze(0)
    return torch.sum(stress * area[:, None], dim=2)


def _viscous_energy_flux(
    stress: torch.Tensor,
    velocity: torch.Tensor,
    sound_speed_gradient: torch.Tensor,
    heat_coefficient: torch.Tensor,
    area_vectors: torch.Tensor,
) -> torch.Tensor:
    area = area_vectors.movedim(-1, 0).unsqueeze(0)
    stress_power = torch.sum(velocity[:, :, None] * stress, dim=1)
    mechanical_flux = torch.sum(stress_power * area, dim=1)
    heat_flux = heat_coefficient * torch.sum(sound_speed_gradient[:, 0] * area, dim=1)
    return mechanical_flux + heat_flux


def _face_value_averages(
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        0.5
        * (
            values[:, :, 1:-2, 2:-2, 2:-2]
            + values[:, :, 2:-1, 2:-2, 2:-2]
        ),
        0.5
        * (
            values[:, :, 2:-2, 1:-2, 2:-2]
            + values[:, :, 2:-2, 2:-1, 2:-2]
        ),
        0.5
        * (
            values[:, :, 2:-2, 2:-2, 1:-2]
            + values[:, :, 2:-2, 2:-2, 2:-1]
        ),
    )


def compute_interior_viscous_flow_balance(
    fields: torch.Tensor,
    geometry: StructuredBlockGeometry,
    flow_conditions: torch.Tensor,
    *,
    gamma: float = 1.4,
    reference_temperature_k: float = 300.0,
    use_turbulent_viscosity: bool = True,
    nu_tilde_index: int = 5,
    cv1: float = 7.1,
    residual_filter: float = 1.0,
    prandtl: float = 0.72,
    turbulent_prandtl: float = 0.90,
) -> torch.Tensor:
    """Return integrated viscous momentum and energy divergence."""
    geometry = geometry.to(fields.device, dtype=fields.dtype)
    velocity = fields[:, 1:4]
    velocity_gradients = _correct_face_gradients(
        _face_gradients(compute_interior_nodal_gradients(velocity, geometry)),
        velocity,
        geometry.cell_centers,
    )
    sound_speed_squared = float(gamma) * fields[:, 4:5] / fields[:, 0:1]
    sound_speed_gradients = _correct_face_gradients(
        _face_gradients(
            compute_interior_nodal_gradients(sound_speed_squared, geometry)
        ),
        sound_speed_squared,
        geometry.cell_centers,
    )
    face_velocities = _face_value_averages(
        fields[:, 1:4],
    )
    mu_laminar, mu_eddy = compute_cell_viscosity(
        fields,
        flow_conditions,
        gamma=gamma,
        reference_temperature_k=reference_temperature_k,
        use_turbulent_viscosity=use_turbulent_viscosity,
        nu_tilde_index=nu_tilde_index,
        cv1=cv1,
    )
    laminar_face_viscosities = _face_viscosities(mu_laminar)
    eddy_face_viscosities = _face_viscosities(mu_eddy)
    face_areas = (
        geometry.i_face_area_vectors[2:-2, 2:-2, 2:-2],
        geometry.j_face_area_vectors[2:-2, 2:-2, 2:-2],
        geometry.k_face_area_vectors[2:-2, 2:-2, 2:-2],
    )
    face_fluxes: list[torch.Tensor] = []
    for gradient, sound_gradient, velocity_face, mu_lam, mu_eddy_face, area in zip(
        velocity_gradients,
        sound_speed_gradients,
        face_velocities,
        laminar_face_viscosities,
        eddy_face_viscosities,
        face_areas,
    ):
        viscosity = float(residual_filter) * (mu_lam + mu_eddy_face)
        stress = _viscous_stress(gradient, viscosity)
        momentum_flux = _viscous_momentum_flux(stress, area)
        heat_coefficient = float(residual_filter) * (
            mu_lam / (float(prandtl) * (float(gamma) - 1.0))
            + mu_eddy_face
            / (float(turbulent_prandtl) * (float(gamma) - 1.0))
        )
        energy_flux = _viscous_energy_flux(
            stress,
            velocity_face,
            sound_gradient,
            heat_coefficient,
            area,
        )
        face_fluxes.append(torch.cat((momentum_flux, energy_flux[:, None]), dim=1))
    i_flux, j_flux, k_flux = face_fluxes
    return (
        i_flux[:, :, 1:] - i_flux[:, :, :-1]
        + j_flux[:, :, :, 1:] - j_flux[:, :, :, :-1]
        + k_flux[:, :, :, :, 1:] - k_flux[:, :, :, :, :-1]
    )


__all__ = [
    "compute_cell_viscosity",
    "compute_interior_nodal_gradients",
    "compute_interior_viscous_flow_balance",
]
