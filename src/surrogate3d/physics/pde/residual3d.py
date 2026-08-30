"""Interior finite-volume residual operators for primitive 3D flow fields."""

from __future__ import annotations

import torch

from surrogate3d.physics.block_geometry import StructuredBlockGeometry
from surrogate3d.physics.pde.viscous3d import (
    compute_interior_viscous_flow_balance,
)
from surrogate3d.physics.pde.sa3d import compute_interior_sa_residual


FLOW_RESIDUAL_NAMES = ("Rc", "Rmx", "Rmy", "Rmz", "RE")
FULL_RESIDUAL_NAMES = (*FLOW_RESIDUAL_NAMES, "RSA")


def _central_inviscid_face_flux(
    left: torch.Tensor,
    right: torch.Tensor,
    area_vectors: torch.Tensor,
    *,
    gamma: float,
) -> torch.Tensor:
    """Return central inviscid mass and momentum flux through structured faces."""
    rho_left = left[:, 0]
    rho_right = right[:, 0]
    velocity_left = left[:, 1:4]
    velocity_right = right[:, 1:4]
    pressure_left = left[:, 4]
    pressure_right = right[:, 4]

    area = area_vectors.movedim(-1, 0).unsqueeze(0)
    normal_velocity_left = torch.sum(velocity_left * area, dim=1)
    normal_velocity_right = torch.sum(velocity_right * area, dim=1)
    mass_left = rho_left * normal_velocity_left
    mass_right = rho_right * normal_velocity_right

    mass_flux = 0.5 * (mass_left + mass_right)
    momentum_flux = 0.5 * (
        velocity_left * mass_left.unsqueeze(1)
        + velocity_right * mass_right.unsqueeze(1)
        + (pressure_left + pressure_right).unsqueeze(1) * area
    )
    kinetic_left = 0.5 * rho_left * torch.sum(velocity_left**2, dim=1)
    kinetic_right = 0.5 * rho_right * torch.sum(velocity_right**2, dim=1)
    total_energy_left = pressure_left / (float(gamma) - 1.0) + kinetic_left
    total_energy_right = pressure_right / (float(gamma) - 1.0) + kinetic_right
    energy_flux = 0.5 * (
        normal_velocity_left * (total_energy_left + pressure_left)
        + normal_velocity_right * (total_energy_right + pressure_right)
    )
    return torch.cat(
        (mass_flux.unsqueeze(1), momentum_flux, energy_flux.unsqueeze(1)),
        dim=1,
    )


def _interior_face_fluxes(
    fields: torch.Tensor,
    geometry: StructuredBlockGeometry,
    *,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    i_flux = _central_inviscid_face_flux(
        fields[:, :, :-1, :, :],
        fields[:, :, 1:, :, :],
        geometry.i_face_area_vectors[1:-1, :, :, :],
        gamma=gamma,
    )
    j_flux = _central_inviscid_face_flux(
        fields[:, :, :, :-1, :],
        fields[:, :, :, 1:, :],
        geometry.j_face_area_vectors[:, 1:-1, :, :],
        gamma=gamma,
    )
    k_flux = _central_inviscid_face_flux(
        fields[:, :, :, :, :-1],
        fields[:, :, :, :, 1:],
        geometry.k_face_area_vectors[:, :, 1:-1, :],
        gamma=gamma,
    )
    return i_flux, j_flux, k_flux


def _cell_spectral_radii(
    fields: torch.Tensor,
    geometry: StructuredBlockGeometry,
    *,
    gamma: float,
    adis: float,
    acoustic_scale_factor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    velocity = fields[:, 1:4]
    sound_speed_sq = gamma * fields[:, 4] / fields[:, 0]

    def _directional_radius(area_vectors: torch.Tensor, axis: int) -> torch.Tensor:
        if axis == 0:
            area_sum = area_vectors[1:] + area_vectors[:-1]
        elif axis == 1:
            area_sum = area_vectors[:, 1:] + area_vectors[:, :-1]
        else:
            area_sum = area_vectors[:, :, 1:] + area_vectors[:, :, :-1]
        area = area_sum.movedim(-1, 0).unsqueeze(0)
        projected_velocity = torch.sum(velocity * area, dim=1)
        acoustic = torch.sqrt(sound_speed_sq * torch.sum(area * area, dim=1))
        return 0.5 * (torch.abs(projected_velocity) + acoustic_scale_factor * acoustic)

    radius_i = _directional_radius(geometry.i_face_area_vectors, 0)
    radius_j = _directional_radius(geometry.j_face_area_vectors, 1)
    radius_k = _directional_radius(geometry.k_face_area_vectors, 2)
    radius_floor = torch.finfo(fields.dtype).eps
    radius_i = radius_i.clamp_min(radius_floor)
    radius_j = radius_j.clamp_min(radius_floor)
    radius_k = radius_k.clamp_min(radius_floor)

    ratio_ij = (radius_i / radius_j) ** adis
    ratio_jk = (radius_j / radius_k) ** adis
    ratio_ki = (radius_k / radius_i) ** adis
    scaled_i = radius_i * (1.0 + 1.0 / ratio_ij + ratio_ki)
    scaled_j = radius_j * (1.0 + 1.0 / ratio_jk + ratio_ij)
    scaled_k = radius_k * (1.0 + 1.0 / ratio_ki + ratio_jk)
    return scaled_i, scaled_j, scaled_k


def _entropy_sensor(
    fields: torch.Tensor,
    *,
    gamma: float,
    p_inf_corr: float,
    rho_inf: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    entropy = fields[:, 4] / fields[:, 0] ** gamma
    sensor_limit = 0.001 * p_inf_corr / rho_inf**gamma

    def _axis_sensor(axis: int) -> torch.Tensor:
        values = entropy.movedim(axis, -1)
        second_difference = values[..., 2:] - 2.0 * values[..., 1:-1] + values[..., :-2]
        denominator = values[..., 2:] + 2.0 * values[..., 1:-1] + values[..., :-2] + sensor_limit
        interior = torch.abs(second_difference / denominator)
        sensor = torch.zeros_like(values)
        sensor[..., 1:-1] = interior
        return sensor.movedim(-1, axis)

    return _axis_sensor(1), _axis_sensor(2), _axis_sensor(3)


def _dissipation_flow_state(fields: torch.Tensor, *, gamma: float) -> torch.Tensor:
    density = fields[:, 0:1]
    velocity = fields[:, 1:4]
    pressure = fields[:, 4:5]
    total_energy = pressure / (float(gamma) - 1.0) + 0.5 * density * torch.sum(
        velocity**2,
        dim=1,
        keepdim=True,
    )
    return torch.cat(
        (
            density,
            density * velocity,
            total_energy + pressure,
        ),
        dim=1,
    )


def _jameson_face_dissipation(
    conservative: torch.Tensor,
    spectral_radius: torch.Tensor,
    sensor: torch.Tensor,
    *,
    spatial_axis: int,
    vis2: float,
    vis4: float,
    dss_max: float,
) -> torch.Tensor:
    state = conservative.movedim(spatial_axis, -1)
    radius = spectral_radius.movedim(spatial_axis - 1, -1)
    switch = sensor.movedim(spatial_axis - 1, -1)

    left_left = state[..., :-3]
    left = state[..., 1:-2]
    right = state[..., 2:-1]
    right_right = state[..., 3:]
    first_difference = right - left
    third_difference = right_right - left_left - 3.0 * first_difference

    face_radius = 0.5 * (radius[..., 1:-2] + radius[..., 2:-1])
    face_sensor = torch.maximum(switch[..., 1:-2], switch[..., 2:-1]).clamp_max(dss_max)
    second_order = vis2 * face_radius * face_sensor
    fourth_order = torch.clamp(vis4 * face_radius - second_order, min=0.0)
    dissipation = (
        second_order.unsqueeze(1) * first_difference
        - fourth_order.unsqueeze(1) * third_difference
    )
    return dissipation.movedim(-1, spatial_axis)


def _jameson_face_fluxes(
    fields: torch.Tensor,
    geometry: StructuredBlockGeometry,
    central_fluxes: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    gamma: float,
    adis: float,
    acoustic_scale_factor: float,
    vis2: float,
    vis4: float,
    dss_max: float,
    p_inf_corr: float,
    rho_inf: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    radii = _cell_spectral_radii(
        fields,
        geometry,
        gamma=gamma,
        adis=adis,
        acoustic_scale_factor=acoustic_scale_factor,
    )
    sensors = _entropy_sensor(
        fields,
        gamma=gamma,
        p_inf_corr=p_inf_corr,
        rho_inf=rho_inf,
    )
    conservative = _dissipation_flow_state(fields, gamma=gamma)
    dissipation = tuple(
        _jameson_face_dissipation(
            conservative,
            radius,
            sensor,
            spatial_axis=axis + 2,
            vis2=vis2,
            vis4=vis4,
            dss_max=dss_max,
        )
        for axis, (radius, sensor) in enumerate(zip(radii, sensors))
    )
    i_flux, j_flux, k_flux = central_fluxes
    return (
        i_flux[:, :, 1:-1, :, :] - dissipation[0],
        j_flux[:, :, :, 1:-1, :] - dissipation[1],
        k_flux[:, :, :, :, 1:-1] - dissipation[2],
    )


def compute_interior_residual(
    fields: torch.Tensor,
    geometry: StructuredBlockGeometry,
    *,
    flow_conditions: torch.Tensor | None = None,
    include_viscous: bool = True,
    include_sa: bool = True,
    dissipation_mode: str = "jameson",
    gamma: float = 1.4,
    adis: float = 2.0 / 3.0,
    acoustic_scale_factor: float = 1.0,
    vis2: float = 0.25,
    vis4: float = 0.0156,
    dss_max: float = 0.25,
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute interior flow and optional SA residuals with ADFLOW-style fluxes.

    The state convention is primitive
    `[Density, VelocityX, VelocityY, VelocityZ, Pressure, ...]`. Boundary
    cells are left at zero and excluded by the returned boolean mask.
    """
    if fields.ndim != 5:
        raise ValueError(f"Expected fields with shape (B,C,I,J,K), got {tuple(fields.shape)}")
    if fields.shape[1] < 5:
        raise ValueError("3D inviscid residual requires Density, three velocities, and Pressure")

    cell_shape = tuple(int(value) for value in fields.shape[-3:])
    geometry_shape = tuple(int(value) for value in geometry.cell_volumes.shape)
    if cell_shape != geometry_shape:
        raise ValueError(f"Field/geometry cell-shape mismatch: {cell_shape} != {geometry_shape}")
    mode = str(dissipation_mode).lower()
    if mode not in {"jameson", "none"}:
        raise ValueError(f"dissipation_mode must be 'jameson' or 'none', got {dissipation_mode!r}")
    if include_viscous and flow_conditions is None:
        raise ValueError("include_viscous=True requires flow_conditions")
    if include_sa and flow_conditions is None:
        raise ValueError("include_sa=True requires flow_conditions")
    margin = 2 if mode == "jameson" or include_viscous else 1
    minimum_size = 2 * margin + 1
    if any(size < minimum_size for size in cell_shape):
        raise ValueError(
            f"Interior 3D residual with dissipation_mode={mode!r} requires at least "
            f"{minimum_size} cells along every block axis, "
            f"got {cell_shape}"
        )

    geometry = geometry.to(fields.device, dtype=fields.dtype)
    i_flux, j_flux, k_flux = _interior_face_fluxes(
        fields,
        geometry,
        gamma=gamma,
    )
    if mode == "jameson":
        i_flux, j_flux, k_flux = _jameson_face_fluxes(
            fields,
            geometry,
            (i_flux, j_flux, k_flux),
            gamma=gamma,
            adis=adis,
            acoustic_scale_factor=acoustic_scale_factor,
            vis2=vis2,
            vis4=vis4,
            dss_max=dss_max,
            p_inf_corr=p_inf_corr,
            rho_inf=rho_inf,
        )
    elif margin == 2:
        i_flux = i_flux[:, :, 1:-1]
        j_flux = j_flux[:, :, :, 1:-1]
        k_flux = k_flux[:, :, :, :, 1:-1]

    if margin == 2:
        balance = (
            i_flux[:, :, 1:, 2:-2, 2:-2] - i_flux[:, :, :-1, 2:-2, 2:-2]
            + j_flux[:, :, 2:-2, 1:, 2:-2] - j_flux[:, :, 2:-2, :-1, 2:-2]
            + k_flux[:, :, 2:-2, 2:-2, 1:] - k_flux[:, :, 2:-2, 2:-2, :-1]
        )
    else:
        balance = (
            i_flux[:, :, 1:, 1:-1, 1:-1] - i_flux[:, :, :-1, 1:-1, 1:-1]
            + j_flux[:, :, 1:-1, 1:, 1:-1] - j_flux[:, :, 1:-1, :-1, 1:-1]
            + k_flux[:, :, 1:-1, 1:-1, 1:] - k_flux[:, :, 1:-1, 1:-1, :-1]
        )
    if include_viscous:
        viscous_flow = compute_interior_viscous_flow_balance(
            fields,
            geometry,
            flow_conditions,
            gamma=gamma,
            reference_temperature_k=reference_temperature_k,
            use_turbulent_viscosity=use_turbulent_viscosity,
            nu_tilde_index=nu_tilde_index,
            cv1=cv1,
            residual_filter=residual_filter,
            prandtl=prandtl,
            turbulent_prandtl=turbulent_prandtl,
        )
        balance = torch.cat(
            (
                balance[:, :1],
                balance[:, 1:] - viscous_flow,
            ),
            dim=1,
        )
    residual_names = FULL_RESIDUAL_NAMES if include_sa else FLOW_RESIDUAL_NAMES
    residual = torch.zeros(
        (fields.shape[0], len(residual_names), *cell_shape),
        dtype=fields.dtype,
        device=fields.device,
    )
    interior = (slice(margin, -margin),) * 3
    residual[(slice(None), slice(0, len(FLOW_RESIDUAL_NAMES)), *interior)] = (
        balance / geometry.cell_volumes[interior].unsqueeze(0).unsqueeze(0)
    )
    if include_sa:
        residual[(slice(None), 5, *interior)] = compute_interior_sa_residual(
            fields,
            geometry,
            flow_conditions,
            gamma=gamma,
            reference_temperature_k=reference_temperature_k,
            nu_tilde_index=nu_tilde_index,
            cv1=cv1,
            use_ft2=sa_use_ft2,
            approx_sa=sa_approx,
            require_adflow_wall_distance=require_adflow_wall_distance,
        )
    valid_mask = torch.zeros(cell_shape, dtype=torch.bool, device=fields.device)
    valid_mask[interior] = True
    return residual, valid_mask


__all__ = [
    "FLOW_RESIDUAL_NAMES",
    "FULL_RESIDUAL_NAMES",
    "compute_interior_residual",
]
