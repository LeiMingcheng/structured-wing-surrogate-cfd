"""ADFLOW-style interior Spalart-Allmaras residual for structured 3D blocks."""

from __future__ import annotations

import torch

from surrogate3d.physics.block_geometry import StructuredBlockGeometry
from surrogate3d.physics.pde.viscous3d import compute_cell_viscosity


def _interior_direction_data(
    values: torch.Tensor,
    geometry: StructuredBlockGeometry,
    *,
    axis: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    if axis == 0:
        center = values[:, 2:-2, 2:-2, 2:-2]
        minus = values[:, 1:-3, 2:-2, 2:-2]
        plus = values[:, 3:-1, 2:-2, 2:-2]
        area_minus = geometry.i_face_area_vectors[2:-3, 2:-2, 2:-2]
        area_plus = geometry.i_face_area_vectors[3:-2, 2:-2, 2:-2]
    elif axis == 1:
        center = values[:, 2:-2, 2:-2, 2:-2]
        minus = values[:, 2:-2, 1:-3, 2:-2]
        plus = values[:, 2:-2, 3:-1, 2:-2]
        area_minus = geometry.j_face_area_vectors[2:-2, 2:-3, 2:-2]
        area_plus = geometry.j_face_area_vectors[2:-2, 3:-2, 2:-2]
    elif axis == 2:
        center = values[:, 2:-2, 2:-2, 2:-2]
        minus = values[:, 2:-2, 2:-2, 1:-3]
        plus = values[:, 2:-2, 2:-2, 3:-1]
        area_minus = geometry.k_face_area_vectors[2:-2, 2:-2, 2:-3]
        area_plus = geometry.k_face_area_vectors[2:-2, 2:-2, 3:-2]
    else:
        raise ValueError(f"axis must be 0, 1, or 2, got {axis}")
    return center, minus, plus, area_minus, area_plus


def _cell_center_velocity_gradient(
    velocity: torch.Tensor,
    geometry: StructuredBlockGeometry,
) -> torch.Tensor:
    numerator = torch.zeros(
        (
            velocity.shape[0],
            3,
            3,
            velocity.shape[2] - 4,
            velocity.shape[3] - 4,
            velocity.shape[4] - 4,
        ),
        device=velocity.device,
        dtype=velocity.dtype,
    )
    for axis in range(3):
        _, _, _, area_minus, area_plus = _interior_direction_data(
            velocity[:, 0],
            geometry,
            axis=axis,
        )
        if axis == 0:
            value_minus = velocity[:, :, 1:-3, 2:-2, 2:-2]
            value_plus = velocity[:, :, 3:-1, 2:-2, 2:-2]
        elif axis == 1:
            value_minus = velocity[:, :, 2:-2, 1:-3, 2:-2]
            value_plus = velocity[:, :, 2:-2, 3:-1, 2:-2]
        else:
            value_minus = velocity[:, :, 2:-2, 2:-2, 1:-3]
            value_plus = velocity[:, :, 2:-2, 2:-2, 3:-1]
        area_minus = area_minus.movedim(-1, 0).unsqueeze(0).unsqueeze(0)
        area_plus = area_plus.movedim(-1, 0).unsqueeze(0).unsqueeze(0)
        numerator = numerator + (
            value_plus.unsqueeze(2) * area_plus
            - value_minus.unsqueeze(2) * area_minus
        )
    volumes = geometry.cell_volumes[2:-2, 2:-2, 2:-2]
    return numerator / (2.0 * volumes.unsqueeze(0).unsqueeze(0).unsqueeze(0))


def _sa_source(
    fields: torch.Tensor,
    geometry: StructuredBlockGeometry,
    mu_laminar: torch.Tensor,
    *,
    nu_tilde_index: int,
    cb1: float,
    cb2: float,
    kappa: float,
    cw2: float,
    cw3: float,
    ct3: float,
    ct4: float,
    cv1: float,
    use_ft2: bool,
    approx_sa: bool,
) -> torch.Tensor:
    interior = (slice(2, -2),) * 3
    velocity_gradient = _cell_center_velocity_gradient(fields[:, 1:4], geometry)
    strain = 0.5 * (
        velocity_gradient + velocity_gradient.transpose(1, 2)
    )
    divergence = (
        velocity_gradient[:, 0, 0]
        + velocity_gradient[:, 1, 1]
        + velocity_gradient[:, 2, 2]
    )
    strain_magnitude_squared = torch.sum(strain**2, dim=(1, 2))
    production_basis = torch.sqrt(
        2.0 * strain_magnitude_squared - (2.0 / 3.0) * divergence**2
    )

    rho = fields[:, 0][(slice(None), *interior)]
    nu_tilde = fields[:, nu_tilde_index][(slice(None), *interior)]
    nu_laminar = mu_laminar[(slice(None), *interior)] / rho
    wall_distance = geometry.wall_distance[interior].unsqueeze(0)
    distance_squared_inverse = 1.0 / wall_distance**2
    chi = nu_tilde / nu_laminar
    chi2 = chi**2
    chi3 = chi**3
    fv1 = chi3 / (chi3 + float(cv1) ** 3)
    fv2 = 1.0 - chi / (1.0 + chi * fv1)
    ft2 = (
        float(ct3) * torch.exp(-float(ct4) * chi2)
        if use_ft2
        else torch.zeros_like(chi)
    )

    kappa_squared_inverse = 1.0 / float(kappa) ** 2
    corrected_production = production_basis + (
        nu_tilde * fv2 * kappa_squared_inverse * distance_squared_inverse
    )
    corrected_production = torch.clamp(corrected_production, min=1.0e-10)
    ratio = (
        nu_tilde
        * kappa_squared_inverse
        * distance_squared_inverse
        / corrected_production
    ).clamp_max(10.0)
    g_function = ratio + float(cw2) * (ratio**6 - ratio)
    cw3_sixth = float(cw3) ** 6
    wall_function = g_function * (
        (1.0 + cw3_sixth) / (g_function**6 + cw3_sixth)
    ) ** (1.0 / 6.0)
    cw1 = float(cb1) / float(kappa) ** 2 + (1.0 + float(cb2)) / (2.0 / 3.0)
    production = (
        torch.zeros_like(production_basis)
        if approx_sa
        else float(cb1) * (1.0 - ft2) * production_basis
    )
    destruction = distance_squared_inverse * (
        kappa_squared_inverse
        * float(cb1)
        * ((1.0 - ft2) * fv2 + ft2)
        - cw1 * wall_function
    )
    return (production + destruction * nu_tilde) * nu_tilde


def _sa_advection(
    fields: torch.Tensor,
    geometry: StructuredBlockGeometry,
    *,
    nu_tilde_index: int,
) -> torch.Tensor:
    volumes = geometry.cell_volumes[2:-2, 2:-2, 2:-2]
    velocity = fields[:, 1:4, 2:-2, 2:-2, 2:-2]
    nu_tilde = fields[:, nu_tilde_index]
    residual = torch.zeros_like(nu_tilde[:, 2:-2, 2:-2, 2:-2])
    for axis in range(3):
        center, minus, plus, area_minus, area_plus = _interior_direction_data(
            nu_tilde,
            geometry,
            axis=axis,
        )
        average_area = (area_plus + area_minus).movedim(-1, 0).unsqueeze(0)
        contravariant_velocity = (
            0.5
            * torch.sum(velocity * average_area, dim=1)
            / volumes.unsqueeze(0)
        )
        upwind_difference = torch.where(
            contravariant_velocity > 0.0,
            center - minus,
            plus - center,
        )
        residual = residual - contravariant_velocity * upwind_difference
    return residual


def _sa_diffusion(
    fields: torch.Tensor,
    geometry: StructuredBlockGeometry,
    mu_laminar: torch.Tensor,
    *,
    nu_tilde_index: int,
    cb2: float,
) -> torch.Tensor:
    nu_tilde = fields[:, nu_tilde_index]
    nu_laminar = mu_laminar / fields[:, 0]
    volumes = geometry.cell_volumes
    center_volume = volumes[2:-2, 2:-2, 2:-2]
    residual = torch.zeros_like(nu_tilde[:, 2:-2, 2:-2, 2:-2])
    cb3_inverse = 1.0 / (2.0 / 3.0)

    for axis in range(3):
        center, minus, plus, area_minus, area_plus = _interior_direction_data(
            nu_tilde,
            geometry,
            axis=axis,
        )
        _, nu_minus, nu_plus, _, _ = _interior_direction_data(
            nu_laminar,
            geometry,
            axis=axis,
        )
        if axis == 0:
            volume_minus = volumes[1:-3, 2:-2, 2:-2]
            volume_plus = volumes[3:-1, 2:-2, 2:-2]
        elif axis == 1:
            volume_minus = volumes[2:-2, 1:-3, 2:-2]
            volume_plus = volumes[2:-2, 3:-1, 2:-2]
        else:
            volume_minus = volumes[2:-2, 2:-2, 1:-3]
            volume_plus = volumes[2:-2, 2:-2, 3:-1]

        minus_metric = (
            area_minus
            * (2.0 / (center_volume + volume_minus))[..., None]
        )
        plus_metric = (
            area_plus
            * (2.0 / (center_volume + volume_plus))[..., None]
        )
        average_metric = (
            0.5
            * (area_plus + area_minus)
            / center_volume[..., None]
        )
        ttm = torch.sum(minus_metric * average_metric, dim=-1).unsqueeze(0)
        ttp = torch.sum(plus_metric * average_metric, dim=-1).unsqueeze(0)

        central_adjustment = -float(cb2) * center * cb3_inverse
        coefficient_minus = (
            0.5 * (nu_minus + nu_laminar[:, 2:-2, 2:-2, 2:-2])
            + (1.0 + float(cb2)) * 0.5 * (minus + center)
        ) * ttm * cb3_inverse
        coefficient_plus = (
            0.5 * (nu_plus + nu_laminar[:, 2:-2, 2:-2, 2:-2])
            + (1.0 + float(cb2)) * 0.5 * (plus + center)
        ) * ttp * cb3_inverse
        coefficient_minus = torch.clamp(
            coefficient_minus + ttm * central_adjustment,
            min=0.0,
        )
        coefficient_plus = torch.clamp(
            coefficient_plus + ttp * central_adjustment,
            min=0.0,
        )
        residual = residual + (
            coefficient_minus * minus
            - (coefficient_minus + coefficient_plus) * center
            + coefficient_plus * plus
        )
    return residual


def compute_interior_sa_residual(
    fields: torch.Tensor,
    geometry: StructuredBlockGeometry,
    flow_conditions: torch.Tensor,
    *,
    gamma: float = 1.4,
    reference_temperature_k: float = 300.0,
    nu_tilde_index: int = 5,
    cv1: float = 7.1,
    cb1: float = 0.1355,
    cb2: float = 0.622,
    kappa: float = 0.41,
    cw2: float = 0.3,
    cw3: float = 2.0,
    ct3: float = 1.2,
    ct4: float = 0.5,
    use_ft2: bool = True,
    approx_sa: bool = False,
    require_adflow_wall_distance: bool = True,
) -> torch.Tensor:
    """Compute ADFLOW's per-volume interior SA residual on the two-cell mask."""
    if fields.shape[1] <= nu_tilde_index:
        raise ValueError(
            f"SA residual requires field channel {nu_tilde_index}, "
            f"got {fields.shape[1]} channels"
        )
    if (
        require_adflow_wall_distance
        and geometry.wall_distance_source != "adflow_turbulent_distance"
    ):
        raise ValueError(
            "SA residual requires ADFLOW TurbulentDistance geometry; "
            f"got {geometry.wall_distance_source!r}"
        )
    geometry = geometry.to(fields.device, dtype=fields.dtype)
    mu_laminar, _ = compute_cell_viscosity(
        fields,
        flow_conditions,
        gamma=gamma,
        reference_temperature_k=reference_temperature_k,
        use_turbulent_viscosity=False,
        nu_tilde_index=nu_tilde_index,
        cv1=cv1,
    )
    source = _sa_source(
        fields,
        geometry,
        mu_laminar,
        nu_tilde_index=nu_tilde_index,
        cb1=cb1,
        cb2=cb2,
        kappa=kappa,
        cw2=cw2,
        cw3=cw3,
        ct3=ct3,
        ct4=ct4,
        cv1=cv1,
        use_ft2=use_ft2,
        approx_sa=approx_sa,
    )
    advection = _sa_advection(
        fields,
        geometry,
        nu_tilde_index=nu_tilde_index,
    )
    diffusion = _sa_diffusion(
        fields,
        geometry,
        mu_laminar,
        nu_tilde_index=nu_tilde_index,
        cb2=cb2,
    )
    return -(source + advection + diffusion)


__all__ = ["compute_interior_sa_residual"]
