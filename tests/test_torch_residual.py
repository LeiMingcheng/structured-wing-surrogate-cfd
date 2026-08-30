import numpy as np
import pytest
import torch

from surrogate3d.physics.block_geometry import extract_block_geometry
from surrogate3d.physics.pde.residual3d import compute_interior_residual
from surrogate3d.physics.pde.viscous3d import (
    compute_interior_nodal_gradients,
    compute_interior_viscous_flow_balance,
)
from surrogate3d.physics.residual import PDEResidualCalculator
from surrogate3d.physics.residual_sample import ResidualBlockSample, ResidualCaseSample


def _orthogonal_geometry(n_cells: int = 5):
    i, j, k = np.meshgrid(
        np.arange(n_cells + 1, dtype=np.float64),
        np.arange(n_cells + 1, dtype=np.float64),
        np.arange(n_cells + 1, dtype=np.float64),
        indexing="ij",
    )
    nodes = np.stack((i, j, k), axis=-1)
    return extract_block_geometry(nodes, block_index=0, zone_name="block0")


def _uniform_fields(n_cells: int = 5) -> torch.Tensor:
    fields = torch.zeros((1, 6, n_cells, n_cells, n_cells), dtype=torch.float64)
    fields[:, 0] = 1.2
    fields[:, 1] = 0.8
    fields[:, 2] = -0.1
    fields[:, 3] = 0.2
    fields[:, 4] = 0.9
    fields[:, 5] = 1.0e-3
    return fields


def _case_sample(fields: torch.Tensor, geometry) -> ResidualCaseSample:
    block = ResidualBlockSample(
        block_index=0,
        block_name="block0",
        fields=fields,
        coords_center=geometry.cell_centers.permute(3, 0, 1, 2).unsqueeze(0),
        coords_vertex=geometry.vertex_coords.permute(3, 0, 1, 2).unsqueeze(0),
        cell_volumes=geometry.cell_volumes.unsqueeze(0),
        wall_distance=geometry.wall_distance.unsqueeze(0),
        geometry=geometry,
    )
    return ResidualCaseSample(
        flow_conditions=torch.tensor([[0.7, 2.0, 1.0]], dtype=fields.dtype),
        blocks={0: block},
        metadata={"sample_id": "synthetic"},
    )


def test_uniform_primitive_state_has_zero_interior_residual():
    geometry = _orthogonal_geometry()
    residual, valid_mask = compute_interior_residual(
        _uniform_fields(),
        geometry,
        include_viscous=False,
        include_sa=False,
        dissipation_mode="none",
    )

    assert residual.shape == (1, 5, 5, 5, 5)
    assert int(valid_mask.sum()) == 27
    torch.testing.assert_close(
        residual[:, :, valid_mask],
        torch.zeros((1, 5, 27), dtype=torch.float64),
    )


def test_linear_x_velocity_matches_continuity_divergence():
    geometry = _orthogonal_geometry()
    fields = _uniform_fields()
    x = geometry.cell_centers[..., 0]
    fields[:, 0] = 2.0
    fields[:, 1] = x
    fields[:, 2:4] = 0.0
    fields[:, 4] = 1.0

    residual, valid_mask = compute_interior_residual(
        fields,
        geometry,
        include_viscous=False,
        include_sa=False,
        dissipation_mode="none",
    )

    torch.testing.assert_close(
        residual[0, 0, valid_mask],
        torch.full((27,), 2.0, dtype=torch.float64),
    )


def test_calculator_aggregates_components_and_keeps_gradients():
    geometry = _orthogonal_geometry()
    fields = _uniform_fields().requires_grad_(True)
    x = geometry.cell_centers[..., 0]
    fields_with_gradient = fields.clone()
    fields_with_gradient[:, 1] = x
    sample = _case_sample(fields_with_gradient, geometry)

    score, result = PDEResidualCalculator(
        backend_config={"require_adflow_wall_distance": False}
    )(sample)
    (-score).backward()

    assert result.formulation == "interior_jameson_continuity_momentum_energy_viscous_sa"
    assert set(result.component_rms) == {"Rc", "Rmx", "Rmy", "Rmz", "RE", "RSA"}
    assert result.blocks[0].residual.shape == (1, 6, 5, 5, 5)
    assert fields.grad is not None
    assert torch.isfinite(fields.grad).all()
    assert result.component_rms["Rc"].detach().item() > 0.0


def test_default_sa_mode_requires_adflow_wall_distance():
    geometry = _orthogonal_geometry()
    sample = _case_sample(_uniform_fields(), geometry)

    with pytest.raises(ValueError, match="ADFLOW TurbulentDistance"):
        PDEResidualCalculator()(sample)


def test_backend_rejects_blocks_without_interior_cells():
    geometry = _orthogonal_geometry(n_cells=2)
    with pytest.raises(ValueError, match="at least 5 cells"):
        compute_interior_residual(
            _uniform_fields(n_cells=2),
            geometry,
            flow_conditions=torch.tensor([[0.7, 2.0, 1.0]], dtype=torch.float64),
        )


def test_adflow_nodal_gradient_recovers_physical_coordinate_basis():
    geometry = _orthogonal_geometry(n_cells=6)
    coordinates = geometry.cell_centers.permute(3, 0, 1, 2).unsqueeze(0)

    gradients = compute_interior_nodal_gradients(coordinates, geometry)
    expected = torch.eye(3, dtype=torch.float64).view(1, 3, 3, 1, 1, 1)

    torch.testing.assert_close(
        gradients,
        expected.expand_as(gradients),
        atol=1.0e-12,
        rtol=1.0e-12,
    )


def test_adflow_nodal_gradient_is_exact_on_affine_mesh():
    n_cells = 6
    i, j, k = np.meshgrid(
        np.arange(n_cells + 1, dtype=np.float64),
        np.arange(n_cells + 1, dtype=np.float64),
        np.arange(n_cells + 1, dtype=np.float64),
        indexing="ij",
    )
    logical = np.stack((i, j, k), axis=-1)
    transform = np.array(
        [
            [1.2, 0.1, -0.2],
            [0.2, 0.9, 0.15],
            [-0.1, 0.25, 1.1],
        ],
        dtype=np.float64,
    )
    nodes = logical @ transform.T
    geometry = extract_block_geometry(nodes, block_index=0, zone_name="affine")
    coordinates = geometry.cell_centers.permute(3, 0, 1, 2).unsqueeze(0)

    gradients = compute_interior_nodal_gradients(coordinates, geometry)
    expected = torch.eye(3, dtype=torch.float64).view(1, 3, 3, 1, 1, 1)

    torch.testing.assert_close(
        gradients,
        expected.expand_as(gradients),
        atol=2.0e-12,
        rtol=2.0e-12,
    )


def test_quadratic_velocity_has_expected_viscous_momentum_divergence():
    geometry = _orthogonal_geometry(n_cells=7)
    fields = _uniform_fields(n_cells=7)
    x = geometry.cell_centers[..., 0]
    fields[:, 0] = 1.0
    fields[:, 1] = x**2
    fields[:, 2:4] = 0.0
    fields[:, 4] = 1.0
    fields[:, 5] = 0.0
    flow = torch.tensor(
        [[1.0 / np.sqrt(1.4), 0.0, 1.0]],
        dtype=torch.float64,
    )

    balance = compute_interior_viscous_flow_balance(
        fields,
        geometry,
        flow,
        use_turbulent_viscosity=False,
    )

    torch.testing.assert_close(
        balance[:, 0],
        torch.full_like(balance[:, 0], 8.0 / 3.0),
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    torch.testing.assert_close(
        balance[:, 1:3],
        torch.zeros_like(balance[:, 1:3]),
        atol=1.0e-12,
        rtol=1.0e-12,
    )


def test_linear_pressure_matches_inviscid_energy_divergence():
    geometry = _orthogonal_geometry()
    fields = _uniform_fields()
    x = geometry.cell_centers[..., 0]
    fields[:, 0] = 1.0
    fields[:, 1] = 1.0
    fields[:, 2:4] = 0.0
    fields[:, 4] = x + 1.0

    residual, valid_mask = compute_interior_residual(
        fields,
        geometry,
        include_viscous=False,
        include_sa=False,
        dissipation_mode="none",
    )

    torch.testing.assert_close(
        residual[0, 4, valid_mask],
        torch.full((27,), 3.5, dtype=torch.float64),
    )
