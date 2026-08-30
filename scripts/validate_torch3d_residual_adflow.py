"""Compare the Torch 3D residual with residual fields exported by ADFLOW."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from surrogate3d.physics.block_geometry import (
    _iter_volume_zones,
    extract_case_block_geometry_from_volume_cgns,
)
from surrogate3d.physics.pde.residual3d import compute_interior_residual
from surrogate3d.physics.pde.viscous3d import compute_cell_viscosity
from parametric_swept_wing_cfd.export.cgns_state import (
    _find_solution_group,
    _node_data,
    _rind,
    _to_ijk,
    _trim_rind,
)


STATE_FIELDS = (
    "Density",
    "VelocityX",
    "VelocityY",
    "VelocityZ",
    "Pressure",
    "TurbulentSANuTilde",
)
RESIDUAL_FIELDS = (
    "ResDensity",
    "ResMomentumX",
    "ResMomentumY",
    "ResMomentumZ",
    "ResEnergyStagnationDensity",
    "ResTurbulentSANuTilde",
)
COMPONENT_NAMES = ("Rc", "Rmx", "Rmy", "Rmz", "RE", "RSA")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adflow-cgns", type=Path, required=True)
    parser.add_argument("--mach", type=float, required=True)
    parser.add_argument("--aoa", type=float, required=True)
    parser.add_argument("--reynolds", type=float, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser


def _read_cell_field(solution, name: str, rind: np.ndarray) -> np.ndarray:
    raw = np.asarray(_node_data(solution[name]))
    return _to_ijk(_trim_rind(raw, rind, ndim=3)).astype(np.float64, copy=False)


def _comparison_metrics(torch_values: torch.Tensor, adflow_values: torch.Tensor) -> dict[str, float]:
    torch_rms = torch.sqrt(torch.mean(torch_values**2))
    adflow_rms = torch.sqrt(torch.mean(adflow_values**2))
    scale = torch.sum(torch_values * adflow_values) / torch.sum(torch_values**2)
    return {
        "torch_rms": float(torch_rms.item()),
        "adflow_rms": float(adflow_rms.item()),
        "correlation": float(
            torch.corrcoef(torch.stack((torch_values, adflow_values)))[0, 1].item()
        ),
        "adflow_over_torch_scale": float(scale.item()),
        "relative_rmse": float(
            (torch.sqrt(torch.mean((torch_values - adflow_values) ** 2)) / adflow_rms).item()
        ),
        "scaled_relative_rmse": float(
            (torch.sqrt(torch.mean((scale * torch_values - adflow_values) ** 2)) / adflow_rms).item()
        ),
    }


def validate(args: argparse.Namespace) -> dict:
    cgns_path = args.adflow_cgns.expanduser().resolve()
    device = torch.device(args.device)
    flow_conditions = torch.tensor(
        [[args.mach, args.aoa, args.reynolds]],
        device=device,
        dtype=torch.float64,
    )
    torch_components: dict[str, list[torch.Tensor]] = {
        name: [] for name in COMPONENT_NAMES
    }
    adflow_components: dict[str, list[torch.Tensor]] = {
        name: [] for name in COMPONENT_NAMES
    }
    torch_eddy_viscosity: list[torch.Tensor] = []
    adflow_eddy_viscosity: list[torch.Tensor] = []
    blocks: list[dict[str, object]] = []
    geometry_by_block = extract_case_block_geometry_from_volume_cgns(
        cgns_path,
        dtype=torch.float64,
    )

    with torch.no_grad(), h5py.File(cgns_path, "r") as h5file:
        for block_index, (zone_name, zone) in enumerate(_iter_volume_zones(h5file)):
            solution = _find_solution_group(zone)
            rind = _rind(solution, ndim=3)
            state = torch.from_numpy(
                np.stack(
                    [_read_cell_field(solution, name, rind) for name in STATE_FIELDS],
                    axis=0,
                )[None]
            ).to(device=device)
            adflow_residual = torch.from_numpy(
                np.stack(
                    [_read_cell_field(solution, name, rind) for name in RESIDUAL_FIELDS],
                    axis=0,
                )[None]
            ).to(device=device)
            geometry = geometry_by_block[block_index]
            torch_residual, valid_mask = compute_interior_residual(
                state,
                geometry,
                flow_conditions=flow_conditions,
                include_sa=True,
            )
            _, mu_eddy = compute_cell_viscosity(state, flow_conditions)
            adflow_mu_eddy = torch.from_numpy(
                _read_cell_field(solution, "ViscosityEddy", rind)
            ).to(device=device)

            for component_index, name in enumerate(COMPONENT_NAMES):
                torch_components[name].append(
                    torch_residual[0, component_index][valid_mask].cpu()
                )
                adflow_components[name].append(
                    adflow_residual[0, component_index][valid_mask].cpu()
                )
            torch_eddy_viscosity.append(mu_eddy[0][valid_mask].cpu())
            adflow_eddy_viscosity.append(adflow_mu_eddy[valid_mask].cpu())
            blocks.append(
                {
                    "block_index": int(block_index),
                    "zone_name": str(zone_name),
                    "cell_shape": [int(value) for value in state.shape[-3:]],
                    "valid_cells": int(valid_mask.sum().item()),
                }
            )

    component_metrics = {
        name: _comparison_metrics(
            torch.cat(torch_components[name]),
            torch.cat(adflow_components[name]),
        )
        for name in COMPONENT_NAMES
    }
    viscosity_metrics = _comparison_metrics(
        torch.cat(torch_eddy_viscosity),
        torch.cat(adflow_eddy_viscosity),
    )
    return {
        "source_cgns": str(cgns_path),
        "formulation": "interior_jameson_continuity_momentum_energy_viscous_sa",
        "flow_conditions": {
            "mach": float(args.mach),
            "aoa": float(args.aoa),
            "reynolds": float(args.reynolds),
        },
        "valid_cells": int(sum(block["valid_cells"] for block in blocks)),
        "components": component_metrics,
        "eddy_viscosity": viscosity_metrics,
        "blocks": blocks,
        "scope": {
            "included": [
                "continuity",
                "three_momentum_equations",
                "energy_equation",
                "sa_transport_equation",
                "scalar_jameson_dissipation",
                "laminar_viscosity",
                "sa_eddy_viscosity",
            ],
            "not_included": [
                "boundary_and_cross_block_halo_cells",
            ],
        },
    }


def main() -> None:
    args = build_parser().parse_args()
    payload = validate(args)
    output_path = args.output_json.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(payload["components"], indent=2, sort_keys=True))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
