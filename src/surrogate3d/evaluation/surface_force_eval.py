"""Field-based 3D surface force evaluation using Cp/Cf wall data."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from parametric_swept_wing_cfd.export import export_surface_state

from surrogate3d.evaluation.force_eval import (
    compute_force_error_metrics,
    read_reference_force_coefficients,
)
from surrogate3d.physics.surface_forces import (
    compute_surface_force_coefficients_torch,
    prepare_surface_force_geometry,
)


@dataclass(frozen=True)
class SurfaceForceConfig:
    """Field-name and coefficient settings for 3D wall-surface force evaluation."""

    cp_field: str = "CoefPressure"
    cf_fields: tuple[str, str, str] = ("SkinFrictionX", "SkinFrictionY", "SkinFrictionZ")
    compute_viscous: bool = True
    device: str = "cpu"
    dtype: str = "float64"


def surface_force_config_from_dict(config: Mapping[str, Any] | None = None) -> SurfaceForceConfig:
    payload = dict(config or {})
    if "cf_fields" in payload:
        payload["cf_fields"] = tuple(str(value) for value in payload["cf_fields"])
    return SurfaceForceConfig(**payload)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    return value


def read_surface_force_case_metadata(
    *,
    input_json: str | Path,
    conds_json: str | Path,
) -> dict[str, float]:
    input_json = Path(input_json).expanduser().resolve()
    conds_json = Path(conds_json).expanduser().resolve()
    with input_json.open("r", encoding="utf-8") as stream:
        wing = json.load(stream)
    with conds_json.open("r", encoding="utf-8") as stream:
        conds = json.load(stream)
    return {
        "area_ref": float(wing["surface_area"]),
        "chord_ref": float(wing.get("chord_ref", wing.get("ref_chord", 1.0))),
        "x_ref": float(wing.get("x_ref", wing.get("xRef", 0.25))),
        "y_ref": float(wing.get("y_ref", 0.0) or 0.0),
        "z_ref": float(wing.get("z_ref", 0.0) or 0.0),
        "aoa_deg": float(conds.get("AA", wing.get("AA", 0.0))),
    }


def _surface_field_index_map(field_names: Sequence[str]) -> dict[str, int]:
    return {str(name): idx for idx, name in enumerate(field_names)}


def compute_surface_force_coefficients_from_npz(
    surface_state_npz: str | Path,
    *,
    case_metadata: Mapping[str, float],
    config: SurfaceForceConfig | Mapping[str, Any] | None = None,
) -> dict[str, float]:
    cfg = config if isinstance(config, SurfaceForceConfig) else surface_force_config_from_dict(config)
    surface_state_npz = Path(surface_state_npz).expanduser().resolve()
    with np.load(surface_state_npz, allow_pickle=False) as data:
        field_names = [str(value) for value in data["field_names"].tolist()]
        field_indices = _surface_field_index_map(field_names)
        cp_index = int(field_indices[cfg.cp_field])
        cf_indices = tuple(int(field_indices[name]) for name in cfg.cf_fields)

        geometry = prepare_surface_force_geometry(
            centers=data["coords"],
            area_vectors=data["area_vector"],
            cell_areas=data["cell_area"],
            device=cfg.device,
            dtype=getattr(torch, str(cfg.dtype)),
        )
        surface_fields = torch.as_tensor(
            np.asarray(data["fields"], dtype=np.float32),
            device=torch.device(cfg.device),
            dtype=getattr(torch, str(cfg.dtype)),
        )

    coeffs = compute_surface_force_coefficients_torch(
        surface_fields,
        geometry,
        aoa_deg=float(case_metadata["aoa_deg"]),
        area_ref=float(case_metadata["area_ref"]),
        chord_ref=float(case_metadata["chord_ref"]),
        moment_center=(
            float(case_metadata["x_ref"]),
            float(case_metadata["y_ref"]),
            float(case_metadata["z_ref"]),
        ),
        cp_index=cp_index,
        cf_indices=cf_indices,
        compute_viscous=bool(cfg.compute_viscous),
    )
    return {
        "CL": float(coeffs["CL"].detach().cpu().item()),
        "CD": float(coeffs["CD"].detach().cpu().item()),
        "CMz": float(coeffs["CMz"].detach().cpu().item()),
        "Cx": float(coeffs["Cx"].detach().cpu().item()),
        "Cy": float(coeffs["Cy"].detach().cpu().item()),
        "Cz": float(coeffs["Cz"].detach().cpu().item()),
        "CMx": float(coeffs["CMx"].detach().cpu().item()),
        "CMy": float(coeffs["CMy"].detach().cpu().item()),
        "pressure_force_coeff": [float(v) for v in coeffs["pressure_force_coeff"].detach().cpu().tolist()],
        "viscous_force_coeff": [float(v) for v in coeffs["viscous_force_coeff"].detach().cpu().tolist()],
    }


def run_surface_force_evaluation(
    *,
    source_case_dir: str | Path,
    output_dir: str | Path,
    config: SurfaceForceConfig | Mapping[str, Any] | None = None,
    surface_cgns: str | Path | None = None,
    summary_path: str | Path | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = config if isinstance(config, SurfaceForceConfig) else surface_force_config_from_dict(config)
    source_case_dir = Path(source_case_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = (
        Path(summary_path).expanduser().resolve()
        if summary_path is not None
        else output_dir / "surface_force_summary.json"
    )
    surface_cgns = (
        Path(surface_cgns).expanduser().resolve()
        if surface_cgns is not None
        else source_case_dir / "wing_000_surf.cgns"
    )
    surface_state_npz = output_dir / "surface_state.npz"
    surface_meta_json = output_dir / "surface_state_meta.json"
    export_surface_state(
        surface_cgns,
        surface_state_npz,
        metadata_json=surface_meta_json,
    )

    case_metadata = read_surface_force_case_metadata(
        input_json=source_case_dir.parent / "input.json",
        conds_json=source_case_dir / "conds.json",
    )
    reference = read_reference_force_coefficients(source_case_dir / "output.json")
    candidate = compute_surface_force_coefficients_from_npz(
        surface_state_npz,
        case_metadata=case_metadata,
        config=cfg,
    )
    metrics = compute_force_error_metrics(reference, candidate)
    with np.load(surface_state_npz, allow_pickle=False) as data:
        field_names = [str(value) for value in data["field_names"].tolist()]

    summary: dict[str, Any] = {
        "force_eval_method": "surface_cp_cf_torch",
        "force_eval_candidate_key": "surface_cp_cf_torch",
        "reference": reference,
        "surface_cp_cf_torch": candidate,
        "surface_state_npz": str(surface_state_npz),
        "surface_state_meta_json": str(surface_meta_json),
        "source_case_dir": str(source_case_dir),
        "surface_cgns": str(surface_cgns),
        "case_metadata": dict(case_metadata),
        "field_names": field_names,
        "config": asdict(cfg),
        "metadata": dict(metadata or {}),
    }
    summary.update(metrics)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True, default=_json_default)
        stream.write("\n")
    summary["summary_path"] = str(summary_path)
    return summary


__all__ = [
    "SurfaceForceConfig",
    "compute_force_error_metrics",
    "compute_surface_force_coefficients_from_npz",
    "read_reference_force_coefficients",
    "read_surface_force_case_metadata",
    "run_surface_force_evaluation",
    "surface_force_config_from_dict",
]
