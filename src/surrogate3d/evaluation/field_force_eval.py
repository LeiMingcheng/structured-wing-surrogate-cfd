"""Field-based 3D force evaluation on surrogate-predicted volume states."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from surrogate3d.evaluation.force_eval import (
    compute_force_error_metrics,
    read_reference_force_coefficients,
)
from surrogate3d.evaluation.inference import WholeCasePrediction
from surrogate3d.physics.surface_forces import compute_surface_field_metrics
from surrogate3d.physics.field_force import (
    ForceEvaluatorConfig,
    compute_force_coefficients_torch,
    force_evaluator_config_from_dict,
    prepare_volume_force_geometry,
)
from surrogate3d.physics.wall_geometry import wall_geometry_from_prediction_batch


@dataclass(frozen=True)
class ForceEvalConfig(ForceEvaluatorConfig):
    """Configuration for whole-case predictor-side volume force evaluation."""

    device: str = "cpu"
    dtype: str = "float64"

    def torch_dtype(self) -> torch.dtype:
        return getattr(torch, str(self.dtype))

    def candidate_key(self) -> str:
        return "force_torch"


def force_eval_config_from_dict(
    config: Mapping[str, Any] | None = None,
) -> ForceEvalConfig:
    payload = dict(config or {})
    if "velocity_indices" in payload:
        payload["velocity_indices"] = tuple(int(value) for value in payload["velocity_indices"])
    return ForceEvalConfig(**payload)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    return value


def _require_single_case_prediction(prediction: WholeCasePrediction) -> None:
    sample_ids = prediction.batch.get("sample_id")
    if not isinstance(sample_ids, list) or len(sample_ids) != 1:
        raise ValueError("run_force_evaluation currently expects a single-case WholeCasePrediction")


def _prediction_source_case_dir(prediction: WholeCasePrediction) -> Path:
    values = prediction.batch.get("source_case_dir")
    if not isinstance(values, list) or not values or not str(values[0]).strip():
        raise ValueError("Prediction batch does not include a valid source_case_dir[0]")
    return Path(str(values[0])).expanduser().resolve()


def _prediction_volume_cgns(prediction: WholeCasePrediction, source_case_dir: Path) -> Path:
    basenames = prediction.batch.get("cgns_basename")
    basename = str(basenames[0]).strip() if isinstance(basenames, list) and basenames else ""
    if basename:
        return (source_case_dir / basename).expanduser().resolve()
    return (source_case_dir / "wing_000_vol.cgns").expanduser().resolve()


def read_force_case_metadata(
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


def _coefficient_record(coeffs: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "CL",
        "CD",
        "CMz",
        "Cx",
        "Cy",
        "Cz",
        "CMx",
        "CMy",
        "pressure_force_coeff",
        "viscous_force_coeff",
    )
    record: dict[str, Any] = {}
    for key in keys:
        value = coeffs[key]
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                record[key] = float(value.detach().cpu().item())
            else:
                record[key] = value.detach().cpu().tolist()
        else:
            record[key] = value
    return record


def run_force_evaluation(
    prediction: WholeCasePrediction,
    *,
    output_dir: str | Path,
    config: ForceEvalConfig | Mapping[str, Any] | None = None,
    source_case_dir: str | Path | None = None,
    volume_cgns: str | Path | None = None,
    summary_path: str | Path | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate predicted/target volume fields via wall Cp/Cf recovery."""
    _require_single_case_prediction(prediction)
    cfg = config if isinstance(config, ForceEvalConfig) else force_eval_config_from_dict(config)
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = (
        Path(summary_path).expanduser().resolve()
        if summary_path is not None
        else output_dir / "force_summary.json"
    )

    resolved_source_case_dir = (
        Path(source_case_dir).expanduser().resolve()
        if source_case_dir is not None
        else _prediction_source_case_dir(prediction)
    )
    resolved_volume_cgns = (
        Path(volume_cgns).expanduser().resolve()
        if volume_cgns is not None
        else _prediction_volume_cgns(prediction, resolved_source_case_dir)
    )
    case_metadata = read_force_case_metadata(
        input_json=resolved_source_case_dir.parent / "input.json",
        conds_json=resolved_source_case_dir / "conds.json",
    )
    reference = read_reference_force_coefficients(resolved_source_case_dir / "output.json")

    torch_dtype = cfg.torch_dtype()
    block_geometry = wall_geometry_from_prediction_batch(
        resolved_volume_cgns,
        prediction.batch,
        dtype=torch_dtype,
    )
    prepared_geometry = prepare_volume_force_geometry(
        block_geometry,
        device=cfg.device,
        dtype=torch_dtype,
    )
    moment_center = (
        float(case_metadata["x_ref"]),
        float(case_metadata["y_ref"]),
        float(case_metadata["z_ref"]),
    )
    pred_coeffs = compute_force_coefficients_torch(
        prediction.pred_physical,
        block_geometry,
        prediction.batch["flow_conditions"],
        area_ref=float(case_metadata["area_ref"]),
        chord_ref=float(case_metadata["chord_ref"]),
        moment_center=moment_center,
        config=cfg,
        prepared_geometry=prepared_geometry,
        device=cfg.device,
        dtype=torch_dtype,
        return_surface_fields=True,
    )
    target_coeffs = compute_force_coefficients_torch(
        prediction.target_physical,
        block_geometry,
        prediction.batch["flow_conditions"],
        area_ref=float(case_metadata["area_ref"]),
        chord_ref=float(case_metadata["chord_ref"]),
        moment_center=moment_center,
        config=cfg,
        prepared_geometry=prepared_geometry,
        device=cfg.device,
        dtype=torch_dtype,
        return_surface_fields=True,
    )

    pred_record = _coefficient_record(pred_coeffs)
    target_record = _coefficient_record(target_coeffs)
    pred_reference_metrics = compute_force_error_metrics(reference, pred_record)
    target_reference_metrics = compute_force_error_metrics(reference, target_record)
    pred_target_metrics = compute_force_error_metrics(target_record, pred_record)
    wall_field_metrics = compute_surface_field_metrics(
        pred_coeffs["surface_fields"],
        target_coeffs["surface_fields"],
        cp_index=0,
        cf_indices=(1, 2, 3),
    )

    summary: dict[str, Any] = {
        "force_eval_method": "force_torch",
        "force_eval_candidate_key": cfg.candidate_key(),
        "reference": reference,
        cfg.candidate_key(): pred_record,
        "target_force": target_record,
        "pred_vs_reference": pred_reference_metrics,
        "target_vs_reference": target_reference_metrics,
        "pred_vs_target": pred_target_metrics,
        "wall_field_metrics": wall_field_metrics,
        "volume_cgns": str(resolved_volume_cgns),
        "source_case_dir": str(resolved_source_case_dir),
        "case_metadata": dict(case_metadata),
        "config": asdict(cfg),
        "metadata": dict(metadata or {}),
        "force_mode": cfg.mode,
        "is_complete_force": cfg.is_complete_force(),
        "viscous_model": cfg.viscous_model(),
        "block_surface_cell_counts": prepared_geometry["block_cell_counts"],
        "n_surface_cells": int(prepared_geometry["n_surface_cells"]),
        "surface_field_names": ["Cp", "CfX", "CfY", "CfZ"],
    }
    summary.update(pred_reference_metrics)
    if cfg.includes_viscous_force():
        summary["note"] = (
            "Viscous term uses a first-cell wall-shear proxy from the predicted volume field, "
            "not ADFLOW's wall-flux reconstruction."
        )
    else:
        summary["note"] = (
            "Pressure-only mode excludes viscous force and is not a complete total-force result."
        )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True, default=_json_default)
        stream.write("\n")
    summary["summary_path"] = str(summary_path)
    return summary


__all__ = [
    "ForceEvalConfig",
    "force_eval_config_from_dict",
    "read_force_case_metadata",
    "run_force_evaluation",
]
