"""Stage-1 exported-residual evaluation for surrogate3d restart predictions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from parametric_swept_wing_cfd.core.execution import run_subprocess
from parametric_swept_wing_cfd.export import export_volume_state

from surrogate3d.evaluation.adflow_restart import (
    ADflowRestartEvalConfig,
    build_adflow_restart_eval_command,
    build_adflow_restart_eval_env,
    restart_eval_config_from_dict,
)


DEFAULT_RESIDUAL_FIELDS = ("ResDensity", "ResTurbulentSANuTilde")


@dataclass(frozen=True)
class ResidualChannelMetricsConfig:
    """Configuration for stage-1 exported residual-channel comparisons."""

    field_names: tuple[str, ...] = DEFAULT_RESIDUAL_FIELDS
    near_wall_layers: int = 4


def residual_metrics_config_from_dict(
    config: Mapping[str, Any] | None = None,
) -> ResidualChannelMetricsConfig:
    payload = dict(config or {})
    if "field_names" in payload:
        payload["field_names"] = tuple(str(value) for value in payload["field_names"])
    if "near_wall_layers" in payload:
        payload["near_wall_layers"] = int(payload["near_wall_layers"])
    return ResidualChannelMetricsConfig(**payload)


def _tail_text(path: Path, *, n_lines: int = 40) -> str:
    if not path.exists():
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return ""
    return "\n".join(lines[-n_lines:])


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return value


def _resume_proxy_note(config: ADflowRestartEvalConfig) -> Optional[str]:
    if config.cycle_protocol == "resume" and config.cycles == 1 and config.inner_cycles == 1:
        return (
            "Stage-1 residual check uses a one-step restart evaluation proxy: "
            "resume, cycles=1, inner_cycles=1."
        )
    return None


def _residual_eval_method_name(config: ADflowRestartEvalConfig) -> str:
    preset = (config.solver_preset or "").strip().lower()
    if config.cycle_protocol == "resume" and config.cycles == 1 and config.inner_cycles == 1:
        base = "one_step_exported_residual_proxy"
    elif config.cycle_protocol == "resume":
        base = f"resume{int(config.cycles)}_inner{int(config.inner_cycles)}_exported_residual_proxy"
    else:
        base = f"singlecall_ncycles{int(config.cycles)}_exported_residual_proxy"
    return f"{preset}_{base}" if preset else base


def _residual_candidate_key(config: ADflowRestartEvalConfig) -> str:
    preset = (config.solver_preset or "").strip().lower()
    if config.cycle_protocol == "resume" and config.cycles == 1 and config.inner_cycles == 1:
        return f"surrogate_one_step_exported_residual_{preset}" if preset else "surrogate_one_step_exported_residual"
    return "surrogate_exported_residual_eval"


def _load_exported_residual_fields(
    npz_path: str | Path,
    *,
    field_names: Sequence[str],
) -> dict[str, Any]:
    npz_path = Path(npz_path).expanduser().resolve()
    with np.load(npz_path, allow_pickle=False) as data:
        exported_field_names = [str(value) for value in np.asarray(data["extra_fields"]).astype(str).tolist()]
        missing = [name for name in field_names if name not in exported_field_names]
        if missing:
            raise KeyError(f"{npz_path} is missing residual field(s): {missing}")
        field_indices = [exported_field_names.index(name) for name in field_names]
        return {
            "field_names": tuple(str(name) for name in field_names),
            "extra": np.asarray(data["extra"], dtype=np.float64)[:, field_indices],
            "cell_volumes": np.asarray(data["cell_volumes"], dtype=np.float64),
            "block_id": np.asarray(data["block_id"], dtype=np.int32),
            "ijk": np.asarray(data["ijk"], dtype=np.int32),
            "block_names": np.asarray(data["block_names"]).astype(str),
            "block_cell_shape": np.asarray(data["block_cell_shape"], dtype=np.int32),
        }


def _validate_matching_export_layout(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> None:
    if tuple(reference["field_names"]) != tuple(candidate["field_names"]):
        raise ValueError(
            "Residual field-name mismatch: "
            f"{tuple(reference['field_names'])} != {tuple(candidate['field_names'])}"
        )
    for key in ("extra", "cell_volumes", "block_id", "ijk"):
        if np.asarray(reference[key]).shape != np.asarray(candidate[key]).shape:
            raise ValueError(
                f"Residual export shape mismatch for {key}: "
                f"{np.asarray(reference[key]).shape} != {np.asarray(candidate[key]).shape}"
            )
    if not np.array_equal(reference["block_id"], candidate["block_id"]):
        raise ValueError("Residual exports use different block_id layouts")
    if not np.array_equal(reference["ijk"], candidate["ijk"]):
        raise ValueError("Residual exports use different cell ijk layouts")
    if not np.array_equal(reference["block_cell_shape"], candidate["block_cell_shape"]):
        raise ValueError("Residual exports use different block shapes")


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    weight_sum = float(np.sum(weights, dtype=np.float64))
    if weight_sum <= 0.0:
        return float(np.mean(values, dtype=np.float64))
    return float(np.sum(values * weights, dtype=np.float64) / weight_sum)


def _weighted_rms(values: np.ndarray, weights: np.ndarray) -> float:
    weight_sum = float(np.sum(weights, dtype=np.float64))
    if weight_sum <= 0.0:
        return float(math.sqrt(np.mean(np.square(values), dtype=np.float64)))
    return float(math.sqrt(np.sum(np.square(values) * weights, dtype=np.float64) / weight_sum))


def _compute_scalar_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    cell_volumes: np.ndarray,
) -> dict[str, float]:
    signed_error = np.asarray(candidate - reference, dtype=np.float64)
    abs_error = np.abs(signed_error)
    ref_abs = np.abs(reference)
    cand_abs = np.abs(candidate)
    return {
        "reference_mean_abs": float(np.mean(ref_abs, dtype=np.float64)),
        "reference_rms": float(math.sqrt(np.mean(np.square(reference), dtype=np.float64))),
        "candidate_mean_abs": float(np.mean(cand_abs, dtype=np.float64)),
        "candidate_rms": float(math.sqrt(np.mean(np.square(candidate), dtype=np.float64))),
        "score_gap": float(
            math.sqrt(np.mean(np.square(candidate), dtype=np.float64))
            - math.sqrt(np.mean(np.square(reference), dtype=np.float64))
        ),
        "mae": float(np.mean(abs_error, dtype=np.float64)),
        "rmse": float(math.sqrt(np.mean(np.square(signed_error), dtype=np.float64))),
        "max_abs_error": float(np.max(abs_error)),
        "mean_signed_error": float(np.mean(signed_error, dtype=np.float64)),
        "volume_weighted_mae": _weighted_mean(abs_error, cell_volumes),
        "volume_weighted_rmse": _weighted_rms(signed_error, cell_volumes),
    }


def _compute_field_metric_bundle(
    reference_values: np.ndarray,
    candidate_values: np.ndarray,
    cell_volumes: np.ndarray,
    field_names: Sequence[str],
) -> dict[str, Any]:
    fields: dict[str, dict[str, float]] = {}
    mae_values: list[float] = []
    rmse_values: list[float] = []
    vw_rmse_values: list[float] = []
    score_gap_values: list[float] = []
    for index, field_name in enumerate(field_names):
        metrics = _compute_scalar_metrics(
            reference_values[:, index],
            candidate_values[:, index],
            cell_volumes,
        )
        fields[str(field_name)] = metrics
        mae_values.append(float(metrics["mae"]))
        rmse_values.append(float(metrics["rmse"]))
        vw_rmse_values.append(float(metrics["volume_weighted_rmse"]))
        score_gap_values.append(float(metrics["score_gap"]))
    overall = {
        "field_mae_mean": float(np.mean(mae_values, dtype=np.float64)) if mae_values else 0.0,
        "field_rmse_mean": float(np.mean(rmse_values, dtype=np.float64)) if rmse_values else 0.0,
        "field_volume_weighted_rmse_mean": float(np.mean(vw_rmse_values, dtype=np.float64))
        if vw_rmse_values
        else 0.0,
        "field_score_gap_mean": float(np.mean(score_gap_values, dtype=np.float64)) if score_gap_values else 0.0,
    }
    return {"fields": fields, "overall": overall}


def compare_exported_residual_channels(
    *,
    reference_volume_cgns: str | Path,
    candidate_volume_cgns: str | Path,
    output_dir: str | Path,
    metrics_config: ResidualChannelMetricsConfig | Mapping[str, Any] | None = None,
    summary_path: str | Path | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = (
        metrics_config
        if isinstance(metrics_config, ResidualChannelMetricsConfig)
        else residual_metrics_config_from_dict(metrics_config)
    )
    reference_volume_cgns = Path(reference_volume_cgns).expanduser().resolve()
    candidate_volume_cgns = Path(candidate_volume_cgns).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = (
        Path(summary_path).expanduser().resolve()
        if summary_path is not None
        else output_dir / "residual_channel_summary.json"
    )

    reference_npz = output_dir / "reference_full_state.npz"
    candidate_npz = output_dir / "candidate_full_state.npz"
    reference_meta = output_dir / "reference_full_state_meta.json"
    candidate_meta = output_dir / "candidate_full_state_meta.json"
    export_volume_state(
        reference_volume_cgns,
        reference_npz,
        metadata_json=reference_meta,
        extra_fields=cfg.field_names,
        compress=False,
    )
    export_volume_state(
        candidate_volume_cgns,
        candidate_npz,
        metadata_json=candidate_meta,
        extra_fields=cfg.field_names,
        compress=False,
    )

    reference = _load_exported_residual_fields(reference_npz, field_names=cfg.field_names)
    candidate = _load_exported_residual_fields(candidate_npz, field_names=cfg.field_names)
    _validate_matching_export_layout(reference, candidate)

    field_names = tuple(str(name) for name in reference["field_names"])
    near_wall_layers = max(0, int(cfg.near_wall_layers))
    near_wall_mask = reference["ijk"][:, 2] < near_wall_layers if near_wall_layers > 0 else None

    global_metrics = _compute_field_metric_bundle(
        reference["extra"],
        candidate["extra"],
        reference["cell_volumes"],
        field_names,
    )
    near_wall_metrics = None
    if near_wall_mask is not None and bool(np.any(near_wall_mask)):
        near_wall_metrics = _compute_field_metric_bundle(
            reference["extra"][near_wall_mask],
            candidate["extra"][near_wall_mask],
            reference["cell_volumes"][near_wall_mask],
            field_names,
        )

    block_summaries: list[dict[str, Any]] = []
    block_ids = sorted(int(value) for value in np.unique(reference["block_id"]).tolist())
    for block_id in block_ids:
        block_mask = reference["block_id"] == block_id
        block_summary: dict[str, Any] = {
            "block_index": int(block_id),
            "block_name": str(reference["block_names"][block_id]),
            "cell_shape_ijk": [int(v) for v in reference["block_cell_shape"][block_id].tolist()],
            "n_cells": int(np.count_nonzero(block_mask)),
            "metrics": _compute_field_metric_bundle(
                reference["extra"][block_mask],
                candidate["extra"][block_mask],
                reference["cell_volumes"][block_mask],
                field_names,
            ),
        }
        if near_wall_mask is not None:
            block_near_wall_mask = block_mask & near_wall_mask
            block_summary["near_wall_cells"] = int(np.count_nonzero(block_near_wall_mask))
            if bool(np.any(block_near_wall_mask)):
                block_summary["near_wall_metrics"] = _compute_field_metric_bundle(
                    reference["extra"][block_near_wall_mask],
                    candidate["extra"][block_near_wall_mask],
                    reference["cell_volumes"][block_near_wall_mask],
                    field_names,
                )
        block_summaries.append(block_summary)

    summary: dict[str, Any] = {
        "residual_fields": list(field_names),
        "reference_volume_cgns": str(reference_volume_cgns),
        "candidate_volume_cgns": str(candidate_volume_cgns),
        "reference_export_npz": str(reference_npz),
        "candidate_export_npz": str(candidate_npz),
        "reference_export_meta_json": str(reference_meta),
        "candidate_export_meta_json": str(candidate_meta),
        "metrics_config": asdict(cfg),
        "global_metrics": global_metrics,
        "metadata": dict(metadata or {}),
        "n_cells": int(reference["extra"].shape[0]),
        "blocks": block_summaries,
    }
    if near_wall_metrics is not None:
        summary["near_wall_layers"] = int(near_wall_layers)
        summary["near_wall_n_cells"] = int(np.count_nonzero(near_wall_mask))
        summary["near_wall_metrics"] = near_wall_metrics

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True, default=_json_default)
        stream.write("\n")
    summary["summary_path"] = str(summary_path)
    return summary


def run_restart_residual_evaluation(
    *,
    restart_cgns: str | Path,
    source_case_dir: str | Path,
    output_dir: str | Path,
    config: ADflowRestartEvalConfig | Mapping[str, Any] | None = None,
    metrics_config: ResidualChannelMetricsConfig | Mapping[str, Any] | None = None,
    summary_path: str | Path | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = (
        config
        if isinstance(config, ADflowRestartEvalConfig)
        else restart_eval_config_from_dict(config)
    )
    if not cfg.write_volume_solution:
        cfg = replace(cfg, write_volume_solution=True)

    metrics_cfg = (
        metrics_config
        if isinstance(metrics_config, ResidualChannelMetricsConfig)
        else residual_metrics_config_from_dict(metrics_config)
    )
    restart_cgns = Path(restart_cgns).expanduser().resolve()
    source_case_dir = Path(source_case_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = (
        Path(summary_path).expanduser().resolve()
        if summary_path is not None
        else output_dir / "restart_residual_summary.json"
    )

    command = build_adflow_restart_eval_command(
        source_case_dir=source_case_dir,
        restart_cgns=restart_cgns,
        config=cfg,
    )
    env = build_adflow_restart_eval_env(cfg)
    driver_log = output_dir / "adflow_driver.log"
    adflow_log = output_dir / "adflow.log"
    run_start = time.perf_counter()
    success, error_message, return_code = run_subprocess(
        command,
        output_dir,
        env,
        log_file=driver_log,
        timeout=int(cfg.timeout_s),
    )
    elapsed_sec = time.perf_counter() - run_start

    output_json_path = output_dir / "output.json"
    output_payload = None
    if output_json_path.exists():
        with output_json_path.open("r", encoding="utf-8") as stream:
            output_payload = json.load(stream)

    candidate_volume_cgns = output_dir / "wing_000_vol.cgns"
    reference_volume_cgns = source_case_dir / "wing_000_vol.cgns"
    candidate_key = _residual_candidate_key(cfg)
    summary: dict[str, Any] = {
        "residual_eval_method": _residual_eval_method_name(cfg),
        "residual_eval_candidate_key": candidate_key,
        "restart_cgns": str(restart_cgns),
        "source_case_dir": str(source_case_dir),
        "reference_volume_cgns": str(reference_volume_cgns),
        "candidate_volume_cgns": str(candidate_volume_cgns),
        "residual_fields": list(metrics_cfg.field_names),
        "adflow_run": {
            "success": bool(success),
            "return_code": int(return_code),
            "elapsed_sec": float(elapsed_sec),
            "run_dir": str(output_dir),
            "output_json": str(output_json_path) if output_json_path.exists() else None,
            "adflow_log": str(adflow_log),
            "adflow_driver_log": str(driver_log),
            "command": command,
            "config": asdict(cfg),
            "error_message": error_message or None,
        },
        "metadata": dict(metadata or {}),
    }
    note = _resume_proxy_note(cfg)
    if note:
        summary["note"] = note

    if output_payload is not None:
        summary["adflow_run"]["results"] = output_payload.get("results") or {}
        summary["adflow_run"]["working_conditions"] = output_payload.get("working_conditions")
        summary["adflow_run"]["solver_call_protocol"] = output_payload.get("solver_call_protocol")

    if success and candidate_volume_cgns.exists():
        comparison = compare_exported_residual_channels(
            reference_volume_cgns=reference_volume_cgns,
            candidate_volume_cgns=candidate_volume_cgns,
            output_dir=output_dir / "residual_channels",
            metrics_config=metrics_cfg,
            summary_path=output_dir / f"{candidate_key}_residual_channels.json",
            metadata={
                **dict(metadata or {}),
                "restart_cgns": str(restart_cgns),
                "candidate_key": candidate_key,
            },
        )
        summary[candidate_key] = {
            "global_metrics": comparison["global_metrics"],
            "near_wall_metrics": comparison.get("near_wall_metrics"),
            "summary_path": comparison["summary_path"],
        }
        summary["residual_channel_summary_json"] = str(comparison["summary_path"])
        summary["residual_channel_global_rmse_mean"] = float(
            comparison["global_metrics"]["overall"]["field_rmse_mean"]
        )
        summary["residual_channel_global_vw_rmse_mean"] = float(
            comparison["global_metrics"]["overall"]["field_volume_weighted_rmse_mean"]
        )
    else:
        if error_message:
            summary["adflow_run"]["error_message"] = error_message
        adflow_tail = _tail_text(adflow_log)
        if adflow_tail:
            summary["adflow_run"]["adflow_log_tail"] = adflow_tail
        driver_tail = _tail_text(driver_log)
        if driver_tail:
            summary["adflow_run"]["driver_log_tail"] = driver_tail

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True, default=_json_default)
        stream.write("\n")
    summary["summary_path"] = str(summary_path)
    return summary


__all__ = [
    "DEFAULT_RESIDUAL_FIELDS",
    "ResidualChannelMetricsConfig",
    "compare_exported_residual_channels",
    "residual_metrics_config_from_dict",
    "run_restart_residual_evaluation",
]
