"""NK recovery and whole-field metrics for surrogate3d CGNS restarts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Mapping

import numpy as np
import torch

from NK_resume.metrics import field_metrics, residual_metrics
from parametric_swept_wing_cfd.core.execution import run_subprocess
from parametric_swept_wing_cfd.export import export_volume_state

from surrogate3d.evaluation.adflow_restart import (
    ADflowRestartEvalConfig,
    build_adflow_restart_eval_command,
    build_adflow_restart_eval_env,
)
from surrogate3d.evaluation.field_force_eval import read_force_case_metadata
from surrogate3d.evaluation.force_eval import compute_force_error_metrics
from surrogate3d.physics.field_force import (
    ForceEvaluatorConfig,
    compute_force_coefficients_torch,
    prepare_volume_force_geometry,
)
from surrogate3d.physics.surface_forces import compute_surface_field_metrics
from surrogate3d.physics.wall_geometry import extract_case_wall_geometry_from_volume_cgns


@dataclass(frozen=True)
class NKRecoveryConfig:
    """Execution controls for one 3D surrogate-to-NK recovery."""

    mpi_ranks: int = 8
    cycles: int = 10
    inner_cycles: int = 1
    l2_tol: float = 1.0e-8
    timeout_s: int = 7200
    python_executable: str = sys.executable
    write_each_solution: bool = False


def _restart_config(config: NKRecoveryConfig) -> ADflowRestartEvalConfig:
    return ADflowRestartEvalConfig(
        mpi_ranks=int(config.mpi_ranks),
        python_executable=str(config.python_executable),
        cycle_protocol="resume",
        cycles=int(config.cycles),
        inner_cycles=int(config.inner_cycles),
        solver_preset="nk",
        l2_tol=float(config.l2_tol),
        timeout_s=int(config.timeout_s),
        write_volume_solution=True,
        write_surface_solution=False,
        write_each_solution=bool(config.write_each_solution),
    )


def _load_export(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def _validate_layout(reference: Mapping[str, np.ndarray], candidate: Mapping[str, np.ndarray]) -> None:
    for key in ("state", "cell_volumes", "block_id", "ijk", "block_cell_shape"):
        if np.asarray(reference[key]).shape != np.asarray(candidate[key]).shape:
            raise ValueError(
                f"Recovery export layout mismatch for {key}: "
                f"{np.asarray(reference[key]).shape} != {np.asarray(candidate[key]).shape}"
            )
    for key in ("block_id", "ijk", "block_cell_shape", "block_names", "state_fields"):
        if not np.array_equal(reference[key], candidate[key]):
            raise ValueError(f"Recovery export layout mismatch for {key}")


def _normalized_flat_state(state: np.ndarray, normalizer: torch.nn.Module) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(state, dtype=np.float32).T).unsqueeze(0)
    transformed = normalizer.transform(tensor)
    return transformed.squeeze(0).T.detach().cpu().numpy()


def _weighted_comparison(
    candidate: np.ndarray,
    reference: np.ndarray,
    volumes: np.ndarray,
) -> dict[str, float]:
    weights = np.asarray(volumes, dtype=np.float64).reshape(-1, 1)
    error = np.asarray(candidate, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    weighted_count = float(np.sum(weights) * error.shape[1])
    mse = float(np.sum(weights * error**2) / weighted_count)
    return {
        "mse": mse,
        "rmse": float(math.sqrt(mse)),
    }


def _volume_weighted_metrics(
    recovered: np.ndarray,
    reference: np.ndarray,
    initial: np.ndarray,
    volumes: np.ndarray,
) -> dict[str, Any]:
    recovered_metrics = _weighted_comparison(recovered, reference, volumes)
    initial_metrics = _weighted_comparison(initial, reference, volumes)
    return {
        "candidate_vs_reference": recovered_metrics,
        "initial_vs_reference": initial_metrics,
        "improvement": {
            "relative_mse_reduction": 1.0 - recovered_metrics["mse"] / initial_metrics["mse"],
        },
    }


def _parse_residual_history(
    log_path: Path,
    *,
    threshold: float,
    uniform_total_residual: float,
) -> dict[str, Any]:
    resrho: list[float] = []
    total: list[float] = []
    budgets: list[int] = []
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) < 13:
            continue
        try:
            int(parts[0])
            local_iteration = int(parts[1])
            iteration = int(parts[2])
            rho_value = float(parts[7])
            total_value = float(parts[12])
        except ValueError:
            continue
        if math.isfinite(rho_value) and math.isfinite(total_value):
            # Every resumed solver call repeats an Iter=0 snapshot. Retain only
            # the first baseline and each subsequent solved state.
            if total and local_iteration == 0:
                continue
            budgets.append(iteration)
            resrho.append(rho_value)
            total.append(total_value)
    if not total:
        raise RuntimeError(f"No residual history could be parsed from {log_path}")
    if not math.isfinite(uniform_total_residual) or uniform_total_residual <= 0.0:
        raise ValueError("uniform_total_residual must be positive and finite")
    total_ratio = [value / uniform_total_residual for value in total]
    resrho_ratio = [value / resrho[0] for value in resrho]
    return {
        "total_residual": residual_metrics(
            total_ratio,
            budgets=budgets,
            threshold=float(threshold),
            metadata={
                "source": "adflow.log",
                "value_kind": "ratio_to_uniform_totalR0",
                "uniform_totalR0": uniform_total_residual,
            },
        ),
        "density_residual": residual_metrics(
            resrho_ratio,
            budgets=budgets,
            threshold=float(threshold),
            metadata={"source": "adflow.log", "value_kind": "ratio_to_first_ResRho"},
        ),
    }


def _block_fields(export: Mapping[str, np.ndarray]) -> dict[int, torch.Tensor]:
    state = np.asarray(export["state"], dtype=np.float64)
    block_ids = np.asarray(export["block_id"], dtype=np.int32)
    shapes = np.asarray(export["block_cell_shape"], dtype=np.int32)
    fields: dict[int, torch.Tensor] = {}
    for block_index, shape_array in enumerate(shapes):
        shape = tuple(int(value) for value in shape_array)
        block = state[block_ids == block_index].reshape(*shape, state.shape[1])
        fields[block_index] = torch.from_numpy(np.moveaxis(block, -1, 0)).unsqueeze(0)
    return fields


def _coefficient_record(coefficients: Mapping[str, Any]) -> dict[str, float]:
    return {
        key: float(torch.as_tensor(coefficients[key]).reshape(-1)[0].detach().cpu())
        for key in ("CL", "CD", "CMz", "Cx", "Cy")
    }


def _force_torch_metrics(
    *,
    initial_export: Mapping[str, np.ndarray],
    recovered_export: Mapping[str, np.ndarray],
    target_export: Mapping[str, np.ndarray],
    source_case_dir: Path,
) -> dict[str, Any]:
    volume_cgns = source_case_dir / "wing_000_vol.cgns"
    geometry = extract_case_wall_geometry_from_volume_cgns(volume_cgns, dtype=torch.float64)
    prepared = prepare_volume_force_geometry(geometry, device="cpu", dtype=torch.float64)
    case_metadata = read_force_case_metadata(
        input_json=source_case_dir.parent / "input.json",
        conds_json=source_case_dir / "conds.json",
    )
    with (source_case_dir / "conds.json").open("r", encoding="utf-8") as stream:
        conditions = json.load(stream)
    flow = torch.tensor(
        [[conditions["M3"], conditions["AA"], conditions["re"]]],
        dtype=torch.float64,
    )
    kwargs = {
        "area_ref": float(case_metadata["area_ref"]),
        "chord_ref": float(case_metadata["chord_ref"]),
        "moment_center": (
            float(case_metadata["x_ref"]),
            float(case_metadata["y_ref"]),
            float(case_metadata["z_ref"]),
        ),
        "config": ForceEvaluatorConfig(),
        "prepared_geometry": prepared,
        "device": "cpu",
        "dtype": torch.float64,
    }
    initial_result = compute_force_coefficients_torch(
        _block_fields(initial_export),
        geometry,
        flow,
        return_surface_fields=True,
        **kwargs,
    )
    recovered_result = compute_force_coefficients_torch(
        _block_fields(recovered_export),
        geometry,
        flow,
        return_surface_fields=True,
        **kwargs,
    )
    target_result = compute_force_coefficients_torch(
        _block_fields(target_export),
        geometry,
        flow,
        return_surface_fields=True,
        **kwargs,
    )
    initial = _coefficient_record(initial_result)
    recovered = _coefficient_record(recovered_result)
    target = _coefficient_record(target_result)
    return {
        "initial": initial,
        "recovered": recovered,
        "target": target,
        "initial_vs_target": compute_force_error_metrics(target, initial),
        "recovered_vs_target": compute_force_error_metrics(target, recovered),
        "wall_fields": {
            "initial_vs_target": compute_surface_field_metrics(
                initial_result["surface_fields"],
                target_result["surface_fields"],
                cp_index=0,
                cf_indices=(1, 2, 3),
            ),
            "recovered_vs_target": compute_surface_field_metrics(
                recovered_result["surface_fields"],
                target_result["surface_fields"],
                cp_index=0,
                cf_indices=(1, 2, 3),
            ),
        },
    }


def _find_recovered_cgns(output_dir: Path) -> Path:
    candidates = sorted(output_dir.glob("wing_*_vol.cgns"))
    if not candidates:
        raise FileNotFoundError(f"NK recovery did not write a wing_*_vol.cgns under {output_dir}")
    return candidates[-1]


def summarize_nk_recovery(
    *,
    restart_cgns: str | Path,
    source_case_dir: str | Path,
    output_dir: str | Path,
    normalizer: torch.nn.Module,
    config: NKRecoveryConfig,
    elapsed_s: float,
    command: list[str],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize an already-completed 3D NK recovery run."""

    restart_cgns = Path(restart_cgns).expanduser().resolve()
    source_case_dir = Path(source_case_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    recovered_cgns = _find_recovered_cgns(output_dir)

    with tempfile.TemporaryDirectory(prefix="nk_recovery_metrics_", dir=output_dir) as temp_text:
        temp_dir = Path(temp_text)
        paths = {
            "initial": temp_dir / "initial.npz",
            "recovered": temp_dir / "recovered.npz",
            "target": temp_dir / "target.npz",
        }
        export_volume_state(restart_cgns, paths["initial"], dtype="float32", compress=False)
        export_volume_state(recovered_cgns, paths["recovered"], dtype="float32", compress=False)
        export_volume_state(
            source_case_dir / "wing_000_vol.cgns",
            paths["target"],
            dtype="float32",
            compress=False,
        )
        exports = {name: _load_export(path) for name, path in paths.items()}

        _validate_layout(exports["target"], exports["initial"])
        _validate_layout(exports["target"], exports["recovered"])
        state_fields = [str(value) for value in exports["target"]["state_fields"].tolist()]

        physical_metrics = field_metrics(
            exports["recovered"]["state"],
            exports["target"]["state"],
            initial=exports["initial"]["state"],
            channel_axis=1,
            channel_names=state_fields,
            metadata={"space": "physical", "source": "surrogate3d_cgns"},
        )
        normalized = {
            name: _normalized_flat_state(export["state"], normalizer)
            for name, export in exports.items()
        }
        normalized_metrics = field_metrics(
            normalized["recovered"],
            normalized["target"],
            initial=normalized["initial"],
            channel_axis=1,
            channel_names=state_fields,
            metadata={"space": "training_normalized", "source": "surrogate3d_cgns"},
        )
        weighted_metrics = _volume_weighted_metrics(
            normalized["recovered"],
            normalized["target"],
            normalized["initial"],
            exports["target"]["cell_volumes"],
        )
        force_metrics_payload = _force_torch_metrics(
            initial_export=exports["initial"],
            recovered_export=exports["recovered"],
            target_export=exports["target"],
            source_case_dir=source_case_dir,
        )

    output_json = output_dir / "output.json"
    solver_output = json.loads(output_json.read_text(encoding="utf-8"))
    trace = solver_output["solver_call_trace"]
    if not trace:
        raise ValueError(f"Empty solver_call_trace in {output_json}")
    reported_totalr0 = np.asarray(
        [float(point["totalr0_reported"]) for point in trace],
        dtype=np.float64,
    )
    if (
        not np.isfinite(reported_totalr0).all()
        or np.any(reported_totalr0 <= 0.0)
        or not np.all(reported_totalr0 == reported_totalr0[0])
    ):
        raise ValueError(f"Invalid or inconsistent uniform totalR0 in {output_json}")
    uniform_total_residual = float(reported_totalr0[0])
    summary = {
        "schema_version": "surrogate3d_nk_recovery_v1",
        "method": "surrogate_cgns_restart_to_nk_recovery",
        "restart_cgns": str(restart_cgns),
        "recovered_cgns": str(recovered_cgns),
        "target_cgns": str((source_case_dir / "wing_000_vol.cgns").resolve()),
        "config": asdict(config),
        "command": command,
        "elapsed_s": float(elapsed_s),
        "solver_output": solver_output,
        "residual": _parse_residual_history(
            output_dir / "adflow.log",
            threshold=float(config.l2_tol),
            uniform_total_residual=uniform_total_residual,
        ),
        "field_metrics": {
            "physical": physical_metrics,
            "normalized": normalized_metrics,
            "normalized_cell_volume_weighted": weighted_metrics,
        },
        "force_torch": force_metrics_payload,
        "metadata": dict(metadata or {}),
    }
    summary_path = output_dir / "nk_recovery_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def run_nk_recovery(
    *,
    restart_cgns: str | Path,
    source_case_dir: str | Path,
    output_dir: str | Path,
    normalizer: torch.nn.Module,
    config: NKRecoveryConfig | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Recover one surrogate3d restart and compute NK_resume metric schemas."""

    cfg = config or NKRecoveryConfig()
    restart_cfg = _restart_config(cfg)
    restart_cgns = Path(restart_cgns).expanduser().resolve()
    source_case_dir = Path(source_case_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    command = build_adflow_restart_eval_command(
        source_case_dir=source_case_dir,
        restart_cgns=restart_cgns,
        config=restart_cfg,
    )
    started = time.perf_counter()
    success, error_message, return_code = run_subprocess(
        command,
        output_dir,
        build_adflow_restart_eval_env(restart_cfg),
        log_file=output_dir / "adflow_driver.log",
        timeout=int(cfg.timeout_s),
    )
    elapsed_s = time.perf_counter() - started
    if not success:
        raise RuntimeError(
            f"NK recovery failed with return code {return_code}: {error_message or 'unknown error'}"
        )

    return summarize_nk_recovery(
        restart_cgns=restart_cgns,
        source_case_dir=source_case_dir,
        output_dir=output_dir,
        normalizer=normalizer,
        config=cfg,
        elapsed_s=elapsed_s,
        command=command,
        metadata=metadata,
    )


__all__ = ["NKRecoveryConfig", "run_nk_recovery", "summarize_nk_recovery"]
