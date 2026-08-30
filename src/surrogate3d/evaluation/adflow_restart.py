"""Shared ADFLOW restart-evaluation helpers for surrogate3d physics checks."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, TypeVar

from parametric_swept_wing_cfd.config import DEFAULT_L2_TOL, DEFAULT_NCORE_PER_JOB, PYTHON_PATH, setup_environment


def default_adflow_eval_python() -> str:
    """Pick a Python executable that matches the active conda env when possible."""
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidate = Path(conda_prefix).expanduser().resolve() / "bin" / "python"
        if candidate.exists():
            return str(candidate)
    if PYTHON_PATH:
        candidate = Path(PYTHON_PATH).expanduser()
        if candidate.exists():
            return str(candidate.resolve())
    return str(Path(sys.executable).resolve())


@dataclass(frozen=True)
class ADflowRestartEvalConfig:
    """Execution settings shared by restart-based physics evaluators."""

    mpi_ranks: int = DEFAULT_NCORE_PER_JOB
    python_executable: str = default_adflow_eval_python()
    cycle_protocol: str = "resume"
    cycles: int = 1
    inner_cycles: int = 1
    solver_calls: Optional[int] = None
    solver_preset: Optional[str] = None
    l2_tol: float = DEFAULT_L2_TOL
    timeout_s: int = 7200
    write_volume_solution: bool = False
    write_surface_solution: bool = False
    write_each_solution: bool = False
    verbose: bool = False
    solution_precision: Optional[str] = None
    refresh_residual_before_write: bool = False
    write_full_flow_residual: bool = False


TConfig = TypeVar("TConfig", bound=ADflowRestartEvalConfig)


def restart_eval_config_from_dict(
    config: Mapping[str, Any] | None = None,
    *,
    config_cls: type[TConfig] = ADflowRestartEvalConfig,
) -> TConfig:
    payload = dict(config or {})
    if "mpi_ranks" in payload:
        payload["mpi_ranks"] = int(payload["mpi_ranks"])
    if "cycles" in payload:
        payload["cycles"] = int(payload["cycles"])
    if "inner_cycles" in payload:
        payload["inner_cycles"] = int(payload["inner_cycles"])
    if "solver_calls" in payload and payload["solver_calls"] is not None:
        payload["solver_calls"] = int(payload["solver_calls"])
    if "timeout_s" in payload:
        payload["timeout_s"] = int(payload["timeout_s"])
    if "l2_tol" in payload:
        payload["l2_tol"] = float(payload["l2_tol"])
    return config_cls(**payload)


def build_adflow_restart_eval_command(
    *,
    source_case_dir: str | Path,
    restart_cgns: str | Path,
    config: ADflowRestartEvalConfig,
) -> list[str]:
    source_case_dir = Path(source_case_dir).expanduser().resolve()
    restart_cgns = Path(restart_cgns).expanduser().resolve()
    command = [
        "mpirun",
        "-np",
        str(int(config.mpi_ranks)),
        str(Path(config.python_executable).expanduser().resolve()),
        "-m",
        "parametric_swept_wing_cfd.solver.adflow",
        "--grid",
        str((source_case_dir / "wing_000_vol.cgns").resolve()),
        "--input",
        str((source_case_dir.parent / "input.json").resolve()),
        "--conds",
        str((source_case_dir / "conds.json").resolve()),
        "--restart-file",
        str(restart_cgns),
        "--cycle-protocol",
        str(config.cycle_protocol),
        "--cycles",
        str(int(config.cycles)),
        "--inner-cycles",
        str(int(config.inner_cycles)),
        "--l2-tol",
        str(float(config.l2_tol)),
    ]
    if config.solver_calls is not None:
        command.extend(["--solver-calls", str(int(config.solver_calls))])
    if config.solver_preset:
        command.extend(["--solver-preset", str(config.solver_preset)])
    if not config.write_volume_solution:
        command.append("--no-volume-solution")
    if not config.write_surface_solution:
        command.append("--no-surface-solution")
    if config.write_each_solution:
        command.append("--write-each-solution")
    if config.solution_precision is not None:
        command.extend(["--solution-precision", str(config.solution_precision)])
    if config.refresh_residual_before_write:
        command.append("--refresh-residual-before-write")
    if config.write_full_flow_residual:
        command.append("--write-full-flow-residual")
    if config.verbose:
        command.append("--verbose")
    return command


def build_adflow_restart_eval_env(config: ADflowRestartEvalConfig) -> dict[str, str]:
    env = setup_environment()
    python_executable = str(Path(config.python_executable).expanduser().resolve())
    env["PIPELINE_PYTHON"] = python_executable
    conda_bin = str(Path(python_executable).parent)
    path_parts = [part for part in env.get("PATH", "").split(":") if part]
    path_parts = [part for part in path_parts if part != conda_bin]
    path_parts.insert(0, conda_bin)
    env["PATH"] = ":".join(path_parts)
    return env


__all__ = [
    "ADflowRestartEvalConfig",
    "build_adflow_restart_eval_command",
    "build_adflow_restart_eval_env",
    "default_adflow_eval_python",
    "restart_eval_config_from_dict",
]
