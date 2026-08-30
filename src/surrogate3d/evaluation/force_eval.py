"""Restart-based ADFLOW force evaluation for surrogate3d predictions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any, Mapping, Optional

from mpi4py import MPI
from adflow import ADFLOW
from baseclasses import AeroProblem

from parametric_swept_wing_cfd.core.execution import run_subprocess
from parametric_swept_wing_cfd.solver.adflow import redirect_stdout

from surrogate3d.evaluation.adflow_restart import (
    ADflowRestartEvalConfig,
    build_adflow_restart_eval_env,
    restart_eval_config_from_dict,
)


@dataclass(frozen=True)
class ADflowForceEvalConfig(ADflowRestartEvalConfig):
    """Execution settings for restart-based ADFLOW force evaluation."""

    def method_name(self) -> str:
        return "adflow_pure_force_eval0"

    def candidate_key(self) -> str:
        return "surrogate_adflow_force_eval"

    def run_suffix(self) -> str:
        suffix = self.candidate_key()
        return f"{suffix}_np{int(self.mpi_ranks)}"

    def proxy_note(self) -> Optional[str]:
        return None


def force_eval_config_from_dict(config: Mapping[str, Any] | None = None) -> ADflowForceEvalConfig:
    return restart_eval_config_from_dict(config, config_cls=ADflowForceEvalConfig)


def _force_triplet_from_results(results: Mapping[str, Any]) -> dict[str, float]:
    return {
        "CL": float(results["wing_cl"]),
        "CD": float(results["wing_cd"]),
        "CMz": float(results["wing_cmz"]),
    }


def read_reference_force_coefficients(output_json: str | Path) -> dict[str, float]:
    output_json = Path(output_json).expanduser().resolve()
    with output_json.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    results = payload.get("results") or {}
    missing = [key for key in ("wing_cl", "wing_cd", "wing_cmz") if key not in results]
    if missing:
        raise KeyError(f"Reference output is missing force coefficients {missing}: {output_json}")
    return _force_triplet_from_results(results)


def compute_force_error_metrics(
    reference: Mapping[str, float],
    candidate: Mapping[str, float],
) -> dict[str, float]:
    ref_cl = float(reference["CL"])
    ref_cd = float(reference["CD"])
    ref_cmz = float(reference["CMz"])
    cand_cl = float(candidate["CL"])
    cand_cd = float(candidate["CD"])
    cand_cmz = float(candidate["CMz"])

    dcl = cand_cl - ref_cl
    dcd = cand_cd - ref_cd
    dcmz = cand_cmz - ref_cmz
    eps = 1.0e-12
    return {
        "abs_dCL": abs(dcl),
        "abs_dCD": abs(dcd),
        "abs_dCMz": abs(dcmz),
        "dCL": dcl,
        "dCD": dcd,
        "dCMz": dcmz,
        "rel_CL": abs(dcl) / max(abs(ref_cl), eps),
        "rel_CD": abs(dcd) / max(abs(ref_cd), eps),
        "rel_CMz": abs(dcmz) / max(abs(ref_cmz), eps),
        "force_l1_cl_cd": abs(dcl) + abs(dcd),
        "force_l1_cl_cd_cmz": abs(dcl) + abs(dcd) + abs(dcmz),
        "force_l2_cl_cd": (dcl * dcl + dcd * dcd) ** 0.5,
    }


def build_adflow_force_eval_command(
    *,
    source_case_dir: str | Path,
    restart_cgns: str | Path,
    config: ADflowForceEvalConfig,
) -> list[str]:
    source_case_dir = Path(source_case_dir).expanduser().resolve()
    restart_cgns = Path(restart_cgns).expanduser().resolve()
    helper = Path(__file__).with_name("force_eval.py").resolve()
    return [
        "mpirun",
        "-np",
        str(int(config.mpi_ranks)),
        str(Path(config.python_executable).expanduser().resolve()),
        str(helper),
        "--grid",
        str((source_case_dir / "wing_000_vol.cgns").resolve()),
        "--input",
        str((source_case_dir.parent / "input.json").resolve()),
        "--conds",
        str((source_case_dir / "conds.json").resolve()),
        "--restart",
        str(restart_cgns),
    ]


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


def _path(text: str) -> str:
    return str(Path(text).expanduser().resolve())


@redirect_stdout("adflow.log")
def _run_force_eval_main(*, grid: str, input_json: str, conds_json: str, restart: str) -> None:
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        with open(conds_json, "r", encoding="utf-8") as handle:
            conds = json.load(handle)
        with open(input_json, "r", encoding="utf-8") as handle:
            wing = json.load(handle)
    else:
        conds = None
        wing = None
    conds = comm.bcast(conds, root=0)
    wing = comm.bcast(wing, root=0)

    opts = {
        "gridFile": grid,
        "outputDirectory": ".",
        "monitorvariables": ["resrho", "cl", "cd", "cmz", "resturb"],
        "writeTecplotSurfaceSolution": False,
        "writeSurfaceSolution": False,
        "writeVolumeSolution": False,
        "equationType": "RANS",
        "MGCycle": "3w",
        "restartFile": restart,
        "nCyclesCoarse": 500,
        "nCycles": 0,
    }
    solver = ADFLOW(comm=comm, options=opts)
    ap = AeroProblem(
        name="wing",
        mach=conds["M3"],
        alpha=conds["AA"],
        reynolds=conds["re"],
        reynoldsLength=float(wing.get("reynolds_length", 1.0)),
        T=conds["temp"],
        areaRef=float(wing["surface_area"]),
        chordRef=float(wing.get("chord_ref", wing.get("ref_chord", 1.0))),
        xRef=float(wing.get("x_ref", wing.get("xRef", 0.25))),
        evalFuncs=["cl", "cd", "cmx", "cmy", "cmz"],
    )
    if wing.get("y_ref") is not None:
        ap.yRef = float(wing["y_ref"])
    if wing.get("z_ref") is not None:
        ap.zRef = float(wing["z_ref"])
    solver(ap, writeSolution=False)
    funcs = {}
    solver.evalFunctions(ap, funcs)
    if comm.rank == 0:
        payload = {"results": {key: float(value) for key, value in funcs.items()}}
        with open("output.json", "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        print(json.dumps(payload, indent=2))


def run_restart_force_evaluation(
    *,
    restart_cgns: str | Path,
    source_case_dir: str | Path,
    output_dir: str | Path,
    config: ADflowForceEvalConfig | Mapping[str, Any] | None = None,
    summary_path: str | Path | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = config if isinstance(config, ADflowForceEvalConfig) else force_eval_config_from_dict(config)
    restart_cgns = Path(restart_cgns).expanduser().resolve()
    source_case_dir = Path(source_case_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = (
        Path(summary_path).expanduser().resolve()
        if summary_path is not None
        else output_dir / "force_summary.json"
    )
    reference_output_json = source_case_dir / "output.json"
    reference = read_reference_force_coefficients(reference_output_json)

    command = build_adflow_force_eval_command(
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

    candidate_key = cfg.candidate_key()
    summary: dict[str, Any] = {
        "force_eval_method": cfg.method_name(),
        "force_eval_candidate_key": candidate_key,
        "reference": reference,
        "reference_output_json": str(reference_output_json),
        "restart_cgns": str(restart_cgns),
        "source_case_dir": str(source_case_dir),
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
    note = cfg.proxy_note()
    if note:
        summary["note"] = note

    if output_payload is not None:
        results = output_payload.get("results") or {}
        if {"wing_cl", "wing_cd", "wing_cmz"}.issubset(results):
            candidate = _force_triplet_from_results(results)
            summary[candidate_key] = candidate
            summary.update(compute_force_error_metrics(reference, candidate))
        summary["adflow_run"]["results"] = results
        summary["adflow_run"]["working_conditions"] = output_payload.get("working_conditions")
        summary["adflow_run"]["solver_call_protocol"] = output_payload.get("solver_call_protocol")

    if not success:
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
    "ADflowForceEvalConfig",
    "build_adflow_force_eval_command",
    "compute_force_error_metrics",
    "force_eval_config_from_dict",
    "read_reference_force_coefficients",
    "run_restart_force_evaluation",
]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--conds", required=True)
    parser.add_argument("--restart", required=True)
    args = parser.parse_args()
    _run_force_eval_main(
        grid=_path(args.grid),
        input_json=_path(args.input),
        conds_json=_path(args.conds),
        restart=_path(args.restart),
    )
