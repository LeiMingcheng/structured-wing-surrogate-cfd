"""Evaluate surrogate3d checkpoints, render diagnostics, and export NK payloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from surrogate3d.evaluation import (
    ADflowForceEvalConfig,
    predict_case,
    run_restart_force_evaluation,
)
from surrogate3d.nk_resume import write_whole_case_payload
from surrogate3d.nk_resume.cgns_restart import DEFAULT_RESTART_FIELD_FLOORS, write_prediction_restart_cgns
from surrogate3d.visualization import plot_prediction_diagnostics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="surrogate3d checkpoint .pt")
    parser.add_argument("--index", type=Path, default=None, help="Override dataset index.csv")
    parser.add_argument("--output-dir", type=Path, required=True, help="Evaluation output directory")
    parser.add_argument("--case-index", type=int, default=None, help="Whole-case dataset index to evaluate")
    parser.add_argument(
        "--holdout-start",
        type=int,
        default=4,
        help="Default case index when --case-index is omitted; overfit small run used first 4 cases",
    )
    parser.add_argument("--device", default="cuda", help="Torch device")
    parser.add_argument(
        "--fields",
        default="Pressure,VelocityX,TurbulentSANuTilde",
        help="Comma-separated fields to visualize",
    )
    parser.add_argument(
        "--slice-axis",
        default="z",
        choices=["i", "j", "k", "x", "y", "z"],
        help="3D slice axis. Use x/y/z for physical-coordinate cuts, i/j/k for block-index cuts.",
    )
    parser.add_argument("--slice-index", type=int, default=None, help="Single slice index; default uses --slice-fractions")
    parser.add_argument(
        "--slice-fractions",
        default="0.18,0.35,0.55,0.75",
        help="Comma-separated normalized stations used when --slice-index is omitted",
    )
    parser.add_argument(
        "--surface-layer-k",
        type=int,
        default=0,
        help="Cell-centered near-wall k-layer. CGNS wall is the k=1 node boundary, so default is k=0.",
    )
    parser.add_argument("--surface-stride", type=int, default=1, help="Subsample wall-surface rendering")
    parser.add_argument("--slice-stride", type=int, default=1, help="Subsample slice rendering")
    parser.add_argument(
        "--slice-wall-layers",
        type=int,
        default=36,
        help="For i/j slices, only render this many near-wall k-layers; <=0 renders full farfield slices.",
    )
    parser.add_argument(
        "--min-slice-axis-points",
        type=int,
        default=32,
        help="Skip blocks with fewer cells along the selected slice axis",
    )
    parser.add_argument("--surface-layer-j", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--surface-projection", default=None, choices=["xy", "xz", "yz"], help=argparse.SUPPRESS)
    parser.add_argument("--slice-projection", default=None, choices=["xy", "xz", "yz"], help=argparse.SUPPRESS)
    parser.add_argument("--no-figures", action="store_true", help="Skip PNG visualization")
    parser.add_argument("--no-payload", action="store_true", help="Skip HDF5 payload export")
    parser.add_argument("--no-cgns-restart", action="store_true", help="Skip CGNS restart export")
    parser.add_argument(
        "--write-target-cgns-restart",
        action="store_true",
        help="Also export a CGNS restart containing the dataset target state for ADFLOW/NK controls.",
    )
    parser.add_argument(
        "--restart-positivity-projection",
        action="store_true",
        help="Floor density/pressure/SA-nu-tilde only in the exported CGNS restart.",
    )
    parser.add_argument(
        "--template-cgns",
        type=Path,
        default=None,
        help="Template volume CGNS. Defaults to source_case_dir/wing_000_vol.cgns.",
    )
    parser.add_argument(
        "--run-force-eval",
        action="store_true",
        help="Run restart-based ADFLOW force evaluation against the exported surrogate restart.",
    )
    parser.add_argument(
        "--force-eval-ranks",
        type=int,
        default=10,
        help="MPI ranks for restart-based ADFLOW force evaluation (default: 10).",
    )
    parser.add_argument(
        "--force-cycles",
        type=int,
        default=1,
        help="ADFLOW --cycles for restart-based force evaluation (default: 1).",
    )
    parser.add_argument(
        "--force-inner-cycles",
        type=int,
        default=1,
        help="ADFLOW --inner-cycles for restart-based force evaluation (default: 1).",
    )
    parser.add_argument(
        "--force-cycle-protocol",
        choices=["resume", "single-call"],
        default="resume",
        help="ADFLOW cycle protocol for restart-based force evaluation (default: resume).",
    )
    parser.add_argument(
        "--force-solver-preset",
        choices=["prod", "pseudo", "ank1", "ank2", "nk"],
        default=None,
        help="Optional ADFLOW solver preset for restart-based force evaluation.",
    )
    parser.add_argument(
        "--force-solver-calls",
        type=int,
        default=None,
        help="Optional override for ADFLOW repeated solver(ap) calls during force eval.",
    )
    parser.add_argument(
        "--force-l2-tol",
        type=float,
        default=1.0e-10,
        help="ADFLOW L2 tolerance for restart-based force evaluation.",
    )
    parser.add_argument(
        "--force-timeout-s",
        type=int,
        default=7200,
        help="Timeout in seconds for restart-based force evaluation.",
    )
    parser.add_argument(
        "--force-python",
        type=Path,
        default=None,
        help="Python executable used inside the ADFLOW force-eval subprocess.",
    )
    parser.add_argument(
        "--force-write-volume-solution",
        action="store_true",
        help="Keep ADFLOW volume CGNS output during force evaluation.",
    )
    parser.add_argument(
        "--force-write-surface-solution",
        action="store_true",
        help="Keep ADFLOW surface CGNS output during force evaluation.",
    )
    return parser


def _fields_from_arg(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def main() -> int:
    args = build_parser().parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction = predict_case(
        args.checkpoint,
        index_path=args.index,
        case_index=args.case_index,
        holdout_start=args.holdout_start,
        device=args.device,
    )
    batch = prediction.batch
    sample_id = str(batch["sample_id"][0])
    cond_idx = int(batch["cond_idx"][0].detach().cpu())
    source_case_dir = Path(str(batch["source_case_dir"][0])).expanduser().resolve()
    case_stem = f"{sample_id}_cond{cond_idx}_case{prediction.case_index}"

    artifacts: dict[str, Any] = {
        "case_index": prediction.case_index,
        "sample_id": sample_id,
        "cond_idx": cond_idx,
        "source_case_dir": str(source_case_dir),
    }

    if not args.no_figures:
        figure_dir = output_dir / "figures" / case_stem
        figure_paths = plot_prediction_diagnostics(
            prediction,
            output_dir=figure_dir,
            fields=_fields_from_arg(args.fields),
            surface_layer_k=int(args.surface_layer_k),
            slice_axis=args.slice_axis,
            slice_index=args.slice_index,
            slice_fractions=args.slice_fractions,
            surface_stride=int(args.surface_stride),
            slice_stride=int(args.slice_stride),
            slice_wall_layers=int(args.slice_wall_layers) if int(args.slice_wall_layers) > 0 else None,
            min_slice_axis_points=int(args.min_slice_axis_points),
            surface_projection=args.surface_projection,
            slice_projection=args.slice_projection,
            surface_layer_j=args.surface_layer_j,
        )
        artifacts["figures"] = [str(path) for path in figure_paths]

    if not args.no_payload:
        payload_path = output_dir / "payloads" / f"{case_stem}_whole_case_payload.h5"
        write_whole_case_payload(
            payload_path,
            predicted_blocks=prediction.pred_physical,
            target_blocks=prediction.target_physical,
            batch=batch,
            block_catalog=prediction.checkpoint_payload.get("block_catalog"),
            metadata={
                "checkpoint": str(args.checkpoint.expanduser().resolve()),
                "case_index": prediction.case_index,
                "role": "holdout" if prediction.case_index >= int(args.holdout_start) else "train_or_unspecified",
            },
        )
        artifacts["payload_h5"] = str(payload_path)

    template_cgns = args.template_cgns
    if template_cgns is None:
        template_cgns = source_case_dir / "wing_000_vol.cgns"

    restart_path: Path | None = None
    if not args.no_cgns_restart:
        field_floors = DEFAULT_RESTART_FIELD_FLOORS if args.restart_positivity_projection else None
        restart_suffix = "surrogate_restart_projected_vol.cgns" if field_floors else "surrogate_restart_vol.cgns"
        restart_path = output_dir / "nk_restart" / case_stem / restart_suffix
        write_prediction_restart_cgns(
            template_cgns=template_cgns,
            output_cgns=restart_path,
            predicted_blocks=prediction.pred_physical,
            batch=batch,
            field_floors=field_floors,
            metadata={
                "checkpoint": str(args.checkpoint.expanduser().resolve()),
                "case_index": prediction.case_index,
                "sample_id": sample_id,
                "cond_idx": cond_idx,
                "restart_projection": "positivity" if field_floors else "raw",
            },
        )
        artifacts["restart_cgns"] = str(restart_path)
        artifacts["restart_projection"] = "positivity" if field_floors else "raw"
        artifacts["template_cgns"] = str(Path(template_cgns).expanduser().resolve())

    if args.write_target_cgns_restart:
        target_restart_path = output_dir / "nk_restart" / case_stem / "target_restart_vol.cgns"
        write_prediction_restart_cgns(
            template_cgns=template_cgns,
            output_cgns=target_restart_path,
            predicted_blocks=prediction.target_physical,
            batch=batch,
            field_floors=None,
            metadata={
                "checkpoint": str(args.checkpoint.expanduser().resolve()),
                "case_index": prediction.case_index,
                "sample_id": sample_id,
                "cond_idx": cond_idx,
                "restart_projection": "target",
            },
        )
        artifacts["target_restart_cgns"] = str(target_restart_path)
        artifacts["target_template_cgns"] = str(Path(template_cgns).expanduser().resolve())

    if args.run_force_eval:
        if restart_path is None:
            field_floors = DEFAULT_RESTART_FIELD_FLOORS if args.restart_positivity_projection else None
            restart_suffix = "surrogate_restart_projected_vol.cgns" if field_floors else "surrogate_restart_vol.cgns"
            restart_path = output_dir / "nk_restart" / case_stem / restart_suffix
            write_prediction_restart_cgns(
                template_cgns=template_cgns,
                output_cgns=restart_path,
                predicted_blocks=prediction.pred_physical,
                batch=batch,
                field_floors=field_floors,
                metadata={
                    "checkpoint": str(args.checkpoint.expanduser().resolve()),
                    "case_index": prediction.case_index,
                    "sample_id": sample_id,
                    "cond_idx": cond_idx,
                    "restart_projection": "positivity" if field_floors else "raw",
                },
            )
            artifacts["restart_cgns"] = str(restart_path)
            artifacts["restart_projection"] = "positivity" if field_floors else "raw"
            artifacts["template_cgns"] = str(Path(template_cgns).expanduser().resolve())

        force_config = ADflowForceEvalConfig(
            mpi_ranks=int(args.force_eval_ranks),
            python_executable=(
                str(args.force_python.expanduser().resolve())
                if args.force_python is not None
                else ADflowForceEvalConfig().python_executable
            ),
            cycle_protocol=str(args.force_cycle_protocol),
            cycles=int(args.force_cycles),
            inner_cycles=int(args.force_inner_cycles),
            solver_calls=args.force_solver_calls,
            solver_preset=args.force_solver_preset,
            l2_tol=float(args.force_l2_tol),
            timeout_s=int(args.force_timeout_s),
            write_volume_solution=bool(args.force_write_volume_solution),
            write_surface_solution=bool(args.force_write_surface_solution),
        )
        force_run_dir = output_dir / "force_eval" / f"{case_stem}_{force_config.run_suffix()}"
        force_summary = run_restart_force_evaluation(
            restart_cgns=restart_path,
            source_case_dir=source_case_dir,
            output_dir=force_run_dir,
            summary_path=force_run_dir / f"{case_stem}_force_summary.json",
            config=force_config,
            metadata={
                "checkpoint": str(args.checkpoint.expanduser().resolve()),
                "case_index": prediction.case_index,
                "sample_id": sample_id,
                "cond_idx": cond_idx,
                "restart_projection": artifacts.get("restart_projection"),
            },
        )
        artifacts["force_eval_summary_json"] = str(force_summary["summary_path"])
        artifacts["force_eval_method"] = str(force_summary["force_eval_method"])
        if "force_l1_cl_cd" in force_summary:
            artifacts["force_eval_force_l1_cl_cd"] = float(force_summary["force_l1_cl_cd"])
        if force_summary.get("force_eval_candidate_key") in force_summary:
            candidate_key = str(force_summary["force_eval_candidate_key"])
            artifacts["force_eval_candidate_key"] = candidate_key
            artifacts["force_eval_candidate"] = force_summary[candidate_key]

    summary_path = output_dir / f"{case_stem}_summary.json"
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(_to_jsonable(artifacts), stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(_to_jsonable({"summary": str(summary_path), **artifacts}), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
