"""Run surrogate3d inference followed by 3D NK recovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from surrogate3d.evaluation.inference import predict_case
from surrogate3d.evaluation.nk_recovery import NKRecoveryConfig, run_nk_recovery
from surrogate3d.nk_resume.cgns_restart import write_prediction_restart_cgns
from surrogate3d.training.run import build_normalizer_from_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--case-index", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mpi-ranks", type=int, default=8)
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--inner-cycles", type=int, default=1)
    parser.add_argument("--l2-tol", type=float, default=1.0e-8)
    parser.add_argument("--timeout-s", type=int, default=7200)
    return parser


def main() -> int:
    args = _parser().parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction = predict_case(
        args.checkpoint,
        case_index=int(args.case_index),
        device=args.device,
    )
    source_case_dir = Path(str(prediction.batch["source_case_dir"][0])).expanduser().resolve()
    basename = str(prediction.batch["cgns_basename"][0]).strip() or "wing_000_vol.cgns"
    restart_path = output_dir / "surrogate_restart_vol.cgns"
    write_prediction_restart_cgns(
        template_cgns=source_case_dir / basename,
        output_cgns=restart_path,
        predicted_blocks=prediction.pred_physical,
        batch=prediction.batch,
        metadata={
            "checkpoint": str(args.checkpoint.expanduser().resolve()),
            "case_index": int(args.case_index),
            "sample_id": str(prediction.batch["sample_id"][0]),
            "cond_idx": int(prediction.batch["cond_idx"][0].detach().cpu().item()),
        },
    )
    normalizer = build_normalizer_from_config(prediction.config)
    summary = run_nk_recovery(
        restart_cgns=restart_path,
        source_case_dir=source_case_dir,
        output_dir=output_dir,
        normalizer=normalizer,
        config=NKRecoveryConfig(
            mpi_ranks=int(args.mpi_ranks),
            cycles=int(args.cycles),
            inner_cycles=int(args.inner_cycles),
            l2_tol=float(args.l2_tol),
            timeout_s=int(args.timeout_s),
        ),
        metadata={
            "checkpoint": str(args.checkpoint.expanduser().resolve()),
            "checkpoint_epoch": int(prediction.checkpoint_payload.get("epoch", -1)),
            "checkpoint_global_step": int(prediction.checkpoint_payload.get("global_step", -1)),
            "case_index": int(args.case_index),
            "sample_id": str(prediction.batch["sample_id"][0]),
            "cond_idx": int(prediction.batch["cond_idx"][0].detach().cpu().item()),
            "family": str(prediction.batch["family"][0]),
        },
    )
    normalized = summary["field_metrics"]["normalized"]
    print(
        json.dumps(
            {
                "summary_path": summary["summary_path"],
                "elapsed_s": summary["elapsed_s"],
                "normalized_initial": normalized["initial_vs_reference"]["summary"],
                "normalized_recovered": normalized["candidate_vs_reference"]["summary"],
                "normalized_improvement": normalized["improvement"],
                "force_torch": summary["force_torch"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
