"""Train whole-case 3D direct surrogate models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from surrogate3d.training.run import apply_overrides, load_training_config, train_from_config


def _parse_tuple3(value: str) -> list[int]:
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected comma-separated triplet, e.g. 8,4,8")
    return [int(part) for part in parts]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="YAML config file")
    parser.add_argument("--index", type=Path, default=None, help="Structured block dataset index.csv")
    parser.add_argument("--output-dir", type=Path, default=None, help="Run/checkpoint output directory")
    parser.add_argument("--device", default=None, help="Torch device, e.g. cuda, cuda:0, cpu")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--max-steps", type=int, default=None, help="Stop after this many optimizer steps")
    parser.add_argument("--batch-size", type=int, default=None, help="Whole-case batch size")
    parser.add_argument("--num-cases", type=int, default=None, help="Optional cap on whole-case samples")
    parser.add_argument("--val-fraction", type=float, default=None, help="Validation fraction")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader worker count")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--width", type=int, default=None, help="Model width")
    parser.add_argument("--condition-dim", type=int, default=None, help="Condition embedding dimension")
    parser.add_argument("--n-layers", type=int, default=None, help="Local spectral layer count")
    parser.add_argument("--global-layers", type=int, default=None, help="Global block-token transformer layer count")
    parser.add_argument("--patch-size", type=_parse_tuple3, default=None, help="Patch size as pi,pj,pk")
    parser.add_argument("--modes", type=_parse_tuple3, default=None, help="FNO modes as mi,mj,mk")
    parser.add_argument("--lr", type=float, default=None, help="AdamW learning rate")
    parser.add_argument("--weight-decay", type=float, default=None, help="AdamW weight decay")
    parser.add_argument("--log-every", type=int, default=None, help="Log every N steps")
    parser.add_argument("--save-every", type=int, default=None, help="Save every N steps; 0 disables")
    parser.add_argument("--no-normalize", action="store_true", help="Disable state normalization")
    parser.add_argument("--scale-turbulent", action="store_true", help="Use PIR-DM log-SA turbulent scaling")
    parser.add_argument("--turbulent-stats", type=Path, default=None, help="turbulent_scale_stats.json")
    return parser


def cli_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {
        "data.index_path": str(args.index) if args.index is not None else None,
        "data.batch_size": args.batch_size,
        "data.num_cases": args.num_cases,
        "data.val_fraction": args.val_fraction,
        "data.num_workers": args.num_workers,
        "runtime.device": args.device,
        "training.epochs": args.epochs,
        "training.max_steps": args.max_steps,
        "experiment.seed": args.seed,
        "training.log_every": args.log_every,
        "model.width": args.width,
        "model.condition_dim": args.condition_dim,
        "model.n_layers": args.n_layers,
        "model.global_layers": args.global_layers,
        "model.patch_size": args.patch_size,
        "model.modes": args.modes,
        "training.optimizer.lr": args.lr,
        "training.optimizer.weight_decay": args.weight_decay,
        "training.output_dir": str(args.output_dir) if args.output_dir is not None else None,
        "training.save_every_steps": args.save_every,
        "normalization.scale_turbulent": True if args.scale_turbulent else None,
        "normalization.turbulent_stats_path": str(args.turbulent_stats)
        if args.turbulent_stats is not None
        else None,
    }
    if args.no_normalize:
        overrides["normalization.normalize"] = False
    return overrides


def main() -> int:
    args = build_parser().parse_args()
    config = load_training_config(args.config)
    config = apply_overrides(config, cli_overrides(args))
    summary = train_from_config(config)
    if bool(summary.get("_is_main_process", True)):
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
