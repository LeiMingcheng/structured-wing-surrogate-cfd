"""Export completed wing CFD cases as structured HDF5 block shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from parametric_swept_wing_cfd.export import build_structured_block_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=16)
    parser.add_argument("--limit-cases", type=int, default=0)
    parser.add_argument("--compression", choices=("gzip", "lzf", "none"), default="lzf")
    parser.add_argument("--keep-case-exports", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    compression = None if args.compression == "none" else args.compression
    metadata = build_structured_block_dataset(
        dataset_root=args.dataset_root.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        shard_size=args.shard_size,
        limit_cases=args.limit_cases,
        compression=compression,
        keep_exports=args.keep_case_exports,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
