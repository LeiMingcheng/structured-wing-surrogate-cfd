"""Audit tracked release files for private paths and oversized artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


PRIVATE_MARKERS = (
    b"/root/" + b"shared-nvme",
    b"/" + b"home/",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-bytes", type=int, default=10 * 1024 * 1024)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = Path(__file__).resolve().parents[1]
    output = subprocess.check_output(
        ["git", "ls-files", "-z"],
        cwd=root,
    )
    paths = [root / raw.decode("utf-8") for raw in output.split(b"\0") if raw]
    failures: list[str] = []
    for path in paths:
        if path.stat().st_size > args.max_bytes:
            failures.append(f"oversized: {path.relative_to(root)} ({path.stat().st_size} bytes)")
        data = path.read_bytes()
        for marker in PRIVATE_MARKERS:
            if marker in data:
                failures.append(f"private path marker {marker!r}: {path.relative_to(root)}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"audited {len(paths)} tracked files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
