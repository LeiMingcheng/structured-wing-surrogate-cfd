"""Create a deterministic multi-case structured dataset for DDP smoke tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


STATE_FIELDS = (
    "Density",
    "VelocityX",
    "VelocityY",
    "VelocityZ",
    "Pressure",
    "TurbulentSANuTilde",
)
BLOCK_SHAPES = ((8, 4, 8), (8, 4, 8))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-cases", type=int, default=4)
    return parser


def _coordinates(shape: tuple[int, int, int], block_index: int) -> np.ndarray:
    i, j, k = np.meshgrid(
        np.linspace(0.0, 1.0, shape[0], dtype=np.float32),
        np.linspace(0.0, 1.0, shape[1], dtype=np.float32),
        np.linspace(0.0, 1.0, shape[2], dtype=np.float32),
        indexing="ij",
    )
    return np.stack((i, j + float(block_index), k), axis=0)


def create_dataset(output_dir: Path, n_cases: int) -> Path:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    rows: list[dict[str, object]] = []

    for block_index, shape in enumerate(BLOCK_SHAPES):
        shard_name = f"block_{block_index}.h5"
        state = rng.normal(size=(n_cases, len(STATE_FIELDS), *shape)).astype(np.float32)
        state[:, 0] += 2.0
        state[:, 4] += 2.0
        state[:, 5] = np.abs(state[:, 5]) * 1.0e-3
        coords = np.repeat(_coordinates(shape, block_index)[None], n_cases, axis=0)
        volumes = np.ones((n_cases, *shape), dtype=np.float32)
        flow = np.asarray(
            [[0.76 + 0.01 * case_index, 1.0 + case_index, 2.0e7] for case_index in range(n_cases)],
            dtype=np.float32,
        )
        forces = np.asarray(
            [[0.2 + 0.01 * case_index, 0.01, -0.03] for case_index in range(n_cases)],
            dtype=np.float32,
        )
        geometry = np.asarray(
            [[1.0, 2.0, 3.0, float(case_index)] for case_index in range(n_cases)],
            dtype=np.float32,
        )
        with h5py.File(output_dir / shard_name, "w") as h5_file:
            h5_file.create_dataset("state", data=state)
            h5_file.create_dataset("coords", data=coords)
            h5_file.create_dataset("cell_volumes", data=volumes)
            h5_file.create_dataset("flow", data=flow)
            h5_file.create_dataset("forces", data=forces)
            h5_file.create_dataset("geometry", data=geometry)

        for case_index in range(n_cases):
            rows.append(
                {
                    "sample_id": f"synthetic-wing-{case_index:02d}",
                    "cond_idx": 0,
                    "family": "synthetic",
                    "band": "ddp-smoke",
                    "cgns_basename": "wing_vol.cgns",
                    "source_case_dir": str(output_dir / "external-case-placeholder"),
                    "mach": float(flow[case_index, 0]),
                    "aoa": float(flow[case_index, 1]),
                    "reynolds": float(flow[case_index, 2]),
                    "block_index": block_index,
                    "block_name": f"block{block_index}",
                    "block_signature": f"block{block_index}_8x4x8",
                    "shard_path": shard_name,
                    "local_index": case_index,
                    "ni": shape[0],
                    "nj": shape[1],
                    "nk": shape[2],
                }
            )

    metadata = {
        "version": "structured-wing-ddp-smoke",
        "geometry_schema": "synthetic",
        "geometry_dim": 4,
        "geometry_feature_names": ["g0", "g1", "g2", "case_offset"],
        "state_fields": list(STATE_FIELDS),
        "flow_names": ["mach", "aoa", "reynolds"],
        "force_names": ["wing_cl", "wing_cd", "wing_cmz"],
        "cell_volumes": True,
        "norm": {
            "state_mean": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0e-3],
            "state_std": [1.0] * len(STATE_FIELDS),
            "flow_mean": [0.0] * 3,
            "flow_std": [1.0] * 3,
            "force_mean": [0.0] * 3,
            "force_std": [1.0] * 3,
        },
        "block_catalog": [
            {
                "block_index": block_index,
                "block_name": f"block{block_index}",
                "block_signature": f"block{block_index}_8x4x8",
                "cell_shape_ijk": list(shape),
                "n_records": n_cases,
            }
            for block_index, shape in enumerate(BLOCK_SHAPES)
        ],
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    index_path = output_dir / "index.csv"
    pd.DataFrame(rows).to_csv(index_path, index=False)
    return index_path


def main() -> int:
    args = build_parser().parse_args()
    index_path = create_dataset(args.output_dir, args.n_cases)
    print(index_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
