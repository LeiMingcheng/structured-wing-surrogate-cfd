from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tiny_dataset_index(tmp_path: Path) -> Path:
    block_shapes = ((8, 4, 8), (8, 4, 8))
    rows = []
    rng = np.random.default_rng(7)
    for block_index, shape in enumerate(block_shapes):
        shard_name = f"block_{block_index}.h5"
        shard_path = tmp_path / shard_name
        coords = np.stack(
            np.meshgrid(
                np.linspace(0.0, 1.0, shape[0], dtype=np.float32),
                np.linspace(0.0, 1.0, shape[1], dtype=np.float32),
                np.linspace(0.0, 1.0, shape[2], dtype=np.float32),
                indexing="ij",
            ),
            axis=0,
        )
        state = rng.normal(size=(1, 6, *shape)).astype(np.float32)
        state[:, 0] += 2.0
        state[:, 4] += 2.0
        state[:, 5] = np.abs(state[:, 5]) * 1.0e-3
        with h5py.File(shard_path, "w") as h5_file:
            h5_file.create_dataset("state", data=state)
            h5_file.create_dataset("coords", data=coords[None])
            h5_file.create_dataset("cell_volumes", data=np.ones((1, *shape), dtype=np.float32))
            h5_file.create_dataset("flow", data=np.asarray([[0.78, 2.0, 2.0e7]], dtype=np.float32))
            h5_file.create_dataset("forces", data=np.asarray([[0.3, 0.02, -0.1]], dtype=np.float32))
            h5_file.create_dataset("geometry", data=np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32))
        rows.append(
            {
                "sample_id": "synthetic-wing",
                "cond_idx": 0,
                "family": "synthetic",
                "band": "ci",
                "cgns_basename": "wing_vol.cgns",
                "source_case_dir": str(tmp_path / "case"),
                "mach": 0.78,
                "aoa": 2.0,
                "reynolds": 2.0e7,
                "block_index": block_index,
                "block_name": f"block{block_index}",
                "block_signature": f"block{block_index}_8x4x8",
                "shard_path": shard_name,
                "local_index": 0,
                "ni": shape[0],
                "nj": shape[1],
                "nk": shape[2],
            }
        )

    metadata = {
        "version": "structured-wing-ci-fixture",
        "geometry_dim": 4,
        "geometry_feature_names": ["g0", "g1", "g2", "g3"],
        "cell_volumes": True,
        "state_fields": [
            "Density",
            "VelocityX",
            "VelocityY",
            "VelocityZ",
            "Pressure",
            "TurbulentSANuTilde",
        ],
        "norm": {
            "state_mean": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0e-3],
            "state_std": [1.0] * 6,
            "flow_mean": [0.0] * 3,
            "flow_std": [1.0] * 3,
            "force_mean": [0.0] * 3,
            "force_std": [1.0] * 3,
        },
        "block_catalog": [
            {
                "block_index": index,
                "block_name": f"block{index}",
                "block_signature": f"block{index}_8x4x8",
                "cell_shape_ijk": list(shape),
                "n_records": 1,
            }
            for index, shape in enumerate(block_shapes)
        ],
    }
    (tmp_path / "metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    index_path = tmp_path / "index.csv"
    pd.DataFrame(rows).to_csv(index_path, index=False)
    return index_path
