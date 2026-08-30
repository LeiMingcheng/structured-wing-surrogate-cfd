import json
from pathlib import Path

import h5py
import pandas as pd
import pytest
import torch

from surrogate3d.data.case_dataset import WholeCaseBlockDataset, whole_case_collate
from surrogate3d.training.volume_guard import require_dataset_cell_volumes, require_volume_blocks


def _write_min_dataset(root: Path, *, include_cell_volumes: bool) -> Path:
    shard_path = root / "block0_0000.h5"
    coords = torch.tensor(
        [[[[[0.5]], [[1.5]]], [[[0.5]], [[0.5]]], [[[0.5]], [[0.5]]]]],
        dtype=torch.float32,
    )
    with h5py.File(shard_path, "w") as h5_file:
        h5_file.create_dataset(
            "state",
            data=torch.arange(12, dtype=torch.float32).reshape(1, 6, 2, 1, 1).numpy(),
        )
        h5_file.create_dataset("coords", data=coords.numpy())
        if include_cell_volumes:
            h5_file.create_dataset(
                "cell_volumes",
                data=torch.tensor([[[[1.0]], [[2.0]]]], dtype=torch.float32).numpy(),
            )
        h5_file.create_dataset(
            "flow",
            data=torch.tensor([[0.78, 2.5, 3.0e6]], dtype=torch.float32).numpy(),
        )
        h5_file.create_dataset(
            "forces",
            data=torch.tensor([[0.3, 0.02, -0.1]], dtype=torch.float32).numpy(),
        )
        h5_file.create_dataset(
            "geometry",
            data=torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32).numpy(),
        )

    metadata = {
        "block_catalog": [
            {
                "block_index": 0,
                "block_name": "blk0",
                "block_signature": "blk0_sig",
                "cell_shape_ijk": [2, 1, 1],
                "n_records": 1,
            }
        ],
        "cell_volumes": include_cell_volumes,
    }
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    pd.DataFrame(
        [
            {
                "sample_id": "sample0",
                "cond_idx": 0,
                "family": "fam",
                "band": "band",
                "cgns_basename": "wing_000_vol.cgns",
                "source_case_dir": "/tmp/case0",
                "tier3_variant": 0,
                "mach": 0.78,
                "aoa": 2.5,
                "reynolds": 3.0e6,
                "block_index": 0,
                "block_name": "blk0",
                "block_signature": "blk0_sig",
                "shard_path": shard_path.name,
                "local_index": 0,
                "ni": 2,
                "nj": 1,
                "nk": 1,
            }
        ]
    ).to_csv(root / "index.csv", index=False)
    return root / "index.csv"


def test_require_dataset_cell_volumes_rejects_legacy_metadata(tmp_path):
    index_path = _write_min_dataset(tmp_path, include_cell_volumes=False)
    with pytest.raises(RuntimeError, match="cell_volumes: true"):
        require_dataset_cell_volumes(index_path, use_volume_weighted_mse=True)


def test_require_dataset_cell_volumes_accepts_volume_metadata(tmp_path):
    index_path = _write_min_dataset(tmp_path, include_cell_volumes=True)
    require_dataset_cell_volumes(index_path, use_volume_weighted_mse=True)


def test_trainer_raises_when_volume_weighted_loss_has_no_volume_batch(tmp_path):
    index_path = _write_min_dataset(tmp_path, include_cell_volumes=False)
    dataset = WholeCaseBlockDataset(index_path)
    batch = whole_case_collate([dataset[0]])
    dataset.close()

    with pytest.raises(RuntimeError, match="requires per-block cell volumes"):
        require_volume_blocks(batch)
