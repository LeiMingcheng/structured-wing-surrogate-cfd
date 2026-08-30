import json

import h5py
import pandas as pd
import torch

from surrogate3d.data.case_dataset import WholeCaseBlockDataset, whole_case_collate


def _write_min_dataset(root, *, include_cell_volumes: bool):
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
        "geometry_dim": 4,
        "geometry_feature_names": ["a", "b", "c", "d"],
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


def test_whole_case_dataset_reads_cell_volumes_from_shard(tmp_path):
    index_path = _write_min_dataset(tmp_path, include_cell_volumes=True)
    dataset = WholeCaseBlockDataset(index_path)
    sample = dataset[0]
    batch = whole_case_collate([sample])
    dataset.close()

    assert "cell_volumes" in sample["blocks"][0]
    assert batch["blocks"][0]["cell_volumes"].shape == (1, 2, 1, 1)
    assert torch.allclose(
        batch["blocks"][0]["cell_volumes"],
        torch.tensor([[[[1.0]], [[2.0]]]], dtype=torch.float32),
    )


def test_whole_case_dataset_supports_legacy_shards_without_cell_volumes(tmp_path):
    index_path = _write_min_dataset(tmp_path, include_cell_volumes=False)
    dataset = WholeCaseBlockDataset(index_path)
    sample = dataset[0]
    batch = whole_case_collate([sample])
    dataset.close()

    assert "cell_volumes" not in sample["blocks"][0]
    assert "cell_volumes" not in batch["blocks"][0]


def test_whole_case_dataset_can_match_legacy_double_aligned_geometry(tmp_path):
    index_path = _write_min_dataset(tmp_path, include_cell_volumes=True)
    dataset = WholeCaseBlockDataset(
        index_path,
        legacy_double_alignment_feature_names_by_family={"fam": ["a", "c", "d"]},
    )
    geometry = dataset[0]["geometry"]
    dataset.close()

    assert torch.equal(geometry, torch.tensor([1.0, 0.0, 2.0, 3.0]))
