from pathlib import Path

import torch
from torch.utils.data import DataLoader

from surrogate3d.data import (
    WholeCaseBlockDataset,
    create_normalizer_from_metadata,
    write_turbulent_scale_stats,
    whole_case_collate,
)
from surrogate3d.models import HierarchicalGlobalFNO3D
from surrogate3d.training.config import WholeCaseDirectTrainerConfig
from surrogate3d.training.run import build_pirdm_training_config
from surrogate3d.training.trainer import WholeCaseDirectTrainer


def _tiny_model(dataset: WholeCaseBlockDataset) -> HierarchicalGlobalFNO3D:
    return HierarchicalGlobalFNO3D(
        block_catalog=dataset.get_block_catalog(),
        geometry_dim=dataset.geometry_dim,
        width=4,
        condition_dim=8,
        n_layers=1,
        global_layers=1,
        post_global_layers=1,
        modes=(1, 1, 1),
        patch_size=(4, 2, 4),
        coord_channels=6,
    )


def test_whole_case_dataset_collate(tiny_dataset_index: Path):
    dataset = WholeCaseBlockDataset(tiny_dataset_index)
    sample = dataset[0]
    assert len(sample["blocks"]) == 2
    loader = DataLoader(dataset, batch_size=1, collate_fn=whole_case_collate)
    batch = next(iter(loader))
    assert batch["geometry"].shape == (1, 4)
    assert len(batch["blocks"]) == 2
    first_block = batch["blocks"][0]
    assert first_block["state"].shape == (1, 6, 8, 4, 8)
    assert first_block["coords"].shape == (1, 6, 8, 4, 8)
    assert torch.allclose(first_block["coords"][:, :3], first_block["coords_pde"])


def test_metadata_normalizer_roundtrip(tiny_dataset_index: Path):
    write_turbulent_scale_stats(tiny_dataset_index, turbulent_channel_idx=5)
    normalizer = create_normalizer_from_metadata(tiny_dataset_index.parent / "metadata.json")
    x = torch.rand(2, 6, 8, 4, 8)
    restored = normalizer.inverse_transform(normalizer.transform(x))
    assert torch.allclose(restored, x, atol=1.0e-6)


def test_global_fno_forward_shapes(tiny_dataset_index: Path):
    dataset = WholeCaseBlockDataset(tiny_dataset_index)
    batch = whole_case_collate([dataset[0]])
    model = _tiny_model(dataset)
    pred = model(batch["geometry"], batch["flow_conditions"], batch["blocks"])
    for block_index, block in batch["blocks"].items():
        assert pred[block_index].shape == block["target"].shape


def test_cpu_fixture_runs_one_optimizer_step(tiny_dataset_index: Path):
    dataset = WholeCaseBlockDataset(tiny_dataset_index)
    batch = whole_case_collate([dataset[0]])
    model = _tiny_model(dataset)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
    trainer = WholeCaseDirectTrainer(
        model,
        optimizer,
        config=WholeCaseDirectTrainerConfig(),
        device=torch.device("cpu"),
    )
    before = model.stems["0"].weight.detach().clone()
    result = trainer.train_step(batch)
    after = model.stems["0"].weight.detach()
    assert torch.isfinite(result["loss"])
    assert not torch.equal(before, after)


def test_pirdm_training_config_scheduler_controls():
    config = {
        "training": {
            "epochs": 3,
            "optimizer": {"name": "adamw", "lr": 2.0e-4, "weight_decay": 0.0},
            "warmup_ratio": 0.1,
            "eta_min_ratio": 0.05,
            "gradient_clip_norm": 1.0,
            "loss": {"use_volume_weighted_mse": False},
        }
    }
    training_config = build_pirdm_training_config(config)
    assert training_config.optimizer["name"] == "adamw"
    assert training_config.optimizer["lr"] == 2.0e-4
    assert training_config.warmup_ratio == 0.1
    assert training_config.eta_min_ratio == 0.05
