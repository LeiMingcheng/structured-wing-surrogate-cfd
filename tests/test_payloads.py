from pathlib import Path

import numpy as np

from surrogate3d.data import WholeCaseBlockDataset, whole_case_collate
from surrogate3d.nk_resume import (
    DEFAULT_RESTART_FIELD_FLOORS,
    apply_field_floors,
    load_whole_case_payload,
    write_whole_case_payload,
)


def test_write_and_load_whole_case_payload(tmp_path, tiny_dataset_index: Path):
    dataset = WholeCaseBlockDataset(tiny_dataset_index)
    batch = whole_case_collate([dataset[0]])
    predicted = {block_index: block["target"] for block_index, block in batch["blocks"].items()}
    payload_path = write_whole_case_payload(
        tmp_path / "case_payload.h5",
        predicted_blocks=predicted,
        target_blocks=predicted,
        batch=batch,
        block_catalog=dataset.get_block_catalog(),
        metadata={"test": True},
    )
    payload = load_whole_case_payload(payload_path)
    assert payload["version"] == "surrogate3d-whole-case-payload-v1"
    assert payload["sample_id"] == batch["sample_id"][0]
    assert len(payload["blocks"]) == 2
    assert payload["blocks"][0]["predicted_state"].shape == (6, 8, 4, 8)
    assert payload["blocks"][0]["coords"].shape[0] == 3
    assert payload["blocks"][0]["coords_model"].shape[0] == 6


def test_apply_field_floors_projects_restart_positive_channels():
    state = -np.ones((6, 2, 2, 2), dtype=np.float32)
    projected = apply_field_floors(state, field_floors=DEFAULT_RESTART_FIELD_FLOORS)
    assert np.isclose(projected[0].min(), DEFAULT_RESTART_FIELD_FLOORS["Density"])
    assert np.isclose(projected[4].min(), DEFAULT_RESTART_FIELD_FLOORS["Pressure"])
    assert np.isclose(projected[5].min(), DEFAULT_RESTART_FIELD_FLOORS["TurbulentSANuTilde"])
    assert projected[1].min() == -1.0
    assert state[0].min() == -1.0
