"""Checkpoint loading and whole-case prediction helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Optional

import torch

from surrogate.common.checkpointing import load_model_checkpoint

from surrogate3d.data import WholeCaseBlockDataset, whole_case_collate
from surrogate3d.training.losses import inverse_transform_blocks, transform_target_blocks
from surrogate3d.training.run import build_model_from_config, build_normalizer_from_config


@dataclass
class WholeCasePrediction:
    """One whole-case prediction bundle in model and physical state space."""

    batch: dict[str, Any]
    pred_model: dict[int, torch.Tensor]
    pred_physical: dict[int, torch.Tensor]
    target_model: dict[int, torch.Tensor]
    target_physical: dict[int, torch.Tensor]
    config: dict[str, Any]
    checkpoint_payload: dict[str, Any]
    case_index: int


def _load_checkpoint_payload(path: str | Path, device: torch.device) -> dict[str, Any]:
    return torch.load(Path(path).expanduser().resolve(), map_location=device, weights_only=False)


def select_case_index(
    dataset: WholeCaseBlockDataset,
    *,
    requested_index: Optional[int] = None,
    holdout_start: Optional[int] = None,
) -> int:
    """Select a dataset case, defaulting to the first holdout after train cap."""
    if requested_index is not None:
        idx = int(requested_index)
    elif holdout_start is not None:
        idx = int(holdout_start)
    else:
        idx = 0
    if idx < 0 or idx >= len(dataset):
        raise IndexError(f"case_index={idx} is outside dataset length {len(dataset)}")
    return idx


def load_checkpoint_for_inference(
    checkpoint_path: str | Path,
    *,
    index_path: str | Path | None = None,
    device: str | torch.device = "cuda",
    match_legacy_double_aligned_geometry: bool = False,
) -> tuple[torch.nn.Module, Optional[torch.nn.Module], WholeCaseBlockDataset, dict[str, Any], dict[str, Any]]:
    """Load a surrogate3d checkpoint and construct its dataset/model pair."""
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    payload = _load_checkpoint_payload(checkpoint_path, device)
    config = dict(payload.get("config") or {})
    training_data_cfg = dict(config.get("data") or {})
    dataset_schema = dict(payload.get("dataset_schema") or {})
    if index_path is None and not training_data_cfg.get("index_path"):
        raise KeyError("checkpoint config does not include data.index_path; provide --index")
    selected_index = Path(index_path or training_data_cfg["index_path"]).expanduser().resolve()
    selected_metadata_path = selected_index.parent / "metadata.json"
    with selected_metadata_path.open("r", encoding="utf-8") as stream:
        selected_metadata = json.load(stream)
    geometry_feature_names = [
        str(name)
        for name in (
            dataset_schema.get("geometry_feature_names")
            or selected_metadata["geometry_feature_names"]
        )
    ]
    legacy_feature_names_by_family = None
    if match_legacy_double_aligned_geometry:
        training_index = training_data_cfg.get("index_path")
        if training_index is None:
            raise ValueError("legacy geometry alignment requires the original training index path")
        training_metadata_path = Path(training_index).expanduser().resolve().parent / "metadata.json"
        with training_metadata_path.open("r", encoding="utf-8") as stream:
            training_metadata = json.load(stream)
        subsets_root = training_metadata_path.parent / "_subsets"
        legacy_feature_names_by_family = {}
        for family in training_metadata["families"]:
            subset_metadata_path = subsets_root / str(family) / "metadata.json"
            with subset_metadata_path.open("r", encoding="utf-8") as stream:
                subset_metadata = json.load(stream)
            legacy_feature_names_by_family[str(family)] = [
                str(name) for name in subset_metadata["geometry_feature_names"]
            ]

    if index_path is not None:
        config.setdefault("data", {})
        config["data"]["index_path"] = str(selected_index)
    data_cfg = dict(config.get("data") or {})

    dataset = WholeCaseBlockDataset(
        data_cfg["index_path"],
        num_cases=None,
        cache_size_mb=int(data_cfg.get("cache_size_mb", 256)),
        geometry_feature_names=geometry_feature_names,
        legacy_double_alignment_feature_names_by_family=legacy_feature_names_by_family,
    )
    block_catalog = payload.get("block_catalog") or dataset.get_block_catalog()
    model = build_model_from_config(
        config,
        block_catalog=list(block_catalog),
        dataset_geometry_dim=dataset.geometry_dim,
    ).to(device)
    load_model_checkpoint(
        model,
        checkpoint_path,
        device,
        use_ema=False,
        context=f"surrogate3d inference {checkpoint_path}",
    )
    model.eval()
    normalizer = build_normalizer_from_config(config)
    if normalizer is not None and hasattr(normalizer, "to"):
        normalizer = normalizer.to(device)
    return model, normalizer, dataset, config, payload


@torch.no_grad()
def predict_loaded_case(
    model: torch.nn.Module,
    normalizer: Optional[torch.nn.Module],
    dataset: WholeCaseBlockDataset,
    config: dict[str, Any],
    payload: dict[str, Any],
    *,
    case_index: int,
) -> WholeCasePrediction:
    """Predict one dataset case using an already-loaded model."""
    device = next(model.parameters()).device
    selected = select_case_index(dataset, requested_index=case_index)
    batch = whole_case_collate([dataset[selected]])

    moved = {}
    for key, value in batch.items():
        if key == "blocks":
            moved_blocks = {}
            for block_index, block in value.items():
                moved_blocks[int(block_index)] = {
                    inner_key: inner_value.to(device) if isinstance(inner_value, torch.Tensor) else inner_value
                    for inner_key, inner_value in block.items()
                }
            moved[key] = moved_blocks
        elif isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value

    pred_model = model(
        moved["geometry"],
        moved["flow_conditions"],
        moved["blocks"],
        initial_field=moved.get("initial_field"),
    )
    target_model = transform_target_blocks(moved, normalizer)
    pred_physical = inverse_transform_blocks(pred_model, normalizer)
    target_physical = inverse_transform_blocks(target_model, normalizer)

    return WholeCasePrediction(
        batch=moved,
        pred_model={int(k): v.detach() for k, v in pred_model.items()},
        pred_physical={int(k): v.detach() for k, v in pred_physical.items()},
        target_model={int(k): v.detach() for k, v in target_model.items()},
        target_physical={int(k): v.detach() for k, v in target_physical.items()},
        config=config,
        checkpoint_payload=payload,
        case_index=selected,
    )


@torch.no_grad()
def predict_case(
    checkpoint_path: str | Path,
    *,
    index_path: str | Path | None = None,
    case_index: Optional[int] = None,
    holdout_start: Optional[int] = None,
    device: str | torch.device = "cuda",
) -> WholeCasePrediction:
    """Run one whole-case prediction from a saved checkpoint."""
    model, normalizer, dataset, config, payload = load_checkpoint_for_inference(
        checkpoint_path,
        index_path=index_path,
        device=device,
    )
    selected = select_case_index(dataset, requested_index=case_index, holdout_start=holdout_start)
    return predict_loaded_case(
        model,
        normalizer,
        dataset,
        config,
        payload,
        case_index=selected,
    )


def prediction_to_cpu(prediction: WholeCasePrediction) -> WholeCasePrediction:
    """Move tensors in a prediction bundle to CPU for plotting/serialization."""
    def _move_mapping(blocks: Mapping[int, torch.Tensor]) -> dict[int, torch.Tensor]:
        return {int(k): v.detach().cpu() for k, v in blocks.items()}

    batch = {}
    for key, value in prediction.batch.items():
        if key == "blocks":
            batch["blocks"] = {
                int(block_index): {
                    inner_key: inner_value.detach().cpu() if isinstance(inner_value, torch.Tensor) else inner_value
                    for inner_key, inner_value in block.items()
                }
                for block_index, block in value.items()
            }
        elif isinstance(value, torch.Tensor):
            batch[key] = value.detach().cpu()
        else:
            batch[key] = value
    return WholeCasePrediction(
        batch=batch,
        pred_model=_move_mapping(prediction.pred_model),
        pred_physical=_move_mapping(prediction.pred_physical),
        target_model=_move_mapping(prediction.target_model),
        target_physical=_move_mapping(prediction.target_physical),
        config=prediction.config,
        checkpoint_payload=prediction.checkpoint_payload,
        case_index=prediction.case_index,
    )


__all__ = [
    "WholeCasePrediction",
    "load_checkpoint_for_inference",
    "predict_case",
    "predict_loaded_case",
    "prediction_to_cpu",
    "select_case_index",
]
