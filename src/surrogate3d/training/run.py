"""Reusable training entry points for whole-case 3D direct surrogates."""

from __future__ import annotations

import copy
from dataclasses import asdict, fields
import json
import os
from pathlib import Path
from types import SimpleNamespace
import random
import time
from typing import Any, Mapping, Optional

import torch
import torch.distributed as dist
import yaml

from surrogate.common.checkpointing import load_model_checkpoint
from surrogate.configs import TrainingConfig, TrainingLossConfig
from surrogate.training.engine import create_lr_scheduler, create_optimizer

from surrogate3d.data import create_normalizer_from_metadata, create_whole_case_loaders
from surrogate3d.models import HierarchicalGlobalFNO3D
from surrogate3d.training.config import whole_case_trainer_config_from_dict
from surrogate3d.training.trainer import WholeCaseDirectTrainer
from surrogate3d.training.volume_guard import metadata_path_from_index, require_dataset_cell_volumes


DEFAULT_CONFIG: dict[str, Any] = {
    "experiment": {
        "name": "global_fno",
        "seed": 0,
    },
    "data": {
        "index_path": None,
        "batch_size": 1,
        "val_fraction": 0.2,
        "num_workers": 0,
        "cache_size_mb": 256,
        "split_by_sample_id": True,
    },
    "normalization": {
        "normalize": True,
        "scale_turbulent": False,
        "turbulent_channel_idx": 5,
    },
    "model": {
        "patch_size": [8, 4, 8],
        "coord_channels": 6,
        "width": 16,
        "modes": [8, 4, 4],
        "n_layers": 2,
        "condition_dim": 64,
        "global_layers": 1,
        "input_state_channels": 0,
        "post_global_layers": 2,
    },
    "training": {
        "epochs": 1,
        "max_steps": None,
        "log_every": 1,
        "checkpoint_every_epochs": None,
        "output_dir": "runs/global_fno",
        "optimizer": {
            "name": "adamw",
            "lr": 1.0e-4,
            "weight_decay": 1.0e-4,
        },
        "warmup_ratio": 0.0,
        "eta_min_ratio": 1.0,
        "gradient_clip_norm": 1.0,
        "per_block_equal_weight": True,
        "loss": {
            "use_volume_weighted_mse": False,
        },
    },
    "runtime": {
        "device": "cuda",
        "checkpoint": None,
    },
}


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _use_ddp() -> bool:
    return _world_size() > 1


def _is_main_process() -> bool:
    return _rank() == 0


def _setup_distributed(device: torch.device) -> torch.device:
    if not _use_ddp():
        return device
    if not dist.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
    if device.type == "cuda":
        local_rank = _local_rank()
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")
    return device


def _cleanup_distributed() -> None:
    if dist.is_initialized():
        if torch.cuda.is_available():
            dist.barrier(device_ids=[_local_rank()])
        else:
            dist.barrier()
        dist.destroy_process_group()


def _ddp_average_scalar(value: float, *, device: torch.device) -> float:
    if not _use_ddp() or not dist.is_initialized():
        return float(value)
    tensor = torch.tensor(float(value), device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= float(dist.get_world_size())
    return float(tensor.item())


def _ddp_average_metrics(metrics: Mapping[str, float], *, device: torch.device) -> dict[str, float]:
    if not _use_ddp() or not dist.is_initialized():
        return {str(key): float(value) for key, value in metrics.items()}
    if not metrics:
        return {}
    keys = sorted(str(key) for key in metrics.keys())
    values = torch.tensor([float(metrics[key]) for key in keys], device=device, dtype=torch.float64)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values /= float(dist.get_world_size())
    return {key: float(value) for key, value in zip(keys, values.tolist())}


def _ddp_sum_count(sum_value: float, count_value: int, *, device: torch.device) -> tuple[float, int]:
    if not _use_ddp() or not dist.is_initialized():
        return float(sum_value), int(count_value)
    tensor = torch.tensor([float(sum_value), float(count_value)], device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor[0].item()), int(round(float(tensor[1].item())))


def _ddp_reduce_metric_sums_counts(
    metric_sums: Mapping[str, float],
    metric_counts: Mapping[str, int],
    *,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, int]]:
    keys = sorted(set(metric_sums.keys()) | set(metric_counts.keys()))
    if not _use_ddp() or not dist.is_initialized():
        return (
            {str(key): float(metric_sums.get(key, 0.0)) for key in keys},
            {str(key): int(metric_counts.get(key, 0)) for key in keys},
        )
    if not keys:
        return {}, {}
    sum_tensor = torch.tensor([float(metric_sums.get(key, 0.0)) for key in keys], device=device, dtype=torch.float64)
    count_tensor = torch.tensor([float(metric_counts.get(key, 0)) for key in keys], device=device, dtype=torch.float64)
    dist.all_reduce(sum_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
    reduced_sums = {key: float(value) for key, value in zip(keys, sum_tensor.tolist())}
    reduced_counts = {key: int(round(float(value))) for key, value in zip(keys, count_tensor.tolist())}
    return reduced_sums, reduced_counts


def _wrap_ddp_model(model: torch.nn.Module, *, device: torch.device) -> torch.nn.Module:
    if not _use_ddp():
        return model
    if device.type == "cuda":
        return torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[_local_rank()],
            output_device=_local_rank(),
        )
    return torch.nn.parallel.DistributedDataParallel(model)


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    try:
        import numpy as np

        np.random.seed(int(seed))
    except Exception:
        pass
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def deep_update(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_update(dict(out[key]), value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_training_config(config_path: str | Path | None = None) -> dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)
    if config_path is None:
        return config
    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream) or {}
    return deep_update(config, loaded)


def apply_overrides(config: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(config)
    for dotted_key, value in overrides.items():
        if value is None:
            continue
        cursor = out
        parts = str(dotted_key).split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return out


def _merged_training_dict(config: Mapping[str, Any]) -> dict[str, Any]:
    """Merge legacy 3D training keys into PIR-DM's training config shape."""
    train_cfg = dict(config.get("training") or {})

    legacy_optimizer = dict(config.get("optimizer") or {})
    if legacy_optimizer:
        optimizer = {"name": "adamw", **dict(train_cfg.get("optimizer") or {})}
        for key, value in legacy_optimizer.items():
            optimizer.setdefault(key, value)
        train_cfg["optimizer"] = optimizer

    legacy_checkpoint = dict(config.get("checkpoint") or {})
    if legacy_checkpoint.get("output_dir") and not train_cfg.get("output_dir"):
        train_cfg["output_dir"] = legacy_checkpoint["output_dir"]
    if legacy_checkpoint.get("save_every") is not None and train_cfg.get("save_every_steps") is None:
        train_cfg["save_every_steps"] = legacy_checkpoint["save_every"]

    legacy_loss_fields = {field.name for field in fields(TrainingLossConfig)}
    loss_cfg = dict(train_cfg.get("loss") or {})
    for key in legacy_loss_fields:
        if key in train_cfg and key not in loss_cfg:
            loss_cfg[key] = train_cfg[key]
    train_cfg["loss"] = loss_cfg
    return train_cfg


def build_pirdm_training_config(config: Mapping[str, Any]) -> TrainingConfig:
    """Build PIR-DM's canonical TrainingConfig for shared training controls."""
    train_cfg = _merged_training_dict(config)
    allowed = {field.name for field in fields(TrainingConfig)}
    data = {key: value for key, value in train_cfg.items() if key in allowed}
    training_config = TrainingConfig(**data)
    training_config.validate()
    return training_config


def _seed_from_config(config: Mapping[str, Any]) -> int:
    experiment_cfg = dict(config.get("experiment") or {})
    train_cfg = _merged_training_dict(config)
    return int(experiment_cfg.get("seed", train_cfg.get("seed", 0)))


def _device_from_config(config: Mapping[str, Any]) -> torch.device:
    runtime_cfg = dict(config.get("runtime") or {})
    train_cfg = _merged_training_dict(config)
    requested = runtime_cfg.get("device", train_cfg.get("device", "cuda"))
    device = torch.device(str(requested))
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return _setup_distributed(device)


def _resume_path_from_config(config: Mapping[str, Any]) -> Optional[Path]:
    train_cfg = _merged_training_dict(config)
    training_checkpoint = train_cfg.get("checkpoint_path")
    if training_checkpoint not in (None, "", "none", "None"):
        return Path(training_checkpoint).expanduser().resolve()
    runtime_cfg = dict(config.get("runtime") or {})
    checkpoint = runtime_cfg.get("checkpoint")
    if checkpoint in (None, "", "none", "None"):
        return None
    return Path(checkpoint).expanduser().resolve()


def _load_training_state_from_config(config: Mapping[str, Any]) -> bool:
    train_cfg = _merged_training_dict(config)
    if "load_training_state" in train_cfg:
        return bool(train_cfg.get("load_training_state"))
    runtime_cfg = dict(config.get("runtime") or {})
    if "load_training_state" in runtime_cfg:
        return bool(runtime_cfg.get("load_training_state"))
    return True


def _output_dir_from_config(training_config: TrainingConfig) -> Path:
    output_dir = training_config.output_dir or DEFAULT_CONFIG["training"]["output_dir"]
    return Path(output_dir).expanduser().resolve()


def _max_steps_from_config(config: Mapping[str, Any]) -> Optional[int]:
    value = _merged_training_dict(config).get("max_steps")
    if value in (None, "none", "None"):
        return None
    return int(value)


def _log_every_from_config(config: Mapping[str, Any]) -> int:
    return max(1, int(_merged_training_dict(config).get("log_every", 1)))


def _save_every_steps_from_config(config: Mapping[str, Any]) -> int:
    return int(_merged_training_dict(config).get("save_every_steps", 0) or 0)


def _trainer_config_from_training(
    training_config: TrainingConfig,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    train_cfg = _merged_training_dict(config)
    trainer_cfg = training_config.to_direct_trainer_config()
    for key in (
        "per_block_equal_weight",
        "loss_block_indices",
        "volume_weighted_mse_weight",
        "surface_cp_weight",
        "surface_cf_weight",
        "surface_loss_dtype",
    ):
        if key in train_cfg:
            trainer_cfg[key] = train_cfg[key]
    return trainer_cfg


def build_model_from_config(
    config: Mapping[str, Any],
    *,
    block_catalog: list[dict[str, Any]],
    dataset_geometry_dim: int | None = None,
) -> HierarchicalGlobalFNO3D:
    model_cfg = dict(config.get("model") or {})
    geometry_dim = model_cfg.pop("geometry_dim", None)
    if geometry_dim is None:
        if dataset_geometry_dim is None:
            raise ValueError("geometry_dim is not set in config and dataset metadata did not provide it")
        geometry_dim = int(dataset_geometry_dim)
    return HierarchicalGlobalFNO3D(
        block_catalog=block_catalog,
        geometry_dim=int(geometry_dim),
        flow_condition_dim=int(model_cfg.pop("flow_condition_dim", 3)),
        coord_channels=int(model_cfg.pop("coord_channels", 6)),
        out_channels=int(model_cfg.pop("out_channels", 6)),
        patch_size=tuple(int(v) for v in model_cfg.pop("patch_size", [8, 4, 8])),
        width=int(model_cfg.pop("width", 16)),
        modes=tuple(int(v) for v in model_cfg.pop("modes", [8, 4, 4])),
        n_layers=int(model_cfg.pop("n_layers", 2)),
        condition_dim=int(model_cfg.pop("condition_dim", 64)),
        global_layers=int(model_cfg.pop("global_layers", 1)),
        input_state_channels=int(model_cfg.pop("input_state_channels", 0)),
        post_global_layers=int(model_cfg.pop("post_global_layers", 2)),
    )


def build_normalizer_from_config(config: Mapping[str, Any]) -> Optional[torch.nn.Module]:
    data_cfg = dict(config.get("data") or {})
    norm_cfg = dict(config.get("normalization") or {})
    index_path = data_cfg.get("index_path")
    if index_path is None:
        raise ValueError("data.index_path is required")
    return create_normalizer_from_metadata(
        norm_cfg.get("metadata_path") or metadata_path_from_index(index_path),
        normalize=bool(norm_cfg.get("normalize", True)),
        scale_turbulent=bool(norm_cfg.get("scale_turbulent", False)),
        turbulent_stats_path=norm_cfg.get("turbulent_stats_path"),
        turbulent_channel_idx=int(norm_cfg.get("turbulent_channel_idx", 5)),
    )


def save_training_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    config: Mapping[str, Any],
    pirdm_training_config: TrainingConfig,
    epoch: int,
    global_step: int,
    metrics: Mapping[str, Any],
    history: list[dict[str, float]],
    block_catalog: list[dict[str, Any]],
    dataset_metadata: Mapping[str, Any],
    normalizer: Optional[torch.nn.Module],
    split: Any,
) -> Path:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    model_to_save = model.module if hasattr(model, "module") else model
    payload = {
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": None if lr_scheduler is None else lr_scheduler.state_dict(),
        "model_info": model_to_save.get_model_info() if hasattr(model_to_save, "get_model_info") else {},
        "config": dict(config),
        "pirdm_training_config": asdict(pirdm_training_config),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "metrics": dict(metrics),
        "history": list(history),
        "block_catalog": list(block_catalog),
        "dataset_schema": {
            "version": dataset_metadata.get("version"),
            "geometry_schema": dataset_metadata.get("geometry_schema"),
            "geometry_dim": dataset_metadata.get("geometry_dim"),
            "geometry_feature_names": list(
                dataset_metadata.get("geometry_feature_names") or []
            ),
            "state_fields": list(dataset_metadata.get("state_fields") or []),
            "flow_names": list(dataset_metadata.get("flow_names") or []),
            "force_names": list(dataset_metadata.get("force_names") or []),
        },
        "normalizer_config": normalizer.get_config() if hasattr(normalizer, "get_config") else None,
        "split": {
            "train_indices": list(getattr(split, "train_indices", [])),
            "val_indices": list(getattr(split, "val_indices", [])),
            "split_key": str(getattr(split, "split_key", "")),
        },
    }
    torch.save(payload, path)
    return path


def _load_training_resume(
    checkpoint_path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    trainer: WholeCaseDirectTrainer,
    device: torch.device,
    load_training_state: bool,
) -> tuple[int, list[dict[str, float]], dict[str, Any]]:
    payload = load_model_checkpoint(
        model,
        checkpoint_path,
        device,
        use_ema=False,
        context=f"surrogate3d training resume {checkpoint_path}",
    )
    if not load_training_state:
        trainer.global_step = 0
        return 0, [], dict(payload)
    if "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if lr_scheduler is not None and payload.get("lr_scheduler_state_dict") is not None:
        lr_scheduler.load_state_dict(payload["lr_scheduler_state_dict"])
    trainer.global_step = int(payload.get("global_step", 0) or 0)
    history = list(payload.get("history", []))
    start_epoch = int(payload.get("epoch", -1) or -1) + 1
    return max(start_epoch, 0), history, dict(payload)


def train_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
    data_cfg = dict(config.get("data") or {})
    training_config = build_pirdm_training_config(config)
    trainer_config = whole_case_trainer_config_from_dict(
        _trainer_config_from_training(training_config, config)
    )

    require_dataset_cell_volumes(
        data_cfg["index_path"],
        use_volume_weighted_mse=bool(trainer_config.use_volume_weighted_mse),
    )

    seed = _seed_from_config(config)
    _set_seed(seed)

    device = _device_from_config(config)
    try:
        loaders = create_whole_case_loaders(
            data_cfg["index_path"],
            batch_size=int(data_cfg.get("batch_size", 1)),
            val_fraction=float(data_cfg.get("val_fraction", 0.2)),
            seed=seed,
            num_cases=data_cfg.get("num_cases"),
            num_workers=int(data_cfg.get("num_workers", 0)),
            pin_memory=bool(data_cfg.get("pin_memory", False)),
            shuffle_train=bool(data_cfg.get("shuffle_train", True)),
            split_by_sample_id=bool(data_cfg.get("split_by_sample_id", True)),
            cache_size_mb=int(data_cfg.get("cache_size_mb", 256)),
            use_ddp=_use_ddp(),
        )
        normalizer = build_normalizer_from_config(config)
        model = build_model_from_config(
            config,
            block_catalog=loaders.dataset.get_block_catalog(),
            dataset_geometry_dim=int(loaders.dataset.metadata.get("geometry_dim", 0) or 0),
        )

        pirdm_config = SimpleNamespace(training=training_config)
        optimizer = create_optimizer(model, pirdm_config)
        lr_scheduler = create_lr_scheduler(
            optimizer,
            pirdm_config,
            steps_per_epoch=len(loaders.train_loader),
        )

        history: list[dict[str, float]] = []
        start_epoch = 0
        resume_stub = SimpleNamespace(global_step=0)
        resume_path = _resume_path_from_config(config)
        if resume_path is not None:
            start_epoch, history, _ = _load_training_resume(
                resume_path,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                trainer=resume_stub,
                device=device,
                load_training_state=_load_training_state_from_config(config),
            )

        model = _wrap_ddp_model(model.to(device), device=device)
        trainer = WholeCaseDirectTrainer(
            model,
            optimizer,
            config=trainer_config,
            normalizer=normalizer,
            lr_scheduler=lr_scheduler,
            device=device,
        )
        trainer.global_step = int(getattr(resume_stub, "global_step", 0))

        output_dir = _output_dir_from_config(training_config)
        output_dir.mkdir(parents=True, exist_ok=True)
        resolved_config_path = output_dir / "resolved_config.json"
        if _is_main_process():
            with resolved_config_path.open("w", encoding="utf-8") as stream:
                json.dump(config, stream, indent=2, sort_keys=True)
                stream.write("\n")

        max_steps = _max_steps_from_config(config)
        log_every = _log_every_from_config(config)
        save_every_steps = _save_every_steps_from_config(config)
        checkpoint_every_epochs = training_config.checkpoint_every_epochs
        total_epochs = int(training_config.epochs)
        last_metrics: dict[str, Any] = {}
        started = time.time()

        stop = False
        for epoch in range(start_epoch, total_epochs):
            train_sampler = getattr(loaders.train_loader, "sampler", None)
            if hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(epoch)

            train_loss_sum = 0.0
            train_loss_count = 0
            train_metric_sums: dict[str, float] = {}
            train_metric_counts: dict[str, int] = {}
            for batch in loaders.train_loader:
                result = trainer.train_step(batch, epoch=epoch)
                loss_value = float(result["loss"].detach().cpu())
                train_loss_sum += loss_value
                train_loss_count += 1
                last_metrics = {key: float(value) for key, value in result["metrics"].items()}
                for key, value in last_metrics.items():
                    train_metric_sums[key] = float(train_metric_sums.get(key, 0.0) + float(value))
                    train_metric_counts[key] = int(train_metric_counts.get(key, 0) + 1)
                if trainer.global_step % log_every == 0:
                    reduced_metrics = _ddp_average_metrics(last_metrics, device=device)
                    reduced_loss = _ddp_average_scalar(loss_value, device=device)
                    if _is_main_process():
                        print(
                            json.dumps(
                                {
                                    "epoch": epoch,
                                    "step": trainer.global_step,
                                    "loss": reduced_loss,
                                    "mse": reduced_metrics.get("mse"),
                                    "volume_weighted_mse": reduced_metrics.get("volume_weighted_mse"),
                                    "surface_cp_mse": reduced_metrics.get("surface_cp_mse"),
                                    "surface_cf_mse": reduced_metrics.get("surface_cf_mse"),
                                    "physical_mse": reduced_metrics.get("physical_mse"),
                                    "lr": float(optimizer.param_groups[0]["lr"]),
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                if save_every_steps > 0 and trainer.global_step % save_every_steps == 0 and _is_main_process():
                    save_training_checkpoint(
                        output_dir / f"step_{trainer.global_step:08d}.pt",
                        model=trainer.model,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        config=config,
                        pirdm_training_config=training_config,
                        epoch=epoch,
                        global_step=trainer.global_step,
                        metrics=last_metrics,
                        history=history,
                        block_catalog=loaders.dataset.get_block_catalog(),
                        dataset_metadata=loaders.dataset.metadata,
                        normalizer=normalizer,
                        split=loaders.split,
                    )
                if max_steps is not None and trainer.global_step >= max_steps:
                    stop = True
                    break

            train_loss_sum, train_loss_count = _ddp_sum_count(train_loss_sum, train_loss_count, device=device)
            train_metric_sums, train_metric_counts = _ddp_reduce_metric_sums_counts(
                train_metric_sums,
                train_metric_counts,
                device=device,
            )

            epoch_metrics = {
                "epoch": float(epoch),
                "global_step": float(trainer.global_step),
                "train_loss": float(train_loss_sum / max(1, train_loss_count)),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
            for key in sorted(train_metric_sums.keys()):
                epoch_metrics[f"train_{key}"] = float(
                    train_metric_sums[key] / max(1, train_metric_counts.get(key, 1))
                )

            if loaders.val_loader is not None:
                val_sampler = getattr(loaders.val_loader, "sampler", None)
                if hasattr(val_sampler, "set_epoch"):
                    val_sampler.set_epoch(epoch)
                val_loss_sum = 0.0
                val_loss_count = 0
                val_metric_sums: dict[str, float] = {}
                val_metric_counts: dict[str, int] = {}
                for batch in loaders.val_loader:
                    result = trainer.validate_step(batch, epoch=epoch)
                    loss_value = float(result["loss"].detach().cpu())
                    val_loss_sum += loss_value
                    val_loss_count += 1
                    val_metrics = {key: float(value) for key, value in result["metrics"].items()}
                    for key, value in val_metrics.items():
                        val_metric_sums[key] = float(val_metric_sums.get(key, 0.0) + float(value))
                        val_metric_counts[key] = int(val_metric_counts.get(key, 0) + 1)
                val_loss_sum, val_loss_count = _ddp_sum_count(val_loss_sum, val_loss_count, device=device)
                val_metric_sums, val_metric_counts = _ddp_reduce_metric_sums_counts(
                    val_metric_sums,
                    val_metric_counts,
                    device=device,
                )
                epoch_metrics["val_loss"] = float(val_loss_sum / max(1, val_loss_count))
                for key in sorted(val_metric_sums.keys()):
                    epoch_metrics[f"val_{key}"] = float(
                        val_metric_sums[key] / max(1, val_metric_counts.get(key, 1))
                    )

            if _is_main_process():
                history.append(epoch_metrics)
                print(json.dumps(epoch_metrics, sort_keys=True), flush=True)

                if (
                    checkpoint_every_epochs is not None
                    and (epoch + 1) % int(checkpoint_every_epochs) == 0
                ):
                    save_training_checkpoint(
                        output_dir / f"checkpoint_epoch_{epoch + 1:04d}.pt",
                        model=trainer.model,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        config=config,
                        pirdm_training_config=training_config,
                        epoch=epoch,
                        global_step=trainer.global_step,
                        metrics=last_metrics,
                        history=history,
                        block_catalog=loaders.dataset.get_block_catalog(),
                        dataset_metadata=loaders.dataset.metadata,
                        normalizer=normalizer,
                        split=loaders.split,
                    )
            if stop:
                break

        if _is_main_process():
            final_checkpoint = save_training_checkpoint(
                output_dir / "last.pt",
                model=trainer.model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                config=config,
                pirdm_training_config=training_config,
                epoch=int(history[-1]["epoch"]) if history else start_epoch,
                global_step=trainer.global_step,
                metrics=last_metrics,
                history=history,
                block_catalog=loaders.dataset.get_block_catalog(),
                dataset_metadata=loaders.dataset.metadata,
                normalizer=normalizer,
                split=loaders.split,
            )
            summary = {
                "global_step": int(trainer.global_step),
                "history": history,
                "last_checkpoint": str(final_checkpoint),
                "resolved_config": str(resolved_config_path),
                "n_train_cases": len(loaders.split.train_indices),
                "n_val_cases": len(loaders.split.val_indices),
                "n_parameters": int((trainer.model.module if hasattr(trainer.model, "module") else trainer.model).count_parameters()[0]),
                "device": str(device),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "elapsed_s": float(time.time() - started),
                "_is_main_process": True,
            }
            with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
                json.dump(summary, stream, indent=2, sort_keys=True)
                stream.write("\n")
        else:
            summary = {
                "global_step": int(trainer.global_step),
                "device": str(device),
                "_is_main_process": False,
            }
        return summary
    finally:
        if _use_ddp():
            _cleanup_distributed()


__all__ = [
    "DEFAULT_CONFIG",
    "apply_overrides",
    "build_model_from_config",
    "build_normalizer_from_config",
    "build_pirdm_training_config",
    "deep_update",
    "load_training_config",
    "metadata_path_from_index",
    "save_training_checkpoint",
    "train_from_config",
]
