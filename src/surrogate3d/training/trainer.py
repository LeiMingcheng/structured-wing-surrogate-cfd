"""PIR-DM-compatible trainer for whole-case 3D direct models."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import torch

from surrogate.direct.training import DirectTrainer

from surrogate3d.training.config import WholeCaseDirectTrainerConfig, whole_case_trainer_config_from_dict
from surrogate3d.training.losses import (
    inverse_transform_blocks,
    move_whole_case_batch,
    transform_target_blocks,
    whole_case_mse,
    whole_case_volume_weighted_mse,
)
from surrogate3d.training.surface_losses import WholeCaseSurfaceLossHelper
from surrogate3d.training.volume_guard import require_volume_blocks, volume_blocks_from_batch


class WholeCaseDirectTrainer(DirectTrainer):
    """Direct trainer adapter for dict-of-blocks whole-case models."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        *,
        config: Optional[WholeCaseDirectTrainerConfig | Dict[str, Any]] = None,
        normalizer: Optional[torch.nn.Module] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        device: str | torch.device = "cuda",
    ) -> None:
        if isinstance(config, WholeCaseDirectTrainerConfig):
            trainer_config = config
        else:
            trainer_config = whole_case_trainer_config_from_dict(dict(config or {}))
        self._validate_supported_loss_terms(trainer_config)
        super().__init__(
            model=model,
            optimizer=optimizer,
            config=trainer_config,
            normalizer=normalizer,
            device=device,
        )
        self.config: WholeCaseDirectTrainerConfig
        self.lr_scheduler = lr_scheduler
        self.surface_losses = WholeCaseSurfaceLossHelper()

    @staticmethod
    def _validate_supported_loss_terms(config: WholeCaseDirectTrainerConfig) -> None:
        unsupported = {
            "wall_mse_weight": config.wall_mse_weight,
            "wall_cp_weight": config.wall_cp_weight,
            "gradient_loss_weight": config.gradient_loss_weight,
            "flow_perceptual_weight": config.flow_perceptual_weight,
            "res_loss_weight": config.res_loss_weight if config.res_loss_enabled else 0.0,
        }
        enabled = [name for name, value in unsupported.items() if float(value or 0.0) > 0.0]
        if enabled:
            raise NotImplementedError(
                "WholeCaseDirectTrainer currently supports supervised MSE and optional "
                f"volume-weighted MSE only; 2D PIR-DM regularizers are not yet valid for 3D: {enabled}"
            )

    def _move_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return move_whole_case_batch(batch, self.device)

    def _target_blocks_from(self, batch: Mapping[str, Any]) -> dict[int, torch.Tensor]:
        return transform_target_blocks(batch, self.normalizer)

    def _volume_blocks_from(self, batch: Mapping[str, Any]) -> Optional[dict[int, torch.Tensor]]:
        return volume_blocks_from_batch(batch)

    def forward_batch(self, batch: Dict[str, Any]) -> dict[int, torch.Tensor]:
        batch = self._move_batch(batch)
        return self.model(
            batch["geometry"],
            batch["flow_conditions"],
            batch["blocks"],
            initial_field=batch.get("initial_field"),
        )

    def compute_loss(
        self,
        pred: Mapping[int, torch.Tensor],
        target: Mapping[int, torch.Tensor],
        batch: Dict[str, Any],
        *,
        epoch: int = 0,
        include_training_regularizers: bool = True,
        use_volume_weighted_base: bool = True,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        del epoch, include_training_regularizers
        mse_loss, metric_tensors = whole_case_mse(
            pred,
            target,
            block_indices=self.config.loss_block_indices,
            per_block_equal_weight=self.config.per_block_equal_weight,
        )
        total_loss = torch.zeros((), device=mse_loss.device, dtype=mse_loss.dtype)
        metric_tensors["mse"] = mse_loss.detach()

        vw_loss = None
        if self.config.use_volume_weighted_mse:
            volumes = require_volume_blocks(batch)
            volume_loss, volume_metrics = whole_case_volume_weighted_mse(
                pred,
                target,
                volumes,
                block_indices=self.config.loss_block_indices,
                per_block_equal_weight=self.config.per_block_equal_weight,
                alpha=self.config.volume_weight_alpha,
                w_max=self.config.volume_weight_max,
            )
            metric_tensors.update(volume_metrics)
            vw_loss = volume_loss
        if vw_loss is not None and float(self.config.volume_weighted_mse_weight) > 0.0:
            total_loss = total_loss + float(self.config.volume_weighted_mse_weight) * vw_loss

        pred_phys: Optional[dict[int, torch.Tensor]] = None
        target_phys: Optional[dict[int, torch.Tensor]] = None

        if self.normalizer is not None:
            pred_phys = inverse_transform_blocks(pred, self.normalizer)
            target_phys = inverse_transform_blocks(target, self.normalizer)
            phys_mse, _ = whole_case_mse(
                pred_phys,
                target_phys,
                block_indices=self.config.loss_block_indices,
                per_block_equal_weight=self.config.per_block_equal_weight,
            )
            metric_tensors["physical_mse"] = phys_mse.detach()
        else:
            pred_phys = {int(k): v for k, v in pred.items()}
            target_phys = {int(k): v for k, v in target.items()}

        if (
            (float(self.config.surface_cp_weight) > 0.0 or float(self.config.surface_cf_weight) > 0.0)
            and pred_phys is not None
            and target_phys is not None
        ):
            surface_dtype = getattr(torch, str(self.config.surface_loss_dtype))
            cp_loss, cf_loss = self.surface_losses.compute_surface_losses(
                pred_phys=pred_phys,
                target_phys=target_phys,
                batch=batch,
                device=next(iter(pred.values())).device,
                dtype=surface_dtype,
            )
            metric_tensors["surface_cp_mse"] = cp_loss.detach().to(dtype=next(iter(pred.values())).dtype)
            metric_tensors["surface_cf_mse"] = cf_loss.detach().to(dtype=next(iter(pred.values())).dtype)
            if float(self.config.surface_cp_weight) > 0.0:
                total_loss = total_loss + float(self.config.surface_cp_weight) * cp_loss.to(total_loss.dtype)
            if float(self.config.surface_cf_weight) > 0.0:
                total_loss = total_loss + float(self.config.surface_cf_weight) * cf_loss.to(total_loss.dtype)

        if total_loss.detach().abs().item() == 0.0:
            if vw_loss is not None and use_volume_weighted_base:
                total_loss = vw_loss
            else:
                total_loss = mse_loss

        metrics = {key: float(value.detach().item()) for key, value in metric_tensors.items()}
        metrics["base_loss"] = float(total_loss.detach().item())
        metrics["total_loss"] = float(total_loss.detach().item())
        metrics["volume_weighted_mse_weight"] = float(self.config.volume_weighted_mse_weight)
        metrics["surface_cp_weight"] = float(self.config.surface_cp_weight)
        metrics["surface_cf_weight"] = float(self.config.surface_cf_weight)
        return total_loss, metrics

    def _detach_blocks(self, blocks: Mapping[int, torch.Tensor]) -> dict[int, torch.Tensor]:
        return {int(k): v.detach() for k, v in blocks.items()}

    def train_step(self, batch: Dict[str, Any], *, epoch: int = 0) -> Dict[str, Any]:
        if self.optimizer is None:
            raise RuntimeError("WholeCaseDirectTrainer.train_step requires an optimizer")
        self.model.train()
        batch = self._move_batch(batch)
        target = self._target_blocks_from(batch)
        pred = self.model(
            batch["geometry"],
            batch["flow_conditions"],
            batch["blocks"],
            initial_field=batch.get("initial_field"),
        )
        loss, metrics = self.compute_loss(pred, target, batch, epoch=epoch)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.config.gradient_clip_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                float(self.config.gradient_clip_norm),
            )
            metrics["grad_norm"] = float(grad_norm)
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            metrics["lr"] = float(self.optimizer.param_groups[0]["lr"])
        self.global_step += 1
        pred_detached = self._detach_blocks(pred)
        output: Dict[str, Any] = {
            "loss": loss.detach(),
            "metrics": metrics,
            "prediction": pred_detached,
        }
        if self.normalizer is not None:
            output["prediction_physical"] = inverse_transform_blocks(pred_detached, self.normalizer)
        return output

    @torch.no_grad()
    def validate_step(self, batch: Dict[str, Any], *, epoch: int = 0) -> Dict[str, Any]:
        self.model.eval()
        batch = self._move_batch(batch)
        target = self._target_blocks_from(batch)
        pred = self.model(
            batch["geometry"],
            batch["flow_conditions"],
            batch["blocks"],
            initial_field=batch.get("initial_field"),
        )
        loss, metrics = self.compute_loss(
            pred,
            target,
            batch,
            epoch=epoch,
            include_training_regularizers=False,
        )
        pred_detached = self._detach_blocks(pred)
        output: Dict[str, Any] = {
            "loss": loss.detach(),
            "metrics": metrics,
            "prediction": pred_detached,
        }
        if self.normalizer is not None:
            output["prediction_physical"] = inverse_transform_blocks(pred_detached, self.normalizer)
        return output


__all__ = ["WholeCaseDirectTrainer"]
