"""Co-Student LightningModule for RF-DETR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
from rfdetr._namespace import _namespace_from_configs
from rfdetr.training.callbacks.ema import RFDETREMACallback
from rfdetr.training.module_model import RFDETRModelModule
from rfdetr.training.param_groups import get_param_dict
from rfdetr.utilities.logger import get_logger
from rfdetr.utilities.tensors import NestedTensor

logger = get_logger()

from co_student.pseudo_labels import (
    cvt_detections,
    merge_ground_truth,
    result_to_detections,
    revision_pred,
)
from co_student.train_config import build_lr_lambda


@dataclass
class CoStudentConfig:
    """Hyperparameters for Co-Student pseudo-label co-training."""

    student_score_thresh: float = 0.5
    teacher_score_thresh: float = 0.6
    matching_iou_thresh: float = 0.5
    nms_thresh: float = 0.5


class CoStudentRFDETRModule(RFDETRModelModule):
    """RF-DETR training with Co-Student weak/strong branches and EMA teacher."""

    def __init__(
        self,
        model_config,
        train_config,
        costudent_config: Optional[CoStudentConfig] = None,
    ) -> None:
        super().__init__(model_config, train_config)
        self.costudent_config = costudent_config or CoStudentConfig()
        self._warned_multi_scale: bool = False

    def configure_optimizers(self) -> Dict[str, Any]:
        """AdamW + LambdaLR with multi-epoch step decay from CoStudentTrainConfig."""
        tc = self.train_config
        ns = _namespace_from_configs(self.model_config, tc)

        model_for_params = getattr(self.model, "_orig_mod", self.model)
        param_dicts = get_param_dict(ns, model_for_params)
        param_dicts = [p for p in param_dicts if p["params"].requires_grad]
        optimizer = torch.optim.AdamW(
            param_dicts,
            lr=tc.lr,
            weight_decay=tc.weight_decay,
            fused=self._use_fused_optimizer,
        )

        total_steps = int(self.trainer.estimated_stepping_batches)
        steps_per_epoch = max(1, total_steps // tc.epochs)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=build_lr_lambda(
                tc,
                total_steps=total_steps,
                steps_per_epoch=steps_per_epoch,
            ),
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def on_train_batch_start(
        self,
        batch: Union[dict[str, Any], Tuple[Any, Any]],
        batch_idx: int,
    ) -> None:
        """Skip parent multi-scale hook for Co-Student dict batches."""
        if isinstance(batch, dict):
            tc = self.train_config
            if (
                tc.multi_scale
                and not tc.do_random_resize_via_padding
                and not self._warned_multi_scale
            ):
                logger.warning(
                    "Co-Student training ignores multi_scale batch resizing; "
                    "views are fixed at model resolution by triple-view augmentation."
                )
                self._warned_multi_scale = True
            return
        super().on_train_batch_start(batch, batch_idx)

    def _get_teacher_model(self) -> Optional[torch.nn.Module]:
        if self.trainer is None:
            return None
        for callback in self.trainer.callbacks:
            if isinstance(callback, RFDETREMACallback) and callback._average_model is not None:
                return callback._average_model.module.model
        return None

    @staticmethod
    def _image_size_from_target(target: dict[str, Any]) -> tuple[int, int]:
        if "size" in target:
            return int(target["size"][0]), int(target["size"][1])
        orig = target["orig_size"]
        return int(orig[0]), int(orig[1])

    def _compute_branch_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss_dict = self.criterion(outputs, targets)
        weight_dict = self.criterion.weight_dict
        loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)
        return loss, loss_dict

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        cfg = self.costudent_config
        raw_samples, raw_targets = batch["raw"]
        weak_samples, weak_targets = batch["weak"]
        strong_samples, strong_targets = batch["strong"]
        transforms = batch["transforms"]
        batch_size = len(weak_targets)

        weak_outputs = self.model(weak_samples, weak_targets)
        strong_outputs = self.model(strong_samples, strong_targets)

        with torch.no_grad():
            def _decode_batch(
                outputs: dict[str, torch.Tensor],
                targets: tuple[dict[str, Any], ...],
                score_thresh: float,
            ) -> list:
                orig_sizes = torch.stack([t["size"] for t in targets])
                image_sizes = [self._image_size_from_target(t) for t in targets]
                core = {k: v for k, v in outputs.items() if k != "aux_outputs"}
                results = self.postprocess(core, orig_sizes)
                return [
                    result_to_detections(results[i], image_sizes[i], score_thresh, cfg.nms_thresh)
                    for i in range(batch_size)
                ]

            weak_preds = _decode_batch(weak_outputs, weak_targets, cfg.student_score_thresh)
            strong_preds = _decode_batch(strong_outputs, strong_targets, cfg.student_score_thresh)

            teacher_model = self._get_teacher_model()
            if teacher_model is not None:
                teacher_outputs = teacher_model(raw_samples)
                teacher_preds = _decode_batch(teacher_outputs, raw_targets, cfg.teacher_score_thresh)
                for i in range(batch_size):
                    tm = transforms[i]
                    weak_size = self._image_size_from_target(weak_targets[i])
                    strong_size = self._image_size_from_target(strong_targets[i])
                    teacher_weak = cvt_detections(
                        teacher_preds[i], tm["raw"], tm["weak"], weak_size
                    )
                    teacher_strong = cvt_detections(
                        teacher_preds[i], tm["raw"], tm["strong"], strong_size
                    )
                    weak_preds[i] = revision_pred(teacher_weak, weak_preds[i])
                    strong_preds[i] = revision_pred(teacher_strong, strong_preds[i])

            weak_merged_targets = [
                merge_ground_truth(
                    weak_targets[i],
                    strong_preds[i],
                    cfg.matching_iou_thresh,
                    source_tm=transforms[i]["strong"],
                    target_tm=transforms[i]["weak"],
                )
                for i in range(batch_size)
            ]
            strong_merged_targets = [
                merge_ground_truth(
                    strong_targets[i],
                    weak_preds[i],
                    cfg.matching_iou_thresh,
                    source_tm=transforms[i]["weak"],
                    target_tm=transforms[i]["strong"],
                )
                for i in range(batch_size)
            ]

        weak_loss, weak_dict = self._compute_branch_loss(weak_outputs, weak_merged_targets)
        strong_loss, strong_dict = self._compute_branch_loss(strong_outputs, strong_merged_targets)

        loss = 0.5 * (weak_loss + strong_loss)

        loss_scaled = loss / self.trainer.accumulate_grad_batches
        train_log_sync_dist = bool(self.train_config.train_log_sync_dist)
        train_log_on_step = bool(self.train_config.train_log_on_step)

        log_dict = {f"train/weak_{k}": v for k, v in weak_dict.items()}
        log_dict.update({f"train/strong_{k}": v for k, v in strong_dict.items()})
        log_dict["train/weak_loss"] = weak_loss
        log_dict["train/strong_loss"] = strong_loss

        self.log_dict(
            log_dict,
            on_step=train_log_on_step,
            on_epoch=True,
            sync_dist=train_log_sync_dist,
            batch_size=batch_size,
        )
        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            on_step=train_log_on_step,
            on_epoch=True,
            sync_dist=train_log_sync_dist,
            batch_size=batch_size,
        )

        optimizer = self.optimizers()
        if isinstance(optimizer, list):
            optimizer = optimizer[0]
        group_lrs = [pg["lr"] for pg in optimizer.param_groups if "lr" in pg]
        if group_lrs:
            self.log("train/lr", group_lrs[0], prog_bar=True, on_step=True, on_epoch=False)

        return loss_scaled
