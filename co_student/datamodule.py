"""DataModule for Co-Student training (triple-view CPU augmentation)."""

from __future__ import annotations

from functools import partial
from typing import Any, Optional, Tuple

import torch
from rfdetr._namespace import _namespace_from_configs
from rfdetr.datasets import build_dataset
from rfdetr.training.module_data import RFDETRDataModule

from co_student.collate import collate_costudent_batch
from co_student.dataset import (
    CocoSplitPaths,
    build_coco_from_paths,
    build_costudent_train_dataset,
    build_costudent_train_from_roboflow,
)


class CoStudentDataModule(RFDETRDataModule):
    """RF-DETR datamodule with CoStudent-style raw / weak / strong train views.
    Based on https://github.com/hustvl/CoStudent

    Training uses :class:`~co_student.costudent_dataset.CoStudentCocoDataset`
    and records 3×3 geometric transform matrices per view. Validation keeps
    the standard RF-DETR pipeline.

    When ``train_paths`` / ``val_paths`` are set, COCO data is loaded from
    explicit image directories and annotation paths instead of the default
    Roboflow layout under ``dataset_dir``.
    """

    def __init__(
        self,
        model_config,
        train_config,
        *,
        train_paths: Optional[CocoSplitPaths] = None,
        val_paths: Optional[CocoSplitPaths] = None,
    ) -> None:
        super().__init__(model_config, train_config)
        self.train_paths = train_paths
        self.val_paths = val_paths

        block_size = model_config.patch_size * model_config.num_windows
        self._train_collate_fn = partial(collate_costudent_batch, block_size=block_size)

    def train_dataloader(self):
        loader = super().train_dataloader()
        loader.collate_fn = self._train_collate_fn
        return loader

    def transfer_batch_to_device(
        self,
        batch: Any,
        device: torch.device,
        dataloader_idx: int,
    ) -> Any:
        """Move Co-Student dict batches or standard (samples, targets) tuples to *device*."""
        if not isinstance(batch, dict):
            return super().transfer_batch_to_device(batch, device, dataloader_idx)
        return self._transfer_costudent_batch(batch, device)

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        """Skip Kornia GPU aug for triple-view train batches (already augmented on CPU)."""
        if isinstance(batch, dict):
            return batch
        return super().on_after_batch_transfer(batch, dataloader_idx)

    @staticmethod
    def _transfer_costudent_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
        non_blocking = device.type == "cuda"

        def _move_branch(samples_targets: Tuple[Any, tuple]) -> Tuple[Any, tuple]:
            samples, targets = samples_targets
            samples = samples.to(device, non_blocking=non_blocking)
            targets = tuple(
                {k: v.to(device, non_blocking=non_blocking) for k, v in t.items()} for t in targets
            )
            return samples, targets

        return {
            "raw": _move_branch(batch["raw"]),
            "weak": _move_branch(batch["weak"]),
            "strong": _move_branch(batch["strong"]),
            "transforms": batch["transforms"],
        }

    def setup(self, stage: str) -> None:
        resolution = self.model_config.resolution
        ns = _namespace_from_configs(self.model_config, self.train_config)
        ns.augmentation_backend = "cpu"
        seed = int(getattr(self.train_config, "seed", 0))

        if stage == "fit":
            if self._dataset_train is None:
                self._dataset_train = self._build_train_dataset(ns, resolution, seed)
            if self._dataset_val is None:
                self._dataset_val = self._build_val_dataset(ns, resolution)
            self._kornia_setup_done = True
        elif stage == "validate":
            if self._dataset_val is None:
                self._dataset_val = self._build_val_dataset(ns, resolution)
        elif stage in ("test", "predict"):
            super().setup(stage)

    def _build_train_dataset(self, ns, resolution: int, seed: int):
        if self.train_paths is not None:
            return build_costudent_train_dataset(
                ns,
                resolution,
                self.train_paths,
                augment_seed=seed,
            )
        if not getattr(self.train_config, "dataset_dir", None):
            raise ValueError("train_config.dataset_dir is required for Co-Student training")
        return build_costudent_train_from_roboflow(ns, resolution, augment_seed=seed)

    def _build_val_dataset(self, ns, resolution: int):
        if self.val_paths is not None:
            return build_coco_from_paths("val", ns, resolution, self.val_paths)
        return build_dataset("val", ns, resolution)
