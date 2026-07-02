"""COCO dataset helpers for torchvision Faster R-CNN training."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torchvision.transforms.functional as F
from rfdetr.utilities.logger import get_logger
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode

from co_student.dataset import (
    CocoSplitPaths,
    build_coco_base_from_paths,
    filter_unannotated_train_images,
    resolve_coco_ann_path,
    summarize_coco_split,
)

logger = get_logger()


def class_names_from_ann(ann_path: str | Path) -> list[str]:
    """Return category names sorted by COCO category id."""
    with open(resolve_coco_ann_path(ann_path), encoding="utf-8") as handle:
        data = json.load(handle)
    categories = sorted(data["categories"], key=lambda cat: cat["id"])
    return [str(cat["name"]) for cat in categories]


class RandomHorizontalFlip:
    """Flip image and boxes with probability *prob*."""

    def __init__(self, prob: float = 0.5) -> None:
        self.prob = prob

    def __call__(
        self,
        image: Any,
        target: dict[str, torch.Tensor],
    ) -> tuple[Any, dict[str, torch.Tensor]]:
        if random.random() >= self.prob:
            return image, target

        width, _ = image.size
        image = F.hflip(image)
        boxes = target["boxes"].clone()
        boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
        flipped = dict(target)
        flipped["boxes"] = boxes
        return image, flipped


class TorchvisionCocoDataset(Dataset):
    """Wrap RF-DETR :class:`CocoDetection` for torchvision detection training."""

    def __init__(
        self,
        base: Dataset,
        *,
        train: bool,
        horizontal_flip_prob: float = 0.5,
    ) -> None:
        self.base = base
        self.train = train
        self.horizontal_flip = RandomHorizontalFlip(horizontal_flip_prob) if train else None

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image, target = self.base[index]
        sample = {
            key: target[key]
            for key in ("boxes", "labels", "image_id", "area", "iscrowd")
            if key in target
        }
        # Faster R-CNN reserves label 0 for background.
        sample["labels"] = sample["labels"] + 1

        if self.horizontal_flip is not None:
            image, sample = self.horizontal_flip(image, sample)

        tensor_image = F.to_tensor(image)
        return tensor_image, sample


def collate_faster_rcnn_batch(
    batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]],
) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
    images, targets = zip(*batch)
    return list(images), list(targets)


def build_faster_rcnn_datasets(
    train_paths: CocoSplitPaths,
    val_paths: CocoSplitPaths | None,
    *,
    horizontal_flip_prob: float = 0.5,
    keep_unannotated: bool = False,
) -> tuple[TorchvisionCocoDataset, TorchvisionCocoDataset | None, list[str]]:
    """Build train and optional val datasets plus sorted class names."""
    class_names = class_names_from_ann(train_paths.ann_path)
    args_stub = type("Args", (), {"segmentation_head": False})()

    train_summary = summarize_coco_split(train_paths.ann_path)
    logger.info(
        "Faster R-CNN train COCO JSON: %d images (%d annotated, %d unannotated)",
        train_summary.total_images,
        train_summary.annotated_images,
        train_summary.unannotated_images,
    )
    train_base = filter_unannotated_train_images(
        build_coco_base_from_paths(
            "train",
            args_stub,
            0,
            train_paths,
            remap_category_ids=True,
        ),
        keep_unannotated=keep_unannotated,
    )
    logger.info("Faster R-CNN train dataloader: %d samples", len(train_base))
    train_dataset = TorchvisionCocoDataset(
        train_base,
        train=True,
        horizontal_flip_prob=horizontal_flip_prob,
    )

    val_dataset = None
    if val_paths is not None:
        val_base = build_coco_base_from_paths(
            "val",
            args_stub,
            0,
            val_paths,
            remap_category_ids=True,
        )
        val_summary = summarize_coco_split(val_paths.ann_path)
        logger.info(
            "Faster R-CNN val COCO JSON: %d images (%d annotated, %d unannotated)",
            val_summary.total_images,
            val_summary.annotated_images,
            val_summary.unannotated_images,
        )
        val_dataset = TorchvisionCocoDataset(val_base, train=False)

    return train_dataset, val_dataset, class_names


@dataclass(frozen=True)
class FasterRCNNDataConfig:
    batch_size: int = 2
    num_workers: int = 4


def build_faster_rcnn_dataloaders(
    train_paths: CocoSplitPaths,
    val_paths: CocoSplitPaths | None,
    *,
    config: FasterRCNNDataConfig | None = None,
    horizontal_flip_prob: float = 0.5,
    keep_unannotated: bool = False,
) -> tuple[DataLoader, DataLoader | None, list[str]]:
    """Return train/val dataloaders and class names."""
    cfg = config or FasterRCNNDataConfig()
    train_dataset, val_dataset, class_names = build_faster_rcnn_datasets(
        train_paths,
        val_paths,
        horizontal_flip_prob=horizontal_flip_prob,
        keep_unannotated=keep_unannotated,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_faster_rcnn_batch,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=collate_faster_rcnn_batch,
            pin_memory=torch.cuda.is_available(),
        )
    return train_loader, val_loader, class_names


def resize_for_inference(image: Any, *, min_size: int = 800, max_size: int = 1333) -> torch.Tensor:
    """Resize a PIL image like torchvision detection models do at inference."""
    tensor = F.to_tensor(image)
    _, height, width = tensor.shape
    scale = min_size / min(height, width)
    if max(height, width) * scale > max_size:
        scale = max_size / max(height, width)

    new_height = max(1, int(round(height * scale)))
    new_width = max(1, int(round(width * scale)))
    return F.resize(
        tensor,
        [new_height, new_width],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
