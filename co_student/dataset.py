"""COCO dataset loading with explicit image and annotation paths."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from rfdetr.datasets.coco import (
    CocoDetection,
    _resolve_runtime_augmentation_backend,
    make_coco_transforms,
    make_coco_transforms_square_div_64,
)
from rfdetr.utilities.logger import get_logger

from co_student.costudent_dataset import CoStudentCocoDataset

logger = get_logger()


@dataclass(frozen=True)
class CocoSplitPaths:
    """Paths for one COCO split (train or val)."""

    image_dir: Path
    ann_path: Path


def resolve_coco_ann_path(path: str | Path) -> Path:
    """Resolve a COCO annotation file from a path or directory.

    If *path* is a directory, looks for ``_annotations.coco.json`` first, then
    a single ``*.json`` file in that directory.

    Args:
        path: Path to a ``.json`` file or a directory containing one.

    Returns:
        Resolved path to the annotation JSON file.

    Raises:
        FileNotFoundError: If no annotation file can be resolved.
        ValueError: If the directory contains multiple JSON files.
    """
    resolved = Path(path).expanduser().resolve()
    if resolved.is_file():
        return resolved
    if not resolved.is_dir():
        raise FileNotFoundError(f"Annotation path does not exist: {resolved}")

    preferred = resolved / "_annotations.coco.json"
    if preferred.is_file():
        return preferred

    json_files = sorted(resolved.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No .json annotation file found in {resolved}")
    if len(json_files) > 1:
        names = ", ".join(p.name for p in json_files)
        raise ValueError(
            f"Multiple JSON files in {resolved}; specify the file explicitly. Found: {names}"
        )
    return json_files[0]


def count_categories(ann_path: str | Path) -> int:
    """Return the number of categories in a COCO annotation file."""
    with open(resolve_coco_ann_path(ann_path), encoding="utf-8") as f:
        data = json.load(f)
    return len(data["categories"])


def build_coco_base_from_paths(
    image_set: str,
    args: Any,
    resolution: int,
    paths: CocoSplitPaths,
    *,
    remap_category_ids: bool = True,
) -> CocoDetection:
    """Build :class:`CocoDetection` without transforms (for Co-Student triple aug)."""
    img_folder = Path(paths.image_dir).expanduser().resolve()
    ann_file = resolve_coco_ann_path(paths.ann_path)

    if not img_folder.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {img_folder}")

    include_masks = getattr(args, "segmentation_head", False)
    logger.info(
        "Building COCO %s base dataset (no CPU transforms): images=%s annotations=%s",
        image_set,
        img_folder,
        ann_file,
    )
    return CocoDetection(
        img_folder,
        ann_file,
        transforms=None,
        include_masks=include_masks,
        remap_category_ids=remap_category_ids,
    )


def build_costudent_train_dataset(
    args: Any,
    resolution: int,
    paths: CocoSplitPaths,
    *,
    augment_seed: int = 0,
    flip_prob: float = 0.5,
    remap_category_ids: bool = True,
) -> CoStudentCocoDataset:
    """Training dataset with raw / weak / strong views and transform matrices."""
    base = build_coco_base_from_paths(
        "train",
        args,
        resolution,
        paths,
        remap_category_ids=remap_category_ids,
    )
    return CoStudentCocoDataset(
        base,
        resolution,
        augment_seed=augment_seed,
        flip_prob=flip_prob,
    )


def build_costudent_train_from_roboflow(
    args: Any,
    resolution: int,
    *,
    augment_seed: int = 0,
    flip_prob: float = 0.5,
) -> CoStudentCocoDataset:
    """Co-Student train set from Roboflow layout under ``args.dataset_dir``."""
    root = Path(args.dataset_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Roboflow dataset path does not exist: {root}")

    img_folder = root / "train"
    ann_file = root / "train" / "_annotations.coco.json"
    include_masks = getattr(args, "segmentation_head", False)
    logger.info(
        "Building Co-Student Roboflow train base: images=%s annotations=%s resolution=%s",
        img_folder,
        ann_file,
        resolution,
    )
    base = CocoDetection(
        img_folder,
        ann_file,
        transforms=None,
        include_masks=include_masks,
        remap_category_ids=True,
    )
    return CoStudentCocoDataset(
        base,
        resolution,
        augment_seed=augment_seed,
        flip_prob=flip_prob,
    )


def build_coco_from_paths(
    image_set: str,
    args: Any,
    resolution: int,
    paths: CocoSplitPaths,
    *,
    remap_category_ids: bool = True,
) -> CocoDetection:
    """Build :class:`CocoDetection` from explicit image and annotation paths."""
    img_folder = Path(paths.image_dir).expanduser().resolve()
    ann_file = resolve_coco_ann_path(paths.ann_path)

    if not img_folder.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {img_folder}")

    square_resize_div_64 = getattr(args, "square_resize_div_64", False)
    include_masks = getattr(args, "segmentation_head", False)
    aug_config = getattr(args, "aug_config", None)
    augmentation_backend = getattr(args, "augmentation_backend", "cpu")
    resolved_backend = _resolve_runtime_augmentation_backend(augmentation_backend)
    gpu_postprocess = resolved_backend != "cpu"

    transform_kwargs = dict(
        multi_scale=getattr(args, "multi_scale", False),
        expanded_scales=getattr(args, "expanded_scales", False),
        skip_random_resize=not getattr(args, "do_random_resize_via_padding", False),
        patch_size=getattr(args, "patch_size", 16),
        num_windows=getattr(args, "num_windows", 4),
        aug_config=aug_config,
        gpu_postprocess=gpu_postprocess,
    )

    if square_resize_div_64:
        transforms = make_coco_transforms_square_div_64(image_set, resolution, **transform_kwargs)
    else:
        transforms = make_coco_transforms(image_set, resolution, **transform_kwargs)

    logger.info(
        "Building COCO %s dataset: images=%s annotations=%s resolution=%s",
        image_set,
        img_folder,
        ann_file,
        resolution,
    )
    return CocoDetection(
        img_folder,
        ann_file,
        transforms=transforms,
        include_masks=include_masks,
        remap_category_ids=remap_category_ids,
    )


def split_paths_from_args(
    image_dir: str | Path,
    ann_path: str | Path,
) -> CocoSplitPaths:
    """Create split paths, resolving the annotation directory or file."""
    return CocoSplitPaths(
        image_dir=Path(image_dir).expanduser().resolve(),
        ann_path=resolve_coco_ann_path(ann_path),
    )
