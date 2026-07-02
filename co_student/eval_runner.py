"""Shared COCO evaluation loop for detection predictors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm

from co_student.coco_eval_utils import (
    build_label_to_coco_category_id,
    detections_to_coco_predictions,
    run_coco_eval,
    save_eval_outputs,
)
from co_student.predictors.base import DetectionPredictor
from co_student.predictors.faster_rcnn import FasterRCNNPredictor
from co_student.predictors.rfdetr import RFDETRPredictor
from co_student.sahi_inference import SahiInferenceConfig, SahiPredictor


def _open_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def run_predictor_on_coco(
    predictor: DetectionPredictor,
    *,
    image_dir: Path,
    coco_gt: COCO,
    threshold: float = 0.0,
    max_images: int | None = None,
    sahi: bool = False,
    sahi_overlap: float = 0.2,
    sahi_slice_size: int | None = None,
    sahi_postprocess: str = "GREEDYNMM",
) -> tuple[list[dict[str, Any]], int, SahiPredictor | None]:
    """Run *predictor* over COCO images and return raw prediction dicts."""
    img_ids = coco_gt.getImgIds()
    if max_images is not None:
        img_ids = img_ids[:max_images]
    if not img_ids:
        raise ValueError("No images found in the COCO annotation file.")

    class_names = predictor.class_names
    label_to_category_id = build_label_to_coco_category_id(class_names, coco_gt)

    sahi_predictor: SahiPredictor | None = None
    if sahi:
        if not isinstance(predictor, RFDETRPredictor):
            raise ValueError("--sahi is only supported for RF-DETR checkpoints")
        sahi_predictor = predictor.build_sahi_predictor(
            threshold=threshold,
            config=SahiInferenceConfig(
                slice_size=sahi_slice_size,
                overlap_ratio=sahi_overlap,
                postprocess_type=sahi_postprocess,
            ),
        )

    predictions: list[dict[str, Any]] = []
    missing_images = 0

    for img_id in tqdm(img_ids, desc="Inference"):
        info = coco_gt.loadImgs(img_id)[0]
        image_path = image_dir / info["file_name"]
        if not image_path.is_file():
            missing_images += 1
            continue

        rgb_image = _open_rgb_image(image_path)
        if sahi_predictor is not None:
            detections = sahi_predictor.predict(rgb_image)
        else:
            detections = predictor.predict(rgb_image, threshold=threshold)

        predictions.extend(
            detections_to_coco_predictions(
                detections,
                int(img_id),
                label_to_category_id,
            )
        )

    return predictions, missing_images, sahi_predictor


def evaluate_predictor_on_coco(
    predictor: DetectionPredictor,
    *,
    image_dir: Path,
    ann_path: Path,
    output_dir: Path,
    threshold: float = 0.0,
    max_images: int | None = None,
    sahi: bool = False,
    sahi_overlap: float = 0.2,
    sahi_slice_size: int | None = None,
    sahi_postprocess: str = "GREEDYNMM",
    eval_config_extra: dict[str, Any] | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Run inference plus COCO eval and write outputs under *output_dir*."""
    coco_gt = COCO(str(ann_path))
    predictions, missing_images, sahi_predictor = run_predictor_on_coco(
        predictor,
        image_dir=image_dir,
        coco_gt=coco_gt,
        threshold=threshold,
        max_images=max_images,
        sahi=sahi,
        sahi_overlap=sahi_overlap,
        sahi_slice_size=sahi_slice_size,
        sahi_postprocess=sahi_postprocess,
    )

    if missing_images:
        print(f"Warning: skipped {missing_images} image(s) missing from {image_dir}")

    if not predictions:
        raise ValueError(
            "No predictions produced. Check checkpoint, category mapping, and threshold."
        )

    _, overall, per_class = run_coco_eval(coco_gt, predictions, iou_type="bbox")
    img_ids = coco_gt.getImgIds()
    if max_images is not None:
        img_ids = img_ids[:max_images]
    eval_config = {
        "image_dir": str(image_dir),
        "ann_file": str(ann_path),
        "threshold": threshold,
        "num_images": len(img_ids),
        "num_predictions": len(predictions),
        "sahi": sahi,
        "sahi_overlap": sahi_overlap if sahi else None,
        "sahi_slice_size": sahi_predictor.slice_size if sahi_predictor else None,
        "sahi_postprocess": sahi_postprocess if sahi else None,
    }
    if eval_config_extra:
        eval_config.update(eval_config_extra)

    save_eval_outputs(
        output_dir,
        predictions=predictions,
        overall=overall,
        per_class=per_class,
        label_to_category_id=build_label_to_coco_category_id(predictor.class_names, coco_gt),
        class_names=predictor.class_names,
        eval_config=eval_config,
    )
    return overall, per_class


def evaluate_faster_rcnn_model(
    model: torch.nn.Module,
    class_names: list[str],
    *,
    image_dir: Path,
    ann_path: Path,
    device: str | torch.device,
) -> dict[str, float]:
    """Evaluate an in-memory Faster R-CNN model without writing files."""
    predictor = FasterRCNNPredictor.from_model(model, class_names, device=device)
    coco_gt = COCO(str(ann_path))
    predictions, _, _ = run_predictor_on_coco(
        predictor,
        image_dir=image_dir,
        coco_gt=coco_gt,
        threshold=0.0,
    )
    if not predictions:
        return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0, "AR100": 0.0}
    _, overall, _ = run_coco_eval(coco_gt, predictions, iou_type="bbox")
    return overall
