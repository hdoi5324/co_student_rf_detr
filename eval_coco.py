#!/usr/bin/env python3
"""Run RF-DETR inference on a COCO dataset and evaluate against ground truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image
from pycocotools.coco import COCO
from rfdetr import RFDETR
from tqdm import tqdm

from co_student.coco_eval_utils import (
    build_label_to_coco_category_id,
    detections_to_coco_predictions,
    run_coco_eval,
    save_eval_outputs,
)
from co_student.dataset import resolve_coco_ann_path
from co_student.sahi_inference import SahiInferenceConfig, SahiPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to an inference-ready .pth (e.g. checkpoint_best_ema.pth)",
    )
    parser.add_argument(
        "--image-dir",
        required=True,
        help="Directory containing COCO images",
    )
    parser.add_argument(
        "--ann-file",
        required=True,
        help="Path to ground-truth COCO JSON (file or directory)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for predictions and evaluation metrics",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Minimum detection confidence for COCO eval (default: 0.0, COCO sweeps scores)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Cap the number of images evaluated (default: all in COCO JSON)",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Call model.optimize_for_inference() before predicting",
    )

    sahi = parser.add_argument_group("SAHI sliced inference")
    sahi.add_argument(
        "--sahi",
        action="store_true",
        help="Run sliced inference on full-resolution images (default: single resized pass)",
    )
    sahi.add_argument(
        "--sahi-overlap",
        type=float,
        default=0.2,
        help="Fractional overlap between adjacent inference tiles (default: 0.2)",
    )
    sahi.add_argument(
        "--sahi-slice-size",
        type=int,
        default=None,
        help="Tile height/width in pixels (default: checkpoint model resolution)",
    )
    sahi.add_argument(
        "--sahi-postprocess",
        default="GREEDYNMM",
        choices=["GREEDYNMM", "NMS"],
        help="How to merge overlapping tile predictions (default: GREEDYNMM)",
    )
    return parser.parse_args()


def _open_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _print_overall_metrics(overall: dict[str, float]) -> None:
    print("\nOverall COCO metrics")
    print(f"  AP @[0.50:0.95] = {overall['AP']:.4f}")
    print(f"  AP @0.50       = {overall['AP50']:.4f}")
    print(f"  AP @0.75       = {overall['AP75']:.4f}")
    print(f"  AR @100        = {overall['AR100']:.4f}")


def _print_per_class_metrics(per_class: list[dict[str, object]]) -> None:
    print("\nPer-class metrics")
    print(f"{'Class':<24} {'AP':>8} {'AP50':>8} {'AP75':>8} {'AR100':>8}")
    print("-" * 60)
    for row in per_class:
        name = str(row["category_name"])
        print(
            f"{name:<24} "
            f"{row['AP']:>8.4f} "
            f"{row['AP50']:>8.4f} "
            f"{row['AP75']:>8.4f} "
            f"{row['AR100']:>8.4f}"
        )


def main() -> None:
    args = parse_args()

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    image_dir = Path(args.image_dir).expanduser().resolve()
    ann_path = resolve_coco_ann_path(args.ann_file)
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")
    if not image_dir.is_dir():
        raise SystemExit(f"Image directory not found: {image_dir}")

    coco_gt = COCO(str(ann_path))
    img_ids = coco_gt.getImgIds()
    if args.max_images is not None:
        img_ids = img_ids[: args.max_images]
    if not img_ids:
        raise SystemExit("No images found in the COCO annotation file.")

    model = RFDETR.from_checkpoint(checkpoint)
    if args.optimize:
        model.optimize_for_inference()

    class_names = model.class_names
    label_to_category_id = build_label_to_coco_category_id(class_names, coco_gt)

    sahi_predictor: SahiPredictor | None = None
    if args.sahi:
        sahi_predictor = SahiPredictor.from_rfdetr(
            model,
            threshold=args.threshold,
            config=SahiInferenceConfig(
                slice_size=args.sahi_slice_size,
                overlap_ratio=args.sahi_overlap,
                postprocess_type=args.sahi_postprocess,
            ),
        )
        print(
            f"SAHI inference: {sahi_predictor.slice_size}x{sahi_predictor.slice_size} tiles, "
            f"overlap={args.sahi_overlap}, postprocess={args.sahi_postprocess}"
        )
    else:
        print("Standard inference: full-image resize to model resolution")

    predictions: list[dict[str, object]] = []
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
            detections = model.predict(rgb_image, threshold=args.threshold)

        predictions.extend(
            detections_to_coco_predictions(
                detections,
                int(img_id),
                label_to_category_id,
            )
        )

    if missing_images:
        print(f"Warning: skipped {missing_images} image(s) missing from {image_dir}")

    if not predictions:
        raise SystemExit(
            "No predictions produced. Check checkpoint, category mapping, and threshold."
        )

    _, overall, per_class = run_coco_eval(coco_gt, predictions, iou_type="bbox")

    eval_config = {
        "checkpoint": str(checkpoint),
        "image_dir": str(image_dir),
        "ann_file": str(ann_path),
        "threshold": args.threshold,
        "num_images": len(img_ids),
        "num_predictions": len(predictions),
        "sahi": args.sahi,
        "sahi_overlap": args.sahi_overlap if args.sahi else None,
        "sahi_slice_size": sahi_predictor.slice_size if sahi_predictor else None,
        "sahi_postprocess": args.sahi_postprocess if args.sahi else None,
    }
    save_eval_outputs(
        output_dir,
        predictions=predictions,
        overall=overall,
        per_class=per_class,
        label_to_category_id=label_to_category_id,
        class_names=class_names,
        eval_config=eval_config,
    )

    _print_overall_metrics(overall)
    _print_per_class_metrics(per_class)
    print(f"\nWrote predictions and metrics to {output_dir}")


if __name__ == "__main__":
    main()
