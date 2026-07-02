#!/usr/bin/env python3
"""Run detection inference on a COCO dataset and evaluate against ground truth."""

from __future__ import annotations

import argparse
from pathlib import Path

from co_student.dataset import resolve_coco_ann_path
from co_student.eval_runner import evaluate_predictor_on_coco
from co_student.predictors import load_predictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to an inference-ready checkpoint (.pth for RF-DETR or Faster R-CNN)",
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
        help="Call model.optimize_for_inference() before predicting (RF-DETR only)",
    )

    sahi = parser.add_argument_group("SAHI sliced inference")
    sahi.add_argument(
        "--sahi",
        action="store_true",
        help="Run sliced inference on full-resolution images (RF-DETR only)",
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

    predictor = load_predictor(checkpoint, optimize=args.optimize)
    if args.sahi:
        print(
            f"SAHI inference: overlap={args.sahi_overlap}, postprocess={args.sahi_postprocess}"
        )
    else:
        print("Standard inference: full-image pass")

    overall, per_class = evaluate_predictor_on_coco(
        predictor,
        image_dir=image_dir,
        ann_path=ann_path,
        output_dir=output_dir,
        threshold=args.threshold,
        max_images=args.max_images,
        sahi=args.sahi,
        sahi_overlap=args.sahi_overlap,
        sahi_slice_size=args.sahi_slice_size,
        sahi_postprocess=args.sahi_postprocess,
        eval_config_extra={"checkpoint": str(checkpoint)},
    )

    _print_overall_metrics(overall)
    _print_per_class_metrics(per_class)
    print(f"\nWrote predictions and metrics to {output_dir}")


if __name__ == "__main__":
    main()
