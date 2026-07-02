#!/usr/bin/env python3
"""Visualize checkpoint predictions on images (optional COCO GT overlay)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import supervision as sv
from PIL import Image
from pycocotools.coco import COCO

from co_student.dataset import resolve_coco_ann_path
from co_student.predictors import RFDETRPredictor, load_predictor
from co_student.sahi_inference import SahiInferenceConfig, SahiPredictor

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


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
        help="Directory containing images to run inference on",
    )
    parser.add_argument(
        "--ann-file",
        default=None,
        help="Optional COCO JSON for image list and --show-gt overlays",
    )
    parser.add_argument(
        "--output-dir",
        default="./viz_outputs",
        help="Directory to write annotated images (default: ./viz_outputs)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Minimum detection confidence (default: 0.5)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Cap the number of images processed (default: all)",
    )
    parser.add_argument(
        "--show-gt",
        action="store_true",
        help="Draw ground-truth boxes from --ann-file in green",
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


def _iter_images(
    image_dir: Path,
    coco: COCO | None,
    max_images: int | None,
) -> Iterator[tuple[Path, int | None]]:
    if coco is not None:
        img_ids = coco.getImgIds()
        if max_images is not None:
            img_ids = img_ids[:max_images]
        for img_id in img_ids:
            info = coco.loadImgs(img_id)[0]
            yield image_dir / info["file_name"], img_id
        return

    paths = sorted(
        p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
    )
    if max_images is not None:
        paths = paths[:max_images]
    for path in paths:
        yield path, None


def _open_rgb_image(path: Path) -> Image.Image:
    """Load an image as RGB (grayscale and palette modes are expanded to 3 channels)."""
    with Image.open(path) as image:
        return image.convert("RGB")


def _class_name(class_names: list[str], class_id: int) -> str:
    if 0 <= class_id < len(class_names):
        return class_names[class_id]
    return str(class_id)


def _detection_labels(detections: sv.Detections, class_names: list[str]) -> list[str]:
    if detections.class_id is None or detections.confidence is None:
        return []
    return [
        f"{_class_name(class_names, int(cid))} {conf:.2f}"
        for cid, conf in zip(detections.class_id, detections.confidence)
    ]


def _annotate_predictions(
    scene: np.ndarray,
    detections: sv.Detections,
    class_names: list[str],
) -> np.ndarray:
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    mask_annotator = sv.MaskAnnotator()

    annotated = scene.copy()
    if detections.mask is not None:
        annotated = mask_annotator.annotate(annotated, detections)
    annotated = box_annotator.annotate(annotated, detections)
    return label_annotator.annotate(annotated, detections, _detection_labels(detections, class_names))


def _draw_gt_boxes(
    scene: np.ndarray,
    coco: COCO,
    img_id: int,
) -> np.ndarray:
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    bgr = cv2.cvtColor(scene, cv2.COLOR_RGB2BGR)

    for ann in anns:
        if "bbox" not in ann:
            continue
        x, y, w, h = [int(round(v)) for v in ann["bbox"]]
        cv2.rectangle(bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cat = coco.loadCats(ann["category_id"])[0]
        label = cat.get("name", str(ann["category_id"]))
        cv2.putText(
            bgr,
            f"GT {label}",
            (x, max(y - 4, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _save_rgb_image(path: Path, rgb: np.ndarray) -> None:
    Image.fromarray(rgb).save(path)


def _load_scene(image: Image.Image, detections: sv.Detections) -> np.ndarray:
    source = detections.metadata.get("source_image") if detections.metadata else None
    if source is not None:
        arr = np.asarray(source)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        return arr
    return np.asarray(image)


def main() -> None:
    args = parse_args()

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    image_dir = Path(args.image_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")
    if not image_dir.is_dir():
        raise SystemExit(f"Image directory not found: {image_dir}")
    if args.show_gt and not args.ann_file:
        raise SystemExit("--show-gt requires --ann-file")

    coco: COCO | None = None
    if args.ann_file:
        ann_path = resolve_coco_ann_path(args.ann_file)
        coco = COCO(str(ann_path))

    output_dir.mkdir(parents=True, exist_ok=True)

    predictor = load_predictor(checkpoint, optimize=args.optimize)
    class_names = predictor.class_names

    sahi_predictor: SahiPredictor | None = None
    if args.sahi:
        if not isinstance(predictor, RFDETRPredictor):
            raise SystemExit("--sahi is only supported for RF-DETR checkpoints")
        sahi_predictor = predictor.build_sahi_predictor(
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

    count = 0

    for image_path, img_id in _iter_images(image_dir, coco, args.max_images):
        if not image_path.is_file():
            print(f"skip missing image: {image_path}")
            continue

        rgb_image = _open_rgb_image(image_path)
        if sahi_predictor is not None:
            detections = sahi_predictor.predict(rgb_image)
        else:
            detections = predictor.predict(rgb_image, threshold=args.threshold)
        scene = _load_scene(rgb_image, detections)
        annotated = _annotate_predictions(scene, detections, class_names)

        n_dets = len(detections)
        if n_dets == 0:
            if sahi_predictor is not None:
                max_conf = sahi_predictor.max_confidence(rgb_image)
            else:
                max_conf = predictor.max_confidence(rgb_image)
            print(
                f"saved {output_dir / f'{image_path.stem}_pred.jpg'} "
                f"(0 predictions at threshold={args.threshold}, max_conf={max_conf:.3f})"
            )
        else:
            top_conf = float(detections.confidence.max())
            print(
                f"saved {output_dir / f'{image_path.stem}_pred.jpg'} "
                f"({n_dets} predictions, top_conf={top_conf:.3f})"
            )

        if args.show_gt and coco is not None and img_id is not None:
            annotated = _draw_gt_boxes(annotated, coco, img_id)

        out_path = output_dir / f"{image_path.stem}_pred.jpg"
        _save_rgb_image(out_path, annotated)
        count += 1

    if count == 0:
        raise SystemExit("No images were processed. Check --image-dir and --ann-file paths.")
    print(f"Wrote {count} image(s) to {output_dir}")


if __name__ == "__main__":
    main()
