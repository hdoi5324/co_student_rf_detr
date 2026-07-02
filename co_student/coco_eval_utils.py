"""COCO prediction export and evaluation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import supervision as sv
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def build_label_to_coco_category_id(class_names: list[str], coco: COCO) -> dict[int, int]:
    """Map model label indices to COCO ``category_id`` values in *coco*.

    Prefer name matching between ``class_names`` and COCO categories. Fall back to
    the RF-DETR ``remap_category_ids`` convention (sorted category ids → 0..N-1).
    """
    cats = coco.loadCats(coco.getCatIds())
    name_to_id = {str(cat["name"]).lower(): int(cat["id"]) for cat in cats}

    by_name: dict[int, int] = {}
    for index, name in enumerate(class_names):
        cat_id = name_to_id.get(str(name).lower())
        if cat_id is not None:
            by_name[index] = cat_id

    if len(by_name) == len(class_names):
        return by_name

    sorted_ids = sorted(int(cat_id) for cat_id in coco.getCatIds())
    if len(sorted_ids) == len(class_names):
        return {index: cat_id for index, cat_id in enumerate(sorted_ids)}

    raise ValueError(
        "Could not map model class indices to COCO category ids. "
        f"Model has {len(class_names)} classes ({class_names!r}); "
        f"COCO categories are {[cat['name'] for cat in cats]!r} (ids={sorted_ids})."
    )


def detections_to_coco_predictions(
    detections: sv.Detections,
    image_id: int,
    label_to_category_id: dict[int, int],
) -> list[dict[str, Any]]:
    """Convert :class:`supervision.Detections` to COCO result dicts."""
    if len(detections) == 0:
        return []

    results: list[dict[str, Any]] = []
    for box, score, class_id in zip(
        detections.xyxy,
        detections.confidence,
        detections.class_id,
    ):
        label = int(class_id)
        if label not in label_to_category_id:
            continue
        x1, y1, x2, y2 = (float(v) for v in box)
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        if width < 1.0 or height < 1.0:
            continue
        results.append(
            {
                "image_id": int(image_id),
                "category_id": int(label_to_category_id[label]),
                "bbox": [x1, y1, width, height],
                "score": float(score),
            }
        )
    return results


def _mean_valid(values: np.ndarray) -> float:
    valid = values[values > -1]
    if valid.size == 0:
        return float("nan")
    return float(np.mean(valid))


def summarize_per_class_ap(coco_eval: COCOeval, coco_gt: COCO) -> list[dict[str, Any]]:
    """Return per-class AP metrics from a completed :class:`COCOeval`."""
    cat_ids = list(coco_eval.params.catIds)
    cat_id_to_name = {
        int(cat["id"]): str(cat["name"]) for cat in coco_gt.loadCats(cat_ids)
    }
    precisions = coco_eval.eval["precision"]
    recalls = coco_eval.eval["recall"]

    rows: list[dict[str, Any]] = []
    for index, cat_id in enumerate(cat_ids):
        cat_id = int(cat_id)
        ap = _mean_valid(precisions[:, :, index, 0, -1])
        ap50 = _mean_valid(precisions[0, :, index, 0, -1])
        ap75 = _mean_valid(precisions[5, :, index, 0, -1]) if precisions.shape[0] > 5 else float("nan")
        ar = _mean_valid(recalls[:, index, 0, -1])
        rows.append(
            {
                "category_id": cat_id,
                "category_name": cat_id_to_name.get(cat_id, str(cat_id)),
                "AP": ap,
                "AP50": ap50,
                "AP75": ap75,
                "AR100": ar,
            }
        )
    return rows


def _capture_coco_summary(coco_eval: COCOeval) -> dict[str, float]:
    stats = coco_eval.stats
    keys = [
        "AP",
        "AP50",
        "AP75",
        "AP_small",
        "AP_medium",
        "AP_large",
        "AR1",
        "AR10",
        "AR100",
        "AR_small",
        "AR_medium",
        "AR_large",
    ]
    return {key: float(value) for key, value in zip(keys, stats)}


def run_coco_eval(
    coco_gt: COCO,
    predictions: list[dict[str, Any]],
    *,
    iou_type: str = "bbox",
) -> tuple[COCOeval, dict[str, float], list[dict[str, Any]]]:
    """Evaluate *predictions* against *coco_gt* and return summary tables."""
    if not predictions:
        raise ValueError("No predictions to evaluate.")

    coco_dt = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    overall = _capture_coco_summary(coco_eval)
    per_class = summarize_per_class_ap(coco_eval, coco_gt)
    return coco_eval, overall, per_class


def save_eval_outputs(
    output_dir: Path,
    *,
    predictions: list[dict[str, Any]],
    overall: dict[str, float],
    per_class: list[dict[str, Any]],
    label_to_category_id: dict[int, int],
    class_names: list[str],
    eval_config: dict[str, Any],
) -> None:
    """Write predictions and metric tables under *output_dir*."""
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = output_dir / "predictions_coco.json"
    predictions_path.write_text(json.dumps(predictions, indent=2), encoding="utf-8")

    summary_path = output_dir / "metrics_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "overall": overall,
                "per_class": per_class,
                "label_to_category_id": {str(k): v for k, v in label_to_category_id.items()},
                "class_names": class_names,
                **eval_config,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    csv_path = output_dir / "metrics_per_class.csv"
    header = "category_id,category_name,AP,AP50,AP75,AR100"
    lines = [header]
    for row in per_class:
        lines.append(
            f"{row['category_id']},{row['category_name']},"
            f"{row['AP']:.6f},{row['AP50']:.6f},{row['AP75']:.6f},{row['AR100']:.6f}"
        )
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
