"""Pseudo-label denoising and merging (ported from Co-Student FCOS)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from torchvision.ops import batched_nms

from co_student.geometric import TransformMatrix, cvt_boxes_xyxy, cvt_masks
from rfdetr.utilities import box_ops


@dataclass
class Detections:
    """Absolute xyxy detections in pixel coordinates, optionally with instance masks."""

    boxes: torch.Tensor
    labels: torch.Tensor
    scores: torch.Tensor
    image_size: tuple[int, int]
    masks: Optional[torch.Tensor] = None  # (N, H, W) bool, aligned with image_size


def bbox_overlaps(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Pairwise IoU matrix, shape (N, M)."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
    iou, _ = box_ops.box_iou(boxes1, boxes2)
    return iou


def revision_pred(anchor: Detections, candidate: Detections) -> Detections:
    """Denoise *candidate* using higher-confidence *anchor* predictions (Revision_PRED)."""
    anchor = _detections_float32(anchor)
    candidate = _detections_float32(candidate)

    if anchor.boxes.numel() == 0 or candidate.boxes.numel() == 0:
        return _cat_detections(anchor, candidate)

    boxes1 = anchor.boxes
    scores1 = anchor.scores
    classes1 = anchor.labels
    masks1 = anchor.masks

    boxes2 = candidate.boxes.clone()
    scores2 = candidate.scores.clone()
    classes2 = candidate.labels.clone()
    masks2 = candidate.masks.clone() if candidate.masks is not None else None

    ious = bbox_overlaps(boxes1, boxes2)
    if ious.numel() == 0:
        return _cat_detections(anchor, candidate)

    while True:
        refine_gt_inds = (ious > 0.5).any(dim=0)
        if not refine_gt_inds.any():
            break

        refine_inds = ious.max(dim=0)[1]
        refine_pred_scores = scores1[refine_inds]
        need_refine = refine_pred_scores >= scores2

        lower_scores_inds = (~refine_gt_inds | ~need_refine) & refine_gt_inds
        lower_idx = torch.where(lower_scores_inds)[0]
        if lower_idx.numel() == 0:
            break

        index = (refine_inds[lower_scores_inds], lower_idx)
        ious.index_put_(index, ious.new_full((lower_idx.numel(),), 0.5))

    refine_gt_inds = (ious > 0.5).any(dim=0)
    if refine_gt_inds.any():
        refine_inds = ious.max(dim=0)[1][refine_gt_inds]
        gt_idx = torch.where(refine_gt_inds)[0]
        boxes2[gt_idx] = boxes1[refine_inds]
        classes2[gt_idx] = classes1[refine_inds]
        scores2[gt_idx] = scores1[refine_inds]
        if masks1 is not None:
            if masks2 is None:
                h, w = candidate.image_size
                masks2 = torch.zeros(
                    (boxes2.shape[0], h, w),
                    dtype=torch.bool,
                    device=boxes2.device,
                )
            masks2[gt_idx] = masks1[refine_inds]

    missing_inds = (ious < 0.5).all(dim=1)
    missing = Detections(
        boxes=boxes1[missing_inds],
        labels=classes1[missing_inds],
        scores=scores1[missing_inds],
        image_size=anchor.image_size,
        masks=masks1[missing_inds] if masks1 is not None else None,
    )
    refined = Detections(
        boxes=boxes2,
        labels=classes2,
        scores=scores2,
        image_size=candidate.image_size,
        masks=masks2,
    )
    return _cat_detections(missing, refined)


def cvt_detections(
    detections: Detections,
    source_tm: TransformMatrix,
    target_tm: TransformMatrix,
    target_size: tuple[int, int],
) -> Detections:
    """Warp detections from source view coordinates into the target view."""
    h, w = target_size
    boxes, keep = cvt_boxes_xyxy(detections.boxes, source_tm, target_tm, w, h)
    if boxes.numel() == 0:
        return Detections(
            boxes=boxes,
            labels=detections.labels.new_zeros((0,), dtype=torch.int64),
            scores=detections.scores.new_zeros((0,)),
            image_size=target_size,
            masks=_empty_masks(detections.masks, target_size),
        )

    masks = None
    if detections.masks is not None:
        masks = cvt_masks(detections.masks, source_tm, target_tm, target_size, keep)

    return _detections_float32(
        Detections(
            boxes=boxes,
            labels=detections.labels[keep],
            scores=detections.scores[keep],
            image_size=target_size,
            masks=masks,
        )
    )


def merge_ground_truth(
    sparse_target: dict[str, Any],
    predictions: Detections,
    iou_threshold: float,
    *,
    source_tm: Optional[TransformMatrix] = None,
    target_tm: Optional[TransformMatrix] = None,
) -> dict[str, Any]:
    """Append pseudo-labels that do not overlap sparse GT (merge_ground_truth)."""
    h, w = _image_hw(sparse_target)
    if source_tm is not None and target_tm is not None:
        predictions = cvt_detections(predictions, source_tm, target_tm, (h, w))

    gt_boxes = target_to_xyxy(sparse_target)
    gt_labels = sparse_target["labels"]
    gt_masks = sparse_target.get("masks")

    merged = dict(sparse_target)
    if predictions.boxes.numel() == 0:
        return merged

    if gt_boxes.numel() == 0:
        return detections_to_target(predictions, sparse_target)

    iou_matrix, _ = box_ops.box_iou(gt_boxes, predictions.boxes)
    class_match = gt_labels.reshape(-1, 1) == predictions.labels.reshape(1, -1)
    matched = (iou_matrix > iou_threshold) & class_match
    unlabeled = matched.sum(dim=0) == 0

    if not unlabeled.any():
        return merged

    pseudo_boxes = predictions.boxes[unlabeled]
    pseudo_labels = predictions.labels[unlabeled]
    pseudo_masks = predictions.masks[unlabeled] if predictions.masks is not None else None

    all_xyxy = torch.cat([gt_boxes, pseudo_boxes], dim=0)
    all_labels = torch.cat([gt_labels, pseudo_labels], dim=0)
    all_masks = None
    if gt_masks is not None or pseudo_masks is not None:
        if gt_masks is None:
            gt_masks = torch.zeros((gt_boxes.shape[0], h, w), dtype=torch.bool, device=gt_boxes.device)
        if pseudo_masks is None:
            pseudo_masks = torch.zeros(
                (pseudo_boxes.shape[0], h, w),
                dtype=torch.bool,
                device=pseudo_boxes.device,
            )
        all_masks = torch.cat([gt_masks, pseudo_masks], dim=0)

    pseudo = Detections(
        boxes=all_xyxy,
        labels=all_labels,
        scores=torch.ones(all_labels.shape, device=all_labels.device),
        image_size=(h, w),
        masks=all_masks,
    )
    return detections_to_target(pseudo, sparse_target)


def target_to_xyxy(target: dict[str, Any]) -> torch.Tensor:
    """Convert normalized cxcywh target boxes to absolute xyxy."""
    h, w = _image_hw(target)
    boxes = target["boxes"]
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 4))
    xyxy = box_ops.box_cxcywh_to_xyxy(boxes)
    scale = boxes.new_tensor([w, h, w, h])
    return xyxy * scale


def detections_to_target(detections: Detections, template: dict[str, Any]) -> dict[str, Any]:
    """Build an RF-DETR target dict from absolute xyxy detections."""
    h, w = detections.image_size
    skip_keys = {"boxes", "labels", "area", "iscrowd", "masks"}
    target = {k: v for k, v in template.items() if k not in skip_keys}
    if detections.boxes.numel() == 0:
        device = template["labels"].device if template["labels"].numel() else detections.boxes.device
        target["boxes"] = torch.zeros((0, 4), device=device)
        target["labels"] = torch.zeros((0,), dtype=torch.int64, device=device)
        target["area"] = torch.zeros((0,), device=device)
        target["iscrowd"] = torch.zeros((0,), dtype=torch.int64, device=device)
        if "masks" in template or detections.masks is not None:
            target["masks"] = torch.zeros((0, h, w), dtype=torch.bool, device=device)
        return target

    xyxy = detections.boxes
    cxcywh = box_ops.box_xyxy_to_cxcywh(xyxy)
    scale = xyxy.new_tensor([w, h, w, h])
    boxes = cxcywh / scale

    areas = (xyxy[:, 2] - xyxy[:, 0]).clamp(min=0) * (xyxy[:, 3] - xyxy[:, 1]).clamp(min=0)
    target["boxes"] = boxes
    target["labels"] = detections.labels.to(dtype=torch.int64)
    target["area"] = areas
    target["iscrowd"] = torch.zeros_like(target["labels"])
    if detections.masks is not None:
        target["masks"] = detections.masks.to(dtype=torch.bool)
    elif "masks" in template:
        target["masks"] = torch.zeros((0, h, w), dtype=torch.bool, device=boxes.device)
    return target


def result_to_detections(
    result: dict[str, torch.Tensor],
    image_size: tuple[int, int],
    score_threshold: float,
    nms_threshold: float,
    *,
    include_masks: bool = False,
) -> Detections:
    """Filter a single-image postprocess result into Detections."""
    boxes = result["boxes"]
    scores = result["scores"]
    labels = result["labels"]
    raw_masks = result.get("masks") if include_masks else None

    keep = scores >= score_threshold
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    if raw_masks is not None:
        raw_masks = raw_masks[keep]
        if raw_masks.ndim == 4 and raw_masks.shape[1] == 1:
            raw_masks = raw_masks.squeeze(1)
        raw_masks = raw_masks > 0

    if boxes.numel() == 0:
        return Detections(
            boxes=boxes.new_zeros((0, 4)),
            labels=labels.new_zeros((0,), dtype=torch.int64),
            scores=scores.new_zeros((0,)),
            image_size=image_size,
            masks=_empty_masks(raw_masks, image_size),
        )

    keep = batched_nms(boxes, scores, labels, nms_threshold)
    masks = raw_masks[keep] if raw_masks is not None else None
    return Detections(
        boxes=boxes[keep],
        labels=labels[keep],
        scores=scores[keep],
        image_size=image_size,
        masks=masks,
    )


def outputs_for_postprocess(outputs: dict[str, Any]) -> dict[str, Any]:
    """Strip aux outputs and densify sparse training-time ``pred_masks`` dicts."""
    core = {k: v for k, v in outputs.items() if k != "aux_outputs"}
    pred_masks = core.get("pred_masks")
    if isinstance(pred_masks, dict):
        spatial = pred_masks["spatial_features"]
        queries = pred_masks["query_features"]
        bias = pred_masks["bias"]
        core["pred_masks"] = torch.einsum("bchw,bnc->bnhw", spatial, queries) + bias
    return core


def outputs_to_detections(
    outputs: dict[str, torch.Tensor],
    postprocess: torch.nn.Module,
    image_size: tuple[int, int],
    score_threshold: float,
    nms_threshold: float,
    *,
    include_masks: bool = False,
) -> Detections:
    """Decode model outputs into filtered absolute xyxy detections."""
    h, w = image_size
    orig_sizes = torch.tensor([[h, w]], device=outputs["pred_logits"].device)
    results = postprocess(outputs_for_postprocess(outputs), orig_sizes)[0]
    return result_to_detections(
        results,
        image_size,
        score_threshold,
        nms_threshold,
        include_masks=include_masks,
    )


def _detections_float32(detections: Detections) -> Detections:
    """Cast box/score tensors to float32 (teacher under AMP may be bfloat16)."""
    masks = detections.masks
    if masks is not None:
        masks = masks.bool()
    return Detections(
        boxes=detections.boxes.float(),
        labels=detections.labels,
        scores=detections.scores.float(),
        image_size=detections.image_size,
        masks=masks,
    )


def _cat_detections(a: Detections, b: Detections) -> Detections:
    if a.boxes.numel() == 0:
        return b
    if b.boxes.numel() == 0:
        return a

    masks = None
    if a.masks is not None or b.masks is not None:
        h, w = a.image_size
        device = a.boxes.device
        a_masks = a.masks if a.masks is not None else torch.zeros((a.boxes.shape[0], h, w), dtype=torch.bool, device=device)
        b_masks = b.masks if b.masks is not None else torch.zeros((b.boxes.shape[0], h, w), dtype=torch.bool, device=device)
        masks = torch.cat([a_masks, b_masks], dim=0)

    return Detections(
        boxes=torch.cat([a.boxes, b.boxes], dim=0),
        labels=torch.cat([a.labels, b.labels], dim=0),
        scores=torch.cat([a.scores, b.scores], dim=0),
        image_size=a.image_size,
        masks=masks,
    )


def _empty_masks(masks: Optional[torch.Tensor], image_size: tuple[int, int]) -> Optional[torch.Tensor]:
    if masks is None:
        return None
    h, w = image_size
    return masks.new_zeros((0, h, w), dtype=torch.bool)


def _image_hw(target: dict[str, Any]) -> tuple[int, int]:
    size = target.get("size")
    if size is None:
        orig = target["orig_size"]
        h, w = int(orig[0]), int(orig[1])
    else:
        h, w = int(size[0]), int(size[1])
    return h, w
