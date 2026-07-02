"""Tests for mask-aware Co-Student pseudo labels."""

from __future__ import annotations

import torch

from co_student.geometric import cvt_masks, hflip_matrix, resize_matrix
from co_student.pseudo_labels import (
    Detections,
    cvt_detections,
    merge_ground_truth,
    outputs_for_postprocess,
    revision_pred,
)


def _sample_mask(h: int, w: int) -> torch.Tensor:
    mask = torch.zeros((h, w), dtype=torch.bool)
    mask[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = True
    return mask


def test_outputs_for_postprocess_densifies_sparse_masks():
    spatial = torch.randn(2, 4, 8, 8)
    queries = torch.randn(2, 10, 4)
    bias = torch.tensor(0.1)
    outputs = {
        "pred_logits": torch.randn(2, 10, 3),
        "pred_boxes": torch.rand(2, 10, 4),
        "pred_masks": {
            "spatial_features": spatial,
            "query_features": queries,
            "bias": bias,
        },
        "aux_outputs": [{"pred_logits": torch.randn(2, 10, 3)}],
    }
    core = outputs_for_postprocess(outputs)
    assert "aux_outputs" not in core
    assert isinstance(core["pred_masks"], torch.Tensor)
    assert core["pred_masks"].shape == (2, 10, 8, 8)
    expected = torch.einsum("bchw,bnc->bnhw", spatial, queries) + bias
    assert torch.allclose(core["pred_masks"], expected)


def test_cvt_masks_resize():
    h0, w0, h1, w1 = 100, 200, 512, 512
    masks = _sample_mask(h0, w0).unsqueeze(0)
    identity = resize_matrix(w0, h0, w0, h0)  # original -> original (masks live in source view)
    resize = resize_matrix(w0, h0, w1, h1)
    keep = torch.tensor([True])
    warped = cvt_masks(masks, identity, resize, (h1, w1), keep)
    assert warped.shape == (1, h1, w1)
    assert warped.any()


def test_cvt_detections_warps_masks():
    h, w = 512, 512
    boxes = torch.tensor([[128.0, 128.0, 384.0, 384.0]])
    masks = _sample_mask(h, w).unsqueeze(0)
    detections = Detections(
        boxes=boxes,
        labels=torch.tensor([0]),
        scores=torch.tensor([0.9]),
        image_size=(h, w),
        masks=masks,
    )
    m_flip = hflip_matrix(w)
    out = cvt_detections(detections, m_flip, m_flip, (h, w))
    assert out.masks is not None
    assert out.masks.shape == (1, h, w)
    assert out.masks.any()


def test_revision_pred_copies_masks():
    h, w = 256, 256
    anchor = Detections(
        boxes=torch.tensor([[50.0, 50.0, 150.0, 150.0]]),
        labels=torch.tensor([1]),
        scores=torch.tensor([0.9]),
        image_size=(h, w),
        masks=_sample_mask(h, w).unsqueeze(0),
    )
    candidate = Detections(
        boxes=torch.tensor([[52.0, 52.0, 148.0, 148.0]]),
        labels=torch.tensor([1]),
        scores=torch.tensor([0.6]),
        image_size=(h, w),
        masks=torch.zeros((1, h, w), dtype=torch.bool),
    )
    out = revision_pred(anchor, candidate)
    assert out.masks is not None
    assert out.masks[0].any()


def test_merge_ground_truth_appends_pseudo_masks():
    h, w = 256, 256
    gt_mask = torch.zeros((1, h, w), dtype=torch.bool)
    gt_mask[0, 10:40, 10:40] = True
    sparse = {
        "boxes": torch.tensor([[0.05, 0.05, 0.15, 0.15]]),
        "labels": torch.tensor([0]),
        "area": torch.tensor([100.0]),
        "iscrowd": torch.tensor([0]),
        "masks": gt_mask,
        "size": torch.tensor([h, w]),
        "orig_size": torch.tensor([h, w]),
    }
    pseudo_mask = torch.zeros((1, h, w), dtype=torch.bool)
    pseudo_mask[0, 120:180, 120:180] = True
    predictions = Detections(
        boxes=torch.tensor([[120.0, 120.0, 180.0, 180.0]]),
        labels=torch.tensor([0]),
        scores=torch.tensor([0.85]),
        image_size=(h, w),
        masks=pseudo_mask,
    )
    merged = merge_ground_truth(sparse, predictions, iou_threshold=0.5)
    assert merged["masks"].shape[0] == 2
    assert merged["labels"].shape[0] == 2
    assert merged["masks"][1].any()
