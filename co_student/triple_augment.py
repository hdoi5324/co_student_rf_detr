"""Raw / weak / strong view augmentation with 3x3 transform matrices."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.v2 import ToDtype, ToImage

from co_student.geometric import (
    affine_matrix_from_cv2,
    append_transform,
    hflip_matrix,
    resize_matrix,
)
from rfdetr.datasets.transforms import Normalize

_to_image = ToImage()
_to_float = ToDtype(torch.float32, scale=True)
_normalize = Normalize()


@dataclass
class AugmentedView:
    image: torch.Tensor
    target: dict[str, Any]
    transform_matrix: np.ndarray
    image_size: tuple[int, int]


@dataclass
class TripleViewSample:
    raw: AugmentedView
    weak: AugmentedView
    strong: AugmentedView


class TripleViewAugmentor:
    """Three independent pipelines aligned with FCOS Co-Student (Raw / Nor / Str) from ."""

    def __init__(
        self,
        resolution: int,
        *,
        flip_prob: float = 0.5,
        seed: int = 0,
    ) -> None:
        self.resolution = resolution
        self.flip_prob = flip_prob
        self.seed = seed

    def __call__(self, image: Image.Image, target: dict[str, Any], index: int) -> TripleViewSample:
        rng = np.random.default_rng(self.seed + index)
        base = _pil_to_rgb_np(image)
        ann = _annotation_tensors(target)
        meta = _base_meta(target)

        raw_img, raw_ann, raw_m = _pipeline_raw(base, ann, self.resolution, rng)
        weak_img, weak_ann, weak_m = _pipeline_weak(base, ann, self.resolution, rng, self.flip_prob)
        strong_img, strong_ann, strong_m = _pipeline_strong(
            base, ann, self.resolution, rng, self.flip_prob
        )

        return TripleViewSample(
            raw=_finalize_view(raw_img, raw_ann, raw_m, meta),
            weak=_finalize_view(weak_img, weak_ann, weak_m, meta),
            strong=_finalize_view(strong_img, strong_ann, strong_m, meta),
        )


def _base_meta(target: dict[str, Any]) -> dict[str, Any]:
    return {
        k: v
        for k, v in target.items()
        if k not in {"boxes", "labels", "area", "iscrowd", "masks", "size"}
    }


@dataclass
class _AnnotationTensors:
    boxes: torch.Tensor
    labels: torch.Tensor
    area: torch.Tensor
    iscrowd: torch.Tensor


def _annotation_tensors(target: dict[str, Any]) -> _AnnotationTensors:
    return _AnnotationTensors(
        boxes=target["boxes"].clone(),
        labels=target["labels"].clone(),
        area=target["area"].clone(),
        iscrowd=target["iscrowd"].clone(),
    )


def _finalize_view(
    image: np.ndarray,
    ann: _AnnotationTensors,
    matrix: np.ndarray,
    meta: dict[str, Any],
) -> AugmentedView:
    h, w = image.shape[:2]
    target = copy.deepcopy(meta)
    target["boxes"] = ann.boxes
    target["labels"] = ann.labels
    target["iscrowd"] = ann.iscrowd
    xyxy = ann.boxes
    target["area"] = (xyxy[:, 2] - xyxy[:, 0]).clamp(min=0) * (xyxy[:, 3] - xyxy[:, 1]).clamp(min=0)
    target["orig_size"] = meta.get("orig_size", torch.tensor([h, w]))
    target["size"] = torch.tensor([h, w])

    pil = Image.fromarray(image)
    tensor = _to_float(_to_image(pil))
    tensor, target = _normalize(tensor, target)
    return AugmentedView(
        image=tensor,
        target=target,
        transform_matrix=matrix.astype(np.float32),
        image_size=(h, w),
    )


def _box_keep_mask(boxes: torch.Tensor, w: int, h: int) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.zeros(0, dtype=torch.bool)
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    valid &= (boxes[:, 2] - boxes[:, 0] >= 2) & (boxes[:, 3] - boxes[:, 1] >= 2)
    valid &= (boxes[:, 0] >= 0) & (boxes[:, 1] >= 0)
    valid &= (boxes[:, 2] <= w) & (boxes[:, 3] <= h)
    return valid


def _filter_ann(ann: _AnnotationTensors, w: int, h: int) -> _AnnotationTensors:
    keep = _box_keep_mask(ann.boxes, w, h)
    return _AnnotationTensors(
        boxes=ann.boxes[keep],
        labels=ann.labels[keep],
        area=ann.area[keep],
        iscrowd=ann.iscrowd[keep],
    )


def _pipeline_raw(
    image: np.ndarray,
    ann: _AnnotationTensors,
    resolution: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, _AnnotationTensors, np.ndarray]:
    del rng
    return _resize_only(image, ann, resolution)


def _pipeline_weak(
    image: np.ndarray,
    ann: _AnnotationTensors,
    resolution: int,
    rng: np.random.Generator,
    flip_prob: float,
) -> tuple[np.ndarray, _AnnotationTensors, np.ndarray]:
    image, ann, m = _resize_only(image, ann, resolution)
    if rng.random() < flip_prob:
        image, ann, m_flip = _horizontal_flip(image, ann)
        m = append_transform(m, m_flip)
    image = _color_jitter(image, rng, strength=0.4)
    return image, ann, m


def _pipeline_strong(
    image: np.ndarray,
    ann: _AnnotationTensors,
    resolution: int,
    rng: np.random.Generator,
    flip_prob: float,
) -> tuple[np.ndarray, _AnnotationTensors, np.ndarray]:
    image, ann, m = _resize_only(image, ann, resolution)
    if rng.random() < flip_prob:
        image, ann, m_flip = _horizontal_flip(image, ann)
        m = append_transform(m, m_flip)
    image = _color_jitter(image, rng, strength=0.8)
    image, ann, m_aff = _random_affine(image, ann, rng)
    m = append_transform(m, m_aff)
    image = _random_erase(image, rng)
    return image, ann, m


def _resize_only(
    image: np.ndarray,
    ann: _AnnotationTensors,
    resolution: int,
) -> tuple[np.ndarray, _AnnotationTensors, np.ndarray]:
    h0, w0 = image.shape[:2]
    image = cv2.resize(image, (resolution, resolution), interpolation=cv2.INTER_LINEAR)
    m = resize_matrix(w0, h0, resolution, resolution)
    ann = _transform_ann(ann, m, resolution, resolution)
    return image, ann, m


def _horizontal_flip(
    image: np.ndarray,
    ann: _AnnotationTensors,
) -> tuple[np.ndarray, _AnnotationTensors, np.ndarray]:
    image = np.ascontiguousarray(image[:, ::-1, :])
    w = image.shape[1]
    boxes = ann.boxes.clone()
    x0, x1 = boxes[:, 0].clone(), boxes[:, 2].clone()
    boxes[:, 0] = w - x1
    boxes[:, 2] = w - x0
    return image, _AnnotationTensors(boxes, ann.labels, ann.area, ann.iscrowd), hflip_matrix(w)


def _transform_ann(
    ann: _AnnotationTensors,
    matrix: np.ndarray,
    max_w: int,
    max_h: int,
) -> _AnnotationTensors:
    if ann.boxes.numel() == 0:
        return ann
    from co_student.geometric import bbox2points, points2bbox

    m = torch.tensor(matrix, dtype=torch.float32)
    points = bbox2points(ann.boxes)
    points_h = torch.cat([points, torch.ones(points.shape[0], 1)], dim=1)
    out = (m @ points_h.t()).t()
    out = out[:, :2] / out[:, 2:3]
    warped = points2bbox(out, max_w, max_h)
    return _filter_ann(
        _AnnotationTensors(warped, ann.labels, ann.area, ann.iscrowd),
        max_w,
        max_h,
    )


def _color_jitter(image: np.ndarray, rng: np.random.Generator, strength: float) -> np.ndarray:
    out = image.astype(np.float32)
    for _ in range(2):
        factor = 1.0 + (rng.random() - 0.5) * strength
        out = np.clip(out * factor, 0, 255)
    if rng.random() < 0.5:
        gray = out.mean(axis=2, keepdims=True)
        sat = 1.0 + (rng.random() - 0.5) * strength
        out = np.clip(gray + (out - gray) * sat, 0, 255)
    return out.astype(np.uint8)


def _random_affine(
    image: np.ndarray,
    ann: _AnnotationTensors,
    rng: np.random.Generator,
) -> tuple[np.ndarray, _AnnotationTensors, np.ndarray]:
    h, w = image.shape[:2]
    angle = float(rng.uniform(-15, 15))
    scale = float(rng.uniform(0.9, 1.1))
    tx = float(rng.uniform(-0.1, 0.1) * w)
    ty = float(rng.uniform(-0.1, 0.1) * h)
    center = (w / 2, h / 2)
    m2 = cv2.getRotationMatrix2D(center, angle, scale)
    m2[:, 2] += (tx, ty)
    image = cv2.warpAffine(
        image,
        m2,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(114, 114, 114),
    )
    ann = _transform_ann(ann, affine_matrix_from_cv2(m2), w, h)
    return image, ann, affine_matrix_from_cv2(m2)


def _random_erase(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = image.shape[:2]
    n = int(rng.integers(1, 6))
    out = image.copy()
    for _ in range(n):
        ew = int(rng.uniform(0.02, 0.2) * w)
        eh = int(rng.uniform(0.02, 0.2) * h)
        if ew < 1 or eh < 1:
            continue
        x0 = int(rng.integers(0, max(1, w - ew)))
        y0 = int(rng.integers(0, max(1, h - eh)))
        out[y0 : y0 + eh, x0 : x0 + ew] = 114
    return out


def _pil_to_rgb_np(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"))
