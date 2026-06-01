"""Geometric transforms and box warping (FCOS Co-Student convention)."""

from __future__ import annotations

from typing import Union

import numpy as np
import torch

TransformMatrix = Union[np.ndarray, torch.Tensor]


def identity_matrix(dtype=np.float32) -> np.ndarray:
    return np.eye(3, dtype=dtype)


def compose_matrix(*matrices: np.ndarray) -> np.ndarray:
    """Compose transforms applied in order (maps original image coords -> current view)."""
    out = identity_matrix()
    for m in matrices:
        out = m @ out
    return out.astype(np.float32)


def append_transform(base: np.ndarray, new: np.ndarray) -> np.ndarray:
    """Apply *new* after *base* (both map original -> their stage)."""
    return compose_matrix(base, new)


def resize_matrix(w0: float, h0: float, w1: float, h1: float) -> np.ndarray:
    sx, sy = w1 / w0, h1 / h0
    return np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float32)


def hflip_matrix(width: float) -> np.ndarray:
    return np.array([[-1, 0, width], [0, 1, 0], [0, 0, 1]], dtype=np.float32)


def affine_matrix_from_cv2(m2x3: np.ndarray) -> np.ndarray:
    m = identity_matrix()
    m[:2, :] = m2x3
    return m


def bbox2points(box: torch.Tensor) -> torch.Tensor:
    min_x, min_y, max_x, max_y = torch.split(box[:, :4], [1, 1, 1, 1], dim=1)
    return torch.cat([min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y], dim=1).reshape(-1, 2)


def points2bbox(point: torch.Tensor, max_w: float, max_h: float) -> torch.Tensor:
    point = point.reshape(-1, 4, 2)
    if point.shape[0] == 0:
        return point.new_zeros(0, 4)
    min_xy = point.min(dim=1)[0]
    max_xy = point.max(dim=1)[0]
    xmin = min_xy[:, 0].clamp(min=0, max=max_w)
    ymin = min_xy[:, 1].clamp(min=0, max=max_h)
    xmax = max_xy[:, 0].clamp(min=0, max=max_w)
    ymax = max_xy[:, 1].clamp(min=0, max=max_h)
    min_xy = torch.stack([xmin, ymin], dim=1)
    max_xy = torch.stack([xmax, ymax], dim=1)
    return torch.cat([min_xy, max_xy], dim=1)


def warp_matrix(source_tm: TransformMatrix, target_tm: TransformMatrix) -> np.ndarray:
    """Return M such that p_target = M @ p_source (homogeneous)."""
    s = _to_numpy(source_tm)
    t = _to_numpy(target_tm)
    return t @ np.linalg.inv(s)


def cvt_boxes_xyxy(
    boxes: torch.Tensor,
    source_tm: TransformMatrix,
    target_tm: TransformMatrix,
    max_w: float,
    max_h: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Warp absolute xyxy boxes from source view coordinates to target view.

    Returns:
        warped boxes and a boolean ``keep`` mask into the input *boxes*.
    """
    if boxes.numel() == 0:
        return boxes, torch.zeros(0, dtype=torch.bool, device=boxes.device)
    m_t = torch.tensor(warp_matrix(source_tm, target_tm), device=boxes.device, dtype=torch.float32)
    points = bbox2points(boxes[:, :4])
    points_h = torch.cat([points, points.new_ones(points.shape[0], 1)], dim=1)
    target_points = (m_t @ points_h.t()).t()
    target_points = target_points[:, :2] / target_points[:, 2:3]
    warped = points2bbox(target_points, max_w, max_h)
    min_x, min_y, max_x, max_y = warped.unbind(dim=1)
    keep = (min_x < max_x) & (min_y < max_y)
    return warped[keep], keep


def _to_numpy(matrix: TransformMatrix) -> np.ndarray:
    if isinstance(matrix, torch.Tensor):
        return matrix.detach().cpu().numpy().astype(np.float32)
    return np.asarray(matrix, dtype=np.float32)
