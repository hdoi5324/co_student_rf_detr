"""Model-agnostic detection predictor protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import supervision as sv
import torch
from PIL import Image


@runtime_checkable
class DetectionPredictor(Protocol):
    """Run inference and return :class:`supervision.Detections`."""

    class_names: list[str]

    def predict(
        self,
        image: str | Image.Image,
        *,
        threshold: float = 0.5,
    ) -> sv.Detections: ...


def load_predictor(
    checkpoint: Path | str,
    *,
    device: str | None = None,
    optimize: bool = False,
) -> DetectionPredictor:
    """Load an RF-DETR or Faster R-CNN checkpoint for inference."""
    from co_student.predictors.faster_rcnn import FasterRCNNPredictor
    from co_student.predictors.rfdetr import RFDETRPredictor

    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        meta = payload.get("meta")
        if isinstance(meta, dict) and meta.get("model_type") == "faster_rcnn":
            return FasterRCNNPredictor.from_checkpoint(path, device=device)
        if "model_config" in payload:
            return RFDETRPredictor.from_checkpoint(path, device=device, optimize=optimize)

    try:
        return RFDETRPredictor.from_checkpoint(path, device=device, optimize=optimize)
    except Exception:
        return FasterRCNNPredictor.from_checkpoint(path, device=device)
