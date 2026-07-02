"""Torchvision Faster R-CNN checkpoint predictor."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import supervision as sv
import torch
from PIL import Image
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from co_student.torchvision_coco import resize_for_inference


def _resolve_device(device: str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _open_rgb_image(image: str | Image.Image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    with Image.open(image) as opened:
        return opened.convert("RGB")


def build_faster_rcnn_model(
    num_foreground_classes: int,
    *,
    pretrained: bool = False,
) -> torch.nn.Module:
    """Build Faster R-CNN with a ResNet50-FPN backbone."""
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        num_foreground_classes + 1,
    )
    return model


def _predictions_to_detections(
    prediction: dict[str, torch.Tensor],
    *,
    threshold: float,
    label_offset: int,
) -> sv.Detections:
    boxes = prediction["boxes"].detach().cpu().numpy()
    scores = prediction["scores"].detach().cpu().numpy()
    labels = prediction["labels"].detach().cpu().numpy()

    keep = scores >= threshold
    if not np.any(keep):
        return sv.Detections.empty()

    class_ids = labels[keep].astype(np.int64) - label_offset
    return sv.Detections(
        xyxy=boxes[keep].astype(np.float32),
        confidence=scores[keep].astype(np.float32),
        class_id=class_ids,
    )


@dataclass
class FasterRCNNPredictor:
    """Wrap a torchvision Faster R-CNN model for shared inference utilities."""

    model: torch.nn.Module
    class_names: list[str]
    device: torch.device
    label_offset: int = 1

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Path | str,
        *,
        device: str | None = None,
    ) -> FasterRCNNPredictor:
        path = Path(checkpoint).expanduser().resolve()
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "model" not in payload:
            raise ValueError(f"Not a Faster R-CNN checkpoint: {path}")

        meta = payload.get("meta")
        if not isinstance(meta, dict):
            raise ValueError(f"Checkpoint missing meta block: {path}")

        class_names = [str(name) for name in meta.get("class_names", [])]
        num_classes = int(meta.get("num_classes", len(class_names)))
        label_offset = int(meta.get("label_offset", 1))
        if num_classes <= 0:
            raise ValueError(f"Invalid num_classes in checkpoint meta: {num_classes}")

        resolved_device = _resolve_device(device)
        model = build_faster_rcnn_model(num_classes, pretrained=False)
        model.load_state_dict(payload["model"])
        model.to(resolved_device)
        model.eval()

        if not class_names:
            class_names = [str(index) for index in range(num_classes)]

        return cls(
            model=model,
            class_names=class_names,
            device=resolved_device,
            label_offset=label_offset,
        )

    @classmethod
    def from_model(
        cls,
        model: torch.nn.Module,
        class_names: list[str],
        *,
        device: str | torch.device | None = None,
        label_offset: int = 1,
    ) -> FasterRCNNPredictor:
        resolved_device = device if isinstance(device, torch.device) else _resolve_device(
            str(device) if device is not None else None
        )
        model = model.to(resolved_device)
        model.eval()
        return cls(
            model=model,
            class_names=list(class_names),
            device=resolved_device,
            label_offset=label_offset,
        )

    def predict(
        self,
        image: str | Image.Image,
        *,
        threshold: float = 0.5,
    ) -> sv.Detections:
        rgb = _open_rgb_image(image)
        tensor = resize_for_inference(rgb).to(self.device)
        with torch.inference_mode():
            outputs = self.model([tensor])
        prediction = outputs[0]
        detections = _predictions_to_detections(
            prediction,
            threshold=threshold,
            label_offset=self.label_offset,
        )
        detections.metadata = {"source_image": np.asarray(rgb)}
        return detections

    def max_confidence(self, image: str | Image.Image) -> float:
        detections = self.predict(image, threshold=0.001)
        if len(detections) == 0 or detections.confidence is None:
            return 0.0
        return float(detections.confidence.max())


def save_faster_rcnn_checkpoint(
    path: Path | str,
    model: torch.nn.Module,
    *,
    class_names: list[str],
    epoch: int,
    extra_meta: dict[str, Any] | None = None,
) -> None:
    """Write an inference-ready Faster R-CNN checkpoint."""
    ckpt_path = Path(path).expanduser().resolve()
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {
        "model_type": "faster_rcnn",
        "backbone": "resnet50_fpn",
        "class_names": list(class_names),
        "num_classes": len(class_names),
        "label_offset": 1,
        "epoch": epoch,
    }
    if extra_meta:
        meta.update(extra_meta)
    torch.save({"model": model.state_dict(), "meta": meta, "epoch": epoch}, ckpt_path)
