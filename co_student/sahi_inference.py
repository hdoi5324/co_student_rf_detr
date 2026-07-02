"""SAHI sliced inference helpers for RF-DETR checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import supervision as sv
import torch
from PIL import Image
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.prediction import PredictionResult

if TYPE_CHECKING:
    from rfdetr.detr import RFDETR


@dataclass(frozen=True)
class SahiInferenceConfig:
    """Tile geometry and merge settings for SAHI inference."""

    slice_size: int | None = None
    overlap_ratio: float = 0.2
    postprocess_type: str = "GREEDYNMM"
    postprocess_match_metric: str = "IOS"
    postprocess_match_threshold: float = 0.5
    postprocess_class_agnostic: bool = False
    perform_standard_pred: bool = False


def resolve_slice_size(model: RFDETR, slice_size: int | None = None) -> int:
    """Return explicit slice size or the model's native input resolution."""
    if slice_size is not None:
        return int(slice_size)
    return int(model.model_config.resolution)


def _resolve_device(device: str | None) -> str:
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_category_mapping(class_names: list[str]) -> dict[str, str]:
    return {str(index): name for index, name in enumerate(class_names)}


def build_sahi_detection_model(
    model: RFDETR,
    *,
    threshold: float,
    device: str | None = None,
) -> Any:
    """Wrap a loaded :class:`RFDETR` instance for SAHI tile inference."""
    return AutoDetectionModel.from_pretrained(
        model_type="roboflow",
        model=model,
        confidence_threshold=threshold,
        category_mapping=build_category_mapping(model.class_names),
        device=_resolve_device(device),
    )


def _image_to_rgb_numpy(image: str | Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        return np.ascontiguousarray(arr)
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))
    with Image.open(image) as opened:
        return np.asarray(opened.convert("RGB"))


def prediction_result_to_supervision(
    result: PredictionResult,
    *,
    source_image: np.ndarray,
) -> sv.Detections:
    """Convert merged SAHI predictions to :class:`supervision.Detections`."""
    if not result.object_prediction_list:
        return sv.Detections.empty()

    xyxy_rows: list[list[float]] = []
    scores: list[float] = []
    class_ids: list[int] = []
    masks: list[np.ndarray] = []
    has_mask = False

    for prediction in result.object_prediction_list:
        shifted = prediction.get_shifted_object_prediction()
        xyxy_rows.append(list(shifted.bbox.to_xyxy()))
        scores.append(float(shifted.score.value))
        class_ids.append(int(shifted.category.id))
        if shifted.mask is not None:
            has_mask = True
            masks.append(shifted.mask.bool_mask)

    detections = sv.Detections(
        xyxy=np.asarray(xyxy_rows, dtype=np.float32),
        confidence=np.asarray(scores, dtype=np.float32),
        class_id=np.asarray(class_ids, dtype=np.int64),
        mask=np.asarray(masks, dtype=bool) if has_mask else None,
    )
    detections.metadata = {"source_image": source_image}
    return detections


@dataclass
class SahiPredictor:
    """Reusable SAHI predictor for an RF-DETR checkpoint."""

    rfdetr_model: RFDETR
    detection_model: Any
    config: SahiInferenceConfig
    slice_size: int
    threshold: float

    @classmethod
    def from_rfdetr(
        cls,
        model: RFDETR,
        *,
        threshold: float,
        config: SahiInferenceConfig | None = None,
        device: str | None = None,
    ) -> SahiPredictor:
        resolved = config or SahiInferenceConfig()
        slice_size = resolve_slice_size(model, resolved.slice_size)
        return cls(
            rfdetr_model=model,
            detection_model=build_sahi_detection_model(
                model,
                threshold=threshold,
                device=device,
            ),
            config=resolved,
            slice_size=slice_size,
            threshold=threshold,
        )

    def predict(self, image: str | Image.Image | np.ndarray) -> sv.Detections:
        rgb = _image_to_rgb_numpy(image)
        result = get_sliced_prediction(
            image=rgb,
            detection_model=self.detection_model,
            slice_height=self.slice_size,
            slice_width=self.slice_size,
            overlap_height_ratio=self.config.overlap_ratio,
            overlap_width_ratio=self.config.overlap_ratio,
            auto_slice_resolution=False,
            perform_standard_pred=self.config.perform_standard_pred,
            postprocess_type=self.config.postprocess_type,
            postprocess_match_metric=self.config.postprocess_match_metric,
            postprocess_match_threshold=self.config.postprocess_match_threshold,
            postprocess_class_agnostic=self.config.postprocess_class_agnostic,
            verbose=0,
        )
        return prediction_result_to_supervision(result, source_image=rgb)

    def max_confidence(self, image: str | Image.Image | np.ndarray) -> float:
        """Return the highest merged confidence at a very low threshold."""
        old_threshold = float(self.detection_model.confidence_threshold)
        try:
            self.detection_model.confidence_threshold = 0.001
            detections = self.predict(image)
            if len(detections) == 0 or detections.confidence is None:
                return 0.0
            return float(detections.confidence.max())
        finally:
            self.detection_model.confidence_threshold = old_threshold
