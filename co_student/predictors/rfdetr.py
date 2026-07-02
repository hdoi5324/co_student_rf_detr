"""RF-DETR checkpoint predictor."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import supervision as sv
import torch
from PIL import Image
from rfdetr import RFDETR

if TYPE_CHECKING:
    from co_student.sahi_inference import SahiPredictor


@dataclass
class RFDETRPredictor:
    """Wrap a loaded :class:`RFDETR` model for shared inference utilities."""

    model: RFDETR

    @property
    def class_names(self) -> list[str]:
        return list(self.model.class_names)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Path | str,
        *,
        device: str | None = None,
        optimize: bool = False,
    ) -> RFDETRPredictor:
        del device  # RFDETR selects device internally.
        model = RFDETR.from_checkpoint(checkpoint)
        if optimize:
            model.optimize_for_inference()
        return cls(model=model)

    def predict(
        self,
        image: str | Image.Image,
        *,
        threshold: float = 0.5,
    ) -> sv.Detections:
        return self.model.predict(image, threshold=threshold)

    def max_confidence(self, image: str | Image.Image) -> float:
        detections = self.model.predict(image, threshold=0.001)
        if len(detections) == 0 or detections.confidence is None:
            return 0.0
        return float(detections.confidence.max())

    def build_sahi_predictor(
        self,
        *,
        threshold: float,
        config: object | None = None,
        device: str | None = None,
    ) -> SahiPredictor:
        from co_student.sahi_inference import SahiInferenceConfig, SahiPredictor

        resolved = config if isinstance(config, SahiInferenceConfig) else SahiInferenceConfig()
        return SahiPredictor.from_rfdetr(
            self.model,
            threshold=threshold,
            config=resolved,
            device=device,
        )
