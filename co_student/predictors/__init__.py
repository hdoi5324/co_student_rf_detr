"""Checkpoint predictors shared by eval and visualization scripts."""

from co_student.predictors.base import DetectionPredictor, load_predictor
from co_student.predictors.faster_rcnn import FasterRCNNPredictor
from co_student.predictors.rfdetr import RFDETRPredictor

__all__ = [
    "DetectionPredictor",
    "FasterRCNNPredictor",
    "RFDETRPredictor",
    "load_predictor",
]
