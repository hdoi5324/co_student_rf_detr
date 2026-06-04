"""Extended TrainConfig with multi-epoch LR step decay."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable

from pydantic import Field
from rfdetr.config import TrainConfig

if TYPE_CHECKING:
    from rfdetr.config import TrainConfig as TrainConfigType


class CoStudentTrainConfig(TrainConfig):
    """TrainConfig with optional LR drops at multiple epoch boundaries."""

    lr_drop_epochs: list[int] = Field(
        default_factory=list,
        description=(
            "Epoch indices at which to multiply the LR by lr_drop_gamma. "
            "When empty, a single drop at lr_drop is used (RF-DETR default)."
        ),
    )
    lr_drop_gamma: float = Field(
        default=0.1,
        gt=0.0,
        lt=1.0,
        description="LR multiplier applied at each epoch in lr_drop_epochs (or at lr_drop).",
    )


def resolve_lr_drop_epochs(train_config: TrainConfigType) -> list[int]:
    """Return sorted unique epoch indices where step decay applies."""
    extra = getattr(train_config, "lr_drop_epochs", None)
    if extra:
        return sorted({int(e) for e in extra})
    return [int(train_config.lr_drop)]


def resolve_lr_drop_gamma(train_config: TrainConfigType) -> float:
    return float(getattr(train_config, "lr_drop_gamma", 0.1))


def build_lr_lambda(
    train_config: TrainConfigType,
    *,
    total_steps: int,
    steps_per_epoch: int,
) -> Callable[[int], float]:
    """Build LambdaLR multiplier matching RF-DETR warmup + step/cosine schedules."""
    tc = train_config
    warmup_steps = int(steps_per_epoch * tc.warmup_epochs)
    drop_epochs = resolve_lr_drop_epochs(tc)
    gamma = resolve_lr_drop_gamma(tc)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        if tc.lr_scheduler == "cosine":
            progress = float(current_step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            return tc.lr_min_factor + (1 - tc.lr_min_factor) * 0.5 * (
                1 + math.cos(math.pi * progress)
            )
        multiplier = 1.0
        for drop_epoch in drop_epochs:
            if current_step >= drop_epoch * steps_per_epoch:
                multiplier *= gamma
        return multiplier

    return lr_lambda
