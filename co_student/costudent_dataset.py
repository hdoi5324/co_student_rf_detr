"""COCO dataset wrapper that emits raw / weak / strong Co-Student views."""

from __future__ import annotations

from typing import Any, Optional

from rfdetr.datasets.coco import CocoDetection

from co_student.triple_augment import TripleViewAugmentor, TripleViewSample


class CoStudentCocoDataset:
    """Wrap :class:`CocoDetection` with triple-view augmentation and transform matrices."""

    def __init__(
        self,
        base: CocoDetection,
        resolution: int,
        *,
        augment_seed: int = 0,
        flip_prob: float = 0.5,
    ) -> None:
        self.base = base
        self.augmentor = TripleViewAugmentor(
            resolution,
            flip_prob=flip_prob,
            seed=augment_seed,
        )

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> TripleViewSample:
        image, target = self.base[index]
        return self.augmentor(image, target, index)

    @property
    def coco(self):
        return self.base.coco
