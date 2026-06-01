"""Collate function for triple-view Co-Student batches."""

from __future__ import annotations

from functools import partial
from typing import Any

from rfdetr.utilities.tensors import nested_tensor_from_tensor_list

from co_student.triple_augment import TripleViewSample


def collate_costudent_batch(
    batch: list[TripleViewSample],
    block_size: int | None = None,
) -> dict[str, Any]:
    """Collate triple views into NestedTensors and per-image transform matrices."""
    raw_images, raw_targets = [], []
    weak_images, weak_targets = [], []
    strong_images, strong_targets = [], []
    transforms = []

    for sample in batch:
        raw_images.append(sample.raw.image)
        raw_targets.append(sample.raw.target)
        weak_images.append(sample.weak.image)
        weak_targets.append(sample.weak.target)
        strong_images.append(sample.strong.image)
        strong_targets.append(sample.strong.target)
        transforms.append(
            {
                "raw": sample.raw.transform_matrix,
                "weak": sample.weak.transform_matrix,
                "strong": sample.strong.transform_matrix,
            }
        )

    nest = partial(nested_tensor_from_tensor_list, block_size=block_size)
    return {
        "raw": (nest(raw_images), tuple(raw_targets)),
        "weak": (nest(weak_images), tuple(weak_targets)),
        "strong": (nest(strong_images), tuple(strong_targets)),
        "transforms": transforms,
    }
