"""Pre-slice COCO training data with SAHI for RF-DETR windowed training."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from sahi.slicing import slice_coco

from co_student.dataset import CocoSplitPaths


@dataclass(frozen=True)
class SahiSliceConfig:
    """Parameters for SAHI COCO slicing."""

    slice_size: int
    overlap_ratio: float = 0.2
    min_area_ratio: float = 0.1
    ignore_negative_samples: bool = True


def _manifest_path(cache_dir: Path) -> Path:
    return cache_dir / "sahi_manifest.json"


def _ann_path(cache_dir: Path, config: SahiSliceConfig) -> Path:
    name = _output_ann_stem(config)
    return cache_dir / "images" / f"{name}_coco.json"


def _output_ann_stem(config: SahiSliceConfig) -> str:
    neg = "no_neg" if config.ignore_negative_samples else "with_neg"
    return f"sliced_{config.slice_size}_o{config.overlap_ratio}_ma{config.min_area_ratio}_{neg}"


def _source_fingerprint(train_paths: CocoSplitPaths) -> dict[str, object]:
    ann_stat = train_paths.ann_path.stat()
    return {
        "image_dir": str(train_paths.image_dir),
        "ann_path": str(train_paths.ann_path),
        "ann_mtime_ns": ann_stat.st_mtime_ns,
        "ann_size": ann_stat.st_size,
    }


def _cache_is_valid(cache_dir: Path, train_paths: CocoSplitPaths, config: SahiSliceConfig) -> bool:
    manifest_file = _manifest_path(cache_dir)
    ann_file = _ann_path(cache_dir, config)
    if not manifest_file.is_file() or not ann_file.is_file():
        return False
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    expected = {
        "source": _source_fingerprint(train_paths),
        "slice_config": asdict(config),
    }
    return manifest == expected


def prepare_sliced_train_paths(
    train_paths: CocoSplitPaths,
    *,
    slice_size: int,
    overlap_ratio: float = 0.2,
    min_area_ratio: float = 0.1,
    ignore_negative_samples: bool = True,
    cache_dir: Path,
) -> CocoSplitPaths:
    """Slice training images/annotations and return paths to the cached COCO split.

    Slices are written to ``cache_dir/images`` with a matching COCO JSON. Re-slicing
    runs only when the source annotation or slice parameters change.
    """
    config = SahiSliceConfig(
        slice_size=slice_size,
        overlap_ratio=overlap_ratio,
        min_area_ratio=min_area_ratio,
        ignore_negative_samples=ignore_negative_samples,
    )
    cache_dir = cache_dir.expanduser().resolve()
    images_dir = cache_dir / "images"
    ann_file = _ann_path(cache_dir, config)

    if _cache_is_valid(cache_dir, train_paths, config):
        return CocoSplitPaths(image_dir=images_dir, ann_path=ann_file)

    images_dir.mkdir(parents=True, exist_ok=True)
    ann_stem = _output_ann_stem(config)
    slice_coco(
        coco_annotation_file_path=str(train_paths.ann_path),
        image_dir=str(train_paths.image_dir),
        output_coco_annotation_file_name=ann_stem,
        output_dir=str(images_dir),
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap_ratio,
        overlap_width_ratio=overlap_ratio,
        min_area_ratio=min_area_ratio,
        ignore_negative_samples=ignore_negative_samples,
        verbose=False,
    )

    if not ann_file.is_file():
        raise FileNotFoundError(f"SAHI slice did not produce expected annotation file: {ann_file}")

    manifest = {
        "source": _source_fingerprint(train_paths),
        "slice_config": asdict(config),
        "output_ann": str(ann_file),
    }
    _manifest_path(cache_dir).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return CocoSplitPaths(image_dir=images_dir, ann_path=ann_file)
