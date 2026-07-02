"""Merge multiple COCO training sources into one cached split."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from co_student.dataset import CocoSplitPaths, resolve_coco_ann_path

_MANIFEST_NAME = "merge_manifest.json"
_ANN_NAME = "instances_train.json"
_IMAGES_DIR_NAME = "images"


@dataclass(frozen=True)
class CocoTrainSource:
    """One COCO training source (image directory + annotation file)."""

    name: str
    image_dir: Path
    ann_path: Path


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip())
    return slug.strip("_") or "source"


def parse_train_source_arg(value: str, *, index: int) -> CocoTrainSource:
    """Parse ``image_dir:ann_path`` or ``name:image_dir:ann_path``."""
    parts = value.split(":")
    if len(parts) == 2:
        image_dir, ann_path = parts
        name = _slugify(Path(ann_path).stem) or f"source_{index}"
    elif len(parts) >= 3:
        name = _slugify(parts[0])
        image_dir = parts[1]
        ann_path = ":".join(parts[2:])
    else:
        raise ValueError(
            "Each --train-source must be image_dir:ann_path or name:image_dir:ann_path"
        )
    return CocoTrainSource(
        name=name,
        image_dir=Path(image_dir).expanduser().resolve(),
        ann_path=resolve_coco_ann_path(ann_path),
    )


def load_train_manifest(path: str | Path) -> list[CocoTrainSource]:
    """Load training sources from a JSON manifest file."""
    manifest_path = Path(path).expanduser().resolve()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))

    if isinstance(data, dict):
        entries = data.get("train_sources", data.get("sources"))
    else:
        entries = data
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"No train_sources found in manifest: {manifest_path}")

    sources: list[CocoTrainSource] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Manifest entry {index} must be an object")
        image_dir = entry.get("image_dir")
        ann_file = entry.get("ann_file") or entry.get("ann_path")
        if not image_dir or not ann_file:
            raise ValueError(
                f"Manifest entry {index} requires image_dir and ann_file/ann_path"
            )
        name = _slugify(str(entry.get("name") or Path(str(ann_file)).stem or f"source_{index}"))
        sources.append(
            CocoTrainSource(
                name=name,
                image_dir=Path(image_dir).expanduser().resolve(),
                ann_path=resolve_coco_ann_path(ann_file),
            )
        )
    return sources


def resolve_train_sources(
    *,
    manifest: str | Path | None,
    train_source_args: list[str] | None,
) -> list[CocoTrainSource]:
    """Resolve training sources from a manifest and/or repeated CLI arguments."""
    sources: list[CocoTrainSource] = []
    if manifest:
        sources.extend(load_train_manifest(manifest))
    if train_source_args:
        offset = len(sources)
        sources.extend(
            parse_train_source_arg(arg, index=offset + index)
            for index, arg in enumerate(train_source_args)
        )
    if not sources:
        raise ValueError("Provide at least one training source via --train-manifest or --train-source")
    return sources


def _source_fingerprint(source: CocoTrainSource) -> dict[str, object]:
    ann_stat = source.ann_path.stat()
    return {
        "name": source.name,
        "image_dir": str(source.image_dir),
        "ann_path": str(source.ann_path),
        "ann_mtime_ns": ann_stat.st_mtime_ns,
        "ann_size": ann_stat.st_size,
    }


def _manifest_path(cache_dir: Path) -> Path:
    return cache_dir / _MANIFEST_NAME


def _cache_is_valid(cache_dir: Path, sources: list[CocoTrainSource]) -> bool:
    manifest_file = _manifest_path(cache_dir)
    ann_file = cache_dir / _ANN_NAME
    if not manifest_file.is_file() or not ann_file.is_file():
        return False
    try:
        saved = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    expected = {"sources": [_source_fingerprint(source) for source in sources]}
    return saved == expected


def _build_unified_categories(
    coco_dicts: list[dict],
) -> tuple[list[dict], list[dict[str, int]]]:
    """Return unified categories and per-source original-id to unified-id maps."""
    by_name: dict[str, dict[str, str]] = {}
    for coco in coco_dicts:
        for cat in coco.get("categories", []):
            key = str(cat["name"]).strip().lower()
            if key not in by_name:
                by_name[key] = {
                    "name": str(cat["name"]),
                    "supercategory": str(cat.get("supercategory", "")),
                }

    categories: list[dict] = []
    name_to_unified_id: dict[str, int] = {}
    for index, key in enumerate(sorted(by_name), start=1):
        categories.append(
            {
                "id": index,
                "name": by_name[key]["name"],
                "supercategory": by_name[key]["supercategory"],
            }
        )
        name_to_unified_id[key] = index

    per_source_maps: list[dict[int, int]] = []
    for coco in coco_dicts:
        mapping: dict[int, int] = {}
        for cat in coco.get("categories", []):
            key = str(cat["name"]).strip().lower()
            if key not in name_to_unified_id:
                raise ValueError(
                    f"Category {cat['name']!r} missing from unified category table"
                )
            mapping[int(cat["id"])] = name_to_unified_id[key]
        per_source_maps.append(mapping)
    return categories, per_source_maps


def _link_or_copy_image(source_path: Path, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() or dest_path.is_symlink():
        dest_path.unlink()
    try:
        dest_path.symlink_to(source_path.resolve())
    except OSError:
        import shutil

        shutil.copy2(source_path, dest_path)


def _merge_coco_dicts(sources: list[CocoTrainSource], *, images_dir: Path) -> dict:
    coco_dicts = [
        json.loads(source.ann_path.read_text(encoding="utf-8")) for source in sources
    ]
    categories, category_maps = _build_unified_categories(coco_dicts)

    merged_images: list[dict] = []
    merged_annotations: list[dict] = []
    used_file_names: set[str] = set()
    next_image_id = 1
    next_ann_id = 1

    for source, coco, category_map in zip(sources, coco_dicts, category_maps):
        if not source.image_dir.is_dir():
            raise FileNotFoundError(f"Image directory does not exist: {source.image_dir}")

        images_by_id = {int(image["id"]): image for image in coco.get("images", [])}
        old_to_new_image_id: dict[int, int] = {}

        for old_image_id in sorted(images_by_id):
            image = dict(images_by_id[old_image_id])
            file_name = Path(str(image["file_name"])).name
            merged_file_name = file_name
            if merged_file_name in used_file_names:
                merged_file_name = f"{source.name}__{file_name}"

            source_image_path = source.image_dir / file_name
            if not source_image_path.is_file():
                alt_path = source.image_dir / str(image["file_name"])
                if alt_path.is_file():
                    source_image_path = alt_path
                else:
                    raise FileNotFoundError(
                        f"Image not found for source {source.name}: {source_image_path}"
                    )

            dest_image_path = images_dir / merged_file_name
            _link_or_copy_image(source_image_path, dest_image_path)

            image["id"] = next_image_id
            image["file_name"] = merged_file_name
            merged_images.append(image)
            old_to_new_image_id[old_image_id] = next_image_id
            used_file_names.add(merged_file_name)
            next_image_id += 1

        for ann in coco.get("annotations", []):
            old_image_id = int(ann["image_id"])
            if old_image_id not in old_to_new_image_id:
                continue
            new_ann = dict(ann)
            new_ann["id"] = next_ann_id
            new_ann["image_id"] = old_to_new_image_id[old_image_id]
            new_ann["category_id"] = category_map[int(ann["category_id"])]
            merged_annotations.append(new_ann)
            next_ann_id += 1

    return {
        "info": {"description": "Merged Co-Student RF-DETR training split"},
        "licenses": coco_dicts[0].get("licenses", []),
        "categories": categories,
        "images": merged_images,
        "annotations": merged_annotations,
    }


def merge_coco_sources(
    sources: list[CocoTrainSource],
    cache_dir: Path,
) -> CocoSplitPaths:
    """Merge *sources* into a cached COCO split and return its paths."""
    if not sources:
        raise ValueError("At least one training source is required")

    cache_dir = cache_dir.expanduser().resolve()
    images_dir = cache_dir / _IMAGES_DIR_NAME
    ann_path = cache_dir / _ANN_NAME

    if _cache_is_valid(cache_dir, sources):
        return CocoSplitPaths(image_dir=images_dir, ann_path=ann_path)

    cache_dir.mkdir(parents=True, exist_ok=True)
    if images_dir.exists():
        for path in images_dir.iterdir():
            if path.is_file() or path.is_symlink():
                path.unlink()
    else:
        images_dir.mkdir(parents=True, exist_ok=True)

    merged = _merge_coco_dicts(sources, images_dir=images_dir)

    ann_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    manifest = {
        "sources": [_source_fingerprint(source) for source in sources],
        "num_images": len(merged["images"]),
        "num_annotations": len(merged["annotations"]),
        "categories": merged["categories"],
        "output_ann": str(ann_path),
    }
    _manifest_path(cache_dir).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return CocoSplitPaths(image_dir=images_dir, ann_path=ann_path)


def sources_to_config(sources: list[CocoTrainSource]) -> list[dict[str, str]]:
    return [
        {
            "name": source.name,
            "image_dir": str(source.image_dir),
            "ann_file": str(source.ann_path),
        }
        for source in sources
    ]
