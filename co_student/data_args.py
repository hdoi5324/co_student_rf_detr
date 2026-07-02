"""Shared dataset CLI arguments and path resolution for training scripts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from co_student.coco_merge import merge_coco_sources, resolve_train_sources, sources_to_config
from co_student.dataset import CocoSplitPaths, split_paths_from_args


@dataclass(frozen=True)
class ResolvedDataset:
    """Resolved train/val COCO paths for a training run."""

    dataset_dir: str
    train_paths: CocoSplitPaths
    val_paths: CocoSplitPaths | None
    train_sources_config: list[dict[str, str]]


def add_dataset_args(parser: argparse.ArgumentParser) -> None:
    """Register dataset path arguments shared by training entry points."""
    data = parser.add_argument_group("dataset")
    data.add_argument(
        "--train-manifest",
        default=None,
        help="JSON manifest listing train_sources (image_dir + ann_file per source)",
    )
    data.add_argument(
        "--train-source",
        action="append",
        default=None,
        metavar="SPEC",
        help=(
            "Training source as image_dir:ann_path or name:image_dir:ann_path. "
            "Repeat for multiple sources."
        ),
    )
    data.add_argument(
        "--train-merge-cache-dir",
        default=None,
        help="Cache directory for merged training COCO (default: <output-dir>/merged_train)",
    )
    data.add_argument(
        "--val-image-dir",
        default=None,
        help="Directory containing validation images",
    )
    data.add_argument(
        "--val-ann-dir",
        default=None,
        help="Directory containing the validation COCO JSON",
    )
    data.add_argument(
        "--val-ann-file",
        default=None,
        help="Path to validation COCO JSON (overrides --val-ann-dir if both are set)",
    )
    data.add_argument(
        "--keep-unannotated",
        action="store_true",
        help="Keep training images with no instance annotations (default: drop them)",
    )


def _ann_arg(file_arg: str | None, dir_arg: str | None) -> str | None:
    if file_arg:
        return file_arg
    return dir_arg


def resolve_dataset_paths(
    args: argparse.Namespace,
    *,
    output_dir: Path,
) -> ResolvedDataset:
    """Merge training sources and resolve optional validation split paths."""
    try:
        train_sources = resolve_train_sources(
            manifest=args.train_manifest,
            train_source_args=args.train_source,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    merge_cache_dir = Path(args.train_merge_cache_dir or output_dir / "merged_train")
    train_paths = merge_coco_sources(train_sources, merge_cache_dir)
    train_sources_config = sources_to_config(train_sources)

    val_ann = _ann_arg(args.val_ann_file, args.val_ann_dir)
    val_paths = None
    if args.val_image_dir and val_ann:
        val_paths = split_paths_from_args(args.val_image_dir, val_ann)

    dataset_dir = str(train_paths.image_dir.parent)
    return ResolvedDataset(
        dataset_dir=dataset_dir,
        train_paths=train_paths,
        val_paths=val_paths,
        train_sources_config=train_sources_config,
    )
