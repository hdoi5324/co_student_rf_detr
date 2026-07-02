#!/usr/bin/env python3
"""Train RF-DETR with Co-Student for sparsely annotated object detection or segmentation."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from rfdetr.training.trainer import build_trainer

from co_student.checkpoint_callback import EnrichInferenceCheckpointsCallback
from co_student.checkpoint_resume import (
    checkpoint_needs_weights_only_resume,
    load_checkpoint_dict,
    load_weights_only_checkpoint,
)
from co_student.coco_eval_callback import CoStudentCOCOEvalCallback
from co_student.data_args import add_dataset_args, resolve_dataset_paths
from co_student.datamodule import CoStudentDataModule
from co_student.dataset import count_categories, summarize_coco_split
from co_student.mean_teacher_ema import CoStudentMeanTeacherCallback
from co_student.module import CoStudentConfig, CoStudentRFDETRModule
from co_student.sahi_slice import prepare_sliced_train_paths
from co_student.train_config import CoStudentTrainConfig
from rfdetr.training.callbacks.coco_eval import COCOEvalCallback
from rfdetr.training.callbacks.ema import RFDETREMACallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_dataset_args(parser)
    model = parser.add_argument_group("model")
    model.add_argument(
        "--task",
        default="detection",
        choices=["detection", "segmentation"],
        help="Train a bbox detector (default) or an RF-DETR Seg model with mask supervision",
    )
    model.add_argument("--model", default="nano", choices=["nano", "small", "medium", "large"])
    model.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Freeze DINOv2 backbone weights (ModelConfig freeze_encoder=True)",
    )
    model.add_argument(
        "--focal-alpha",
        type=float,
        default=0.25,
        help="Focal loss alpha for classification loss and Hungarian matching (default: 0.25)",
    )

    pseudo = parser.add_argument_group("Co-Student pseudo labels")
    pseudo.add_argument(
        "--pseudo-labels",
        dest="pseudo_labels",
        action="store_true",
        default=None,
        help="Enable bbox pseudo-label merging across weak/strong views (default: on for detection)",
    )
    pseudo.add_argument(
        "--no-pseudo-labels",
        dest="pseudo_labels",
        action="store_false",
        help="Train on sparse GT only (default for segmentation; required until mask pseudo-labels exist)",
    )

    parser.add_argument("--output-dir", default="./outputs/costudent", help="Checkpoints and logs")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-encoder", type=float, default=1.5e-4)
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="AdamW weight decay (default: TrainConfig default, 1e-4)",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=float,
        default=3.0,
        help="Linear LR warmup length in epochs",
    )
    lr_sched = parser.add_argument_group("learning rate schedule")
    lr_sched.add_argument(
        "--lr-scheduler",
        default="step",
        choices=["step", "cosine"],
        help="LR schedule after warmup (default: step)",
    )
    lr_sched.add_argument(
        "--lr-drop",
        type=int,
        default=None,
        help="Single epoch for one LR step (×0.1). Default: 100 (no drop within 100 epochs). "
        "Ignored when --lr-drop-epochs is set.",
    )
    lr_sched.add_argument(
        "--lr-drop-epochs",
        default=None,
        metavar="EPOCHS",
        help="Comma-separated epochs to multiply LR by --lr-drop-gamma (e.g. 40,50)",
    )
    lr_sched.add_argument(
        "--lr-drop-gamma",
        type=float,
        default=0.1,
        help="LR multiplier at each drop epoch (default: 0.1)",
    )
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume")
    parser.add_argument(
        "--resume-weights-only",
        action="store_true",
        help=(
            "Load model weights, EMA, and epoch counter only (fresh optimizer). "
            "Auto-enabled when the checkpoint has no optimizer/LR scheduler state."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--student-score-thresh", type=float, default=0.5)
    parser.add_argument("--teacher-score-thresh", type=float, default=0.6)
    parser.add_argument("--matching-iou-thresh", type=float, default=0.5)
    parser.add_argument("--no-ema", action="store_true", help="Disable EMA teacher")
    ema = parser.add_argument_group("teacher EMA (MeanTeacher)")
    ema.add_argument(
        "--ema-decay",
        type=float,
        default=0.999,
        help="EMA momentum when using RF-DETR EMA (ignored with --mean-teacher)",
    )
    ema.add_argument(
        "--ema-tau",
        type=int,
        default=0,
        help="RF-DETR EMA warm-up steps (0 = constant decay; ignored with --mean-teacher)",
    )
    ema.add_argument(
        "--ema-update-interval",
        type=int,
        default=1,
        help="Update teacher every N optimizer steps (CoStudent interval=1)",
    )
    ema.add_argument(
        "--mean-teacher",
        action="store_true",
        help="Use CoStudent/cvpods MeanTeacher momentum schedule instead of RF-DETR EMA tau ramp",
    )
    ema.add_argument(
        "--ema-warm-up",
        type=int,
        default=100,
        help="MeanTeacher warm_up (only with --mean-teacher; 0 matches CoStudent FCOS config)",
    )

    logging = parser.add_argument_group("logging")
    logging.add_argument("--wandb", action="store_true", help="Log metrics and config to Weights & Biases")
    logging.add_argument(
        "--wandb-project",
        default="co-student-rf-detr",
        help="W&B project name (used with --wandb)",
    )
    logging.add_argument(
        "--wandb-run",
        default=None,
        help="W&B run name (default: unique name from model + UTC timestamp)",
    )

    sahi = parser.add_argument_group("SAHI training slices")
    sahi.add_argument(
        "--sahi-slice",
        action="store_true",
        help="Pre-slice training COCO into model-resolution windows before training",
    )
    sahi.add_argument(
        "--sahi-overlap",
        type=float,
        default=0.2,
        help="Fractional overlap between adjacent train tiles (default: 0.2)",
    )
    sahi.add_argument(
        "--sahi-min-area-ratio",
        type=float,
        default=0.1,
        help="Drop clipped boxes retaining less than this fraction of area (default: 0.1)",
    )
    sahi.add_argument(
        "--sahi-keep-negative-samples",
        action="store_true",
        help="Include sliced tiles with no annotations (default: drop empty tiles)",
    )
    sahi.add_argument(
        "--sahi-cache-dir",
        default=None,
        help="Directory for cached sliced train data (default: <output-dir>/sahi_train)",
    )
    return parser.parse_args()


MODEL_MAP_DETECTION = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
}

MODEL_MAP_SEGMENTATION = {
    "nano": "RFDETRSegNano",
    "small": "RFDETRSegSmall",
    "medium": "RFDETRSegMedium",
    "large": "RFDETRSegLarge",
}


def _resolve_pseudo_labels(args: argparse.Namespace) -> bool:
    if args.pseudo_labels is not None:
        return args.pseudo_labels
    return True


def _validate_task_args(args: argparse.Namespace, use_pseudo_labels: bool) -> None:
    del args, use_pseudo_labels


def _parse_epoch_list(value: str) -> list[int]:
    epochs = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not epochs:
        raise SystemExit("--lr-drop-epochs must list at least one epoch (e.g. 40,50)")
    if any(e < 0 for e in epochs):
        raise SystemExit("--lr-drop-epochs values must be non-negative integers")
    return epochs


def _log_wandb_config(trainer, config: dict) -> None:
    """Push hyperparameters and run metadata to the active W&B run."""
    try:
        from pytorch_lightning.loggers import WandbLogger
    except ImportError:
        return

    loggers = trainer.loggers
    if not loggers:
        return
    if not isinstance(loggers, list):
        loggers = [loggers]
    for logger in loggers:
        if isinstance(logger, WandbLogger) and logger.experiment is not None:
            logger.experiment.config.update(config, allow_val_change=True)


def _align_num_classes(wrapper, train_ann_path: Path | None, dataset_dir: str) -> None:
    if train_ann_path is not None:
        num_classes = count_categories(train_ann_path)
        if wrapper.model_config.num_classes != num_classes:
            wrapper.model_config.num_classes = num_classes
            if hasattr(wrapper, "model") and wrapper.model is not None:
                wrapper.model.args.num_classes = num_classes
        return
    wrapper._align_num_classes_from_dataset(dataset_dir)


def main() -> None:
    args = parse_args()
    use_pseudo_labels = _resolve_pseudo_labels(args)
    _validate_task_args(args, use_pseudo_labels)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved = resolve_dataset_paths(args, output_dir=output_dir)
    dataset_dir = resolved.dataset_dir
    train_paths = resolved.train_paths
    val_paths = resolved.val_paths
    train_sources_config = resolved.train_sources_config
    print(f"Merged training data: {train_paths.ann_path}")
    print(f"  images: {train_paths.image_dir}")
    print(f"  sources: {len(train_sources_config)}")
    train_summary = summarize_coco_split(train_paths.ann_path)
    print(
        f"  COCO JSON: {train_summary.total_images} images "
        f"({train_summary.annotated_images} annotated, "
        f"{train_summary.unannotated_images} unannotated)"
    )
    if args.keep_unannotated:
        print("  keeping unannotated training images")
    else:
        print("  dropping unannotated training images (default)")

    import rfdetr.variants as variants

    model_map = MODEL_MAP_DETECTION if args.task == "detection" else MODEL_MAP_SEGMENTATION
    model_cls = getattr(variants, model_map[args.model])
    wrapper = model_cls(freeze_encoder=args.freeze_encoder)

    sahi_slice_config: dict[str, object] | None = None
    if args.sahi_slice:
        slice_size = int(wrapper.model_config.resolution)
        cache_dir = Path(args.sahi_cache_dir or output_dir / "sahi_train")
        train_paths = prepare_sliced_train_paths(
            train_paths,
            slice_size=slice_size,
            overlap_ratio=args.sahi_overlap,
            min_area_ratio=args.sahi_min_area_ratio,
            ignore_negative_samples=not args.sahi_keep_negative_samples,
            cache_dir=cache_dir,
        )
        sahi_slice_config = {
            "enabled": True,
            "slice_size": slice_size,
            "overlap_ratio": args.sahi_overlap,
            "min_area_ratio": args.sahi_min_area_ratio,
            "keep_negative_samples": args.sahi_keep_negative_samples,
            "cache_dir": str(cache_dir),
            "train_image_dir": str(train_paths.image_dir),
            "train_ann_file": str(train_paths.ann_path),
        }
        print(
            f"SAHI train slicing: {slice_size}x{slice_size} tiles, "
            f"overlap={args.sahi_overlap}, "
            f"keep_negative_samples={args.sahi_keep_negative_samples}"
        )
        print(f"  images: {train_paths.image_dir}")
        print(f"  annotations: {train_paths.ann_path}")

    if args.task == "segmentation":
        print(
            "Segmentation task: RF-DETR Seg with mask-aware Co-Student pseudo labels "
            "(box IoU matching, raster mask warp across views)."
        )
    elif not use_pseudo_labels:
        print("Detection task with pseudo-label merging disabled; training on sparse GT only.")

    if args.wandb_run:
        wandb_run_name = args.wandb_run
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        wandb_run_name = f"{args.model}-costudent-{stamp}-{uuid4().hex[:6]}"

    lr_drop_epochs: list[int] = []
    if args.lr_drop_epochs:
        lr_drop_epochs = _parse_epoch_list(args.lr_drop_epochs)
    lr_drop = args.lr_drop if args.lr_drop is not None else (max(lr_drop_epochs) if lr_drop_epochs else 100)

    train_config = CoStudentTrainConfig(
        dataset_dir=dataset_dir,
        output_dir=str(output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        lr_encoder=args.lr_encoder,
        warmup_epochs=args.warmup_epochs,
        lr_scheduler=args.lr_scheduler,
        lr_drop=lr_drop,
        lr_drop_epochs=lr_drop_epochs,
        lr_drop_gamma=args.lr_drop_gamma,
        resume=args.resume,
        seed=args.seed,
        use_ema=not args.no_ema,
        ema_decay=args.ema_decay,
        ema_tau=args.ema_tau,
        ema_update_interval=args.ema_update_interval,
        focal_alpha=args.focal_alpha,
        aug_config={},
        augmentation_backend="cpu",
        dataset_file="coco",
        wandb=args.wandb,
        project=args.wandb_project if args.wandb else None,
        run=wandb_run_name if args.wandb else None,
        **({"weight_decay": args.weight_decay} if args.weight_decay is not None else {}),
    )

    costudent_config = CoStudentConfig(
        student_score_thresh=args.student_score_thresh,
        teacher_score_thresh=args.teacher_score_thresh,
        matching_iou_thresh=args.matching_iou_thresh,
        use_pseudo_labels=use_pseudo_labels,
    )

    train_ann_path = train_paths.ann_path
    _align_num_classes(wrapper, train_ann_path, dataset_dir)

    module = CoStudentRFDETRModule(
        model_config=wrapper.model_config,
        train_config=train_config,
        costudent_config=costudent_config,
    )
    datamodule = CoStudentDataModule(
        model_config=wrapper.model_config,
        train_config=train_config,
        train_paths=train_paths,
        val_paths=val_paths,
        keep_unannotated=args.keep_unannotated,
    )

    trainer = build_trainer(train_config, wrapper.model_config)

    trainer.callbacks = [
        CoStudentCOCOEvalCallback(
            max_dets=train_config.eval_max_dets,
            segmentation=wrapper.model_config.segmentation_head,
            eval_interval=train_config.eval_interval,
            log_per_class_metrics=train_config.log_per_class_metrics,
        )
        if isinstance(cb, COCOEvalCallback)
        else cb
        for cb in trainer.callbacks
    ]
    trainer.callbacks.append(EnrichInferenceCheckpointsCallback(output_dir))

    if args.mean_teacher and not args.no_ema:
        trainer.callbacks = [
            cb
            for cb in trainer.callbacks
            if not isinstance(cb, RFDETREMACallback)
        ]
        trainer.callbacks.append(
            CoStudentMeanTeacherCallback(
                momentum=args.ema_decay,
                warm_up=args.ema_warm_up,
                update_interval_steps=args.ema_update_interval,
            )
        )

    if args.wandb:
        _log_wandb_config(
            trainer,
            {
                "task": args.task,
                "model": args.model,
                "freeze_encoder": args.freeze_encoder,
                "use_pseudo_labels": use_pseudo_labels,
                "keep_unannotated": args.keep_unannotated,
                "num_classes": wrapper.model_config.num_classes,
                **train_config.model_dump(),
                **costudent_config.__dict__,
                "train_image_dir": str(train_paths.image_dir),
                "train_ann_file": str(train_paths.ann_path),
                "train_sources": train_sources_config,
                "val_image_dir": str(val_paths.image_dir) if val_paths else None,
                "val_ann_file": str(val_paths.ann_path) if val_paths else None,
                **({"sahi_slice": sahi_slice_config} if sahi_slice_config else {}),
            },
        )

    resume_path = train_config.resume
    weights_only = args.resume_weights_only
    if resume_path:
        peek = load_checkpoint_dict(resume_path)
        if checkpoint_needs_weights_only_resume(peek):
            if not weights_only:
                print(
                    "Checkpoint has no optimizer/LR scheduler state; "
                    "using weights-only resume (fresh optimizer, same epoch counter)."
                )
            weights_only = True

    if resume_path and weights_only:
        load_weights_only_checkpoint(resume_path, module, trainer)
        trainer.fit(module, datamodule=datamodule, ckpt_path=None)
    else:
        trainer.fit(module, datamodule=datamodule, ckpt_path=resume_path)

    class_names = getattr(datamodule, "class_names", None)
    config_path = output_dir / "costudent_config.json"
    config_path.write_text(
        json.dumps(
            {
                "task": args.task,
                "model": args.model,
                "model_name": model_map[args.model],
                "freeze_encoder": args.freeze_encoder,
                "use_pseudo_labels": use_pseudo_labels,
                "keep_unannotated": args.keep_unannotated,
                "model_config": wrapper.model_config.model_dump(),
                "class_names": list(class_names) if class_names else None,
                "train_config": train_config.model_dump(),
                "costudent_config": costudent_config.__dict__,
                "train_paths": {
                    "image_dir": str(train_paths.image_dir),
                    "ann_file": str(train_paths.ann_path),
                },
                "train_sources": train_sources_config,
                "val_paths": (
                    {"image_dir": str(val_paths.image_dir), "ann_file": str(val_paths.ann_path)}
                    if val_paths
                    else None
                ),
                "sahi_slice": sahi_slice_config,
            },
            indent=2,
        )
    )
    print(f"Training complete. Config saved to {config_path}")


if __name__ == "__main__":
    main()
