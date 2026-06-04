#!/usr/bin/env python3
"""Train RF-DETR with Co-Student for sparsely annotated object detection."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from rfdetr.training.trainer import build_trainer

from co_student.coco_eval_callback import CoStudentCOCOEvalCallback
from co_student.datamodule import CoStudentDataModule
from co_student.dataset import count_categories, split_paths_from_args
from co_student.mean_teacher_ema import CoStudentMeanTeacherCallback
from co_student.module import CoStudentConfig, CoStudentRFDETRModule
from co_student.train_config import CoStudentTrainConfig
from rfdetr.training.callbacks.coco_eval import COCOEvalCallback
from rfdetr.training.callbacks.ema import RFDETREMACallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    data = parser.add_argument_group("dataset")
    data.add_argument(
        "--dataset-dir",
        default=None,
        help="Roboflow/COCO dataset root (legacy layout). Optional when train paths are set.",
    )
    data.add_argument(
        "--train-image-dir",
        default=None,
        help="Directory containing training images",
    )
    data.add_argument(
        "--train-ann-dir",
        default=None,
        help="Directory containing the training COCO JSON (e.g. _annotations.coco.json)",
    )
    data.add_argument(
        "--train-ann-file",
        default=None,
        help="Path to training COCO JSON (overrides --train-ann-dir if both are set)",
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

    parser.add_argument("--output-dir", default="./output/costudent", help="Checkpoints and logs")
    parser.add_argument("--model", default="nano", choices=["nano", "small", "medium", "large"])
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Freeze DINOv2 backbone weights (ModelConfig freeze_encoder=True)",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lr-encoder", type=float, default=1.5e-5)
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
        default=0,
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
    return parser.parse_args()


MODEL_MAP = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
}


def _parse_epoch_list(value: str) -> list[int]:
    epochs = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not epochs:
        raise SystemExit("--lr-drop-epochs must list at least one epoch (e.g. 40,50)")
    if any(e < 0 for e in epochs):
        raise SystemExit("--lr-drop-epochs values must be non-negative integers")
    return epochs


def _ann_arg(file_arg: str | None, dir_arg: str | None) -> str | None:
    if file_arg:
        return file_arg
    return dir_arg


def _resolve_dataset_args(args: argparse.Namespace) -> tuple[str, object | None, object | None]:
    train_ann = _ann_arg(args.train_ann_file, args.train_ann_dir)
    val_ann = _ann_arg(args.val_ann_file, args.val_ann_dir)
    custom_train = args.train_image_dir is not None and train_ann is not None

    if custom_train:
        train_paths = split_paths_from_args(args.train_image_dir, train_ann)
        val_paths = None
        if args.val_image_dir and val_ann:
            val_paths = split_paths_from_args(args.val_image_dir, val_ann)
        dataset_dir = args.dataset_dir or str(train_paths.image_dir.parent)
        return dataset_dir, train_paths, val_paths

    if not args.dataset_dir:
        raise SystemExit(
            "Provide either --dataset-dir (Roboflow layout) or both "
            "--train-image-dir and --train-ann-dir/--train-ann-file."
        )
    return args.dataset_dir, None, None


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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir, train_paths, val_paths = _resolve_dataset_args(args)

    import rfdetr.variants as variants

    model_cls = getattr(variants, MODEL_MAP[args.model])
    wrapper = model_cls(freeze_encoder=args.freeze_encoder)

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
        aug_config={},
        augmentation_backend="cpu",
        dataset_file="roboflow" if train_paths is None else "coco",
        wandb=args.wandb,
        project=args.wandb_project if args.wandb else None,
        run=wandb_run_name if args.wandb else None,
        **({"weight_decay": args.weight_decay} if args.weight_decay is not None else {}),
    )

    costudent_config = CoStudentConfig(
        student_score_thresh=args.student_score_thresh,
        teacher_score_thresh=args.teacher_score_thresh,
        matching_iou_thresh=args.matching_iou_thresh,
    )

    train_ann_path = train_paths.ann_path if train_paths else None
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
                "model": args.model,
                "freeze_encoder": args.freeze_encoder,
                "num_classes": wrapper.model_config.num_classes,
                **train_config.model_dump(),
                **costudent_config.__dict__,
                "train_image_dir": str(train_paths.image_dir) if train_paths else None,
                "train_ann_file": str(train_paths.ann_path) if train_paths else None,
                "val_image_dir": str(val_paths.image_dir) if val_paths else None,
                "val_ann_file": str(val_paths.ann_path) if val_paths else None,
            },
        )

    trainer.fit(module, datamodule=datamodule, ckpt_path=train_config.resume)

    config_path = output_dir / "costudent_config.json"
    config_path.write_text(
        json.dumps(
            {
                "freeze_encoder": args.freeze_encoder,
                "train_config": train_config.model_dump(),
                "costudent_config": costudent_config.__dict__,
                "train_paths": (
                    {"image_dir": str(train_paths.image_dir), "ann_file": str(train_paths.ann_path)}
                    if train_paths
                    else None
                ),
                "val_paths": (
                    {"image_dir": str(val_paths.image_dir), "ann_file": str(val_paths.ann_path)}
                    if val_paths
                    else None
                ),
            },
            indent=2,
        )
    )
    print(f"Training complete. Config saved to {config_path}")


if __name__ == "__main__":
    main()
