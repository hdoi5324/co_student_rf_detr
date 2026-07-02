#!/usr/bin/env python3
"""Train a vanilla torchvision Faster R-CNN baseline on COCO data."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import torch
from torch.optim.lr_scheduler import MultiStepLR

from co_student.data_args import add_dataset_args, resolve_dataset_paths
from co_student.eval_runner import evaluate_faster_rcnn_model
from co_student.predictors.faster_rcnn import build_faster_rcnn_model, save_faster_rcnn_checkpoint
from co_student.torchvision_coco import (
    FasterRCNNDataConfig,
    build_faster_rcnn_dataloaders,
)
from co_student.dataset import summarize_coco_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_dataset_args(parser)

    train = parser.add_argument_group("training")
    train.add_argument("--output-dir", default="./outputs/faster_rcnn", help="Checkpoints and logs")
    train.add_argument("--epochs", type=int, default=26, help="Training epochs (torchvision default: 26)")
    train.add_argument("--batch-size", type=int, default=2, help="Per-step batch size")
    train.add_argument("--num-workers", type=int, default=4, help="DataLoader worker processes")
    train.add_argument(
        "--lr",
        type=float,
        default=None,
        help="SGD learning rate (default: 0.02 * batch_size / 16)",
    )
    train.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    train.add_argument("--weight-decay", type=float, default=1e-4, help="SGD weight decay")
    train.add_argument(
        "--lr-milestones",
        default=None,
        metavar="EPOCHS",
        help="Comma-separated LR drop epochs (default: 16,22 scaled to --epochs)",
    )
    train.add_argument("--lr-gamma", type=float, default=0.1, help="LR multiplier at each milestone")
    train.add_argument("--seed", type=int, default=42, help="Random seed")
    train.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Train from scratch instead of COCO-pretrained Faster R-CNN weights",
    )
    train.add_argument(
        "--eval-interval",
        type=int,
        default=1,
        help="Run validation COCO eval every N epochs (default: 1)",
    )
    train.add_argument("--resume", default=None, help="Resume from a Faster R-CNN checkpoint .pth")

    logging = parser.add_argument_group("logging")
    logging.add_argument("--wandb", action="store_true", help="Log metrics to Weights & Biases")
    logging.add_argument("--wandb-project", default="co-student-rf-detr", help="W&B project name")
    logging.add_argument("--wandb-run", default=None, help="W&B run name")
    return parser.parse_args()


def _parse_milestones(value: str) -> list[int]:
    milestones = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not milestones:
        raise SystemExit("--lr-milestones must list at least one epoch")
    return milestones


def _default_milestones(epochs: int) -> list[int]:
    if epochs <= 1:
        return []
    first = max(1, int(round(epochs * 16 / 26)))
    second = max(first + 1, int(round(epochs * 22 / 26)))
    if second >= epochs:
        second = epochs - 1
    return [first, second]


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_model(num_classes: int, *, pretrained: bool) -> torch.nn.Module:
    return build_faster_rcnn_model(num_classes, pretrained=pretrained)


def _train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    running_loss = 0.0
    num_batches = 0

    for images, targets in loader:
        images = [image.to(device) for image in images]
        targets = [{key: value.to(device) for key, value in target.items()} for target in targets]

        loss_dict = model(images, targets)
        losses = sum(loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += float(losses.detach().cpu())
        num_batches += 1

    mean_loss = running_loss / max(num_batches, 1)
    print(f"Epoch {epoch}: train_loss={mean_loss:.4f}")
    return mean_loss


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved = resolve_dataset_paths(args, output_dir=output_dir)
    train_paths = resolved.train_paths
    val_paths = resolved.val_paths

    print(f"Training data: {train_paths.ann_path}")
    print(f"  images: {train_paths.image_dir}")
    train_summary = summarize_coco_split(train_paths.ann_path)
    print(
        f"  COCO JSON: {train_summary.total_images} images "
        f"({train_summary.annotated_images} annotated, "
        f"{train_summary.unannotated_images} unannotated)"
    )
    if val_paths is not None:
        print(f"Validation data: {val_paths.ann_path}")
        print(f"  images: {val_paths.image_dir}")
        val_summary = summarize_coco_split(val_paths.ann_path)
        print(
            f"  COCO JSON: {val_summary.total_images} images "
            f"({val_summary.annotated_images} annotated, "
            f"{val_summary.unannotated_images} unannotated)"
        )
    else:
        print("No validation split provided; skipping COCO eval.")

    if args.keep_unannotated:
        print("  keeping unannotated training images")
    else:
        print("  dropping unannotated training images (default)")

    train_loader, _, class_names = build_faster_rcnn_dataloaders(
        train_paths,
        val_paths,
        config=FasterRCNNDataConfig(batch_size=args.batch_size, num_workers=args.num_workers),
        keep_unannotated=args.keep_unannotated,
    )
    print(f"  dataloader: {len(train_loader.dataset)} train samples")
    num_classes = len(class_names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(num_classes, pretrained=not args.no_pretrained).to(device)

    start_epoch = 0
    best_ap = -1.0
    if args.resume:
        resume_path = Path(args.resume).expanduser().resolve()
        payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "model" not in payload:
            raise SystemExit(f"Not a Faster R-CNN checkpoint: {resume_path}")
        model.load_state_dict(payload["model"])
        start_epoch = int(payload.get("epoch", 0)) + 1
        meta = payload.get("meta", {})
        if isinstance(meta, dict):
            class_names = [str(name) for name in meta.get("class_names", class_names)]
        print(f"Resumed from {resume_path} at epoch {start_epoch}")

    lr = args.lr if args.lr is not None else 0.02 * args.batch_size / 16
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)

    milestones = (
        _parse_milestones(args.lr_milestones)
        if args.lr_milestones
        else _default_milestones(args.epochs)
    )
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=args.lr_gamma) if milestones else None

    wandb_run = None
    if args.wandb:
        import wandb

        run_name = args.wandb_run
        if not run_name:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            run_name = f"faster-rcnn-{stamp}-{uuid4().hex[:6]}"
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "model_type": "faster_rcnn",
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": lr,
                "milestones": milestones,
                "num_classes": num_classes,
                "pretrained": not args.no_pretrained,
                "keep_unannotated": args.keep_unannotated,
                "train_ann_file": str(train_paths.ann_path),
                "val_ann_file": str(val_paths.ann_path) if val_paths else None,
            },
        )

    history: list[dict[str, float | int]] = []

    for epoch in range(start_epoch, args.epochs):
        train_loss = _train_one_epoch(model, train_loader, optimizer, device, epoch)

        metrics: dict[str, float] = {"train_loss": train_loss}
        if val_paths is not None and (epoch + 1) % max(args.eval_interval, 1) == 0:
            overall = evaluate_faster_rcnn_model(
                model,
                class_names,
                image_dir=val_paths.image_dir,
                ann_path=val_paths.ann_path,
                device=device,
            )
            metrics.update(overall)
            print(
                f"Epoch {epoch}: val_AP={overall['AP']:.4f} "
                f"AP50={overall['AP50']:.4f} AP75={overall['AP75']:.4f}"
            )
            if overall["AP"] > best_ap:
                best_ap = overall["AP"]
                save_faster_rcnn_checkpoint(
                    output_dir / "checkpoint_best.pth",
                    model,
                    class_names=class_names,
                    epoch=epoch,
                    extra_meta={
                        "val_ap": overall["AP"],
                        "train_paths": {
                            "image_dir": str(train_paths.image_dir),
                            "ann_file": str(train_paths.ann_path),
                        },
                        "val_paths": {
                            "image_dir": str(val_paths.image_dir),
                            "ann_file": str(val_paths.ann_path),
                        },
                    },
                )

        save_faster_rcnn_checkpoint(
            output_dir / "checkpoint_last.pth",
            model,
            class_names=class_names,
            epoch=epoch,
            extra_meta={
                "train_paths": {
                    "image_dir": str(train_paths.image_dir),
                    "ann_file": str(train_paths.ann_path),
                },
                "val_paths": (
                    {
                        "image_dir": str(val_paths.image_dir),
                        "ann_file": str(val_paths.ann_path),
                    }
                    if val_paths
                    else None
                ),
            },
        )

        history.append({"epoch": epoch, **metrics})
        if wandb_run is not None:
            wandb_run.log(metrics, step=epoch)

        if scheduler is not None:
            scheduler.step()

    config_path = output_dir / "faster_rcnn_config.json"
    config_path.write_text(
        json.dumps(
            {
                "model_type": "faster_rcnn",
                "backbone": "resnet50_fpn",
                "class_names": class_names,
                "num_classes": num_classes,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": lr,
                "milestones": milestones,
                "pretrained": not args.no_pretrained,
                "keep_unannotated": args.keep_unannotated,
                "train_paths": {
                    "image_dir": str(train_paths.image_dir),
                    "ann_file": str(train_paths.ann_path),
                },
                "val_paths": (
                    {
                        "image_dir": str(val_paths.image_dir),
                        "ann_file": str(val_paths.ann_path),
                    }
                    if val_paths
                    else None
                ),
                "train_sources": resolved.train_sources_config,
                "history": history,
                "best_ap": best_ap if best_ap >= 0 else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Training complete. Config saved to {config_path}")


if __name__ == "__main__":
    main()
