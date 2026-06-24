"""Checkpoint enrichment so saved weights work with ``RFDETR.from_checkpoint()``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from rfdetr.training.callbacks.best_model import BestModelCallback

_INFERENCE_CHECKPOINTS = (
    "checkpoint_best_regular.pth",
    "checkpoint_best_ema.pth",
    "checkpoint_best_total.pth",
)

LAST_EMA_CHECKPOINT = "checkpoint_last_ema.pth"


def _model_config_dict(pl_module: LightningModule) -> dict[str, Any] | None:
    model_config = getattr(pl_module, "model_config", None)
    if model_config is None:
        return None
    if isinstance(model_config, dict):
        return dict(model_config)
    model_dump = getattr(model_config, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, dict):
            return dumped
    return None


def _train_args_dict(trainer: Trainer, pl_module: LightningModule) -> dict[str, Any]:
    train_config = pl_module.train_config
    datamodule = getattr(trainer, "datamodule", None)
    dataset_class_names = getattr(datamodule, "class_names", None)
    if (
        dataset_class_names is not None
        and hasattr(train_config, "model_copy")
        and getattr(train_config, "class_names", None) is None
    ):
        train_config = train_config.model_copy(update={"class_names": dataset_class_names})
    if hasattr(train_config, "model_dump"):
        args = train_config.model_dump()
        return dict(args) if isinstance(args, dict) else {}
    return dict(train_config) if isinstance(train_config, dict) else {}


def _enriched_args(args: dict[str, Any], model_config: dict[str, Any]) -> dict[str, Any]:
    enriched_args = dict(args)
    enriched_args["num_classes"] = model_config.get("num_classes")
    for key in (
        "group_detr",
        "num_queries",
        "segmentation_head",
        "patch_size",
        "freeze_encoder",
        "resolution",
    ):
        if key in model_config:
            enriched_args.setdefault(key, model_config[key])
    return enriched_args


def _apply_inference_metadata(
    checkpoint: dict[str, Any],
    pl_module: LightningModule,
    model_config: dict[str, Any],
    *,
    args: dict[str, Any] | None = None,
) -> None:
    checkpoint["model_config"] = model_config
    base_args = args if args is not None else checkpoint.get("args")
    if isinstance(base_args, dict):
        checkpoint["args"] = _enriched_args(base_args, model_config)
    else:
        checkpoint["args"] = _enriched_args({}, model_config)


def _get_ema_callback(trainer: Trainer) -> Any | None:
    for callback in trainer.callbacks:
        if callable(getattr(callback, "get_ema_model_state_dict", None)):
            return callback
    return None


def _ema_state_dict_from_last_ckpt(output_dir: Path) -> dict[str, torch.Tensor] | None:
    last_ckpt = output_dir / "last.ckpt"
    if not last_ckpt.is_file():
        return None
    checkpoint = torch.load(last_ckpt, map_location="cpu", weights_only=False)
    callbacks = checkpoint.get("callbacks")
    if not isinstance(callbacks, dict):
        return None
    ema_callback_state = callbacks.get("RFDETREMACallback")
    if not isinstance(ema_callback_state, dict):
        return None
    wrapped = ema_callback_state.get("average_model_state_dict")
    if not isinstance(wrapped, dict):
        return None
    prefix = "module.model."
    model_state = {
        key.removeprefix(prefix): value
        for key, value in wrapped.items()
        if isinstance(key, str) and key.startswith(prefix)
    }
    return model_state or None


def _extract_ema_state_dict(trainer: Trainer, output_dir: Path) -> dict[str, torch.Tensor] | None:
    ema_callback = _get_ema_callback(trainer)
    if ema_callback is not None:
        state_dict = ema_callback.get_ema_model_state_dict()
        if state_dict:
            return state_dict
    return _ema_state_dict_from_last_ckpt(output_dir)


def enrich_inference_checkpoint(
    path: Path | str,
    pl_module: LightningModule,
    *,
    trainer: Trainer | None = None,
    ema: bool = False,
) -> None:
    """Add inference metadata and optionally write an EMA-weight checkpoint.

    When ``ema=False`` (default), enrich an existing RF-DETR ``.pth`` checkpoint in place
    with ``model_config`` and ``args.num_classes`` so :func:`RFDETR.from_checkpoint` works.

    When ``ema=True``, build a new checkpoint at *path* using the EMA teacher weights from
    the active EMA callback, or fall back to ``last.ckpt`` callback state.
    """
    ckpt_path = Path(path)
    model_config = _model_config_dict(pl_module)
    if model_config is None:
        return

    if ema:
        if trainer is None:
            raise ValueError("trainer is required when ema=True")
        ema_state_dict = _extract_ema_state_dict(trainer, ckpt_path.parent)
        if ema_state_dict is None:
            return
        args_dict = _train_args_dict(trainer, pl_module)
        model_name = BestModelCallback._resolve_model_name(pl_module)
        checkpoint = BestModelCallback._build_checkpoint_payload(
            ema_state_dict,
            args_dict,
            trainer,
            model_name=model_name,
            model_config_dict=model_config,
        )
        _apply_inference_metadata(checkpoint, pl_module, model_config, args=args_dict)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, ckpt_path)
        return

    if not ckpt_path.is_file():
        return

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        return

    _apply_inference_metadata(checkpoint, pl_module, model_config)
    torch.save(checkpoint, ckpt_path)


def _checkpoint_interval(pl_module: LightningModule) -> int:
    interval = getattr(pl_module.train_config, "checkpoint_interval", 10)
    return max(1, int(interval))


def _should_save_interval_checkpoint(trainer: Trainer, pl_module: LightningModule) -> bool:
    """Return whether RF-DETR's ``checkpoint_{epoch}.ckpt`` archive fires this epoch."""
    interval = _checkpoint_interval(pl_module)
    epoch = int(trainer.current_epoch)
    return interval == 1 or (epoch + 1) % interval == 0


def _should_save_last_checkpoint(pl_module: LightningModule) -> bool:
    """Return whether RF-DETR's ``last.ckpt`` resume checkpoint fires every epoch."""
    return _checkpoint_interval(pl_module) != 1


class EnrichInferenceCheckpointsCallback(Callback):
    """Write inference-ready ``.pth`` checkpoints, including EMA mirrors of PTL saves."""

    def __init__(self, output_dir: str | Path) -> None:
        self._output_dir = Path(output_dir)

    def _save_ema_checkpoint(self, trainer: Trainer, pl_module: LightningModule, path: Path) -> None:
        if not getattr(pl_module.train_config, "use_ema", True):
            return
        enrich_inference_checkpoint(path, pl_module, trainer=trainer, ema=True)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not trainer.is_global_zero or getattr(trainer, "sanity_checking", False):
            return

        if _should_save_last_checkpoint(pl_module):
            self._save_ema_checkpoint(trainer, pl_module, self._output_dir / LAST_EMA_CHECKPOINT)

        if _should_save_interval_checkpoint(trainer, pl_module):
            epoch = int(trainer.current_epoch)
            self._save_ema_checkpoint(
                trainer,
                pl_module,
                self._output_dir / f"checkpoint_{epoch}_ema.pth",
            )

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not trainer.is_global_zero:
            return
        for name in _INFERENCE_CHECKPOINTS:
            enrich_inference_checkpoint(self._output_dir / name, pl_module)
        self._save_ema_checkpoint(trainer, pl_module, self._output_dir / LAST_EMA_CHECKPOINT)
