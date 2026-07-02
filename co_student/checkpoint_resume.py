"""Resume training without restoring optimizer or LR scheduler state."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from pytorch_lightning import Callback, LightningModule, Trainer

logger = logging.getLogger(__name__)


class RestoreTrainingProgressCallback(Callback):
    """Restore epoch and global_step from a Lightning checkpoint at fit start."""

    def __init__(self, checkpoint: dict[str, Any]) -> None:
        super().__init__()
        self._checkpoint = checkpoint

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        loops = self._checkpoint.get("loops")
        if isinstance(loops, dict) and "fit_loop" in loops:
            trainer.fit_loop.load_state_dict(loops["fit_loop"])
            logger.info(
                "Restored training progress (epoch=%s, global_step=%s).",
                trainer.current_epoch,
                trainer.global_step,
            )
            return

        epoch = int(self._checkpoint.get("epoch", 0))
        global_step = int(self._checkpoint.get("global_step", 0))
        trainer.fit_loop.epoch_progress.current.completed = epoch
        trainer.fit_loop.epoch_progress.current.started = epoch
        trainer.fit_loop.epoch_progress.current.processed = epoch
        trainer.fit_loop.global_step = global_step
        logger.info(
            "Restored training progress from checkpoint metadata (epoch=%s, global_step=%s).",
            epoch,
            global_step,
        )


def load_checkpoint_dict(ckpt_path: str | Path) -> dict[str, Any]:
    """Load a Lightning checkpoint dict from disk."""
    path = Path(ckpt_path).expanduser().resolve()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Expected dict checkpoint at {path}, got {type(checkpoint)}")
    return checkpoint


def checkpoint_needs_weights_only_resume(checkpoint: dict[str, Any]) -> bool:
    """Return True when a checkpoint cannot restore optimizer or scheduler state."""
    return "optimizer_states" not in checkpoint or "lr_schedulers" not in checkpoint


def _restore_callback_states(trainer: Trainer, saved_callbacks: dict[str, Any]) -> None:
    ema_state: dict[str, Any] | None = None
    for key, state in saved_callbacks.items():
        if isinstance(state, dict) and ("EMA" in key or "MeanTeacher" in key):
            ema_state = state
            break

    for callback in trainer.callbacks:
        state = saved_callbacks.get(callback.state_key)
        if isinstance(state, dict):
            callback.load_state_dict(state)
            continue
        if ema_state is not None and "EMA" in callback.__class__.__qualname__:
            callback.load_state_dict(ema_state)


def load_weights_only_checkpoint(
    ckpt_path: str | Path,
    module: LightningModule,
    trainer: Trainer,
) -> dict[str, Any]:
    """Load model weights and callback state; append a progress-restore callback."""
    checkpoint = load_checkpoint_dict(ckpt_path)

    state_dict = checkpoint.get("state_dict")
    if isinstance(state_dict, dict):
        incompatible = module.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            logger.warning(
                "Partial checkpoint load: missing=%d unexpected=%d",
                len(incompatible.missing_keys),
                len(incompatible.unexpected_keys),
            )

    saved_callbacks = checkpoint.get("callbacks")
    if isinstance(saved_callbacks, dict):
        _restore_callback_states(trainer, saved_callbacks)

    trainer.callbacks.append(RestoreTrainingProgressCallback(checkpoint))
    return checkpoint
