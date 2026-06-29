#!/usr/bin/env python3
"""Export EMA teacher weights from a PyTorch Lightning ``.ckpt`` to an inference ``.pth``."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from rfdetr.training.callbacks.best_model import BestModelCallback

from co_student.checkpoint_callback import _enriched_args
from co_student.dataset import count_categories, resolve_coco_ann_path

_EMA_CALLBACK_KEYS = ("RFDETREMACallback", "CoStudentMeanTeacherCallback")
_MODEL_PREFIX = "module.model."
_REFERENCE_PTH_NAMES = (
    "checkpoint_best_regular.pth",
    "checkpoint_best_total.pth",
    "checkpoint_best_ema.pth",
)
_MODEL_MAP_DETECTION = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
}
_MODEL_MAP_SEGMENTATION = {
    "nano": "RFDETRSegNano",
    "small": "RFDETRSegSmall",
    "medium": "RFDETRSegMedium",
    "large": "RFDETRSegLarge",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt",
        default="outputs/costudent/last.ckpt",
        help="Path to a PyTorch Lightning checkpoint (e.g. last.ckpt)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .pth path (default: <ckpt_stem>_ema_inference.pth beside the .ckpt)",
    )
    parser.add_argument(
        "--costudent-config",
        default=None,
        help="costudent_config.json for model_config/args (default: beside the .ckpt)",
    )
    parser.add_argument(
        "--model",
        default=None,
        choices=["nano", "small", "medium", "large"],
        help="Model size override when costudent_config.json has no model field",
    )
    parser.add_argument(
        "--reference-pth",
        default=None,
        help="Optional fallback: copy metadata from an existing inference .pth",
    )
    return parser.parse_args()


def _extract_ema_state_dict(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Pull detection-model weights from the RF-DETR EMA callback state."""
    callbacks = checkpoint.get("callbacks")
    if not isinstance(callbacks, dict):
        raise SystemExit("Checkpoint has no callbacks dict; is this a Lightning .ckpt file?")

    ema_callback_state: dict[str, Any] | None = None
    for key in _EMA_CALLBACK_KEYS:
        state = callbacks.get(key)
        if isinstance(state, dict):
            ema_callback_state = state
            break

    if ema_callback_state is None:
        names = ", ".join(callbacks)
        raise SystemExit(
            f"No EMA callback state found (looked for {_EMA_CALLBACK_KEYS}). "
            f"Callbacks present: {names}"
        )

    wrapped = ema_callback_state.get("average_model_state_dict")
    if not isinstance(wrapped, dict):
        raise SystemExit(
            "RFDETREMACallback state is present but average_model_state_dict is missing. "
            "Was EMA enabled during training (use_ema=True)?"
        )

    model_state = {
        key.removeprefix(_MODEL_PREFIX): value
        for key, value in wrapped.items()
        if isinstance(key, str) and key.startswith(_MODEL_PREFIX)
    }
    if not model_state:
        raise SystemExit(
            f"No tensors with prefix {_MODEL_PREFIX!r} in average_model_state_dict."
        )
    return model_state


def _resolve_costudent_config(ckpt_path: Path, config_arg: str | None) -> Path:
    if config_arg:
        path = Path(config_arg).expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"costudent_config.json not found: {path}")
        return path
    default = ckpt_path.parent / "costudent_config.json"
    if default.is_file():
        return default
    raise SystemExit(
        f"No costudent_config.json beside {ckpt_path}. "
        "Pass --costudent-config or use --reference-pth."
    )


def _class_names_from_coco(ann_path: str | Path) -> list[str]:
    with open(resolve_coco_ann_path(ann_path), encoding="utf-8") as f:
        data = json.load(f)
    categories = sorted(data["categories"], key=lambda cat: cat["id"])
    return [cat["name"] for cat in categories]


def _infer_model_size(costudent: dict[str, Any]) -> str | None:
    model = costudent.get("model")
    if isinstance(model, str) and model in _MODEL_MAP_DETECTION:
        return model

    run = (costudent.get("train_config") or {}).get("run")
    if isinstance(run, str):
        match = re.match(r"^(nano|small|medium|large)-", run)
        if match:
            return match.group(1)
    return None


def _build_model_config(costudent: dict[str, Any], model_size: str) -> tuple[dict[str, Any], str]:
    task = costudent.get("task", "detection")
    model_map = _MODEL_MAP_DETECTION if task == "detection" else _MODEL_MAP_SEGMENTATION
    model_name = costudent.get("model_name")
    if not isinstance(model_name, str) or not model_name:
        model_name = model_map[model_size]

    saved_config = costudent.get("model_config")
    if isinstance(saved_config, dict):
        model_config = dict(saved_config)
    else:
        import rfdetr.variants as variants

        model_cls = getattr(variants, model_name)
        wrapper = model_cls(freeze_encoder=bool(costudent.get("freeze_encoder", False)))
        train_paths = costudent.get("train_paths") or {}
        ann_file = train_paths.get("ann_file")
        if ann_file:
            wrapper.model_config.num_classes = count_categories(ann_file)
        model_config = wrapper.model_config.model_dump()

    return model_config, model_name


def _metadata_from_costudent_config(
    costudent: dict[str, Any],
    *,
    model_override: str | None,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    model_size = model_override or _infer_model_size(costudent)
    if model_size is None:
        raise SystemExit(
            "Could not determine model size from costudent_config.json. "
            "Re-run training with the updated script (saves model field) or pass --model."
        )

    model_config, model_name = _build_model_config(costudent, model_size)
    train_config = costudent.get("train_config")
    if not isinstance(train_config, dict):
        raise SystemExit("costudent_config.json is missing train_config")

    args_dict = dict(train_config)
    class_names = costudent.get("class_names")
    if class_names is None:
        train_paths = costudent.get("train_paths") or {}
        ann_file = train_paths.get("ann_file")
        if ann_file:
            class_names = _class_names_from_coco(ann_file)
    if class_names:
        args_dict["class_names"] = class_names

    return model_config, args_dict, model_name


def _metadata_from_reference(reference: Path) -> tuple[dict[str, Any], dict[str, Any], str | None]:
    ref_ckpt = torch.load(reference, map_location="cpu", weights_only=False)
    if not isinstance(ref_ckpt, dict):
        raise SystemExit(f"Reference checkpoint is not a dict: {reference}")

    model_config = ref_ckpt.get("model_config")
    args = ref_ckpt.get("args")
    model_name = ref_ckpt.get("model_name")

    if not isinstance(model_config, dict):
        raise SystemExit(f"Reference .pth has no model_config dict: {reference}")
    if not isinstance(args, dict):
        raise SystemExit(f"Reference .pth has no args dict: {reference}")

    model_name = model_name if isinstance(model_name, str) and model_name.strip() else None
    return model_config, args, model_name


def _resolve_reference_pth(ckpt_path: Path, reference_pth: str | None) -> Path | None:
    if reference_pth:
        path = Path(reference_pth).expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"Reference .pth not found: {path}")
        return path

    search_dir = ckpt_path.parent
    for name in _REFERENCE_PTH_NAMES:
        candidate = search_dir / name
        if candidate.is_file():
            return candidate
    return None


def _trainer_stub(checkpoint: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        current_epoch=int(checkpoint.get("epoch", 0)),
        global_step=int(checkpoint.get("global_step", 0)),
    )


def convert_ckpt_to_ema_pth(
    ckpt_path: Path,
    output_path: Path,
    *,
    costudent_config_path: Path | None = None,
    model_override: str | None = None,
    reference_pth: Path | None = None,
) -> None:
    print(f"Loading Lightning checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise SystemExit(f"Expected a dict checkpoint, got {type(checkpoint)}")

    ema_state_dict = _extract_ema_state_dict(checkpoint)
    print(f"Extracted {len(ema_state_dict)} EMA model tensors from RFDETREMACallback")

    if costudent_config_path is not None:
        print(f"Using metadata from: {costudent_config_path}")
        costudent = json.loads(costudent_config_path.read_text(encoding="utf-8"))
        model_config, args_dict, model_name = _metadata_from_costudent_config(
            costudent,
            model_override=model_override,
        )
    elif reference_pth is not None:
        print(f"Using metadata from: {reference_pth}")
        model_config, args_dict, model_name = _metadata_from_reference(reference_pth)
        if model_name is None:
            raise SystemExit(f"Reference .pth has no model_name: {reference_pth}")
    else:
        raise SystemExit("Provide --costudent-config or --reference-pth")

    payload = BestModelCallback._build_checkpoint_payload(
        ema_state_dict,
        args_dict,
        _trainer_stub(checkpoint),
        model_name=model_name,
        model_config_dict=model_config,
    )
    payload["model_config"] = dict(model_config)
    payload["args"] = _enriched_args(dict(args_dict), dict(model_config))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(f"Wrote inference checkpoint: {output_path}")
    print("Load with: RFDETR.from_checkpoint(...) or the matching RFDETR* variant class")


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.ckpt).expanduser().resolve()
    if not ckpt_path.is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        output_path = ckpt_path.with_name(f"{ckpt_path.stem}_ema_inference.pth")

    reference_pth = _resolve_reference_pth(ckpt_path, args.reference_pth)
    costudent_config_path: Path | None = None
    if args.costudent_config or not args.reference_pth:
        try:
            costudent_config_path = _resolve_costudent_config(ckpt_path, args.costudent_config)
        except SystemExit:
            if reference_pth is None:
                raise
            costudent_config_path = None

    convert_ckpt_to_ema_pth(
        ckpt_path,
        output_path,
        costudent_config_path=costudent_config_path,
        model_override=args.model,
        reference_pth=reference_pth if costudent_config_path is None else None,
    )


if __name__ == "__main__":
    main()
