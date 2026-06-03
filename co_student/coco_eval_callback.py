"""COCO eval callback that prints both student and EMA teacher validation tables."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from rfdetr.evaluation.f1_sweep import sweep_confidence_thresholds
from rfdetr.evaluation.matching import (
    build_matching_data,
    distributed_merge_matching_data,
    init_matching_accumulator,
    merge_matching_data,
)
from rfdetr.training.callbacks.coco_eval import COCOEvalCallback


class CoStudentCOCOEvalCallback(COCOEvalCallback):
    """Extends RF-DETR validation logging with printed EMA (teacher) metric tables."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._f1_local_ema: dict[int, dict[str, Any]] = init_matching_accumulator()
        self.map_metric_ap50: Any = None
        self.map_metric_ema_ap50: Any = None
        self._ap50_by_cid_cache: dict[int, float] = {}

    def _make_ap50_map_metric(self, device: torch.device) -> Any:
        from torchmetrics.detection import MeanAveragePrecision

        iou_type: Any = ["bbox", "segm"] if self._segmentation else "bbox"
        return MeanAveragePrecision(
            iou_type=iou_type,
            iou_thresholds=[0.5],
            class_metrics=True,
            max_detection_thresholds=[1, 10, self._max_dets],
            backend="faster_coco_eval",
        ).to(device)

    @staticmethod
    def _normalize_metrics_tensors(metrics: dict[str, Any]) -> dict[str, Any]:
        if "classes" not in metrics:
            return metrics
        if metrics["classes"].ndim == 0:
            metrics = dict(metrics)
            metrics["classes"] = metrics["classes"].unsqueeze(0)
            for key in list(metrics):
                val = metrics[key]
                if isinstance(val, torch.Tensor) and val.ndim == 0 and "per_class" in key:
                    metrics[key] = val.unsqueeze(0)
        return metrics

    def _per_class_ap50_from_metric(self, metric: Any) -> dict[int, float]:
        """Per-class AP@50 from a metric trained with ``iou_thresholds=[0.5]`` only."""
        if metric is None:
            return {}
        metrics = self._normalize_metrics_tensors(metric.compute())
        pfx = "bbox_" if self._segmentation else ""
        pc_key = f"{pfx}map_per_class"
        if pc_key not in metrics or "classes" not in metrics:
            return {}
        return {int(c): float(ap) for c, ap in zip(metrics["classes"], metrics[pc_key])}

    def on_validation_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: dict[str, Any],
        batch: Any,
        batch_idx: int,
    ) -> None:
        preds = self._convert_preds(outputs["results"])
        targets = self._convert_targets(outputs["targets"])

        self.map_metric.update(preds, targets)

        if self.map_metric_ap50 is None:
            self.map_metric_ap50 = self._make_ap50_map_metric(pl_module.device)
        self.map_metric_ap50.update(preds, targets)

        iou_type = "segm" if self._segmentation else "bbox"
        merge_matching_data(
            self._f1_local,
            build_matching_data(preds, targets, iou_threshold=0.5, iou_type=iou_type),
        )

        ema_cb = self._get_ema_callback(trainer)
        if ema_cb is None or ema_cb._average_model is None:
            return

        if self.map_metric_ema is None:
            from torchmetrics.detection import MeanAveragePrecision

            ema_iou_type: Any = ["bbox", "segm"] if self._segmentation else "bbox"
            self.map_metric_ema = MeanAveragePrecision(
                iou_type=ema_iou_type,
                class_metrics=True,
                max_detection_thresholds=[1, 10, self._max_dets],
                backend="faster_coco_eval",
            ).to(pl_module.device)

        samples, _ = batch
        orig_sizes = torch.stack([t["orig_size"] for t in outputs["targets"]]).to(pl_module.device)
        ema_underlying = ema_cb._average_model.module.model
        with torch.no_grad():
            ema_underlying.eval()
            ema_outputs = ema_underlying(samples)
            ema_results = pl_module.postprocess(ema_outputs, orig_sizes)
        ema_preds = self._convert_preds(ema_results)
        self.map_metric_ema.update(ema_preds, targets)
        if self.map_metric_ema_ap50 is None:
            self.map_metric_ema_ap50 = self._make_ap50_map_metric(pl_module.device)
        self.map_metric_ema_ap50.update(ema_preds, targets)
        merge_matching_data(
            self._f1_local_ema,
            build_matching_data(ema_preds, targets, iou_threshold=0.5, iou_type=iou_type),
        )

    def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        if self._eval_interval > 1:
            current_epoch = int(getattr(trainer, "current_epoch", 0)) + 1
            max_epochs = getattr(trainer, "max_epochs", None)
            is_last_epoch = isinstance(max_epochs, int) and max_epochs > 0 and current_epoch >= max_epochs
            if current_epoch % self._eval_interval != 0 and not is_last_epoch:
                self.map_metric.reset()
                if self.map_metric_ema is not None:
                    self.map_metric_ema.reset()
                if self.map_metric_ap50 is not None:
                    self.map_metric_ap50.reset()
                if self.map_metric_ema_ap50 is not None:
                    self.map_metric_ema_ap50.reset()
                self._f1_local = init_matching_accumulator()
                self._f1_local_ema = init_matching_accumulator()
                return
        self._compute_and_log(trainer, pl_module, "val")

    def _compute_and_log(self, trainer: Any, pl_module: Any, split: str) -> None:
        ema_snapshot: tuple[dict[str, Any], dict[int, dict[str, Any]]] | None = None
        if self.map_metric_ema is not None:
            ema_snapshot = (
                self.map_metric_ema.compute(),
                distributed_merge_matching_data(self._f1_local_ema),
            )

        self._ap50_by_cid_cache = self._per_class_ap50_from_metric(self.map_metric_ap50)

        orig_print = self._print_metrics_tables

        def _print_student(tr: Any, sp: str, ov: dict[str, float], pc: list) -> None:
            orig_print(tr, sp, ov, pc, subtitle="Student")

        self._print_metrics_tables = _print_student  # type: ignore[method-assign]
        try:
            super()._compute_and_log(trainer, pl_module, split)
        finally:
            self._print_metrics_tables = orig_print

        if self.map_metric_ap50 is not None:
            self.map_metric_ap50.reset()

        if ema_snapshot is None or not getattr(trainer, "is_global_zero", True):
            if ema_snapshot is not None:
                self._f1_local_ema = init_matching_accumulator()
            return

        ema_metrics, merged_ema = ema_snapshot
        self._ap50_by_cid_cache = self._per_class_ap50_from_metric(self.map_metric_ema_ap50)
        if self.map_metric_ema_ap50 is not None:
            self.map_metric_ema_ap50.reset()
        pfx = "bbox_" if self._segmentation else ""
        mar_key = f"{pfx}mar_{self._max_dets}"
        ema_overall, ema_per_class = self._display_tables_from_metrics(
            ema_metrics,
            merged_ema,
            pfx=pfx,
            mar_key=mar_key,
            split=split,
            pl_module=pl_module,
            log_prefix="ema_",
        )
        self._print_metrics_tables(
            trainer,
            split,
            ema_overall,
            ema_per_class,
            subtitle="EMA Teacher",
        )
        self._f1_local_ema = init_matching_accumulator()

    def _display_tables_from_metrics(
        self,
        metrics: dict[str, Any],
        merged_f1: dict[int, dict[str, Any]],
        *,
        pfx: str,
        mar_key: str,
        split: str,
        pl_module: Any,
        log_prefix: str,
    ) -> tuple[dict[str, float], list[dict[str, Any]]]:
        """Build overall and per-class rows for terminal tables (no scalar logging)."""
        metrics = self._normalize_metrics_tensors(metrics)

        overall: dict[str, float] = {
            "mAP 50:95": float(metrics[f"{pfx}map"]),
            "mAP 50": float(metrics[f"{pfx}map_50"]),
            "mAP 75": float(metrics[f"{pfx}map_75"]),
            f"mAR @{self._max_dets}": float(metrics[mar_key]),
        }

        f1_by_cid: dict[int, dict[str, float]] = {}
        if merged_f1:
            sorted_ids = sorted(merged_f1.keys())
            per_class_list = [merged_f1[cid] for cid in sorted_ids]
            classes_with_gt = [i for i, cid in enumerate(sorted_ids) if merged_f1[cid]["total_gt"] > 0]
            f1_results = sweep_confidence_thresholds(
                per_class_list, np.linspace(0, 1, 101), classes_with_gt
            )
            best = max(f1_results, key=lambda x: x["macro_f1"])
            overall["F1"] = float(best["macro_f1"])
            overall["Precision"] = float(best["macro_precision"])
            overall["Recall"] = float(best["macro_recall"])
            for k, cid in enumerate(sorted_ids):
                f1_by_cid[cid] = {
                    "f1": float(best["per_class_f1"][k]),
                    "precision": float(best["per_class_prec"][k]),
                    "recall": float(best["per_class_rec"][k]),
                }
        else:
            overall["F1"] = 0.0
            overall["Precision"] = 0.0
            overall["Recall"] = 0.0

        ar_pc_key = f"{pfx}mar_{self._max_dets}_per_class"
        ar_by_cid: dict[int, float] = {}
        if ar_pc_key in metrics and "classes" in metrics:
            for class_id, ar in zip(metrics["classes"], metrics[ar_pc_key]):
                ar_by_cid[int(class_id)] = float(ar)

        per_class = self._build_per_class_rows(
            metrics=metrics,
            pfx=pfx,
            split=split,
            pl_module=pl_module,
            ar_by_cid=ar_by_cid,
            f1_by_cid=f1_by_cid,
            log_prefix=log_prefix,
        )
        return overall, per_class

    def _build_per_class_rows(
        self,
        metrics: dict[str, Any],
        pfx: str,
        split: str,
        pl_module: Any,
        ar_by_cid: dict[int, float],
        f1_by_cid: dict[int, dict[str, float]],
        log_prefix: str = "",
    ) -> list[dict[str, Any]]:
        """Build per-class rows; skip ``pl_module.log`` when ``log_prefix`` is set (display-only)."""
        if not self._log_per_class_metrics:
            return []

        pc_key = f"{pfx}map_per_class"
        if pc_key not in metrics or "classes" not in metrics:
            return []

        per_class: list[dict[str, Any]] = []
        for class_id, ap in zip(metrics["classes"], metrics[pc_key]):
            ap_f = float(ap)
            ar_f = ar_by_cid.get(int(class_id), float("nan"))
            if ap_f < 0 and (ar_f != ar_f or ar_f < 0):
                continue
            idx = int(class_id)
            name = self._cat_id_to_name.get(idx, str(idx))
            if not log_prefix:
                pl_module.log(f"{split}/AP/{name}", ap)
            ap50_f = self._ap50_by_cid_cache.get(idx, float("nan"))
            if not log_prefix:
                pl_module.log(f"{split}/AP50/{name}", ap50_f if ap50_f == ap50_f else 0.0)
            row: dict[str, Any] = {
                "name": name,
                "ap": ap_f,
                "ap50": ap50_f,
                "ar": ar_f,
            }
            row.update(
                f1_by_cid.get(idx, {"f1": float("nan"), "precision": float("nan"), "recall": float("nan")})
            )
            per_class.append(row)
        return per_class

    def _print_metrics_tables(
        self,
        trainer: Any,
        split: str,
        overall: dict[str, float],
        per_class: list[dict[str, Any]],
        subtitle: str | None = None,
    ) -> None:
        title_pfx = split.capitalize()
        if subtitle:
            title_pfx = f"{title_pfx} — {subtitle}"
        if not getattr(trainer, "is_global_zero", True):
            return
        try:
            from rich.console import Console
            from rich.table import Table
        except ImportError:
            return

        def _fmt(v: float) -> str:
            if v != v or v < 0:
                return "—"
            return f"{v:.4f}"

        console = Console(force_terminal=True)

        def _render_all() -> None:
            console.print(self._render_overall_merged(title_pfx, overall))
            if per_class:
                t2 = Table(
                    title=f"{title_pfx} — Per-class Metrics",
                    title_style="bold cyan",
                    show_header=True,
                    header_style="bold cyan",
                )
                t2.add_column("Class", style="dim", no_wrap=True)
                t2.add_column("AP 50:95", justify="right")
                t2.add_column("AP 50", justify="right")
                t2.add_column("AR", justify="right")
                t2.add_column("F1", justify="right")
                t2.add_column("Precision", justify="right")
                t2.add_column("Recall", justify="right")
                for row in per_class:
                    t2.add_row(
                        row["name"],
                        _fmt(row["ap"]),
                        _fmt(row["ap50"]),
                        _fmt(row["ar"]),
                        _fmt(row["f1"]),
                        _fmt(row["precision"]),
                        _fmt(row["recall"]),
                    )
                console.print(t2)

        import contextlib

        if self._in_notebook:
            if self._output_widget is None:
                with contextlib.suppress(ImportError):
                    import ipywidgets as widgets
                    from IPython.display import display

                    self._output_widget = widgets.Output()
                    display(self._output_widget)

            if self._output_widget is not None:
                self._output_widget.clear_output(wait=True)
                with self._output_widget:
                    _render_all()
                return

        _render_all()
