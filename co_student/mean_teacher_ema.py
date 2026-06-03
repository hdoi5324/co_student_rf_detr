"""MeanTeacher-style EMA for Co-Student (matches cvpods ``hooks.MeanTeacher``)."""

from __future__ import annotations

from rfdetr.training.callbacks.ema import RFDETREMACallback


class CoStudentMeanTeacherCallback(RFDETREMACallback):
    """EMA teacher with CoStudent / cvpods MeanTeacher momentum schedule.

    Update rule (same as ``MeanTeacher.momentum_update``)::

        teacher = m * teacher + (1 - m) * student

    with::

        m = min(momentum, 1 - (1 + warm_up) / (step + 1 + warm_up))

    where *step* is the 1-indexed EMA update count. At ``step == 1`` and
    ``warm_up == 0``, ``m = 0.5`` (teacher blends evenly), then ``m`` approaches
    *momentum* (default ``0.999``).

    RF-DETR's default :class:`~rfdetr.training.callbacks.ema.RFDETREMACallback`
    uses constant ``ema_decay`` (and optional ``ema_tau`` ramp). Use this callback
    when you want parity with::

        hooks.MeanTeacher(runner, momentum=0.999, interval=1, warm_up=0, ...)
    """

    def __init__(
        self,
        momentum: float = 0.999,
        warm_up: int = 0,
        update_interval_steps: int = 1,
        use_buffers: bool = True,
    ) -> None:
        super().__init__(
            decay=momentum,
            tau=0,
            use_buffers=use_buffers,
            update_interval_steps=update_interval_steps,
        )
        self._momentum = momentum
        self._warm_up = warm_up

    def _avg_fn(
        self,
        averaged_param,
        model_param,
        num_averaged: int,
    ):
        step = num_averaged + 1
        m = min(
            self._momentum,
            1.0 - (1.0 + self._warm_up) / (step + 1.0 + self._warm_up),
        )
        return averaged_param * m + model_param * (1.0 - m)
