"""Training callbacks: early stopping and the LR scheduler factory."""

from __future__ import annotations

from typing import Any


class EarlyStopping:
    """Track the monitored metric and signal when to stop.

    ``step`` returns True when the metric *improved* (so the caller can snapshot
    the best checkpoint). ``should_stop`` becomes True after ``patience`` epochs
    with no improvement.
    """

    def __init__(self, patience: int = 25, mode: str = "max", min_delta: float = 0.0) -> None:
        if mode not in ("max", "min"):
            raise ValueError("mode must be 'max' or 'min'.")
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best: float | None = None
        self.best_epoch: int = -1
        self.num_bad: int = 0
        self.should_stop: bool = False

    def _is_better(self, value: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "max":
            return value > self.best + self.min_delta
        return value < self.best - self.min_delta

    def step(self, value: float, epoch: int) -> bool:
        improved = self._is_better(value)
        if improved:
            self.best = value
            self.best_epoch = epoch
            self.num_bad = 0
        else:
            self.num_bad += 1
            if self.num_bad >= self.patience:
                self.should_stop = True
        return improved


def build_scheduler(optimizer, scheduler_cfg: dict[str, Any], mode: str = "max"):
    """Build a ReduceLROnPlateau scheduler (the only documented scheduler)."""
    from torch.optim.lr_scheduler import ReduceLROnPlateau

    name = str(scheduler_cfg.get("name", "reduce_on_plateau"))
    if name != "reduce_on_plateau":
        raise ValueError(f"Unsupported scheduler '{name}'.")
    return ReduceLROnPlateau(
        optimizer,
        mode=mode,
        factor=float(scheduler_cfg.get("factor", 0.5)),
        patience=int(scheduler_cfg.get("patience", 10)),
        min_lr=float(scheduler_cfg.get("min_lr", 1e-5)),
    )
