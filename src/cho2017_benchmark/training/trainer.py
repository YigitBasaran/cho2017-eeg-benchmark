"""Generic PyTorch trainer used by both neural experiments.

The trainer fits with AdamW + ReduceLROnPlateau, early-stops on the validation
monitor metric, restores the best-validation weights, and never reads the test
set. It is deliberately model-agnostic.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score

from ..reproducibility import set_seed
from .callbacks import EarlyStopping, build_scheduler


def resolve_device(device: str = "auto") -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


@dataclass
class TrainResult:
    best_epoch: int
    best_val_metric: float
    history: pd.DataFrame
    train_time_seconds: float


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        *,
        device: str = "cpu",
        optimizer_cfg: dict[str, Any] | None = None,
        scheduler_cfg: dict[str, Any] | None = None,
        max_epochs: int = 200,
        patience: int = 25,
        monitor: str = "val_balanced_accuracy",
        monitor_mode: str = "max",
        grad_clip: float | None = None,
        mixed_precision: bool = False,
        seed: int = 42,
        class_weights: np.ndarray | None = None,
    ) -> None:
        self.model = model
        self.device = device
        self.optimizer_cfg = optimizer_cfg or {}
        self.scheduler_cfg = scheduler_cfg or {"name": "reduce_on_plateau"}
        self.max_epochs = max_epochs
        self.patience = patience
        self.monitor = monitor
        self.monitor_mode = monitor_mode
        self.grad_clip = grad_clip
        self.mixed_precision = mixed_precision and str(device).startswith("cuda")
        self.seed = seed
        self.class_weights = class_weights

    def _make_optimizer(self):
        from torch.optim import AdamW

        return AdamW(
            self.model.parameters(),
            lr=float(self.optimizer_cfg.get("lr", 1e-3)),
            weight_decay=float(self.optimizer_cfg.get("weight_decay", 1e-4)),
        )

    def fit(
        self,
        train_loader,
        val_loader,
        *,
        on_improve: Callable[[int, float], None] | None = None,
    ) -> TrainResult:
        set_seed(self.seed)
        self.model.to(self.device)
        optimizer = self._make_optimizer()
        scheduler = build_scheduler(optimizer, self.scheduler_cfg, mode=self.monitor_mode)
        weight = (torch.tensor(self.class_weights, dtype=torch.float32, device=self.device)
                  if self.class_weights is not None else None)
        criterion = nn.CrossEntropyLoss(weight=weight)
        scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)
        early = EarlyStopping(self.patience, mode=self.monitor_mode)

        best_state = copy.deepcopy(self.model.state_dict())
        best_epoch, best_metric = 0, (-np.inf if self.monitor_mode == "max" else np.inf)
        history: list[dict[str, Any]] = []
        start = time.perf_counter()

        for epoch in range(self.max_epochs):
            train_loss = self._train_epoch(train_loader, optimizer, criterion, scaler)
            val_loss, val_balacc = self._validate(val_loader, criterion)
            monitor_value = val_balacc if self.monitor == "val_balanced_accuracy" else -val_loss
            scheduler.step(val_balacc if self.monitor_mode == "max" else val_loss)

            improved = early.step(monitor_value, epoch)
            if improved:
                best_state = copy.deepcopy(self.model.state_dict())
                best_epoch = epoch
                best_metric = monitor_value
                if on_improve is not None:
                    on_improve(epoch, monitor_value)

            history.append({
                "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                "val_balanced_accuracy": val_balacc,
                "lr": optimizer.param_groups[0]["lr"],
            })
            if early.should_stop:
                break

        self.model.load_state_dict(best_state)
        elapsed = time.perf_counter() - start
        return TrainResult(best_epoch, float(best_metric), pd.DataFrame(history), elapsed)

    def _train_epoch(self, loader, optimizer, criterion, scaler) -> float:
        self.model.train()
        total, n = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            optimizer.zero_grad(set_to_none=True)
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    logits = self.model(xb)
                    loss = criterion(logits, yb)
                scaler.scale(loss).backward()
                if self.grad_clip:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                if self.grad_clip:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                optimizer.step()
            total += float(loss.item()) * xb.size(0)
            n += xb.size(0)
        return total / max(n, 1)

    def _validate(self, loader, criterion) -> tuple[float, float]:
        self.model.eval()
        total, n = 0.0, 0
        ys, preds = [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits = self.model(xb)
                loss = criterion(logits, yb)
                total += float(loss.item()) * xb.size(0)
                n += xb.size(0)
                ys.append(yb.cpu().numpy())
                preds.append(logits.argmax(1).cpu().numpy())
        y = np.concatenate(ys) if ys else np.array([])
        p = np.concatenate(preds) if preds else np.array([])
        balacc = float(balanced_accuracy_score(y, p)) if len(y) else 0.0
        return total / max(n, 1), balacc

    def evaluate(self, loader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(y_true, y_pred, y_prob)`` where y_prob is softmax over classes."""
        self.model.to(self.device)
        self.model.eval()
        ys, preds, probs = [], [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(self.device)
                logits = self.model(xb)
                prob = torch.softmax(logits, dim=1)
                ys.append(yb.numpy())
                preds.append(logits.argmax(1).cpu().numpy())
                probs.append(prob.cpu().numpy())
        return (
            np.concatenate(ys) if ys else np.array([]),
            np.concatenate(preds) if preds else np.array([]),
            np.concatenate(probs) if probs else np.empty((0, 2)),
        )
