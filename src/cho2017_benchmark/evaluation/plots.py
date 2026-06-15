"""Reusable matplotlib plotting helpers.

Functions return a Matplotlib ``Figure`` so notebooks can display inline and also
save via :func:`save_fig`. The backend is left to the caller (notebooks use the
inline backend; headless scripts/tests may set ``Agg``).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_DPI = 200


def save_fig(fig, path: Path | str, dpi: int = DEFAULT_DPI) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def confusion_matrix_plot(cm: np.ndarray, class_names, *, normalized: bool = False, title: str = "Confusion matrix"):
    cm = np.asarray(cm, dtype=float)
    if normalized:
        with np.errstate(invalid="ignore", divide="ignore"):
            cm = cm / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() if cm.size else 1)
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fmt = ".2f" if normalized else ".0f"
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def subject_accuracy_sorted(metrics: pd.DataFrame, *, metric: str = "accuracy", title: str | None = None):
    data = metrics.sort_values(metric).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(data)), data[metric])
    ax.axhline(0.5, color="red", ls="--", lw=1, label="chance (0.5)")
    ax.set_xticks(range(len(data)), data["subject_id"], rotation=90, fontsize=7)
    ax.set_ylabel(metric)
    ax.set_title(title or f"Per-subject {metric} (sorted)")
    ax.legend()
    fig.tight_layout()
    return fig


def subject_accuracy_distribution(metrics: pd.DataFrame, *, metric: str = "accuracy"):
    fig, ax = plt.subplots(figsize=(4, 5))
    ax.violinplot(metrics[metric].dropna().to_numpy(), showmeans=True, showmedians=True)
    ax.scatter(np.random.default_rng(0).normal(1, 0.04, len(metrics)), metrics[metric],
               alpha=0.5, s=15)
    ax.axhline(0.5, color="red", ls="--", lw=1)
    ax.set_xticks([1], [metric])
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} distribution")
    fig.tight_layout()
    return fig


def accuracy_histogram(metrics: pd.DataFrame, *, metric: str = "accuracy"):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(metrics[metric].dropna(), bins=np.linspace(0, 1, 21), edgecolor="black")
    ax.axvline(0.5, color="red", ls="--", lw=1)
    ax.set_xlabel(metric)
    ax.set_ylabel("number of subjects")
    ax.set_title(f"{metric} histogram")
    fig.tight_layout()
    return fig


def accuracy_vs_train_trials(metrics: pd.DataFrame, *, metric: str = "accuracy"):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(metrics["n_train"], metrics[metric], alpha=0.6)
    ax.axhline(0.5, color="red", ls="--", lw=1)
    ax.set_xlabel("number of clean training trials")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs training-set size")
    fig.tight_layout()
    return fig


def training_time_by_subject(metrics: pd.DataFrame):
    data = metrics.sort_values("train_time_seconds").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(data)), data["train_time_seconds"])
    ax.set_xticks(range(len(data)), data["subject_id"], rotation=90, fontsize=7)
    ax.set_ylabel("training time (s)")
    ax.set_title("Training time by subject")
    fig.tight_layout()
    return fig


def latency_summary(metrics: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4))
    cols = ["inference_mean_ms", "inference_median_ms", "inference_p95_ms"]
    present = [c for c in cols if c in metrics]
    ax.boxplot([metrics[c].dropna() for c in present], labels=[c.replace("inference_", "") for c in present])
    ax.set_ylabel("ms / trial")
    ax.set_title("Inference latency")
    fig.tight_layout()
    return fig
