"""Paired, participant-level statistical analysis.

Because all models are evaluated on the same participants, comparisons are paired
on participant-level scores. Neural seeds are averaged per participant first
(:func:`aggregate_seeds_per_subject`) so multiple seeds are never treated as
independent participants. Across the three model pairs, p-values are Holm-corrected.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

MODEL_PAIRS = [("csp_lda", "eegnet"), ("csp_lda", "atcnet"), ("eegnet", "atcnet")]


@dataclass
class PairedTestResult:
    comparison: str
    metric: str
    n_pairs: int
    mean_diff: float
    median_diff: float
    ci_low: float
    ci_high: float
    raw_p: float
    corrected_p: float
    effect_size: float
    effect_size_name: str
    significant: bool


def aggregate_seeds_per_subject(
    metrics: pd.DataFrame, value_cols: list[str] | None = None
) -> pd.DataFrame:
    """Average per-(model, subject) across seeds. One row per participant per model."""
    if value_cols is None:
        value_cols = [c for c in ("accuracy", "balanced_accuracy", "macro_f1",
                                  "cohen_kappa", "roc_auc") if c in metrics.columns]
    ok = metrics[metrics.get("status", "ok") == "ok"] if "status" in metrics else metrics
    grouped = (
        ok.groupby(["model_name", "subject_id"], as_index=False)[value_cols].mean()
    )
    return grouped


def wilcoxon_paired(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Wilcoxon signed-rank test statistic and p-value for paired a, b."""
    from scipy.stats import wilcoxon

    diff = np.asarray(a) - np.asarray(b)
    if np.allclose(diff, 0):
        return 0.0, 1.0
    try:
        stat, p = wilcoxon(a, b)
        return float(stat), float(p)
    except ValueError:
        return float("nan"), 1.0


def rank_biserial(a: np.ndarray, b: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation effect size for a - b."""
    from scipy.stats import rankdata

    diff = np.asarray(a) - np.asarray(b)
    diff = diff[diff != 0]
    if diff.size == 0:
        return 0.0
    ranks = rankdata(np.abs(diff))
    w_plus = ranks[diff > 0].sum()
    w_minus = ranks[diff < 0].sum()
    total = w_plus + w_minus
    return float((w_plus - w_minus) / total) if total else 0.0


def cohens_dz(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's dz for paired samples (mean diff / std diff)."""
    diff = np.asarray(a) - np.asarray(b)
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else 0.0


def bootstrap_ci_mean_diff(
    a: np.ndarray, b: np.ndarray, *, n_boot: int = 10000, seed: int = 42, alpha: float = 0.05
) -> tuple[float, float]:
    """Bootstrap CI of the mean paired difference (a - b)."""
    diff = np.asarray(a) - np.asarray(b)
    n = diff.size
    if n == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot = diff[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo = float(np.percentile(boot, 100 * alpha / 2))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return lo, hi


def holm_correction(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down correction; preserves input order."""
    m = len(pvals)
    order = np.argsort(pvals)
    corrected = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running_max = max(running_max, val)
        corrected[idx] = min(1.0, running_max)
    return corrected


def paired_model_comparison(
    per_subject: pd.DataFrame,
    metric: str,
    *,
    alpha: float = 0.05,
    effect: str = "rank_biserial",
    seed: int = 42,
) -> list[PairedTestResult]:
    """Run the three paired model comparisons for *metric* with Holm correction.

    *per_subject* must have columns ``model_name, subject_id, <metric>`` (one row
    per participant per model; use :func:`aggregate_seeds_per_subject` first).
    """
    wide = per_subject.pivot_table(index="subject_id", columns="model_name", values=metric)
    results: list[dict] = []
    raw_ps: list[float] = []
    for model_a, model_b in MODEL_PAIRS:
        if model_a not in wide or model_b not in wide:
            continue
        pair = wide[[model_a, model_b]].dropna()
        a = pair[model_a].to_numpy()
        b = pair[model_b].to_numpy()
        diff = b - a  # second minus first
        _, p = wilcoxon_paired(a, b)
        es = rank_biserial(b, a) if effect == "rank_biserial" else cohens_dz(b, a)
        lo, hi = bootstrap_ci_mean_diff(b, a, seed=seed)
        raw_ps.append(p)
        results.append({
            "comparison": f"{model_b}_vs_{model_a}", "metric": metric,
            "n_pairs": int(len(pair)), "mean_diff": float(diff.mean()),
            "median_diff": float(np.median(diff)), "ci_low": lo, "ci_high": hi,
            "raw_p": p, "effect_size": es,
            "effect_size_name": effect,
        })

    corrected = holm_correction(raw_ps) if raw_ps else []
    out: list[PairedTestResult] = []
    for res, cp in zip(results, corrected):
        out.append(PairedTestResult(
            corrected_p=cp, significant=bool(cp < alpha), **res
        ))
    return out


def comparison_table(results: list[PairedTestResult]) -> pd.DataFrame:
    return pd.DataFrame([asdict(r) for r in results])
