"""Consistent inference-latency measurement.

Per-trial latency is measured at batch size 1 after a warm-up, with CUDA
synchronisation around the timed region. Data loading / preprocessing are
excluded. The reporting device is recorded so GPU neural inference is never
silently compared against CPU CSP inference.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np


@dataclass
class LatencyResult:
    mean_ms: float
    median_ms: float
    p95_ms: float
    device: str
    n_reps: int

    def as_dict(self) -> dict[str, float | str | int]:
        return asdict(self)

    def to_inference_dict(self) -> dict[str, float]:
        return {"mean_ms": self.mean_ms, "median_ms": self.median_ms, "p95_ms": self.p95_ms}


def _summarize(times_ms: list[float], device: str) -> LatencyResult:
    arr = np.asarray(times_ms, dtype=float)
    return LatencyResult(
        mean_ms=float(arr.mean()), median_ms=float(np.median(arr)),
        p95_ms=float(np.percentile(arr, 95)), device=device, n_reps=len(arr),
    )


def measure_latency(
    model,
    sample,
    *,
    device: str = "cpu",
    warmup: int = 20,
    reps: int = 200,
) -> LatencyResult:
    """Measure per-trial (batch size 1) forward latency of a torch model."""
    import torch

    model = model.to(device)
    model.eval()
    x = sample.to(device)
    if x.dim() == 2:
        x = x.unsqueeze(0)
    is_cuda = str(device).startswith("cuda")

    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if is_cuda:
            torch.cuda.synchronize()
        times_ms: list[float] = []
        for _ in range(reps):
            if is_cuda:
                torch.cuda.synchronize()
            start = time.perf_counter()
            model(x)
            if is_cuda:
                torch.cuda.synchronize()
            times_ms.append((time.perf_counter() - start) * 1000.0)
    return _summarize(times_ms, str(device))


def measure_callable_latency(
    predict_fn: Callable[[np.ndarray], object],
    sample: np.ndarray,
    *,
    device: str = "cpu",
    warmup: int = 10,
    reps: int = 200,
) -> LatencyResult:
    """Measure per-trial latency of an arbitrary callable (e.g. the CSP+LDA pipeline)."""
    for _ in range(warmup):
        predict_fn(sample)
    times_ms: list[float] = []
    for _ in range(reps):
        start = time.perf_counter()
        predict_fn(sample)
        times_ms.append((time.perf_counter() - start) * 1000.0)
    return _summarize(times_ms, device)


# Backwards-friendly alias used by the CSP notebook.
measure_csp_pipeline_latency = measure_callable_latency
