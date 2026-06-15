"""EEGNet construction (Braindecode EEGNetv4 / EEGNet) behind a local wrapper.

The wrapper is tolerant of Braindecode's argument renames across versions
(``in_chans``/``n_chans``, ``n_classes``/``n_outputs``,
``input_window_samples``/``n_times``) and ensures the model returns **raw
logits** (no internal (log-)softmax), so it can be trained with
``nn.CrossEntropyLoss``.
"""

from __future__ import annotations

import inspect
from typing import Any

import torch.nn as nn


def _braindecode_eegnet_class():
    import braindecode.models as models

    # Prefer the current name (EEGNet); fall back to the deprecated EEGNetv4
    # alias on older Braindecode releases.
    cls = getattr(models, "EEGNet", None) or getattr(models, "EEGNetv4", None)
    if cls is None:  # pragma: no cover - depends on installed version
        raise ImportError("Neither EEGNet nor EEGNetv4 found in braindecode.models.")
    return cls


def eegnet_default_config() -> dict[str, Any]:
    return {
        "F1": 8, "D": 2, "F2": 16, "kernel_length": 64,
        "depthwise_kernel_length": 16, "drop_prob": 0.25,
    }


def build_eegnet(
    n_chans: int = 64,
    n_outputs: int = 2,
    n_times: int = 512,
    *,
    drop_prob: float = 0.25,
    **overrides: Any,
) -> nn.Module:
    """Construct a Braindecode EEGNet returning raw logits of shape ``(B, n_outputs)``."""
    cls = _braindecode_eegnet_class()
    params = inspect.signature(cls.__init__).parameters

    kwargs: dict[str, Any] = {}
    # channels
    if "n_chans" in params:
        kwargs["n_chans"] = n_chans
    elif "in_chans" in params:
        kwargs["in_chans"] = n_chans
    # outputs
    if "n_outputs" in params:
        kwargs["n_outputs"] = n_outputs
    elif "n_classes" in params:
        kwargs["n_classes"] = n_outputs
    # time samples
    if "n_times" in params:
        kwargs["n_times"] = n_times
    elif "input_window_samples" in params:
        kwargs["input_window_samples"] = n_times
    # dropout + logits
    if "drop_prob" in params:
        kwargs["drop_prob"] = drop_prob
    if "add_log_softmax" in params:
        kwargs["add_log_softmax"] = False
    if "final_conv_length" in params and "final_conv_length" not in overrides:
        kwargs["final_conv_length"] = "auto"

    for key, value in overrides.items():
        if key in params:
            kwargs[key] = value

    return cls(**kwargs)
