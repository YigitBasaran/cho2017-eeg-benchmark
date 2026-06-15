"""Central model factory and shape/softmax assertions."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .atcnet import build_atcnet
from .eegnet import build_eegnet


def build_model(
    model_name: str,
    cfg: Any,
    *,
    n_chans: int = 64,
    n_times: int = 512,
    n_outputs: int = 2,
) -> tuple[nn.Module, dict[str, Any]]:
    """Construct a model from its config; return ``(model, resolved_info)``."""
    params = dict(cfg.get("model.params", {}) or {})
    if model_name == "eegnet":
        model = build_eegnet(n_chans, n_outputs, n_times, **params)
        info = {"name": "eegnet", "source": "braindecode", "params": params}
        return model, info

    if model_name == "atcnet":
        inp = dict(cfg.get("model.input", {}) or {})
        source_pref = str(cfg.get("model.source", "braindecode"))
        allow_local = bool(cfg.get("model.allow_local_fallback", False))
        model, source = build_atcnet(
            n_chans, n_outputs, n_times,
            prefer_braindecode=(source_pref == "braindecode"),
            allow_local_fallback=allow_local,
            sfreq=float(inp.get("sfreq", 256.0)),
            input_window_seconds=float(inp.get("input_window_seconds", 2.0)),
            **params,
        )
        info = {"name": "atcnet", "source": source, "params": params, "input": inp}
        return model, info

    raise KeyError(f"Unknown model '{model_name}'. Known: eegnet, atcnet.")


def count_trainable_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def assert_no_softmax(
    model: nn.Module,
    *,
    n_chans: int = 64,
    n_times: int = 512,
    n_outputs: int = 2,
    batch: int = 2,
) -> dict[str, Any]:
    """Verify the model returns raw logits ``(batch, n_outputs)`` with no softmax.

    Primary check: scan the module tree for ``nn.Softmax`` / ``nn.LogSoftmax``.
    Secondary: a forward pass yields the right shape and the output does not look
    like a probability distribution.
    """
    offenders = [type(m).__name__ for m in model.modules()
                 if isinstance(m, (nn.Softmax, nn.LogSoftmax))]
    if offenders:
        raise AssertionError(
            f"Model contains a softmax layer ({offenders}); it must return raw "
            "logits for use with CrossEntropyLoss."
        )

    was_training = model.training
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(batch, n_chans, n_times))
    if was_training:
        model.train()

    if tuple(out.shape) != (batch, n_outputs):
        raise AssertionError(f"Model output shape {tuple(out.shape)} != ({batch}, {n_outputs}).")

    sums = out.sum(dim=-1)
    looks_like_probs = bool(torch.all(out >= 0)) and bool(
        torch.allclose(sums, torch.ones_like(sums), atol=1e-3)
    )
    if looks_like_probs:
        raise AssertionError(
            "Model output looks like a probability distribution (non-negative, rows "
            "sum to 1); expected raw logits."
        )
    return {"output_shape": tuple(out.shape), "no_softmax_modules": True}
