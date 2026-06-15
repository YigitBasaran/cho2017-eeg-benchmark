"""ATCNet construction.

Braindecode's ``ATCNet`` is the DEFAULT implementation. A local PyTorch
re-implementation (:class:`LocalATCNet`) is provided as an *opt-in* fallback
(``allow_local_fallback=True``); its results must be labelled separately
(``model_source == "local_fallback"``) and never mixed with the Braindecode
results.

:class:`LocalATCNet` follows Altaheri et al., "Physics-informed attention
temporal convolutional network for EEG-based motor imagery classification"
(IEEE TII 2023): an EEGNet-style convolutional block, sliding temporal windows,
multi-head self-attention, a TCN, and per-window classification heads averaged
into the output. Documented deviations from the TensorFlow reference:

- attention head dimension is ``embed_dim / num_heads`` (PyTorch
  ``nn.MultiheadAttention``) rather than the reference's fixed ``key_dim=8``;
- the depthwise-conv max-norm weight constraint is omitted;
- per-window **logits** are averaged (the reference averages softmax outputs);
  this is equivalent for argmax and appropriate for ``CrossEntropyLoss``.

All variants return raw logits ``(B, n_outputs)``.
"""

from __future__ import annotations

import inspect
from typing import Any

import torch
import torch.nn as nn


# --- Braindecode ATCNet (default) -----------------------------------------


def _braindecode_atcnet_class():
    import braindecode.models as models

    cls = getattr(models, "ATCNet", None)
    if cls is None:  # pragma: no cover - depends on installed version
        raise ImportError("braindecode.models.ATCNet is not available in this version.")
    return cls


def build_atcnet_braindecode(
    n_chans: int = 64,
    n_outputs: int = 2,
    *,
    n_times: int = 512,
    sfreq: float = 256.0,
    input_window_seconds: float = 2.0,
    **params: Any,
) -> nn.Module:
    """Construct Braindecode ATCNet returning raw logits ``(B, n_outputs)``."""
    cls = _braindecode_atcnet_class()
    sig = inspect.signature(cls.__init__).parameters

    kwargs: dict[str, Any] = {}
    if "n_chans" in sig:
        kwargs["n_chans"] = n_chans
    elif "in_chans" in sig:
        kwargs["in_chans"] = n_chans
    if "n_outputs" in sig:
        kwargs["n_outputs"] = n_outputs
    elif "n_classes" in sig:
        kwargs["n_classes"] = n_outputs
    # Time can be specified by n_times or by window-seconds + sfreq.
    if "n_times" in sig:
        kwargs["n_times"] = n_times
    if "sfreq" in sig:
        kwargs["sfreq"] = sfreq
    if "input_window_seconds" in sig and "n_times" not in sig:
        kwargs["input_window_seconds"] = input_window_seconds
    if "add_log_softmax" in sig:
        kwargs["add_log_softmax"] = False

    for key, value in params.items():
        if key in sig:
            kwargs[key] = value

    return cls(**kwargs)


# --- Local PyTorch fallback ------------------------------------------------


class _Chomp1d(nn.Module):
    def __init__(self, chomp: int) -> None:
        super().__init__()
        self.chomp = chomp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.chomp] if self.chomp > 0 else x


class _TCNResidualBlock(nn.Module):
    def __init__(self, ch: int, kernel: int, dilation: int, drop: float) -> None:
        super().__init__()
        pad = (kernel - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, kernel, padding=pad, dilation=dilation),
            _Chomp1d(pad), nn.BatchNorm1d(ch), nn.ELU(), nn.Dropout(drop),
            nn.Conv1d(ch, ch, kernel, padding=pad, dilation=dilation),
            _Chomp1d(pad), nn.BatchNorm1d(ch), nn.ELU(), nn.Dropout(drop),
        )
        self.act = nn.ELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.net(x) + x)


class _ConvBlock(nn.Module):
    """EEGNet-style temporal/spatial/separable conv block."""

    def __init__(self, n_chans, F1, D, kern1, kern2, pool1, pool2, drop) -> None:
        super().__init__()
        F2 = F1 * D
        self.temporal = nn.Sequential(
            nn.Conv2d(1, F1, (1, kern1), padding=(0, kern1 // 2), bias=False),
            nn.BatchNorm2d(F1),
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(F1, F2, (n_chans, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F2), nn.ELU(), nn.AvgPool2d((1, pool1)), nn.Dropout(drop),
        )
        self.separable = nn.Sequential(
            nn.Conv2d(F2, F2, (1, kern2), padding=(0, kern2 // 2), groups=F2, bias=False),
            nn.Conv2d(F2, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2), nn.ELU(), nn.AvgPool2d((1, pool2)), nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, 1, C, T)
        x = self.temporal(x)
        x = self.depthwise(x)
        x = self.separable(x)
        return x  # (B, F2, 1, Tc)


class LocalATCNet(nn.Module):
    def __init__(
        self,
        n_chans: int = 64,
        n_outputs: int = 2,
        n_times: int = 512,
        *,
        n_windows: int = 5,
        conv_block_n_filters: int = 16,
        conv_block_depth_mult: int = 2,
        conv_block_kernel_length_1: int = 64,
        conv_block_kernel_length_2: int = 16,
        conv_block_pool_size_1: int = 8,
        conv_block_pool_size_2: int = 7,
        conv_block_dropout: float = 0.3,
        num_heads: int = 2,
        att_drop_prob: float = 0.5,
        tcn_depth: int = 2,
        tcn_kernel_size: int = 4,
        tcn_drop_prob: float = 0.3,
        **_ignored: Any,
    ) -> None:
        super().__init__()
        F2 = conv_block_n_filters * conv_block_depth_mult
        if F2 % num_heads != 0:
            num_heads = 1  # keep embed_dim divisible by heads
        self.n_windows = n_windows
        self.conv_block = _ConvBlock(
            n_chans, conv_block_n_filters, conv_block_depth_mult,
            conv_block_kernel_length_1, conv_block_kernel_length_2,
            conv_block_pool_size_1, conv_block_pool_size_2, conv_block_dropout,
        )
        with torch.no_grad():
            tc = self.conv_block(torch.zeros(1, 1, n_chans, n_times)).shape[-1]
        self.Tc = int(tc)
        self.Tw = self.Tc - n_windows + 1
        if self.Tw < 1:
            raise ValueError(
                f"ATCNet condensed time axis Tc={self.Tc} too small for n_windows="
                f"{n_windows} (window length {self.Tw} < 1). Reduce n_windows or "
                "conv_block_pool_size_2."
            )

        self.attn_norm = nn.LayerNorm(F2)
        self.mha = nn.MultiheadAttention(F2, num_heads, dropout=att_drop_prob, batch_first=True)
        self.attn_drop = nn.Dropout(att_drop_prob)
        self.tcn = nn.Sequential(*[
            _TCNResidualBlock(F2, tcn_kernel_size, 2 ** i, tcn_drop_prob)
            for i in range(tcn_depth)
        ])
        self.heads = nn.ModuleList([nn.Linear(F2, n_outputs) for _ in range(n_windows)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T)
        x = x.unsqueeze(1)               # (B, 1, C, T)
        x = self.conv_block(x).squeeze(2)  # (B, F2, Tc)
        x = x.permute(0, 2, 1)           # (B, Tc, F2)
        window_logits = []
        for i in range(self.n_windows):
            w = x[:, i:i + self.Tw, :]   # (B, Tw, F2)
            attn, _ = self.mha(self.attn_norm(w), self.attn_norm(w), self.attn_norm(w))
            w = w + self.attn_drop(attn)
            t = self.tcn(w.permute(0, 2, 1))  # (B, F2, Tw)
            window_logits.append(self.heads[i](t[:, :, -1]))  # (B, n_outputs)
        return torch.stack(window_logits, dim=0).mean(dim=0)


# --- Dispatcher ------------------------------------------------------------

_LOCAL_PARAM_NAMES = set(inspect.signature(LocalATCNet.__init__).parameters)


def build_atcnet(
    n_chans: int = 64,
    n_outputs: int = 2,
    n_times: int = 512,
    *,
    prefer_braindecode: bool = True,
    allow_local_fallback: bool = False,
    sfreq: float = 256.0,
    input_window_seconds: float = 2.0,
    **params: Any,
) -> tuple[nn.Module, str]:
    """Build ATCNet; return ``(model, source)`` where source is
    ``"braindecode"`` or ``"local_fallback"``.

    Braindecode is the default. The local fallback is used only when explicitly
    enabled; otherwise a Braindecode failure raises (no silent substitution).
    """
    if prefer_braindecode:
        try:
            model = build_atcnet_braindecode(
                n_chans, n_outputs, n_times=n_times, sfreq=sfreq,
                input_window_seconds=input_window_seconds, **params,
            )
            return model, "braindecode"
        except Exception as exc:  # noqa: BLE001
            if not allow_local_fallback:
                raise RuntimeError(
                    "Braindecode ATCNet construction failed and allow_local_fallback "
                    f"is False (no silent substitution): {exc}"
                ) from exc
    elif not allow_local_fallback:
        raise ValueError("prefer_braindecode=False requires allow_local_fallback=True.")

    local_params = {k: v for k, v in params.items() if k in _LOCAL_PARAM_NAMES}
    model = LocalATCNet(n_chans=n_chans, n_outputs=n_outputs, n_times=n_times, **local_params)
    return model, "local_fallback"
