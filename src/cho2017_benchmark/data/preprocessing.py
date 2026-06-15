"""Canonical, per-epoch preprocessing.

To prevent zero-phase filtering from leaking information across split
boundaries, each trial is processed *inside its own epoch* (never across the
whole recording). The order is:

    raw epoch -> common average reference -> zero-phase 8-30 Hz band-pass
    -> resample 512 -> 256 Hz -> crop the central 0.5-2.5 s task window
    -> assert final shape 64 x 512.

Because the extracted epoch (e.g. [-1.5, 4.0] s) extends >= 1 s beyond the crop
on both sides, filtfilt edge transients never reach the task window.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import gcd
from typing import Any

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

from .epoching import EpochedSubject


@dataclass(frozen=True)
class PreprocessingConfig:
    reference: str = "car"
    l_freq: float = 8.0
    h_freq: float = 30.0
    filter_method: str = "iir_butter_filtfilt"
    iir_order: int = 4
    pad_type: str = "odd"
    crop_start_s: float = 0.5
    crop_end_s: float = 2.5
    srate_in: int = 512
    srate_out: int = 256
    eps: float = 1e-7

    def hashable(self) -> dict[str, Any]:
        return dict(sorted(asdict(self).items()))

    @classmethod
    def from_config(cls, cfg: Any) -> "PreprocessingConfig":
        d = cfg.preprocessing_dict()
        return cls(
            reference=d["reference"], l_freq=d["l_freq"], h_freq=d["h_freq"],
            filter_method=d["filter_method"], iir_order=d["iir_order"],
            pad_type=d["pad_type"], crop_start_s=d["crop_start_s"],
            crop_end_s=d["crop_end_s"], srate_in=d["srate_in"],
            srate_out=d["srate_out"], eps=d["eps"],
        )

    def expected_n_times(self) -> int:
        return int(round((self.crop_end_s - self.crop_start_s) * self.srate_out))


def common_average_reference(x: np.ndarray) -> np.ndarray:
    """Subtract the across-channel mean at each time sample (channel axis = -2)."""
    return x - x.mean(axis=-2, keepdims=True)


def bandpass_zero_phase(x: np.ndarray, srate: float, pp: PreprocessingConfig) -> np.ndarray:
    """Zero-phase Butterworth band-pass along the last axis (per epoch)."""
    sos = butter(
        pp.iir_order, [pp.l_freq, pp.h_freq], btype="band", fs=srate, output="sos"
    )
    return sosfiltfilt(sos, x, axis=-1, padtype=pp.pad_type)


def resample_signal(x: np.ndarray, srate_in: int, srate_out: int) -> np.ndarray:
    """Polyphase resample along the last axis."""
    if srate_in == srate_out:
        return x
    g = gcd(int(srate_out), int(srate_in))
    return resample_poly(x, int(srate_out) // g, int(srate_in) // g, axis=-1)


def crop_window(
    x: np.ndarray, *, epoch_tmin_s: float, srate_out: int, pp: PreprocessingConfig
) -> np.ndarray:
    """Crop ``[crop_start_s, crop_end_s)`` relative to onset from a resampled epoch."""
    start = int(round((pp.crop_start_s - epoch_tmin_s) * srate_out))
    n_times = pp.expected_n_times()
    end = start + n_times
    if start < 0 or end > x.shape[-1]:
        raise ValueError(
            f"Crop [{pp.crop_start_s}, {pp.crop_end_s}] s (samples {start}:{end}) "
            f"does not fit a resampled epoch of length {x.shape[-1]} "
            f"(epoch_tmin={epoch_tmin_s} s)."
        )
    return x[..., start:end]


def preprocess_epochs(
    epoched: EpochedSubject,
    pp: PreprocessingConfig,
    resolution: dict[str, Any] | None = None,
) -> np.ndarray:
    """Run the canonical per-epoch pipeline on the EEG; return ``(n, 64, n_times)``."""
    if epoched.srate != pp.srate_in:
        raise ValueError(
            f"Epoch srate {epoched.srate} != preprocessing srate_in {pp.srate_in}."
        )
    if resolution is not None:
        crop = resolution.get("primary_crop_relative_to_event_seconds")
        if crop and (abs(crop[0] - pp.crop_start_s) > 1e-6 or abs(crop[1] - pp.crop_end_s) > 1e-6):
            raise ValueError(
                f"Config crop [{pp.crop_start_s}, {pp.crop_end_s}] disagrees with "
                f"resolved primary crop {crop}."
            )

    x = epoched.X_eeg.astype(np.float64)
    if pp.reference == "car":
        x = common_average_reference(x)
    elif pp.reference not in ("none", None):
        raise ValueError(f"Unknown reference '{pp.reference}'.")
    x = bandpass_zero_phase(x, epoched.srate, pp)
    x = resample_signal(x, pp.srate_in, pp.srate_out)
    x = crop_window(x, epoch_tmin_s=epoched.epoch_tmin_s, srate_out=pp.srate_out, pp=pp)

    expected = pp.expected_n_times()
    if x.shape[1:] != (epoched.X_eeg.shape[1], expected):
        raise AssertionError(
            f"Processed EEG shape {x.shape} != (n, {epoched.X_eeg.shape[1]}, {expected})."
        )
    return np.ascontiguousarray(x, dtype=np.float32)
