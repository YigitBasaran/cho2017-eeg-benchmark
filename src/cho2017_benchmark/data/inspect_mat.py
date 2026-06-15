"""Structural inspection of the raw ``.mat`` files and layout resolution.

This module never *guesses* how the dataset is laid out. It inspects real files
and writes two artifacts:

- ``results/eda/reports/mat_structure_report.md`` -- a human-readable report of
  what each inspected file actually contains;
- ``data/metadata/mat_layout_resolution.json`` -- a machine-readable resolution
  of the load-bearing structural facts (array orientation, event semantics,
  MATLAB index base, bad-trial scope, rejection-reason mapping, run-metadata
  availability, pre-stimulus availability), each accompanied by a justification.

The resolution file is a hard gate: :func:`load_layout_resolution` is consumed
by :mod:`cho2017_benchmark.data.prepare`, which refuses to run without a present,
validated resolution.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .. import paths
from .download import subject_id_str

EXPECTED_SRATE = 512
N_EEG = 64
N_EMG = 4
N_TOTAL = 68
RESOLUTION_FILENAME = "mat_layout_resolution.json"
REPORT_FILENAME = "mat_structure_report.md"

# Subjects worth inspecting explicitly (typical, excluded, and boundary ids).
DEFAULT_INSPECTION_SUBJECTS = (1, 20, 29, 33, 34, 52)


# --- Low-level introspection ----------------------------------------------


def _is_struct(obj: Any) -> bool:
    return hasattr(obj, "_fieldnames")


def _describe_value(value: Any, max_items: int = 24) -> dict[str, Any]:
    """Describe an arbitrary MATLAB value (struct / array / scalar)."""
    if _is_struct(value):
        return {
            "kind": "struct",
            "fields": {f: _describe_value(getattr(value, f), max_items)
                       for f in value._fieldnames},
        }
    arr = np.asarray(value)
    if arr.dtype == object:
        # cell array: describe each cell shallowly
        flat = arr.ravel()
        cells = [_describe_value(c, max_items) for c in flat[:max_items]]
        return {"kind": "cell", "shape": list(arr.shape), "cells": cells}
    out: dict[str, Any] = {"kind": "array", "shape": list(arr.shape), "dtype": str(arr.dtype)}
    if arr.dtype.kind in "iuf" and arr.size > 0:
        out["min"] = float(np.nanmin(arr))
        out["max"] = float(np.nanmax(arr))
        out["n_nonzero"] = int(np.count_nonzero(arr))
        if arr.size <= max_items:
            out["values"] = arr.ravel().tolist()
    return out


def _shape(value: Any) -> list[int]:
    try:
        return list(np.asarray(value).shape)
    except Exception:
        return []


def _channel_axis(shape: Sequence[int]) -> int | None:
    """Index of the axis whose length matches the EEG/EMG channel count."""
    for axis, size in enumerate(shape):
        if size in (N_EEG, N_TOTAL):
            return axis
    return None


# --- Per-subject inspection ------------------------------------------------


@dataclass
class MatInspection:
    subject_id: str
    path: str
    ok: bool
    error: str | None = None
    top_keys: list[str] = field(default_factory=list)
    eeg_fields: list[str] = field(default_factory=list)
    field_shapes: dict[str, list[int]] = field(default_factory=dict)
    field_dtypes: dict[str, str] = field(default_factory=dict)
    srate: float | None = None
    n_imagery_trials: int | None = None
    imagery_left_shape: list[int] = field(default_factory=list)
    imagery_right_shape: list[int] = field(default_factory=list)
    n_channels: int | None = None
    channel_axis: int | None = None
    time_axis_len: int | None = None
    duration_s: float | None = None
    imagery_event_desc: dict[str, Any] = field(default_factory=dict)
    n_event_onsets_binary: int | None = None
    n_event_onsets_index: int | None = None
    event_len_matches_time: bool | None = None
    onset_spacing_median: int | None = None
    first_onsets: list[int] = field(default_factory=list)
    frame_values: list[float] = field(default_factory=list)
    bad_trial_desc: dict[str, Any] = field(default_factory=dict)
    has_rest: bool = False
    has_noise: bool = False
    notes: list[str] = field(default_factory=list)


def inspect_subject(path: Path | str) -> MatInspection:
    """Inspect one ``sNN.mat`` file and return structured observations."""
    path = Path(path)
    sid = path.stem
    insp = MatInspection(subject_id=sid, path=str(path), ok=False)
    try:
        from scipy.io import loadmat

        mat = loadmat(str(path), struct_as_record=False, squeeze_me=True)
    except Exception as exc:  # noqa: BLE001
        insp.error = f"loadmat failed: {exc}"
        return insp

    insp.top_keys = [k for k in mat if not k.startswith("__")]
    if "eeg" not in mat:
        insp.error = "no 'eeg' struct present"
        return insp
    eeg = mat["eeg"]
    if not _is_struct(eeg):
        insp.error = "'eeg' is not a struct"
        return insp

    insp.eeg_fields = list(eeg._fieldnames)
    for name in insp.eeg_fields:
        value = getattr(eeg, name)
        insp.field_shapes[name] = _shape(value)
        try:
            insp.field_dtypes[name] = str(np.asarray(value).dtype)
        except Exception:
            insp.field_dtypes[name] = type(value).__name__

    insp.has_rest = "rest" in insp.eeg_fields
    insp.has_noise = "noise" in insp.eeg_fields

    if "srate" in insp.eeg_fields:
        try:
            insp.srate = float(np.asarray(getattr(eeg, "srate")).ravel()[0])
        except Exception:
            insp.srate = None

    if "n_imagery_trials" in insp.eeg_fields:
        try:
            insp.n_imagery_trials = int(np.asarray(getattr(eeg, "n_imagery_trials")).ravel()[0])
        except Exception:
            insp.n_imagery_trials = None

    if "imagery_left" in insp.eeg_fields:
        insp.imagery_left_shape = _shape(getattr(eeg, "imagery_left"))
    if "imagery_right" in insp.eeg_fields:
        insp.imagery_right_shape = _shape(getattr(eeg, "imagery_right"))

    shape = insp.imagery_left_shape or insp.imagery_right_shape
    if shape:
        axis = _channel_axis(shape)
        insp.channel_axis = axis
        if axis is not None:
            insp.n_channels = shape[axis]
            time_axes = [s for i, s in enumerate(shape) if i != axis]
            insp.time_axis_len = max(time_axes) if time_axes else None
            if insp.srate and insp.time_axis_len:
                insp.duration_s = insp.time_axis_len / insp.srate

    if "imagery_event" in insp.eeg_fields:
        event = np.asarray(getattr(eeg, "imagery_event"))
        insp.imagery_event_desc = _describe_value(getattr(eeg, "imagery_event"))
        flat = event.ravel()
        insp.n_event_onsets_binary = int(np.count_nonzero(flat))
        insp.n_event_onsets_index = int(flat.size)
        if insp.time_axis_len is not None:
            insp.event_len_matches_time = bool(abs(flat.size - insp.time_axis_len) <= 1)
        onsets = np.flatnonzero(flat)
        insp.first_onsets = onsets[:8].astype(int).tolist()
        if onsets.size > 1:
            insp.onset_spacing_median = int(np.median(np.diff(onsets)))

    if "frame" in insp.eeg_fields:
        try:
            insp.frame_values = np.asarray(getattr(eeg, "frame")).ravel().astype(float).tolist()
        except Exception:
            insp.frame_values = []

    if "bad_trial_indices" in insp.eeg_fields:
        insp.bad_trial_desc = _describe_value(getattr(eeg, "bad_trial_indices"))

    insp.ok = True
    return insp


def inspect_files(paths_or_indices: Sequence[Path | str | int]) -> list[MatInspection]:
    out: list[MatInspection] = []
    for item in paths_or_indices:
        if isinstance(item, int):
            p = paths.raw_dir() / f"{subject_id_str(item)}.mat"
        else:
            p = Path(item)
        if not p.exists():
            out.append(MatInspection(subject_id=p.stem, path=str(p), ok=False,
                                     error="file not found"))
            continue
        out.append(inspect_subject(p))
    return out


# --- Layout resolution -----------------------------------------------------


@dataclass
class LayoutResolution:
    """Resolved structural facts, each with a justification."""

    validated: bool
    inspected_subjects: list[str]
    srate: float | None
    n_channels: int | None
    orientation: str           # e.g. "channels_by_time" | "time_by_channels" | "channels_time_trials" | "unknown"
    event_representation: str  # "binary_marker_vector" | "onset_index_array" | "unknown"
    event_semantics: str
    time_zero_definition: str
    imagery_interval_relative_to_event_seconds: list[float] | None
    primary_crop_relative_to_event_seconds: list[float]
    index_base: str            # "zero_based" | "one_based" | "unknown"
    bad_trial_scope: str       # "class_specific" | "global" | "unknown"
    rejection_reason_mapping: dict[str, str]
    run_metadata_available: bool
    pre_stimulus_available: bool
    pre_stimulus_samples: int | None
    justification: dict[str, str]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_layout(
    inspections: Sequence[MatInspection],
    *,
    primary_crop: tuple[float, float] = (0.5, 2.5),
) -> LayoutResolution:
    """Aggregate per-file observations into a justified layout resolution."""
    ok = [i for i in inspections if i.ok]
    warnings: list[str] = []
    just: dict[str, str] = {}
    if not ok:
        return LayoutResolution(
            validated=False, inspected_subjects=[i.subject_id for i in inspections],
            srate=None, n_channels=None, orientation="unknown",
            event_representation="unknown", event_semantics="unknown",
            time_zero_definition="unknown",
            imagery_interval_relative_to_event_seconds=None,
            primary_crop_relative_to_event_seconds=list(primary_crop),
            index_base="unknown", bad_trial_scope="unknown",
            rejection_reason_mapping={}, run_metadata_available=False,
            pre_stimulus_available=False, pre_stimulus_samples=None,
            justification={"error": "no files could be inspected"},
            warnings=["No inspectable files; resolution not validated."],
        )

    # -- sampling rate --
    srates = sorted({i.srate for i in ok if i.srate is not None})
    srate = srates[0] if len(srates) == 1 else None
    if srate is None:
        warnings.append(f"Inconsistent/unknown srate across files: {srates}")
    elif int(srate) != EXPECTED_SRATE:
        warnings.append(f"srate {srate} != expected {EXPECTED_SRATE}")
    just["srate"] = f"Observed srate field across inspected files: {srates}."

    # -- channels / orientation --
    n_channels_vals = sorted({i.n_channels for i in ok if i.n_channels is not None})
    n_channels = n_channels_vals[0] if len(n_channels_vals) == 1 else None
    axes_set = {i.channel_axis for i in ok if i.channel_axis is not None}
    ndims_set = {len(i.imagery_left_shape) for i in ok if i.imagery_left_shape}
    if ndims_set == {2} and axes_set == {0}:
        orientation = "channels_by_time"
    elif ndims_set == {2} and axes_set == {1}:
        orientation = "time_by_channels"
    elif ndims_set == {3}:
        orientation = "channels_time_trials"
    else:
        orientation = "unknown"
        warnings.append(
            f"Ambiguous imagery array orientation (ndims={sorted(ndims_set)}, "
            f"ch_axis={sorted(axes_set)})."
        )
    just["orientation"] = (
        f"imagery_left shapes had ndim={sorted(ndims_set)} with channel axis "
        f"{sorted(axes_set)} (channel count {n_channels_vals})."
    )
    if n_channels not in (N_EEG, N_TOTAL, None):
        warnings.append(f"Unexpected channel count {n_channels} (expected {N_EEG} or {N_TOTAL}).")

    # -- event representation --
    matches = [i.event_len_matches_time for i in ok if i.event_len_matches_time is not None]
    if matches and all(matches):
        event_representation = "binary_marker_vector"
        just["event_representation"] = (
            "imagery_event length matched the signal time axis in all inspected "
            "files -> binary onset-marker vector; onsets = nonzero positions."
        )
    elif matches and not any(matches):
        event_representation = "onset_index_array"
        just["event_representation"] = (
            "imagery_event length did NOT match the time axis -> treated as an "
            "array of onset sample indices."
        )
    else:
        event_representation = "unknown"
        just["event_representation"] = "Could not determine imagery_event representation."
        warnings.append("imagery_event representation ambiguous; verify before epoching.")

    # number of onsets vs n_imagery_trials (sanity, not assumed)
    for i in ok:
        nt = i.n_imagery_trials
        n_on = (i.n_event_onsets_binary if event_representation == "binary_marker_vector"
                else i.n_event_onsets_index)
        if nt is not None and n_on is not None and event_representation != "unknown":
            # n_on may equal trials per class or total; record discrepancies
            if n_on not in (nt, 2 * nt):
                warnings.append(
                    f"{i.subject_id}: onset count {n_on} not in (n_imagery_trials={nt}, 2x)."
                )

    # -- frame -> pre-stimulus + imagery interval --
    # frame is a [start, end] pair whose units (ms / s / samples) are inferred by
    # checking which interpretation produces a per-trial window that fits within
    # the measured onset spacing (a window wider than the spacing would mean
    # overlapping trials, which is impossible for clean epochs).
    pre_stimulus_available = False
    pre_stimulus_samples: int | None = None
    imagery_interval: list[float] | None = None
    spacings = [i.onset_spacing_median for i in ok if i.onset_spacing_median]
    spacing = int(np.median(spacings)) if spacings else None
    frames = [i.frame_values for i in ok if i.frame_values]
    if frames and len(frames[0]) >= 2 and srate:
        f0 = frames[0]
        start, end = float(f0[0]), float(f0[-1])
        width = end - start
        candidates = {
            "milliseconds": (width / 1000.0 * srate, start / 1000.0, end / 1000.0),
            "seconds": (width * srate, start, end),
            "samples": (width, start / srate, end / srate),
        }
        chosen: tuple[str, float, float, float] | None = None
        for unit, (wsamp, s_s, e_s) in candidates.items():
            if wsamp <= 0:
                continue
            if spacing is None or wsamp <= spacing * 1.02:
                if chosen is None or wsamp > chosen[1]:
                    chosen = (unit, wsamp, s_s, e_s)
        if chosen is None:
            wsamp, s_s, e_s = candidates["milliseconds"]
            unit = "milliseconds"
            warnings.append("frame units did not fit onset spacing; assumed milliseconds.")
        else:
            unit, wsamp, s_s, e_s = chosen
        imagery_interval = [round(s_s, 4), round(e_s, 4)]
        if s_s < 0:
            pre_stimulus_available = True
            pre_stimulus_samples = int(round(abs(s_s) * srate))
        just["frame"] = (
            f"frame {f0} interpreted as {unit} (per-trial window ~{int(wsamp)} samples "
            f"vs onset spacing {spacing}); imagery interval ~[{s_s:.3f}, {e_s:.3f}] s "
            "relative to event."
        )
    else:
        warnings.append("No usable frame/onset info; imagery interval / pre-stimulus unverified.")
        just["frame"] = "frame field absent or unusable."

    # event semantics / time zero (verified against frame, NOT assumed)
    if imagery_interval is not None and abs(imagery_interval[0]) < 1e-6:
        event_semantics = "event marks the start of the stored imagery interval (t=0)"
    elif imagery_interval is not None and imagery_interval[0] < 0:
        event_semantics = (
            "event marks the cue; stored interval includes pre-stimulus baseline "
            f"(starts at {imagery_interval[0]} s)"
        )
    else:
        event_semantics = "event onset (exact cue-vs-imagery relationship unverified)"
        warnings.append("imagery_event semantics not fully verified against frame.")
    time_zero_definition = "imagery_event onset sample (t = 0)"
    just["event_semantics"] = (
        "Derived from the frame interval relative to the event; not assumed to be "
        "imagery onset without this evidence."
    )

    # -- bad_trial_indices scope / index base / reason mapping --
    bad_scope, index_base, reason_map, bad_just, bad_warn = _resolve_bad_trials(ok)
    just["bad_trial_scope"] = bad_just.get("scope", "")
    just["index_base"] = bad_just.get("index_base", "")
    just["rejection_reason_mapping"] = bad_just.get("reason", "")
    warnings.extend(bad_warn)

    # -- run metadata --
    run_fields = {"run", "run_id", "runs", "n_runs"}
    run_metadata_available = any(
        any(fname in run_fields for fname in i.eeg_fields) for i in ok
    )
    just["run_metadata"] = (
        "No explicit run field found in eeg struct; runs must be recovered or "
        "marked unrecoverable." if not run_metadata_available
        else "Explicit run metadata present in eeg struct."
    )

    validated = (
        orientation != "unknown"
        and event_representation != "unknown"
        and srate is not None
        and n_channels in (N_EEG, N_TOTAL)
    )
    if not validated:
        warnings.append(
            "Resolution NOT validated: orientation/event/srate/channels not all "
            "confidently determined. Review the report before preparing data."
        )

    return LayoutResolution(
        validated=validated,
        inspected_subjects=[i.subject_id for i in ok],
        srate=srate,
        n_channels=n_channels,
        orientation=orientation,
        event_representation=event_representation,
        event_semantics=event_semantics,
        time_zero_definition=time_zero_definition,
        imagery_interval_relative_to_event_seconds=imagery_interval,
        primary_crop_relative_to_event_seconds=list(primary_crop),
        index_base=index_base,
        bad_trial_scope=bad_scope,
        rejection_reason_mapping=reason_map,
        run_metadata_available=run_metadata_available,
        pre_stimulus_available=pre_stimulus_available,
        pre_stimulus_samples=pre_stimulus_samples,
        justification=just,
        warnings=warnings,
    )


def _resolve_bad_trials(
    inspections: Sequence[MatInspection],
) -> tuple[str, str, dict[str, str], dict[str, str], list[str]]:
    """Resolve scope, index base and reason mapping of bad_trial_indices."""
    warnings: list[str] = []
    just: dict[str, str] = {}

    structs = [i.bad_trial_desc for i in inspections if i.bad_trial_desc]
    if not structs:
        return ("unknown", "unknown", {},
                {"scope": "bad_trial_indices absent or undescribed."}, warnings)

    sample = structs[0]
    # Collect subfield names if it is a struct.
    subfields: list[str] = []
    if sample.get("kind") == "struct":
        subfields = list(sample.get("fields", {}).keys())

    # Reason mapping from subfield names.
    reason_map: dict[str, str] = {}
    for name in subfields:
        low = name.lower()
        if "volt" in low or "amp" in low:
            reason_map[name] = "amplitude"
        elif "emg" in low or "mi" in low or "corr" in low:
            reason_map[name] = "emg_correlation"
        else:
            reason_map[name] = "provided_flag"
    just["reason"] = (
        f"bad_trial_indices subfields {subfields} mapped to reasons via name "
        "heuristics (volt/amp->amplitude, emg/mi/corr->emg_correlation)."
        if subfields else "bad_trial_indices is a flat array; single reason 'provided_flag'."
    )

    # Scope: nested left/right -> class-specific; otherwise inspect inner shapes.
    scope = "unknown"
    nested_names = " ".join(subfields).lower()
    if any(k in nested_names for k in ("left", "right")):
        scope = "class_specific"
    elif subfields:
        # subfields like voltage/mi each may hold {left,right} cells -> class-specific
        inner = sample.get("fields", {})
        inner_is_cell = any(v.get("kind") == "cell" for v in inner.values())
        scope = "class_specific" if inner_is_cell else "global"
    else:
        scope = "global"
    just["scope"] = (
        f"Derived from bad_trial_indices structure (subfields={subfields})."
    )
    if scope == "unknown":
        warnings.append("bad_trial scope could not be determined.")

    # Index base: compare max index against per-class trial counts.
    index_base = _guess_index_base(inspections, warnings)
    just["index_base"] = (
        "Compared max bad-trial index against n_imagery_trials: max==n -> one-based, "
        "max==n-1 with a zero present -> zero-based."
    )
    return scope, index_base, reason_map, just, warnings


def _collect_index_extremes(desc: dict[str, Any], mins: list[float], maxs: list[float]) -> None:
    if not isinstance(desc, dict):
        return
    kind = desc.get("kind")
    if kind == "array":
        if "min" in desc and "max" in desc and desc.get("max", 0) > 0:
            mins.append(desc["min"])
            maxs.append(desc["max"])
    elif kind == "struct":
        for v in desc.get("fields", {}).values():
            _collect_index_extremes(v, mins, maxs)
    elif kind == "cell":
        for v in desc.get("cells", []):
            _collect_index_extremes(v, mins, maxs)


def _guess_index_base(inspections: Sequence[MatInspection], warnings: list[str]) -> str:
    votes_one = 0
    votes_zero = 0
    for i in inspections:
        nt = i.n_imagery_trials
        if not nt or not i.bad_trial_desc:
            continue
        mins: list[float] = []
        maxs: list[float] = []
        _collect_index_extremes(i.bad_trial_desc, mins, maxs)
        if not maxs:
            continue
        gmax = max(maxs)
        gmin = min(mins) if mins else None
        if gmax == nt:
            votes_one += 1
        elif gmax == nt - 1 and gmin == 0:
            votes_zero += 1
        elif gmax <= nt - 1 and (gmin is not None and gmin >= 1):
            votes_one += 1  # 1..n-? still within 1-based range, no zero seen
    if votes_one and not votes_zero:
        return "one_based"
    if votes_zero and not votes_one:
        return "zero_based"
    if votes_one or votes_zero:
        warnings.append(
            f"Index base ambiguous (one-based votes={votes_one}, zero-based={votes_zero}); "
            "defaulting to one_based (MATLAB convention)."
        )
        return "one_based"
    warnings.append("Could not determine bad-trial index base; defaulting to one_based.")
    return "one_based"


# --- Report / resolution IO ------------------------------------------------


def render_report(inspections: Sequence[MatInspection], resolution: LayoutResolution) -> str:
    lines: list[str] = []
    lines.append("# Cho2017 .mat structure inspection report\n")
    lines.append(
        "Bu rapor, ham `.mat` dosyalarinin gercek yapisini belgeler. Loader/epoching "
        "varsayimlari bu gozlemlere ve `mat_layout_resolution.json` dosyasina dayanir; "
        "asla tahmin edilmez.\n"
    )
    lines.append("## Resolved layout (machine-readable mirror)\n")
    res = resolution.to_dict()
    for key in (
        "validated", "srate", "n_channels", "orientation", "event_representation",
        "event_semantics", "time_zero_definition",
        "imagery_interval_relative_to_event_seconds",
        "primary_crop_relative_to_event_seconds", "index_base", "bad_trial_scope",
        "rejection_reason_mapping", "run_metadata_available",
        "pre_stimulus_available", "pre_stimulus_samples",
    ):
        lines.append(f"- **{key}**: `{res[key]}`")
    lines.append("\n### Justifications\n")
    for key, val in resolution.justification.items():
        lines.append(f"- **{key}**: {val}")
    if resolution.warnings:
        lines.append("\n### Warnings\n")
        for w in resolution.warnings:
            lines.append(f"- :warning: {w}")

    lines.append("\n## Per-subject inspection\n")
    for insp in inspections:
        lines.append(f"### {insp.subject_id}")
        if not insp.ok:
            lines.append(f"- ERROR: {insp.error}\n")
            continue
        lines.append(f"- top-level keys: `{insp.top_keys}`")
        lines.append(f"- eeg fields: `{insp.eeg_fields}`")
        lines.append(f"- srate: `{insp.srate}`")
        lines.append(f"- n_imagery_trials: `{insp.n_imagery_trials}`")
        lines.append(f"- imagery_left shape: `{insp.imagery_left_shape}`, "
                     f"imagery_right shape: `{insp.imagery_right_shape}`")
        lines.append(f"- channel axis: `{insp.channel_axis}`, n_channels: `{insp.n_channels}`, "
                     f"time samples: `{insp.time_axis_len}`, duration_s: `{insp.duration_s}`")
        lines.append(f"- imagery_event: nonzero={insp.n_event_onsets_binary}, "
                     f"length={insp.n_event_onsets_index}, "
                     f"len_matches_time={insp.event_len_matches_time}")
        lines.append(f"- frame: `{insp.frame_values}`")
        lines.append(f"- bad_trial_indices: `{json.dumps(insp.bad_trial_desc)[:600]}`")
        lines.append("")
    return "\n".join(lines)


def write_report(
    inspections: Sequence[MatInspection],
    resolution: LayoutResolution,
    dest: Path | str | None = None,
) -> Path:
    dest = Path(dest) if dest is not None else paths.eda_dir() / "reports" / REPORT_FILENAME
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(render_report(inspections, resolution), encoding="utf-8")
    return dest


def write_layout_resolution(
    resolution: LayoutResolution, dest: Path | str | None = None
) -> Path:
    dest = Path(dest) if dest is not None else paths.metadata_dir() / RESOLUTION_FILENAME
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(resolution.to_dict(), indent=2), encoding="utf-8")
    return dest


REQUIRED_RESOLUTION_KEYS = (
    "validated", "orientation", "event_representation", "event_semantics",
    "time_zero_definition", "imagery_interval_relative_to_event_seconds",
    "primary_crop_relative_to_event_seconds", "index_base", "bad_trial_scope",
    "rejection_reason_mapping", "run_metadata_available", "pre_stimulus_available",
)


def load_layout_resolution(
    path: Path | str | None = None, *, require_validated: bool = True
) -> dict[str, Any]:
    """Load the resolution file; raise if missing/invalid (the prepare gate)."""
    path = Path(path) if path is not None else paths.metadata_dir() / RESOLUTION_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Layout resolution file not found: {path}. Run inspect_mat (e.g. "
            "scripts/download_dataset.py --inspect or scripts/validate_dataset.py) "
            "to generate it before preparing data."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED_RESOLUTION_KEYS if k not in data]
    if missing:
        raise ValueError(f"Layout resolution {path} is missing keys: {missing}")
    if require_validated and not data.get("validated", False):
        raise ValueError(
            f"Layout resolution {path} is not validated (validated=false). Review "
            f"{REPORT_FILENAME} and the warnings before preparing data."
        )
    return data


def inspect_and_resolve(
    subjects: Sequence[int] | None = None,
    *,
    write: bool = True,
    primary_crop: tuple[float, float] = (0.5, 2.5),
) -> tuple[list[MatInspection], LayoutResolution]:
    """Inspect the default (or given) subjects and resolve the layout."""
    subjects = list(subjects) if subjects is not None else list(DEFAULT_INSPECTION_SUBJECTS)
    inspections = inspect_files(subjects)
    resolution = resolve_layout(inspections, primary_crop=primary_crop)
    if write:
        write_report(inspections, resolution)
        write_layout_resolution(resolution)
    return inspections, resolution
