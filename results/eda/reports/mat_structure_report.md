# Cho2017 .mat structure inspection report

Bu rapor, ham `.mat` dosyalarinin gercek yapisini belgeler. Loader/epoching varsayimlari bu gozlemlere ve `mat_layout_resolution.json` dosyasina dayanir; asla tahmin edilmez.

## Resolved layout (machine-readable mirror)

- **validated**: `True`
- **srate**: `512.0`
- **n_channels**: `68`
- **orientation**: `channels_by_time`
- **event_representation**: `binary_marker_vector`
- **event_semantics**: `event marks the cue; stored interval includes pre-stimulus baseline (starts at -2.0 s)`
- **time_zero_definition**: `imagery_event onset sample (t = 0)`
- **imagery_interval_relative_to_event_seconds**: `[-2.0, 5.0]`
- **primary_crop_relative_to_event_seconds**: `[0.5, 2.5]`
- **index_base**: `one_based`
- **bad_trial_scope**: `class_specific`
- **rejection_reason_mapping**: `{'bad_trial_idx_voltage': 'amplitude', 'bad_trial_idx_mi': 'emg_correlation'}`
- **run_metadata_available**: `False`
- **pre_stimulus_available**: `True`
- **pre_stimulus_samples**: `1024`

### Justifications

- **srate**: Observed srate field across inspected files: [512.0].
- **orientation**: imagery_left shapes had ndim=[2] with channel axis [0] (channel count [68]).
- **event_representation**: imagery_event length matched the signal time axis in all inspected files -> binary onset-marker vector; onsets = nonzero positions.
- **frame**: frame [-2000.0, 5000.0] interpreted as milliseconds (per-trial window ~3584 samples vs onset spacing 3584); imagery interval ~[-2.000, 5.000] s relative to event.
- **event_semantics**: Derived from the frame interval relative to the event; not assumed to be imagery onset without this evidence.
- **bad_trial_scope**: Derived from bad_trial_indices structure (subfields=['bad_trial_idx_voltage', 'bad_trial_idx_mi']).
- **index_base**: Compared max bad-trial index against n_imagery_trials: max==n -> one-based, max==n-1 with a zero present -> zero-based.
- **rejection_reason_mapping**: bad_trial_indices subfields ['bad_trial_idx_voltage', 'bad_trial_idx_mi'] mapped to reasons via name heuristics (volt/amp->amplitude, emg/mi/corr->emg_correlation).
- **run_metadata**: No explicit run field found in eeg struct; runs must be recovered or marked unrecoverable.

### Warnings

- :warning: Could not determine bad-trial index base; defaulting to one_based.

## Per-subject inspection

### s01
- top-level keys: `['eeg']`
- eeg fields: `['noise', 'rest', 'srate', 'movement_left', 'movement_right', 'movement_event', 'n_movement_trials', 'imagery_left', 'imagery_right', 'n_imagery_trials', 'frame', 'imagery_event', 'comment', 'subject', 'bad_trial_indices', 'psenloc', 'senloc']`
- srate: `512.0`
- n_imagery_trials: `100`
- imagery_left shape: `[68, 358400]`, imagery_right shape: `[68, 358400]`
- channel axis: `0`, n_channels: `68`, time samples: `358400`, duration_s: `700.0`
- imagery_event: nonzero=100, length=358400, len_matches_time=True
- frame: `[-2000.0, 5000.0]`
- bad_trial_indices: `{"kind": "struct", "fields": {"bad_trial_idx_voltage": {"kind": "cell", "shape": [2], "cells": [{"kind": "array", "shape": [0], "dtype": "uint8"}, {"kind": "array", "shape": [0], "dtype": "uint8"}]}, "bad_trial_idx_mi": {"kind": "cell", "shape": [2], "cells": [{"kind": "array", "shape": [0], "dtype": "uint8"}, {"kind": "array", "shape": [0], "dtype": "uint8"}]}}}`
