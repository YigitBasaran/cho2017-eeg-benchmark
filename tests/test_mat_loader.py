import numpy as np
import pytest

from cho2017_benchmark.data import loader


def test_loader_returns_expected_fields(tiny_mat, tiny_resolution):
    rec = loader.load_subject(tiny_mat, resolution=tiny_resolution)
    assert rec.imagery_left.shape == (68, 240)
    assert rec.imagery_right.shape == (68, 240)
    assert rec.imagery_event.shape[0] == 240
    assert rec.srate == 128
    assert rec.n_imagery_trials == 4
    assert "bad_trial_idx_voltage" in rec.bad_trial_indices
    assert "bad_trial_idx_mi" in rec.bad_trial_indices


def test_eeg_emg_split(tiny_mat, tiny_resolution):
    rec = loader.load_subject(tiny_mat, resolution=tiny_resolution)
    eeg, emg = loader.split_eeg_emg(rec.imagery_left)
    assert eeg.shape[0] == 64
    assert emg.shape[0] == 4
    # EEG must be the first 64 rows, EMG the next 4.
    assert np.allclose(eeg, rec.imagery_left[:64])
    assert np.allclose(emg, rec.imagery_left[64:68])


def test_split_requires_68_channels():
    with pytest.raises(ValueError):
        loader.split_eeg_emg(np.zeros((10, 5)))


def test_channel_names():
    names = loader.load_channel_names()
    assert len(names["eeg"]) == 64 and len(names["emg"]) == 4
    assert "C3" in names["eeg"] and "C4" in names["eeg"] and "Cz" in names["eeg"]
