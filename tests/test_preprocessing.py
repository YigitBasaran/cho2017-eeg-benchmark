import numpy as np

from cho2017_benchmark.data.datasets import apply_normalization, compute_normalization
from cho2017_benchmark.data.preprocessing import (
    PreprocessingConfig,
    bandpass_zero_phase,
    common_average_reference,
    preprocess_epochs,
)


def test_preprocess_epochs_shape(tiny_epoched):
    pp = PreprocessingConfig(srate_in=128, srate_out=64, crop_start_s=0.0, crop_end_s=0.5,
                             l_freq=8.0, h_freq=30.0)
    X = preprocess_epochs(tiny_epoched, pp)
    assert X.shape == (8, 64, pp.expected_n_times())
    assert X.dtype == np.float32
    assert np.isfinite(X).all()


def test_zero_phase_filter_changes_signal(tiny_epoched):
    pp = PreprocessingConfig(srate_in=128, srate_out=128, l_freq=8.0, h_freq=30.0)
    filt = bandpass_zero_phase(tiny_epoched.X_eeg.astype(np.float64), 128, pp)
    assert filt.shape == tiny_epoched.X_eeg.shape
    assert not np.allclose(filt, tiny_epoched.X_eeg)


def test_common_average_reference():
    x = np.random.RandomState(0).randn(2, 64, 16)
    xc = common_average_reference(x)
    assert np.allclose(xc.mean(axis=-2), 0.0, atol=1e-9)


def test_train_only_normalization():
    rng = np.random.RandomState(0)
    X_train = (rng.randn(10, 64, 32) * 5 + 3).astype(np.float32)
    stats = compute_normalization(X_train)
    assert stats.mean.shape == (64, 1) and stats.std.shape == (64, 1)
    Xn = apply_normalization(X_train, stats)
    assert abs(float(Xn.mean())) < 1e-4
    assert abs(float(Xn.std()) - 1.0) < 0.05
    # frozen stats applied to "test" data are NOT recomputed from it
    X_test = (rng.randn(5, 64, 32) * 9 + 11).astype(np.float32)
    Xt = apply_normalization(X_test, stats)
    assert not np.allclose(Xt.mean(), 0.0, atol=1e-2)
