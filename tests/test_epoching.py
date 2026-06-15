from cho2017_benchmark.data import epoching, loader


def test_onset_count_matches_trials(tiny_mat, tiny_resolution):
    rec = loader.load_subject(tiny_mat, resolution=tiny_resolution)
    onsets = epoching.find_onsets(rec.imagery_event, tiny_resolution)
    assert len(onsets) == rec.n_imagery_trials == 4


def test_epoch_length_and_labels(tiny_mat, tiny_resolution):
    rec = loader.load_subject(tiny_mat, resolution=tiny_resolution)
    ep = epoching.epoch_subject(rec, tmin_s=-0.1, tmax_s=0.2, resolution=tiny_resolution)
    expected = round(0.1 * 128) + round(0.2 * 128)
    assert ep.n_epoch_samples == expected
    assert ep.X_eeg.shape == (8, 64, expected)
    assert ep.X_emg.shape == (8, 4, expected)
    assert (ep.y == 0).sum() == 4 and (ep.y == 1).sum() == 4


def test_trial_id_deterministic():
    assert epoching.make_trial_id("s01", "left_hand", 0) == "s01_left_hand_000"
    assert epoching.make_trial_id("s07", "right_hand", 12) == "s07_right_hand_012"


def test_recover_runs_protocol_vs_unrecoverable():
    runs, src = epoching.recover_runs(100)
    assert src == "inferred_from_protocol" and runs.max() == 5 and (runs == 1).sum() == 20
    runs6, src6 = epoching.recover_runs(120)
    assert src6 == "inferred_from_protocol" and runs6.max() == 6
    runs_bad, src_bad = epoching.recover_runs(4)
    assert src_bad == "unrecoverable" and (runs_bad == -1).all()
