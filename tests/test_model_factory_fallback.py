import pytest

pytest.importorskip("torch")

from cho2017_benchmark.models.atcnet import LocalATCNet, build_atcnet  # noqa: E402

pytestmark = pytest.mark.requires_torch


def test_explicit_local_fallback_is_labelled():
    model, source = build_atcnet(prefer_braindecode=False, allow_local_fallback=True)
    assert source == "local_fallback"
    assert isinstance(model, LocalATCNet)


def test_no_silent_substitution():
    # Disabling braindecode without enabling the fallback must error, never
    # silently substitute another model.
    with pytest.raises(ValueError):
        build_atcnet(prefer_braindecode=False, allow_local_fallback=False)


def test_local_fallback_param_filtering():
    # Unknown/braindecode-only kwargs are filtered for the local model.
    model, source = build_atcnet(
        prefer_braindecode=False, allow_local_fallback=True,
        n_windows=3, conv_block_pool_size_2=5, some_unknown_kwarg=123,
    )
    assert source == "local_fallback" and model.n_windows == 3
