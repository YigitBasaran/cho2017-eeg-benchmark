import pytest

torch = pytest.importorskip("torch")

from cho2017_benchmark.models.atcnet import LocalATCNet, build_atcnet  # noqa: E402
from cho2017_benchmark.models.factory import assert_no_softmax, count_trainable_parameters  # noqa: E402


# --- LocalATCNet: torch-only, NOT guarded by braindecode ---

@pytest.mark.requires_torch
@pytest.mark.parametrize("batch", [1, 2, 8])
def test_local_atcnet_forward_shape(batch):
    model = LocalATCNet(n_chans=64, n_outputs=2, n_times=512)
    model.eval()
    out = model(torch.randn(batch, 64, 512))
    assert tuple(out.shape) == (batch, 2)


@pytest.mark.requires_torch
def test_local_atcnet_no_softmax_params_and_backward():
    model = LocalATCNet(n_chans=64, n_outputs=2, n_times=512)
    assert_no_softmax(model, n_chans=64, n_times=512, n_outputs=2)
    assert count_trainable_parameters(model) > 0
    out = model(torch.randn(3, 64, 512))
    out.sum().backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


# --- Braindecode ATCNet: the default implementation, guarded ---

@pytest.mark.requires_braindecode
def test_braindecode_atcnet_forward_shape():
    pytest.importorskip("braindecode")
    model, source = build_atcnet(
        n_chans=64, n_outputs=2, n_times=512,
        prefer_braindecode=True, allow_local_fallback=False,
        sfreq=256.0, input_window_seconds=2.0,
    )
    assert source == "braindecode"
    info = assert_no_softmax(model, n_chans=64, n_times=512, n_outputs=2)
    assert info["output_shape"] == (2, 2)
