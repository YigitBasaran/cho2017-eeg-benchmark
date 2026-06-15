import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("braindecode")

from cho2017_benchmark.models.eegnet import build_eegnet  # noqa: E402
from cho2017_benchmark.models.factory import assert_no_softmax, count_trainable_parameters  # noqa: E402

pytestmark = [pytest.mark.requires_torch, pytest.mark.requires_braindecode]


def test_eegnet_forward_shape_and_logits():
    model = build_eegnet(n_chans=64, n_outputs=2, n_times=512)
    info = assert_no_softmax(model, n_chans=64, n_times=512, n_outputs=2)
    assert info["output_shape"] == (2, 2)
    assert count_trainable_parameters(model) > 0


@pytest.mark.parametrize("batch", [1, 4])
def test_eegnet_batch_sizes(batch):
    model = build_eegnet(n_chans=64, n_outputs=2, n_times=512)
    model.eval()
    out = model(torch.randn(batch, 64, 512))
    assert tuple(out.shape) == (batch, 2)


def test_eegnet_backward():
    model = build_eegnet(n_chans=64, n_outputs=2, n_times=512)
    out = model(torch.randn(4, 64, 512))
    out.sum().backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
