"""DINOv2 loads in half precision on any GPU and float32 on CPU; its loaders are stubbed."""
import types

import pytest
import torch


class _Model:
    def to(self, _device):
        return self

    def eval(self):
        return self


@pytest.mark.parametrize("device, dtype", [("cuda", torch.float16), ("cuda:1", torch.float16),
                                           ("mps", torch.float16), ("cpu", torch.float32)])
def test_dinov2_loads_in_half_precision_on_any_gpu(monkeypatch, device, dtype):
    import transformers

    import v2_dinov2

    seen = []
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained",
                        lambda *_a, torch_dtype=None, **_k: seen.append(torch_dtype) or _Model())
    monkeypatch.setattr(transformers.AutoImageProcessor, "from_pretrained",
                        lambda *_a, **_k: types.SimpleNamespace())
    for name in ("_MODEL", "_PROCESSOR", "_DEVICE"):
        monkeypatch.setattr(v2_dinov2, name, None)
    v2_dinov2.load(device)
    assert seen == [dtype]
