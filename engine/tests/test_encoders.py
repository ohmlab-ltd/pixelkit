"""The image encoders load in half precision on any GPU, an explicit card
("cuda:1") included. Nothing is downloaded: the loaders get stand-ins."""
import importlib
import types

import pytest
import torch


class _Model:
    def to(self, _device):
        return self

    def eval(self):
        return self


@pytest.mark.parametrize("module, processor", [("v2_dinov2", "AutoImageProcessor"),
                                               ("v2_siglip", "AutoProcessor")])
@pytest.mark.parametrize("device, dtype", [("cuda", torch.float16), ("cuda:1", torch.float16),
                                           ("mps", torch.float16), ("cpu", torch.float32)])
def test_encoders_load_in_half_precision_on_any_gpu(monkeypatch, module, processor, device, dtype):
    import transformers

    mod = importlib.import_module(module)
    seen = []
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained",
                        lambda *_a, torch_dtype=None, **_k: seen.append(torch_dtype) or _Model())
    monkeypatch.setattr(getattr(transformers, processor), "from_pretrained",
                        lambda *_a, **_k: types.SimpleNamespace())
    for name in ("_MODEL", "_PROCESSOR", "_DEVICE"):
        monkeypatch.setattr(mod, name, None)
    mod.load(device)
    assert seen == [dtype]
