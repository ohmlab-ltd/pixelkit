"""GPU AI upscaling for cover banners (Real-ESRGAN `realesr-general-x4v3`).

A tiny SRVGGNetCompact model (~1.2M params, ~5 MB weights) run on the backend
GPU (RTX 4500 Ada) via the torch already installed. Used to enlarge small
dataset / Project covers into a crisp full-width banner — real detail synthesis,
unlike the Lanczos fallback. The "general v3" model is purpose-built for
real-world / compressed images (what user covers usually are), so it handles
JPEG artefacts better than the heavier x4plus and barely touches VRAM.

Architecture is vendored from xinntao/Real-ESRGAN (BSD-3-Clause). Weights are
loaded from a vendored file if present, else downloaded once to a local cache on
first use; nothing model-related is committed to git. Every entry point fails
soft: if torch/CUDA/weights are unavailable the caller falls back to Lanczos.
"""
from __future__ import annotations

import os
import threading
import urllib.request
from pathlib import Path

# realesr-general-x4v3: SRVGGNetCompact, fixed 4x. ~4.8 MB.
_WEIGHTS_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/"
    "v0.2.5.0/realesr-general-x4v3.pth"
)
_WEIGHTS_NAME = "realesr-general-x4v3.pth"
_NATIVE_SCALE = 4

# One model instance, loaded lazily and kept resident (it's tiny). The lock
# serialises both the lazy load and every GPU inference so concurrent cover
# requests don't run multiple forwards on the GPU at once.
_MODEL = None            # tuple[(net, device)] once loaded
_LOAD_FAILED = False     # don't retry a hard failure (missing torch, etc.) every call
_LOCK = threading.Lock()


def _cache_dir() -> Path:
    d = Path(os.environ.get("PK_MODEL_CACHE", Path.home() / ".cache" / "pixelkit" / "models"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _weights_path() -> Path:
    # Prefer a vendored copy beside this module (lets an operator drop the file
    # in to avoid any runtime download), else the writable cache.
    vendored = Path(__file__).resolve().parent / "models" / _WEIGHTS_NAME
    if vendored.exists():
        return vendored
    return _cache_dir() / _WEIGHTS_NAME


def _ensure_weights() -> Path:
    p = _weights_path()
    if p.exists() and p.stat().st_size > 1_000_000:
        return p
    # Download once into the cache (atomic via a temp file).
    target = _cache_dir() / _WEIGHTS_NAME
    tmp = target.with_suffix(".tmp")
    urllib.request.urlretrieve(_WEIGHTS_URL, tmp)
    tmp.replace(target)
    return target


def _build_srvgg():
    """SRVGGNetCompact (Real-ESRGAN compact generator), matching the
    realesr-general-x4v3 checkpoint: 64 feats, 32 conv blocks, PReLU, 4x."""
    import torch.nn as nn
    import torch.nn.functional as F

    class SRVGGNetCompact(nn.Module):
        def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4):
            super().__init__()
            self.upscale = upscale
            self.body = nn.ModuleList()
            self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
            self.body.append(nn.PReLU(num_parameters=num_feat))
            for _ in range(num_conv):
                self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
                self.body.append(nn.PReLU(num_parameters=num_feat))
            self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
            self.upsampler = nn.PixelShuffle(upscale)

        def forward(self, x):
            out = x
            for layer in self.body:
                out = layer(out)
            out = self.upsampler(out)
            # Long skip from a nearest-neighbour upscale (the model learns the
            # residual on top of it), as in the reference implementation.
            base = F.interpolate(x, scale_factor=self.upscale, mode="nearest")
            return out + base

    return SRVGGNetCompact


def _get_model():
    """Lazy singleton: (net, device). Returns None if the model can't be loaded
    (no torch, no weights, bad checkpoint) — the caller then uses Lanczos."""
    global _MODEL, _LOAD_FAILED
    if _MODEL is not None:
        return _MODEL
    if _LOAD_FAILED:
        return None
    try:
        import torch
        SRVGGNetCompact = _build_srvgg()
        net = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=_NATIVE_SCALE)
        sd = torch.load(_ensure_weights(), map_location="cpu")
        if isinstance(sd, dict):
            sd = sd.get("params") or sd.get("params_ema") or sd
        net.load_state_dict(sd, strict=True)
        net.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        net = net.to(device)
        _MODEL = (net, device)
        print(f"[upscale] realesr-general-x4v3 loaded on {device}", flush=True)
        return _MODEL
    except Exception as e:
        _LOAD_FAILED = True
        print(f"[upscale] model load failed ({e}); AI upscaling disabled, using Lanczos", flush=True)
        return None


def _tiled_forward(net, img, tile: int, pad: int, scale: int):
    """Run the net tile-by-tile so VRAM stays bounded regardless of input size.
    Each tile is processed with `pad` px of context then the pad is cropped off
    the (scaled) output so seams don't show."""
    import torch  # noqa: F401  (img is already a torch tensor)
    b, c, h, w = img.shape
    if tile <= 0 or (h <= tile and w <= tile):
        return net(img)
    out = img.new_zeros(b, c, h * scale, w * scale)
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            y1, x1 = min(y + tile, h), min(x + tile, w)
            py0, px0 = max(y - pad, 0), max(x - pad, 0)
            py1, px1 = min(y1 + pad, h), min(x1 + pad, w)
            o = net(img[:, :, py0:py1, px0:px1])
            oy0, ox0 = (y - py0) * scale, (x - px0) * scale
            oy1, ox1 = oy0 + (y1 - y) * scale, ox0 + (x1 - x) * scale
            out[:, :, y * scale:y1 * scale, x * scale:x1 * scale] = o[:, :, oy0:oy1, ox0:ox1]
    return out


def available() -> bool:
    """True if the model can be (or is) loaded — cheap to call repeatedly."""
    return _get_model() is not None


def upscale_to(pil_img, target_long_edge: int):
    """AI-upscale `pil_img` (Real-ESRGAN 4x) then Lanczos-fit the longest edge to
    `target_long_edge`. Returns a PIL RGB image. Raises on any failure so the
    caller can fall back to Lanczos."""
    import numpy as np
    import torch
    from PIL import Image

    with _LOCK:
        loaded = _get_model()
        if loaded is None:
            raise RuntimeError("upscale model unavailable")
        net, device = loaded
        img = pil_img.convert("RGB")
        arr = np.asarray(img).astype("float32") / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            # tile=256 keeps peak VRAM tiny even for the largest cover we'd ever
            # enlarge (sources are < the target, so well under ~1280 px).
            out = _tiled_forward(net, t, tile=256, pad=16, scale=_NATIVE_SCALE)
            out = out.squeeze(0).clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
        if device == "cuda":
            torch.cuda.empty_cache()
    up = Image.fromarray((out * 255.0 + 0.5).astype("uint8"))
    # The model is a fixed 4x; bring it down to the requested banner size.
    w, h = up.size
    longest = max(w, h)
    if longest > target_long_edge:
        s = target_long_edge / float(longest)
        up = up.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    return up
