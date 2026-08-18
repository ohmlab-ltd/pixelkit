"""SigLIP2 image embedder for the V2 specific pipeline.

Companion to `v2_dinov2.py`. We score queries against reference
centroids using BOTH encoders and combine the per-label sims with
a configurable weight. SigLIP2 is text-aligned (multimodal
contrastive pretraining), so it carries semantic priors DINOv2
doesn't have - it knows what "hare" means as a concept, not just
how a hare's pixels look. Together they cover each other's blind
spots on fine-grained pairs (hare/rabbit, horse standing/lying)
where one model alone is borderline.

API mirrors v2_dinov2 so the resolver can treat them
interchangeably: `encode_image`, `encode_images_batch(tta=True)`,
`EMBEDDING_DIM`, `EMBED_VERSION`, `is_loaded`.

Differences from DINOv2:
  - Uses the HuggingFace AutoProcessor pair (image + text), but we
    only call the image side - text encoder is unused at inference.
  - SigLIP2's vision tower is a ViT-L/16 by default (16 px patches,
    not 14), so INPUT_SIDE must be a multiple of 16. 384 is the
    upstream pretraining resolution (24×16 = 384, 24×24 = 576
    patch tokens).
  - No foreground-only patch_mean here for simplicity - SigLIP's
    image encoder uses GAP-on-patch-tokens by default and the
    fine-grained signal we want is already in the global vector.
    If we revisit, the same masking trick from DINOv2 ports over.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np
import torch
from PIL import Image as PILImage, ImageOps

SIGLIP_MODEL_ID = os.environ.get("V2_SIGLIP_MODEL", "google/siglip2-large-patch16-384")

# Embedding dim by model - SigLIP2 large is 1152, base is 768,
# so400m is 1152. Inferred at load time; constant below is the
# fallback used before the model is loaded.
_DIM_BY_MODEL = {
    "google/siglip2-base-patch16-256": 768,
    "google/siglip2-base-patch16-384": 768,
    "google/siglip2-large-patch16-256": 1152,
    "google/siglip2-large-patch16-384": 1152,
    "google/siglip2-so400m-patch14-384": 1152,
    "google/siglip2-giant-opt-patch16-384": 1536,
}
EMBEDDING_DIM = _DIM_BY_MODEL.get(SIGLIP_MODEL_ID, 1152)

# SigLIP2 large uses 16 px patches → INPUT_SIDE must be a multiple
# of 16. 384 = 24 × 16 → 576 patch tokens, matches upstream
# pretraining. Configurable for the so400m variant which uses 14 px.
INPUT_SIDE = int(os.environ.get("V2_SIGLIP_INPUT_SIDE", "384"))

# Bumped whenever the encode procedure changes. Mirrors EMBED_VERSION
# semantics in v2_dinov2 - references at older versions get re-
# encoded by the centroid backfill on first load.
#
# v1: SigLIP2 large at 384, GAP pooling, 3-view TTA matching DINOv2
EMBED_VERSION = 1

# TTA view count - keep in sync with `_tta_views`. Same 3 views as
# DINOv2 so the two encoders see the same crop variants and the
# combined score is comparable view-by-view.
EMBED_TTA_VIEWS = 3

_DEFAULT_CACHE = Path(__file__).resolve().parent.parent / "models_cache"
os.environ.setdefault("HF_HOME", str(_DEFAULT_CACHE))

_MODEL = None
_PROCESSOR = None
_DEVICE = "cpu"
_LOAD_LOCK = threading.Lock()


def is_loaded() -> bool:
    return _MODEL is not None and _PROCESSOR is not None


def load(device: str = "cpu"):
    """Idempotent loader. Caches the model + processor in module
    globals. fp16 on CUDA - same rationale as DINOv2: SigLIP was
    pretrained at fp16 upstream and the downstream cosine match runs
    on fp32 numpy, so no precision loss propagates.

    Loading is gated by V2_SIGLIP_DISABLED env so we can fall back
    cleanly on hosts without the spare VRAM. The resolver checks
    is_loaded() and skips the SigLIP score path when False, leaving
    DINOv2-only scoring intact.
    """
    global _MODEL, _PROCESSOR, _DEVICE
    if os.environ.get("V2_SIGLIP_DISABLED", "").lower() in ("1", "true", "on"):
        print("[v2-siglip] V2_SIGLIP_DISABLED set - skipping load.")
        return None, None
    with _LOAD_LOCK:
        if is_loaded() and _DEVICE == device:
            return _MODEL, _PROCESSOR
        from transformers import AutoModel, AutoProcessor
        from transformers.image_utils import PILImageResampling
        print(f"[v2-siglip] loading {SIGLIP_MODEL_ID} on {device}...")
        try:
            processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_ID)
        except Exception as e:
            print(f"[v2-siglip] processor load failed: {e}")
            return None, None
        # Force the image side to feed the model an INPUT_SIDE square
        # using bicubic resize and no centre-crop. Mirrors the
        # DINOv2 setup so both encoders see the same crop framing.
        ip = getattr(processor, "image_processor", processor)
        try:
            ip.size = {"height": INPUT_SIDE, "width": INPUT_SIDE}
        except Exception:
            pass
        if hasattr(ip, "crop_size"):
            ip.crop_size = {"height": INPUT_SIDE, "width": INPUT_SIDE}
        if hasattr(ip, "do_center_crop"):
            ip.do_center_crop = False
        if hasattr(ip, "resample"):
            ip.resample = PILImageResampling.BICUBIC
        dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
        try:
            model = AutoModel.from_pretrained(SIGLIP_MODEL_ID, torch_dtype=dtype).to(device).eval()
        except Exception as e:
            print(f"[v2-siglip] model load failed: {e}")
            return None, None
        # We only ever use the vision tower. Drop the text encoder
        # to free VRAM.
        if hasattr(model, "text_model"):
            try:
                model.text_model = None
            except Exception:
                pass
        _MODEL = model
        _PROCESSOR = processor
        _DEVICE = device
        print(
            f"[v2-siglip] ready (dim={EMBEDDING_DIM}, "
            f"input={INPUT_SIDE}x{INPUT_SIDE}, dtype={dtype}, device={device})."
        )
        return _MODEL, _PROCESSOR


def warmup() -> None:
    if not is_loaded():
        return
    try:
        encode_image(PILImage.new("RGB", (224, 224), color=(127, 127, 127)))
        encode_images_batch([
            PILImage.new("RGB", (224, 224), color=(127, 127, 127)),
            PILImage.new("RGB", (224, 224), color=(127, 127, 127)),
        ], tta=True)
    except Exception as e:
        print(f"[v2-siglip] warmup failed: {e}")


def _get_image_features(pixel_values: torch.Tensor) -> torch.Tensor:
    """Pull the pooled image embedding out of the SigLIP vision tower
    as a plain tensor.

    HF SigLIP variants are inconsistent here:
      - SigLIP v1: `get_image_features` returns a tensor.
      - SigLIP v2 (some checkpoints): `get_image_features` returns a
        `BaseModelOutputWithPooling`, NOT a tensor - calling .float()
        on it throws.
      - Older models: no `get_image_features`, so call `vision_model`
        and read `pooler_output` / fall back to mean-of-tokens.

    Normalise all three paths to "return a tensor".
    """
    def _unwrap(x):
        # Already a tensor? done.
        if isinstance(x, torch.Tensor):
            return x
        # ModelOutput-like dataclass - pick the right field.
        if hasattr(x, "image_embeds") and x.image_embeds is not None:
            return x.image_embeds
        if hasattr(x, "pooler_output") and x.pooler_output is not None:
            return x.pooler_output
        if hasattr(x, "last_hidden_state") and x.last_hidden_state is not None:
            return x.last_hidden_state.mean(dim=1)
        raise RuntimeError(f"SigLIP output has no extractable image embedding: {type(x)}")

    if hasattr(_MODEL, "get_image_features"):
        return _unwrap(_MODEL.get_image_features(pixel_values=pixel_values))
    vision = getattr(_MODEL, "vision_model", None)
    if vision is None:
        raise RuntimeError("SigLIP model has neither get_image_features nor vision_model")
    return _unwrap(vision(pixel_values=pixel_values))


def _tta_views(img: PILImage.Image) -> list[PILImage.Image]:
    """Three identity-preserving views - original, h-flip, +10%
    zoom-out with black pad. Same as DINOv2 so both encoders see
    matched crop variants and the combined score is comparable
    view-by-view."""
    w, h = img.size
    views: list[PILImage.Image] = [img, img.transpose(PILImage.FLIP_LEFT_RIGHT)]
    pw = max(2, int(round(w * 0.10)))
    ph = max(2, int(round(h * 0.10)))
    padded = PILImage.new("RGB", (w + 2 * pw, h + 2 * ph), (0, 0, 0))
    padded.paste(img, (pw, ph))
    views.append(padded)
    return views


# SigLIP2 ViT-L/16 - patches are 16 px square. INPUT_SIDE=384
# yields 24×24 = 576 patch tokens per crop. Same fg-mask logic
# as DINOv2 (count non-black pixels per patch) but at a different
# patch size, so it lives here as a parallel constant.
_PATCH_SIZE = 16
_PATCH_FG_PIXEL_FRAC = float(os.environ.get("V2_SIGLIP_PATCH_FG_FRAC", "0.10"))


def _patch_fg_mask(pil: PILImage.Image) -> np.ndarray:
    """Per-patch foreground mask for SigLIP at INPUT_SIDE × INPUT_SIDE
    (default 384). Mirror of DINOv2's helper but with PATCH_SIZE=16.
    Returns a flat (N,) bool array where N = (INPUT_SIDE/16)^2.
    """
    grid_w = INPUT_SIDE // _PATCH_SIZE
    if pil.size != (INPUT_SIDE, INPUT_SIDE):
        pil = pil.resize((INPUT_SIDE, INPUT_SIDE), PILImage.NEAREST)
    arr = np.asarray(pil.convert("RGB"))
    arr = arr.reshape(grid_w, _PATCH_SIZE, grid_w, _PATCH_SIZE, 3)
    nonzero = (arr.sum(axis=-1) > 0)
    counts = nonzero.sum(axis=(1, 3))
    threshold = max(1, int(round(_PATCH_SIZE * _PATCH_SIZE * _PATCH_FG_PIXEL_FRAC)))
    return (counts >= threshold).reshape(-1)


@torch.inference_mode()
def encode_image_patches(pil: PILImage.Image) -> tuple[np.ndarray, np.ndarray]:
    """Per-patch SigLIP encoding for patch-level matching.

    Mirrors DINOv2's `encode_image_patches`: returns (tokens, fg_mask)
    where tokens is (P, D) L2-normalised float32 and fg_mask is (P,)
    bool over patches that overlap the SAM-masked foreground.

    SigLIP doesn't use a CLS token the same way DINOv2 does - its
    vision tower's `last_hidden_state` already corresponds 1:1 to
    the patch grid (no separate CLS prepended), so we keep the
    full sequence as patches.

    No TTA - patches encode position-specific detail; averaging
    across views washes out the spatial signal patch matching uses.
    """
    n_patches = (INPUT_SIDE // _PATCH_SIZE) ** 2
    if not is_loaded():
        return (np.zeros((n_patches, EMBEDDING_DIM), dtype=np.float32),
                np.zeros(n_patches, dtype=bool))
    try:
        img = ImageOps.exif_transpose(pil).convert("RGB")
        fg = _patch_fg_mask(img)
        inputs = _PROCESSOR(images=img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(_DEVICE)
        if pixel_values.dtype != _MODEL.dtype:
            pixel_values = pixel_values.to(_MODEL.dtype)
        vision = getattr(_MODEL, "vision_model", _MODEL)
        out = vision(pixel_values=pixel_values)
        if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            patches = out.last_hidden_state[0]
        elif isinstance(out, torch.Tensor):
            patches = out[0]
        else:
            raise RuntimeError(f"SigLIP vision output has no last_hidden_state: {type(out)}")
        # Defensive: trim to expected patch count if the model
        # ever prepends a CLS-style register token.
        if patches.shape[0] > n_patches:
            patches = patches[-n_patches:]
        patches = torch.nn.functional.normalize(patches.float(), dim=-1)
        return patches.detach().cpu().numpy().astype(np.float32, copy=False), fg
    except Exception as e:
        print(f"[v2-siglip] encode_image_patches failed: {e}")
        return (np.zeros((n_patches, EMBEDDING_DIM), dtype=np.float32),
                np.zeros(n_patches, dtype=bool))


@torch.inference_mode()
def encode_image(pil: PILImage.Image) -> np.ndarray:
    if not is_loaded():
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)
    try:
        img = ImageOps.exif_transpose(pil).convert("RGB")
        inputs = _PROCESSOR(images=img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(_DEVICE)
        if pixel_values.dtype != _MODEL.dtype:
            pixel_values = pixel_values.to(_MODEL.dtype)
        feat = _get_image_features(pixel_values)
        feat = torch.nn.functional.normalize(feat.float(), dim=-1)
        return feat[0].detach().cpu().numpy().astype(np.float32, copy=False)
    except Exception as e:
        print(f"[v2-siglip] encode_image failed: {e}")
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)


@torch.inference_mode()
def encode_images_batch(pils: list[PILImage.Image], *, tta: bool = True) -> np.ndarray:
    """Encode N PIL images → (N, EMBEDDING_DIM) L2-normalised float32.

    Same TTA scheme as DINOv2 - 3 views per image, batched into one
    forward pass, per-view L2 → mean over views → re-L2.
    """
    n = len(pils)
    if n == 0:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    if not is_loaded():
        return np.zeros((n, EMBEDDING_DIM), dtype=np.float32)
    try:
        prepared = [ImageOps.exif_transpose(p).convert("RGB") for p in pils]
        if tta:
            all_views: list[PILImage.Image] = []
            for img in prepared:
                all_views.extend(_tta_views(img))
            inputs = _PROCESSOR(images=all_views, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(_DEVICE)
            if pixel_values.dtype != _MODEL.dtype:
                pixel_values = pixel_values.to(_MODEL.dtype)
            feats = _get_image_features(pixel_values)
            feats = torch.nn.functional.normalize(feats.float(), dim=-1)
            grouped = feats.view(n, EMBED_TTA_VIEWS, -1)
            mean_emb = grouped.mean(dim=1)
            vecs = torch.nn.functional.normalize(mean_emb, dim=-1)
        else:
            inputs = _PROCESSOR(images=prepared, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(_DEVICE)
            if pixel_values.dtype != _MODEL.dtype:
                pixel_values = pixel_values.to(_MODEL.dtype)
            feats = _get_image_features(pixel_values)
            vecs = torch.nn.functional.normalize(feats.float(), dim=-1)
        return vecs.detach().cpu().numpy().astype(np.float32, copy=False)
    except Exception as e:
        print(f"[v2-siglip] encode_images_batch failed: {e}")
        return np.zeros((n, EMBEDDING_DIM), dtype=np.float32)
