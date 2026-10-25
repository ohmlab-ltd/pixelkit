"""DINOv2 embedder for V2 reference crops: L2-normalised float32 vectors."""
from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np
import torch
from PIL import Image as PILImage, ImageOps

DINO_MODEL_ID = os.environ.get("V2_DINO_MODEL", "facebook/dinov2-large")
DINO_POOLING = os.environ.get("V2_DINO_POOLING", "patch_mean")  # patch_mean | cls | cls_patch

# Embedding dimension per model; 1024 for an unknown id.
_DIM_BY_MODEL = {
    "facebook/dinov2-small": 384,
    "facebook/dinov2-base": 768,
    "facebook/dinov2-large": 1024,
    "facebook/dinov2-giant": 1536,
}
EMBEDDING_DIM = _DIM_BY_MODEL.get(DINO_MODEL_ID, 1024)

# Input side in px, a multiple of the 14 px patch: 518 gives 37 x 37 patches.
INPUT_SIDE = int(os.environ.get("V2_DINO_INPUT_SIDE", "518"))
if INPUT_SIDE % 14 != 0:
    raise RuntimeError(
        f"V2_DINO_INPUT_SIDE={INPUT_SIDE} is not a multiple of 14 - ViT-L/14 patches won't align"
    )

# Version of the embedding procedure. A stored reference embedding of
# another version is re-encoded.
EMBED_VERSION = 7

# Number of views `_tta_views` returns.
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
    """Load once per device: fp16 on cuda and mps, fp32 on cpu."""
    global _MODEL, _PROCESSOR, _DEVICE, EMBEDDING_DIM
    with _LOAD_LOCK:
        if is_loaded() and _DEVICE == device:
            return _MODEL, _PROCESSOR
        from transformers import AutoImageProcessor, AutoModel
        from transformers.image_utils import PILImageResampling
        print(f"[v2-dino] loading {DINO_MODEL_ID} on {device}...")
        processor = AutoImageProcessor.from_pretrained(DINO_MODEL_ID)
        # A plain bicubic resize to INPUT_SIDE square; the default
        # centre-crops to 224.
        processor.size = {"height": INPUT_SIDE, "width": INPUT_SIDE}
        if hasattr(processor, "crop_size"):
            processor.crop_size = {"height": INPUT_SIDE, "width": INPUT_SIDE}
        if hasattr(processor, "do_center_crop"):
            processor.do_center_crop = False
        if hasattr(processor, "resample"):
            processor.resample = PILImageResampling.BICUBIC
        # By type: an explicit card ("cuda:1") is still a GPU.
        dtype = torch.float16 if torch.device(device).type in ("cuda", "mps") else torch.float32
        model = AutoModel.from_pretrained(DINO_MODEL_ID, torch_dtype=dtype).to(device).eval()
        # The table covers the known ids; the loaded model has the final word.
        EMBEDDING_DIM = int(getattr(getattr(model, "config", None), "hidden_size", EMBEDDING_DIM))
        _MODEL = model
        _PROCESSOR = processor
        _DEVICE = device
        print(
            f"[v2-dino] ready (pooling={DINO_POOLING}, dim={EMBEDDING_DIM}, "
            f"input={INPUT_SIDE}x{INPUT_SIDE}, dtype={dtype}, device={device})."
        )
        return _MODEL, _PROCESSOR


def warmup() -> None:
    if not is_loaded():
        return
    try:
        encode_image(PILImage.new("RGB", (224, 224), color=(127, 127, 127)))
        # Batches above 1 use other cuDNN kernels.
        encode_images_batch([
            PILImage.new("RGB", (224, 224), color=(127, 127, 127)),
            PILImage.new("RGB", (224, 224), color=(127, 127, 127)),
        ], tta=True)
    except Exception as e:
        print(f"[v2-dino] warmup failed: {e}")


def _pool(tokens: torch.Tensor, fg_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Pool DINOv2 tokens to (..., D) per DINO_POOLING. `fg_mask`, (B, P)
    bool, limits the patch mean to foreground patches; a row with none
    uses all patches."""
    cls_t = tokens[..., 0, :]
    patch_t = tokens[..., 1:, :]
    if DINO_POOLING == "cls":
        return cls_t

    if fg_mask is not None:
        fg = fg_mask.to(patch_t.dtype).unsqueeze(-1)  # (B, P, 1)
        denom = fg.sum(dim=1)                         # (B, 1)
        masked_mean = (patch_t * fg).sum(dim=1) / denom.clamp(min=1.0)
        full_mean = patch_t.mean(dim=-2)
        # (B, 1) bool, broadcast to (B, D).
        has_fg = denom > 0
        patch_mean = torch.where(has_fg, masked_mean, full_mean)
    else:
        patch_mean = patch_t.mean(dim=-2)

    if DINO_POOLING == "cls_patch":
        return 0.5 * cls_t + 0.5 * patch_mean
    return patch_mean


# A patch is foreground when at least this fraction of its pixels are non-black.
PATCH_FG_PIXEL_FRAC = float(os.environ.get("V2_DINO_PATCH_FG_FRAC", "0.10"))
_PATCH_SIZE = 14  # ViT-L/14


def _patch_fg_mask(pil: PILImage.Image) -> np.ndarray:
    """(P,) bool, P = (INPUT_SIDE/14)^2: True for a patch with at least
    PATCH_FG_PIXEL_FRAC non-black pixels. inpaint_bbox_crop fills the
    background with exact black."""
    grid_w = INPUT_SIDE // _PATCH_SIZE
    # NEAREST: bicubic would inflate the foreground at silhouette edges.
    if pil.size != (INPUT_SIDE, INPUT_SIDE):
        pil = pil.resize((INPUT_SIDE, INPUT_SIDE), PILImage.NEAREST)
    arr = np.asarray(pil.convert("RGB"))  # (H, W, 3) uint8
    arr = arr.reshape(grid_w, _PATCH_SIZE, grid_w, _PATCH_SIZE, 3)
    nonzero = (arr.sum(axis=-1) > 0)               # (gw, 14, gw, 14)
    counts = nonzero.sum(axis=(1, 3))              # (gw, gw)
    threshold = max(1, int(round(_PATCH_SIZE * _PATCH_SIZE * PATCH_FG_PIXEL_FRAC)))
    return (counts >= threshold).reshape(-1)       # (P,)


@torch.inference_mode()
def encode_image_patches(pil: PILImage.Image) -> tuple[np.ndarray, np.ndarray]:
    """Per-patch encoding: ((P, D) float32 tokens, L2-normalised per row,
    (P,) bool foreground mask), zeros on failure. No TTA: averaging views
    would blur the spatial detail patch matching uses."""
    n_patches = (INPUT_SIDE // _PATCH_SIZE) ** 2
    if not is_loaded():
        return (np.zeros((n_patches, EMBEDDING_DIM), dtype=np.float32),
                np.zeros(n_patches, dtype=bool))
    try:
        img = ImageOps.exif_transpose(pil).convert("RGB")
        fg = _patch_fg_mask(img)  # (P,) bool
        inputs = _PROCESSOR(images=img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(_DEVICE)
        if pixel_values.dtype != _MODEL.dtype:
            pixel_values = pixel_values.to(_MODEL.dtype)
        outputs = _MODEL(pixel_values=pixel_values, interpolate_pos_encoding=True)
        # last_hidden_state is (1, 1+P, D); index 0 is CLS.
        patches = outputs.last_hidden_state[0, 1:, :]  # (P, D)
        patches = torch.nn.functional.normalize(patches.float(), dim=-1)
        return patches.detach().cpu().numpy().astype(np.float32, copy=False), fg
    except Exception as e:
        print(f"[v2-dino] encode_image_patches failed: {e}")
        return (np.zeros((n_patches, EMBEDDING_DIM), dtype=np.float32),
                np.zeros(n_patches, dtype=bool))


@torch.inference_mode()
def encode_image(pil: PILImage.Image) -> np.ndarray:
    """Encode one image to an (EMBEDDING_DIM,) L2-normalised float32 vector."""
    if not is_loaded():
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)
    try:
        img = ImageOps.exif_transpose(pil).convert("RGB")
        inputs = _PROCESSOR(images=img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(_DEVICE)
        if pixel_values.dtype != _MODEL.dtype:
            pixel_values = pixel_values.to(_MODEL.dtype)
        # (1, P) foreground mask for _pool.
        fg = torch.from_numpy(_patch_fg_mask(img)).unsqueeze(0).to(_DEVICE)
        outputs = _MODEL(pixel_values=pixel_values, interpolate_pos_encoding=True)
        vec = _pool(outputs.last_hidden_state, fg_mask=fg)[0]
        vec = torch.nn.functional.normalize(vec.float(), dim=0)
        return vec.detach().cpu().numpy().astype(np.float32, copy=False)
    except Exception as e:
        print(f"[v2-dino] encode_image failed: {e}")
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)


def _tta_views(img: PILImage.Image) -> list[PILImage.Image]:
    """Original, horizontal flip, and a zoom-out padded 10% per side with
    black, as inpaint_bbox_crop fills. No vertical flip or rotation: some
    labels depend on orientation."""
    w, h = img.size
    views: list[PILImage.Image] = [img]

    views.append(img.transpose(PILImage.FLIP_LEFT_RIGHT))

    pw = max(2, int(round(w * 0.10)))
    ph = max(2, int(round(h * 0.10)))
    padded = PILImage.new("RGB", (w + 2 * pw, h + 2 * ph), (0, 0, 0))
    padded.paste(img, (pw, ph))
    views.append(padded)

    return views


@torch.inference_mode()
def encode_images_batch(pils: list[PILImage.Image], *, tta: bool = True) -> np.ndarray:
    """Encode N images to (N, EMBEDDING_DIM) L2-normalised float32, zeros on
    failure. With `tta`, each row is the re-normalised mean of the
    `_tta_views` embeddings."""
    n = len(pils)
    if n == 0:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    if not is_loaded():
        return np.zeros((n, EMBEDDING_DIM), dtype=np.float32)
    try:
        prepared = [ImageOps.exif_transpose(p).convert("RGB") for p in pils]
        if tta:
            all_views: list[PILImage.Image] = []
            all_fg: list[np.ndarray] = []
            for img in prepared:
                for view in _tta_views(img):
                    all_views.append(view)
                    all_fg.append(_patch_fg_mask(view))
            inputs = _PROCESSOR(images=all_views, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(_DEVICE)
            if pixel_values.dtype != _MODEL.dtype:
                pixel_values = pixel_values.to(_MODEL.dtype)
            fg_tensor = torch.from_numpy(np.stack(all_fg)).to(_DEVICE)  # (n*V, P)
            outputs = _MODEL(pixel_values=pixel_values, interpolate_pos_encoding=True)
            per_view = _pool(outputs.last_hidden_state, fg_mask=fg_tensor)
            per_view = torch.nn.functional.normalize(per_view.float(), dim=-1)
            grouped = per_view.view(n, EMBED_TTA_VIEWS, -1)
            mean_emb = grouped.mean(dim=1)
            vecs = torch.nn.functional.normalize(mean_emb, dim=-1)
        else:
            inputs = _PROCESSOR(images=prepared, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(_DEVICE)
            if pixel_values.dtype != _MODEL.dtype:
                pixel_values = pixel_values.to(_MODEL.dtype)
            all_fg = [_patch_fg_mask(img) for img in prepared]
            fg_tensor = torch.from_numpy(np.stack(all_fg)).to(_DEVICE)  # (n, P)
            outputs = _MODEL(pixel_values=pixel_values, interpolate_pos_encoding=True)
            vecs = _pool(outputs.last_hidden_state, fg_mask=fg_tensor)
            vecs = torch.nn.functional.normalize(vecs.float(), dim=-1)
        return vecs.detach().cpu().numpy().astype(np.float32, copy=False)
    except Exception as e:
        print(f"[v2-dino] encode_images_batch failed: {e}")
        return np.zeros((n, EMBEDDING_DIM), dtype=np.float32)


def center_square_crop(img: PILImage.Image) -> PILImage.Image:
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def inpaint_bbox_crop(
    image: PILImage.Image,
    box: tuple[int, int, int, int] | list[float],
    polygons: list | None,
    *,
    fill_color: tuple[int, int, int] = (0, 0, 0),
) -> PILImage.Image:
    """Crop `image` to `box` and fill the pixels outside `polygons` with
    `fill_color`. Without a usable mask, or when it covers the whole box,
    the plain crop is returned."""
    x0, y0, x1, y1 = (int(round(float(c))) for c in box)
    x0 = max(0, x0); y0 = max(0, y0)
    x1 = max(x0 + 1, x1); y1 = max(y0 + 1, y1)
    crop_pil = image.crop((x0, y0, x1, y1)).convert("RGB")

    if not polygons:
        return crop_pil
    try:
        import cv2
    except ImportError:
        return crop_pil

    cw, ch = crop_pil.size
    if cw < 4 or ch < 4:
        return crop_pil

    mask = np.zeros((ch, cw), dtype=np.uint8)
    for poly in polygons:
        if not poly or len(poly) < 3:
            continue
        try:
            pts = np.asarray(
                [(int(round(float(p[0])) - x0), int(round(float(p[1])) - y0)) for p in poly],
                dtype=np.int32,
            )
        except Exception:
            continue
        if pts.shape[0] >= 3:
            cv2.fillPoly(mask, [pts], 255)

    if not mask.any():
        return crop_pil
    if mask.all():
        return crop_pil

    # Close single-pixel holes, then dilate 1 px: polygons sit slightly
    # inside the true edge.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    arr = np.array(crop_pil)
    fill_mask = (mask == 0)
    arr[fill_mask] = fill_color
    return PILImage.fromarray(arr)
