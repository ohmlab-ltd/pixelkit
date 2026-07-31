"""DINOv2-base embedder for V2 reference crops.

Mirrors the standalone `detect_and_crop.py` script in the repo root:
    * model:    facebook/dinov2-base (768-dim)
    * pooling:  patch_mean (mean of patch tokens, ignoring CLS)
    * output:   L2-normalised float32 (768,)

Kept separate from `embeddings.py` because that one is pinned to
SigLIP 2 for Label Cascade and has its own crop / centre-weighting
pipeline. The V2 onboarding flow wants a plain DINOv2-base embedding
on raw bbox crops with no masking — what the user is already used to
from the script.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np
import torch
from PIL import Image as PILImage, ImageOps

DINO_MODEL_ID = os.environ.get("V2_DINO_MODEL", "facebook/dinov2-large")
DINO_POOLING = os.environ.get("V2_DINO_POOLING", "patch_mean")  # patch_mean | cls | cls_patch

# Embedding dimension follows the model — base/small=768, large=1024,
# giant=1536. Inferred from the loaded model at runtime; the constant
# below is the fallback used before the model is loaded (e.g. for
# backfill returns when DINOv2 hasn't warmed yet).
_DIM_BY_MODEL = {
    "facebook/dinov2-small": 384,
    "facebook/dinov2-base": 768,
    "facebook/dinov2-large": 1024,
    "facebook/dinov2-giant": 1536,
}
EMBEDDING_DIM = _DIM_BY_MODEL.get(DINO_MODEL_ID, 1024)

# DINOv2 ViT-L/14 patches are 14 px square. Inputs MUST be a multiple
# of 14 for the patchify step to consume every pixel and for the
# Transformer tokens to align with non-overlapping physical features.
# 518 = 37 × 14 → 37×37 = 1369 patch tokens (vs the 16×16 = 256 you
# get at the default 224 input). The extra resolution materially
# helps fine-grained discrimination (hare vs rabbit, lying horse vs
# standing horse) at the cost of ~5× more tokens per crop. The
# pretrained model handles the size change via positional-embedding
# interpolation — DINOv2 was trained with that capability.
INPUT_SIDE = int(os.environ.get("V2_DINO_INPUT_SIDE", "518"))
if INPUT_SIDE % 14 != 0:
    raise RuntimeError(
        f"V2_DINO_INPUT_SIDE={INPUT_SIDE} is not a multiple of 14 — ViT-L/14 patches won't align"
    )

# Bumped whenever the embedding procedure changes (preprocessing,
# TTA settings, pooling, model). Any reference embedding written
# under a previous version gets invalidated and re-encoded so
# centroids and live queries are produced by an identical pipeline.
#
# v1: plain crop + DINOv2-base patch_mean
# v2: mask grey-fill + 5-view TTA (h-flip, ±5% zoom, 3% translate)
# v3: switch base → large (768d → 1024d, stronger fine-grained features)
# v4: 518×518 bicubic input (37×37 patch grid, 5× more tokens vs 224)
# v5: black fill (was grey) + close+dilate mask cleanup for cleaner silhouettes
# v6: TTA reduced to 3 views (orig, h-flip, +10% zoom-out) with black pad
# v7: foreground-only patch_mean — pool drops black-bg patches so the
#     embedding is built from object patches only (sharper margin on
#     fine-grained pairs like hare vs rabbit)
EMBED_VERSION = 7

# TTA view count — keep in sync with `_tta_views`.
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
    globals so subsequent calls are no-ops. Uses fp16 on CUDA for a
    free 2× throughput win — DINOv2 features are extremely robust to
    half precision (the pretrained checkpoint was trained at fp16
    upstream) and the downstream cosine-similarity match is computed
    on the cpu/fp32 numpy array, so no precision loss propagates."""
    global _MODEL, _PROCESSOR, _DEVICE
    with _LOAD_LOCK:
        if is_loaded() and _DEVICE == device:
            return _MODEL, _PROCESSOR
        from transformers import AutoImageProcessor, AutoModel
        from transformers.image_utils import PILImageResampling
        print(f"[v2-dino] loading {DINO_MODEL_ID} on {device}...")
        processor = AutoImageProcessor.from_pretrained(DINO_MODEL_ID)
        # Force the processor to feed the model an INPUT_SIDE × INPUT_SIDE
        # square (518 by default) using bicubic interpolation. The
        # default config resizes shortest edge to 256 and centre-crops
        # to 224, which (a) wastes most of the crop and (b) gives only
        # 16×16 patch tokens. Setting size + crop_size to the same
        # value with do_center_crop=False makes the resize the only
        # spatial transform — the entire crop is rescaled into the
        # 518×518 frame and patchified directly.
        processor.size = {"height": INPUT_SIDE, "width": INPUT_SIDE}
        if hasattr(processor, "crop_size"):
            processor.crop_size = {"height": INPUT_SIDE, "width": INPUT_SIDE}
        if hasattr(processor, "do_center_crop"):
            processor.do_center_crop = False
        if hasattr(processor, "resample"):
            processor.resample = PILImageResampling.BICUBIC
        dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
        model = AutoModel.from_pretrained(DINO_MODEL_ID, torch_dtype=dtype).to(device).eval()
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
        # Prime the batched path too — the cuDNN benchmark cache
        # picks different kernels for batch>1, so warming both
        # avoids a cold-start hiccup on the first multi-box request.
        # Warm the TTA path too since it expands batch size 5×.
        encode_images_batch([
            PILImage.new("RGB", (224, 224), color=(127, 127, 127)),
            PILImage.new("RGB", (224, 224), color=(127, 127, 127)),
        ], tta=True)
    except Exception as e:
        print(f"[v2-dino] warmup failed: {e}")


def _pool(tokens: torch.Tensor, fg_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Pool DINOv2 token sequence → (..., D) per the configured strategy.

    `fg_mask` (optional, (B, P) bool) gates patch_mean to foreground
    patches only — patches whose 14×14 input region overlaps the SAM
    silhouette. Without it the mean over all 1369 patches is dominated
    by background tokens (every black patch produces approximately the
    same token, dragging crops toward a fixed point in feature space
    and shrinking the inter-class margin). With it, only patches that
    actually carry object content contribute, so the pool's fine-grained
    signal-to-noise improves on tightly-segmented crops.

    Falls back to full-patch mean when `fg_mask` is None or contains
    zero foreground patches for any image in the batch (degenerate
    silhouette → don't lose the embedding entirely).
    """
    cls_t = tokens[..., 0, :]
    patch_t = tokens[..., 1:, :]
    if DINO_POOLING == "cls":
        return cls_t

    if fg_mask is not None:
        # Weighted mean over foreground patches. If a row of fg_mask is
        # all-False (no foreground), fall back to the full mean for
        # that row only — vectorised so we don't loop per image.
        fg = fg_mask.to(patch_t.dtype).unsqueeze(-1)  # (B, P, 1)
        denom = fg.sum(dim=1)                         # (B, 1)
        masked_mean = (patch_t * fg).sum(dim=1) / denom.clamp(min=1.0)
        full_mean = patch_t.mean(dim=-2)
        # (B, 1) bool → broadcasts to (B, D) under torch.where.
        has_fg = denom > 0
        patch_mean = torch.where(has_fg, masked_mean, full_mean)
    else:
        patch_mean = patch_t.mean(dim=-2)

    if DINO_POOLING == "cls_patch":
        return 0.5 * cls_t + 0.5 * patch_mean
    return patch_mean


# Per-patch foreground threshold: a patch counts as foreground when at
# least this fraction of its 14×14 pixels are non-black after the
# upstream inpaint mask fill. 0.10 (≥20 of 196 pixels) excludes pure-
# background patches and patches the polygon barely touches, but keeps
# silhouette-boundary patches whose object-side content is informative.
PATCH_FG_PIXEL_FRAC = float(os.environ.get("V2_DINO_PATCH_FG_FRAC", "0.10"))
_PATCH_SIZE = 14  # ViT-L/14


def _patch_fg_mask(pil: PILImage.Image) -> np.ndarray:
    """Return a flat bool array of shape (P,) where P = (INPUT_SIDE/14)^2.

    True when the corresponding 14×14 patch in the resized 518×518
    input has ≥ PATCH_FG_PIXEL_FRAC non-black pixels. Background
    pixels after inpaint_bbox_crop are exactly (0, 0, 0); after
    bicubic resize to 518×518 the background interior remains zero
    (only single-pixel-wide boundary blends are non-zero), so a
    bright-line pixel-sum > 0 check is a clean foreground signal.
    """
    grid_w = INPUT_SIDE // _PATCH_SIZE
    # Resize with NEAREST so the foreground decision matches the grid
    # the model will see. Bicubic would smear black/non-black borders
    # and inflate the foreground fraction near silhouette edges.
    if pil.size != (INPUT_SIDE, INPUT_SIDE):
        pil = pil.resize((INPUT_SIDE, INPUT_SIDE), PILImage.NEAREST)
    arr = np.asarray(pil.convert("RGB"))  # (H, W, 3) uint8
    # Reshape into (grid_w, 14, grid_w, 14, 3) and count non-black
    # pixels per (grid_w, grid_w) cell.
    arr = arr.reshape(grid_w, _PATCH_SIZE, grid_w, _PATCH_SIZE, 3)
    nonzero = (arr.sum(axis=-1) > 0)               # (gw, 14, gw, 14)
    counts = nonzero.sum(axis=(1, 3))              # (gw, gw)
    threshold = max(1, int(round(_PATCH_SIZE * _PATCH_SIZE * PATCH_FG_PIXEL_FRAC)))
    return (counts >= threshold).reshape(-1)       # (P,)


@torch.inference_mode()
def encode_image_patches(pil: PILImage.Image) -> tuple[np.ndarray, np.ndarray]:
    """Per-patch encoding for patch-level matching.

    Returns (patch_tokens, fg_mask) where:
      - patch_tokens: (P, D) float32, L2-normalised per row
      - fg_mask: (P,) bool — True where the patch overlaps the
        SAM-masked foreground (per `_patch_fg_mask`).

    P = (INPUT_SIDE / 14)^2 (1369 at 518×518). D = EMBEDDING_DIM
    (1024 for ViT-L/14).

    No TTA — patches already encode position-specific detail, so
    averaging across views washes out the spatial information that
    patch matching exists to use. CLS token is dropped (we only
    care about the patch grid here).

    Returns (zeros, zeros) on any failure so callers can guard
    via fg_mask.any().
    """
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
        # last_hidden_state: (1, 1+P, D). Drop CLS (index 0), keep
        # patches and L2-normalise so cosine sim is dot product.
        patches = outputs.last_hidden_state[0, 1:, :]  # (P, D)
        patches = torch.nn.functional.normalize(patches.float(), dim=-1)
        return patches.detach().cpu().numpy().astype(np.float32, copy=False), fg
    except Exception as e:
        print(f"[v2-dino] encode_image_patches failed: {e}")
        return (np.zeros((n_patches, EMBEDDING_DIM), dtype=np.float32),
                np.zeros(n_patches, dtype=bool))


@torch.inference_mode()
def encode_image(pil: PILImage.Image) -> np.ndarray:
    """Encode a single PIL image → (768,) L2-normalised float32.

    Matches `DINOv2Embedder.encode_image` in detect_and_crop.py:
    EXIF-transpose → RGB → DINOv2 processor → patch_mean pool → L2
    normalise.
    """
    if not is_loaded():
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)
    try:
        img = ImageOps.exif_transpose(pil).convert("RGB")
        inputs = _PROCESSOR(images=img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(_DEVICE)
        if pixel_values.dtype != _MODEL.dtype:
            pixel_values = pixel_values.to(_MODEL.dtype)
        # Foreground patch mask in (1, P) — _pool drops black-only
        # patches from the patch_mean so they don't dilute the
        # embedding. (1, P) instead of (P,) so the mask broadcasts
        # over the (1, 1+P, D) token tensor.
        fg = torch.from_numpy(_patch_fg_mask(img)).unsqueeze(0).to(_DEVICE)
        outputs = _MODEL(pixel_values=pixel_values, interpolate_pos_encoding=True)
        vec = _pool(outputs.last_hidden_state, fg_mask=fg)[0]
        vec = torch.nn.functional.normalize(vec.float(), dim=0)
        return vec.detach().cpu().numpy().astype(np.float32, copy=False)
    except Exception as e:
        print(f"[v2-dino] encode_image failed: {e}")
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)


def _tta_views(img: PILImage.Image) -> list[PILImage.Image]:
    """Three identity-preserving views for multi-view ensembling.
    Per the user's instruction: original crop, horizontal flip, and
    a zoomed-out version with ~10% extra context padding on each
    side. Embeddings of the three are L2-normalised, mean-pooled,
    and re-normalised so the per-image output is the centroid.

    The intuition is that subtle distinctions like ear length or
    limb proportion can flip between views — flipping reveals the
    object from the mirror angle, and the zoomed-out view brings
    in surrounding context so the network sees proportion against
    a wider frame. Voting across views is more discriminative than
    a single tight crop on fine-grained pairs (hare vs rabbit,
    standing vs lying horse).

    Padding fills with BLACK to match the upstream `inpaint_bbox_crop`
    background (which also fills non-mask pixels with black). Using
    the same fill keeps the TTA's zoom-out frame consistent with
    the masked input — no second background colour for the model
    to encode.

    Vertical flip / large rotation are deliberately excluded — they
    invalidate identity for orientation-sensitive labels (number
    plates, faces, "horse standing vs lying", etc.).
    """
    w, h = img.size
    views: list[PILImage.Image] = [img]

    # H-flip.
    views.append(img.transpose(PILImage.FLIP_LEFT_RIGHT))

    # Zoom out: pad 10% of the original side on each edge with black.
    # Total frame grows to (1.2 × w, 1.2 × h) so the object sits
    # ~83% of the original size in the new frame, giving the model
    # more surrounding context once the processor resizes everything
    # back to its canonical input grid.
    pw = max(2, int(round(w * 0.10)))
    ph = max(2, int(round(h * 0.10)))
    padded = PILImage.new("RGB", (w + 2 * pw, h + 2 * ph), (0, 0, 0))
    padded.paste(img, (pw, ph))
    views.append(padded)

    return views


@torch.inference_mode()
def encode_images_batch(pils: list[PILImage.Image], *, tta: bool = True) -> np.ndarray:
    """Encode N PIL images → (N, EMBEDDING_DIM) L2-normalised float32.

    TTA is ON by default for every caller (references at upload, the
    edit-flush PUT, the imports pipeline, click-to-detect, and the
    standalone /embed_crops endpoint) — fine-grained discrimination
    relies on the multi-view smoothing, so the only path that opts
    out is `warmup` where the output is discarded anyway.

    With ``tta=True`` (default) each image is expanded into the three
    identity-preserving views from `_tta_views` (original, horizontal
    flip, +10% zoom-out with black pad). All views go through one
    batched forward pass at 518×518 grouped back to (N, 3, D), each
    view L2-normalised, mean-pooled across views, and re-normalised.
    The per-image output is the centroid of three view embeddings —
    same dim, lower variance, better robustness to lighting / pose
    quirks that would tilt a single pass.

    With ``tta=False`` it's a plain batched encode (used by warmup).

    Returns a zero matrix on any failure so callers can keep their
    array-aligned indexing.
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
            # (n*V, 1+P, D) → pool with per-view fg mask → (n*V, D) →
            # per-view L2 → mean over views → re-L2. Mask gates the
            # patch_mean so each view's embedding is built from its
            # own foreground patches only.
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
    """Crop `image` to `box`, then replace non-mask pixels with a flat
    fill colour so DINOv2 only sees the object SAM identified.

    Default fill is now BLACK (0, 0, 0) — switched from grey on the
    user's instruction. The intuition: grey lands on the centre of
    the ImageNet normalisation distribution and contributes "neutral"
    activations, but in practice the patch_mean pool still averages
    those neutral patches with object patches and dilutes the signal
    by a factor of (n_bg / n_total). A black background activates a
    consistent, high-contrast silhouette boundary at the edge of the
    object — which is exactly the kind of feature DINOv2 was trained
    to encode strongly. Reference and live-query crops both use the
    same procedure, so any systematic bias from the black fill is
    common-mode and cancels in the cosine match. Pass
    `fill_color=(127, 127, 127)` at the call site to revert.

    Falls back to a plain bbox crop when:
      * No polygons supplied (no mask available)
      * Polygons are degenerate / empty
      * Mask covers the entire bbox (nothing to fill)
    """
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
        return crop_pil  # mask landed entirely outside the bbox
    if mask.all():
        return crop_pil  # object fills the entire bbox; nothing to fill

    # Cleaner segmentation step:
    #  1. Morphological CLOSE (3x3 kernel, 1 iter) plugs single-pixel
    #     holes inside the mask that the polygon→raster conversion
    #     leaves behind on noisy SAM contours. Without this the
    #     interior of the object can have stray "fill" patches that
    #     contribute black tokens to the embedding.
    #  2. 1-pixel DILATE feathers the boundary outward so the polygon's
    #     under-shoot (most polygon approximations sit slightly inside
    #     the true mask edge) doesn't clip the object's outer texture.
    # Both ops are cheap on the typical bbox crop (<512 px side).
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    arr = np.array(crop_pil)
    fill_mask = (mask == 0)
    arr[fill_mask] = fill_color
    return PILImage.fromarray(arr)
