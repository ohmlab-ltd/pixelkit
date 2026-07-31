"""Per-segmentation embeddings via DINOv2.

Pipeline for each segmentation:
  1. Rasterise the polygon mask onto a copy of the original image.
  2. Replace non-mask pixels with neutral grey (128). This isolates
     the object so the embedding reflects the object itself, not
     incidental background pixels around it.
  3. Crop to the box bounding box with a small padding ring so the
     model sees a tiny strip of context.
  4. Resize to 224×224 (DINOv2 ViT-B/14 expects multiples of 14;
     224 = 16 patches square, the canonical input size).
  5. Run through DINOv2 ViT-B/14, take the CLS-token embedding,
     L2-normalise. Result: a 768-dim float32 vector.

Storage is per-project, in two files that sit next to the manifest:
  * embeddings.npy   — float32 array, shape (N, 768)
  * embeddings.json  — metadata, list of N rows aligned with the npy

Both rewrite as a unit on every change. Reads load the npy lazily and
keep a per-project in-memory cache keyed by mtime.
"""
from __future__ import annotations

import io
import json
import os
import threading
import time
import uuid as _uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image as PILImage

# Vision encoder for Label Cascade. SigLIP 2 outperforms CLIP and
# DINOv2 on object-level retrieval at this scale, and its image-text
# alignment objective biases the embedding toward "what is the
# object" rather than "what's the overall scene like". Override via
# `EMBED_MODEL` env var if you want to A/B another model.
EMBED_MODEL = os.environ.get("EMBED_MODEL", "google/siglip2-base-patch16-224")
EMBEDDING_DIM = 768
INPUT_SIZE = 224
NEUTRAL_GREY = 128
# 25% of the box size as a padding ring on each side. Wider context
# helps SigLIP attach the embedding to the object's surroundings,
# which is what the centre-weighted alpha then refocuses on the
# object itself. The previous 8% gave the model almost nothing
# beyond the box and recall suffered.
CROP_PAD_FRAC = 0.25
# Centre-weighting parameters. The square-padded crop is fed through
# a smooth alpha mask that keeps the centre at full intensity and
# fades the edges toward neutral grey, so the encoder's attention
# is pulled toward the object (which always sits in the middle of
# the canvas after square padding) rather than incidental context.
# `CENTRE_FALLOFF` controls how sharp the focus is: higher values =
# tighter centre. 1.5 is gentle (corners ≈ 0.6 alpha), 3.0 is heavy.
CENTRE_FALLOFF = 1.6
CENTRE_MIN_ALPHA = 0.45  # corners stay at least this visible
# Bump whenever the encode pipeline changes — model swap, padding
# strategy, centre-weighting, anything that changes the vector for
# the same input. The refresh path requires this to match before
# considering a row up to date, so old rows get rebuilt under the
# new transform.
ENCODER_VERSION = 4

# Same cache directory as VLM/SigLIP shares with the rest of HF.
_DEFAULT_CACHE = Path(__file__).resolve().parent.parent / "models_cache"
EMBED_CACHE_DIR = Path(os.environ.get("EMBED_CACHE_DIR") or os.environ.get("HF_HOME") or _DEFAULT_CACHE)
EMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(EMBED_CACHE_DIR))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(EMBED_CACHE_DIR / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(EMBED_CACHE_DIR / "transformers"))

_MODEL = None
_PROCESSOR = None
_DEVICE = "cpu"
_LOAD_LOCK = threading.Lock()


def is_loaded() -> bool:
    return _MODEL is not None and _PROCESSOR is not None


def load_dinov2(device: str = "cpu"):
    """Load the embedding model lazily. Function name kept for
    backward compatibility with the lifespan loader; under the
    hood this now loads SigLIP 2 by default. Idempotent."""
    global _MODEL, _PROCESSOR, _DEVICE
    with _LOAD_LOCK:
        if is_loaded() and _DEVICE == device:
            return _MODEL, _PROCESSOR
        from transformers import AutoModel, AutoProcessor
        print(f"[embed] loading {EMBED_MODEL} on {device}...")
        dtype = torch.float16 if device == "cuda" else torch.float32
        model = AutoModel.from_pretrained(EMBED_MODEL, torch_dtype=dtype).to(device).eval()
        processor = AutoProcessor.from_pretrained(EMBED_MODEL)
        _MODEL = model
        _PROCESSOR = processor
        _DEVICE = device
        print(f"[embed] ready ({EMBEDDING_DIM}-dim, {dtype}, {device}).")
        return _MODEL, _PROCESSOR


def warmup() -> None:
    if not is_loaded():
        return
    try:
        dummy = PILImage.new("RGB", (INPUT_SIZE, INPUT_SIZE), color=(127, 127, 127))
        encode_image(dummy)
    except Exception as e:
        print(f"[embed] warmup failed: {e}")


# Cached centre-weighting alpha — same shape every call (after the
# square-pad), so we can precompute once per crop size.
_ALPHA_CACHE: dict[int, np.ndarray] = {}


def _centre_alpha(size: int) -> np.ndarray:
    """Smooth radial alpha: 1.0 at the centre, falling off to
    `CENTRE_MIN_ALPHA` at the corners."""
    cached = _ALPHA_CACHE.get(size)
    if cached is not None:
        return cached
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    cy = cx = (size - 1) / 2.0
    # Normalised radial distance — 0 at centre, 1 at corner.
    r = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2) / np.sqrt(2.0)
    alpha = np.exp(-CENTRE_FALLOFF * r * r).astype(np.float32)
    alpha = np.clip(alpha, CENTRE_MIN_ALPHA, 1.0)
    _ALPHA_CACHE[size] = alpha
    return alpha


def _apply_centre_weight(rgb: np.ndarray) -> np.ndarray:
    """Blend the crop with neutral grey using the radial alpha. The
    object (centred via square padding) stays sharp; edges and
    corners fade toward grey so they contribute less to the
    embedding without disappearing entirely."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return rgb
    h, w = rgb.shape[:2]
    if h != w:
        # Defensive — caller should square-pad before this. If not,
        # use the smaller dim so we still get a circular fade.
        side = min(h, w)
    else:
        side = h
    alpha = _centre_alpha(side)
    if alpha.shape != rgb.shape[:2]:
        # Crop doesn't match the cached size (rare path). Resize the
        # alpha to fit instead of blowing the cache on every call.
        alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)
    a = alpha[..., None]
    out = a * rgb.astype(np.float32) + (1.0 - a) * float(NEUTRAL_GREY)
    return np.clip(out, 0, 255).astype(np.uint8)


def _build_mask(polygons: list, height: int, width: int) -> np.ndarray:
    """Rasterise polygon list onto a binary HxW uint8 mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    if not polygons:
        return mask
    contours = []
    for poly in polygons:
        if not poly or len(poly) < 3:
            continue
        try:
            arr = np.asarray(poly, dtype=np.int32)
            if arr.ndim != 2 or arr.shape[1] != 2:
                continue
            contours.append(arr)
        except Exception:
            continue
    if contours:
        cv2.fillPoly(mask, contours, 255)
    return mask


def _isolate_segmentation(image_pil: PILImage.Image, polygons: list, box_xyxy: list[float]) -> PILImage.Image | None:
    """Crop the natural image around the box (with a small padding
    ring) and square-pad with edge-replicated pixels.

    Why no grey-background mask isolation: DINOv2 was trained on
    natural photographs. Replacing background with grey 128 produces
    a strongly out-of-distribution input — the model can still run
    on it but its features collapse toward "something on a grey
    canvas" rather than capturing the object's actual visual
    identity, so two photos of the same kind of object end up with
    very different embeddings. Feeding it the raw box content (with
    natural surroundings) recovers the recall the masking lost.

    Square-pad still happens, but with REFLECT instead of grey fill
    so the canvas continues looking photographic — the model isn't
    drawn to a synthetic grey halo. Only fires when the crop isn't
    already square."""
    if not polygons or len(box_xyxy) != 4:
        return None
    rgb = np.asarray(image_pil.convert("RGB"))
    H, W = rgb.shape[:2]

    x0, y0, x1, y1 = (float(v) for v in box_xyxy)
    bw = max(1.0, x1 - x0)
    bh = max(1.0, y1 - y0)
    pad_w = bw * CROP_PAD_FRAC
    pad_h = bh * CROP_PAD_FRAC
    cx0 = max(0, int(round(x0 - pad_w)))
    cy0 = max(0, int(round(y0 - pad_h)))
    cx1 = min(W, int(round(x1 + pad_w)))
    cy1 = min(H, int(round(y1 + pad_h)))
    if cx1 - cx0 < 4 or cy1 - cy0 < 4:
        return None
    crop = rgb[cy0:cy1, cx0:cx1]

    # Square-pad via reflection so the resize doesn't distort aspect
    # and the model doesn't see synthetic borders.
    ch, cw = crop.shape[:2]
    if ch != cw:
        target = max(ch, cw)
        oy = (target - ch) // 2
        ox = (target - cw) // 2
        try:
            crop = cv2.copyMakeBorder(
                crop,
                top=oy,
                bottom=target - ch - oy,
                left=ox,
                right=target - cw - ox,
                borderType=cv2.BORDER_REFLECT_101,
            )
        except Exception:
            # Fall back to neutral grey on the off chance reflect
            # blows up (e.g. a 1-px crop) — rare path.
            canvas = np.full((target, target, 3), NEUTRAL_GREY, dtype=np.uint8)
            canvas[oy:oy + ch, ox:ox + cw] = crop
            crop = canvas

    # Apply the radial alpha so the object's centre dominates the
    # embedding while edges fade toward neutral grey. This is what
    # gives the encoder a "tighter" focus without a hard mask.
    crop = _apply_centre_weight(crop)
    return PILImage.fromarray(crop)


def encode_image(pil: PILImage.Image) -> np.ndarray:
    """Encode a single PIL image → (768,) L2-normalised float32.

    Tries `get_image_features` first (CLIP / SigLIP / SigLIP-2
    expose it), then falls back to a direct vision-tower call.
    Some transformers versions wrap `get_image_features` output in
    a `BaseModelOutputWithPooling` instead of returning a plain
    Tensor — we unwrap by reaching for `.pooler_output` or the
    CLS token before the L2-normalise so `.norm()` doesn't blow
    up on the container object.
    """
    if not is_loaded():
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)
    try:
        inputs = _PROCESSOR(images=pil, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(_DEVICE)
        if pixel_values.dtype != _MODEL.dtype:
            pixel_values = pixel_values.to(_MODEL.dtype)
        with torch.inference_mode():
            feats: torch.Tensor | None = None
            if hasattr(_MODEL, "get_image_features"):
                try:
                    raw = _MODEL.get_image_features(pixel_values=pixel_values)
                except Exception:
                    raw = None
                if isinstance(raw, torch.Tensor):
                    feats = raw
                elif raw is not None:
                    # Wrapped in BaseModelOutputWithPooling on some
                    # transformers builds. Pooler output is the
                    # canonical image feature; CLS-token of
                    # last_hidden_state is the standard fallback.
                    pooled = getattr(raw, "pooler_output", None)
                    if isinstance(pooled, torch.Tensor):
                        feats = pooled
                    else:
                        lhs = getattr(raw, "last_hidden_state", None)
                        if isinstance(lhs, torch.Tensor):
                            feats = lhs[:, 0, :]
            if feats is None:
                # Direct vision-tower call. Works for DINOv2, and as
                # a hard fallback for SigLIP if `get_image_features`
                # didn't yield anything usable.
                vm = getattr(_MODEL, "vision_model", None) or _MODEL
                vout = vm(pixel_values=pixel_values)
                pooled = getattr(vout, "pooler_output", None)
                if isinstance(pooled, torch.Tensor):
                    feats = pooled
                else:
                    feats = vout.last_hidden_state[:, 0, :]
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return feats.detach().cpu().float().numpy().astype(np.float32, copy=False)[0]
    except Exception as e:
        print(f"[embed] encode_image failed: {e}")
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)


def encode_segmentation(
    image_pil: PILImage.Image,
    polygons: list,
    box_xyxy: list[float],
) -> np.ndarray | None:
    """Compute the embedding for one segmentation. Returns None when
    the polygon is empty or the crop comes out unusable."""
    isolated = _isolate_segmentation(image_pil, polygons, box_xyxy)
    if isolated is None:
        return None
    return encode_image(isolated)


# ---------------------------------------------------------------------------
# Per-project storage (npy + json sidecar).
# ---------------------------------------------------------------------------

def _paths(project_dir: Path) -> tuple[Path, Path]:
    return project_dir / "embeddings.npy", project_dir / "embeddings.json"


_CACHE: dict[str, tuple[float, np.ndarray, list[dict]]] = {}
_CACHE_LOCK = threading.Lock()


def load_store(project_dir: Path) -> tuple[np.ndarray, list[dict]]:
    """Load embeddings + metadata for a project. Returns ((N, D)
    float32 array, list-of-dicts metadata) — empty when nothing
    is stored yet. Cached by mtime so repeated calls are cheap."""
    npy_path, json_path = _paths(project_dir)
    if not npy_path.exists() or not json_path.exists():
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32), []
    key = str(project_dir)
    try:
        mtime = max(npy_path.stat().st_mtime, json_path.stat().st_mtime)
    except OSError:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32), []
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if cached and cached[0] == mtime:
            return cached[1], cached[2]
    try:
        arr = np.load(npy_path)
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        if not isinstance(meta, list):
            meta = []
        if arr.shape[0] != len(meta):
            # Out of sync — rebuild from scratch on next refresh.
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32), []
        with _CACHE_LOCK:
            _CACHE[key] = (mtime, arr, meta)
        return arr, meta
    except Exception as e:
        print(f"[dinov2] load_store failed: {e}")
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32), []


def save_store(project_dir: Path, embeddings: np.ndarray, meta: list[dict]) -> None:
    """Atomic write of the per-project store. Both files swap together
    so a partial failure can't leave the npy and the metadata out of
    sync."""
    npy_path, json_path = _paths(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)
    # Tmp paths must keep the same trailing extension as the target
    # (`.npy` for the array, `.json` for the metadata). `np.save`
    # auto-appends `.npy` when the path doesn't end in it, which
    # silently writes to the wrong file and breaks the os.replace
    # below. Putting the disambiguator BEFORE the extension avoids
    # that.
    npy_tmp = npy_path.with_name(npy_path.stem + ".tmp.npy")
    json_tmp = json_path.with_name(json_path.stem + ".tmp.json")
    try:
        # Pass the open file handle so np.save won't mangle the
        # filename — belt-and-braces with the .tmp.npy suffix above.
        with open(npy_tmp, "wb") as fh:
            np.save(fh, embeddings.astype(np.float32, copy=False))
        json_tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        os.replace(npy_tmp, npy_path)
        os.replace(json_tmp, json_path)
    finally:
        for p in (npy_tmp, json_tmp):
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass
    # Bust cache.
    with _CACHE_LOCK:
        _CACHE.pop(str(project_dir), None)


def upsert_rows(
    project_dir: Path,
    rows: list[tuple[dict, np.ndarray]],
) -> None:
    """Insert or update embedding rows. `rows` is a list of (metadata,
    embedding) tuples; metadata must contain a stable `box_id` so we
    can match against existing rows."""
    if not rows:
        return
    arr, meta = load_store(project_dir)
    by_id: dict[str, int] = {}
    for i, m in enumerate(meta):
        bid = m.get("box_id")
        if bid:
            by_id[str(bid)] = i

    new_arr = arr.tolist() if arr.size else []
    new_meta = list(meta)
    now = _now_iso()
    for m, vec in rows:
        bid = str(m.get("box_id") or "")
        if not bid:
            continue
        m["updated_at"] = now
        if "id" not in m:
            m["id"] = _uuid.uuid4().hex
        if bid in by_id:
            idx = by_id[bid]
            new_meta[idx] = m
            new_arr[idx] = vec.tolist()
        else:
            new_meta.append(m)
            new_arr.append(vec.tolist())
            by_id[bid] = len(new_meta) - 1

    save_store(project_dir, np.asarray(new_arr, dtype=np.float32), new_meta)


def update_label(project_dir: Path, box_ids: list[str], new_label: str) -> int:
    """Update the stored `label` for one or more box ids without
    re-encoding (same visual content, only the textual annotation
    changed). Returns the number of rows updated."""
    if not box_ids:
        return 0
    arr, meta = load_store(project_dir)
    if not meta:
        return 0
    target = {str(b) for b in box_ids}
    n = 0
    now = _now_iso()
    for m in meta:
        if str(m.get("box_id")) in target:
            if m.get("label") != new_label:
                m["label"] = new_label
                m["updated_at"] = now
                n += 1
    if n:
        save_store(project_dir, arr, meta)
    return n


def mark_ignored(project_dir: Path, box_ids: list[str]) -> int:
    """Flag rows so future Label Cascade searches skip them. The row
    keeps its embedding (so the box still gets the visual fingerprint
    treatment for other features that might use it) but `find_similar`
    filters it out, both as a candidate match and as an exclude path
    for the trigger box. Returns the number of rows updated."""
    if not box_ids:
        return 0
    arr, meta = load_store(project_dir)
    if not meta:
        return 0
    target = {str(b) for b in box_ids}
    n = 0
    now = _now_iso()
    for m in meta:
        if str(m.get("box_id")) in target and not m.get("cascade_ignored"):
            m["cascade_ignored"] = True
            m["cascade_ignored_at"] = now
            n += 1
    if n:
        save_store(project_dir, arr, meta)
    return n


def remove_rows(project_dir: Path, box_ids: list[str]) -> int:
    """Drop rows whose `box_id` matches. Used when a box is deleted
    from the manifest; the embedding store would otherwise hold a
    dead reference to it."""
    if not box_ids:
        return 0
    arr, meta = load_store(project_dir)
    if not meta:
        return 0
    target = {str(b) for b in box_ids}
    keep_idx = [i for i, m in enumerate(meta) if str(m.get("box_id")) not in target]
    if len(keep_idx) == len(meta):
        return 0
    new_meta = [meta[i] for i in keep_idx]
    new_arr = arr[keep_idx] if arr.size else arr
    save_store(project_dir, new_arr, new_meta)
    return len(meta) - len(keep_idx)


def find_similar(
    project_dir: Path,
    query: np.ndarray,
    *,
    threshold: float = 0.35,
    max_results: int = 12,
    exclude_box_ids: set[str] | None = None,
    label_filter: str | None = None,
    query_size_frac: float | None = None,
) -> list[dict]:
    """Find rows whose visual embedding is similar to `query`.

    Threshold sits on COSINE alone — that's the primary signal we
    want to gate on. Size shows up as a soft additive bonus / penalty
    to break ties and gently demote wildly-different-scale matches,
    rather than a multiplicative factor that could yank a 0.85
    cosine match below threshold just because the candidate is
    half the relative size.

    Score formula:
        score = cosine + 0.10 * (size_factor - 1)
        size_factor = exp(-0.15 * |log(query_size / row_size)|)

    For matching size: bonus = 0 (no effect).
    For 2× size diff: bonus ≈ -0.01.
    For 10× size diff: bonus ≈ -0.03.

    So size only matters at the margins; it nudges ranking and a
    truly disparate-scale candidate gets a small demerit, but it
    never blocks a genuinely-strong cosine match.

    The 0.62 cosine floor reflects what DINOv2 actually produces
    for "same kind of object, different pose / lighting / instance"
    — typically 0.6–0.85. Anything tighter was missing real matches.
    """
    arr, meta = load_store(project_dir)
    if arr.size == 0:
        return []
    query = query.astype(np.float32, copy=False)
    sims = arr @ query  # (N,) cosine

    if query_size_frac is not None and query_size_frac > 0:
        sizes = np.array(
            [float(m.get("size_frac") or query_size_frac) for m in meta],
            dtype=np.float32,
        )
        ratio = np.clip(sizes / float(query_size_frac), 1e-3, 1e3)
        size_factor = np.exp(-0.15 * np.abs(np.log(ratio)))
        combined = sims + 0.10 * (size_factor - 1.0)
    else:
        combined = sims

    order = np.argsort(-combined)
    excl = exclude_box_ids or set()
    out: list[dict] = []
    for idx in order:
        if len(out) >= max_results:
            break
        cos = float(sims[int(idx)])
        if cos < threshold:
            # Threshold is on cosine itself, not the combined score —
            # we don't want a tiny size bonus dragging an obvious
            # mismatch over the line.
            break
        m = meta[int(idx)]
        if str(m.get("box_id")) in excl:
            continue
        if m.get("cascade_ignored"):
            # User explicitly marked this row as "don't surface in
            # Label Cascade again". Honour that everywhere.
            continue
        if label_filter is not None and m.get("label") != label_filter:
            continue
        out.append({
            **m,
            "similarity": round(cos, 4),
            "score": round(float(combined[int(idx)]), 4),
        })
    return out


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
