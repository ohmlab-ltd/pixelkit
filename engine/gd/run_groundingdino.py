"""Run MM-Grounding-DINO-L on a single image with text prompts.

The previous build used IDEA-Research's GroundingDINO_SwinB checkpoint via the
local repo. We now load `rziga/mm_grounding_dino_large_all` through the
HuggingFace transformers API — same architecture family, retrained on far more
data (+22% LVIS minival mAP vs SwinB), no mmcv/mmdet dependency.
"""
import argparse
import os
import re
from typing import Iterable

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

# Ampere (RTX 30xx) and later GPUs enable TF32 by default, which reduces
# float32 matmul precision and causes transformer models to produce
# different (incorrect) box coordinates vs CPU/MPS. Force full float32.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Kept as named constants so server.py can import them; values are now HF model IDs.
DEFAULT_CONFIG = "rziga/mm_grounding_dino_large_all"
DEFAULT_CHECKPOINT = "rziga/mm_grounding_dino_large_all"


def load_image(image_path):
    """Return (PIL.Image, PIL.Image). Applies EXIF orientation so PIL coordinates
    match what the browser renders. Second copy kept for API parity.
    """
    from PIL import ImageOps
    image_pil = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    return image_pil, image_pil


def load_model(config_path, checkpoint_path, device):
    """Load MM-GD-L at fp32. Mixed-precision inference is handled in
    `predict` via `torch.autocast` rather than by casting the whole
    model up front — `torch_dtype=fp16` on `from_pretrained` doesn't
    reliably cast every submodule of MM-GD-L (the text-encoder query
    projections in particular keep fp32 weights and then mismatch
    fp16 activations at runtime: `mat1 and mat2 must have the same
    dtype`). Autocast resolves this by op-level type promotion so
    every Linear / matmul gets a consistent dtype regardless of how
    the model was loaded."""
    model_id = checkpoint_path or DEFAULT_CHECKPOINT
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    model.eval()
    return {"model": model, "processor": processor, "device": device, "id": model_id}


def _normalize_tags(prompt_or_tags) -> list[str]:
    if isinstance(prompt_or_tags, (list, tuple)):
        tags = [str(t).strip().lower() for t in prompt_or_tags if str(t).strip()]
    else:
        # Accept legacy "a. b. c." prompt strings
        tags = [t.strip().lower() for t in str(prompt_or_tags).replace(",", ".").split(".") if t.strip()]
    seen, out = set(), []
    for t in tags:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def predict(bundle, image, prompt_or_tags, box_threshold, text_threshold, device, nms_iou=0.5):
    """Run MM-GD-L on a single PIL image with a list of text labels.

    Returns (boxes_xyxy_pixels: Tensor[N,4], phrases: list[str]) where each
    phrase is "<label> (<score>)" — same shape server.py already expects.

    `text_threshold` is unused (HF post-processor handles label assignment);
    kept in the signature so the server's threshold UI keeps working.
    """
    model = bundle["model"]
    processor = bundle["processor"]
    tags = _normalize_tags(prompt_or_tags)
    if not tags:
        return torch.zeros((0, 4)), []

    inputs = processor(images=image, text=[tags], return_tensors="pt").to(device)
    # Autocast the forward pass on CUDA: every op runs in the cheaper
    # dtype that PyTorch's amp registry has whitelisted, which gives
    # us most of the VRAM and latency savings of pure-fp16 inference
    # without the dtype-mismatch hazards of casting the model wholesale.
    # Activation memory drops ~40-50% during the encoder forward pass
    # — the difference between fitting on a 12 GB card and OOMing on
    # the deformable-attention `grid_sample` workspace allocation.
    use_autocast = str(device).startswith("cuda")
    with torch.no_grad(), torch.autocast(
        device_type="cuda" if use_autocast else "cpu",
        dtype=torch.float16,
        enabled=use_autocast,
    ):
        outputs = model(**inputs)

    # The rziga/mm_grounding_dino_large_all checkpoint only ships bbox_embed[0]
    # (it was trained with a single shared box head). transformers ≥ 5.x ties
    # that one head across all 6 decoder layers, so the same refinement gets
    # applied six times — the Y dimension of every box collapses and detections
    # render as thin horizontal stripes. The first decoder layer's reference
    # (one refinement pass) reproduces the original mmdet MM-GD behaviour and
    # matches what an IDEA-Research grounding-dino checkpoint produces.
    if (
        getattr(model.config, "model_type", "") == "mm-grounding-dino"
        and getattr(outputs, "intermediate_reference_points", None) is not None
    ):
        outputs.pred_boxes = outputs.intermediate_reference_points[:, 0]

    W, H = image.size
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        threshold=float(box_threshold),
        text_threshold=float(text_threshold),
        target_sizes=[(H, W)],
        text_labels=[tags],
    )[0]

    boxes = results["boxes"].detach().cpu()
    scores = results["scores"].detach().cpu()
    labels = results.get("text_labels") or results.get("labels") or []

    # No detections survived the threshold. The HF post-processor still
    # returns the prompt's `text_labels` regardless of detections, so
    # `len(labels)` and `boxes.shape[0]` desync (boxes=[0,4],
    # labels=[<tag>]) and the downstream `boxes[keep_mask]` blows up
    # with a shape-mismatch IndexError. Bail before that path.
    if boxes.shape[0] == 0:
        print(f"[gd] tags={tags} raw=0 kept=0 (no boxes above threshold)")
        return boxes, []

    # Some HF builds return labels of length != len(boxes) even when boxes
    # is non-empty (text_labels mirrors the prompt). Trim to match so the
    # zip + keep_mask paths below stay aligned.
    if len(labels) != boxes.shape[0]:
        labels = list(labels)[: boxes.shape[0]] + [""] * max(0, boxes.shape[0] - len(labels))

    # Loose thresholds let the model fire on tokens that don't cleanly map to
    # any requested tag — the processor returns those as empty strings or
    # partial substrings. We accept any label that CONTAINS one of the
    # prompted tags as a whole word, then rewrite the label to the tag
    # itself. This recovers detections like "the road" / "road surface"
    # for the "road" prompt while still rejecting unrelated phrases.
    raw_total = len(labels)
    tag_set = {t.lower() for t in tags}
    rewritten: list[str | None] = []
    for lab in labels:
        if not lab:
            rewritten.append(None)
            continue
        tokens = re.findall(r"[a-z0-9]+", lab.lower())
        # Try multi-word tags first (e.g. "stop sign"), then single
        # tokens. First match wins so we preserve user intent.
        match: str | None = None
        for tag in tags:
            tag_tokens = re.findall(r"[a-z0-9]+", tag.lower())
            if not tag_tokens:
                continue
            # All of the tag's tokens must appear as contiguous
            # tokens in the label — handles "stop sign" inside
            # "stop sign on the corner".
            for i in range(len(tokens) - len(tag_tokens) + 1):
                if tokens[i:i + len(tag_tokens)] == tag_tokens:
                    match = tag.lower()
                    break
            if match:
                break
        if match is None:
            # Fallback: any single tag token appearing anywhere in
            # the label is enough. Catches truncations like "road"
            # being parsed as part of a longer phrase.
            for tag in tags:
                if tag.lower() in set(tokens):
                    match = tag.lower()
                    break
        rewritten.append(match)

    keep_mask = torch.tensor([m is not None for m in rewritten], dtype=torch.bool)
    if keep_mask.numel() > 0 and not keep_mask.all():
        boxes = boxes[keep_mask]
        scores = scores[keep_mask]
        labels = [m for m in rewritten if m is not None]
    else:
        labels = [m or (lab.strip().lower() if lab else "") for m, lab in zip(rewritten, labels)]
    print(f"[gd] tags={tags} raw={raw_total} kept={len(labels)}")

    if len(boxes) > 1 and nms_iou is not None:
        from torchvision.ops import batched_nms
        # Per-class NMS — same-class duplicates get suppressed, but a "pothole"
        # and a "road" that share pixels both survive.
        label_to_idx = {l: i for i, l in enumerate(dict.fromkeys(labels))}
        idxs = torch.tensor([label_to_idx[l] for l in labels])
        keep = batched_nms(boxes, scores, idxs, float(nms_iou))
        boxes = boxes[keep]
        scores = scores[keep]
        labels = [labels[i] for i in keep.tolist()]

    if len(boxes) > 1:
        boxes, scores, labels = _drop_contained(boxes, scores, labels, frac=0.7)

    # Cross-class dedupe — when GD slaps two labels on essentially the
    # same patch (the classic "road" + "pavement" overlap), keep only
    # the higher-scoring one. High IoU threshold so legitimate co-
    # located objects (person inside a car frame, helmet on a head,
    # pothole on a road) survive.
    if len(boxes) > 1:
        boxes, scores, labels = _drop_overlapping_crossclass(boxes, scores, labels, iou_thr=0.7)

    phrases = [f"{lab} ({s.item():.2f})" for lab, s in zip(labels, scores)]
    return boxes, phrases


# ── Native-resolution tiled inference ────────────────────────────────
# For very large frames (e.g. 4K drone imagery) the processor's resize
# to ~800/1333px leaves a 27px animal at ~9px — below reliable
# detection. predict_tiled slices the image into overlapping native-
# resolution crops, runs `predict` per crop (plus one full-frame pass
# for objects larger than a tile), offsets everything back to global
# pixel coords, and merges with a truncation-aware class-wise NMS so an
# animal seen whole in one tile beats its clipped twin from the
# neighbouring tile. Same return shape as `predict`.

_PHRASE_RE = re.compile(r"^(.*)\s+\(([0-9.]+)\)\s*$")


def _parse_gd_phrase(phrase: str) -> tuple[str, float]:
    m = _PHRASE_RE.match(phrase or "")
    if not m:
        return (phrase or "").strip().lower(), 0.0
    try:
        return m.group(1).strip().lower(), float(m.group(2))
    except ValueError:
        return m.group(1).strip().lower(), 0.0


def _tile_origins(size: int, tile: int, stride: int) -> list[int]:
    """Top-left origins covering `size` with `tile`-wide windows. The last
    window is shifted back flush with the boundary so the full extent is
    always covered (no sliver tiles)."""
    if size <= tile:
        return [0]
    out = list(range(0, size - tile + 1, stride))
    if out[-1] != size - tile:
        out.append(size - tile)
    return out


def _merge_tiled_candidates(cands, iou_thr=0.5, contain_frac=0.7):
    """Greedy same-class merge across tiles. cands rows are
    [x0,y0,x1,y1, score, label, truncated]. Un-truncated boxes win over
    truncated ones, then score, then area — so the fuller view of an
    object suppresses the crop-edge-clipped duplicate from the
    neighbouring tile, the exact failure mode tiling without overlap
    bakes into a dataset.

    Three guards tuned for small-object aerial frames:
    - containment only dedups boxes of comparable size (≤3×), so a
      full-frame "group blob" can't eat the individuals the tiles resolved;
    - near-coincident tiny boxes (<48px) dedup by centre distance — IoU is
      unreliable at that scale (cross-tile localisation jitter of a few px
      drops IoU below any sane threshold);
    - post-merge, a kept box that mostly contains ≥2 smaller kept same-
      class boxes is a phantom cluster around individuals — dropped
      (cross-pass mirror of _drop_contained's rationale)."""
    def area(c):
        return max(c[2] - c[0], 0.0) * max(c[3] - c[1], 0.0)

    order = sorted(cands, key=lambda c: (c[6], -c[4], -area(c)))
    kept: list[list] = []
    for c in order:
        ca = area(c)
        c_ms = min(c[2] - c[0], c[3] - c[1])
        ccx, ccy = (c[0] + c[2]) / 2.0, (c[1] + c[3]) / 2.0
        dup = False
        for k in kept:
            if k[5] != c[5]:
                continue
            ix0, iy0 = max(c[0], k[0]), max(c[1], k[1])
            ix1, iy1 = min(c[2], k[2]), min(c[3], k[3])
            inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
            if inter <= 0:
                continue
            ka = area(k)
            union = ca + ka - inter
            smaller = min(ca, ka) or 1.0
            comparable = max(ca, ka) <= 3.0 * max(min(ca, ka), 1.0)
            k_ms = min(k[2] - k[0], k[3] - k[1])
            kcx, kcy = (k[0] + k[2]) / 2.0, (k[1] + k[3]) / 2.0
            tiny_same_spot = (
                c_ms < 48 and k_ms < 48
                and abs(ccx - kcx) + abs(ccy - kcy) <= 0.75 * max(min(c_ms, k_ms), 1.0)
            )
            if (
                (union > 0 and inter / union >= iou_thr)
                or (comparable and inter / smaller >= contain_frac)
                or tiny_same_spot
            ):
                dup = True
                break
        if not dup:
            kept.append(c)

    # Phantom-cluster eviction: drop a kept box that mostly contains ≥2
    # clearly smaller kept same-class boxes — it's a group blob around
    # individuals the tile passes resolved.
    if len(kept) > 2:
        drop: set[int] = set()
        for i, k in enumerate(kept):
            ka = area(k)
            inside = 0
            for j, c in enumerate(kept):
                if i == j or c[5] != k[5]:
                    continue
                ca = area(c)
                if ca * 2.0 > ka:
                    continue
                ix0, iy0 = max(c[0], k[0]), max(c[1], k[1])
                ix1, iy1 = min(c[2], k[2]), min(c[3], k[3])
                inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
                if ca > 0 and inter / ca >= 0.7:
                    inside += 1
                    if inside >= 2:
                        drop.add(i)
                        break
        if drop:
            kept = [k for i, k in enumerate(kept) if i not in drop]
    return kept


def predict_tiled(
    bundle,
    image,
    prompt_or_tags,
    box_threshold,
    text_threshold,
    device,
    nms_iou=0.5,
    tile_size=1024,
    overlap=0.2,
    cancel_check=None,
):
    """Tiled native-resolution variant of `predict` for large images.

    Drop-in: same (boxes_xyxy_pixels, phrases) return shape. Falls back
    to plain `predict` when the image isn't meaningfully larger than one
    tile. `cancel_check` (optional callable -> bool) aborts between tile
    passes so job cancellation stays responsive mid-image."""
    W, H = image.size
    tile = max(256, int(tile_size or 1024))
    if max(W, H) <= tile * 1.25:
        return predict(bundle, image, prompt_or_tags, box_threshold, text_threshold, device, nms_iou=nms_iou)
    ov = min(max(float(overlap or 0.0), 0.0), 0.8)
    stride = max(1, int(tile * (1.0 - ov)))
    if cancel_check is not None and cancel_check():
        return torch.zeros((0, 4)), []

    cands: list[list] = []  # [x0,y0,x1,y1, score, label, truncated]

    def _collect(boxes_t, phrases, ox, oy, tx1, ty1):
        edge = 4.0
        for b, ph in zip(boxes_t.tolist(), phrases):
            label, score = _parse_gd_phrase(ph)
            gx0, gy0, gx1, gy1 = b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy
            # A box hugging an interior crop edge is likely mid-object;
            # flag it so the merge prefers a fuller neighbouring view.
            truncated = (
                (gx0 - ox <= edge and ox > 0)
                or (gy0 - oy <= edge and oy > 0)
                or (tx1 - gx1 <= edge and tx1 < W)
                or (ty1 - gy1 <= edge and ty1 < H)
            )
            # Edge tiles in the short dimension extend past the image and
            # PIL zero-pads the crop — clamp so a box straddling the real
            # edge can't carry out-of-frame coords into the manifest.
            gx0 = min(max(gx0, 0.0), float(W)); gy0 = min(max(gy0, 0.0), float(H))
            gx1 = min(max(gx1, 0.0), float(W)); gy1 = min(max(gy1, 0.0), float(H))
            cands.append([gx0, gy0, gx1, gy1, score, label, truncated])

    # Full-frame pass first — catches objects larger than a tile (or
    # dense groups) that per-tile passes only ever see in pieces.
    fb, fp = predict(bundle, image, prompt_or_tags, box_threshold, text_threshold, device, nms_iou=nms_iou)
    _collect(fb, fp, 0, 0, W, H)

    n_tiles = 0
    for oy in _tile_origins(H, tile, stride):
        for ox in _tile_origins(W, tile, stride):
            if cancel_check is not None and cancel_check():
                return torch.zeros((0, 4)), []
            crop = image.crop((ox, oy, ox + tile, oy + tile))
            tb, tp = predict(bundle, crop, prompt_or_tags, box_threshold, text_threshold, device, nms_iou=nms_iou)
            _collect(tb, tp, ox, oy, ox + tile, oy + tile)
            n_tiles += 1

    # Decoupled from the per-pass nms_iou: cross-tile duplicates of small
    # boxes carry localisation jitter, so a high user NMS (e.g. 0.7) would
    # let one animal survive as two boxes from adjacent tiles.
    merge_iou = min(float(nms_iou if nms_iou is not None else 0.5), 0.5)
    kept = _merge_tiled_candidates(cands, iou_thr=merge_iou)
    # Safety cap: at very low thresholds a 4K frame can fire thousands of
    # candidates across tiles; every kept box costs a SAM + VLM pass
    # downstream. Keep the top-N by score (env-tunable, generous).
    max_boxes = int(os.environ.get("GD_TILED_MAX_BOXES", "500"))
    if max_boxes > 0 and len(kept) > max_boxes:
        kept = sorted(kept, key=lambda c: -c[4])[:max_boxes]
        print(f"[gd-tiled] capped to top {max_boxes} boxes by score")
    if kept:
        boxes = torch.tensor([k[:4] for k in kept], dtype=torch.float32)
        scores = torch.tensor([k[4] for k in kept], dtype=torch.float32)
        labels = [k[5] for k in kept]
        if len(kept) > 1:
            # Restore the cross-class dedupe predict() applies within one
            # pass — without it a road/pavement pair detected by DIFFERENT
            # passes survives as stacked boxes.
            boxes, scores, labels = _drop_overlapping_crossclass(boxes, scores, labels, iou_thr=0.7)
        phrases = [f"{lab} ({s.item():.2f})" for lab, s in zip(labels, scores)]
    else:
        boxes = torch.zeros((0, 4))
        phrases = []
    print(f"[gd-tiled] {W}x{H} → {n_tiles} tiles @{tile}px (+full frame): {len(cands)} raw → {len(phrases)} merged")
    return boxes, phrases


def _drop_contained(boxes, scores, labels, frac=0.7):
    """Drop a same-class box if `frac` of its area lies inside a higher-scoring
    same-class box. Catches MM-GD's habit of returning both individual and
    "cluster" boxes for the same physical thing — those don't share enough IoU
    for plain NMS to fire."""
    n = len(boxes)
    order = torch.argsort(scores, descending=True).tolist()
    keep_mask = [True] * n
    areas = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).clamp(min=1.0)

    for ii in range(n):
        i = order[ii]
        if not keep_mask[i]:
            continue
        for jj in range(ii + 1, n):
            j = order[jj]
            if not keep_mask[j]:
                continue
            if labels[i] != labels[j]:
                continue
            x0 = max(boxes[i, 0].item(), boxes[j, 0].item())
            y0 = max(boxes[i, 1].item(), boxes[j, 1].item())
            x1 = min(boxes[i, 2].item(), boxes[j, 2].item())
            y1 = min(boxes[i, 3].item(), boxes[j, 3].item())
            inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
            if inter <= 0:
                continue
            # j is the lower-scoring box. Drop it if mostly inside i, OR if i is
            # mostly inside j (i.e. j is a phantom cluster around i).
            if inter / areas[j].item() >= frac or inter / areas[i].item() >= frac:
                keep_mask[j] = False

    keep_idx = [k for k, m in enumerate(keep_mask) if m]
    keep_t = torch.tensor(keep_idx, dtype=torch.long)
    return boxes[keep_t], scores[keep_t], [labels[k] for k in keep_idx]


def _drop_overlapping_crossclass(boxes, scores, labels, iou_thr=0.7):
    """Drop near-duplicate boxes that disagree on class.

    GD will sometimes assign two labels to essentially the same region
    — "road" + "pavement" on a single patch of tarmac, or "person" +
    "rider" on someone on a bike. Per-class NMS doesn't help (different
    classes) and `_drop_contained` only fires when one box is mostly
    inside another (same-class). This pass catches the leftover case:
    boxes whose IoU is high enough that they're clearly the same
    physical thing, just labelled twice.

    Threshold of 0.7 is the sweet spot for tight stacked-label cleanup
    without clipping legitimate co-located objects: helmets on heads
    typically peak ~0.4–0.5 IoU, potholes on roads ~0.2–0.3, person
    inside a car frame ~0.4–0.6 — all safe. Two detectors landing on
    the same physical thing run 0.8+ and get suppressed.
    """
    n = len(boxes)
    if n < 2:
        return boxes, scores, labels
    order = torch.argsort(scores, descending=True).tolist()
    keep_mask = [True] * n
    areas = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).clamp(min=1.0)

    for ii in range(n):
        i = order[ii]
        if not keep_mask[i]:
            continue
        for jj in range(ii + 1, n):
            j = order[jj]
            if not keep_mask[j]:
                continue
            if labels[i] == labels[j]:
                continue
            x0 = max(boxes[i, 0].item(), boxes[j, 0].item())
            y0 = max(boxes[i, 1].item(), boxes[j, 1].item())
            x1 = min(boxes[i, 2].item(), boxes[j, 2].item())
            y1 = min(boxes[i, 3].item(), boxes[j, 3].item())
            inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
            if inter <= 0:
                continue
            union = areas[i].item() + areas[j].item() - inter
            iou = inter / union if union > 0 else 0.0
            if iou >= iou_thr:
                # j is the lower-scoring one (descending sort) — drop.
                keep_mask[j] = False

    keep_idx = [k for k, m in enumerate(keep_mask) if m]
    keep_t = torch.tensor(keep_idx, dtype=torch.long)
    return boxes[keep_t], scores[keep_t], [labels[k] for k in keep_idx]


def draw(image_pil, boxes_xyxy_pixels, phrases):
    """Annotate a PIL image with pixel-space xyxy boxes."""
    draw = ImageDraw.Draw(image_pil)
    font = ImageFont.load_default()
    boxes_t = boxes_xyxy_pixels if isinstance(boxes_xyxy_pixels, torch.Tensor) else torch.as_tensor(boxes_xyxy_pixels)
    for box, label in zip(boxes_t, phrases):
        x0, y0, x1, y1 = (int(v) for v in box.tolist())
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        draw.rectangle([x0, y0, x1, y1], outline=color, width=4)
        bbox = draw.textbbox((x0, y0), label, font=font)
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), label, fill="white", font=font)
    return image_pil


PREVIEW_MAX_SIDE = 480   # longest edge of the baked preview JPEG
PREVIEW_QUALITY = 72     # JPEG quality — tuned for small file size
PREVIEW_DIM_FACTOR = 0.6  # 0 = pitch black background, 1 = no dim


def draw_preview(image_pil, mask_payloads):
    """Render a thumbnail-quality preview: image downscaled to PREVIEW_MAX_SIDE
    on its longest edge. Pixels inside the segmentation polygons keep their
    original colour; everything outside is darkened so the segmented subjects
    pop. No boxes, no labels. Returns an RGB PIL image (~15-40 KB JPEG).

    `mask_payloads` is a list of {"polygons": [[[x,y], ...], ...]} dicts.
    Entries that are None or have no polygons are skipped.

    Images with no masks at all return the downsized original unmodified —
    we don't dim the whole frame just because nothing's been segmented yet.
    """
    W, H = image_pil.size
    longest = max(W, H)
    scale = PREVIEW_MAX_SIDE / longest if longest > PREVIEW_MAX_SIDE else 1.0
    if scale < 1.0:
        small = image_pil.resize((int(W * scale), int(H * scale)), Image.LANCZOS)
    else:
        small = image_pil.copy()
    base = small.convert("RGB")

    # Build a single-channel mask of the union of all segmentation polygons.
    # Cache the scaled polygons so we can stroke their outlines later
    # without re-walking the input.
    mask = Image.new("L", base.size, 0)
    drw = ImageDraw.Draw(mask)
    scaled_polygons: list[list[tuple[int, int]]] = []
    for payload in mask_payloads:
        if not payload:
            continue
        for polygon in payload.get("polygons") or []:
            if len(polygon) < 3:
                continue
            pts = [(int(p[0] * scale), int(p[1] * scale)) for p in polygon]
            drw.polygon(pts, fill=255)
            scaled_polygons.append(pts)
    if not scaled_polygons:
        return base

    # Darkened copy: blend toward black. Using Image.blend keeps colour
    # information instead of a flat overlay, so the dimmed background still
    # has subtle texture cues (sky → dark blue rather than pure black).
    black = Image.new("RGB", base.size, (0, 0, 0))
    dimmed = Image.blend(base, black, 1.0 - PREVIEW_DIM_FACTOR)

    # Composite: where mask is 255 (inside polygons) keep `base`, where 0
    # (background) use the dimmed version. Soft-edged blur on the mask
    # avoids a harsh stairstep along the polygon boundary.
    mask = mask.filter(ImageFilter.GaussianBlur(radius=1.0))
    result = Image.composite(base, dimmed, mask)

    # White outline along each polygon so the cutout edge reads cleanly
    # against any background. Stroke width scales with the rendered preview
    # — 1px on tiny crops, 2px on larger ones.
    outline_w = max(1, min(base.size) // 240)
    drw_out = ImageDraw.Draw(result)
    for pts in scaled_polygons:
        drw_out.line(pts + [pts[0]], fill=(255, 255, 255), width=outline_w, joint="curve")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", "-i", required=True)
    parser.add_argument("--prompt", "-t", required=True, help="comma- or period-separated tags, e.g. 'pothole, traffic cone'")
    parser.add_argument("--output", "-o", default="output.jpg")
    parser.add_argument("--checkpoint", "-p", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--box-threshold", type=float, default=0.3)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    image_pil, _ = load_image(args.image)
    bundle = load_model(DEFAULT_CONFIG, args.checkpoint, args.device)
    boxes, phrases = predict(
        bundle, image_pil, args.prompt,
        args.box_threshold, args.text_threshold, args.device, nms_iou=args.nms_iou,
    )
    print(f"Found {len(boxes)} detection(s):")
    for p in phrases:
        print(f"  - {p}")
    draw(image_pil.copy(), boxes, phrases).save(args.output)
    print(f"Saved annotated image to {args.output}")


if __name__ == "__main__":
    main()
