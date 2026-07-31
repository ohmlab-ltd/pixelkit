"""SAM2 segmentation pass over MM-GD-L boxes.

Lazy-loaded the first time `segment_boxes` is called so the FastAPI server
doesn't block on SAM2 weights at startup. Returns one polygon set per box —
each polygon is a list of (x, y) pixel pairs, which the frontend can render
directly as an SVG `<polygon>` overlay.
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = ROOT / "weights"

# Hiera-Large: highest accuracy SAM2.1 variant. ~224 M params, ~900 MB.
SAM2_CHECKPOINT = WEIGHTS_DIR / "sam2.1_hiera_large.pt"
SAM2_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

# Approximate the mask boundary down to this pixel tolerance — keeps polygons
# light enough to ship over JSON without losing meaningful shape detail.
POLY_EPSILON = 1.5
# Drop tiny noise contours; below this fraction of the box area they're not
# real features.
MIN_CONTOUR_AREA_FRAC = 0.005
# Gaussian blur sigma applied to the binary mask before contour extraction.
# Rounds off the single-pixel staircase that SAM2's mask-decoder emits, so the
# resulting polygon traces a smooth curve instead of a jagged outline. Kept
# small enough that the boundary shifts at most ~1 px from the raw mask.
# Tuned down again — at 2.6 the smoothing was still pulling fine
# protrusions inward; 2.0 keeps the de-jaggying without over-rounding.
SMOOTH_SIGMA = 2.0
# Chaikin corner-cutting passes applied to each contour after the
# approxPolyDP simplification step. Each pass doubles the vertex
# count; one pass already softens visible jaggies, two looks almost
# spline-smooth without blowing the polygon size up too much.
CHAIKIN_ITERATIONS = 2


def _ensure_weights() -> Path:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    if SAM2_CHECKPOINT.exists():
        return SAM2_CHECKPOINT
    tmp = SAM2_CHECKPOINT.with_suffix(".pt.tmp")
    print(f"[segmentation] downloading SAM2 weights to {SAM2_CHECKPOINT}...")
    urllib.request.urlretrieve(SAM2_CHECKPOINT_URL, tmp)
    tmp.rename(SAM2_CHECKPOINT)
    return SAM2_CHECKPOINT


def _build_predictor(device: str):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    ckpt = _ensure_weights()
    model = build_sam2(SAM2_CONFIG, str(ckpt), device=device)
    return SAM2ImagePredictor(model)


def _chaikin(poly: list[list[float]], iterations: int) -> list[list[float]]:
    """Chaikin corner-cutting smoothing for closed polygons.

    Each iteration replaces every vertex with two new vertices ¼ and
    ¾ of the way along each adjacent edge. Sharp corners round off
    visibly after the first pass; two passes yield a near-spline
    feel. Vertex count doubles per pass so we cap at the caller's
    `iterations` (typically 1 or 2).
    """
    pts = poly
    for _ in range(max(0, iterations)):
        n = len(pts)
        if n < 3:
            return pts
        out: list[list[float]] = []
        for i in range(n):
            px, py = pts[i]
            qx, qy = pts[(i + 1) % n]
            out.append([px * 0.75 + qx * 0.25, py * 0.75 + qy * 0.25])
            out.append([px * 0.25 + qx * 0.75, py * 0.25 + qy * 0.75])
        pts = out
    return pts


def _mask_to_polygons(
    mask: np.ndarray,
    box_area: float,
) -> list[list[list[float]]]:
    """Binary mask → list of polygons (each a flat list of [[x,y], ...]).

    Pipeline: morphological close → Gaussian smooth → threshold →
    contour extraction → simplification + Chaikin corner-cutting.
    """
    mask_u8 = (mask.astype(np.uint8)) * 255
    h, w = mask_u8.shape[:2]

    # Stage 1: morphological close to fill pinholes / one-pixel
    # connectivity gaps that fragment SAM2 output.
    closing_k = max(3, int(round(min(h, w) * 0.01)))
    if closing_k % 2 == 0:
        closing_k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_k, closing_k))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)

    # Stage 2: Gaussian blur + threshold for the smooth boundary.
    blur_k = max(3, int(2 * round(3 * SMOOTH_SIGMA) + 1))
    mask_u8 = cv2.GaussianBlur(mask_u8, (blur_k, blur_k), SMOOTH_SIGMA)
    _, mask_u8 = cv2.threshold(mask_u8, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys: list[list[list[float]]] = []
    min_area = max(8.0, box_area * MIN_CONTOUR_AREA_FRAC)

    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        approx = cv2.approxPolyDP(c, POLY_EPSILON, True)
        if len(approx) < 3:
            continue
        pts = [[float(p[0][0]), float(p[0][1])] for p in approx]
        polys.append(_chaikin(pts, CHAIKIN_ITERATIONS))
    return polys


def segment_point(state: dict, image_pil: Image.Image, point_xy: list[float]) -> dict | None:
    """SAM2 click-to-segment. Returns {polygons, box_xyxy} for the highest-
    scoring mask among SAM2's 3 multimask candidates — picking the best of
    three lets us recover from ambiguous "is this the wheel or the whole
    car?" clicks. Returns None if no usable contour was found."""
    predictor = state.get("segmenter")
    if predictor is None:
        predictor = _build_predictor(state.get("device", "cpu"))
        state["segmenter"] = predictor

    img_rgb = np.array(image_pil)
    predictor.set_image(img_rgb)

    point_coords = np.array([[float(point_xy[0]), float(point_xy[1])]], dtype=np.float32)
    point_labels = np.array([1], dtype=np.int32)  # 1 = foreground
    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=None,
        multimask_output=True,
    )
    # masks shape: (3, H, W) — three candidates at increasing scale.
    # Pick SAM's highest-confidence mask: tighter and more conservative
    # than the largest-area pick, so click-to-detect doesn't over-segment
    # into background. The user can still hand-edit the mask via the
    # painter for occluded cases.
    if masks.ndim == 4:
        masks = masks[0]
    best_idx = int(np.argmax(scores))
    mask = masks[best_idx] > 0
    if not mask.any():
        return None

    ys, xs = np.where(mask)
    x0, y0 = float(xs.min()), float(ys.min())
    x1, y1 = float(xs.max()) + 1.0, float(ys.max()) + 1.0
    box_area = max(1.0, (x1 - x0) * (y1 - y0))
    polys = _mask_to_polygons(mask, box_area)
    if not polys:
        return None
    return {
        "polygons": polys,
        "box_xyxy": [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)],
        "score": float(scores[best_idx]),
    }


def segment_boxes(state: dict, image_pil: Image.Image, boxes_xyxy: list[list[float]]) -> list[dict | None]:
    """Run SAM2 once per image and return one mask payload per input box.

    Payload shape: {"polygons": [[[x,y], ...], ...]} — multiple polygons cover
    the case where SAM2 returns disconnected components (e.g. occluded
    objects). Returns None for boxes where no usable mask was found.
    """
    if not boxes_xyxy:
        return []

    predictor = state.get("segmenter")
    if predictor is None:
        predictor = _build_predictor(state.get("device", "cpu"))
        state["segmenter"] = predictor

    img_rgb = np.array(image_pil)
    predictor.set_image(img_rgb)

    box_arr = np.array(boxes_xyxy, dtype=np.float32)
    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=box_arr,
        multimask_output=False,
    )
    # `masks` shape: (N, 1, H, W) when N>1 boxes are passed, else (1, H, W).
    if masks.ndim == 4:
        masks = masks[:, 0]
    elif masks.ndim == 3 and len(boxes_xyxy) == 1:
        masks = masks[:1]

    payloads: list[dict | None] = []
    for box, mask in zip(boxes_xyxy, masks):
        x0, y0, x1, y1 = box
        box_area = max(1.0, (x1 - x0) * (y1 - y0))
        polys = _mask_to_polygons(mask > 0, box_area)
        payloads.append({"polygons": polys} if polys else None)
    return payloads


# ── Windowed segmentation for small boxes on large images ────────────
# set_image squashes the whole frame into SAM2's fixed 1024×1024 encoder
# input, so a 27px animal on a 4K frame is ~7px in encoder space and
# ~1.8px in the 256² mask-decoder grid — masks come back empty or as
# blobs. Cropping a native-resolution window around small boxes keeps
# them at full pixel size through the encoder. Bonus: upsampling mask
# logits to 1024² windows instead of the full 8.3MP frame LOWERS peak
# VRAM despite the extra encoder passes.

# Boxes smaller than this on their shortest side get routed to a native
# window; larger boxes segment fine through the full-frame pass.
SMALL_BOX_MIN_SIDE = float(os.environ.get("SAM2_SMALL_BOX_MIN_SIDE", "64"))
# Native window edge + context padding kept between a member box and the
# window border (so SAM2 sees surroundings, not a box flush to the crop).
SEG_WINDOW = int(os.environ.get("SAM2_SEG_WINDOW", "1024"))
SEG_WINDOW_PAD = 32


def segment_boxes_windowed(
    state: dict,
    image_pil: Image.Image,
    boxes_xyxy: list[list[float]],
    *,
    window: int | None = None,
    small_min_side: float | None = None,
    cancel_check=None,
) -> list[dict | None]:
    """Drop-in variant of `segment_boxes` for large images: small boxes are
    segmented inside native-resolution windows (greedy clustering, one
    set_image + one batched predict per window, polygons offset back to
    global coords); everything else takes the normal full-image pass.
    Output is positionally aligned with `boxes_xyxy`, exactly like
    `segment_boxes`. Behaviour-identical to `segment_boxes` when the image
    is small or no box qualifies as small."""
    n = len(boxes_xyxy)
    if n == 0:
        return []
    win = int(window or SEG_WINDOW)
    small_side = float(small_min_side if small_min_side is not None else SMALL_BOX_MIN_SIDE)
    W, H = image_pil.size
    # Image already near SAM2's native input scale — plain pass.
    if min(W, H) <= win:
        return segment_boxes(state, image_pil, boxes_xyxy)

    # Clamp incoming boxes; tiled-GD merges can leave a vertex a hair
    # out of bounds, and window placement math assumes in-image boxes.
    clamped: list[list[float]] = []
    for b in boxes_xyxy:
        x0 = min(max(float(b[0]), 0.0), float(W))
        y0 = min(max(float(b[1]), 0.0), float(H))
        x1 = min(max(float(b[2]), 0.0), float(W))
        y1 = min(max(float(b[3]), 0.0), float(H))
        clamped.append([x0, y0, x1, y1])

    interior = win - 2 * SEG_WINDOW_PAD
    small_idx = [
        i for i, b in enumerate(clamped)
        if min(b[2] - b[0], b[3] - b[1]) < small_side
        and (b[2] - b[0]) <= interior and (b[3] - b[1]) <= interior
    ]
    small_set = set(small_idx)
    full_pass_idx = [i for i in range(n) if i not in small_set]

    payloads: list[dict | None] = [None] * n

    # Greedy clustering: centre a window on the first unassigned small
    # box, absorb every small box fully inside its pad-inset interior.
    # Pad requirements relax at image edges (there's no context beyond
    # the frame, so a flush fit is correct there, not a truncation).
    unassigned = list(small_idx)
    windows: list[tuple[int, int, list[int]]] = []
    while unassigned:
        seed = unassigned[0]
        b = clamped[seed]
        cx = (b[0] + b[2]) / 2.0
        cy = (b[1] + b[3]) / 2.0
        wx0 = int(round(min(max(cx - win / 2.0, 0.0), W - win)))
        wy0 = int(round(min(max(cy - win / 2.0, 0.0), H - win)))
        pl = SEG_WINDOW_PAD if wx0 > 0 else 0
        pt = SEG_WINDOW_PAD if wy0 > 0 else 0
        pr = SEG_WINDOW_PAD if wx0 + win < W else 0
        pb = SEG_WINDOW_PAD if wy0 + win < H else 0
        members: list[int] = []
        for i in list(unassigned):
            bb = clamped[i]
            if (
                bb[0] >= wx0 + pl and bb[1] >= wy0 + pt
                and bb[2] <= wx0 + win - pr and bb[3] <= wy0 + win - pb
            ):
                members.append(i)
                unassigned.remove(i)
        if not members:
            # Couldn't place the seed even in its own centred window —
            # send it through the full-image pass with the large boxes.
            unassigned.remove(seed)
            full_pass_idx.append(seed)
            continue
        windows.append((wx0, wy0, members))

    for wx0, wy0, members in windows:
        if cancel_check is not None and cancel_check():
            # Return immediately — falling through would still run the
            # full-image pass below, i.e. exactly the expensive set_image
            # the cancellation was meant to skip.
            return payloads
        crop = image_pil.crop((wx0, wy0, wx0 + win, wy0 + win))
        local = [
            [clamped[i][0] - wx0, clamped[i][1] - wy0, clamped[i][2] - wx0, clamped[i][3] - wy0]
            for i in members
        ]
        try:
            local_payloads = segment_boxes(state, crop, local)
        except Exception as e:
            print(f"[segmentation] window ({wx0},{wy0}) failed: {e}")
            continue
        for i, p in zip(members, local_payloads):
            if not p or not p.get("polygons"):
                continue
            # Pure translation — the window is a crop, not a resize, so
            # polygons shift by the window origin and nothing scales.
            payloads[i] = {
                "polygons": [
                    [[x + wx0, y + wy0] for x, y in poly]
                    for poly in p["polygons"]
                ]
            }

    # One combined full-image pass for the large boxes (+ any fallbacks),
    # AFTER the windows so it costs a single set_image.
    if full_pass_idx and not (cancel_check is not None and cancel_check()):
        try:
            full = segment_boxes(state, image_pil, [clamped[i] for i in full_pass_idx])
            for i, p in zip(full_pass_idx, full):
                payloads[i] = p
        except Exception as e:
            print(f"[segmentation] full-image pass failed: {e}")

    return payloads
