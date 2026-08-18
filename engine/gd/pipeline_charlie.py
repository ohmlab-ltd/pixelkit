"""Pipeline Charlie - clean restart of the labelling pipeline.

Charlie is a from-scratch alternative to V2 (GroundingDINO + SAM2 +
Qwen-VL + DINOv2 + SigLIP). The current step-1 implementation runs
**only SAM3** for promptable concept segmentation: image + text label
in, instance boxes + masks + scores out.

Per the SAM3 model card the canonical call shape is one text prompt
per forward pass:

    inputs = processor(images=image, text="ear", return_tensors="pt").to(dev)
    outputs = model(**inputs)
    results = processor.post_process_instance_segmentation(
        outputs, threshold=0.5, mask_threshold=0.5,
        target_sizes=inputs["original_sizes"].tolist(),
    )[0]

For multi-label requests we loop over the labels and concatenate the
per-label detections. SAM3 image encoding is heavy (~half the cost
per call), so a future optimisation can run `model.get_image_embeddings`
once and reuse the cached embeddings across labels - leaving that for
later iterations.

Future iterations of Charlie will likely add:
- VLM step for label disambiguation (drop-in once VRAM budget allows)
- Embeddings (DINOv2 / SigLIP) for reference-centroid resolution
- Reference-image processing flow

Future pipelines (Delta, Echo, …) will live alongside this one and
will replace SAM3 with whatever's next.

Public surface:
    load_sam3(device)               -> (model, processor)
    is_loaded()                     -> bool
    clear_sam3()                    -> None
    segment_labels(image, labels)   -> (detections, timings_ms)
    segment_point(image, [x, y])    -> (detection_or_none, timings_ms)

Gating: facebook/sam3 is gated on Hugging Face. Set HF_TOKEN in .env
(or the environment) before the first load - transformers picks it
up automatically. Override with SAM3_MODEL_ID for community
repackages or alternate checkpoints.
"""

from __future__ import annotations

import base64
import io
import os
import time
from typing import Iterable

import torch
from PIL import Image

SAM3_MODEL_ID = os.environ.get("SAM3_MODEL_ID", "facebook/sam3")

# ─────────────────────────────────────────────────────────────────────
# LOCKED VALUES - DO NOT CHANGE THESE DEFAULTS WITHOUT EXPLICIT USER
# REQUEST.  They were tuned together against real PPE / construction
# images.  Each individual env var below can still be overridden via
# .env if a specific workflow needs it, but the source-tree defaults
# are stable and any change to them needs to be asked for first.
# ─────────────────────────────────────────────────────────────────────

# Model-level confidence cuts. Kept at SAM3's defaults so the model
# returns its full set of plausible detections - noise rejection is
# handled in post-processing rather than at the model boundary, so we
# don't accidentally drop legit small / lower-confidence objects.
SAM3_THRESHOLD = float(os.environ.get("SAM3_THRESHOLD", "0.5"))           # LOCKED
SAM3_MASK_THRESHOLD = float(os.environ.get("SAM3_MASK_THRESHOLD", "0.5")) # LOCKED
# Interactive paths (click-to-detect, add-box) use lower thresholds
# than the batch labeller so the user can find objects the auto-run
# missed. The strict batch thresholds guard against false positives
# across a whole dataset; for a single click the user is the final
# arbiter, so being more permissive is the right trade-off.
SAM3_INTERACTIVE_THRESHOLD = float(os.environ.get("SAM3_INTERACTIVE_THRESHOLD", "0.05"))
SAM3_INTERACTIVE_MASK_THRESHOLD = float(os.environ.get("SAM3_INTERACTIVE_MASK_THRESHOLD", "0.2"))

# Per-image absolute floor: drop anything whose mask covers less than
# max(SAM3_MIN_AREA_PX, SAM3_MIN_AREA_FRAC × image area) pixels.
# Loose (0.01 % image-relative + 64 px absolute) so genuinely small
# accessories like gloves and hard hats survive; relative filters do
# the bulk of noise removal, this one only catches pixel slivers.
SAM3_MIN_AREA_FRAC = float(os.environ.get("SAM3_MIN_AREA_FRAC", "0.0001"))  # LOCKED - 0.01 %
SAM3_MIN_AREA_PX = int(os.environ.get("SAM3_MIN_AREA_PX", "64"))            # LOCKED

# Within a single label's hits, drop anything whose mask area is below
# this fraction of that label's largest mask. 5 % keeps partial /
# occluded siblings while still killing clear same-label noise.
SAM3_MIN_RELATIVE_AREA = float(os.environ.get("SAM3_MIN_RELATIVE_AREA", "0.05"))  # LOCKED - 5 %

# Cross-label area filter: drop anything whose mask is below this
# fraction of the IMAGE-WIDE largest detection (across all labels).
# DISABLED BY DEFAULT - dominant subjects (a person filling the
# frame) make legit small accessories look tiny relative to them, so
# this filter misclassifies real PPE. Re-enable in .env per workflow
# only if needed.
SAM3_MIN_GLOBAL_RELATIVE_AREA = float(os.environ.get("SAM3_MIN_GLOBAL_RELATIVE_AREA", "0"))  # LOCKED

# Resize incoming images so SAM3 always runs at the same scale
# (longest edge = SAM3_TARGET_LONGEST_EDGE px). Keeps mask resolution
# AND polygons consistent across heterogeneous inputs - a 600×400
# photo and a 4000×3000 photo yield masks with the same vertex
# density, so jaggy edges on small uploads disappear. 0 disables.
SAM3_TARGET_LONGEST_EDGE = int(os.environ.get("SAM3_TARGET_LONGEST_EDGE", "1500"))  # LOCKED
# Interactive tools (click-to-detect / box-to-mask) trade a little mask
# resolution for latency: the vision encoder is ~O(pixels), so 1008 px is
# roughly 2.2x faster than 1500 px per first click on an image. Batch
# labelling keeps the full-resolution target above.
SAM3_INTERACTIVE_LONGEST_EDGE = int(os.environ.get("SAM3_INTERACTIVE_LONGEST_EDGE", "1008"))

# Douglas-Peucker simplification factor for output polygons, as a
# fraction of each contour's perimeter. 0.0015 (0.15 %) collapses
# pixel-step staircases without eating real corners. 0 disables.
SAM3_POLY_SIMPLIFY_EPS = float(os.environ.get("SAM3_POLY_SIMPLIFY_EPS", "0.0015"))  # LOCKED

# Sub-mask outlier filter (post-processing, runs inside
# _mask_to_polygons): within a single detection's polygon set, drop
# sub-polygons whose area is below this fraction of the largest sub-
# polygon. Targets the case where SAM3 returns one good mask plus a
# few stray pixel blobs disconnected from the real object. Set
# generously (5 %) so only clear outliers fall - partial / cropped
# pieces of the same object stay.
SAM3_SUBMASK_MIN_RELATIVE_AREA = float(os.environ.get("SAM3_SUBMASK_MIN_RELATIVE_AREA", "0.05"))

# Pre-contour morphological opening. Erodes then dilates the binary
# mask with a kernel of this size, removing features narrower than
# the kernel width. SAM3 occasionally emits a thin pixel bridge
# between two separate blobs of the same concept (e.g. two gloves
# linked by a 1–3 px line through the body) - cv2.findContours then
# returns ONE contour walking both regions through the bridge, and
# the resample pass collapses the bridge into a single straight
# segment that visually crosses unrelated parts of the image. A 5×5
# open kernel breaks bridges ≤5 px wide while leaving real glove /
# hard hat scale features untouched. 0 disables.
SAM3_MASK_OPEN_PX = int(os.environ.get("SAM3_MASK_OPEN_PX", "5"))

# Pre-contour mask smoothing. Applies a Gaussian blur to the binary
# mask, then re-thresholds at 0.5, before contour extraction. Knocks
# back the pixel-level staircase that produces jaggy polygon edges
# without eating real corners. The radius is in mask pixels at the
# inference resolution (1500 px on longest); 2 gives a 5×5 kernel
# with σ=2, which softens edges by ≈1 px on each side. Set to 0 to
# disable smoothing entirely.
SAM3_MASK_SMOOTH_PX = int(os.environ.get("SAM3_MASK_SMOOTH_PX", "0"))

# Catmull-Rom polygon subdivision (post-processing, additive only).
# After polygon extraction + simplification + outlier removal, run a
# Catmull-Rom spline through every kept vertex and emit
# SAM3_POLY_SUBDIVIDE_SAMPLES new points between each consecutive
# pair. Original vertices are NEVER removed - the curve passes
# through them exactly. Result: a polygon with (samples+1)× as many
# vertices, smooth between corners. Set to 0 to disable.
SAM3_POLY_SUBDIVIDE_SAMPLES = int(os.environ.get("SAM3_POLY_SUBDIVIDE_SAMPLES", "0"))

# Gaussian smoothing on polygon vertices, applied AFTER subdivision.
# Doesn't change vertex count - each vertex gets replaced with a
# Gaussian-weighted mean of itself + its neighbours along the closed
# loop. Smooths the wiggles that Catmull-Rom inevitably traces
# through every original control point. Sigma is in vertex-index
# units, not pixels: σ=6 means the kernel covers ≈12 vertices on
# each side (≈24 on a subdivided polygon, since subdivision blew
# the count up 21×). Set to 0 to disable.
SAM3_POLY_SMOOTH_SIGMA = float(os.environ.get("SAM3_POLY_SMOOTH_SIGMA", "0"))
# Number of Gaussian-smoothing passes. Each pass with sigma σ
# composes to an effective σ_total = σ × √passes (Gaussians
# multiply additively in variance), so 3 × σ=6 ≈ σ_eff=10.4 - visibly
# round curves with no remaining staircase.
SAM3_POLY_SMOOTH_PASSES = int(os.environ.get("SAM3_POLY_SMOOTH_PASSES", "3"))

# Periodic cubic B-spline approximation, applied AFTER Gaussian
# smoothing as the final step in _mask_to_polygons. Unlike Catmull-
# Rom (which interpolates - passes through every control point), a
# B-spline approximates - the curve is pulled toward control points
# but isn't required to touch them. Result: C2 continuity (smooth
# second derivatives, no kinks in curvature) and any residual
# staircase in the control polygon is washed out completely.
# samples_per_segment controls the density of the output curve. Set
# to 0 to disable the spline pass entirely.
SAM3_POLY_SPLINE_SAMPLES = int(os.environ.get("SAM3_POLY_SPLINE_SAMPLES", "0"))

# Final resample step. After all the smoothing the polygon has
# thousands of vertices clustered in tiny segments - when the FE
# draws cubic Bezier curves between every consecutive pair, each
# segment is so short that the aggregate looks like a polyline
# (visible "bumps" at high zoom). Resample to a fixed-perimeter-
# density vertex count: every ~SAM3_POLY_RESAMPLE_PX_SPACING px of
# perimeter gets one control point, with min/max caps. Wide-spaced
# control points + Bezier between them = real smooth curves on the
# FE. Set to 0 to disable (keep dense smoothed output).
SAM3_POLY_RESAMPLE_PX_SPACING = float(os.environ.get("SAM3_POLY_RESAMPLE_PX_SPACING", "0"))
SAM3_POLY_RESAMPLE_MIN = int(os.environ.get("SAM3_POLY_RESAMPLE_MIN", "32"))
SAM3_POLY_RESAMPLE_MAX = int(os.environ.get("SAM3_POLY_RESAMPLE_MAX", "240"))

# Sharp-corner preservation. Detected on the simplified polygon
# (post-DP) by turning angle averaged over a small window of
# vertices on each side, then non-maximum suppressed so clusters of
# noisy vertices don't all get flagged. Corners with a turn ≥
# threshold survive the smoothing pipeline as exact angular vertices.
# Set threshold to 0 to disable preservation entirely.
# 40° catches typical real-world corners (sign edges, box edges,
# bent-pipe joints, the angle where a roof meets a wall) while still
# rejecting soft mask-noise turns.
SAM3_CORNER_ANGLE_THRESHOLD_DEG = float(os.environ.get("SAM3_CORNER_ANGLE_THRESHOLD_DEG", "0"))
# How many vertices on each side to average for the turn-angle
# calculation. Larger window = robust against contour pixel noise;
# too large washes out real corners. Capped per-polygon at n//6 to
# stop the window wrapping to opposite-side vertices on small polys.
SAM3_CORNER_WINDOW = int(os.environ.get("SAM3_CORNER_WINDOW", "4"))
# Non-maximum suppression radius (in vertex-index units). Within ±r
# vertices of a candidate corner, only the SHARPEST corner survives.
# Stops a noisy run of vertices from registering as multiple corners
# in a row, but kept tight so two genuinely adjacent corners (e.g.
# a hexagon's neighbouring vertices) don't suppress each other.
SAM3_CORNER_NMS_RADIUS = int(os.environ.get("SAM3_CORNER_NMS_RADIUS", "2"))
# Polygons with fewer than this many vertices skip corner detection
# entirely - there's not enough room for the window + NMS to give a
# reliable result, and tiny polygons usually round acceptably anyway.
SAM3_CORNER_MIN_VERTICES = int(os.environ.get("SAM3_CORNER_MIN_VERTICES", "8"))
# Maximum distance (in original-image px) a corner is allowed to be
# from the smoothed polygon and still get re-anchored. If smoothing
# pulled the polygon further away than this, the corner is "lost" -
# we leave the smoothed shape alone instead of inserting a thin spike
# that points to where the corner used to be (the visible artefact in
# the pothole/glove screenshots).
SAM3_CORNER_MAX_DIST_PX = float(os.environ.get("SAM3_CORNER_MAX_DIST_PX", "12"))
# When a corner IS close enough to anchor, carve a small wedge into
# the smoothed polygon instead of inserting a thin spike: every
# vertex within carve-radius of the corner is removed and replaced
# with the corner + duplicate, producing a real sharp angle in place
# of the rounded arc the smoothing produced.
SAM3_CORNER_CARVE_RADIUS_PX = float(os.environ.get("SAM3_CORNER_CARVE_RADIUS_PX", "10"))

# ─────────────────────────────────────────────────────────────────────
# Context-aware smoothing - replaces the multi-step
# subdivide/Gaussian/B-spline/resample/corner-detect/anchor stack
# above with a single variable-σ Gaussian. Per-vertex sigma is
# inversely proportional to local curvature, so sharp corners
# (high curvature) get σ→0 and don't move, while smooth-section
# vertices get full σ and average out into a clean curve. There's
# no binary "is it a corner" decision - sigma is a continuous
# gradient, so soft real corners no longer vanish and noisy bumps
# no longer turn into phantom corners.
# ─────────────────────────────────────────────────────────────────────

# Subdivide each polygon by this many samples per segment via Catmull-
# Rom BEFORE the context-aware smoothing pass, so the variable-σ
# Gaussian has enough vertex density to actually smooth. 0 disables
# (smoothing runs on the post-DP polygon directly).
SAM3_CONTEXT_SUBDIVIDE = int(os.environ.get("SAM3_CONTEXT_SUBDIVIDE", "0"))
# Maximum smoothing sigma (vertex-index units). Applied at vertices
# with zero local curvature; falls off linearly to 0 at the corner
# threshold. 0 disables context-aware smoothing entirely.
SAM3_CONTEXT_SMOOTH_MAX_SIGMA = float(os.environ.get("SAM3_CONTEXT_SMOOTH_MAX_SIGMA", "0"))
# FLOOR for the corner threshold (degrees) at which sigma reaches 0.
# Effective threshold per polygon is max(this, avg_turn × sensitivity)
# so small organic shapes (potholes, gloves) - where every vertex
# legitimately turns 20–40° - get judged on their OWN curvature
# distribution, not against a fixed-degree absolute. A vertex is
# treated as a corner only when its turn substantially exceeds the
# polygon's typical turn.
SAM3_CONTEXT_CORNER_DEG = float(os.environ.get("SAM3_CONTEXT_CORNER_DEG", "60"))
# How many times the polygon's average turn the threshold sits at.
# 3.0 means: a vertex must turn ≥3× the polygon's typical turn to
# count as a corner. Higher = even more smoothing on organic shapes.
SAM3_CONTEXT_CORNER_SENSITIVITY = float(os.environ.get("SAM3_CONTEXT_CORNER_SENSITIVITY", "3.0"))
# After smoothing, vertices whose post-smoothing turn angle still
# exceeds this many degrees get a duplicate vertex inserted right
# after them. The duplicate makes the FE's Catmull-Rom-to-Bezier
# produce a real cusp at that vertex instead of a rounded transition.
# Set to 0 to disable corner cusp marking.
SAM3_CONTEXT_DUPE_DEG = float(os.environ.get("SAM3_CONTEXT_DUPE_DEG", "0"))


_MODEL = None
_PROCESSOR = None


def _device_sync() -> None:
    """Barrier for accurate timings on whichever backend is active."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        torch.mps.synchronize()


def _device_empty_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def is_loaded() -> bool:
    return _MODEL is not None and _PROCESSOR is not None


def load_sam3(device: str = "cuda"):
    """Load SAM3 onto the given device. Populates module-level handles
    and returns (model, processor).

    Devices: cuda and mps (Apple Metal) load fp16; cpu loads fp32 and is
    explicitly opt-in upstream (the SAM3 vision transformer runs minutes
    per image on CPU - the server only requests it when the user forced
    PK_DEVICE=cpu)."""
    global _MODEL, _PROCESSOR

    dev = str(device)
    if not (dev in ("mps", "cpu") or dev == "cuda" or dev.startswith("cuda")):
        raise RuntimeError(f"SAM3: unsupported device {device!r} (cuda | mps | cpu)")

    from transformers import Sam3Model, Sam3Processor

    dtype = torch.float32 if dev == "cpu" else torch.float16
    print(f"[charlie] loading {SAM3_MODEL_ID} on {device} ({dtype})...")
    if dev == "cpu":
        print("[charlie] WARNING: CPU inference is extremely slow - expect minutes per image.")
    proc = Sam3Processor.from_pretrained(SAM3_MODEL_ID)
    mdl = Sam3Model.from_pretrained(SAM3_MODEL_ID, torch_dtype=dtype)
    mdl = mdl.to(device).eval()

    _MODEL = mdl
    _PROCESSOR = proc
    print(f"[charlie] {SAM3_MODEL_ID} ready on {device}.")
    return mdl, proc


def clear_sam3() -> None:
    """Drop module references so VRAM can be reclaimed."""
    global _MODEL, _PROCESSOR
    _MODEL = None
    _PROCESSOR = None
    _VISION_CACHE.clear()


def _detect_sharp_corners(
    polygon,
    threshold_deg: float,
    window: int = 3,
    nms_radius: int = 2,
    min_vertices: int = 8,
) -> list[tuple[float, float]]:
    """Detect vertices where the polygon takes a real sharp turn.

    MULTI-SCALE detection. For each vertex, the turn angle is
    computed at every window size from 1 up to `window`, and the
    SHARPEST result wins. This catches both:

      - Sharp single-vertex corners (90° box edges) - fires at w=1.
      - "Spread" corners where the same total angle is spread across
        2–4 vertices (typical road / soft panel boundaries) - fires
        at w=3 or w=4.

    Per-window definition:
        v1 = p_i            − p_{i-w}
        v2 = p_{i+w}        − p_i
        cos θ = v1·v2 / (|v1||v2|)

    Each window is capped per-polygon at n//6 so the vectors don't
    wrap to opposite-side vertices on small polygons. Polygons below
    min_vertices skip detection entirely. NMS picks the sharpest
    representative vertex within each cluster."""
    n = len(polygon)
    if n < min_vertices or threshold_deg <= 0:
        return []
    import math
    cos_threshold = math.cos(math.radians(threshold_deg))
    max_window = max(1, min(window, n // 6))
    if n < 2 * max_window + 1:
        return []
    windows = list(range(1, max_window + 1))

    scores = [0.0] * n
    for i in range(n):
        best = 0.0
        for w in windows:
            prev_i = (i - w) % n
            next_i = (i + w) % n
            v1x = float(polygon[i][0]) - float(polygon[prev_i][0])
            v1y = float(polygon[i][1]) - float(polygon[prev_i][1])
            v2x = float(polygon[next_i][0]) - float(polygon[i][0])
            v2y = float(polygon[next_i][1]) - float(polygon[i][1])
            n1 = math.sqrt(v1x * v1x + v1y * v1y)
            n2 = math.sqrt(v2x * v2x + v2y * v2y)
            if n1 < 1e-6 or n2 < 1e-6:
                continue
            dot = (v1x * v2x + v1y * v2y) / (n1 * n2)
            if dot < cos_threshold:
                score = cos_threshold - dot
                if score > best:
                    best = score
        scores[i] = best

    out: list[tuple[float, float]] = []
    for i in range(n):
        if scores[i] <= 0:
            continue
        is_max = True
        for off in range(-nms_radius, nms_radius + 1):
            if off == 0:
                continue
            j = (i + off) % n
            if scores[j] > scores[i]:
                is_max = False
                break
        if is_max:
            out.append((float(polygon[i][0]), float(polygon[i][1])))
    return out


def _dist_to_segment_sq(px: float, py: float, a, b) -> float:
    """Squared distance from (px, py) to the closed segment a–b."""
    ax = float(a[0]); ay = float(a[1])
    bx = float(b[0]); by = float(b[1])
    abx = bx - ax; aby = by - ay
    ab_len_sq = abx * abx + aby * aby
    if ab_len_sq < 1e-9:
        dx = px - ax; dy = py - ay
        return dx * dx + dy * dy
    t = ((px - ax) * abx + (py - ay) * aby) / ab_len_sq
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    cx = ax + t * abx
    cy_p = ay + t * aby
    dx = px - cx; dy = py - cy_p
    return dx * dx + dy * dy


def _anchor_corners(
    polygon,
    corner_positions: list[tuple[float, float]],
    max_dist_px: float = 12.0,
    carve_radius_px: float = 10.0,
):
    """Re-introduce sharp corners after smoothing.

    For each detected corner:

      1. SKIP if the original corner is more than `max_dist_px` from
         the smoothed polygon. Smoothing has pulled the boundary too
         far for the corner to be reintroduced cleanly - inserting
         it anyway would produce a thin spike pointing out of the
         polygon (the artefact visible in the pothole / vest
         screenshots). The corner stays rounded.

      2. CARVE the corner in: remove every smoothed vertex within
         `carve_radius_px` of the corner and replace that contiguous
         run with the corner + duplicate. This swaps the rounded arc
         the smoothing produced for a real sharp angle. The duplicate
         vertex pair makes the FE's Catmull-Rom-to-Bezier collapse
         to colinear tangents on each side, producing a true cusp.

      3. If no smoothed vertices fall inside the carve radius, fall
         back to inserting the corner at its closest segment - the
         smoothed boundary already runs near the corner, so the
         insertion lands flush rather than as a spike.
    """
    if not corner_positions or len(polygon) < 2:
        return polygon
    poly = list(polygon)
    max_dist_sq = max_dist_px * max_dist_px
    carve_radius_sq = carve_radius_px * carve_radius_px

    for cx, cy in corner_positions:
        n = len(poly)
        if n < 2:
            continue

        # Closest segment + closest single vertex distance.
        best_seg = 0
        best_seg_d2 = float("inf")
        for i in range(n):
            j = (i + 1) % n
            d2 = _dist_to_segment_sq(cx, cy, poly[i], poly[j])
            if d2 < best_seg_d2:
                best_seg_d2 = d2
                best_seg = i
        # Bail if the corner is too far from the smoothed shape -
        # don't draw a spike pointing into empty space.
        if best_seg_d2 > max_dist_sq:
            continue

        # Find contiguous run of vertices within carve radius.
        within = []
        for idx, p in enumerate(poly):
            dx = float(p[0]) - cx
            dy = float(p[1]) - cy
            if dx * dx + dy * dy <= carve_radius_sq:
                within.append(idx)

        c = [round(cx, 1), round(cy, 1)]
        if not within:
            # No vertex inside the carve radius - corner is right on
            # the boundary between two vertices. Insert into segment.
            poly.insert(best_seg + 1, c)
            poly.insert(best_seg + 2, c)
            continue

        # Group `within` indices into contiguous runs (handling the
        # closed-loop wrap-around case).
        within_set = set(within)
        # Walk forward from each index; pick the run that contains
        # the closest-segment endpoints if possible, otherwise the
        # longest contiguous run.
        runs: list[list[int]] = []
        seen_idx: set[int] = set()
        for idx in within:
            if idx in seen_idx:
                continue
            run = [idx]
            seen_idx.add(idx)
            # extend forward
            j = (idx + 1) % n
            while j in within_set and j not in seen_idx:
                run.append(j)
                seen_idx.add(j)
                j = (j + 1) % n
            # extend backward
            j = (idx - 1) % n
            while j in within_set and j not in seen_idx:
                run.insert(0, j)
                seen_idx.add(j)
                j = (j - 1) % n
            runs.append(run)
        # Pick the run nearest to the corner (smallest min distance).
        def _min_d2(run: list[int]) -> float:
            md = float("inf")
            for idx in run:
                p = poly[idx]
                dx = float(p[0]) - cx
                dy = float(p[1]) - cy
                d = dx * dx + dy * dy
                if d < md:
                    md = d
            return md
        target_run = min(runs, key=_min_d2)

        # Replace the contiguous run with [corner, corner].
        # Wrap-around runs split list operations into two parts.
        run_set = set(target_run)
        if all(target_run[k] + 1 == target_run[k + 1] for k in range(len(target_run) - 1)):
            # Simple contiguous, no wrap.
            first = target_run[0]
            last = target_run[-1]
            new_poly = poly[:first] + [c, c] + poly[last + 1:]
        else:
            # Wrap-around - easier to filter then prepend the corner pair.
            kept = [poly[i] for i in range(n) if i not in run_set]
            # Place corner pair at the wrap boundary.
            new_poly = [c, c] + kept
        poly = new_poly

    return poly


def _resample_polygon_arc_length(polygon, n_target: int):
    """Resample `polygon` to `n_target` evenly-spaced points along
    its perimeter. Preserves overall shape (crucially, the order of
    samples) while collapsing dense vertex clusters that prevent the
    FE's Bezier renderer from drawing wide smooth arcs.

    Polygons with too few vertices (or a target ≤ current count)
    pass through untouched."""
    n = len(polygon)
    if n < 4 or n_target < 4 or n_target >= n:
        return polygon
    import numpy as np
    pts = np.asarray(polygon, dtype=np.float64)
    closed = np.vstack([pts, pts[0:1]])  # close the loop for arc-length math
    diffs = np.diff(closed, axis=0)
    seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = float(cum[-1])
    if total <= 0:
        return polygon
    # Target arc-length positions, evenly spaced (excluding the
    # endpoint so the loop closes cleanly when re-wrapped).
    targets = np.linspace(0.0, total, n_target, endpoint=False)
    out = []
    for t in targets:
        idx = int(np.searchsorted(cum, t, side="right") - 1)
        idx = max(0, min(n - 1, idx))
        seg_len = max(1e-9, float(seg_lens[idx]))
        seg_t = (t - cum[idx]) / seg_len
        x = closed[idx, 0] + seg_t * diffs[idx, 0]
        y = closed[idx, 1] + seg_t * diffs[idx, 1]
        out.append([round(float(x), 1), round(float(y), 1)])
    return out


def _periodic_cubic_bspline(polygon, samples_per_segment: int):
    """Periodic uniform cubic B-spline through `polygon`'s vertices.

    For each consecutive quadruple (P_{i-1}, P_i, P_{i+1}, P_{i+2})
    the curve is sampled at samples_per_segment evenly-spaced
    parameter values using the standard uniform cubic B-spline basis:

        B_{-1}(t) = (1 − 3t + 3t² − t³)   / 6
        B_{0}(t)  = (4 − 6t² + 3t³)       / 6
        B_{1}(t)  = (1 + 3t + 3t² − 3t³)  / 6
        B_{2}(t)  = t³                    / 6

    Unlike Catmull-Rom this is APPROXIMATING - the curve is pulled
    toward the control points but doesn't pass through them, so any
    residual staircase in the control polygon is washed out entirely.
    The resulting curve has C2 continuity along its full length.

    Indices wrap around so the spline is closed (no seam at the
    "first" vertex). Polygons with fewer than 4 vertices, or
    samples_per_segment ≤ 0, pass through untouched."""
    n = len(polygon)
    if n < 4 or samples_per_segment <= 0:
        return polygon
    out = []
    for i in range(n):
        p0 = polygon[(i - 1) % n]
        p1 = polygon[i]
        p2 = polygon[(i + 1) % n]
        p3 = polygon[(i + 2) % n]
        for j in range(samples_per_segment):
            t = j / samples_per_segment
            t2 = t * t
            t3 = t2 * t
            b_m1 = (1.0 - 3.0 * t + 3.0 * t2 - t3) / 6.0
            b_0 = (4.0 - 6.0 * t2 + 3.0 * t3) / 6.0
            b_1 = (1.0 + 3.0 * t + 3.0 * t2 - 3.0 * t3) / 6.0
            b_2 = t3 / 6.0
            x = b_m1 * p0[0] + b_0 * p1[0] + b_1 * p2[0] + b_2 * p3[0]
            y = b_m1 * p0[1] + b_0 * p1[1] + b_1 * p2[1] + b_2 * p3[1]
            out.append([round(float(x), 1), round(float(y), 1)])
    return out


def _context_aware_smooth(
    polygon,
    max_sigma: float,
    corner_threshold_deg: float,
    sensitivity: float = 3.0,
    measure_window: int = 2,
):
    """Variable-σ Gaussian smoothing along a closed polygon, with an
    ADAPTIVE corner threshold.

    Each polygon picks its own threshold:
        eff_threshold = max(corner_threshold_deg, avg_turn × sensitivity)

    Per-vertex σ is then:
        σ(i) = max_sigma · max(0, 1 − turn_angle(i) / eff_threshold)

    Vertices whose turn is in line with the polygon's typical
    curvature get σ ≈ max_sigma and smooth out into a clean curve.
    A vertex only counts as a "real corner" when its turn substantially
    exceeds the polygon's average - at which point σ → 0 and it
    doesn't move at all.

    The adaptive part is the key: a small organic shape where every
    vertex turns 20–40° judges corners against its OWN distribution,
    not a fixed-degree absolute, so the polygon as a whole still
    smooths properly. A square with mostly straight edges and four
    90° corners still preserves the corners because the average turn
    is low and 90° towers over it.

    Returns FLOAT polygon (1-decimal rounding) - same vertex count
    as input, repositioned in place.
    """
    import math
    n = len(polygon)
    if n < 5 or max_sigma <= 0 or corner_threshold_deg <= 0:
        return [list(p) for p in polygon]

    eff_window = max(1, min(measure_window, n // 6))

    # Per-vertex turn angle in degrees, measured over a small window
    # so it doesn't jitter pixel-by-pixel along noisy contours.
    turn_angles = [0.0] * n
    for i in range(n):
        prev_i = (i - eff_window) % n
        next_i = (i + eff_window) % n
        v1x = float(polygon[i][0]) - float(polygon[prev_i][0])
        v1y = float(polygon[i][1]) - float(polygon[prev_i][1])
        v2x = float(polygon[next_i][0]) - float(polygon[i][0])
        v2y = float(polygon[next_i][1]) - float(polygon[i][1])
        n1 = math.sqrt(v1x * v1x + v1y * v1y)
        n2 = math.sqrt(v2x * v2x + v2y * v2y)
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        cos_a = max(-1.0, min(1.0, (v1x * v2x + v1y * v2y) / (n1 * n2)))
        turn_angles[i] = math.degrees(math.acos(cos_a))

    # Adaptive threshold: a polygon's "corner" is a turn that's
    # substantially sharper than the polygon's typical turn. Floor at
    # corner_threshold_deg so a near-circle (avg ≈ 360/n, very small)
    # doesn't end up with a tiny threshold that flags everything.
    avg_turn = sum(turn_angles) / max(1, len(turn_angles))
    eff_threshold = max(corner_threshold_deg, avg_turn * sensitivity)

    # Per-vertex sigma - linear falloff from max_sigma at 0° to 0 at
    # eff_threshold.
    sigmas = [
        max(0.0, max_sigma * (1.0 - ta / eff_threshold))
        for ta in turn_angles
    ]

    # Identify "corner-ish" indices - vertices whose turn is at least
    # half the effective threshold. The smoothing kernel for any vertex
    # gets capped so it can't reach past the nearest corner-ish vertex
    # on either side. Without this cap, a midpoint vertex on a straight
    # edge sees its kernel reach across into the next edge through the
    # corner, averaging the corner into the edge midpoint and bulging
    # the straight edge inward.
    corner_floor = eff_threshold * 0.5
    corner_indices = [i for i in range(n) if turn_angles[i] >= corner_floor]
    # For each vertex, distance (in index units, accounting for wrap)
    # to the nearest corner-ish vertex on either side.
    if corner_indices:
        # Sort corners and use them to find left/right distance per vertex.
        corner_sorted = sorted(corner_indices)
        nearest_corner_dist = [n] * n  # default to "no corner reachable"
        for i in range(n):
            # Min wrap-aware distance to any corner index.
            best = n
            for ci in corner_sorted:
                d = abs(i - ci)
                d = min(d, n - d)
                if d < best:
                    best = d
            nearest_corner_dist[i] = best
    else:
        nearest_corner_dist = [n // 2] * n

    # Apply Gaussian per vertex with that vertex's own sigma. Each
    # vertex computes its new position as a weighted mean of its
    # neighbours along the closed loop; the kernel is built fresh
    # per-vertex so different vertices get different smoothing
    # strengths in the same pass.
    smoothed: list[list[float]] = []
    for i in range(n):
        sigma = sigmas[i]
        if sigma < 0.05:
            # No smoothing - keep position exactly.
            smoothed.append(
                [round(float(polygon[i][0]), 1), round(float(polygon[i][1]), 1)]
            )
            continue
        radius = max(1, int(round(sigma * 3)))
        # Cap 1 - never reach the polygon's opposite side on small polys.
        radius = min(radius, max(1, n // 4))
        # Cap 2 - never reach across a corner. dist_to_corner is the
        # number of vertex steps to the nearest corner-ish vertex on
        # either side; halving it gives a kernel that fades to zero
        # before touching the corner, so a straight edge midpoint
        # only averages with other vertices on the SAME edge.
        ndist = nearest_corner_dist[i]
        if ndist < n:
            radius = min(radius, max(1, ndist - 1))
        ks = list(range(-radius, radius + 1))
        ws = [math.exp(-0.5 * (k / sigma) ** 2) for k in ks]
        ws_sum = sum(ws)
        ax = ay = 0.0
        for j, k in enumerate(ks):
            idx = (i + k) % n
            w = ws[j] / ws_sum
            ax += w * float(polygon[idx][0])
            ay += w * float(polygon[idx][1])
        smoothed.append([round(ax, 1), round(ay, 1)])
    return smoothed


def _duplicate_sharp_vertices(
    polygon,
    threshold_deg: float,
    measure_window: int = 2,
):
    """Insert a duplicate vertex right after every vertex whose local
    turn angle ≥ threshold_deg.

    The duplicate is what triggers the FE's Catmull-Rom-to-Bezier
    cusp (the duplicate-pair segment collapses to colinear tangents
    on each side, producing a real sharp angle in the rendered curve
    instead of a rounded transition).

    Runs AFTER context-aware smoothing - so we measure turn angle
    on the *smoothed* polygon, and only vertices that survived the
    smoothing pass as sharp end up duplicated."""
    import math
    n = len(polygon)
    if n < 3 or threshold_deg <= 0:
        return [list(p) for p in polygon]
    eff_window = max(1, min(measure_window, n // 6))

    out: list[list[float]] = []
    for i in range(n):
        out.append([round(float(polygon[i][0]), 1), round(float(polygon[i][1]), 1)])
        prev_i = (i - eff_window) % n
        next_i = (i + eff_window) % n
        v1x = float(polygon[i][0]) - float(polygon[prev_i][0])
        v1y = float(polygon[i][1]) - float(polygon[prev_i][1])
        v2x = float(polygon[next_i][0]) - float(polygon[i][0])
        v2y = float(polygon[next_i][1]) - float(polygon[i][1])
        n1 = math.sqrt(v1x * v1x + v1y * v1y)
        n2 = math.sqrt(v2x * v2x + v2y * v2y)
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        cos_a = max(-1.0, min(1.0, (v1x * v2x + v1y * v2y) / (n1 * n2)))
        angle = math.degrees(math.acos(cos_a))
        if angle >= threshold_deg:
            out.append(
                [round(float(polygon[i][0]), 1), round(float(polygon[i][1]), 1)]
            )
    return out


def _gaussian_smooth_polygon(polygon, sigma: float):
    """Apply a 1-D Gaussian filter along a closed polygon's vertex
    list - moves each vertex toward the local mean of its neighbours
    so wiggles between control points smooth out, without changing
    the vertex count. The first/last entries wrap around (closed
    loop). Returns FLOAT coordinates (rounded to 1 decimal) so
    sub-pixel positions survive the smoothing → keeps the FE
    renderer's Bezier curves smooth at extreme zoom."""
    n = len(polygon)
    if n < 5 or sigma <= 0:
        return polygon
    import numpy as np
    pts = np.asarray(polygon, dtype=np.float64)              # (N, 2)
    radius = max(1, int(round(sigma * 3)))
    if radius >= n:
        radius = max(1, n // 2)
    idx = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (idx / sigma) ** 2)
    kernel /= kernel.sum()
    pad = np.concatenate([pts[-radius:], pts, pts[:radius]], axis=0)
    smoothed = np.zeros_like(pts)
    for i in range(2 * radius + 1):
        smoothed += kernel[i] * pad[i:i + n]
    return [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in smoothed]


def _catmull_rom_subdivide(polygon, samples_per_segment: int):
    """Catmull-Rom corner-cut subdivision with NO point removal.

    For each consecutive vertex pair (P1, P2), inserts
    `samples_per_segment` interpolated points between them, sampled
    along a centripetal Catmull-Rom curve fitted through (P0, P1, P2,
    P3). All original vertices are retained exactly; the result is a
    denser polygon that reads as smooth curves through the originals.

    Polygons with fewer than 3 vertices, or samples_per_segment ≤ 0,
    pass through untouched."""
    n = len(polygon)
    if n < 3 or samples_per_segment <= 0:
        return polygon
    out = []
    for i in range(n):
        p0 = polygon[(i - 1) % n]
        p1 = polygon[i]
        p2 = polygon[(i + 1) % n]
        p3 = polygon[(i + 2) % n]
        out.append([round(float(p1[0]), 1), round(float(p1[1]), 1)])
        for j in range(1, samples_per_segment + 1):
            t = j / (samples_per_segment + 1)
            t2 = t * t
            t3 = t2 * t
            x = 0.5 * (
                (2.0 * p1[0])
                + (-p0[0] + p2[0]) * t
                + (2.0 * p0[0] - 5.0 * p1[0] + 4.0 * p2[0] - p3[0]) * t2
                + (-p0[0] + 3.0 * p1[0] - 3.0 * p2[0] + p3[0]) * t3
            )
            y = 0.5 * (
                (2.0 * p1[1])
                + (-p0[1] + p2[1]) * t
                + (2.0 * p0[1] - 5.0 * p1[1] + 4.0 * p2[1] - p3[1]) * t2
                + (-p0[1] + 3.0 * p1[1] - 3.0 * p2[1] + p3[1]) * t3
            )
            out.append([round(float(x), 1), round(float(y), 1)])
    return out


def _shoelace_area(points: list[list[int]]) -> float:
    """Absolute polygon area via the shoelace formula. Used by the
    sub-mask outlier filter to compare disjoint pieces of one
    detection's mask against each other."""
    n = len(points)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x0, y0 = points[i][0], points[i][1]
        x1, y1 = points[(i + 1) % n][0], points[(i + 1) % n][1]
        s += float(x0) * float(y1) - float(x1) * float(y0)
    return abs(s) * 0.5


def _mask_to_polygons(mask) -> list[list[list[int]]]:
    """Convert a 2D bool/uint8 mask to a list of (x, y) polygon contours.

    Pipeline:
      1. Coerce mask to uint8 binary.
      2. cv2.findContours to extract pixel-walking outlines.
      3. Douglas-Peucker simplify each contour by SAM3_POLY_SIMPLIFY_EPS
         × perimeter - collapses single-pixel staircases into smooth
         curves while preserving real corners. Skipped when the env
         knob is 0.
      4. Sub-mask outlier filter (negative-only): when a detection
         comes back with multiple disjoint polygons, drop sub-polygons
         whose area is below SAM3_SUBMASK_MIN_RELATIVE_AREA × the
         largest sub-polygon's area. Only smaller outliers are removed
         - the biggest piece is always kept and never trimmed.

    Empty / degenerate (<3 vertex) contours are dropped."""
    import cv2
    import numpy as np

    if isinstance(mask, torch.Tensor):
        m = mask.detach().to(torch.uint8).cpu().numpy()
    elif isinstance(mask, np.ndarray):
        m = mask
    else:
        m = np.asarray(mask)

    if m.ndim == 3:
        m = m.squeeze()
    if m.dtype != np.uint8:
        m = (m > 0).astype(np.uint8)

    # Step -1 - morphological opening. Breaks thin pixel bridges
    # between disjoint blobs of the same concept BEFORE the contour
    # walker has a chance to trace them as a single contour. The
    # canonical case: a high-vis vest has front + back panels, SAM3
    # returns ONE instance with a single mask, and a single-pixel
    # sliver across the body (artefact of the model's coarse output)
    # makes cv2.findContours walk both panels through the sliver.
    # Resampling then turns the sliver into a straight line crossing
    # the body. Opening with a 5×5 kernel removes ≤5 px features
    # without affecting glove / vest scale.
    if SAM3_MASK_OPEN_PX > 0:
        before_blobs = int(cv2.connectedComponents(m)[0] - 1)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (SAM3_MASK_OPEN_PX, SAM3_MASK_OPEN_PX)
        )
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
        after_blobs = int(cv2.connectedComponents(m)[0] - 1)
        if after_blobs > before_blobs:
            print(
                f"[charlie] morph-open: split mask into "
                f"{after_blobs} blobs (was {before_blobs}) - bridge removed"
            )

    # Step 0 - pre-contour mask smoothing. Gaussian blur of the
    # binary mask, then re-threshold at 0.5. Softens the pixel-level
    # staircase along the boundary so the contour walker emits fewer
    # tiny zigzags; downstream Douglas-Peucker then removes the
    # remaining redundant vertices, producing visibly smooth polygons.
    if SAM3_MASK_SMOOTH_PX > 0:
        k = 2 * SAM3_MASK_SMOOTH_PX + 1
        m_blur = cv2.GaussianBlur(m.astype(np.float32), (k, k), float(SAM3_MASK_SMOOTH_PX))
        m = (m_blur > 0.5).astype(np.uint8)

    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys: list[list[list[int]]] = []
    for c in contours:
        if c.shape[0] < 3:
            continue
        if SAM3_POLY_SIMPLIFY_EPS > 0:
            eps = SAM3_POLY_SIMPLIFY_EPS * cv2.arcLength(c, True)
            if eps > 0:
                simplified = cv2.approxPolyDP(c, eps, True)
                # Don't replace if simplification collapsed below 3
                # vertices (degenerate shape).
                if simplified.shape[0] >= 3:
                    c = simplified
        polys.append([[int(p[0][0]), int(p[0][1])] for p in c])

    # NOTE: corner detection now lives AFTER the sub-mask outlier
    # filter (step 4 below). Detecting before the filter caused an
    # index-misalignment bug - if the filter dropped a sub-polygon,
    # the corner_positions_per_poly list still had its entry, so
    # corners detected on the DROPPED polygon ended up inserted into
    # a SURVIVING polygon, drawing spurious lines from one
    # sub-segmentation to where another used to be.
    corner_positions_per_poly: list[list[tuple[float, float]]] = []

    # Step 4 - negative-outlier sub-mask filter.
    if len(polys) > 1 and SAM3_SUBMASK_MIN_RELATIVE_AREA > 0:
        areas = [_shoelace_area(p) for p in polys]
        max_area = max(areas)
        if max_area > 0:
            floor = max_area * SAM3_SUBMASK_MIN_RELATIVE_AREA
            kept = [p for p, a in zip(polys, areas) if a >= floor]
            n_dropped = len(polys) - len(kept)
            if n_dropped > 0:
                print(
                    f"[charlie] sub-mask outlier filter: kept {len(kept)}/{len(polys)} "
                    f"polygons (dropped {n_dropped} below {SAM3_SUBMASK_MIN_RELATIVE_AREA*100:.0f}% "
                    f"× max={int(max_area)} px²)"
                )
            polys = kept

    # Detect sharp corners NOW (post-outlier-filter, pre-smoothing).
    # corner_positions_per_poly[i] aligns with polys[i] from this
    # point on - every later step (subdivide / smooth / B-spline /
    # resample / anchor) is per-polygon and preserves count, so the
    # alignment holds.
    if SAM3_CORNER_ANGLE_THRESHOLD_DEG > 0:
        for poly in polys:
            corner_positions_per_poly.append(
                _detect_sharp_corners(
                    poly,
                    SAM3_CORNER_ANGLE_THRESHOLD_DEG,
                    window=SAM3_CORNER_WINDOW,
                    nms_radius=SAM3_CORNER_NMS_RADIUS,
                    min_vertices=SAM3_CORNER_MIN_VERTICES,
                )
            )
    else:
        corner_positions_per_poly = [[] for _ in polys]

    # ── Context-aware smoothing pipeline ──────────────────────────
    # Step 5a - Catmull-Rom subdivision for vertex density. The
    # context-aware Gaussian only repositions existing vertices, so
    # the polygon needs a reasonable density before smoothing for the
    # result to look smooth.
    if SAM3_CONTEXT_SUBDIVIDE > 0 and polys:
        polys = [_catmull_rom_subdivide(p, SAM3_CONTEXT_SUBDIVIDE) for p in polys]

    # Step 5b - variable-sigma Gaussian. Per-vertex σ scales inversely
    # with local curvature: smooth sections smooth, sharp corners
    # stay put. Single algorithm replaces the old subdivide+Gaussian+
    # B-spline+resample+corner-detect/anchor stack.
    if SAM3_CONTEXT_SMOOTH_MAX_SIGMA > 0 and polys:
        polys = [
            _context_aware_smooth(
                p,
                SAM3_CONTEXT_SMOOTH_MAX_SIGMA,
                SAM3_CONTEXT_CORNER_DEG,
                sensitivity=SAM3_CONTEXT_CORNER_SENSITIVITY,
            )
            for p in polys
        ]

    # Step 5c - duplicate vertices that survived smoothing as sharp
    # so the FE Bezier renderer produces real cusps there.
    if SAM3_CONTEXT_DUPE_DEG > 0 and polys:
        polys = [
            _duplicate_sharp_vertices(p, SAM3_CONTEXT_DUPE_DEG)
            for p in polys
        ]
    # ──────────────────────────────────────────────────────────────

    # ── Legacy multi-step smoothing (disabled by default in v2 of
    #    the pipeline; kept under env knobs for revertability) ─────

    # Step 5 - Catmull-Rom subdivision (additive: every original
    # vertex is retained exactly, plus N interpolated points between
    # each pair).
    if SAM3_POLY_SUBDIVIDE_SAMPLES > 0 and polys:
        polys = [_catmull_rom_subdivide(p, SAM3_POLY_SUBDIVIDE_SAMPLES) for p in polys]

    # Step 6 - Gaussian smoothing along the now-dense vertex list.
    # Same point count, but each vertex gets replaced with a
    # Gaussian-weighted local mean so the wiggles Catmull-Rom traces
    # through every staircase control point smooth out into a clean
    # curve. Runs after subdivision because more vertices = better
    # local-mean estimate for the same sigma. Multiple passes give
    # an effective σ_total = σ × √passes - cheaper than running one
    # huge kernel and gives a smoother (more "viscous-fluid") result
    # than a single pass.
    if SAM3_POLY_SMOOTH_SIGMA > 0 and polys and SAM3_POLY_SMOOTH_PASSES > 0:
        for _ in range(SAM3_POLY_SMOOTH_PASSES):
            polys = [_gaussian_smooth_polygon(p, SAM3_POLY_SMOOTH_SIGMA) for p in polys]

    # Step 7 - periodic cubic B-spline approximation. Replaces each
    # polygon with a smooth curve that's pulled toward the (already-
    # smoothed) control points but not constrained to pass through
    # them. C2 continuity end-to-end - any residual staircase in the
    # control vertices is fully washed out.
    if SAM3_POLY_SPLINE_SAMPLES > 0 and polys:
        polys = [_periodic_cubic_bspline(p, SAM3_POLY_SPLINE_SAMPLES) for p in polys]

    # Step 8 - arc-length resample to a sparse, evenly-spaced control
    # set. The FE renders polygons as cubic Bezier <path>s that pass
    # through every point - with thousands of clustered vertices each
    # Bezier segment is too short to actually curve, so the aggregate
    # reads as a polyline and shows the underlying staircase. Down-
    # sampling to ~one point per 30 px of perimeter (clamped 20–120)
    # gives the renderer wide arcs to interpolate, producing visually
    # smooth shapes at any zoom.
    if SAM3_POLY_RESAMPLE_PX_SPACING > 0 and polys:
        import numpy as np
        resampled: list[list[list[int]]] = []
        for p in polys:
            if len(p) < 4:
                resampled.append(p)
                continue
            arr = np.asarray(p, dtype=np.float64)
            closed = np.vstack([arr, arr[0:1]])
            seg_lens = np.sqrt((np.diff(closed, axis=0) ** 2).sum(axis=1))
            perimeter = float(seg_lens.sum())
            target = int(round(perimeter / SAM3_POLY_RESAMPLE_PX_SPACING))
            target = max(SAM3_POLY_RESAMPLE_MIN, min(SAM3_POLY_RESAMPLE_MAX, target))
            resampled.append(_resample_polygon_arc_length(p, target))
        polys = resampled

    # Step 9 - re-anchor sharp corners detected pre-smoothing. Snaps
    # the nearest smoothed vertex to the exact corner position and
    # inserts a duplicate so the FE Bezier renderer produces a real
    # cusp. Sign edges, building corners, anything truly angular in
    # the source comes through intact instead of rounded into a
    # bezier curve.
    if SAM3_CORNER_ANGLE_THRESHOLD_DEG > 0 and polys:
        for pi, poly in enumerate(polys):
            if pi < len(corner_positions_per_poly) and corner_positions_per_poly[pi]:
                polys[pi] = _anchor_corners(
                    poly,
                    corner_positions_per_poly[pi],
                    max_dist_px=SAM3_CORNER_MAX_DIST_PX,
                    carve_radius_px=SAM3_CORNER_CARVE_RADIUS_PX,
                )

    return polys


def _crop_to_jpg_b64(image: Image.Image, box: list[int], max_side: int = 384, quality: int = 80) -> str:
    """Encode a crop of `image` (clipped to box, downscaled to max_side
    on the long edge) as a base64 JPEG. Used to populate the per-
    detection thumbnails in the pipeline popup."""
    W, H = image.size
    x0 = max(0, min(W, int(box[0])))
    y0 = max(0, min(H, int(box[1])))
    x1 = max(0, min(W, int(box[2])))
    y1 = max(0, min(H, int(box[3])))
    if x1 <= x0 or y1 <= y0:
        return ""
    crop = image.crop((x0, y0, x1, y1))
    long_side = max(crop.size)
    if long_side > max_side:
        scale = max_side / long_side
        new_size = (max(1, int(crop.size[0] * scale)), max(1, int(crop.size[1] * scale)))
        crop = crop.resize(new_size, Image.LANCZOS)
    buf = io.BytesIO()
    crop.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _normalise_labels(labels: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    clean: list[str] = []
    for raw in labels:
        s = (raw or "").strip()
        if s and s.lower() not in seen:
            seen.add(s.lower())
            clean.append(s)
    return clean


def _build_detection(
    image: Image.Image,
    box: list[int],
    mask,
    score: float,
    label: str,
    include_crop: bool,
    scale_back: float = 1.0,
) -> dict:
    """Pack one SAM3 hit into the V2-compatible detection envelope.

    `scale_back` rescales the box and polygon vertices from inference
    coords to original-image coords (multiplier ≥ 1). Use 1.0 when
    SAM3 ran on the original image. The crop thumbnail is taken from
    the (already-original) `image` at the rescaled box coords so the
    FE never has to know the resize happened.

    The fields populated by other models in V2 (embedding,
    embedding_siglip, vlm_*) are returned empty/null so the FE can
    iterate `detections` without restructuring."""
    polys = _mask_to_polygons(mask)
    if scale_back != 1.0 and polys:
        # Keep float precision through the scale-back so smooth sub-
        # pixel positions survive into the response. SVG <path> in
        # the FE handles floats fine; PIL rasterisation in the cover-
        # photo path coerces to int at draw time.
        polys = [
            [[round(float(p[0]) * scale_back, 1), round(float(p[1]) * scale_back, 1)] for p in poly]
            for poly in polys
        ]
    box_orig = [int(round(c * scale_back)) for c in box[:4]]
    return {
        "box": box_orig,
        "mask": {"polygons": polys} if polys else None,
        "embedding": [],
        "embedding_siglip": [],
        "gd_label": label,
        "gd_variant": label,
        "gd_score": float(score),
        "vlm_label": None,
        "vlm_score": None,
        "vlm_ms": 0.0,
        "crop_jpg_b64": _crop_to_jpg_b64(image, box_orig) if include_crop else "",
    }


def _resize_for_inference(image: Image.Image, target: int | None = None) -> tuple[Image.Image, float]:
    """Resize `image` so its longest edge is (target or SAM3_TARGET_LONGEST_EDGE)
    px. Returns (image_inf, scale_back) where scale_back multiplies
    inference-space coords back to original-image coords.

    Resizes in BOTH directions - small images upscale, large ones
    downscale, so SAM3 always sees the same input scale and the
    resulting masks have the same effective resolution before being
    scaled back. This is the fix for jaggy edges on small uploads:
    a 600 px image used to produce 600-px-resolution polygons; now it
    gets 1500-px-resolution polygons rescaled down, which read as
    smooth curves at any display size.

    Returns the original image untouched only when the resize target
    is 0 / disabled, or the input is already exactly at target size.
    """
    if (target or SAM3_TARGET_LONGEST_EDGE) <= 0:
        return image, 1.0
    W, H = image.size
    longest = max(W, H)
    if longest == (target or SAM3_TARGET_LONGEST_EDGE):
        return image, 1.0
    inf_scale = (target or SAM3_TARGET_LONGEST_EDGE) / longest
    new_size = (max(1, int(round(W * inf_scale))), max(1, int(round(H * inf_scale))))
    # LANCZOS works well for both up and down - for upscaling it gives
    # smoother interpolation than NEAREST without introducing the
    # ringing that BICUBIC sometimes shows on edges.
    return image.resize(new_size, Image.LANCZOS), 1.0 / inf_scale


# Cross-call vision-embed cache for INTERACTIVE tools. A click re-runs the
# heavy SAM3 vision encoder on the same image every time; caching the encoder
# output per image makes every click after the first skip it entirely
# (the per-label text passes are cheap by comparison). Keyed by caller-supplied
# id + inference size; tiny LRU because entries hold GPU tensors.
from collections import OrderedDict as _OrderedDict

_VISION_CACHE: "_OrderedDict[str, tuple]" = _OrderedDict()
_VISION_CACHE_MAX = 3


def _get_image_embeds_cached(image: Image.Image, cache_key: str | None):
    if not cache_key:
        return _try_get_image_embeds(image)
    key = f"{cache_key}:{image.size[0]}x{image.size[1]}"
    hit = _VISION_CACHE.get(key)
    if hit is not None:
        _VISION_CACHE.move_to_end(key)
        return hit
    out = _try_get_image_embeds(image)
    if out[0] is not None:
        _VISION_CACHE[key] = out
        while len(_VISION_CACHE) > _VISION_CACHE_MAX:
            _VISION_CACHE.popitem(last=False)
    return out


def _try_get_image_embeds(image: Image.Image):
    """Best-effort SAM3 vision-encoder pass that returns a
    Sam3VisionEncoderOutput suitable for passing back into
    Sam3Model.forward(vision_embeds=...). When this succeeds, every
    subsequent forward for the same image can skip the heavy image
    encoder - a per-click speedup of ~N× when scanning N labels.

    Returns (vision_embeds_or_none, original_sizes_or_none, processed_pixel_values_or_none).
    Falls back to (None, None, None) on any error so callers can do a
    plain forward path without losing correctness."""
    if _MODEL is None or _PROCESSOR is None:
        return None, None, None
    try:
        proc_inputs = _PROCESSOR(images=image, return_tensors="pt").to(_MODEL.device)
    except Exception as e:
        print(f"[charlie] vision-embed cache: processor pre-pass failed: {e}")
        return None, None, None

    pixel_values = proc_inputs.get("pixel_values")
    original_sizes = proc_inputs.get("original_sizes")

    # Try a few methods exposed by SAM-family transformers - these
    # vary across versions, so we probe in order of preference.
    for attr in ("get_image_embeddings", "get_vision_embeddings", "encode_image"):
        method = getattr(_MODEL, attr, None)
        if method is None:
            continue
        try:
            with torch.inference_mode():
                vision_embeds = method(pixel_values=pixel_values)
            return vision_embeds, original_sizes, pixel_values
        except Exception as e:
            print(f"[charlie] vision-embed cache: {attr}() raised {type(e).__name__}: {e}")
            continue

    # Last resort - call the vision encoder submodule directly.
    vision_encoder = getattr(_MODEL, "vision_encoder", None) or getattr(_MODEL, "vision_model", None)
    if vision_encoder is not None:
        try:
            with torch.inference_mode():
                vision_embeds = vision_encoder(pixel_values=pixel_values)
            return vision_embeds, original_sizes, pixel_values
        except Exception as e:
            print(f"[charlie] vision-embed cache: vision_encoder direct call raised {type(e).__name__}: {e}")

    return None, None, None


def _run_text_with_cached_vision(label: str, vision_embeds, original_sizes, pixel_values):
    """Run SAM3 text-prompt path using a cached vision_embeds. Falls
    back to a plain forward if any of the cache-aware code paths
    error. Returns the same shape as a normal model(**inputs) call."""
    text_inputs = _PROCESSOR(text=label, return_tensors="pt").to(_MODEL.device)
    # Drop pixel_values from the text-inputs dict if the processor
    # decided to fabricate any (some HF processors do).
    text_inputs.pop("pixel_values", None)
    text_inputs.pop("original_sizes", None)
    forward_kwargs = dict(text_inputs)
    if original_sizes is not None:
        forward_kwargs["original_sizes"] = original_sizes
    if vision_embeds is not None:
        forward_kwargs["vision_embeds"] = vision_embeds
    else:
        # No cache - caller should have supplied pixel_values.
        if pixel_values is not None:
            forward_kwargs["pixel_values"] = pixel_values
    with torch.inference_mode():
        outputs = _MODEL(**forward_kwargs)
    return outputs, text_inputs


def _segment_one_label(
    image: Image.Image,
    label: str,
    *,
    threshold: float | None = None,
    mask_threshold: float | None = None,
    min_relative_area: float | None = None,
) -> tuple[list[tuple], float, float, int]:
    """One SAM3 forward pass for a single text prompt.

    Returns (raw_results, predict_ms, post_ms, dropped_tiny) where
    raw_results is a list of (box, mask, score) tuples already in
    original-image coords, and dropped_tiny is the count of detections
    filtered out for being below the area threshold.

    `threshold` / `mask_threshold` / `min_relative_area` override the
    SAM3_* module-level defaults for this single call only - used by
    the project page's per-run SAM3 knobs.
    """
    if _MODEL is None or _PROCESSOR is None:
        raise RuntimeError("SAM3 not loaded - call load_sam3() first")

    import numpy as np

    eff_threshold = SAM3_THRESHOLD if threshold is None else float(threshold)
    eff_mask_threshold = SAM3_MASK_THRESHOLD if mask_threshold is None else float(mask_threshold)
    eff_min_relative = SAM3_MIN_RELATIVE_AREA if min_relative_area is None else float(min_relative_area)

    W, H = image.size
    min_area_px = max(SAM3_MIN_AREA_PX, int(W * H * SAM3_MIN_AREA_FRAC))

    inputs = outputs = None
    raw: list[tuple] = []
    predict_ms = 0.0
    post_ms = 0.0
    dropped_tiny = 0
    try:
        inputs = _PROCESSOR(
            images=image,
            text=label,
            return_tensors="pt",
        ).to(_MODEL.device)

        t_predict = time.perf_counter()
        with torch.inference_mode():
            outputs = _MODEL(**inputs)
        _device_sync()
        predict_ms = (time.perf_counter() - t_predict) * 1000.0

        t_post = time.perf_counter()
        target_sizes = inputs.get("original_sizes")
        if target_sizes is not None and hasattr(target_sizes, "tolist"):
            target_sizes = target_sizes.tolist()
        else:
            target_sizes = [[H, W]]

        results = _PROCESSOR.post_process_instance_segmentation(
            outputs,
            threshold=eff_threshold,
            mask_threshold=eff_mask_threshold,
            target_sizes=target_sizes,
        )[0]
        post_ms = (time.perf_counter() - t_post) * 1000.0

        boxes_t = results["boxes"]
        masks_t = results["masks"]
        scores_t = results["scores"]

        # Pass 1 - collect every hit alongside its mask area so the
        # relative filter can compute the per-label max.
        hits: list[tuple] = []  # (box_list, mask, score, area)
        for i in range(len(boxes_t)):
            box = boxes_t[i]
            mask = masks_t[i]
            score = float(scores_t[i])
            box_list = box.tolist() if hasattr(box, "tolist") else list(box)
            if isinstance(mask, torch.Tensor):
                mask_area = int(mask.to(torch.uint8).sum().item())
            else:
                mask_area = int((np.asarray(mask) > 0).sum())
            hits.append((box_list, mask, score, mask_area))

        # Pass 2a - absolute floor: kill anything below max(floor px,
        # image-relative fraction). Catches noise on small images.
        kept_floor = [h for h in hits if h[3] >= min_area_px]
        n_dropped_floor = len(hits) - len(kept_floor)
        dropped_tiny += n_dropped_floor

        # Pass 2b - relative-to-max-in-label: anything under
        # eff_min_relative × the biggest surviving mask is treated as
        # a noise fragment that share-labels with a real object, and
        # dropped. Skipped when the label only fired once (nothing
        # to compare against).
        if len(kept_floor) > 1 and eff_min_relative > 0:
            max_area = max(h[3] for h in kept_floor)
            rel_floor = int(max_area * eff_min_relative)
            kept_rel = [h for h in kept_floor if h[3] >= rel_floor]
            n_dropped_rel = len(kept_floor) - len(kept_rel)
            dropped_tiny += n_dropped_rel
        else:
            kept_rel = kept_floor
            n_dropped_rel = 0

        # Surface what each pass actually saw / dropped - invaluable
        # when a real label shows up in the SAM3 output but doesn't
        # make it to the FE. min_area_px is logged because it's the
        # main per-image-resolution variable.
        if hits:
            sample_areas = sorted([h[3] for h in hits])
            print(
                f"[charlie] label={label!r}: sam3={len(hits)} "
                f"areas[min/p50/max]={sample_areas[0]}/{sample_areas[len(sample_areas)//2]}/{sample_areas[-1]} "
                f"floor={min_area_px}px (dropped {n_dropped_floor}) "
                f"rel{eff_min_relative*100:.0f}%={('skipped' if len(kept_floor) <= 1 else f'dropped {n_dropped_rel}')} "
                f"kept={len(kept_rel)}"
            )

        for box_list, mask, score, _area in kept_rel:
            raw.append((box_list, mask, score))
    finally:
        del inputs, outputs
        _device_empty_cache()

    return raw, predict_ms, post_ms, dropped_tiny


def segment_labels(
    image: Image.Image,
    labels: Iterable[str],
    include_crops: bool = True,
    *,
    threshold: float | None = None,
    mask_threshold: float | None = None,
    min_relative_area: float | None = None,
) -> tuple[list[dict], dict]:
    """Run SAM3 promptable concept segmentation for each label and
    concatenate the per-label detections.

    `threshold`, `mask_threshold`, `min_relative_area` override the
    module-level SAM3_* defaults for this call only. Used by the
    project page's per-run SAM3 knobs so a labelling job can opt for
    a stricter detection threshold or a looser mask precision without
    touching the process-wide config.

    Returns:
        (detections, timings_ms)
        timings_ms keys: sam3_predict_ms (sum across labels),
                         sam3_post_ms (sum), encode_crops_ms, total_ms,
                         per_label_ms ({label: ms})
    """
    if _MODEL is None or _PROCESSOR is None:
        raise RuntimeError("SAM3 not loaded - call load_sam3() first")

    t_total = time.perf_counter()
    timings: dict = {
        "resize_ms": 0.0,
        "sam3_predict_ms": 0.0,
        "sam3_post_ms": 0.0,
        "encode_crops_ms": 0.0,
        "total_ms": 0.0,
        "per_label_ms": {},
        "dropped_tiny": 0,
        "inference_size": list(image.size),
        "scale_back": 1.0,
    }

    clean = _normalise_labels(labels)
    if not clean:
        timings["total_ms"] = (time.perf_counter() - t_total) * 1000.0
        return [], timings

    t_resize = time.perf_counter()
    image_inf, scale_back = _resize_for_inference(image)
    timings["resize_ms"] = (time.perf_counter() - t_resize) * 1000.0
    timings["inference_size"] = list(image_inf.size)
    timings["scale_back"] = scale_back

    # Compute the SAM3 vision embedding once and reuse it for every
    # label. This skips the heavy image-encoder forward on labels 2..N,
    # cutting per-image wall-clock time roughly in half for a 2-label
    # project (hare + rabbit) and proportionally more for wider label sets.
    import numpy as np
    t_vis = time.perf_counter()
    vision_embeds, original_sizes, pixel_values = _try_get_image_embeds(image_inf)
    timings["vision_embed_ms"] = (time.perf_counter() - t_vis) * 1000.0

    all_hits: list[tuple] = []
    for label in clean:
        t_label = time.perf_counter()
        if vision_embeds is not None:
            raw, predict_ms, post_ms, dropped_tiny = _segment_one_label_with_cached_vision(
                image_inf, label,
                vision_embeds, original_sizes, pixel_values,
                threshold=threshold,
                mask_threshold=mask_threshold,
            )
        else:
            raw, predict_ms, post_ms, dropped_tiny = _segment_one_label(
                image_inf, label,
                threshold=threshold,
                mask_threshold=mask_threshold,
                min_relative_area=min_relative_area,
            )
        timings["sam3_predict_ms"] += predict_ms
        timings["sam3_post_ms"] += post_ms
        timings["dropped_tiny"] += dropped_tiny
        timings["per_label_ms"][label] = (time.perf_counter() - t_label) * 1000.0
        for box, mask, score in raw:
            if isinstance(mask, torch.Tensor):
                area = int(mask.to(torch.uint8).sum().item())
            else:
                area = int((np.asarray(mask) > 0).sum())
            all_hits.append((label, box, mask, score, area))

    # Pass 2 - global cross-label area filter. Drops noise specs that
    # share an image with a much bigger detection of any label
    # (typical failure mode: a tiny "road" sliver alongside a large
    # "person"; per-label filter doesn't see across labels so the
    # sliver passes through). Skipped when there's only one hit
    # globally so a single legit small object can't get over-filtered.
    if len(all_hits) > 1 and SAM3_MIN_GLOBAL_RELATIVE_AREA > 0:
        max_area = max(h[4] for h in all_hits)
        global_floor = int(max_area * SAM3_MIN_GLOBAL_RELATIVE_AREA)
        kept_global = [h for h in all_hits if h[4] >= global_floor]
        timings["dropped_tiny"] += len(all_hits) - len(kept_global)
        all_hits = kept_global

    detections: list[dict] = []
    t_crops = time.perf_counter()
    for label, box, mask, score, _area in all_hits:
        detections.append(_build_detection(
            image=image,                    # original-resolution for crops
            box=box,                        # inference-coord
            mask=mask,                      # inference-resolution mask
            score=score,
            label=label,
            include_crop=include_crops,
            scale_back=scale_back,          # rescales box + polygons back to original
        ))
    timings["encode_crops_ms"] += (time.perf_counter() - t_crops) * 1000.0

    timings["total_ms"] = (time.perf_counter() - t_total) * 1000.0
    return detections, timings


# ── Native-resolution tiled inference ────────────────────────────────
# segment_labels resizes EVERY image so its longest edge is
# SAM3_TARGET_LONGEST_EDGE (1500px). On a 4K aerial frame that's a 2.56×
# downscale - a 27px animal lands at ~10px and its mask area falls under
# the absolute area floor, so it never comes back. segment_labels_tiled
# slices the original image into overlapping native-resolution tiles
# whose longest edge equals the inference target, so _resize_for_inference
# is a no-op per tile and small objects keep their pixels. A full-frame
# pass still runs first to catch objects larger than a tile. Detections
# are offset back to original coords (pure vertex translation - only
# polygons leave this module, never bitmaps) and merged with a
# truncation-aware same-label NMS mirroring run_groundingdino's
# _merge_tiled_candidates, so an object seen whole in one tile beats its
# crop-edge-clipped twin from the neighbour.

SAM3_TILED_MAX_DETS = int(os.environ.get("SAM3_TILED_MAX_DETS", "500"))


def _tiled_origins(size: int, tile: int, stride: int) -> list[int]:
    """Window origins covering `size`; the last window shifts back flush
    with the boundary so coverage is complete with no sliver tiles."""
    if size <= tile:
        return [0]
    out = list(range(0, size - tile + 1, stride))
    if out[-1] != size - tile:
        out.append(size - tile)
    return out


def _det_area(det: dict) -> float:
    b = det.get("box") or [0, 0, 0, 0]
    return max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))


def _offset_detection(det: dict, ox: int, oy: int) -> dict:
    """Translate a detection produced on a tile crop back into original-
    image coordinates. Boxes and polygon vertices shift by the tile
    origin; the crop thumbnail already shows the right pixels (it was
    cut from the tile) so it's kept as-is."""
    det["box"] = [
        int(det["box"][0]) + ox, int(det["box"][1]) + oy,
        int(det["box"][2]) + ox, int(det["box"][3]) + oy,
    ]
    mask = det.get("mask")
    if isinstance(mask, dict) and mask.get("polygons"):
        mask["polygons"] = [
            [[round(float(p[0]) + ox, 1), round(float(p[1]) + oy, 1)] for p in poly]
            for poly in mask["polygons"]
        ]
    return det


def _merge_tiled_detections(cands: list[tuple], iou_thr: float = 0.5, contain_frac: float = 0.7) -> list[dict]:
    """Greedy same-label merge of (det, truncated) candidates across the
    full-frame + tile passes. Un-truncated wins over truncated, then
    score, then area - so the fuller view of an object suppresses the
    tile-edge-clipped duplicate.

    Three guards tuned for small-object aerial frames:
    - containment only dedups boxes of comparable size (≤3×), so a
      full-frame "group blob" can't eat the individuals the tiles resolved;
    - near-coincident tiny boxes (<48px) dedup by centre distance - IoU is
      unreliable at that scale (cross-tile localisation jitter of a few px
      drops IoU below any sane threshold);
    - post-merge, a kept box that mostly contains ≥2 smaller kept same-
      label boxes is a phantom cluster around individuals - dropped."""
    order = sorted(
        cands,
        key=lambda c: (c[1], -float(c[0].get("gd_score") or 0.0), -_det_area(c[0])),
    )
    kept: list[dict] = []
    for det, _trunc in order:
        b = det["box"]
        ba = _det_area(det)
        b_ms = min(b[2] - b[0], b[3] - b[1])
        bcx, bcy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
        lbl = (det.get("gd_label") or det.get("gd_variant") or "").lower()
        dup = False
        for k in kept:
            kl = (k.get("gd_label") or k.get("gd_variant") or "").lower()
            if kl != lbl:
                continue
            kb = k["box"]
            ix0, iy0 = max(b[0], kb[0]), max(b[1], kb[1])
            ix1, iy1 = min(b[2], kb[2]), min(b[3], kb[3])
            inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
            if inter <= 0:
                continue
            ka = _det_area(k)
            union = ba + ka - inter
            smaller = min(ba, ka) or 1.0
            comparable = max(ba, ka) <= 3.0 * max(min(ba, ka), 1.0)
            k_ms = min(kb[2] - kb[0], kb[3] - kb[1])
            kcx, kcy = (kb[0] + kb[2]) / 2.0, (kb[1] + kb[3]) / 2.0
            tiny_same_spot = (
                b_ms < 48 and k_ms < 48
                and abs(bcx - kcx) + abs(bcy - kcy) <= 0.75 * max(min(b_ms, k_ms), 1.0)
            )
            if (
                (union > 0 and inter / union >= iou_thr)
                or (comparable and inter / smaller >= contain_frac)
                or tiny_same_spot
            ):
                dup = True
                break
        if not dup:
            kept.append(det)

    # Phantom-cluster eviction: a kept box that mostly contains ≥2 clearly
    # smaller kept same-label boxes is a group blob around individuals the
    # tiles resolved - keep the individuals, drop the blob. Cross-pass
    # mirror of predict()'s _drop_contained rationale.
    if len(kept) > 2:
        drop: set[int] = set()
        for i, k in enumerate(kept):
            ka = _det_area(k)
            kl = (k.get("gd_label") or k.get("gd_variant") or "").lower()
            kb = k["box"]
            inside = 0
            for j, c in enumerate(kept):
                if i == j:
                    continue
                cl = (c.get("gd_label") or c.get("gd_variant") or "").lower()
                if cl != kl:
                    continue
                ca = _det_area(c)
                if ca * 2.0 > ka:
                    continue  # not meaningfully smaller than the candidate blob
                cb = c["box"]
                ix0, iy0 = max(cb[0], kb[0]), max(cb[1], kb[1])
                ix1, iy1 = min(cb[2], kb[2]), min(cb[3], kb[3])
                inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
                if ca > 0 and inter / ca >= 0.7:
                    inside += 1
                    if inside >= 2:
                        drop.add(i)
                        break
        if drop:
            kept = [k for i, k in enumerate(kept) if i not in drop]
    return kept


def _accumulate_timings(total: dict, part: dict) -> None:
    for k, v in part.items():
        if k == "per_label_ms" and isinstance(v, dict):
            slot = total.setdefault("per_label_ms", {})
            for lbl, ms in v.items():
                slot[lbl] = slot.get(lbl, 0.0) + float(ms)
        elif k == "scale_back":
            continue  # ratio, not additive - summed it's meaningless
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            total[k] = total.get(k, 0.0) + float(v)


def segment_labels_tiled(
    image: Image.Image,
    labels: Iterable[str],
    include_crops: bool = True,
    *,
    threshold: float | None = None,
    mask_threshold: float | None = None,
    min_relative_area: float | None = None,
    tile_size: int | None = None,
    overlap: float = 0.2,
    cancel_check=None,
) -> tuple[list[dict], dict]:
    """Tiled native-resolution variant of `segment_labels` for large
    images. Drop-in: same (detections, timings) return shape, detections
    in original-image pixel coords. Falls back to a single pass when the
    image isn't meaningfully larger than one tile. `tile_size` defaults
    to SAM3_TARGET_LONGEST_EDGE so each crop sails through
    _resize_for_inference untouched - any other value gets rescaled to
    the inference target, which defeats the point. `cancel_check`
    (callable -> bool) aborts between tile passes."""
    # Floor the tile size: an unvalidated tiny/negative value from the
    # jobs API would otherwise explode into 10^5+ SAM3 passes.
    tile = max(256, int(tile_size or SAM3_TARGET_LONGEST_EDGE or 1500))
    W, H = image.size
    if max(W, H) <= tile * 1.25:
        return segment_labels(
            image, labels, include_crops,
            threshold=threshold, mask_threshold=mask_threshold,
            min_relative_area=min_relative_area,
        )
    ov = min(max(float(overlap or 0.0), 0.0), 0.8)
    stride = max(1, int(tile * (1.0 - ov)))
    if cancel_check is not None and cancel_check():
        return [], {
            "cancelled": True, "total_ms": 0.0,
            "scale_back": 1.0, "inference_size": [W, H], "per_label_ms": {},
        }

    t_total = time.perf_counter()
    timings: dict = {"per_label_ms": {}}
    cands: list[tuple] = []  # (detection_in_original_coords, truncated)

    def _collect(dets: list[dict], ox: int, oy: int, tx1: int, ty1: int) -> None:
        edge = 4
        for det in dets:
            if ox or oy or tx1 != W or ty1 != H:
                det = _offset_detection(det, ox, oy)
            b = det["box"]
            truncated = (
                (b[0] - ox <= edge and ox > 0)
                or (b[1] - oy <= edge and oy > 0)
                or (tx1 - b[2] <= edge and tx1 < W)
                or (ty1 - b[3] <= edge and ty1 < H)
            )
            # Edge tiles in the short dimension extend past the image and
            # PIL zero-pads the crop - clamp so a box straddling the real
            # edge can't persist out-of-bounds coords.
            det["box"] = [
                min(max(int(b[0]), 0), W), min(max(int(b[1]), 0), H),
                min(max(int(b[2]), 0), W), min(max(int(b[3]), 0), H),
            ]
            cands.append((det, truncated))

    # Full-frame pass first: objects larger than a tile (or dense groups)
    # only ever appear whole here.
    full_dets, full_tim = segment_labels(
        image, labels, include_crops,
        threshold=threshold, mask_threshold=mask_threshold,
        min_relative_area=min_relative_area,
    )
    _accumulate_timings(timings, full_tim)
    _collect(full_dets, 0, 0, W, H)

    n_tiles = 0
    for oy in _tiled_origins(H, tile, stride):
        for ox in _tiled_origins(W, tile, stride):
            if cancel_check is not None and cancel_check():
                # Overwrite the summed (non-additive) diagnostics so the
                # cancelled-result timings aren't garbage if persisted.
                timings["scale_back"] = 1.0
                timings["inference_size"] = [W, H]
                timings["total_ms"] = (time.perf_counter() - t_total) * 1000.0
                timings["cancelled"] = True
                return [], timings
            crop = image.crop((ox, oy, ox + tile, oy + tile))
            tile_dets, tile_tim = segment_labels(
                crop, labels, include_crops,
                threshold=threshold, mask_threshold=mask_threshold,
                min_relative_area=min_relative_area,
            )
            _accumulate_timings(timings, tile_tim)
            _collect(tile_dets, ox, oy, ox + tile, oy + tile)
            n_tiles += 1

    kept = _merge_tiled_detections(cands)
    if SAM3_TILED_MAX_DETS > 0 and len(kept) > SAM3_TILED_MAX_DETS:
        kept = sorted(kept, key=lambda d: -float(d.get("gd_score") or 0.0))[:SAM3_TILED_MAX_DETS]
        print(f"[sam3-tiled] capped to top {SAM3_TILED_MAX_DETS} detections by score")

    timings["inference_size"] = [W, H]
    timings["scale_back"] = 1.0
    timings["tiles"] = n_tiles
    timings["total_ms"] = (time.perf_counter() - t_total) * 1000.0
    print(f"[sam3-tiled] {W}x{H} → {n_tiles} tiles @{tile}px (+full frame): {len(cands)} raw → {len(kept)} merged")
    return kept, timings


def _segment_one_label_with_cached_vision(
    image_inf: Image.Image,
    label: str,
    vision_embeds,
    original_sizes,
    pixel_values,
    *,
    threshold: float | None = None,
    mask_threshold: float | None = None,
) -> tuple[list[tuple], float, float, int]:
    """Per-label SAM3 forward that re-uses a cached vision_embeds when
    available, then runs the standard post-process + filter passes
    from _segment_one_label. Falls back to a plain forward if the
    cache path errors at runtime (vision_embeds rejected, etc).

    `threshold` / `mask_threshold` override the SAM3_* defaults - the
    interactive click-to-detect path uses lower values so a faint
    object still produces a hit instead of a cancelled gesture."""
    import numpy as np
    W, H = image_inf.size
    min_area_px = max(SAM3_MIN_AREA_PX, int(W * H * SAM3_MIN_AREA_FRAC))
    eff_threshold = SAM3_THRESHOLD if threshold is None else float(threshold)
    eff_mask_threshold = SAM3_MASK_THRESHOLD if mask_threshold is None else float(mask_threshold)

    raw: list[tuple] = []
    predict_ms = 0.0
    post_ms = 0.0
    dropped_tiny = 0

    outputs = None
    text_inputs = None
    try:
        t_predict = time.perf_counter()
        try:
            outputs, text_inputs = _run_text_with_cached_vision(
                label, vision_embeds, original_sizes, pixel_values,
            )
        except Exception as e:
            print(f"[charlie] cached-vision forward failed for label={label!r}: {type(e).__name__}: {e}; falling back to full forward.")
            inputs = _PROCESSOR(images=image_inf, text=label, return_tensors="pt").to(_MODEL.device)
            with torch.inference_mode():
                outputs = _MODEL(**inputs)
            text_inputs = inputs
        _device_sync()
        predict_ms = (time.perf_counter() - t_predict) * 1000.0

        t_post = time.perf_counter()
        target_sizes = text_inputs.get("original_sizes") if text_inputs else None
        if target_sizes is None and original_sizes is not None:
            target_sizes = original_sizes
        if target_sizes is not None and hasattr(target_sizes, "tolist"):
            target_sizes = target_sizes.tolist()
        elif target_sizes is None:
            target_sizes = [[H, W]]

        results = _PROCESSOR.post_process_instance_segmentation(
            outputs,
            threshold=eff_threshold,
            mask_threshold=eff_mask_threshold,
            target_sizes=target_sizes,
        )[0]
        post_ms = (time.perf_counter() - t_post) * 1000.0

        boxes_t = results["boxes"]
        masks_t = results["masks"]
        scores_t = results["scores"]

        hits: list[tuple] = []
        for i in range(len(boxes_t)):
            box = boxes_t[i]
            mask = masks_t[i]
            score = float(scores_t[i])
            box_list = box.tolist() if hasattr(box, "tolist") else list(box)
            if isinstance(mask, torch.Tensor):
                mask_area = int(mask.to(torch.uint8).sum().item())
            else:
                mask_area = int((np.asarray(mask) > 0).sum())
            hits.append((box_list, mask, score, mask_area))

        kept_floor = [h for h in hits if h[3] >= min_area_px]
        dropped_tiny += len(hits) - len(kept_floor)

        if len(kept_floor) > 1:
            max_area = max(h[3] for h in kept_floor)
            rel_floor = int(max_area * SAM3_MIN_RELATIVE_AREA)
            kept_rel = [h for h in kept_floor if h[3] >= rel_floor]
            dropped_tiny += len(kept_floor) - len(kept_rel)
        else:
            kept_rel = kept_floor

        for box_list, mask, score, _area in kept_rel:
            raw.append((box_list, mask, score))
    finally:
        del outputs, text_inputs
        _device_empty_cache()

    return raw, predict_ms, post_ms, dropped_tiny


def segment_point(
    image: Image.Image,
    point: list[float],
    candidate_labels: Iterable[str] | None = None,
    cache_key: str | None = None,
) -> tuple[dict | None, dict]:
    """Click-to-detect via SAM3 text prompts.

    SAM3 is a Promptable Concept Segmentation model, not a SAM2-style
    point promptable segmenter - its processor doesn't expose a
    point-prompt channel and its post-process helper is text-driven.
    So instead of trying to coerce SAM3 into pretending it has point
    prompting, we run it once per candidate label (text-prompt path
    that is known to work), collect every detection, and pick the
    one whose mask actually contains the click pixel.

    Returns (detection, timings_ms). Detection envelope is V2-shaped;
    label is set to whichever candidate produced the matching mask.
    """
    if _MODEL is None or _PROCESSOR is None:
        raise RuntimeError("SAM3 not loaded - call load_sam3() first")

    import numpy as np

    t_total = time.perf_counter()
    timings: dict = {
        "resize_ms": 0.0,
        "sam3_predict_ms": 0.0,
        "sam3_post_ms": 0.0,
        "encode_crops_ms": 0.0,
        "total_ms": 0.0,
        "inference_size": list(image.size),
        "scale_back": 1.0,
        "per_label_ms": {},
    }

    if not (isinstance(point, (list, tuple)) and len(point) == 2):
        timings["total_ms"] = (time.perf_counter() - t_total) * 1000.0
        return None, timings
    clean = _normalise_labels(candidate_labels or [])
    if not clean:
        # No labels means no candidate concepts to segment; bail
        # cleanly so the FE shows "no detection" rather than crashing.
        timings["total_ms"] = (time.perf_counter() - t_total) * 1000.0
        return None, timings

    t_resize = time.perf_counter()
    image_inf, scale_back = _resize_for_inference(image, SAM3_INTERACTIVE_LONGEST_EDGE)
    inf_scale = 1.0 / scale_back if scale_back != 0 else 1.0
    timings["resize_ms"] = (time.perf_counter() - t_resize) * 1000.0
    timings["inference_size"] = list(image_inf.size)
    timings["scale_back"] = scale_back

    W_inf, H_inf = image_inf.size
    px_inf = float(point[0]) * inf_scale
    py_inf = float(point[1]) * inf_scale
    px_int = max(0, min(W_inf - 1, int(round(px_inf))))
    py_int = max(0, min(H_inf - 1, int(round(py_inf))))

    # Vision-encoder cache - encode the image ONCE so the per-label
    # forwards skip the expensive vision pass. ~N× speedup across N
    # candidate labels. Falls through cleanly to per-label full
    # forwards if the cache path doesn't work on this transformers
    # build.
    t_cache = time.perf_counter()
    cached_embeds, cached_sizes, cached_pixels = _get_image_embeds_cached(image_inf, cache_key)
    cache_ms = (time.perf_counter() - t_cache) * 1000.0
    timings["vision_encode_ms"] = cache_ms
    timings["vision_cache_hit"] = cached_embeds is not None

    # Run SAM3 with each candidate label as a text prompt, then keep
    # only detections whose mask actually contains the click pixel.
    # Use the interactive-path thresholds so a low-confidence hit
    # still counts - the user clicked on something, we'd rather
    # return a faint match than cancel.
    all_hits: list[tuple] = []  # (label, box, mask, score, area, mask_np)
    for label in clean:
        t_label = time.perf_counter()
        if cached_embeds is not None:
            raw, predict_ms, post_ms, _ = _segment_one_label_with_cached_vision(
                image_inf, label, cached_embeds, cached_sizes, cached_pixels,
                threshold=SAM3_INTERACTIVE_THRESHOLD,
                mask_threshold=SAM3_INTERACTIVE_MASK_THRESHOLD,
            )
        else:
            raw, predict_ms, post_ms, _ = _segment_one_label(
                image_inf, label,
                threshold=SAM3_INTERACTIVE_THRESHOLD,
                mask_threshold=SAM3_INTERACTIVE_MASK_THRESHOLD,
            )
        timings["sam3_predict_ms"] += predict_ms
        timings["sam3_post_ms"] += post_ms
        timings["per_label_ms"][label] = (time.perf_counter() - t_label) * 1000.0
        for box, mask, score in raw:
            m_np = mask.detach().to(torch.uint8).cpu().numpy() if isinstance(mask, torch.Tensor) else np.asarray(mask)
            if m_np.ndim == 3:
                m_np = m_np.squeeze()
            area = int((m_np > 0).sum())
            all_hits.append((label, box, mask, score, area, m_np))

    # Pick a hit whose mask covers the clicked pixel. If multiple do,
    # prefer the smallest-area mask (most specific) so a click on a
    # small object inside a big one resolves to the small one.
    containing = [h for h in all_hits if h[5][py_int, px_int] > 0]
    if not containing:
        # Fallback: bbox containment (safety net for thin masks).
        containing = [
            h for h in all_hits
            if h[1][0] <= px_inf <= h[1][2] and h[1][1] <= py_inf <= h[1][3]
        ]
    if not containing:
        timings["total_ms"] = (time.perf_counter() - t_total) * 1000.0
        print(
            f"[charlie] segment_point: no mask contains click "
            f"({point[0]:.0f},{point[1]:.0f}) inf=({px_int},{py_int}); "
            f"sam3_total={len(all_hits)} candidates"
        )
        return None, timings

    containing.sort(key=lambda h: h[4])  # smallest first
    label, box, mask, score, _area, _m_np = containing[0]
    box_list = box if isinstance(box, list) else (box.tolist() if hasattr(box, "tolist") else list(box))

    t_crops = time.perf_counter()
    detection = _build_detection(
        image=image,
        box=box_list,
        mask=mask,
        score=score,
        label=label,
        include_crop=True,
        scale_back=scale_back,
    )
    timings["encode_crops_ms"] = (time.perf_counter() - t_crops) * 1000.0

    print(
        f"[charlie] segment_point: click={point} → label={label!r} "
        f"area={_area}px² (over {len(containing)} containing hit(s), "
        f"{len(all_hits)} sam3_total)"
    )

    timings["total_ms"] = (time.perf_counter() - t_total) * 1000.0
    return detection, timings


def _bbox_iou(a: list[float], b: list[float]) -> float:
    """Plain xyxy-bbox IoU. Returns 0 when boxes don't overlap or
    when either box is degenerate (zero area)."""
    ax0, ay0, ax1, ay1 = float(a[0]), float(a[1]), float(a[2]), float(a[3])
    bx0, by0, bx1, by1 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    ix0 = max(ax0, bx0); iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1); iy1 = min(ay1, by1)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    aa = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    ab = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = aa + ab - inter
    return float(inter / union) if union > 0 else 0.0


def segment_box(
    image: Image.Image,
    box: list[float],
    candidate_labels: Iterable[str] | None = None,
    cache_key: str | None = None,
) -> tuple[dict | None, dict]:
    """User-drawn bbox → mask via SAM3 text prompts.

    Same reasoning as segment_point: SAM3's input_boxes channel
    isn't a real "give me the object inside this box" prompt, so we
    run text-prompted SAM3 with each candidate label and pick the
    detection whose bbox has the highest IoU with the user's box.

    Returns (detection, timings_ms). Detection.label is whichever
    candidate produced the winning mask.
    """
    if _MODEL is None or _PROCESSOR is None:
        raise RuntimeError("SAM3 not loaded - call load_sam3() first")

    t_total = time.perf_counter()
    timings: dict = {
        "resize_ms": 0.0,
        "sam3_predict_ms": 0.0,
        "sam3_post_ms": 0.0,
        "encode_crops_ms": 0.0,
        "total_ms": 0.0,
        "inference_size": list(image.size),
        "scale_back": 1.0,
        "per_label_iou": {},
    }

    if not (isinstance(box, (list, tuple)) and len(box) == 4):
        timings["total_ms"] = (time.perf_counter() - t_total) * 1000.0
        return None, timings
    clean = _normalise_labels(candidate_labels or [])
    if not clean:
        timings["total_ms"] = (time.perf_counter() - t_total) * 1000.0
        return None, timings

    t_resize = time.perf_counter()
    image_inf, scale_back = _resize_for_inference(image, SAM3_INTERACTIVE_LONGEST_EDGE)
    inf_scale = 1.0 / scale_back if scale_back != 0 else 1.0
    timings["resize_ms"] = (time.perf_counter() - t_resize) * 1000.0
    timings["inference_size"] = list(image_inf.size)
    timings["scale_back"] = scale_back

    user_box_inf = [
        float(box[0]) * inf_scale,
        float(box[1]) * inf_scale,
        float(box[2]) * inf_scale,
        float(box[3]) * inf_scale,
    ]

    t_cache = time.perf_counter()
    cached_embeds, cached_sizes, cached_pixels = _get_image_embeds_cached(image_inf, cache_key)
    timings["vision_encode_ms"] = (time.perf_counter() - t_cache) * 1000.0
    timings["vision_cache_hit"] = cached_embeds is not None

    best_label: str | None = None
    best_iou = 0.0
    best_box = best_mask = None
    best_score = 0.0
    for label in clean:
        if cached_embeds is not None:
            raw, predict_ms, post_ms, _ = _segment_one_label_with_cached_vision(
                image_inf, label, cached_embeds, cached_sizes, cached_pixels,
                threshold=SAM3_INTERACTIVE_THRESHOLD,
                mask_threshold=SAM3_INTERACTIVE_MASK_THRESHOLD,
            )
        else:
            raw, predict_ms, post_ms, _ = _segment_one_label(
                image_inf, label,
                threshold=SAM3_INTERACTIVE_THRESHOLD,
                mask_threshold=SAM3_INTERACTIVE_MASK_THRESHOLD,
            )
        timings["sam3_predict_ms"] += predict_ms
        timings["sam3_post_ms"] += post_ms
        local_best_iou = 0.0
        for hit_box, hit_mask, hit_score in raw:
            iou = _bbox_iou(user_box_inf, hit_box)
            if iou > local_best_iou:
                local_best_iou = iou
            if iou > best_iou:
                best_iou = iou
                best_label = label
                best_box = hit_box
                best_mask = hit_mask
                best_score = float(hit_score)
        timings["per_label_iou"][label] = round(local_best_iou, 4)

    if best_box is None:
        timings["total_ms"] = (time.perf_counter() - t_total) * 1000.0
        print(f"[charlie] segment_box: no SAM3 detection overlapped user box {box}")
        return None, timings

    box_list = best_box if isinstance(best_box, list) else (best_box.tolist() if hasattr(best_box, "tolist") else list(best_box))

    t_crops = time.perf_counter()
    detection = _build_detection(
        image=image,
        box=box_list,
        mask=best_mask,
        score=best_score,
        label=best_label or "",
        include_crop=True,
        scale_back=scale_back,
    )
    timings["encode_crops_ms"] = (time.perf_counter() - t_crops) * 1000.0

    print(
        f"[charlie] segment_box: user_box={box} → label={best_label!r} "
        f"iou={best_iou:.3f}"
    )

    timings["total_ms"] = (time.perf_counter() - t_total) * 1000.0
    return detection, timings


def classify_box(
    image: Image.Image,
    box: list[float],
    candidate_labels: Iterable[str],
) -> tuple[str | None, float | None, dict]:
    """Pick the best matching label for a user-drawn bbox via SAM3
    text prompts. For each candidate label, runs SAM3 over the whole
    image with that label as the prompt, finds the detection whose
    bbox most overlaps the user's box (IoU), and returns the
    candidate that wins.

    Returns (label, score, timings_ms). label/score may be None if no
    candidate produced a sufficiently overlapping detection."""
    if _MODEL is None or _PROCESSOR is None:
        raise RuntimeError("SAM3 not loaded - call load_sam3() first")

    t_total = time.perf_counter()
    timings = {
        "sam3_predict_ms": 0.0,
        "sam3_post_ms": 0.0,
        "total_ms": 0.0,
        "per_label_iou": {},
    }

    candidates = _normalise_labels(candidate_labels)
    if not candidates or not (isinstance(box, (list, tuple)) and len(box) == 4):
        timings["total_ms"] = (time.perf_counter() - t_total) * 1000.0
        return None, None, timings

    image_inf, scale_back = _resize_for_inference(image)
    inf_scale = 1.0 / scale_back if scale_back != 0 else 1.0
    user_box_inf = [
        float(box[0]) * inf_scale,
        float(box[1]) * inf_scale,
        float(box[2]) * inf_scale,
        float(box[3]) * inf_scale,
    ]

    best_label: str | None = None
    best_score: float | None = None
    best_iou = 0.0
    for label in candidates:
        raw, predict_ms, post_ms, _ = _segment_one_label(image_inf, label)
        timings["sam3_predict_ms"] += predict_ms
        timings["sam3_post_ms"] += post_ms
        # Pick the detection within this label whose bbox overlaps
        # the user's box most. Stash it for IoU comparison across
        # labels.
        local_best_iou = 0.0
        local_best_score = 0.0
        for hit_box, _mask, hit_score in raw:
            iou = _bbox_iou(user_box_inf, hit_box)
            if iou > local_best_iou:
                local_best_iou = iou
                local_best_score = float(hit_score)
        timings["per_label_iou"][label] = round(local_best_iou, 4)
        if local_best_iou > best_iou:
            best_iou = local_best_iou
            best_label = label
            best_score = local_best_score

    timings["total_ms"] = (time.perf_counter() - t_total) * 1000.0
    return best_label, best_score, timings
