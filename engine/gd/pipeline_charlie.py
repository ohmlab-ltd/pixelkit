"""Pipeline Charlie: SAM3 promptable concept segmentation for labelling.

facebook/sam3 is gated on Hugging Face and needs HF_TOKEN. SAM3_MODEL_ID
overrides the checkpoint.
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

# Defaults marked LOCKED are tuned together.

# SAM3's defaults; noise is filtered after the model.
SAM3_THRESHOLD = float(os.environ.get("SAM3_THRESHOLD", "0.5"))           # LOCKED
SAM3_MASK_THRESHOLD = float(os.environ.get("SAM3_MASK_THRESHOLD", "0.5")) # LOCKED
# Thresholds for click-to-detect and add-box.
SAM3_INTERACTIVE_THRESHOLD = float(os.environ.get("SAM3_INTERACTIVE_THRESHOLD", "0.05"))
SAM3_INTERACTIVE_MASK_THRESHOLD = float(os.environ.get("SAM3_INTERACTIVE_MASK_THRESHOLD", "0.2"))

# A mask under max(SAM3_MIN_AREA_PX, SAM3_MIN_AREA_FRAC x image area) px is dropped.
SAM3_MIN_AREA_FRAC = float(os.environ.get("SAM3_MIN_AREA_FRAC", "0.0001"))  # LOCKED
SAM3_MIN_AREA_PX = int(os.environ.get("SAM3_MIN_AREA_PX", "64"))            # LOCKED

# A mask under this fraction of its label's largest mask is dropped.
SAM3_MIN_RELATIVE_AREA = float(os.environ.get("SAM3_MIN_RELATIVE_AREA", "0.05"))  # LOCKED

# The same against the largest mask of any label. 0 disables.
SAM3_MIN_GLOBAL_RELATIVE_AREA = float(os.environ.get("SAM3_MIN_GLOBAL_RELATIVE_AREA", "0"))  # LOCKED

# Longest edge in px that images are resized to for SAM3. 0 disables.
SAM3_TARGET_LONGEST_EDGE = int(os.environ.get("SAM3_TARGET_LONGEST_EDGE", "1500"))  # LOCKED
# Longest edge for click-to-detect and box-to-mask.
SAM3_INTERACTIVE_LONGEST_EDGE = int(os.environ.get("SAM3_INTERACTIVE_LONGEST_EDGE", "1008"))

# Douglas-Peucker epsilon as a fraction of contour perimeter. 0 disables.
SAM3_POLY_SIMPLIFY_EPS = float(os.environ.get("SAM3_POLY_SIMPLIFY_EPS", "0.0015"))  # LOCKED

# A detection's sub-polygons under this fraction of its largest are dropped.
SAM3_SUBMASK_MIN_RELATIVE_AREA = float(os.environ.get("SAM3_SUBMASK_MIN_RELATIVE_AREA", "0.05"))

# Morphological opening kernel in px; breaks thin bridges. 0 disables.
SAM3_MASK_OPEN_PX = int(os.environ.get("SAM3_MASK_OPEN_PX", "5"))

# Mask blur radius in px before contouring. 0 disables.
SAM3_MASK_SMOOTH_PX = int(os.environ.get("SAM3_MASK_SMOOTH_PX", "0"))

# Catmull-Rom points added between each vertex pair. 0 disables.
SAM3_POLY_SUBDIVIDE_SAMPLES = int(os.environ.get("SAM3_POLY_SUBDIVIDE_SAMPLES", "0"))

# Vertex smoothing sigma, in vertices. 0 disables.
SAM3_POLY_SMOOTH_SIGMA = float(os.environ.get("SAM3_POLY_SMOOTH_SIGMA", "0"))
SAM3_POLY_SMOOTH_PASSES = int(os.environ.get("SAM3_POLY_SMOOTH_PASSES", "3"))

# Periodic cubic B-spline samples per segment. 0 disables.
SAM3_POLY_SPLINE_SAMPLES = int(os.environ.get("SAM3_POLY_SPLINE_SAMPLES", "0"))

# Resample to one vertex per this many px of perimeter. 0 disables.
SAM3_POLY_RESAMPLE_PX_SPACING = float(os.environ.get("SAM3_POLY_RESAMPLE_PX_SPACING", "0"))
SAM3_POLY_RESAMPLE_MIN = int(os.environ.get("SAM3_POLY_RESAMPLE_MIN", "32"))
SAM3_POLY_RESAMPLE_MAX = int(os.environ.get("SAM3_POLY_RESAMPLE_MAX", "240"))

# Corners turning more than this many degrees stay sharp. 0 disables.
SAM3_CORNER_ANGLE_THRESHOLD_DEG = float(os.environ.get("SAM3_CORNER_ANGLE_THRESHOLD_DEG", "0"))
# Largest turn-angle window, in vertices each side, capped at n//6.
SAM3_CORNER_WINDOW = int(os.environ.get("SAM3_CORNER_WINDOW", "4"))
# Corner non-maximum suppression radius, in vertices.
SAM3_CORNER_NMS_RADIUS = int(os.environ.get("SAM3_CORNER_NMS_RADIUS", "2"))
# Polygons with fewer vertices skip corner detection.
SAM3_CORNER_MIN_VERTICES = int(os.environ.get("SAM3_CORNER_MIN_VERTICES", "8"))
# Corners farther than this many mask px from the smoothed polygon stay rounded.
SAM3_CORNER_MAX_DIST_PX = float(os.environ.get("SAM3_CORNER_MAX_DIST_PX", "12"))
# Radius in mask px within which smoothed vertices are replaced by the corner.
SAM3_CORNER_CARVE_RADIUS_PX = float(os.environ.get("SAM3_CORNER_CARVE_RADIUS_PX", "10"))

# Context-aware smoothing: the Gaussian sigma falls with local curvature.

# Catmull-Rom samples per segment before context-aware smoothing. 0 disables.
SAM3_CONTEXT_SUBDIVIDE = int(os.environ.get("SAM3_CONTEXT_SUBDIVIDE", "0"))
# Sigma in vertices at zero turn. 0 disables.
SAM3_CONTEXT_SMOOTH_MAX_SIGMA = float(os.environ.get("SAM3_CONTEXT_SMOOTH_MAX_SIGMA", "0"))
# Floor in degrees of the corner threshold max(this, average turn x sensitivity).
SAM3_CONTEXT_CORNER_DEG = float(os.environ.get("SAM3_CONTEXT_CORNER_DEG", "60"))
SAM3_CONTEXT_CORNER_SENSITIVITY = float(os.environ.get("SAM3_CONTEXT_CORNER_SENSITIVITY", "3.0"))
# Doubles vertices turning this many degrees after smoothing. 0 disables.
SAM3_CONTEXT_DUPE_DEG = float(os.environ.get("SAM3_CONTEXT_DUPE_DEG", "0"))


_MODEL = None
_PROCESSOR = None


def _model_device() -> "torch.device":
    """The model's device. The helpers below key on it, not on CUDA being
    present: a CUDA sync with the model on the CPU creates a context on GPU 0."""
    try:
        return torch.device(getattr(_MODEL, "device", None) or "cpu")
    except (TypeError, RuntimeError):
        return torch.device("cpu")


def _device_sync() -> None:
    """Barrier for accurate timings on the model's backend."""
    dev = _model_device()
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    elif dev.type == "mps":
        torch.mps.synchronize()


def _device_empty_cache() -> None:
    dev = _model_device()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    elif dev.type == "mps":
        torch.mps.empty_cache()


def is_loaded() -> bool:
    return _MODEL is not None and _PROCESSOR is not None


def load_sam3(device: str = "cuda"):
    """Load SAM3 and return (model, processor). cuda and mps load fp16, cpu fp32."""
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


def _detect_sharp_corners(
    polygon,
    threshold_deg: float,
    window: int = 3,
    nms_radius: int = 2,
    min_vertices: int = 8,
) -> list[tuple[float, float]]:
    """Vertices turning more than `threshold_deg` at any window from 1 to
    `window`, thinned by non-maximum suppression."""
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
    """Squared distance from (px, py) to segment ab."""
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
    """Carve corners back into a smoothed polygon. A corner within
    `max_dist_px` takes the place of the nearest run of vertices within
    `carve_radius_px`, doubled so the frontend draws a cusp."""
    if not corner_positions or len(polygon) < 2:
        return polygon
    poly = list(polygon)
    max_dist_sq = max_dist_px * max_dist_px
    carve_radius_sq = carve_radius_px * carve_radius_px

    for cx, cy in corner_positions:
        n = len(poly)
        if n < 2:
            continue

        best_seg = 0
        best_seg_d2 = float("inf")
        for i in range(n):
            j = (i + 1) % n
            d2 = _dist_to_segment_sq(cx, cy, poly[i], poly[j])
            if d2 < best_seg_d2:
                best_seg_d2 = d2
                best_seg = i
        if best_seg_d2 > max_dist_sq:
            continue

        within = []
        for idx, p in enumerate(poly):
            dx = float(p[0]) - cx
            dy = float(p[1]) - cy
            if dx * dx + dy * dy <= carve_radius_sq:
                within.append(idx)

        c = [round(cx, 1), round(cy, 1)]
        if not within:
            poly.insert(best_seg + 1, c)
            poly.insert(best_seg + 2, c)
            continue

        within_set = set(within)
        runs: list[list[int]] = []
        seen_idx: set[int] = set()
        for idx in within:
            if idx in seen_idx:
                continue
            run = [idx]
            seen_idx.add(idx)
            j = (idx + 1) % n
            while j in within_set and j not in seen_idx:
                run.append(j)
                seen_idx.add(j)
                j = (j + 1) % n
            j = (idx - 1) % n
            while j in within_set and j not in seen_idx:
                run.insert(0, j)
                seen_idx.add(j)
                j = (j - 1) % n
            runs.append(run)
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

        run_set = set(target_run)
        if all(target_run[k] + 1 == target_run[k + 1] for k in range(len(target_run) - 1)):
            first = target_run[0]
            last = target_run[-1]
            new_poly = poly[:first] + [c, c] + poly[last + 1:]
        else:
            # The run wraps past the end, so the pair goes at the start.
            kept = [poly[i] for i in range(n) if i not in run_set]
            new_poly = [c, c] + kept
        poly = new_poly

    return poly


def _resample_polygon_arc_length(polygon, n_target: int):
    """Resample `polygon` to `n_target` points evenly spaced along its
    perimeter. Only reduces the vertex count."""
    n = len(polygon)
    if n < 4 or n_target < 4 or n_target >= n:
        return polygon
    import numpy as np
    pts = np.asarray(polygon, dtype=np.float64)
    closed = np.vstack([pts, pts[0:1]])
    diffs = np.diff(closed, axis=0)
    seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = float(cum[-1])
    if total <= 0:
        return polygon
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
    """Closed uniform cubic B-spline approximating `polygon`, sampled
    `samples_per_segment` times per vertex. The curve does not pass
    through the vertices."""
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
    """Gaussian smoothing of a closed polygon. Sigma falls linearly from
    `max_sigma` at no turn to 0 at max(corner_threshold_deg, average turn x
    sensitivity)."""
    import math
    n = len(polygon)
    if n < 5 or max_sigma <= 0 or corner_threshold_deg <= 0:
        return [list(p) for p in polygon]

    eff_window = max(1, min(measure_window, n // 6))

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

    avg_turn = sum(turn_angles) / max(1, len(turn_angles))
    eff_threshold = max(corner_threshold_deg, avg_turn * sensitivity)

    sigmas = [
        max(0.0, max_sigma * (1.0 - ta / eff_threshold))
        for ta in turn_angles
    ]

    # Kernels stop short of corner-ish vertices so straight edges do not bulge.
    corner_floor = eff_threshold * 0.5
    corner_indices = [i for i in range(n) if turn_angles[i] >= corner_floor]
    if corner_indices:
        corner_sorted = sorted(corner_indices)
        nearest_corner_dist = [n] * n
        for i in range(n):
            best = n
            for ci in corner_sorted:
                d = abs(i - ci)
                d = min(d, n - d)
                if d < best:
                    best = d
            nearest_corner_dist[i] = best
    else:
        nearest_corner_dist = [n // 2] * n

    smoothed: list[list[float]] = []
    for i in range(n):
        sigma = sigmas[i]
        if sigma < 0.05:
            smoothed.append(
                [round(float(polygon[i][0]), 1), round(float(polygon[i][1]), 1)]
            )
            continue
        radius = max(1, int(round(sigma * 3)))
        radius = min(radius, max(1, n // 4))
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
    """Double every vertex turning at least `threshold_deg`; the frontend
    draws a doubled vertex as a cusp."""
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
    """1-D Gaussian filter along a closed polygon's vertices. Keeps the
    vertex count and returns float coordinates."""
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
    """Insert `samples_per_segment` Catmull-Rom points between each vertex
    pair, keeping every original vertex."""
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
    """Absolute polygon area by the shoelace formula."""
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
    """Convert a 2D mask to a list of (x, y) polygons, in mask pixels."""
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
                if simplified.shape[0] >= 3:
                    c = simplified
        polys.append([[int(p[0][0]), int(p[0][1])] for p in c])

    corner_positions_per_poly: list[list[tuple[float, float]]] = []

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
                    f"of max={int(max_area)} sq px)"
                )
            polys = kept

    # After the outlier filter, so corner_positions_per_poly[i] matches polys[i].
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

    # Context-aware smoothing.
    if SAM3_CONTEXT_SUBDIVIDE > 0 and polys:
        polys = [_catmull_rom_subdivide(p, SAM3_CONTEXT_SUBDIVIDE) for p in polys]

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

    if SAM3_CONTEXT_DUPE_DEG > 0 and polys:
        polys = [
            _duplicate_sharp_vertices(p, SAM3_CONTEXT_DUPE_DEG)
            for p in polys
        ]

    # Multi-step smoothing, off by default.
    if SAM3_POLY_SUBDIVIDE_SAMPLES > 0 and polys:
        polys = [_catmull_rom_subdivide(p, SAM3_POLY_SUBDIVIDE_SAMPLES) for p in polys]

    if SAM3_POLY_SMOOTH_SIGMA > 0 and polys and SAM3_POLY_SMOOTH_PASSES > 0:
        for _ in range(SAM3_POLY_SMOOTH_PASSES):
            polys = [_gaussian_smooth_polygon(p, SAM3_POLY_SMOOTH_SIGMA) for p in polys]

    if SAM3_POLY_SPLINE_SAMPLES > 0 and polys:
        polys = [_periodic_cubic_bspline(p, SAM3_POLY_SPLINE_SAMPLES) for p in polys]

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
    """Base64 JPEG of `image` cropped to `box`, the long edge at most `max_side`."""
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
    """Pack one SAM3 hit into the V2 detection dict, other models' fields empty.

    `scale_back` maps the box and mask from inference to `image` coordinates.
    """
    polys = _mask_to_polygons(mask)
    if scale_back != 1.0 and polys:
        polys = [
            [[round(float(p[0]) * scale_back, 1), round(float(p[1]) * scale_back, 1)] for p in poly]
            for poly in polys
        ]
    box_orig = [int(round(c * scale_back)) for c in box[:4]]
    return {
        "box": box_orig,
        "mask": {"polygons": polys} if polys else None,
        "embedding": [],
        "gd_label": label,
        "gd_variant": label,
        "gd_score": float(score),
        "crop_jpg_b64": _crop_to_jpg_b64(image, box_orig) if include_crop else "",
    }


def _resize_for_inference(image: Image.Image, target: int | None = None) -> tuple[Image.Image, float]:
    """Scale `image` up or down so its longest edge is `target` or
    SAM3_TARGET_LONGEST_EDGE px. Returns (image_inf, scale_back), where
    scale_back maps inference coordinates to original ones.
    """
    if (target or SAM3_TARGET_LONGEST_EDGE) <= 0:
        return image, 1.0
    W, H = image.size
    longest = max(W, H)
    if longest == (target or SAM3_TARGET_LONGEST_EDGE):
        return image, 1.0
    inf_scale = (target or SAM3_TARGET_LONGEST_EDGE) / longest
    new_size = (max(1, int(round(W * inf_scale))), max(1, int(round(H * inf_scale))))
    return image.resize(new_size, Image.LANCZOS), 1.0 / inf_scale


# LRU of SAM3 vision-encoder outputs for the interactive tools, keyed by
# caller id and inference size. Entries hold GPU tensors.
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
    """Run only the SAM3 vision encoder, for the vision_embeds argument of
    Sam3Model.forward. Returns (vision_embeds, original_sizes, pixel_values),
    or three Nones on error."""
    if _MODEL is None or _PROCESSOR is None:
        return None, None, None
    try:
        proc_inputs = _PROCESSOR(images=image, return_tensors="pt").to(_MODEL.device)
    except Exception as e:
        print(f"[charlie] vision-embed cache: processor pre-pass failed: {e}")
        return None, None, None

    pixel_values = proc_inputs.get("pixel_values")
    original_sizes = proc_inputs.get("original_sizes")

    # The method name varies across transformers versions.
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
    """SAM3 text-prompt forward on cached vision embeddings. Returns
    (outputs, text_inputs)."""
    text_inputs = _PROCESSOR(text=label, return_tensors="pt").to(_MODEL.device)
    # Some HF processors add pixel_values to text-only inputs.
    text_inputs.pop("pixel_values", None)
    text_inputs.pop("original_sizes", None)
    forward_kwargs = dict(text_inputs)
    if original_sizes is not None:
        forward_kwargs["original_sizes"] = original_sizes
    if vision_embeds is not None:
        forward_kwargs["vision_embeds"] = vision_embeds
    else:
        if pixel_values is not None:
            forward_kwargs["pixel_values"] = pixel_values
    with torch.inference_mode():
        outputs = _MODEL(**forward_kwargs)
    return outputs, text_inputs


def _keep_relative(hits: list[tuple], min_relative: float, ref_score: float | None) -> list[tuple]:
    """Drop (box, mask, score, area) hits under `min_relative` x the largest area.

    Only hits scoring at least `ref_score` set the largest, or all when none
    does: at the interactive threshold SAM3 also returns a near-whole-image mask.
    """
    if len(hits) <= 1 or min_relative <= 0:
        return hits
    ref = [h[3] for h in hits if ref_score is None or h[2] >= ref_score]
    floor = int(max(ref or [h[3] for h in hits]) * min_relative)
    return [h for h in hits if h[3] >= floor]


def _segment_one_label(
    image: Image.Image,
    label: str,
    *,
    threshold: float | None = None,
    mask_threshold: float | None = None,
    min_relative_area: float | None = None,
    relative_ref_score: float | None = None,
) -> tuple[list[tuple], float, float, int]:
    """One SAM3 forward pass for one text prompt.

    Returns (raw, predict_ms, post_ms, dropped_tiny); raw holds (box, mask,
    score) in `image` pixels. The keyword arguments override SAM3_* defaults.
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

        kept_floor = [h for h in hits if h[3] >= min_area_px]
        n_dropped_floor = len(hits) - len(kept_floor)
        dropped_tiny += n_dropped_floor

        kept_rel = _keep_relative(kept_floor, eff_min_relative, relative_ref_score)
        n_dropped_rel = len(kept_floor) - len(kept_rel)
        dropped_tiny += n_dropped_rel

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
    """Run SAM3 for each label and concatenate the detections.

    The keyword arguments override SAM3_* defaults for this call. Returns
    (detections, timings_ms).
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
                min_relative_area=min_relative_area,
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
            image=image,                    # original resolution
            box=box,                        # inference coordinates
            mask=mask,                      # inference resolution
            score=score,
            label=label,
            include_crop=include_crops,
            scale_back=scale_back,
        ))
    timings["encode_crops_ms"] += (time.perf_counter() - t_crops) * 1000.0

    timings["total_ms"] = (time.perf_counter() - t_total) * 1000.0
    return detections, timings


# Tiled inference at native resolution, for small objects in large images.

SAM3_TILED_MAX_DETS = int(os.environ.get("SAM3_TILED_MAX_DETS", "500"))


def _tiled_origins(size: int, tile: int, stride: int) -> list[int]:
    """Window origins covering `size`, the last one flush with the end."""
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
    """Shift a tile detection's box and polygons by the tile origin. The
    crop thumbnail needs no change."""
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
    """Greedy same-label merge of (det, truncated) candidates: untruncated
    first, then score, then area. A kept box mostly containing two or more
    same-label boxes under half its area is dropped."""
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
                    continue
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
            continue  # a ratio, not additive
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
    """`segment_labels` on native-resolution tiles plus the full frame, boxes
    in original-image pixels. The default `tile_size`, SAM3_TARGET_LONGEST_EDGE,
    skips _resize_for_inference. `cancel_check` (callable -> bool) aborts
    between tiles."""
    # The jobs API does not validate tile_size.
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
    cands: list[tuple] = []  # (detection, truncated)

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
            # Edge tiles extend past the image, where PIL zero-pads the crop.
            det["box"] = [
                min(max(int(b[0]), 0), W), min(max(int(b[1]), 0), H),
                min(max(int(b[2]), 0), W), min(max(int(b[3]), 0), H),
            ]
            cands.append((det, truncated))

    # Objects larger than a tile appear whole only in the full-frame pass.
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
    print(f"[sam3-tiled] {W}x{H} -> {n_tiles} tiles @{tile}px (+full frame): {len(cands)} raw -> {len(kept)} merged")
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
    min_relative_area: float | None = None,
    relative_ref_score: float | None = None,
) -> tuple[list[tuple], float, float, int]:
    """`_segment_one_label` on cached vision embeddings, with a plain forward
    if the cached path raises."""
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

        eff_min_relative = SAM3_MIN_RELATIVE_AREA if min_relative_area is None else float(min_relative_area)
        kept_rel = _keep_relative(kept_floor, eff_min_relative, relative_ref_score)
        dropped_tiny += len(kept_floor) - len(kept_rel)

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
    """Click-to-detect. SAM3 has no point prompt, so each candidate label runs
    as a text prompt and the smallest mask containing the click wins. Returns
    (detection or None, timings_ms)."""
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

    t_cache = time.perf_counter()
    cached_embeds, cached_sizes, cached_pixels = _get_image_embeds_cached(image_inf, cache_key)
    cache_ms = (time.perf_counter() - t_cache) * 1000.0
    timings["vision_encode_ms"] = cache_ms
    timings["vision_cache_hit"] = cached_embeds is not None

    all_hits: list[tuple] = []  # (label, box, mask, score, area, mask_np)
    for label in clean:
        t_label = time.perf_counter()
        if cached_embeds is not None:
            raw, predict_ms, post_ms, _ = _segment_one_label_with_cached_vision(
                image_inf, label, cached_embeds, cached_sizes, cached_pixels,
                threshold=SAM3_INTERACTIVE_THRESHOLD,
                mask_threshold=SAM3_INTERACTIVE_MASK_THRESHOLD,
                relative_ref_score=SAM3_THRESHOLD,
            )
        else:
            raw, predict_ms, post_ms, _ = _segment_one_label(
                image_inf, label,
                threshold=SAM3_INTERACTIVE_THRESHOLD,
                mask_threshold=SAM3_INTERACTIVE_MASK_THRESHOLD,
                relative_ref_score=SAM3_THRESHOLD,
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

    containing = [h for h in all_hits if h[5][py_int, px_int] > 0]
    if not containing:
        # Box containment, for thin masks.
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

    containing.sort(key=lambda h: h[4])  # smallest area first
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
        f"[charlie] segment_point: click={point} -> label={label!r} "
        f"area={_area} sq px (over {len(containing)} containing hit(s), "
        f"{len(all_hits)} sam3_total)"
    )

    timings["total_ms"] = (time.perf_counter() - t_total) * 1000.0
    return detection, timings


def _inside_fraction(inner: list[float], outer: list[float]) -> float:
    """Share of xyxy box `inner`'s area that lies inside `outer`."""
    iw = max(0.0, min(float(inner[2]), float(outer[2])) - max(float(inner[0]), float(outer[0])))
    ih = max(0.0, min(float(inner[3]), float(outer[3])) - max(float(inner[1]), float(outer[1])))
    area = max(0.0, float(inner[2]) - float(inner[0])) * max(0.0, float(inner[3]) - float(inner[1]))
    return (iw * ih) / area if area > 0.0 else 0.0


def _bbox_iou(a: list[float], b: list[float]) -> float:
    """IoU of two xyxy boxes."""
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
    """Box-to-mask. SAM3's input_boxes is not an object-in-box prompt, so each
    candidate label runs as a text prompt and the hit best matching the box
    wins. Returns (detection or None, timings_ms)."""
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
                relative_ref_score=SAM3_THRESHOLD,
            )
        else:
            raw, predict_ms, post_ms, _ = _segment_one_label(
                image_inf, label,
                threshold=SAM3_INTERACTIVE_THRESHOLD,
                mask_threshold=SAM3_INTERACTIVE_MASK_THRESHOLD,
                relative_ref_score=SAM3_THRESHOLD,
            )
        timings["sam3_predict_ms"] += predict_ms
        timings["sam3_post_ms"] += post_ms
        local_best_iou = 0.0
        for hit_box, hit_mask, hit_score in raw:
            iou = _bbox_iou(user_box_inf, hit_box)
            if iou > local_best_iou:
                local_best_iou = iou
            # A match overlaps the box or sits inside it; others are neighbours.
            if iou < 0.3 and _inside_fraction(hit_box, user_box_inf) < 0.8:
                continue
            if iou > best_iou:
                best_iou = iou
                best_label = label
                best_box = hit_box
                best_mask = hit_mask
                best_score = float(hit_score)
        timings["per_label_iou"][label] = round(local_best_iou, 4)

    if best_box is None:
        timings["total_ms"] = (time.perf_counter() - t_total) * 1000.0
        print(f"[charlie] segment_box: no SAM3 detection matches user box {box}")
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
        f"[charlie] segment_box: user_box={box} -> label={best_label!r} "
        f"iou={best_iou:.3f}"
    )

    timings["total_ms"] = (time.perf_counter() - t_total) * 1000.0
    return detection, timings


def classify_box(
    image: Image.Image,
    box: list[float],
    candidate_labels: Iterable[str],
) -> tuple[str | None, float | None, dict]:
    """Label for a drawn box: the candidate whose SAM3 hit has the highest IoU
    with it. Returns (label, score, timings_ms), label and score None when no
    hit matches."""
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
        local_best_iou = 0.0
        local_best_score = 0.0
        for hit_box, _mask, hit_score in raw:
            iou = _bbox_iou(user_box_inf, hit_box)
            if iou < 0.3 and _inside_fraction(hit_box, user_box_inf) < 0.8:
                continue  # a neighbour, not the boxed object
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
