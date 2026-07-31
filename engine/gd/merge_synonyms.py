"""Synonym + spatial-adjacency merger using mask alignment.

Two kinds of merges happen here:

  - **Synonym duplicates** of the same physical object — drop the
    lower-score detection, keep the higher one's mask as-is. Three
    signals can fire (strong bbox overlap, mask IoU, mask containment).

  - **Spatially adjacent same-class instances** — combine the
    detections by taking the bbox + polygon union. Same class only,
    fires when the masks are within ~30 pixels of each other and
    the bboxes meaningfully overlap. Catches the "two segments of
    one road that the model split into separate detections" case.

Either case ends with the lower-score detection removed; the kept
one may have its bbox/polygons rewritten when the merge was a union.
"""
from __future__ import annotations

import re
from typing import Any


_PHRASE_PARSE_RE = re.compile(r"^(.+?)\s*\(([0-9.]+)\)\s*$")


def _rasterize_mask(mask: dict | None, W: int, H: int):
    """Burn a polygon mask payload onto a (H, W) bool raster. None when
    the mask is missing or empty so callers can short-circuit math."""
    if not mask:
        return None
    polys = mask.get("polygons") if isinstance(mask, dict) else None
    if not polys:
        return None
    try:
        from PIL import Image as _Image, ImageDraw as _ImageDraw
        import numpy as _np
        img = _Image.new("L", (W, H), 0)
        draw = _ImageDraw.Draw(img)
        for poly in polys:
            if poly and len(poly) >= 3:
                pts = [(float(x), float(y)) for x, y in poly]
                draw.polygon(pts, fill=1)
        return _np.asarray(img, dtype=bool)
    except Exception as e:
        print(f"[mask-merge] rasterize failed: {e}")
        return None


def _bbox_iou(bi: list[float], bj: list[float]) -> tuple[float, float]:
    """Returns (intersection, iou) for two xyxy boxes. inter==0 means
    they don't overlap and the caller can skip mask work entirely."""
    ix0 = max(bi[0], bj[0]); iy0 = max(bi[1], bj[1])
    ix1 = min(bi[2], bj[2]); iy1 = min(bi[3], bj[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0:
        return 0.0, 0.0
    ai = max(1.0, (bi[2] - bi[0]) * (bi[3] - bi[1]))
    aj = max(1.0, (bj[2] - bj[0]) * (bj[3] - bj[1]))
    union = ai + aj - inter
    return inter, (inter / union if union > 0 else 0.0)


def _bbox_union(bi: list[float], bj: list[float]) -> list[float]:
    return [min(bi[0], bj[0]), min(bi[1], bj[1]), max(bi[2], bj[2]), max(bi[3], bj[3])]


def _close_gap_to_polygons(
    raster_a,
    raster_b,
    image_size: tuple[int, int],
    max_bridge_px: int,
):
    """Combine two pre-rasterised binary masks into a single closed
    polygon shape, bridging the gap between disjoint regions.

    Adaptive: measures the actual minimum distance between the two
    regions and dilates by just enough to bridge it (plus a small
    margin), then erodes most of the bulge back. The result is one
    polygon covering both regions with a slim connecting strip,
    rather than a fat halo around each input mask.

    Returns None if the regions are too far apart (> `max_bridge_px`)
    or the contour extraction fails — caller falls back to raw
    polygon concatenation.
    """
    try:
        import numpy as _np
        import cv2 as _cv2
        from scipy.ndimage import binary_dilation as _dilate
        from scipy.ndimage import binary_erosion as _erode
        from scipy.ndimage import distance_transform_edt as _edt
    except Exception:
        return None

    W, H = image_size
    if raster_a is None or raster_b is None:
        return None
    if not raster_a.any() or not raster_b.any():
        return None

    # Measure the actual gap between the two regions, in pixels —
    # the smallest distance from any pixel in B to A. This lets us
    # dilate by gap/2 + margin instead of the worst-case bound.
    try:
        dist_from_a = _edt(~raster_a)
        b_dists = dist_from_a[raster_b]
        if b_dists.size == 0:
            return None
        gap = float(b_dists.min())
    except Exception as ex:
        print(f"[mask-merge] distance transform failed: {ex}")
        return None

    if gap > max_bridge_px:
        # Caller's adjacency check passed (using its own dilation
        # threshold), but the actual minimum gap exceeds the bridge
        # budget — bail out and let the caller keep the raw union.
        return None

    # Bridge: dilate the union by just enough to reach across the
    # gap. A small post-erode trims jaggies without erasing the
    # bridge itself (bridge survives as long as erode < n - gap/2).
    n = max(2, int(gap / 2) + 4)
    e = max(1, min(3, n // 8))
    arr = raster_a | raster_b
    try:
        dilated = _dilate(arr, iterations=n)
        closed = _erode(dilated, iterations=e) if e > 0 else dilated
    except Exception as ex:
        print(f"[mask-merge] dilate/erode failed: {ex}")
        return None
    # Always preserve the original mask pixels — closing may shave
    # thin protrusions, and we never want to lose original detail.
    closed = closed | arr

    closed_u8 = (closed.astype(_np.uint8)) * 255
    try:
        # CHAIN_APPROX_SIMPLE keeps all corner points (only drops
        # redundant collinear ones), which matches the mask area
        # accurately. TC89 simplification was lossier than wanted.
        contours, _hier = _cv2.findContours(
            closed_u8, _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE
        )
    except Exception as e:
        print(f"[mask-merge] findContours failed: {e}")
        return None

    out_polys: list = []
    for cnt in contours:
        if cnt is None or len(cnt) < 3:
            continue
        # cv2 contours are (N, 1, 2); flatten to (N, 2) and round
        # to a few decimal places to keep the manifest tidy.
        pts = cnt.reshape(-1, 2).astype(float)
        out_polys.append([[round(float(x), 2), round(float(y), 2)] for x, y in pts])

    return out_polys or None


def merge_synonyms_by_mask(
    boxes: list[list[float]],
    phrases: list[str],
    variants: list[str],
    masks: list[Any],
    image_size: tuple[int, int],
    *,
    # Synonym-duplicate thresholds (drop-only).
    mask_iou_thresh: float = 0.40,
    mask_containment_thresh: float = 0.70,
    bbox_iou_strong: float = 0.80,
    bbox_iou_min: float = 0.30,
    # Spatial-adjacency union-merge thresholds. The defaults were
    # tuned against `roads/test8.jpg` where SAM produced two road
    # masks ~50 px apart at the closest point and the user expected
    # them merged because they're visibly the same road.
    adjacency_bbox_iou_min: float = 0.20,
    adjacency_dilation_px: int = 60,
):
    """Combine same-class detections that the model produced as duplicates
    or as adjacent fragments of one object. Returns
    `(boxes, phrases, variants, masks)` with the lower-quality entries
    removed; the surviving entries may have updated bbox/mask geometry
    when a union merge fired.

    Five merge signals — first one to match a pair fires:

      1. **Strong bbox IoU** (>= bbox_iou_strong) → drop lower-score.
         Two same-class boxes overlapping this much are almost
         always the same instance even if SAM gave us slightly
         different masks.
      2. **Mask IoU** (>= mask_iou_thresh) with bbox IoU >= bbox_iou_min
         → drop lower-score. Primary synonym-dup signal.
      3. **Mask containment** (>= mask_containment_thresh) with bbox
         IoU >= bbox_iou_min → drop lower-score. Catches coarse +
         tight versions of the same object.
      4. **Spatial adjacency** — bbox IoU >= adjacency_bbox_iou_min,
         masks within `adjacency_dilation_px` of each other (proxied
         via binary dilation), but mask IoU is low. → **Union-merge**:
         combine the bboxes, append polygon lists, drop the lower
         entry. Catches the "two halves of one road" case where the
         detector split a single contiguous object across two
         predictions.
      5. **No-mask fallback** (rare) — bbox IoU >= 0.65 → drop lower.

    Boxes that overlap a corner only (low bbox IoU) and whose masks
    are clearly disjoint shapes (e.g. two adjacent dogs) survive as
    separate detections."""
    n = len(boxes)
    if n <= 1:
        return boxes, phrases, variants, masks

    # Mutate-friendly copies — Signal 4 unions can rewrite kept entries.
    boxes = [list(b) for b in boxes]
    masks = list(masks)

    # Score per phrase decides who survives a merge.
    scores: list[float] = []
    canon: list[str] = []
    for p in phrases:
        m = _PHRASE_PARSE_RE.match(p)
        if m:
            canon.append(m.group(1).strip().lower())
            scores.append(float(m.group(2)))
        else:
            canon.append(p.strip().lower())
            scores.append(0.5)

    W, H = image_size
    rasters: list = [None] * n
    areas: list[float] = [0.0] * n
    try:
        import numpy as _np
        for i in range(n):
            r = _rasterize_mask(masks[i], W, H)
            rasters[i] = r
            if r is not None:
                areas[i] = float(_np.count_nonzero(r))
    except Exception as e:
        print(f"[mask-merge] preflight failed: {e}")

    bbox_areas = [max(1.0, (b[2] - b[0]) * (b[3] - b[1])) for b in boxes]
    order = sorted(range(n), key=lambda i: -scores[i])
    keep = [True] * n
    dropped = 0
    unioned = 0

    try:
        import numpy as _np
        try:
            from scipy.ndimage import binary_dilation as _dilate
            _have_dilate = True
        except Exception:
            _have_dilate = False
            print("[mask-merge] scipy.ndimage unavailable, skipping spatial-adjacency signal")

        for ii in range(n):
            i = order[ii]
            if not keep[i]:
                continue
            for jj in range(ii + 1, n):
                j = order[jj]
                if not keep[j]:
                    continue
                if canon[i] != canon[j]:
                    continue

                bbox_inter, bbox_iou = _bbox_iou(boxes[i], boxes[j])
                if bbox_inter <= 0:
                    continue

                # Signal 1: very high bbox overlap → drop lower.
                if bbox_iou >= bbox_iou_strong:
                    keep[j] = False
                    dropped += 1
                    continue

                ri, rj = rasters[i], rasters[j]

                # Signal 5 fallback: no-mask both → strict bbox IoU.
                if ri is None or rj is None:
                    if bbox_iou >= 0.65:
                        keep[j] = False
                        dropped += 1
                    continue

                inter_pixels = int(_np.logical_and(ri, rj).sum())
                union_pixels = int(_np.logical_or(ri, rj).sum())
                mask_iou = inter_pixels / union_pixels if union_pixels > 0 else 0.0
                contain_i = inter_pixels / areas[i] if areas[i] > 0 else 0.0
                contain_j = inter_pixels / areas[j] if areas[j] > 0 else 0.0

                if bbox_iou >= bbox_iou_min:
                    # Signal 2: mask IoU clears the threshold.
                    if mask_iou >= mask_iou_thresh:
                        keep[j] = False
                        dropped += 1
                        continue
                    # Signal 3: containment.
                    if max(contain_i, contain_j) >= mask_containment_thresh:
                        keep[j] = False
                        dropped += 1
                        continue

                # Signal 4: spatial adjacency — bbox bias overlaps,
                # masks don't but they're close. Union-merge into i.
                if not _have_dilate:
                    continue
                if bbox_iou < adjacency_bbox_iou_min:
                    continue
                # Dilate the SMALLER mask (cheaper) and check whether
                # it now overlaps the larger one.
                if areas[i] >= areas[j]:
                    big, small = ri, rj
                else:
                    big, small = rj, ri
                try:
                    dil = _dilate(small, iterations=int(adjacency_dilation_px))
                except Exception as e:
                    print(f"[mask-merge] dilation failed: {e}")
                    continue
                if not bool((dil & big).any()):
                    continue

                # Merge by union into i. Update bbox, polygon list,
                # raster, area caches so subsequent pairs see the new
                # combined region.
                boxes[i] = _bbox_union(boxes[i], boxes[j])
                bbox_areas[i] = max(1.0, (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]))
                mi = masks[i] or {}
                mj = masks[j] or {}
                polys_i = list(mi.get("polygons") or [])
                polys_j = list(mj.get("polygons") or [])
                # Close the gap so the two regions become a single
                # connected polygon. Uses the actual gap distance (via
                # distance transform) to dilate just enough, so the
                # bulge stays minimal. Falls back to raw concat if
                # the bridge can't be drawn.
                closed_polys = _close_gap_to_polygons(
                    ri, rj, (W, H), int(adjacency_dilation_px),
                )
                merged_polys = closed_polys if closed_polys else polys_i + polys_j
                masks[i] = {**mi, "polygons": merged_polys}
                # Recompute the kept raster so signals 1-3 against
                # later pairs see the union shape.
                new_raster = _rasterize_mask(masks[i], W, H)
                if new_raster is not None:
                    rasters[i] = new_raster
                    areas[i] = float(_np.count_nonzero(new_raster))
                keep[j] = False
                unioned += 1
    except Exception as e:
        print(f"[mask-merge] failed: {e}")
        return boxes, phrases, variants, masks

    if dropped == 0 and unioned == 0:
        return boxes, phrases, variants, masks
    print(
        f"[mask-merge] dropped {dropped} synonym duplicate(s), "
        f"union-merged {unioned} adjacent pair(s) of {n} detections"
    )

    keep_idx = [i for i in range(n) if keep[i]]
    return (
        [boxes[i] for i in keep_idx],
        [phrases[i] for i in keep_idx],
        [variants[i] for i in keep_idx],
        [masks[i] for i in keep_idx],
    )
