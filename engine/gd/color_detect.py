"""Colour-aware label parsing for the GD prompt.

GroundingDINO struggles when a colour adjective is bolted onto an
object noun ("red car" detects fewer cars than "car"; the colour word
pulls the prompt embedding off the object class). Rather than fight
that, we:

  1. Strip the colour word out of the tag before it goes to GD,
     handing the model the bare object noun.
  2. Run the per-mask colour classifier on each resulting detection.
  3. Recombine: if the mask's dominant colour matches the user's
     requested colour, label the detection with the original
     "red car" tag. If not, drop the detection — the user asked for
     red cars, not all cars.

Colour classification samples pixels inside the polygon mask, converts
to HSV, and bins each pixel into one of a small set of named colours
using hand-tuned ranges. Cheap (one OpenCV pass per box) and good
enough for object-level "is this red or not?" decisions; fine-grained
shade discrimination isn't the point.
"""
from __future__ import annotations

from collections import Counter

import cv2
import numpy as np
from PIL import Image as PILImage


# Colours we'll recognise in user tags. Lowercase, single-word.
# Each one either has its own HSV bin (see _classify_pixel_hsv) or
# aliases into one — see COLOR_ALIASES. Ambiguous fashion shades
# ("magenta", "lavender") are intentionally left out: more false
# strips than they prevent.
COLOR_WORDS = {
    # Primary chromatic
    "red", "orange", "yellow", "green", "blue", "purple",
    "pink", "brown",
    # Achromatic
    "black", "white", "grey", "gray",
    # Common metallic / neutral terms — aliased to existing bins
    # because HSV can't really tell them apart from the bin they
    # alias to, but recognising the word lets us strip it from the
    # GD prompt cleanly.
    "silver", "gold", "navy", "beige", "cream", "tan", "maroon",
    "teal", "cyan",
}

# Synonym map for variants we want to collapse into the canonical bin.
# Silver / gold / navy / etc. fall into the closest visual bin so the
# post-segmentation colour match still fires correctly.
COLOR_ALIASES = {
    "gray": "grey",
    "silver": "grey",     # silver looks like a desaturated mid-grey
    "gold": "yellow",     # warm metallic yellow
    "navy": "blue",       # dark blue
    "beige": "brown",     # light brown / tan range
    "cream": "white",     # off-white
    "tan": "brown",
    "maroon": "red",      # dark red
    "teal": "green",      # blue-leaning green; HSV puts it on green side
    "cyan": "blue",       # close to blue in HSV
}


def parse_color_label(tag: str) -> tuple[str | None, str]:
    """Strip a leading or trailing colour word from `tag`.

    Returns (colour, base_object) where colour is one of `COLOR_WORDS`
    canonicalised through `COLOR_ALIASES`, or (None, tag) if no colour
    word is present.

    Only matches whole tokens — "red car" → ("red", "car"),
    "redcar" → (None, "redcar"). Multi-word tags are split on
    whitespace; the first matching token gets stripped, everything
    else is rejoined.
    """
    if not tag:
        return None, tag
    parts = tag.strip().lower().split()
    if not parts:
        return None, tag
    for i, p in enumerate(parts):
        if p in COLOR_WORDS:
            color = COLOR_ALIASES.get(p, p)
            base = " ".join(parts[:i] + parts[i + 1:]).strip()
            if not base:
                # User typed just "red" — there's no object noun to
                # detect, so don't try to be clever. Keep the tag
                # intact and skip colour processing.
                return None, tag
            return color, base
    return None, tag


# HSV bins. Hue ranges in OpenCV are [0, 180); saturation and value
# in [0, 255]. Tuned by eye on real photographic crops — a bit looser
# on saturation than a textbook chart so faded / shadowed pixels still
# fall into the right bin.
def _classify_pixel_hsv(h: int, s: int, v: int) -> str:
    # Achromatic axis first — black, white, and grey have low saturation.
    if v < 50:
        return "black"
    if s < 35 and v > 200:
        return "white"
    if s < 35:
        return "grey"
    # Brown sits in the orange-hue band but at lower value (dark).
    if (h <= 20) and v < 140 and s > 60:
        return "brown"
    # Hue-driven chromatic bins.
    if h < 8 or h >= 168:
        return "red"
    if h < 22:
        return "orange"
    if h < 33:
        return "yellow"
    if h < 85:
        return "green"
    if h < 130:
        return "blue"
    if h < 150:
        return "purple"
    return "pink"


def _build_mask_from_polygons(polygons: list, height: int, width: int) -> np.ndarray:
    """Rasterise polygon list (each [[x,y], ...]) onto an HxW uint8
    mask. Empty polygon list → empty mask."""
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


def mask_dominant_color(
    image_pil: PILImage.Image,
    polygons: list,
    *,
    min_pixels: int = 32,
    sample_cap: int = 20000,
) -> str | None:
    """Return the dominant named colour inside the polygon mask, or
    None if the mask is empty / too small.

    Sampling: full mask if it has fewer than `sample_cap` pixels,
    otherwise an evenly-strided subsample. Subsampling caps cost on
    large masks (a 4 K full-frame mask has ~8 M pixels — pointless to
    classify every one when 20 K already pin the histogram).
    """
    if image_pil is None or not polygons:
        return None
    rgb = np.asarray(image_pil.convert("RGB"))
    H, W = rgb.shape[:2]
    mask = _build_mask_from_polygons(polygons, H, W)
    ys, xs = np.where(mask > 0)
    if len(ys) < min_pixels:
        return None
    if len(ys) > sample_cap:
        stride = len(ys) // sample_cap
        ys = ys[::stride]
        xs = xs[::stride]
    pixels = rgb[ys, xs].reshape(-1, 1, 3).astype(np.uint8)
    hsv = cv2.cvtColor(pixels, cv2.COLOR_RGB2HSV).reshape(-1, 3)
    counts: Counter[str] = Counter()
    for h, s, v in hsv:
        counts[_classify_pixel_hsv(int(h), int(s), int(v))] += 1
    if not counts:
        return None
    # Strip achromatic dominance when there's a chromatic runner-up
    # with reasonable share — a slightly faded red car shouldn't be
    # called "grey" just because most of the body is shaded.
    dominant, dom_count = counts.most_common(1)[0]
    if dominant in {"black", "white", "grey"} and len(counts) > 1:
        chromatic = [(c, n) for c, n in counts.items() if c not in {"black", "white", "grey"}]
        if chromatic:
            chromatic.sort(key=lambda x: -x[1])
            top_color, top_count = chromatic[0]
            if top_count >= 0.30 * dom_count:
                return top_color
    return dominant


def color_matches(observed: str | None, requested: str) -> bool:
    """Loose equality between two colour bins. Exact match for now;
    placeholder for fuzzy neighbours (red↔pink, blue↔purple) if we
    ever need them."""
    if not observed or not requested:
        return False
    return observed == requested
