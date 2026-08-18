"""Image preprocessing helpers used during dataset import."""
from __future__ import annotations

import io

import numpy as np
from PIL import Image as PILImage

# Per-channel difference allowed between a "border" pixel and the
# inferred border colour. Generous enough to absorb JPEG compression
# noise on a clean border (~5-10 per channel) without crossing into
# real content (typical content edges shift by 30+).
DEFAULT_TOLERANCE = 12
# Fraction of a row/column that must look like border for it to be
# treated as border. 0.985 = up to 1.5% non-matching pixels per line,
# enough to ignore the odd compressed pixel on the border edge but
# still bail out at the first real content row (a single content
# object usually paints a full line of non-border pixels).
DEFAULT_MIN_MATCH_FRAC = 0.985


def crop_border(
    img: PILImage.Image,
    tolerance: int = DEFAULT_TOLERANCE,
    min_match_frac: float = DEFAULT_MIN_MATCH_FRAC,
) -> PILImage.Image:
    """Crop a uniform border off an image.

    O(h*w) - one numpy diff and two axis reductions, no Python loops.
    Bails out fast if the four corners disagree on colour (which they
    will for borderless images), so the common case is just a corner-
    sample plus a variance check.

    Returns the input image unchanged when no border is detected so
    callers can short-circuit re-encoding via `cropped is img`.
    """
    if img.width < 8 or img.height < 8:
        return img

    # Convert in RGB so we don't have to special-case L/P/RGBA. We
    # don't need alpha for the border test - the border check is on
    # colour, and most "bordered" images are RGB anyway.
    rgb = img if img.mode == "RGB" else img.convert("RGB")
    arr = np.asarray(rgb, dtype=np.int16)  # int16 so abs(diff) won't overflow uint8
    h, w = arr.shape[:2]

    # Sample all four corners and take the median per channel. Robust
    # to one corner being noise - only fails if 3+ corners disagree,
    # which means the image probably doesn't have a uniform border at
    # all and we want to bail.
    corners = np.stack([
        arr[0, 0], arr[0, -1], arr[-1, 0], arr[-1, -1],
    ])
    border_rgb = np.median(corners, axis=0)
    # Quick reject: if the corners themselves disagree by more than
    # `tolerance`, there's no uniform border to crop.
    corner_spread = int(np.max(np.abs(corners - border_rgb)))
    if corner_spread > tolerance:
        return img

    # Per-pixel max-channel difference from the inferred border colour.
    # Single h*w array, then a boolean is_border mask, then per-row
    # and per-col mean (each O(h*w)). Two final argmaxes are O(h+w).
    diff = np.max(np.abs(arr - border_rgb), axis=2)
    is_border = diff <= tolerance
    row_frac = is_border.mean(axis=1)
    col_frac = is_border.mean(axis=0)

    row_content = row_frac < min_match_frac
    col_content = col_frac < min_match_frac
    if not row_content.any() or not col_content.any():
        # Whole image is border-coloured - nothing to crop to.
        return img

    top = int(np.argmax(row_content))
    bottom = h - 1 - int(np.argmax(row_content[::-1]))
    left = int(np.argmax(col_content))
    right = w - 1 - int(np.argmax(col_content[::-1]))

    if top == 0 and left == 0 and bottom == h - 1 and right == w - 1:
        return img

    new_w = right - left + 1
    new_h = bottom - top + 1
    # Don't crop down to a sliver - if our detection collapsed the
    # image to <10% of either side, something went wrong (probably a
    # photo with a uniform sky and a tiny subject) and the safer
    # default is to keep the original.
    if new_w < w * 0.1 or new_h < h * 0.1:
        return img

    return img.crop((left, top, right + 1, bottom + 1))


def is_blank_image_bytes(data: bytes, *, min_dim: int = 96, min_stdev: float = 6.0) -> tuple[bool, str | None]:
    """True when the bytes decode to something the user won't be able
    to label - too small, all transparent, or a uniform block colour.

    `min_stdev` is the floor on the per-channel standard deviation
    measured over a 32×32 downsample. Real photos clear this trivially
    (JPEG noise alone produces std ≥ 10); placeholders, all-black
    "image not found" stubs, and fully transparent PNGs all fall well
    below it.

    Used at import time so we don't write a blank into the project
    bucket. Returns (is_blank, reason_or_none) - the reason string is
    surfaced in the rejection summary the UI shows the user.
    """
    try:
        with PILImage.open(io.BytesIO(data)) as img:
            img.load()
            if img.width < min_dim or img.height < min_dim:
                return True, f"tiny ({img.width}x{img.height})"
            # Cheap variance probe on a 32×32 downsample - keeps the
            # cost bounded regardless of source resolution. RGBA is
            # flattened onto a checkerboard so a fully-transparent
            # PNG collapses to a uniform image and gets caught.
            sample = img.convert("RGBA").resize((32, 32), PILImage.BILINEAR)
            arr = np.asarray(sample, dtype=np.float32)
            alpha = arr[..., 3:4] / 255.0
            rgb = arr[..., :3] * alpha + 128.0 * (1.0 - alpha)
            stdev = float(rgb.reshape(-1, 3).std(axis=0).mean())
            if stdev < min_stdev:
                return True, f"flat (stdev {stdev:.2f})"
    except Exception as e:
        return True, f"decode failed ({e})"
    return False, None


def maybe_crop_border_bytes(data: bytes, ctype: str | None) -> tuple[bytes, dict]:
    """Convenience wrapper for the URL-import flow.

    Decodes once, runs `crop_border`, and either returns the original
    bytes unchanged (no border found - no re-encode needed) or the
    cropped image re-encoded in its source format. Returns
    (bytes, {width, height}).
    """
    with PILImage.open(io.BytesIO(data)) as img:
        img.load()
        original_size = {"width": img.width, "height": img.height}
        cropped = crop_border(img)
        if cropped is img:
            return data, original_size

        fmt = (img.format or "JPEG").upper()
        out_buf = io.BytesIO()
        save_kwargs: dict = {}
        if fmt in ("JPEG", "JPG"):
            save_kwargs["quality"] = 95
            save_kwargs["optimize"] = True
            # JPEG can't carry alpha - flatten just in case the
            # cropped image somehow ended up with a mode mismatch.
            if cropped.mode != "RGB":
                cropped = cropped.convert("RGB")
            fmt = "JPEG"
        elif fmt == "WEBP":
            save_kwargs["quality"] = 95
        cropped.save(out_buf, format=fmt, **save_kwargs)
        return out_buf.getvalue(), {"width": cropped.width, "height": cropped.height}
