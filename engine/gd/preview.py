"""Labelled-preview renderer (pure PIL). Extracted from the deleted
GroundingDINO module — no model dependencies."""
from PIL import Image, ImageDraw, ImageFilter

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

