"""Box/label resolution helpers shared by derived datasets and export.

Extracted from the deleted training module — pure dict utilities, the
canonical per-image view across V1 (results/editedBoxes) and V2 (imports)
manifest shapes."""
from __future__ import annotations

def _safe_label(s) -> str:
    return (s or "").strip().lower() if isinstance(s, str) else ""

def _box_xyxy(b: dict) -> tuple[float, float, float, float] | None:
    if all(b.get(k) is not None for k in ("x0", "y0", "x1", "y1")):
        try:
            return float(b["x0"]), float(b["y0"]), float(b["x1"]), float(b["y1"])
        except (TypeError, ValueError):
            pass
    v = b.get("box_xyxy")
    if isinstance(v, list) and len(v) == 4:
        try:
            return float(v[0]), float(v[1]), float(v[2]), float(v[3])
        except (TypeError, ValueError):
            pass
    # Segmentation-only detection (e.g. a SAM "road" polygon with no stored
    # bbox): derive the enclosing box from the mask polygons so a detector can
    # still train on it. Without this, seg-class boxes are silently dropped.
    polys = (b.get("mask") or {}).get("polygons")
    if isinstance(polys, list) and polys:
        xs: list[float] = []
        ys: list[float] = []
        for poly in polys:
            if not isinstance(poly, (list, tuple)):
                continue
            for pt in poly:
                if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    try:
                        xs.append(float(pt[0]))
                        ys.append(float(pt[1]))
                    except (TypeError, ValueError):
                        pass
        if xs and ys and max(xs) > min(xs) and max(ys) > min(ys):
            return min(xs), min(ys), max(xs), max(ys)
    return None

def _vlm_rejected(b: dict) -> bool:
    v = b.get("validation")
    return isinstance(v, dict) and v.get("match") is False

def _box_label(b: dict) -> str:
    """Resolve a box's label across every naming convention in play — matches
    the export's `_box_label` (and the FE viewer's predLabel||gdLabel). V2 auto
    detections carry the label under predLabel/gd_label, NOT `label`."""
    for key in ("label", "predLabel", "pred_label", "gd_label", "gdLabel", "gd_variant"):
        v = b.get(key)
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return ""

def _image_index(manifest: dict) -> dict[str, dict]:
    """Unified per-image view across V1 (results[] + top-level editedBoxes) and
    V2 (imports[] with per-entry width/height/editedBoxes/detections). Returns
    {filename: {"width", "height", "boxes"}}. This is the SAME resolution the
    export uses, so training sees exactly the labels the user sees — for V2,
    `editedBoxes` wins when set, else auto `detections` (minus rejected)."""
    out: dict[str, dict] = {}
    for r in (manifest.get("results") or []):
        img = r.get("image")
        sz = r.get("size") or {}
        if not img:
            continue
        out[img] = {"width": int(sz.get("width") or 0),
                    "height": int(sz.get("height") or 0), "boxes": []}
    for img_name, boxes in (manifest.get("editedBoxes") or {}).items():
        if not isinstance(boxes, list):
            continue
        slot = out.setdefault(img_name, {"width": 0, "height": 0, "boxes": []})
        slot["boxes"].extend(b for b in boxes if isinstance(b, dict))
    for entry in (manifest.get("imports") or []):
        if not isinstance(entry, dict):
            continue
        fname = entry.get("filename")
        if not fname:
            continue
        slot = out.setdefault(fname, {"width": 0, "height": 0, "boxes": []})
        w = int(entry.get("width") or 0)
        h = int(entry.get("height") or 0)
        if w and not slot["width"]:
            slot["width"] = w
        if h and not slot["height"]:
            slot["height"] = h
        edited = entry.get("editedBoxes")
        if isinstance(edited, list) and (entry.get("editedBoxesSet") or edited):
            slot["boxes"].extend(b for b in edited if isinstance(b, dict))
        else:
            dets = entry.get("detections") or []
            if isinstance(dets, list):
                slot["boxes"].extend(
                    b for b in dets if isinstance(b, dict) and not b.get("rejected"))
    return out
