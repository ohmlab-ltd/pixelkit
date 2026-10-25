"""The boxes an image's augmented copies were made from. Copies made from boxes
the image no longer has are out of date; training and exports leave them out."""
from __future__ import annotations

import hashlib
import json

_LABEL_KEYS = ("label", "predLabel", "pred_label", "gd_label", "gdLabel", "gd_variant")


def _label(d: dict) -> str:
    for k in _LABEL_KEYS:
        v = d.get(k)
        if v is not None and str(v).strip():
            return str(v).strip().lower()
    return ""


def _xyxy(d: dict) -> list[float] | None:
    if all(k in d for k in ("x0", "y0", "x1", "y1")):
        v = [d["x0"], d["y0"], d["x1"], d["y1"]]
    elif isinstance(d.get("box_xyxy"), (list, tuple)) and len(d["box_xyxy"]) == 4:
        v = d["box_xyxy"]
    elif isinstance(d.get("box"), (list, tuple)) and len(d["box"]) == 4:
        v = d["box"]
    else:
        return None
    try:
        return [round(float(x), 1) for x in v]
    except (TypeError, ValueError):
        return None


def _mask_digest(d: dict) -> str:
    mask = d.get("mask")
    polys = mask.get("polygons") if isinstance(mask, dict) else None
    if not isinstance(polys, list) or not polys:
        return ""
    try:
        flat = [[[round(float(p[0]), 1), round(float(p[1]), 1)] for p in poly
                 if isinstance(p, (list, tuple)) and len(p) >= 2]
                for poly in polys if isinstance(poly, list)]
    except (TypeError, ValueError):
        return "unreadable"
    return hashlib.sha1(json.dumps(flat).encode()).hexdigest()[:16]


def _order(boxes: list[dict]) -> list[dict]:
    return sorted(boxes, key=lambda b: json.dumps(b, sort_keys=True))


def source_boxes(imp: dict) -> list[dict]:
    """The import's boxes as the augmenter reads them: label, box and a digest
    of the mask, in a fixed order. Edited boxes once edited (an explicit empty
    edit means none), else the detections the labeller did not reject."""
    edited = imp.get("editedBoxes")
    if isinstance(edited, list) and (edited or imp.get("editedBoxesSet")):
        dets = edited
    else:
        dets = [d for d in imp.get("detections") or []
                if not (isinstance(d, dict) and d.get("rejected"))]
    return _order([{"label": _label(d), "box": _xyxy(d), "mask": _mask_digest(d)}
                   for d in dets if isinstance(d, dict)])


def is_current(anno: dict, imp: dict) -> bool:
    """True when the copies in `anno` (an annotations.json) were made from the
    boxes `imp` has now. Copies without a recorded source count as current."""
    made_from = anno.get("source") if isinstance(anno, dict) else None
    if not isinstance(made_from, list):
        return True
    return _order([b for b in made_from if isinstance(b, dict)]) == source_boxes(imp)


def relabel_source(anno: dict, relabel) -> bool:
    """Apply a label rename to the recorded boxes, as it is applied to the
    image and its copies; `relabel` returns the new name or None to drop the
    box. True when the record changed."""
    made_from = anno.get("source")
    if not isinstance(made_from, list):
        return False
    kept, changed = [], False
    for b in made_from:
        if not isinstance(b, dict):
            continue
        new = relabel(str(b.get("label") or ""))
        if new is None:
            changed = True
            continue
        new = str(new).strip().lower()
        if new != b.get("label"):
            b = {**b, "label": new}
            changed = True
        kept.append(b)
    if changed:
        anno["source"] = _order(kept)
    return changed
