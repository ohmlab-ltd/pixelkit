"""Derived projects: one crop image per parent detection, one label each."""
from __future__ import annotations

import hashlib
import uuid
from pathlib import Path


def _det_key(filename: str, label: str, xyxy) -> str:
    """Stable identity for a parent detection; a moved box gets a new key."""
    s = (f"{filename}|{label.strip().lower()}|"
         f"{round(xyxy[0])},{round(xyxy[1])},{round(xyxy[2])},{round(xyxy[3])}")
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def enumerate_crops(parent_manifest: dict, labels) -> list[dict]:
    """Parent detections to crop. An empty `labels` selects every label."""
    from boxes import _image_index, _box_xyxy, _box_label, _validation_rejected
    wanted = {str(l).strip().lower() for l in (labels or []) if str(l).strip()}
    out: list[dict] = []
    for fname, info in _image_index(parent_manifest).items():
        for b in (info.get("boxes") or []):
            if not isinstance(b, dict) or _validation_rejected(b):
                continue
            label = _box_label(b)
            if not label or (wanted and label.strip().lower() not in wanted):
                continue
            xy = _box_xyxy(b)
            if not xy or abs(xy[2] - xy[0]) < 2 or abs(xy[3] - xy[1]) < 2:
                continue
            out.append({"key": _det_key(fname, label, xy),
                        "filename": fname, "label": label, "xyxy": xy, "box": b})
    return out


def _crop_box(xyxy, w: int, h: int, padding: float, square: bool = False):
    """Crop rect for a box; `padding` is a fraction of the box size."""
    x0, y0, x1, y1 = xyxy
    px, py = (x1 - x0) * padding, (y1 - y0) * padding
    bx0, by0, bx1, by1 = x0 - px, y0 - py, x1 + px, y1 + py
    if square:
        side = max(bx1 - bx0, by1 - by0)
        if w and h:
            side = min(side, float(w), float(h))
        ccx, ccy = (bx0 + bx1) / 2.0, (by0 + by1) / 2.0
        cx0 = ccx - side / 2.0
        cy0 = ccy - side / 2.0
        if w:
            cx0 = max(0.0, min(cx0, float(w) - side))
        if h:
            cy0 = max(0.0, min(cy0, float(h) - side))
        cx0 = max(0.0, cx0)
        cy0 = max(0.0, cy0)
        cx1 = cx0 + side
        cy1 = cy0 + side
        if w:
            cx1 = min(cx1, float(w))
        if h:
            cy1 = min(cy1, float(h))
        return cx0, cy0, cx1, cy1
    cx0 = max(0.0, bx0)
    cy0 = max(0.0, by0)
    cx1 = min(float(w) if w else bx1, bx1)
    cy1 = min(float(h) if h else by1, by1)
    return cx0, cy0, cx1, cy1


def resync(child_manifest: dict, parent_manifest: dict,
           parent_imports_dir: Path, child_imports_dir: Path, *, now_iso: str) -> dict:
    """Sync the child's crops to the parent's detections, minus suppressed ones.
    Mutates and returns `child_manifest`; never writes the parent."""
    from PIL import Image, ImageOps

    derived = child_manifest.setdefault("derived", {})
    labels = derived.get("labels") or []
    crop_cfg = derived.get("crop") or {}
    padding = float(crop_cfg.get("padding", 0.15))
    min_size = int(crop_cfg.get("minSize", 0) or 0)
    square = bool(crop_cfg.get("square", False))
    # A fixed size forces square crops so the resize keeps the aspect ratio.
    fixed_size = int(crop_cfg.get("fixedSize", 0) or 0)
    use_square = square or fixed_size > 0
    # "inherit" copies the parent label onto each crop; "new" leaves crops unlabelled.
    label_mode = derived.get("labelMode") or "inherit"
    new_labels = label_mode == "new"
    suppressed = set(derived.get("suppressed") or [])

    specs = {s["key"]: s for s in enumerate_crops(parent_manifest, labels)
             if s["key"] not in suppressed}

    keep: list[dict] = []
    have: set[str] = set()
    for entry in (child_manifest.get("imports") or []):
        if not isinstance(entry, dict):
            continue
        k = (entry.get("derivedFrom") or {}).get("detKey")
        dets0 = entry.get("detections") or []
        if new_labels:
            has_user_work = bool(entry.get("editedBoxes")) or any(
                isinstance(d, dict) and (d.get("label") or d.get("gd_label") or d.get("pred_label"))
                for d in dets0)
            up_to_date = (not dets0) or has_user_work
        else:
            up_to_date = bool(dets0) and isinstance(dets0[0], dict) and isinstance(dets0[0].get("box"), list)
        if k and k in specs and k not in have and up_to_date:
            have.add(k)
            keep.append(entry)
        else:
            fn = entry.get("filename")
            if fn:
                try:
                    (child_imports_dir / fn).unlink(missing_ok=True)
                except Exception:
                    pass

    child_imports_dir.mkdir(parents=True, exist_ok=True)
    for key, s in specs.items():
        if key in have:
            continue
        src = parent_imports_dir / s["filename"]
        if not src.exists():
            continue
        try:
            # Upright, like the boxes.
            with Image.open(src) as raw:
                im = ImageOps.exif_transpose(raw).convert("RGB")
            wp, hp = im.size
            cb = _crop_box(s["xyxy"], wp, hp, padding, square=use_square)
            # Boxes are offset by the same whole-pixel origin the crop starts at.
            ox, oy = int(cb[0]), int(cb[1])
            crop = im.crop((ox, oy, int(round(cb[2])), int(round(cb[3]))))
        except Exception:
            continue
        cw, ch = crop.size
        if cw < 2 or ch < 2:
            continue
        sx = sy = 1.0
        if fixed_size > 0:
            sx, sy = fixed_size / float(cw), fixed_size / float(ch)
            crop = crop.resize((fixed_size, fixed_size), Image.LANCZOS)
        elif min_size > 0 and min(cw, ch) < min_size:
            grow = min_size / float(min(cw, ch))
            nw, nh = max(1, round(cw * grow)), max(1, round(ch * grow))
            sx, sy = nw / float(cw), nh / float(ch)
            crop = crop.resize((nw, nh), Image.LANCZOS)
        cw, ch = crop.size
        out_name = f"{uuid.uuid4().hex}.jpg"
        try:
            crop.save(child_imports_dir / out_name, "JPEG", quality=92)
        except Exception:
            continue
        if new_labels:
            dets: list[dict] = []
        else:
            bx = [round((s["xyxy"][0] - ox) * sx, 1), round((s["xyxy"][1] - oy) * sy, 1),
                  round((s["xyxy"][2] - ox) * sx, 1), round((s["xyxy"][3] - oy) * sy, 1)]
            det = {
                # The UI reads box, gd_label, pred_label; export and training read box_xyxy, label.
                "box": bx, "box_xyxy": bx,
                "gd_label": s["label"], "pred_label": s["label"], "label": s["label"],
                "gd_score": 1.0, "score": 1.0,
            }
            src_box = s.get("box") if isinstance(s.get("box"), dict) else {}
            polys = (src_box.get("mask") or {}).get("polygons") if isinstance(src_box.get("mask"), dict) else None
            if isinstance(polys, list) and polys:
                mapped = []
                for poly in polys:
                    if not isinstance(poly, (list, tuple)):
                        continue
                    pts = []
                    for pt in poly:
                        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                            try:
                                px = min(float(cw), max(0.0, (float(pt[0]) - ox) * sx))
                                py = min(float(ch), max(0.0, (float(pt[1]) - oy) * sy))
                                pts.append([round(px, 1), round(py, 1)])
                            except (TypeError, ValueError):
                                pass
                    if len(pts) >= 3:
                        mapped.append(pts)
                if mapped:
                    det["mask"] = {"polygons": mapped}
            dets = [det]
        keep.append({
            "id": f"imp_{uuid.uuid4().hex[:12]}",
            "filename": out_name, "width": cw, "height": ch,
            # No editedBoxes: the UI reads those in the x0/y0 EditableBox shape.
            "detections": dets,
            "labelled": not new_labels,
            "derivedFrom": {"parentFilename": s["filename"], "detKey": key, "label": s["label"]},
        })

    # A crop keeps its picture's origin, such as an Openverse licence.
    parent_sources = {e.get("filename"): e["source"] for e in parent_manifest.get("imports") or []
                      if isinstance(e, dict) and isinstance(e.get("source"), dict)}
    for entry in keep:
        src = parent_sources.get((entry.get("derivedFrom") or {}).get("parentFilename"))
        if src:
            entry["source"] = {**src, "cropped": True}

    child_manifest["imports"] = keep
    if keep and not any(e.get("filename") == child_manifest.get("cover") for e in keep):
        child_manifest["cover"] = keep[0]["filename"]
        child_manifest["cover_blurhash"] = None
    derived["lastSyncedAt"] = now_iso
    derived["parentUpdatedAt"] = parent_manifest.get("updatedAt")
    return child_manifest


def suppress_detkey(child_manifest: dict, detkey: str) -> None:
    """Record a deleted child crop so a re-sync does not re-create it."""
    derived = child_manifest.setdefault("derived", {})
    sup = derived.get("suppressed")
    if not isinstance(sup, list):
        sup = []
    if detkey and detkey not in sup:
        sup.append(detkey)
    derived["suppressed"] = sup
