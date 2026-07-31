"""Derived ("child") projects.

A child project's dataset is per-detection CROPS of a PARENT project: one crop
image per detection, exactly one label each, for a user-selected set of labels.
A person-with-PPE parent image becomes (in the child) one "person" crop, one
"glove" crop, one "helmet" crop, … — one image, one box, one label.

Linking is ONE-WAY (parent → child):
  - the parent's current detections drive the child (re-sync = a diff),
  - a parent image/detection removed → its child crop is removed,
  - a child crop deleted by the user is remembered (`suppressed` tombstone) so a
    re-sync does NOT resurrect it,
  - nothing here ever writes the parent (sync only READS the parent manifest +
    its on-disk originals and writes the CHILD).

This module is deliberately free of any `server` import (server orchestrates:
it loads/saves manifests + passes the imports dirs). The tiny manifest readers
are imported lazily from `training` so the child sees exactly the boxes the
user sees / the trainer trains on.
"""
from __future__ import annotations

import hashlib
import uuid
from pathlib import Path


def _det_key(filename: str, label: str, xyxy) -> str:
    """Stable identity for a parent detection (V2 boxes carry no id). A moved
    box becomes a new key (old crop removed, new one added) — acceptable."""
    s = (f"{filename}|{label.strip().lower()}|"
         f"{round(xyxy[0])},{round(xyxy[1])},{round(xyxy[2])},{round(xyxy[3])}")
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def enumerate_crops(parent_manifest: dict, labels) -> list[dict]:
    """The parent detections to crop: label in `labels` (empty = all), not
    VLM-rejected, with a real box. Returns [{key, filename, label, xyxy}]."""
    from training import _image_index, _box_xyxy, _box_label, _vlm_rejected
    wanted = {str(l).strip().lower() for l in (labels or []) if str(l).strip()}
    out: list[dict] = []
    for fname, info in _image_index(parent_manifest).items():
        for b in (info.get("boxes") or []):
            if not isinstance(b, dict) or _vlm_rejected(b):
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
    """Box → crop rect with `padding` (fraction of box size) of context,
    clamped to the image.

    When `square` (ROI mode): the rect is forced to a 1:1 square centred on the
    box — side = the longer padded edge, capped to the image's shorter side and
    shifted inward so it always stays fully inside the image. The output crop is
    therefore exactly square (no letterboxing needed), so derived crops are
    square ROIs rather than long thin slivers."""
    x0, y0, x1, y1 = xyxy
    px, py = (x1 - x0) * padding, (y1 - y0) * padding
    bx0, by0, bx1, by1 = x0 - px, y0 - py, x1 + px, y1 + py
    if square:
        side = max(bx1 - bx0, by1 - by0)
        if w and h:
            side = min(side, float(w), float(h))  # can't exceed the image
        ccx, ccy = (bx0 + bx1) / 2.0, (by0 + by1) / 2.0
        cx0 = ccx - side / 2.0
        cy0 = ccy - side / 2.0
        # Shift the square inward so it stays fully on-image (keeps it square).
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
    """Make the child's crops match the parent's current selected detections
    (minus suppressed). Mutates + returns `child_manifest`; adds/removes crop
    files on disk. Never touches the parent."""
    from PIL import Image

    derived = child_manifest.setdefault("derived", {})
    labels = derived.get("labels") or []
    crop_cfg = derived.get("crop") or {}
    padding = float(crop_cfg.get("padding", 0.15))
    min_size = int(crop_cfg.get("minSize", 0) or 0)
    square = bool(crop_cfg.get("square", False))
    # Optional fixed size: every crop is resized to exactly fixed_size ×
    # fixed_size so all derived images share one size. It only makes sense from a
    # square region (else the resize distorts the aspect), so it forces square
    # cropping. 0 = off.
    fixed_size = int(crop_cfg.get("fixedSize", 0) or 0)
    use_square = square or fixed_size > 0
    # Label source: "inherit" (default) carries the parent label onto each crop;
    # "new" leaves crops UNLABELLED for the user to label fresh, while still
    # remembering the parent label (derivedFrom.label) so the UI can show it.
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
            # "new" crops are a blank ROI canvas. A fresh one (no detections) or
            # one the user has already drawn/labelled is up-to-date and kept
            # as-is. A crop from before this behaviour existed still carries the
            # inherited box (label-less, no user edits) — treat THAT as stale so
            # a re-sync re-derives it back to blank, deleting the unwanted
            # box/mask/segmented cover without touching real user work.
            has_user_work = bool(entry.get("editedBoxes")) or any(
                isinstance(d, dict) and (d.get("label") or d.get("gd_label") or d.get("pred_label"))
                for d in dets0)
            up_to_date = (not dets0) or has_user_work
        else:
            # "Current" inherit crop = has the FE `box` key. Older shapes (x0/y0,
            # box_xyxy-only, or editedBoxes-based) are outdated → dropped +
            # re-derived fresh below, so a re-sync heals them.
            up_to_date = bool(dets0) and isinstance(dets0[0], dict) and isinstance(dets0[0].get("box"), list)
        if k and k in specs and k not in have and up_to_date:
            have.add(k)
            keep.append(entry)            # still valid + current shape — keep as-is
        else:                              # orphaned / dup / deselected / outdated → drop + delete file
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
            im = Image.open(src).convert("RGB")
        except Exception:
            continue
        wp, hp = im.size
        cb = _crop_box(s["xyxy"], wp, hp, padding, square=use_square)
        crop = im.crop((int(cb[0]), int(cb[1]), int(round(cb[2])), int(round(cb[3]))))
        cw, ch = crop.size
        if cw < 2 or ch < 2:
            continue
        # Scale small crops up so every derived image is at least `min_size` on
        # its shorter side (preserves aspect). Box + mask coords scale with it.
        scale = 1.0
        if fixed_size > 0:
            # Fixed size: resize every (square) crop to exactly fixed_size ×
            # fixed_size so all derived images are the same size. Uniform scale
            # (square in → square out), so box + mask coords scale cleanly.
            scale = fixed_size / float(max(1, min(cw, ch)))
            crop = crop.resize((fixed_size, fixed_size), Image.LANCZOS)
            cw, ch = crop.size
        elif min_size > 0 and min(cw, ch) < min_size:
            scale = min_size / float(min(cw, ch))
            crop = crop.resize((max(1, round(cw * scale)), max(1, round(ch * scale))), Image.LANCZOS)
            cw, ch = crop.size
        out_name = f"{uuid.uuid4().hex}.jpg"
        try:
            crop.save(child_imports_dir / out_name, "JPEG", quality=92)
        except Exception:
            continue
        # In "new" label mode the crop is a BLANK ROI canvas: no inherited box,
        # mask, segmented cover or label is copied across — the user draws and
        # labels everything from scratch. Only derivedFrom.label is kept, purely
        # as the muted reference the UI shows. In "inherit" mode we carry the
        # parent box + label + segmentation onto the crop as before.
        if new_labels:
            dets: list[dict] = []
        else:
            bx = [round((s["xyxy"][0] - cb[0]) * scale, 1), round((s["xyxy"][1] - cb[1]) * scale, 1),
                  round((s["xyxy"][2] - cb[0]) * scale, 1), round((s["xyxy"][3] - cb[1]) * scale, 1)]
            det = {
                # The FE unwraps a manifest detection via `box` (NOT box_xyxy) and
                # reads the label from gd_label/pred_label (NOT `label`). The
                # export/trainer read `box_xyxy` + label/pred_label. Set ALL of
                # them so boxes, labels and masks render in every code path.
                "box": bx, "box_xyxy": bx,
                "gd_label": s["label"], "pred_label": s["label"], "label": s["label"],
                "gd_score": 1.0, "score": 1.0,
            }
            # Carry the segmentation through: offset each polygon point into the
            # crop's coordinate space (clamped to the crop) so masks pull through.
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
                                px = min(float(cw), max(0.0, (float(pt[0]) - cb[0]) * scale))
                                py = min(float(ch), max(0.0, (float(pt[1]) - cb[1]) * scale))
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
            # detections only (box_xyxy shape). We deliberately DON'T set
            # editedBoxes/editedBoxesSet: the FE reads editedBoxes as the
            # EditableBox (x0/y0) shape, not the box_xyxy detection shape, so
            # setting them hid the boxes+labels. The gallery + editor + trainer
            # all fall back to `detections`, which carries box_xyxy + label + mask.
            # In "new" mode this is [] — a blank ROI image with nothing on it.
            "detections": dets,
            # "new" mode crops are unlabelled until the user assigns a label.
            "labelled": not new_labels,
            # derivedFrom.label is ALWAYS the original parent label — the source
            # of truth the UI surfaces as a reference, even in "new" mode where
            # the active crop label is blank.
            "derivedFrom": {"parentFilename": s["filename"], "detKey": key, "label": s["label"]},
        })

    child_manifest["imports"] = keep
    # Cover: first crop, so the workspace card has a thumbnail.
    if keep and not any(e.get("filename") == child_manifest.get("cover") for e in keep):
        child_manifest["cover"] = keep[0]["filename"]
        child_manifest["cover_blurhash"] = None
    derived["lastSyncedAt"] = now_iso
    derived["parentUpdatedAt"] = parent_manifest.get("updatedAt")
    return child_manifest


def suppress_detkey(child_manifest: dict, detkey: str) -> None:
    """Tombstone a child crop the user deleted, so re-sync won't bring it back."""
    derived = child_manifest.setdefault("derived", {})
    sup = derived.get("suppressed")
    if not isinstance(sup, list):
        sup = []
    if detkey and detkey not in sup:
        sup.append(detkey)
    derived["suppressed"] = sup
