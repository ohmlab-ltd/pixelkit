"""YOLOX training and int8 ONNX export for the Neuro N6.

Input contract: int8, scale 1/255, zero-point -128, so the board's uint8
RGB888 reads as [0,1] RGB, letterboxed with 114/255 gray. Output: raw head
rows dx, dy, dw, dh, obj and one cls per class; obj and cls are sigmoided,
the box terms in grid units."""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import queue
import random
import re
import textwrap
import threading
import time
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import urllib.request
import zipfile
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch

import aug_source
import store
import workspace
from vendor.yolox import YOLOX, YOLOPAFPN, YOLOXHead, postprocess
from vendor.yolox.network_blocks import BaseConv
from vendor.yolox.qat import ActFQ, set_qat

_RELEASE = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/"

# Nano has 0.9 M parameters, Tiny 5 M and about 5x the MACs. sha256 is that of the
# release file at `url`.
MODEL_SPECS: dict[str, dict] = {
    "nano": {"label": "YOLOX-Nano", "width": 0.25, "depth": 0.33, "depthwise": True,
             "dir": "yolox_nano", "file": "yolox_nano.pth", "url": _RELEASE + "yolox_nano.pth",
             "sha256": "cd28f55fbbc1829f99d9ac9b38a16d259a22889739c8728ea877610201feff7b",
             "mb_per_image": 70.0},
    "tiny": {"label": "YOLOX-Tiny", "width": 0.375, "depth": 0.33, "depthwise": False,
             "dir": "yolox_tiny", "file": "yolox_tiny.pth", "url": _RELEASE + "yolox_tiny.pth",
             "sha256": "9de513de589ac98bb92d3bca53b5af7b9acfa9b0bacb831f7999d0f7afaee8f0",
             "mb_per_image": 150.0},
}
MODEL_DIRS = tuple(spec["dir"] for spec in MODEL_SPECS.values())
RUN_ID_RE = re.compile(r"^[0-9]{8}T[0-9]{6}Z$")
# First Neuro N6 core with PostProcess_PixelKitYOLOX; the exported sketch checks it.
CORE_MIN = (0, 1, 3)


def _split_for(name: str, train_split: float) -> str:
    """The exporters' hash split, as server._split_for; used when
    stratification is off."""
    if train_split >= 0.999:
        return "train"
    h = int(hashlib.sha1(name.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "train" if h < train_split else "val"


@dataclass
class TrainConfig:
    epochs: int = 50
    imgsz: int = 256           # square input side, px
    batch: int = 0             # 0 = sized from free VRAM
    lr: float = 0.01           # scaled by batch/64
    train_split: float = 0.8
    device: str = "cuda"
    pretrained: bool = True
    size: str = "nano"         # MODEL_SPECS key
    # None = the threshold with the best F1 on the int8 val predictions.
    conf_thres: float | None = None
    nms_thres: float = 0.45
    # Quantisation-aware fine-tuning epochs after the float run. 0 = PTQ only.
    qat_epochs: int = 8
    # PTQ calibration, used when qat_epochs is 0.
    calib_images: int = 200
    calib_percentile: float = 99.9
    seed: int = 0
    warmup_epochs: int = 3
    # Compile with ST Edge AI when installed; a model it rejects is not exported.
    verify_neuron6: bool = True
    # Also train on generated augmentations of train-split images.
    use_augmentations: bool = True
    # Per-class split; off = the exporters' hash split.
    stratify: bool = True
    # Run id to warm-start from instead of the COCO checkpoint.
    init_run: str | None = None
    # Held-out dataset id: all of this one trains. Labels match by name.
    eval_project: str | None = None
    # Run id to resume: weights, optimiser, history and settings.
    resume_run: str | None = None
    checkpoint_every: int = 1
    # Worst val images rendered into the gallery.
    failure_limit: int = 24
    # None = timestamp.
    run_id: str | None = None


@dataclass
class Sample:
    path: Path
    boxes: np.ndarray  # (n, 4) xyxy pixels
    classes: np.ndarray  # (n,)
    split: str
    w: int = 0
    h: int = 0
    name: str = ""            # filename the split is keyed on
    source: str = "original"  # "original" | "augmented" | "eval"
    parent: str = ""          # augmented copies: the original's filename
    import_id: str = ""
    measured: bool = False    # w/h come from decoding the file, not the manifest


def _box_xyxy(b: dict) -> tuple[float, float, float, float] | None:
    """A box from flat x0..y1 keys, a `box_xyxy` list or a `box` list, as the
    export builders read it."""
    try:
        if all(k in b for k in ("x0", "y0", "x1", "y1")):
            x0, y0, x1, y1 = float(b["x0"]), float(b["y0"]), float(b["x1"]), float(b["y1"])
        elif isinstance(b.get("box_xyxy"), (list, tuple)) and len(b["box_xyxy"]) == 4:
            x0, y0, x1, y1 = (float(v) for v in b["box_xyxy"])
        elif isinstance(b.get("box"), (list, tuple)) and len(b["box"]) == 4:
            x0, y0, x1, y1 = (float(v) for v in b["box"])
        else:
            return None
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (x0, y0, x1, y1)):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _box_label(b: dict) -> str:
    for k in ("label", "predLabel", "pred_label", "gdLabel", "gd_label"):
        v = b.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip().lower()
    return ""


def _import_boxes(imp: dict, cls_idx: dict[str, int], ignored: Counter | None = None
                  ) -> tuple[list, list]:
    """An import's boxes as the exports read them (server._image_index): its
    edited boxes once edited, else the detections the labeller did not
    reject. Boxes whose label is not a class are counted in `ignored`."""
    edited = imp.get("editedBoxes")
    if isinstance(edited, list) and (edited or imp.get("editedBoxesSet")):
        boxes_src = edited
    else:
        dets = imp.get("detections")
        boxes_src = ([d for d in dets if isinstance(d, dict) and not d.get("rejected")]
                     if isinstance(dets, list) else [])
    boxes, cls = [], []
    for b in boxes_src:
        if not isinstance(b, dict):
            continue
        rect = _box_xyxy(b)
        if rect is None:
            continue
        label = _box_label(b)
        if label not in cls_idx:
            if label and ignored is not None:
                ignored[label] += 1
            continue
        boxes.append(rect)
        cls.append(cls_idx[label])
    return boxes, cls


def _dataset_classes(m: dict) -> list[str]:
    seen: set[str] = set()
    out = []
    for t in m.get("tags") or []:
        if not isinstance(t, str):
            continue
        t = t.strip().lower()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _originals(project_id: str, m: dict, cls_idx: dict[str, int], source: str = "original",
               ignored: Counter | None = None) -> tuple[list[Sample], int]:
    """Labelled and background images of one dataset in manifest order.
    Returns (samples, count of other imports with no usable label)."""
    ds_dir = store.dataset_dir(project_id)
    samples: list[Sample] = []
    unlabelled = 0
    for imp in m.get("imports") or []:
        if not isinstance(imp, dict):
            continue
        fname = imp.get("filename")
        if not fname:
            continue
        boxes, cls = _import_boxes(imp, cls_idx, ignored)
        if not boxes and not imp.get("background"):
            unlabelled += 1
            continue
        path = ds_dir / "images" / fname
        if not path.is_file():
            continue
        samples.append(Sample(
            path=path,
            boxes=np.array(boxes, dtype=np.float32).reshape(-1, 4),
            classes=np.array(cls, dtype=np.float32),
            split="train",
            w=int(imp.get("width") or 0),
            h=int(imp.get("height") or 0),
            name=str(fname),
            source=source,
            import_id=str(imp.get("id") or ""),
        ))
    return samples, unlabelled


def _augmented(project_id: str, m: dict, parents: dict[str, Sample], cls_idx: dict[str, int]
               ) -> tuple[list[Sample], int]:
    """Generated copies of train-split images under
    <dataset>/augmentations/<import_id>/, and the number of images whose
    copies are out of date and left out. A copy that lost every box is
    dropped unless its parent is a background image."""
    root = store.dataset_dir(project_id) / "augmentations"
    if not root.is_dir():
        return [], 0
    out: list[Sample] = []
    stale = 0
    for imp in m.get("imports") or []:
        if not isinstance(imp, dict):
            continue
        import_id = imp.get("id")
        parent = imp.get("filename") or ""
        try:
            n_aug = int(imp.get("n_augmentations") or 0)
        except (TypeError, ValueError):
            n_aug = 0
        if not import_id or not parent or n_aug <= 0:
            continue
        parent_s = parents.get(parent)
        if parent_s is None or parent_s.split != "train":
            continue
        anno_path = root / str(import_id) / "annotations.json"
        if not anno_path.is_file():
            continue
        try:
            data = json.loads(anno_path.read_text("utf-8"))
        except (OSError, ValueError):
            continue
        if not aug_source.is_current(data, imp):
            stale += 1
            continue
        try:
            cw, ch = int(data.get("width") or 0), int(data.get("height") or 0)
        except (TypeError, ValueError):
            cw = ch = 0
        copies = data.get("copies") or {}
        if not isinstance(copies, dict):
            continue
        stem = Path(parent).stem
        for copy_name, dets in sorted(copies.items()):
            if not isinstance(dets, list) or "/" in copy_name or "\\" in copy_name:
                continue
            jpeg = root / str(import_id) / copy_name
            if not jpeg.is_file():
                continue
            boxes, cls = [], []
            for det in dets:
                if not isinstance(det, dict):
                    continue
                rect = _box_xyxy(det)
                label = _box_label(det)
                if rect is None or label not in cls_idx:
                    continue
                boxes.append(rect)
                cls.append(cls_idx[label])
            if not boxes and len(parent_s.boxes):
                continue
            out.append(Sample(
                path=jpeg,
                boxes=np.array(boxes, dtype=np.float32).reshape(-1, 4),
                classes=np.array(cls, dtype=np.float32),
                split="train",
                w=cw, h=ch,
                name=f"aug_{stem}_{Path(copy_name).stem}.jpg",
                source="augmented",
                parent=parent,
                import_id=str(import_id),
            ))
    return out, stale


def stratified_split(samples: list[Sample], train_split: float, seed: int = 0) -> None:
    """Assign train/val per class rather than per image, repeatably for a seed.

    A class with n >= 2 images gets round(n * (1 - train_split)) val images,
    at least one, and keeps one in train. Background images are a class."""
    if not samples:
        return
    if train_split >= 0.999:
        for s in samples:
            s.split = "train"
        return
    if train_split <= 0.001:
        for s in samples:
            s.split = "val"
        return
    keys = [hashlib.sha1(f"{seed}:{s.name}".encode("utf-8")).hexdigest() for s in samples]
    img_classes = [sorted({int(c) for c in s.classes.tolist()}) or [-1] for s in samples]
    total: Counter = Counter()
    for cs in img_classes:
        total.update(cs)
    want: dict[int, int] = {}
    for c, n in total.items():
        if n >= 2:
            want[c] = min(n - 1, max(1, int(round(n * (1.0 - train_split)))))
        else:
            want[c] = 0
    val_count: dict[int, int] = defaultdict(int)
    order = sorted(range(len(samples)),
                   key=lambda i: (min((total[c] for c in img_classes[i]), default=0), keys[i]))
    for i in order:
        cs = img_classes[i]
        need = any(val_count[c] < want[c] for c in cs)
        # A class never loses its last train image to another class's quota.
        room = all(val_count[c] < total[c] - 1 for c in cs)
        if cs and need and room:
            samples[i].split = "val"
            for c in cs:
                val_count[c] += 1
        else:
            samples[i].split = "train"


def _restore_split(samples: list[Sample], saved: dict) -> tuple[int, int]:
    """Give each image the split it had when the run started; an image added
    since trains. Returns (images added, images gone) since then."""
    val = {str(n) for n in saved.get("val") or []}
    known = val | {str(n) for n in saved.get("train") or []}
    for s in samples:
        s.split = "val" if s.name in val else "train"
    names = {s.name for s in samples}
    return len(names - known), len(known - names)


def load_samples(project_id: str, train_split: float, *, use_augmentations: bool = True,
                 stratify: bool = True, seed: int = 0, eval_project: str | None = None,
                 saved_split: dict | None = None) -> tuple[list[Sample], list[str], dict]:
    """Returns (samples, classes in dataset tag order, summary counts).
    `saved_split`, a resumed run's {"train": [...], "val": [...]} file names,
    replaces the split."""
    m = store.load(project_id)
    if not m:
        raise ValueError("dataset not found")
    classes = _dataset_classes(m)
    if not classes:
        raise ValueError("dataset has no labels")
    cls_idx = {c: i for i, c in enumerate(classes)}
    ignored: Counter = Counter()
    samples, unlabelled = _originals(project_id, m, cls_idx, ignored=ignored)
    info: dict = {"originals": len(samples), "augmented": 0, "unlabelled": unlabelled,
                  "background": sum(1 for s in samples if not len(s.boxes)),
                  # Box counts by label, most first.
                  "ignoredLabels": dict(sorted(ignored.items(), key=lambda kv: (-kv[1], kv[0]))),
                  "evalImages": 0, "evalMissingClasses": [], "evalIgnoredLabels": []}

    eval_samples: list[Sample] = []
    if eval_project:
        if eval_project == project_id:
            raise ValueError("the held-out dataset must be a different dataset")
        em = store.load(eval_project)
        if not em:
            raise ValueError("held-out dataset not found")
        for s in samples:
            s.split = "train"
        eval_samples, _ = _originals(eval_project, em, cls_idx, source="eval")
        for s in eval_samples:
            s.split = "val"
        present = {int(c) for s in eval_samples for c in s.classes.tolist()}
        info["evalImages"] = len(eval_samples)
        info["evalMissingClasses"] = [c for i, c in enumerate(classes) if i not in present]
        info["evalIgnoredLabels"] = sorted(t for t in _dataset_classes(em) if t not in cls_idx)
    elif saved_split is not None:
        info["splitAdded"], info["splitGone"] = _restore_split(samples, saved_split)
    elif stratify:
        stratified_split(samples, train_split, seed)
    else:
        for s in samples:
            s.split = _split_for(s.name, train_split)

    if use_augmentations:
        parents = {s.name: s for s in samples}
        aug, info["augmentedStale"] = _augmented(project_id, m, parents, cls_idx)
        info["augmented"] = len(aug)
        samples = samples + aug
    return samples + eval_samples, classes, info


def split_summary(samples: list[Sample], classes: list[str]) -> list[dict]:
    """Per-class image and box counts by split, augmented boxes apart."""
    rows = []
    for ci, name in enumerate(classes):
        r = {"name": name, "trainImages": 0, "valImages": 0, "trainBoxes": 0,
             "valBoxes": 0, "augmentedBoxes": 0}
        for s in samples:
            n = int((s.classes == ci).sum())
            if n == 0:
                continue
            if s.split == "val":
                r["valImages"] += 1
                r["valBoxes"] += n
            elif s.source == "augmented":
                r["augmentedBoxes"] += n
            else:
                r["trainImages"] += 1
                r["trainBoxes"] += n
        rows.append(r)
    return rows


# Pre-flight

def _file_fingerprint(path: Path) -> str | None:
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            head = f.read(256 * 1024)
    except OSError:
        return None
    return f"{size}:{hashlib.blake2b(head, digest_size=8).hexdigest()}"


def preflight_samples(samples: list[Sample], classes: list[str], info: dict, cfg: TrainConfig
                      ) -> dict:
    """Checks before training. Blockers stop the run; warnings go into the
    report."""
    blockers: list[str] = []
    warnings: list[str] = []
    train_s = [s for s in samples if s.split == "train"]
    val_s = [s for s in samples if s.split == "val"]
    n_orig_train = sum(1 for s in train_s if s.source == "original")
    rows = split_summary(samples, classes)

    n_labelled = n_orig_train + sum(1 for s in val_s if s.source == "original")
    if n_labelled < 2 or (not train_s):
        blockers.append(f"training needs at least 2 labelled images, have {n_labelled}")
    elif not any(len(s.boxes) for s in train_s):
        blockers.append("no training image has boxes: background images alone teach nothing")
    if cfg.eval_project and not val_s:
        blockers.append("the held-out dataset has no labelled images with this dataset's labels")
    if cfg.eval_project and info.get("evalMissingClasses"):
        missing = ", ".join(info["evalMissingClasses"])
        (blockers if len(info["evalMissingClasses"]) == len(classes) else warnings).append(
            f"the held-out dataset has no boxes for: {missing}")
    if cfg.eval_project and info.get("evalIgnoredLabels"):
        warnings.append("held-out labels not in this dataset are ignored: "
                        + ", ".join(info["evalIgnoredLabels"][:8]))
    if not blockers and not val_s:
        warnings.append("too few images for a val split: accuracy will be measured on the "
                        "training images and will read higher than it is")
    elif val_s and len(val_s) < 5:
        warnings.append(f"only {len(val_s)} val image(s): accuracy figures will be noisy")
    if info.get("unlabelled"):
        warnings.append(f"{info['unlabelled']} image(s) have no boxes and are not used")
    if info.get("ignoredLabels"):
        warnings.append(f"{sum(info['ignoredLabels'].values())} box(es) with labels not in this "
                        "dataset are left out: " + ", ".join(list(info["ignoredLabels"])[:8]))
    if info.get("augmentedStale"):
        warnings.append(f"{info['augmentedStale']} image(s) changed after their augmentations "
                        "were made, so those copies are left out: click Update on the "
                        "Augmentations tab to remake them")

    for r in rows:
        if r["trainBoxes"] == 0 and r["augmentedBoxes"] == 0:
            warnings.append(f"'{r['name']}' has no training boxes: the model cannot learn it")
        elif r["trainBoxes"] < 10:
            warnings.append(f"'{r['name']}' has only {r['trainBoxes']} training box(es): "
                            "expect poor recall")
        if val_s and r["valBoxes"] == 0 and r["trainBoxes"] > 0:
            warnings.append(f"'{r['name']}' has no val boxes: its accuracy is not measured")

    # A box under 8 px on its short side is below what the stride-8 head localises.
    small: Counter = Counter()
    per_class: Counter = Counter()
    for s in train_s:
        if s.w <= 0 or s.h <= 0:
            continue
        scale = cfg.imgsz / max(s.w, s.h)
        side = np.minimum(s.boxes[:, 2] - s.boxes[:, 0], s.boxes[:, 3] - s.boxes[:, 1]) * scale
        for c, tiny in zip(s.classes.tolist(), (side < 8.0).tolist()):
            per_class[int(c)] += 1
            if tiny:
                small[int(c)] += 1
    for ci, name in enumerate(classes):
        if per_class[ci] and small[ci] / per_class[ci] >= 0.25:
            pct = int(round(100 * small[ci] / per_class[ci]))
            warnings.append(f"{pct}% of '{name}' boxes are under 8 px at {cfg.imgsz}x{cfg.imgsz}: "
                            "try a larger input size or crop closer")
    for r in rows:
        r["smallBoxesPct"] = (int(round(100 * small[classes.index(r['name'])]
                                        / per_class[classes.index(r['name'])]))
                              if per_class[classes.index(r["name"])] else 0)

    if val_s and n_orig_train:
        fp_train: dict[str, str] = {}
        for s in train_s:
            if s.source != "original":
                continue
            k = _file_fingerprint(s.path)
            if k:
                fp_train.setdefault(k, s.name)
        leaks = 0
        for s in val_s:
            k = _file_fingerprint(s.path)
            if k and k in fp_train:
                leaks += 1
        if leaks:
            warnings.append(f"{leaks} val image(s) are byte-identical to a training image: "
                            "val accuracy will read higher than it is (remove the duplicates)")

    return {
        "ok": not blockers,
        "blockers": blockers,
        "warnings": warnings,
        "classes": rows,
        "images": {
            "train": n_orig_train,
            "val": len(val_s),
            "augmented": int(info.get("augmented") or 0),
            "augmentedStale": int(info.get("augmentedStale") or 0),
            "unlabelled": int(info.get("unlabelled") or 0),
            "background": int(info.get("background") or 0),
        },
        "imgsz": cfg.imgsz,
        "size": cfg.size,
        "evalProject": cfg.eval_project,
    }


def preflight(project_id: str, cfg: TrainConfig) -> dict:
    """Load the dataset the way train() will and run the checks."""
    samples, classes, info = load_samples(
        project_id, cfg.train_split, use_augmentations=cfg.use_augmentations,
        stratify=cfg.stratify, seed=cfg.seed, eval_project=cfg.eval_project)
    return preflight_samples(samples, classes, info, cfg)


# Preprocessing

def _letterbox_fit(w: int, h: int, size: int) -> tuple[float, int, int]:
    """Where letterbox() puts a w x h picture: (scale, dx, dy)."""
    scale = min(size / w, size / h)
    return scale, (size - int(round(w * scale))) // 2, (size - int(round(h * scale))) // 2


def letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    """RGB uint8 -> (size,size,3) with 114 padding. Returns (img, scale, dx, dy)."""
    h, w = img.shape[:2]
    scale, dx, dy = _letterbox_fit(w, h, size)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.full((size, size, 3), 114, dtype=np.uint8)
    out[dy : dy + nh, dx : dx + nw] = resized
    return out, scale, dx, dy


def _dims(s: Sample) -> tuple[int, int]:
    """(h, w) as decoded, cached. The manifest can hold an EXIF-rotated
    JPEG's dims unrotated."""
    if s.measured and s.h > 0 and s.w > 0:
        return s.h, s.w
    img = _read_rgb(s.path)
    if img is None:
        return (s.h or 1, s.w or 1)
    s.h, s.w = img.shape[:2]
    s.measured = True
    return s.h, s.w


def _read_rgb(path: Path) -> np.ndarray | None:
    """Decode an image file to RGB; None when unreadable."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    try:
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except cv2.error:
        return None
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _center_square(img: np.ndarray, boxes: np.ndarray, cls: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centre square of the picture; a box is kept if 60% of it stays inside."""
    h, w = img.shape[:2]
    s = min(h, w)
    x0, y0 = (w - s) // 2, (h - s) // 2
    img = img[y0:y0 + s, x0:x0 + s].copy()
    if len(boxes):
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        boxes = boxes - np.array([x0, y0, x0, y0], dtype=boxes.dtype)
        boxes = boxes.clip(0, s - 1)
        kept = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        keep = (kept >= 0.6 * np.maximum(area, 1e-6)) & (boxes[:, 2] - boxes[:, 0] > 2) \
            & (boxes[:, 3] - boxes[:, 1] > 2)
        boxes, cls = boxes[keep], cls[keep]
    return img, boxes, cls


def _augment(img: np.ndarray, boxes: np.ndarray, cls: np.ndarray
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = img.shape[:2]
    if h != w and random.random() < 0.3:
        img, boxes, cls = _center_square(img, boxes, cls)
        h, w = img.shape[:2]
    if random.random() < 0.5:
        img = img[:, ::-1].copy()
        if len(boxes):
            boxes = boxes.copy()
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
    if random.random() < 0.8:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.int16)
        hsv[..., 0] = (hsv[..., 0] + random.randint(-8, 8)) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * random.uniform(0.7, 1.3), 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * random.uniform(0.7, 1.3), 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    # Scale jitter, background images included.
    if random.random() < 0.5:
        s = random.uniform(0.75, 1.25)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, s)
        img = cv2.warpAffine(img, M, (w, h), borderValue=(114, 114, 114))
        pts = boxes.reshape(-1, 2)
        pts = np.hstack([pts, np.ones((len(pts), 1))]) @ M.T
        boxes = pts.reshape(-1, 4).astype(np.float32)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h - 1)
        keep = (boxes[:, 2] - boxes[:, 0] > 2) & (boxes[:, 3] - boxes[:, 1] > 2)
        boxes, cls = boxes[keep], cls[keep]
    return img, boxes, cls


def _make_batch(samples: list[Sample], idxs: list[int], size: int, train: bool,
                max_objs: int = 100) -> tuple[torch.Tensor, torch.Tensor]:
    imgs = np.zeros((len(idxs), 3, size, size), dtype=np.float32)
    targets = np.zeros((len(idxs), max_objs, 5), dtype=np.float32)
    for bi, si in enumerate(idxs):
        s = samples[si]
        img = _read_rgb(s.path)
        if img is None:
            continue
        boxes = s.boxes.copy()
        cls = s.classes.copy()
        if train:
            img, boxes, cls = _augment(img, boxes, cls)
        img, scale, dx, dy = letterbox(img, size)
        # [0,1], not [0,255]: the input contract in the module docstring.
        imgs[bi] = img.astype(np.float32).transpose(2, 0, 1) / 255.0
        j = 0
        for k in range(len(boxes)):
            if j >= max_objs:
                break
            x0, y0, x1, y1 = boxes[k] * scale
            x0, x1 = x0 + dx, x1 + dx
            y0, y1 = y0 + dy, y1 + dy
            # Clipped to the canvas: a box with no anchor near it crashes the head's matching.
            x0, x1 = max(0.0, x0), min(float(size), x1)
            y0, y1 = max(0.0, y0), min(float(size), y1)
            if x1 - x0 < 2.0 or y1 - y0 < 2.0:
                continue
            targets[bi, j] = [cls[k], (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0]
            j += 1
    return torch.from_numpy(imgs), torch.from_numpy(targets)


def _letterboxed_rgb(s: Sample, size: int) -> np.ndarray | None:
    img = _read_rgb(s.path)
    if img is None:
        return None
    return letterbox(img, size)[0]


def _gt_letterbox(s: Sample, imgsz: int) -> np.ndarray:
    """Ground truth of a sample in the model's letterbox space: (m,5)."""
    hh, ww = _dims(s)
    scale, dx, dy = _letterbox_fit(ww, hh, imgsz)
    g = s.boxes * scale
    g[:, [0, 2]] += dx
    g[:, [1, 3]] += dy
    g = np.clip(g, 0.0, float(imgsz))
    keep = (g[:, 2] - g[:, 0] >= 1.0) & (g[:, 3] - g[:, 1] >= 1.0)
    return np.hstack([g[keep], s.classes[keep][:, None]]).astype(np.float32)


def _batch_iter(samples: list[Sample], order: list[int], batch: int, size: int,
                train: bool, workers: int = 4, lookahead: int = 6):
    """Yield (imgs, targets) batches prepared ahead on a thread pool; cv2
    releases the GIL."""
    chunks = [order[i:i + batch] for i in range(0, len(order), batch)]
    chunks = [c for c in chunks if c]
    if not chunks:
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending: "queue.Queue" = queue.Queue()
        it = iter(chunks)
        for _ in range(min(lookahead, len(chunks))):
            pending.put(pool.submit(_make_batch, samples, next(it), size, train))
        while not pending.empty():
            fut = pending.get()
            nxt = next(it, None)
            if nxt is not None:
                pending.put(pool.submit(_make_batch, samples, nxt, size, train))
            yield fut.result()


# Evaluation: VOC AP@0.5

def _voc_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def ap50(preds: list[np.ndarray], gts: list[np.ndarray], n_classes: int,
         iou_thr: float = 0.5) -> tuple[float, list[float | None]]:
    """preds[i]: (n,6) x0,y0,x1,y1,score,cls per image; gts[i]: (m,5) x0..y1,cls.

    Returns (mean over classes with ground truth, per-class AP indexed by
    class, None where a class has no ground truth)."""
    per_class: list[float | None] = []
    for c in range(n_classes):
        rows = []  # (score, is_tp)
        n_gt = 0
        for pi in range(len(preds)):
            gt = gts[pi]
            gt_c = gt[gt[:, 4] == c][:, :4] if len(gt) else np.zeros((0, 4))
            n_gt += len(gt_c)
            pr = preds[pi]
            pr_c = pr[pr[:, 5] == c] if len(pr) else np.zeros((0, 6))
            used = np.zeros(len(gt_c), dtype=bool)
            for p in pr_c[np.argsort(-pr_c[:, 4])] if len(pr_c) else []:
                if not len(gt_c):
                    rows.append((p[4], 0))
                    continue
                xx0 = np.maximum(p[0], gt_c[:, 0]); yy0 = np.maximum(p[1], gt_c[:, 1])
                xx1 = np.minimum(p[2], gt_c[:, 2]); yy1 = np.minimum(p[3], gt_c[:, 3])
                inter = np.maximum(0, xx1 - xx0) * np.maximum(0, yy1 - yy0)
                area_p = (p[2] - p[0]) * (p[3] - p[1])
                area_g = (gt_c[:, 2] - gt_c[:, 0]) * (gt_c[:, 3] - gt_c[:, 1])
                iou = inter / np.maximum(area_p + area_g - inter, 1e-9)
                best = int(np.argmax(iou))
                if iou[best] >= iou_thr and not used[best]:
                    used[best] = True
                    rows.append((p[4], 1))
                else:
                    rows.append((p[4], 0))
        if n_gt == 0:
            per_class.append(None)
            continue
        if not rows:
            per_class.append(0.0)
            continue
        rows.sort(key=lambda r: -r[0])
        tp = np.cumsum([r[1] for r in rows])
        fp = np.cumsum([1 - r[1] for r in rows])
        per_class.append(_voc_ap(tp / n_gt, tp / np.maximum(tp + fp, 1e-9)))
    present = [a for a in per_class if a is not None]
    mean = float(np.mean(present)) if present else 0.0
    return mean, per_class


# Contract constants: uint8 camera bytes re-encoded as int8 must mean [0,1].
CONTRACT_SCALE = 1.0 / 255.0
CONTRACT_ZP = -128


def _decode_raw(raw: np.ndarray, imgsz: int) -> np.ndarray:
    """YOLOX's grid decode of raw head rows, in numpy, as yolox_pp.c does on
    the M55. Row order is stride 8, then 16, then 32."""
    out = np.array(raw, dtype=np.float32, copy=True)
    grids, strides = [], []
    for s_ in (8, 16, 32):
        g = imgsz // s_
        yv, xv = np.meshgrid(np.arange(g), np.arange(g), indexing="ij")
        grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
        strides.append(np.full((g * g, 1), s_, dtype=np.float32))
    grid = np.concatenate(grids, 0)[None].astype(np.float32)
    stride = np.concatenate(strides, 0)[None]
    out[..., :2] = (out[..., :2] + grid) * stride
    out[..., 2:4] = np.exp(np.clip(out[..., 2:4], -20, 20)) * stride
    return out


def _verify_input_contract(path: Path) -> dict:
    """Pin the graph input's quantisation to the contract, then check the
    saved file. stedgeai reads the input as uint8 channel-last, which is
    correct only for scale 1/255, zero-point -128."""
    import onnx
    from onnx import helper, numpy_helper as nh

    m = onnx.load(str(path))
    g = m.graph
    inp = g.input[0].name
    init = {i.name: i for i in g.initializer}
    q = next((n for n in g.node if n.op_type == "QuantizeLinear" and n.input[0] == inp), None)
    calibrated = None
    if q is not None:
        consts = {n.output[0]: n for n in g.node if n.op_type == "Constant"}

        def _val(name):
            if name in init:
                return nh.to_array(init[name])
            if name in consts:
                return nh.to_array(consts[name].attribute[0].t)
            raise ValueError(f"input quantiser parameter {name!r} is not a constant")

        cal_s, cal_z = float(_val(q.input[1])), int(_val(q.input[2]))
        calibrated = {"scale": cal_s, "zeroPoint": cal_z}
        # A calibrated range outside [0,1] means training fed another input range.
        lo, hi = cal_s * (-128 - cal_z), cal_s * (127 - cal_z)
        if lo < -0.02 or hi > 1.02:
            raise ValueError(
                f"export rejected: input calibrated to scale={cal_s:.8f} zp={cal_z} "
                f"(range [{lo:.3f}, {hi:.3f}]), but the Neuro N6 contract needs "
                f"scale={CONTRACT_SCALE:.8f} zp={CONTRACT_ZP} over [0, 1]. The model was "
                "calibrated on the wrong input range - fix the preprocessing rather than "
                "rewriting the finished network.")
        # A sub-range of [0,1] is pinned to the contract.
        for idx, value in ((1, np.float32(CONTRACT_SCALE)), (2, np.int8(CONTRACT_ZP))):
            name = q.input[idx]
            if name in init:
                init[name].CopyFrom(nh.from_array(value, name))
            else:
                consts[name].attribute[0].t.CopyFrom(nh.from_array(value, name))
    else:
        # No input quantiser: insert one with the contract's parameters.
        s_n, z_n = inp + "_pk_scale", inp + "_pk_zp"
        g.initializer.extend([nh.from_array(np.float32(CONTRACT_SCALE), s_n),
                              nh.from_array(np.int8(CONTRACT_ZP), z_n)])
        qout, dout = inp + "_pk_q", inp + "_pk_dq"
        for n in g.node:
            for i, nm in enumerate(n.input):
                if nm == inp:
                    n.input[i] = dout
        g.node.insert(0, helper.make_node("DequantizeLinear", [qout, s_n, z_n], [dout],
                                          name=inp + "_pk_DQ"))
        g.node.insert(0, helper.make_node("QuantizeLinear", [inp, s_n, z_n], [qout],
                                          name=inp + "_pk_Q"))
    onnx.save(m, str(path))

    # The check reads the saved file back.
    m = onnx.load(str(path))
    g = m.graph
    arr = {i.name: nh.to_array(i) for i in g.initializer}
    for n in g.node:
        if n.op_type == "Constant":
            arr[n.output[0]] = nh.to_array(n.attribute[0].t)
    q = next((n for n in g.node
              if n.op_type == "QuantizeLinear" and n.input[0] == g.input[0].name), None)
    if q is None:
        raise ValueError("export rejected: graph input is not quantised (Neuro N6 "
                         "needs an asymmetric [0,1] input; see the module docstring)")
    scale, zp = float(arr[q.input[1]]), int(arr[q.input[2]])
    if abs(scale - CONTRACT_SCALE) > 1e-9 or zp != CONTRACT_ZP or arr[q.input[2]].dtype != np.int8:
        raise ValueError(
            f"export rejected: input quantisation is scale={scale} zp={zp} "
            f"({arr[q.input[2]].dtype}), the Neuro N6 contract needs "
            f"scale={CONTRACT_SCALE} zp={CONTRACT_ZP} (int8)")
    return {"scale": scale, "zeroPoint": zp,
            "range": [round(scale * (-128 - zp), 6), round(scale * (127 - zp), 6)],
            "calibrated": calibrated}


def _decode_rows(out: torch.Tensor, conf: float, nms: float, n_classes: int) -> np.ndarray:
    dets = postprocess(out, n_classes, conf, nms)[0]
    if dets is None:
        return np.zeros((0, 6), dtype=np.float32)
    d = dets.cpu().numpy()
    # postprocess rows: x0,y0,x1,y1,obj,cls_score,cls
    return np.stack([d[:, 0], d[:, 1], d[:, 2], d[:, 3], d[:, 4] * d[:, 5], d[:, 6]], axis=1)


# Inference with an exported model

def open_onnx(path: Path):
    import onnxruntime as ort

    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def infer_image(sess, img_rgb: np.ndarray, imgsz: int, n_classes: int, conf: float,
                nms: float) -> np.ndarray:
    """Run an exported raw-head ONNX on one RGB image. Returns (n,6)
    rows x0,y0,x1,y1,score,cls in the original image's pixels."""
    h, w = img_rgb.shape[:2]
    lb, scale, dx, dy = letterbox(img_rgb, imgsz)
    x = lb.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    raw = sess.run(None, {"images": x})[0]
    out = torch.from_numpy(_decode_raw(raw, imgsz))
    rows = _decode_rows(out, conf, nms, n_classes)
    if not len(rows):
        return rows
    rows = rows.copy()
    rows[:, [0, 2]] = (rows[:, [0, 2]] - dx) / scale
    rows[:, [1, 3]] = (rows[:, [1, 3]] - dy) / scale
    rows[:, [0, 2]] = rows[:, [0, 2]].clip(0, w)
    rows[:, [1, 3]] = rows[:, [1, 3]].clip(0, h)
    keep = (rows[:, 2] - rows[:, 0] >= 1.0) & (rows[:, 3] - rows[:, 1] >= 1.0)
    return rows[keep]


# Training run

class Cancelled(Exception):
    pass


def init_run_problem(run_dir: Path | None, size: str) -> str | None:
    """Why a run of `size` cannot start from the run in `run_dir`, or None.
    Runs of the other size share almost no weight shapes."""
    if run_dir is None or not (run_dir / "best.pt").is_file():
        return "cannot start from that run: it has no saved weights"
    if run_dir.parent.name != MODEL_SPECS[size]["dir"]:
        other = next((s for s, spec in MODEL_SPECS.items() if spec["dir"] == run_dir.parent.name), "other")
        return (f"a {size.capitalize()} run cannot start from a {other.capitalize()} run: "
                "pick a run of the same size, or the COCO checkpoint")
    return None


def find_run_dir(project_id: str, run_id: str) -> Path | None:
    """<dataset>/models/<model dir>/<run> for whichever size produced it."""
    if not run_id or not RUN_ID_RE.fullmatch(run_id):
        return None
    try:
        root = store.dataset_dir(project_id) / "models"
    except KeyError:
        return None
    for d in MODEL_DIRS:
        p = root / d / run_id
        if p.is_dir():
            return p
    return None


def _build_model(n_classes: int, size: str = "nano", conv_stem: bool = True) -> YOLOX:
    spec = MODEL_SPECS[size]
    backbone = YOLOPAFPN(depth=spec["depth"], width=spec["width"], depthwise=spec["depthwise"])
    head = YOLOXHead(num_classes=n_classes, width=spec["width"], depthwise=spec["depthwise"])
    model = YOLOX(backbone, head)
    if conv_stem:
        # Stride-2 conv stem: Neural-ART runs the Focus stem's Slices in software.
        base_ch = int(spec["width"] * 64)
        model.backbone.backbone.stem = BaseConv(3, base_ch, ksize=6, stride=2, act="silu")

    def init(m):
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eps, m.momentum = 1e-3, 0.03
    model.apply(init)
    model.head.initialize_biases(1e-2)
    return model


def _fq_keys(model: YOLOX) -> set[str]:
    """State-dict keys of the fake-quant observers (never warm-started)."""
    keys = set()
    for name, m in model.named_modules():
        if isinstance(m, ActFQ):
            for b in ("lo", "hi", "seen", "n"):
                keys.add(f"{name}.{b}" if name else b)
    return keys


class _BadCheckpoint(Exception):
    """A COCO checkpoint that is incomplete, unreadable or not the release file."""


# Cached checkpoints whose hash matched in this process: (path, size, mtime).
_HASHED: set[tuple[str, int, int]] = set()


def _open_read(path: Path, tries: int = 40, wait: float = 0.05):
    """open(path, "rb"), retried: on Windows it fails while another run
    replaces the file."""
    for i in range(tries):
        try:
            return open(path, "rb")
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(wait)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with _open_read(path) as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _discard(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _fetch_pretrained(spec: dict, dest: Path, cancel=None) -> None:
    """Download spec["url"] to dest. A body shorter than its Content-Length,
    or whose SHA-256 is not spec["sha256"], raises _BadCheckpoint and is not
    kept. The temp name is unique to the thread."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f"{dest.name}.{os.getpid()}.{threading.get_ident()}.{os.urandom(4).hex()}.tmp")
    try:
        digest, got = hashlib.sha256(), 0
        # Timeout and cancel poll: this thread holds the GPU gate.
        with urllib.request.urlopen(spec["url"], timeout=30) as r, open(tmp, "wb") as f:  # noqa: S310
            length = str(r.headers.get("Content-Length") or "")
            while True:
                if cancel is not None and cancel.is_set():
                    raise Cancelled()
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                digest.update(chunk)
                got += len(chunk)
        # read() returns b"" when the server closes the connection early.
        if length.isdigit() and got != int(length):
            raise _BadCheckpoint(f"the COCO checkpoint download stopped at {got} of {length} bytes")
        if spec.get("sha256") and digest.hexdigest() != spec["sha256"]:
            raise _BadCheckpoint("the downloaded COCO checkpoint failed its checksum")
        _replace_retry(tmp, dest)
    finally:
        _discard(tmp)


def _check_cached(path: Path, sha256: str | None) -> None:
    """Hash a cached checkpoint before its first load in this process; one
    that does not match is deleted."""
    if not sha256:
        return
    st = path.stat()
    key = (str(path), st.st_size, st.st_mtime_ns)
    if key in _HASHED:
        return
    if _sha256(path) != sha256:
        _discard(path)
        raise _BadCheckpoint("the cached COCO checkpoint failed its checksum and was deleted")
    _HASHED.add(key)


def _load_pretrained(model: YOLOX, cancel=None, size: str = "nano",
                     notes: list[str] | None = None) -> bool:
    """Warm-start from Megvii's COCO checkpoint, cached in
    <workspace>/weights/yolox/. Class-dependent head layers are skipped. On
    failure returns False and adds the reason to `notes`."""
    spec = MODEL_SPECS[size]
    dest = workspace.weights_dir() / "yolox" / spec["file"]
    try:
        if not dest.is_file():
            _fetch_pretrained(spec, dest, cancel)
        _check_cached(dest, spec.get("sha256"))
        try:
            with _open_read(dest) as f:
                ckpt = torch.load(f, map_location="cpu", weights_only=True)
        except Exception as e:  # noqa: BLE001
            # Deleted, so the next run downloads it again.
            _discard(dest)
            raise _BadCheckpoint("the cached COCO checkpoint could not be read and was deleted") from e
        state = dict(ckpt.get("model", ckpt))
        own = model.state_dict()

        # The Focus stem's 3x3 conv on 12 channels maps exactly onto the 6x6 stride-2 conv.
        f_w = state.pop("backbone.backbone.stem.conv.conv.weight", None)
        tgt = "backbone.backbone.stem.conv.weight"
        if f_w is not None and tgt in own and own[tgt].shape[-1] == 6:
            k6 = torch.zeros_like(own[tgt])                 # [out, 3, 6, 6]
            for grp, (py, px) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        k6[:, :, 2 * di + py + 2, 2 * dj + px + 2] = \
                            f_w[:, grp * 3:(grp + 1) * 3, di + 1, dj + 1]
            # The checkpoint was trained on OpenCV's BGR in [0,255]; the graph
            # takes RGB in [0,1].
            state[tgt] = k6.flip(1) * 255.0
        # Focus holds its BN one level deeper than BaseConv.
        for suffix in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
            v = state.pop(f"backbone.backbone.stem.conv.bn.{suffix}", None)
            if v is not None:
                state[f"backbone.backbone.stem.bn.{suffix}"] = v

        loadable = {
            k: v for k, v in state.items()
            if k in own and own[k].shape == v.shape
        }
        model.load_state_dict(loadable, strict=False)
        return True
    except Cancelled:
        raise
    except Exception as e:  # noqa: BLE001
        if notes is not None:
            why = str(e) if isinstance(e, _BadCheckpoint) else "COCO checkpoint unavailable (offline?)"
            notes.append(f"{why}: trained from scratch")
        return False


def _load_run_weights(model: YOLOX, run_dir: Path, classes: list[str] | None = None) -> dict:
    """Warm-start from another run's best.pt: shape-matched tensors only, no
    fake-quant observers. When the run's labels differ from `classes`, a label
    both have keeps its class-prediction channel and the others start fresh."""
    ck = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    state = dict(ck.get("model", ck))
    own = model.state_dict()
    skip = _fq_keys(model)
    run_classes = list(ck.get("classes") or [])
    reused: list[str] = []
    if classes is not None and run_classes and run_classes != classes:
        pairs = [(j, run_classes.index(c)) for j, c in enumerate(classes) if c in run_classes]
        reused = [classes[j] for j, _ in pairs]
        # head.cls_preds.<level>.weight/bias: one output channel per class.
        for k in [k for k in state if k.startswith("head.cls_preds.")]:
            v = state.pop(k)
            if pairs and k in own and v.shape[1:] == own[k].shape[1:]:
                t = own[k].clone()
                for j, i in pairs:
                    t[j] = v[i]
                state[k] = t
    loadable = {k: v for k, v in state.items()
                if k in own and k not in skip and own[k].shape == v.shape}
    model.load_state_dict(loadable, strict=False)
    return {"run": run_dir.name, "loaded": len(loadable),
            "skipped": len([k for k in own if k not in loadable and k not in skip]),
            "classes": run_classes,
            # Labels whose class channels came from the run when the labels differ.
            "reusedClasses": reused,
            # A run that does not record it may descend from the checkpoint.
            "cocoWeights": bool(ck.get("cocoWeights", True))}


def _check_cancel(cancel) -> None:
    if cancel is not None and cancel.is_set():
        raise Cancelled()


def _gpu_turn(st: dict) -> None:
    """Lend the card between two units of GPU work, if the engine asks."""
    turn = st.get("gpu_turn")
    if turn is not None:
        turn()


def _is_oom(e: BaseException) -> bool:
    oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_cls is not None and isinstance(e, oom_cls):
        return True
    msg = str(e).lower()
    return isinstance(e, RuntimeError) and ("out of memory" in msg or "cudnn_status_alloc_failed" in msg)


# Minimum optimiser steps per epoch; caps the batch on small datasets.
MIN_STEPS_PER_EPOCH = 15
# history.json refresh period. The server treats a run whose file is
# younger than a few periods as live.
HEARTBEAT_SECONDS = 30


def _auto_batch(device: torch.device, imgsz: int, size: str, n_train: int) -> int:
    """60% of free VRAM over the per-image peak (mb_per_image at 256x256),
    capped at 64 and by MIN_STEPS_PER_EPOCH. _epoch() halves it on OOM."""
    by_steps = max(2, int(round(n_train / MIN_STEPS_PER_EPOCH)))
    if device.type != "cuda":
        return max(2, min(8, by_steps, n_train))
    try:
        free, _total = torch.cuda.mem_get_info(device)
    except Exception:  # noqa: BLE001
        return max(2, min(16, by_steps, n_train))
    per_image = MODEL_SPECS[size]["mb_per_image"] * 1e6 * (imgsz / 256.0) ** 2
    b = int(free * 0.6 / per_image)
    b = max(4, min(64, b))
    b = min(b, by_steps)
    if b >= 16:
        b -= b % 8
    return max(2, min(b, n_train))


def _epoch(samples, train_s: list[int], st: dict, imgsz: int, step_fn, cancel, notes: list[str]
           ) -> float:
    """One pass over train_s; returns the mean finite loss. On CUDA OOM the
    batch halves, st["on_oom"] is called and the epoch restarts. An epoch
    with no finite step raises."""
    while True:
        batch = st["batch"]
        steps = max(1, len(train_s) // batch)
        st["steps"] = steps
        order = train_s[:]
        random.shuffle(order)
        # Full batches only, for BN statistics.
        order = order[: steps * batch]
        loss_sum, n, bad = 0.0, 0, 0
        it = _batch_iter(samples, order, batch, imgsz, train=True)
        oom = False
        try:
            for step, (imgs, targets) in enumerate(it):
                _check_cancel(cancel)
                v = step_fn(step, imgs, targets)
                if math.isfinite(v):
                    loss_sum += v
                    n += 1
                else:
                    bad += 1
                _gpu_turn(st)
        except RuntimeError as e:
            if not _is_oom(e) or batch <= 2:
                raise
            oom = True
        finally:
            it.close()
        if not oom:
            if bad and not n:
                raise RuntimeError("training diverged: every step of the epoch had a non-finite "
                                   "loss (check the boxes, or lower the learning rate)")
            if bad:
                notes.append(f"{bad} step(s) with a non-finite loss were skipped")
            return loss_sum / max(1, n)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        st["batch"] = max(2, batch // 2)
        notes.append(f"out of memory at batch {batch}: continuing at {st['batch']}")
        on_oom = st.get("on_oom")
        if on_oom is not None:
            on_oom()


def _json_safe(obj):
    """Strict JSON: non-finite floats become null, as the server's orjson
    rejects NaN."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj) if math.isfinite(float(obj)) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def _replace_retry(tmp: Path, path: Path, tries: int = 40, wait: float = 0.05) -> None:
    """os.replace, retried: on Windows it fails while another process reads
    the target."""
    for i in range(tries):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if i == tries - 1:
                try:
                    tmp.unlink()
                except OSError:
                    pass
                raise
            time.sleep(wait)


def _json_text(data) -> str:
    return json.dumps(_json_safe(data), indent=2, allow_nan=False)


def _write_json_atomic(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(_json_text(data), "utf-8", newline="\n")
    _replace_retry(tmp, path)


def _save_ckpt(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    torch.save(payload, tmp)
    _replace_retry(tmp, path)


def train(
    project_id: str,
    cfg: TrainConfig,
    progress: Callable[[dict], None] | None = None,
    cancel=None,
    gpu_turn: Callable[[], None] | None = None,
) -> dict:
    """Train and export. On cuda:N that device is current for the duration,
    since the vendored head allocates its grids there. `gpu_turn` is called
    between units of GPU work to lend the card to labelling."""
    device = torch.device(cfg.device)
    try:
        if device.type == "cuda" and device.index is not None:
            with torch.cuda.device(device):
                return _train_impl(project_id, cfg, progress, cancel, gpu_turn)
        return _train_impl(project_id, cfg, progress, cancel, gpu_turn)
    finally:
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _restore_cfg(saved: dict, cfg: TrainConfig) -> TrainConfig:
    """The run's saved settings, with device, verify_neuron6 and resume_run
    from the request."""
    names = {f.name for f in fields(TrainConfig)}
    merged = {k: v for k, v in saved.items() if k in names}
    out = TrainConfig(**merged)
    out.device = cfg.device
    out.verify_neuron6 = cfg.verify_neuron6
    out.resume_run = cfg.resume_run
    return out


def _train_impl(project_id: str, cfg: TrainConfig, progress, cancel, gpu_turn=None) -> dict:
    t0 = time.time()
    if cfg.size not in MODEL_SPECS:
        raise ValueError(f"unknown model size {cfg.size!r} (nano or tiny)")

    # A resumed run's checkpoint carries its settings.
    ck = None
    if cfg.resume_run:
        run_dir = find_run_dir(project_id, cfg.resume_run)
        if run_dir is None or not (run_dir / "ckpt_last.pt").is_file():
            raise ValueError("nothing to resume: that run has no checkpoint")
        ck = torch.load(run_dir / "ckpt_last.pt", map_location="cpu", weights_only=True)
        cfg = _restore_cfg(ck.get("cfg") or {}, cfg)
        cfg.run_id = ck["run_id"]

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # A resume keeps the run's split; a checkpoint without one is split again.
    saved_split = ck.get("split") if ck is not None else None
    samples, classes, info = load_samples(
        project_id, cfg.train_split, use_augmentations=cfg.use_augmentations,
        stratify=cfg.stratify, seed=cfg.seed, eval_project=cfg.eval_project,
        saved_split=saved_split if isinstance(saved_split, dict) else None)
    checks = preflight_samples(samples, classes, info, cfg)
    if checks["blockers"]:
        raise ValueError("; ".join(checks["blockers"]))
    if ck is not None and list(ck.get("classes") or []) != classes:
        raise ValueError("the dataset's labels changed since that run started; start a new run")
    split_names = {k: sorted(s.name for s in samples if s.source == "original" and s.split == k)
                   for k in ("train", "val")}
    train_s = [i for i, s in enumerate(samples) if s.split == "train"]
    val_s = [i for i, s in enumerate(samples) if s.split == "val"]
    scored_on_train = not val_s
    if scored_on_train:  # no val split: score on train originals
        val_s = [i for i in train_s if samples[i].source == "original"] or list(train_s)

    device = torch.device(cfg.device)
    use_amp = device.type == "cuda"
    spec = MODEL_SPECS[cfg.size]
    model = _build_model(len(classes), cfg.size).to(device)

    run_id = cfg.run_id or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_dir = store.dataset_dir(project_id) / "models" / spec["dir"] / run_id
    if ck is None and cfg.run_id is None:
        # Two runs started in the same second must not share a folder.
        for _ in range(5):
            try:
                out_dir.mkdir(parents=True, exist_ok=False)
                break
            except FileExistsError:
                time.sleep(1.0)
                run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
                out_dir = out_dir.with_name(run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = str((store.load(project_id) or {}).get("name") or "model")
    slug = _c_slug(dataset_name)
    history: list[dict] = list(ck.get("history") or []) if ck is not None else []
    notes: list[str] = list(ck.get("notes") or []) if ck is not None else []
    total_epochs = cfg.epochs + cfg.qat_epochs

    run_state = {"status": "training", "error": None, "final": False}
    history_lock = threading.Lock()

    def write_history(status: str, error: str | None = None) -> None:
        """An I/O failure on history.json is noted once and otherwise ignored."""
        run_state["status"] = status
        run_state["error"] = error
        try:
            with history_lock:
                _write_json_atomic(out_dir / "history.json", {
                    "run": run_id, "model": spec["dir"], "size": cfg.size, "status": status,
                    "epochs": cfg.epochs, "qatEpochs": cfg.qat_epochs, "imgsz": cfg.imgsz,
                    "classes": classes, "history": list(history), "notes": list(notes),
                    "error": error, "updatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                })
        except OSError as e:
            msg = f"could not update history.json ({type(e).__name__})"
            if msg not in notes:
                notes.append(msg)

    # Heartbeat: other processes judge a run live by the age of history.json.
    stop_beat = threading.Event()

    def _beat() -> None:
        while not stop_beat.wait(HEARTBEAT_SECONDS):
            if run_state["final"]:
                return
            write_history(run_state["status"], run_state["error"])

    beat_thread = threading.Thread(target=_beat, name="pk-train-heartbeat", daemon=True)

    # Weights: resume, previous run, COCO or scratch.
    init_info: dict | None = None
    pretrained_used = False
    if ck is not None:
        model.load_state_dict(ck["model"], strict=False)
        init_info = ck.get("init")
        pretrained_used = bool(ck.get("pretrained"))
    elif cfg.init_run:
        prev = find_run_dir(project_id, cfg.init_run)
        problem = init_run_problem(prev, cfg.size)
        if problem:
            raise ValueError(problem)
        init_info = _load_run_weights(model, prev, classes)
        if init_info["classes"] and init_info["classes"] != classes:
            kept = len(init_info["reusedClasses"])
            notes.append(f"started from run {prev.name}, whose labels differ: "
                         + (f"its class layers were reused by label name for {kept} of "
                            f"{len(classes)} label(s)" if kept else "its class layers were not reused"))
    elif cfg.pretrained:
        pretrained_used = _load_pretrained(model, cancel, cfg.size, notes)
    # The weights descend from Megvii's COCO checkpoint, directly or through earlier runs.
    coco_weights = pretrained_used or (init_info is not None and bool(init_info.get("cocoWeights", True)))
    if ck is not None and "cocoWeights" in ck:
        coco_weights = bool(ck["cocoWeights"])

    # Batch and optimiser
    if ck is not None:
        batch = int(ck.get("batch") or 0) or _auto_batch(device, cfg.imgsz, cfg.size, len(train_s))
    else:
        batch = cfg.batch if cfg.batch > 0 else _auto_batch(device, cfg.imgsz, cfg.size, len(train_s))
    batch = max(2, min(batch, len(train_s)))
    st = {"batch": batch, "steps": max(1, len(train_s) // batch), "gpu_turn": gpu_turn}

    pg_bn, pg_w, pg_b = [], [], []
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            pg_bn.append(m.weight)
        elif hasattr(m, "weight") and isinstance(m.weight, torch.nn.Parameter):
            pg_w.append(m.weight)
        if hasattr(m, "bias") and isinstance(getattr(m, "bias"), torch.nn.Parameter):
            pg_b.append(m.bias)
    opt = torch.optim.SGD(
        [{"params": pg_bn, "weight_decay": 0.0},
         {"params": pg_w, "weight_decay": 5e-4},
         {"params": pg_b, "weight_decay": 0.0}],
        lr=cfg.lr * batch / 64, momentum=0.9, nesterov=True,
    )
    scaler_ref = [torch.amp.GradScaler("cuda", enabled=use_amp)]
    if ck is not None and ck.get("phase") == "float":
        try:
            opt.load_state_dict(ck["opt"])
            if ck.get("scaler") and use_amp:
                scaler_ref[0].load_state_dict(ck["scaler"])
        except Exception:  # noqa: BLE001
            notes.append("optimiser state could not be restored; momentum restarted")

    def _on_oom() -> None:
        # An OOM inside scaler.step() leaves the scaler mid-update.
        scaler_ref[0] = torch.amp.GradScaler("cuda", enabled=use_amp)
        opt.zero_grad(set_to_none=True)

    st["on_oom"] = _on_oom

    def lr_at(epoch: int, step: int) -> float:
        lr = cfg.lr * st["batch"] / 64
        t = epoch + step / st["steps"]
        if t < cfg.warmup_epochs:
            return lr * (t / cfg.warmup_epochs) ** 2
        p = (t - cfg.warmup_epochs) / max(1e-9, cfg.epochs - cfg.warmup_epochs)
        return lr * 0.05 + lr * 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    @torch.no_grad()
    def evaluate(net) -> tuple[float, list[float]]:
        net.eval()
        preds, gts = [], []
        for si in val_s:
            imgs, _ = _make_batch(samples, [si], cfg.imgsz, train=False)
            out = net(imgs.to(device))
            preds.append(_decode_rows(out, 0.01, cfg.nms_thres, len(classes)))
            gts.append(_gt_letterbox(samples[si], cfg.imgsz))
            _gpu_turn(st)
        net.train()
        return ap50(preds, gts, len(classes))

    epoch_secs: list[float] = []

    def report_progress(index: int, phase: str) -> None:
        if not progress:
            return
        eta = None
        if epoch_secs:
            # A QAT epoch takes about 1.8x a float epoch.
            per = float(np.mean(epoch_secs[-5:]))
            left_float = max(0, cfg.epochs - index)
            left_qat = max(0, total_epochs - max(index, cfg.epochs))
            eta = per * (left_float + (1.8 if index < cfg.epochs else 1.0) * left_qat)
        progress({"index": index, "total": total_epochs, "phase": phase,
                  "eta_s": (round(eta) if eta is not None else None)})

    best_ap = float(ck.get("best_ap", -1.0)) if ck is not None else -1.0
    best_state = ck.get("best_state") if ck is not None else None
    start_epoch = 0
    if ck is not None:
        start_epoch = cfg.epochs if ck.get("phase") == "qat" else int(ck.get("epoch") or 0)
        notes.append(f"resumed at epoch {start_epoch + 1}")
        if info.get("splitAdded"):
            notes.append(f"{info['splitAdded']} image(s) added since the run started went to train")
        if info.get("splitGone"):
            notes.append(f"{info['splitGone']} image(s) deleted or left without boxes since the "
                         "run started were dropped")
    write_history("training")
    beat_thread.start()

    def save_checkpoint(phase: str, epoch: int) -> None:
        _save_ckpt(out_dir / "ckpt_last.pt", {
            "phase": phase, "epoch": epoch, "run_id": run_id,
            "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "opt": opt.state_dict() if phase == "float" else None,
            "scaler": scaler_ref[0].state_dict() if (phase == "float" and use_amp) else None,
            "best_ap": best_ap, "best_state": best_state,
            "history": history, "notes": notes, "classes": classes,
            "cfg": asdict(cfg), "batch": st["batch"], "init": init_info,
            "pretrained": pretrained_used, "cocoWeights": coco_weights,
            "split": split_names,
        })

    def float_step(epoch: int):
        def step_fn(step: int, imgs, targets) -> float:
            for g in opt.param_groups:
                g["lr"] = lr_at(epoch, step)
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=use_amp):
                out = model(imgs, targets)
                loss = out["total_loss"]
            opt.zero_grad(set_to_none=True)
            scaler = scaler_ref[0]
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            return float(loss.detach())
        return step_fn

    try:
        model.train()
        for epoch in range(start_epoch, cfg.epochs):
            _check_cancel(cancel)
            random.seed(cfg.seed + epoch + 1)  # same shuffles after a resume
            t_ep = time.time()
            mean_loss = _epoch(samples, train_s, st, cfg.imgsz, float_step(epoch), cancel, notes)
            mean_ap = None
            if (epoch + 1) % 5 == 0 or epoch + 1 == cfg.epochs:
                mean_ap, _ = evaluate(model)
                if mean_ap >= best_ap:
                    best_ap = mean_ap
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epoch_secs.append(time.time() - t_ep)
            history.append({"epoch": epoch + 1, "phase": "float", "loss": round(mean_loss, 4),
                            "lr": round(lr_at(epoch, st["steps"] - 1), 6),
                            "ap50": (round(mean_ap, 4) if mean_ap is not None else None),
                            "seconds": round(epoch_secs[-1], 1), "batch": st["batch"]})
            if (epoch + 1) % max(1, cfg.checkpoint_every) == 0 or epoch + 1 == cfg.epochs:
                save_checkpoint("float", epoch + 1)
            write_history("training")
            report_progress(epoch + 1, f"loss {mean_loss:.2f}"
                            + (f" · AP50 {mean_ap:.3f}" if mean_ap is not None else ""))
        if best_state is not None:
            model.load_state_dict(best_state)
        _check_cancel(cancel)
        fp32_ap, fp32_per_class = evaluate(model)

        # Float export with a raw head: ST Edge AI cannot shape-infer the
        # in-graph decode's ScatterND/Expand/Where.
        model.head.decode_in_inference = False
        fp32_onnx = out_dir / f"{slug}_fp32.onnx"
        _export_onnx(model, fp32_onnx, cfg.imgsz, qat=False)
        model.to(device)
        # QAT evaluation needs the in-graph decode.
        model.head.decode_in_inference = True

        # Quantisation-aware fine-tuning
        qat_ap = None
        if cfg.qat_epochs > 0:
            _check_cancel(cancel)
            save_checkpoint("qat", cfg.epochs)  # a crash here resumes straight into QAT
            write_history("training")
            qat_ap = _qat_finetune(model, samples, train_s, st, cfg, device, evaluate,
                                   report_progress, cancel, history, notes, epoch_secs,
                                   lambda: write_history("training"))

        torch.save({"model": model.state_dict(), "classes": classes, "imgsz": cfg.imgsz,
                    "size": cfg.size, "qat": cfg.qat_epochs > 0, "cocoWeights": coco_weights},
                   out_dir / "best.pt")

        _check_cancel(cancel)
        write_history("exporting")
        int8_onnx = out_dir / f"{slug}_int8.onnx"
        if cfg.qat_epochs > 0:
            # The QAT graph exports its own Q/DQ pairs; no calibration pass.
            _export_onnx(model, int8_onnx, cfg.imgsz, qat=True)
            _fold_weight_quant(int8_onnx)
            contract = _verify_input_contract(int8_onnx)
        else:
            contract = _ptq_quantize(fp32_onnx, int8_onnx, samples, train_s, cfg, notes)
        int8_ap, int8_per_class, preds, gts = _score_onnx(int8_onnx, samples, val_s, cfg, classes)

        # Threshold and failure analysis
        _check_cancel(cancel)
        import train_eval

        if cfg.conf_thres is None:
            threshold = train_eval.calibrate_threshold(preds, gts, len(classes))
        else:
            threshold = {"conf": float(cfg.conf_thres), "calibrated": False, "sweep":
                         train_eval.calibrate_threshold(preds, gts, len(classes))["sweep"]}
            tp = fp = fn = 0
            for row in train_eval.class_report(preds, gts, threshold["conf"], classes):
                tp, fp, fn = tp + row["tp"], fp + row["fp"], fn + row["fn"]
            p = tp / (tp + fp) if tp + fp else 0.0
            r = tp / (tp + fn) if tp + fn else 0.0
            threshold.update({"precision": round(p, 4), "recall": round(r, 4),
                              "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0})
        conf = float(threshold["conf"])
        class_rep = train_eval.class_report(preds, gts, conf, classes)
        confusion = train_eval.confusion_matrix(preds, gts, conf, len(classes))
        failures_dir = out_dir / "failures"
        gallery = train_eval.write_failure_gallery(
            preds, gts, conf, classes, [samples[i].name for i in val_s],
            lambda i: _letterboxed_rgb(samples[val_s[i]], cfg.imgsz),
            failures_dir, limit=cfg.failure_limit)
        _write_json_atomic(out_dir / "failures.json", gallery)
        _write_json_atomic(out_dir / "predictions.json",
                           _val_predictions(samples, val_s, preds, gts, conf, cfg, project_id, classes))

        _check_cancel(cancel)
        neuron6 = _verify_neuron6(int8_onnx, out_dir) if cfg.verify_neuron6 else {
            "verified": False, "available": None, "reason": "verification disabled"}
        if neuron6.get("error"):
            # Rejected by an installed compiler: no bundle; the ONNX and scores stay.
            _write_json_atomic(out_dir / "report.json", {
                "run": run_id, "model": spec["dir"], "size": cfg.size, "exportRejected": True,
                "ap50_fp32": round(fp32_ap, 4), "ap50_int8": round(int8_ap, 4),
                "threshold": threshold, "neuron6": neuron6, "history": history})
            run_state["final"] = True
            write_history("rejected", neuron6["error"])
            (out_dir / "ckpt_last.pt").unlink(missing_ok=True)
            raise ValueError(f"export rejected by ST Edge AI: {neuron6['error']}")

        report = {
            "run": run_id,
            "model": spec["dir"],
            "size": cfg.size,
            "classes": classes,
            "images": {"train": len([i for i in train_s if samples[i].source == "original"]),
                       "val": len(val_s), "augmented": int(info.get("augmented") or 0),
                       "background": int(info.get("background") or 0),
                       "scoredOnTrain": scored_on_train},
            "split": {"ratio": cfg.train_split, "stratified": bool(cfg.stratify and not cfg.eval_project),
                      "evalProject": cfg.eval_project},
            "epochs": cfg.epochs,
            "imgsz": cfg.imgsz,
            "batch": st["batch"],
            "pretrained": pretrained_used,
            "initRun": (init_info or {}).get("run"),
            "cocoWeights": coco_weights,
            "resumed": ck is not None,
            "head": "raw",
            "input": {**contract, "layout": "NCHW float [0,1]",
                      "note": "stedgeai re-encodes to uint8 channel-last"},
            "ap50_fp32": round(fp32_ap, 4),
            "ap50_int8": round(int8_ap, 4),
            "qat_epochs": cfg.qat_epochs,
            "ap50_qat_sim": (round(qat_ap, 4) if qat_ap is not None else None),
            # None = no ground truth for that class in the val split.
            "per_class_fp32": {c: (round(a, 4) if a is not None else None)
                               for c, a in zip(classes, fp32_per_class)},
            "per_class_int8": {c: (round(a, 4) if a is not None else None)
                               for c, a in zip(classes, int8_per_class)},
            "threshold": threshold,
            "classReport": class_rep,
            "confusion": confusion,
            "confusionLabels": classes + ["background"],
            "failures": gallery,
            "history": history,
            "notes": notes,
            "warnings": checks["warnings"],
            "neuron6": neuron6,
            "minutes": round((time.time() - t0) / 60, 1),
        }
        _write_bundle(out_dir, slug, dataset_name, classes, cfg, report, int8_onnx, conf)
        (out_dir / "ckpt_last.pt").unlink(missing_ok=True)
        run_state["final"] = True
        write_history("done")
        return json.loads((out_dir / "report.json").read_text("utf-8"))
    except Cancelled:
        if not run_state["final"]:
            run_state["final"] = True
            write_history("interrupted", "cancelled")
        raise
    except Exception as e:  # noqa: BLE001
        if not run_state["final"]:
            run_state["final"] = True
            write_history("interrupted" if (out_dir / "ckpt_last.pt").is_file() else "failed",
                          f"{type(e).__name__}: {e}")
        raise
    finally:
        stop_beat.set()
        if beat_thread.is_alive():
            beat_thread.join(timeout=2.0)


# Neuro N6 build verification

def _verify_neuron6(int8_onnx: Path, out_dir: Path) -> dict:
    """Compile the export with ST Edge AI as the core's pragma pipeline does;
    the logs go to <run>/neuron6/."""
    import neuron6_verify

    try:
        return neuron6_verify.verify(int8_onnx, out_dir / "neuron6")
    except Exception as e:  # noqa: BLE001
        return {"verified": False, "available": True, "error": f"{type(e).__name__}: {e}"}


# Export helpers

def _export_onnx(model: YOLOX, path: Path, imgsz: int, qat: bool) -> None:
    """Trace to ONNX on CPU. With qat=True activation fake quants export as
    Q/DQ pairs; conv weights come out float, for _fold_weight_quant."""
    set_qat(model, qat)
    model.eval().cpu()
    model.head.decode_in_inference = False
    torch.onnx.export(
        model, torch.zeros(1, 3, imgsz, imgsz), str(path), opset_version=17,
        input_names=["images"], output_names=["raw"], dynamo=False,
    )


def _fold_weight_quant(path: Path) -> int:
    """Give a QAT export int8 weights. Constant Q/DQ parameters become
    initializers, which ST Edge AI requires, and each float conv weight gets
    per-channel symmetric int8 with scale max|w| / 127, as qat.fq_weight
    simulates. Returns the number of convs re-quantised."""
    import onnx
    from onnx import helper, numpy_helper as nh

    m = onnx.load(str(path))
    g = m.graph
    keep = []
    for n in g.node:
        if n.op_type == "Constant" and len(n.attribute) == 1 and n.attribute[0].name == "value":
            t = onnx.TensorProto()
            t.CopyFrom(n.attribute[0].t)
            t.name = n.output[0]
            g.initializer.append(t)
        else:
            keep.append(n)
    init = {i.name: i for i in g.initializer}
    producer = {o: n for n in keep for o in n.output}

    # The exporter shares identical parameters through Identity nodes; each
    # conv gets its own copy.
    def _resolve(name):
        seen = 0
        while name not in init and name in producer and producer[name].op_type == "Identity" and seen < 8:
            name = producer[name].input[0]
            seen += 1
        return name if name in init else None

    for n in keep:
        if n.op_type == "Conv":
            for k in (1, 2):
                if len(n.input) > k and n.input[k] not in init:
                    src = _resolve(n.input[k])
                    if src is not None:
                        copy = onnx.TensorProto()
                        copy.CopyFrom(init[src])
                        copy.name = f"{src}__{n.name.strip('/').replace('/', '_')}_{k}"
                        g.initializer.append(copy)
                        init[copy.name] = copy
                        n.input[k] = copy.name
    keep = [n for n in keep if not (n.op_type == "Identity" and n.output[0] not in
                                    {i for m_ in keep for i in m_.input})]

    out_nodes, count = [], 0
    for n in keep:
        if n.op_type == "Conv" and len(n.input) >= 2 and n.input[1] in init:
            w_t = init[n.input[1]]
            if w_t.data_type == onnx.TensorProto.FLOAT:
                w = nh.to_array(w_t)
                amax = np.maximum(np.abs(w).reshape(w.shape[0], -1).max(axis=1), 1e-8)
                scale = (amax / 127.0).astype(np.float32)
                q = np.clip(np.round(w / scale.reshape(-1, *([1] * (w.ndim - 1)))),
                            -128, 127).astype(np.int8)
                base = n.input[1]
                g.initializer.remove(w_t)
                g.initializer.extend([
                    nh.from_array(q, base + "_q"),
                    nh.from_array(scale, base + "_scale"),
                    nh.from_array(np.zeros(w.shape[0], np.int8), base + "_zp"),
                ])
                out_nodes.append(helper.make_node(
                    "DequantizeLinear", [base + "_q", base + "_scale", base + "_zp"],
                    [base + "_dq"], name=base + "_DQ", axis=0))
                n.input[1] = base + "_dq"
                count += 1
                # Bias as int32 with scale x_scale * w_scale, as in ORT's QDQ output.
                src = producer.get(n.input[0])
                if (len(n.input) >= 3 and n.input[2] in init
                        and init[n.input[2]].data_type == onnx.TensorProto.FLOAT
                        and src is not None and src.op_type == "DequantizeLinear"
                        and src.input[1] in init):
                    x_scale = float(np.ravel(nh.to_array(init[src.input[1]]))[0])
                    b_t = init[n.input[2]]
                    b = nh.to_array(b_t)
                    b_scale = (scale * x_scale).astype(np.float32)
                    # float64: float32 cannot hold 2**31-1, so a saturated bias wraps to INT_MIN.
                    b_q = np.clip(np.round(b.astype(np.float64) / b_scale.astype(np.float64)),
                                  -2**31, 2**31 - 1).astype(np.int32)
                    bbase = n.input[2]
                    g.initializer.remove(b_t)
                    g.initializer.extend([
                        nh.from_array(b_q, bbase + "_q"),
                        nh.from_array(b_scale, bbase + "_scale"),
                        nh.from_array(np.zeros(len(b), np.int32), bbase + "_zp"),
                    ])
                    out_nodes.append(helper.make_node(
                        "DequantizeLinear", [bbase + "_q", bbase + "_scale", bbase + "_zp"],
                        [bbase + "_dq"], name=bbase + "_DQ", axis=0))
                    n.input[2] = bbase + "_dq"
        out_nodes.append(n)
    del g.node[:]
    g.node.extend(out_nodes)
    onnx.save(m, str(path))
    return count


def _qat_finetune(model: YOLOX, samples, train_s, st: dict, cfg: TrainConfig, device,
                  evaluate, report_progress, cancel, history: list[dict], notes: list[str],
                  epoch_secs: list[float], on_epoch: Callable[[], None]) -> float:
    """Fine-tune in fp32 with int8 simulated throughout the graph. Returns the
    simulated-int8 AP@0.5 on the val split."""
    from vendor.yolox.qat import reset_observers, set_calibrate, unseeded

    set_qat(model, True)
    reset_observers(model)  # a warm start must not carry another run's ranges
    batch = st["batch"]
    # Seed observers on both branches: the head's output fake quant is inference-only.
    set_calibrate(model, True)
    with torch.no_grad():
        seed = train_s[: min(len(train_s), max(batch * 8, cfg.calib_images))]
        model.train()
        for imgs, targets in _batch_iter(samples, seed, batch, cfg.imgsz, train=False):
            model(imgs.to(device), targets.to(device))
            _gpu_turn(st)
        model.eval()
        for imgs, _t in _batch_iter(samples, seed, batch, cfg.imgsz, train=False):
            model(imgs.to(device))
            _gpu_turn(st)
    set_calibrate(model, False)
    missing = unseeded(model)
    if missing:
        raise RuntimeError(f"QAT observers never saw data: {missing[:5]}")
    _check_cancel(cancel)
    model.train()
    lr = cfg.lr * st["batch"] / 64
    qat_lr = lr * 0.1
    opt = torch.optim.SGD(model.parameters(), lr=qat_lr, momentum=0.9,
                          nesterov=True, weight_decay=5e-4)
    ap = None

    def qat_step(qe: int):
        def step_fn(step: int, imgs, targets) -> float:
            p = (qe + step / st["steps"]) / max(1, cfg.qat_epochs)
            for g in opt.param_groups:
                g["lr"] = qat_lr * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            loss = model(imgs, targets)["total_loss"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            # With BatchNorm folded, gradient norms reach 100-200.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            return float(loss.detach())
        return step_fn

    for qe in range(cfg.qat_epochs):
        _check_cancel(cancel)
        random.seed(cfg.seed + cfg.epochs + qe + 1)
        t_ep = time.time()
        mean_loss = _epoch(samples, train_s, st, cfg.imgsz, qat_step(qe), cancel, notes)
        if qe + 1 == cfg.qat_epochs or (qe + 1) % 4 == 0:
            ap, _ = evaluate(model)
        epoch_secs.append(time.time() - t_ep)
        history.append({"epoch": cfg.epochs + qe + 1, "phase": "qat", "loss": round(mean_loss, 4),
                        "lr": round(float(opt.param_groups[0]["lr"]), 6),
                        "ap50": (round(ap, 4) if ap is not None else None),
                        "seconds": round(epoch_secs[-1], 1), "batch": st["batch"]})
        on_epoch()
        report_progress(cfg.epochs + qe + 1, f"QAT loss {mean_loss:.2f}"
                        + (f" · int8 AP50 {ap:.3f}" if ap is not None else ""))
    if ap is None:
        ap, _ = evaluate(model)
    return ap


# int8 PTQ fallback: ONNX Runtime static QDQ

def _ptq_quantize(fp32_path: Path, int8_path: Path, samples, train_s, cfg: TrainConfig,
                  notes: list[str] | None = None) -> dict:
    """Post-training quantisation for qat_epochs=0, percentile-calibrated on
    training images. Returns the input contract record."""
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod,
                                          QuantFormat, QuantType, quantize_static)

    calib_idxs = train_s[: cfg.calib_images]

    class Reader(CalibrationDataReader):
        def __init__(self):
            self._it = iter(calib_idxs)

        def get_next(self):
            si = next(self._it, None)
            if si is None:
                return None
            return {"images": _make_batch(samples, [si], cfg.imgsz, train=False)[0].numpy()}

    def _run(method, extra):
        quantize_static(
            str(fp32_path), str(int8_path), Reader(),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
            calibrate_method=method,
            extra_options=extra,
        )

    try:
        _run(CalibrationMethod.Percentile, {"CalibPercentile": cfg.calib_percentile})
    except Exception as e:  # noqa: BLE001
        # The percentile calibrator fails on a tensor with no range; MinMax does not.
        if notes is not None:
            notes.append(f"percentile calibration failed ({type(e).__name__}); used MinMax")
        _run(CalibrationMethod.MinMax, {})
    return _verify_input_contract(int8_path)


def _score_onnx(path: Path, samples, val_s, cfg: TrainConfig, classes: list[str]
                ) -> tuple[float, list[float], list[np.ndarray], list[np.ndarray]]:
    """AP@0.5 of an exported raw-head ONNX on the val split, with the float
    model's decode and NMS. Also returns the predictions down to score 0.01
    and the ground truth, in letterbox space."""
    sess = open_onnx(path)
    preds, gts = [], []
    for si in val_s:
        imgs, _ = _make_batch(samples, [si], cfg.imgsz, train=False)
        raw = sess.run(None, {"images": imgs.numpy()})[0]
        out = torch.from_numpy(_decode_raw(raw, cfg.imgsz))
        preds.append(_decode_rows(out, 0.01, cfg.nms_thres, len(classes)))
        gts.append(_gt_letterbox(samples[si], cfg.imgsz))
    mean, per_class = ap50(preds, gts, len(classes))
    return mean, per_class, preds, gts


def _val_predictions(samples: list[Sample], val_s: list[int], preds: list[np.ndarray],
                     gts: list[np.ndarray], conf: float, cfg: TrainConfig, project_id: str,
                     classes: list[str]) -> dict:
    """Each val image's truth and its predictions at `conf`, in the image's
    pixels. A prediction's `kind` is tp, wrong or fp; a truth box nothing
    found is `missed`."""
    import train_eval

    images = []
    for k, si in enumerate(val_s):
        s = samples[si]
        h, w = _dims(s)
        scale, dx, dy = _letterbox_fit(w, h, cfg.imgsz)
        off = np.array([dx, dy, dx, dy], dtype=np.float32)
        pred = np.asarray(preds[k], dtype=np.float32).reshape(-1, 6)
        gt = np.asarray(gts[k], dtype=np.float32).reshape(-1, 5)
        matches = train_eval.match_image(pred, gt, conf)
        kind = {pi: kd for kd, pi, _gi in matches if pi >= 0}
        missed = {gi for kd, _pi, gi in matches if kd == "fn"}
        gbox = np.round((gt[:, :4] - off) / scale, 1).tolist()
        pbox = np.round((pred[:, :4] - off) / scale, 1).tolist()
        images.append({
            "project": cfg.eval_project if s.source == "eval" else project_id,
            "image": s.name, "width": w, "height": h,
            "truth": [{"box": gbox[gi], "class": int(g[4]), "missed": gi in missed}
                      for gi, g in enumerate(gt)],
            "predicted": [{"box": pbox[pi], "class": int(p[5]), "score": round(float(p[4]), 3),
                           "kind": kind[pi]} for pi, p in enumerate(pred) if pi in kind],
        })
    return {"conf": conf, "classes": classes, "images": images}


# Neuro N6 bundle

_C_KEYWORDS = frozenset("""auto break case char const continue default do double else enum extern
float for goto if inline int long register restrict return short signed sizeof static struct
switch typedef union unsigned void volatile while bool true false""".split())


def _c_slug(name: str) -> str:
    """Dataset name -> a slug valid as a C identifier, a macro prefix and an
    Arduino sketch name, which stops at 63 characters."""
    slug = re.sub(r"[^a-z0-9]+", "_", (name or "").lower()).strip("_")[:40].rstrip("_") or "model"
    if slug[0].isdigit() or slug in _C_KEYWORDS or slug.upper() in _C_KEYWORDS:
        slug = "pk_" + slug
    return slug


def _c_str(s: str) -> str:
    """Escape a class name for a C string literal in the UTF-8 labels header.
    A control character becomes octal escapes of its UTF-8 bytes: an octal
    escape ends after three digits, a hex one would take in a digit after it."""
    out = []
    for ch in s:
        if ch in '\\"':
            out.append("\\" + ch)
        elif unicodedata.category(ch) == "Cc":
            out.extend(f"\\{b:03o}" for b in ch.encode("utf-8"))
        else:
            out.append(ch)
    return "".join(out)


def _c_comment(s: str) -> str:
    """Printable ASCII that cannot end a C block comment."""
    return re.sub(r"[^\x20-\x7e]", "?", s or "").replace("*/", "* /")


def _sketch(slug: str, dataset_name: str, classes: list[str], cfg: TrainConfig,
            report: dict, model_file: str) -> str:
    """The exported sketch. Everything that varies comes from <slug>_labels.h."""
    U = slug.upper()
    name = _c_comment(dataset_name).strip() or slug
    shown = [_c_comment(c) for c in classes[:6]]
    if len(classes) > 6:
        shown.append(f"and {len(classes) - 6} more")
    ap = report.get("ap50_int8")
    core = ".".join(map(str, CORE_MIN))

    def para(text: str) -> list[str]:
        return textwrap.wrap(text, 76, initial_indent="  ", subsequent_indent="  ")

    return "\n".join([
        "/*",
        f"  {name} detector",
        "",
        *para(f"Detects {', '.join(shown)} in the camera picture and streams labelled "
              "boxes to Neuro Studio, over USB or WiFi."),
        "",
        "  Hardware: Neuro N6 with an OV5640 camera; the MAYA-W476 radio for WiFi.",
        "",
        *para(f"Model: {MODEL_SPECS[cfg.size]['label']}, {cfg.imgsz}x{cfg.imgsz} input, "
              f"PixelKit run {report.get('run')}."),
        *(para(f"AP@0.5 {ap:.3f} on its validation images, as int8.")
          if isinstance(ap, (int, float)) else []),
        *para(f"The sketch, {slug}_labels.h and {model_file} belong together."),
        "*/",
        "",
        f"// Stops the build on a core older than {core}, or on another board.",
        "#ifndef NEURON6_CORE_VERSION_AT_LEAST",
        f'#error "This sketch needs the Neuro N6 core {core} or later"',
        f"#elif !NEURON6_CORE_VERSION_AT_LEAST({', '.join(map(str, CORE_MIN))})",
        f'#error "This sketch needs the Neuro N6 core {core} or later"',
        "#endif",
        "",
        "// 0: stream over USB. 1: stream over WiFi from the board's own network: with",
        "// the board on USB, pick WiFi (the board's network) in Neuro Studio and join",
        "// the network it shows. The Serial Monitor prints its name and password too.",
        "#define STREAM_OVER_WIFI 0",
        "",
        "#include <OV5640_Arduino.h>",
        "#include <PostProcess.h>",
        "#if STREAM_OVER_WIFI",
        "#include <MayaW476.h>",
        "#endif",
        "",
        f'#include "{slug}_labels.h"',
        "",
        "// Compiled with ST Edge AI during the build; the weights are flashed with the sketch.",
        f'#pragma neuron6 model="{model_file}" name={slug}',
        "",
        "// OV5640 camera(CARRIER_TFT); for the TFT carrier.",
        "OV5640 camera;",
        "",
        f"NEURON6_DECLARE_MODEL({slug});",
        "",
        "IPostProcessor *detector = nullptr;",
        "",
        "// Minimum box score. Neuro Studio shows it as a slider; a value chosen there",
        f"// belongs in {U}_CONF_THRESHOLD.",
        f"static float confidence = {U}_CONF_THRESHOLD;",
        "",
        "void setup() {",
        "  VisionConfig cfg;",
        "  cfg.camera = CAMERA_VGA;",
        "  cfg.fps = FPS_45;",
        f"  cfg.nn.width  = {U}_INPUT_W;",
        f"  cfg.nn.height = {U}_INPUT_H;",
        "  // The whole picture, padded to a square with grey bars, as in training.",
        "  cfg.nn.aspect = ASPECT_LETTERBOX;",
        "",
        "#if STREAM_OVER_WIFI",
        "  // A network named after the board, with a password unique to it.",
        "  Serial.begin(115200);",
        "  MayaW476.beginAP(MayaW476.getHostname(), MayaW476.devicePassword());",
        "  cfg.transport = TRANSPORT_WIFI_SOFTAP;",
        "#else",
        "  // TRANSPORT_LCD draws the boxes on the TFT carrier's display instead.",
        "  cfg.transport = TRANSPORT_USB;",
        "#endif",
        "",
        '  n6_tune_float("confidence", &confidence, 0.01f, 0.99f);',
        f"  detector = PostProcess_PixelKitYOLOX(confidence, {U}_NMS_THRESHOLD,",
        f"                                       {U}_CLASSES, {U}_NUM_CLASSES);",
        f"  Vision.begin(&NN_Instance_{slug}, detector, cfg);",
        "}",
        "",
        "void loop() {",
        "#if STREAM_OVER_WIFI",
        "  // The network name and password every 10 s.",
        "  static uint32_t nextNote = 0;",
        "  if ((int32_t)(millis() - nextNote) >= 0) {",
        "    nextNote = millis() + 10000;",
        '    Serial.print("WiFi network ");',
        "    Serial.print(MayaW476.getHostname());",
        '    Serial.print(", password ");',
        "    Serial.println(MayaW476.devicePassword());",
        "  }",
        "#endif",
        "",
        "  // Calling the factory again updates the settings of the one instance.",
        f"  static float applied = {U}_CONF_THRESHOLD;",
        "  if (confidence != applied) {",
        "    applied = confidence;",
        f"    PostProcess_PixelKitYOLOX(confidence, {U}_NMS_THRESHOLD, {U}_CLASSES, {U}_NUM_CLASSES);",
        "  }",
        "",
        "  Vision.run();",
        "",
        "  // Each box: centre and size as fractions of the model's input, score and class.",
        "  const od_pp_outBuffer_t *boxes = nullptr;",
        "  const int n = PostProcess_ZooResults(detector, &boxes);",
        "  for (int i = 0; i < n; i++) {",
        f"    // {U}_CLASSES[boxes[i].class_index], score boxes[i].conf",
        "  }",
        "}",
        "",
    ])


def _write_bundle(out_dir: Path, slug: str, dataset_name: str, classes: list[str],
                  cfg: TrainConfig, report: dict, int8_onnx: Path, conf_thres: float) -> None:
    guard = f"{slug.upper()}_LABELS_H"
    thr = report.get("threshold") or {}
    if thr.get("calibrated"):
        thr_note = (f"/* Calibrated on the val split: best F1 {thr.get('f1')} "
                    f"(precision {thr.get('precision')}, recall {thr.get('recall')}). */")
    else:
        thr_note = "/* Confidence threshold as configured for this run. */"
    label = MODEL_SPECS[cfg.size]["label"]
    name = _c_comment(dataset_name).strip() or slug
    labels_h = "\n".join([
        f"#ifndef {guard}",
        f"#define {guard}",
        f"/* Generated by PixelKit: {label} trained on the {name} dataset. */",
        f"#define {slug.upper()}_NUM_CLASSES {len(classes)}",
        f"#define {slug.upper()}_INPUT_W {cfg.imgsz}",
        f"#define {slug.upper()}_INPUT_H {cfg.imgsz}",
        thr_note,
        f"#define {slug.upper()}_CONF_THRESHOLD {conf_thres:.2f}f",
        f"#define {slug.upper()}_NMS_THRESHOLD {cfg.nms_thres}f",
        f"static const char* const {slug.upper()}_CLASSES[{len(classes)}] = {{",
        *[f'  "{_c_str(c)}",' for c in classes],
        "};",
        f"#endif /* {guard} */",
        "",
    ])
    # LF on every OS: the sketch folder may end up in the core, which checks.
    (out_dir / f"{slug}_labels.h").write_text(labels_h, "utf-8", newline="\n")
    sketch = _sketch(slug, dataset_name, classes, cfg, report, int8_onnx.name)
    (out_dir / f"{slug}.ino").write_text(sketch, "utf-8", newline="\n")
    files = [out_dir / f"{slug}.ino", int8_onnx, out_dir / f"{slug}_labels.h"]
    # Every export is a modified YOLOX network; most also descend from Megvii's weights.
    coco = report.get("cocoWeights", bool(report.get("pretrained") or report.get("initRun")))
    weights = f" and fine-tuned from Megvii's COCO-pretrained {label} checkpoint" if coco else ""
    (out_dir / "NOTICE.txt").write_text(
        f"{int8_onnx.name} is a modified version of YOLOX. PixelKit built it from\n"
        f"the YOLOX network as modified by Ohm Lab Ltd{weights},\n"
        "trained it on your dataset and quantised it to int8.\n\n"
        "YOLOX: https://github.com/Megvii-BaseDetection/YOLOX\n"
        "Copyright (c) 2021-2022 Megvii Inc. All rights reserved.\n"
        "Changes: Copyright 2026 Ohm Lab Ltd.\n"
        "Licensed under the Apache License, Version 2.0; a copy is in LICENSE.txt.\n"
        "Keep this notice and LICENSE.txt with the model when you share it.\n",
        "utf-8", newline="\n")
    (out_dir / "LICENSE.txt").write_bytes(
        (Path(__file__).resolve().parent / "vendor" / "yolox" / "LICENSE").read_bytes())
    files += [out_dir / "NOTICE.txt", out_dir / "LICENSE.txt"]

    # One sketch folder, <slug>/, written under a temp name. report.json goes
    # in from memory and onto disk last: a run with report.json lists as done.
    zpath = out_dir / "export.zip"
    tmp = zpath.with_name(f"{zpath.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in files:
                zf.write(f, f"{slug}/{f.name}")
            zf.writestr(f"{slug}/report.json", _json_text(report))
        _replace_retry(tmp, zpath)
    finally:
        _discard(tmp)
    _write_json_atomic(out_dir / "report.json", report)
