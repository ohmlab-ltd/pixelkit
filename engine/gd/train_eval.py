"""Post-training evaluation for a detector run.

preds[i] is an (n, 6) array of x0, y0, x1, y1, score, cls rows and gts[i]
an (m, 5) array of x0, y0, x1, y1, cls, both in letterbox space."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(n,4) x (m,4) xyxy -> (n,m) IoU."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    xx0 = np.maximum(a[:, None, 0], b[None, :, 0])
    yy0 = np.maximum(a[:, None, 1], b[None, :, 1])
    xx1 = np.minimum(a[:, None, 2], b[None, :, 2])
    yy1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(xx1 - xx0, 0, None) * np.clip(yy1 - yy0, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def match_image(pred: np.ndarray, gt: np.ndarray, conf: float, iou_thr: float = 0.5
                ) -> list[tuple[str, int, int]]:
    """Greedy matching, best score first, of one image's predictions at or
    above `conf` to its ground truth: same-class matches at `iou_thr` first,
    then any class ("wrong"). Returns (kind, pred_index, gt_index) triples,
    -1 for none."""
    pred = np.asarray(pred, dtype=np.float32).reshape(-1, 6)
    gt = np.asarray(gt, dtype=np.float32).reshape(-1, 5)
    keep = np.where(pred[:, 4] >= conf)[0]
    keep = keep[np.argsort(-pred[keep, 4], kind="stable")]
    used = np.zeros(len(gt), dtype=bool)
    assigned: dict[int, tuple[str, int]] = {}
    ious = _iou_matrix(pred[:, :4], gt[:, :4]) if len(keep) and len(gt) else None
    if ious is not None:
        for pi in keep:
            pi = int(pi)
            row = ious[pi].copy()
            row[used] = -1.0
            cand = np.where((row >= iou_thr) & (gt[:, 4] == pred[pi, 5]))[0]
            if len(cand):
                gi = int(cand[np.argmax(row[cand])])
                assigned[pi] = ("tp", gi)
                used[gi] = True
        for pi in keep:
            pi = int(pi)
            if pi in assigned:
                continue
            row = ious[pi].copy()
            row[used] = -1.0
            cand = np.where(row >= iou_thr)[0]
            if len(cand):
                gi = int(cand[np.argmax(row[cand])])
                assigned[pi] = ("wrong", gi)
                used[gi] = True
    out: list[tuple[str, int, int]] = []
    for pi in keep:
        pi = int(pi)
        kind, gi = assigned.get(pi, ("fp", -1))
        out.append((kind, pi, gi))
    for gi in np.where(~used)[0]:
        out.append(("fn", -1, int(gi)))
    return out


def _counts(preds, gts, conf: float, n_classes: int, iou_thr: float = 0.5):
    """Per-class tp / fp / fn at a threshold, plus the confusion matrix."""
    tp = np.zeros(n_classes, dtype=np.int64)
    fp = np.zeros(n_classes, dtype=np.int64)
    fn = np.zeros(n_classes, dtype=np.int64)
    cm = np.zeros((n_classes + 1, n_classes + 1), dtype=np.int64)
    bg = n_classes
    for pred, gt in zip(preds, gts):
        pred = np.asarray(pred, dtype=np.float32).reshape(-1, 6)
        gt = np.asarray(gt, dtype=np.float32).reshape(-1, 5)
        for kind, pi, gi in match_image(pred, gt, conf, iou_thr):
            pc = int(pred[pi, 5]) if pi >= 0 else -1
            gc = int(gt[gi, 4]) if gi >= 0 else -1
            if not (0 <= pc < n_classes or pc == -1) or not (0 <= gc < n_classes or gc == -1):
                continue  # class id outside the dataset
            if kind == "tp":
                tp[pc] += 1
                cm[gc, pc] += 1
            elif kind == "wrong":
                fp[pc] += 1
                fn[gc] += 1
                cm[gc, pc] += 1
            elif kind == "fp":
                fp[pc] += 1
                cm[bg, pc] += 1
            else:
                fn[gc] += 1
                cm[gc, bg] += 1
    return tp, fp, fn, cm


def _prf(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    r = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return float(p), float(r), float(f1)


def calibrate_threshold(preds, gts, n_classes: int, *, lo: float = 0.05, hi: float = 0.95,
                        step: float = 0.02, fallback: float = 0.25) -> dict:
    """The confidence threshold with the best F1 over all classes pooled, at
    the middle of a tie. With F1 0 everywhere it returns `fallback` and
    "calibrated": False."""
    sweep = []
    exact = []  # unrounded f1, for the argmax
    t = lo
    while t <= hi + 1e-9:
        conf = round(t, 4)
        tp, fp, fn, _cm = _counts(preds, gts, conf, n_classes)
        p, r, f1 = _prf(int(tp.sum()), int(fp.sum()), int(fn.sum()))
        sweep.append({"conf": conf, "precision": round(p, 4), "recall": round(r, 4),
                      "f1": round(f1, 4)})
        exact.append(f1)
        t += step
    if not sweep or max(exact) <= 0.0:
        return {"conf": fallback, "f1": 0.0, "precision": 0.0, "recall": 0.0,
                "sweep": sweep, "calibrated": False, "fallback": True}
    top = max(exact)
    plateau = [i for i, f in enumerate(exact) if abs(f - top) < 1e-9]
    # The first contiguous run of best rows.
    run = [plateau[0]]
    for i in plateau[1:]:
        if i == run[-1] + 1:
            run.append(i)
        else:
            break
    best = sweep[run[len(run) // 2]]
    return {"conf": best["conf"], "f1": best["f1"], "precision": best["precision"],
            "recall": best["recall"], "sweep": sweep, "calibrated": True}


def class_report(preds, gts, conf: float, classes: list[str]) -> list[dict]:
    """Per-class counts at a threshold. `support` is the ground-truth box
    count; with support 0, recall is None."""
    n = len(classes)
    tp, fp, fn, _cm = _counts(preds, gts, conf, n)
    out = []
    for c in range(n):
        p, r, f1 = _prf(int(tp[c]), int(fp[c]), int(fn[c]))
        support = int(tp[c] + fn[c])
        out.append({
            "name": classes[c],
            "support": support,
            "tp": int(tp[c]), "fp": int(fp[c]), "fn": int(fn[c]),
            "precision": round(p, 4) if tp[c] + fp[c] > 0 else None,
            "recall": round(r, 4) if support > 0 else None,
            "f1": round(f1, 4) if support > 0 or tp[c] + fp[c] > 0 else None,
        })
    return out


def confusion_matrix(preds, gts, conf: float, n_classes: int) -> list[list[int]]:
    """Rows are truth, columns prediction; the last row and column are
    background."""
    _tp, _fp, _fn, cm = _counts(preds, gts, conf, n_classes)
    return cm.tolist()


def rank_failures(preds, gts, conf: float, limit: int = 24, iou_thr: float = 0.5) -> list[dict]:
    """Val images with errors at a threshold, worst first, as {"index",
    "fn", "fp", "wrong", "matches"}."""
    rows = []
    for i, (pred, gt) in enumerate(zip(preds, gts)):
        m = match_image(pred, gt, conf, iou_thr)
        n_fn = sum(1 for k, _p, _g in m if k == "fn")
        n_fp = sum(1 for k, _p, _g in m if k == "fp")
        n_wrong = sum(1 for k, _p, _g in m if k == "wrong")
        if n_fn + n_fp + n_wrong == 0:
            continue
        rows.append({"index": i, "fn": n_fn, "fp": n_fp, "wrong": n_wrong, "matches": m})
    # Misses and "wrong" matches weigh double.
    rows.sort(key=lambda r: (-(2 * r["fn"] + 2 * r["wrong"] + r["fp"]), -r["fn"], r["index"]))
    return rows[:limit]


# RGB colours for the failure images.
_COL_GT = (40, 200, 90)        # ground truth, thin
_COL_TP = (70, 130, 240)       # correct detection, thin
_COL_FP = (230, 60, 60)        # spurious detection
_COL_FN = (250, 200, 40)       # missed ground truth
_COL_WRONG = (250, 130, 30)    # right place, wrong class


def render_failure(img_rgb: np.ndarray, pred: np.ndarray, gt: np.ndarray,
                   matches: list[tuple[str, int, int]], classes: list[str],
                   out_path: Path, view: int = 384) -> dict:
    """Draw one val image's errors on its letterboxed input and write a
    JPEG. Returns the items drawn, boxes in letterbox space."""
    pred = np.asarray(pred, dtype=np.float32).reshape(-1, 6)
    gt = np.asarray(gt, dtype=np.float32).reshape(-1, 5)
    h, w = img_rgb.shape[:2]
    s = view / max(1, max(h, w))
    canvas = cv2.resize(img_rgb, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_LINEAR)
    canvas = np.ascontiguousarray(canvas)

    def _name(c: int) -> str:
        return classes[c] if 0 <= c < len(classes) else str(c)

    def _rect(box, colour, thick, text=None):
        x0, y0, x1, y1 = [int(round(float(v) * s)) for v in box]
        cv2.rectangle(canvas, (x0, y0), (x1, y1), colour, thick)
        if text:
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            ty = y0 - 3 if y0 - th - 6 >= 0 else y1 + th + 4
            cv2.rectangle(canvas, (x0, ty - th - 3), (x0 + tw + 4, ty + 2), colour, -1)
            cv2.putText(canvas, text, (x0 + 2, ty - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (20, 20, 20), 1, cv2.LINE_AA)

    items: list[dict] = []
    for g in gt:
        _rect(g[:4], _COL_GT, 1)
    for kind, pi, gi in matches:
        if kind == "tp":
            _rect(pred[pi, :4], _COL_TP, 1)
    for kind, pi, gi in matches:
        if kind == "fp":
            p = pred[pi]
            _rect(p[:4], _COL_FP, 2, f"{_name(int(p[5]))} {p[4]:.2f} (spurious)")
            items.append({"kind": "fp", "class": _name(int(p[5])), "score": round(float(p[4]), 3),
                          "box": [round(float(v), 1) for v in p[:4]]})
        elif kind == "fn":
            g = gt[gi]
            _rect(g[:4], _COL_FN, 2, f"{_name(int(g[4]))} (missed)")
            items.append({"kind": "fn", "class": _name(int(g[4])), "score": None,
                          "box": [round(float(v), 1) for v in g[:4]]})
        elif kind == "wrong":
            p, g = pred[pi], gt[gi]
            _rect(p[:4], _COL_WRONG, 2, f"{_name(int(g[4]))} as {_name(int(p[5]))} {p[4]:.2f}")
            items.append({"kind": "wrong", "class": _name(int(g[4])), "predicted": _name(int(p[5])),
                          "score": round(float(p[4]), 3),
                          "box": [round(float(v), 1) for v in p[:4]]})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR),
                           [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise RuntimeError("could not encode failure image")
    buf.tofile(str(out_path))
    return {"items": items, "width": int(canvas.shape[1]), "height": int(canvas.shape[0])}


def write_failure_gallery(preds, gts, conf: float, classes: list[str], names: list[str],
                          load_letterboxed: Callable[[int], np.ndarray | None],
                          out_dir: Path, limit: int = 24) -> list[dict]:
    """Render the worst `limit` val images to <out_dir>/NN_<stem>.jpg and
    return the gallery manifest. `load_letterboxed(i)` returns image i's
    letterboxed RGB, or None to skip it."""
    import re

    gallery = []
    for k, row in enumerate(rank_failures(preds, gts, conf, limit=limit)):
        i = row["index"]
        img = load_letterboxed(i)
        if img is None:
            continue
        stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(names[i]).stem)[:48] or "image"
        fname = f"{k:02d}_{stem}.jpg"
        meta = render_failure(img, preds[i], gts[i], row["matches"], classes, out_dir / fname)
        gallery.append({
            "file": fname, "image": names[i], "index": i,
            "fn": row["fn"], "fp": row["fp"], "wrong": row["wrong"],
            **meta,
        })
    return gallery
