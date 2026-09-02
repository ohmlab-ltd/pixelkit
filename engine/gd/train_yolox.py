"""YOLOX-Nano training + int8 ONNX export for the Neuro N6.

Everything runs locally against one PixelKit dataset:

    run = train(project_id, TrainConfig(...), progress=cb, cancel=ev)

  1. Builds YOLOX-Nano (vendor/yolox, Apache-2.0) with the dataset's
     class count; optionally warm-starts from Megvii's COCO nano
     checkpoint (downloaded once into <workspace>/weights/yolox/).
  2. Trains on the dataset's labelled images with the SAME
     deterministic sha1 train/val split the exporters use, light
     augmentation (hflip + HSV + scale jitter), SGD + cosine schedule,
     AMP on CUDA.
  3. Evaluates VOC-style AP@0.5 per class on the val split (no
     pycocotools dependency).
  4. Exports fp32 ONNX (decoded head: [cx,cy,w,h,obj,cls...] rows) and
     a full-integer QDQ int8 ONNX calibrated on the dataset's own
     images, then re-scores BOTH on the val split so report.json
     states the float->int8 mAP delta honestly.
  5. Writes the Neuro N6 bundle: model + labels.h + sketch snippet +
     report.json + export.zip under <dataset>/models/yolox_nano/<run>/.

INPUT CONTRACT (Neuro N6 platform contract, not a model choice). The
DCMIPP DMAs packed interleaved RGB888 straight into the model's input
buffer, and stedgeai is told `--input-data-type uint8
--inputs-ch-position chlast`. That re-encoding is only correct when the
graph input is quantised ASYMMETRICALLY over [0,1]: scale 1/255,
zero-point -128 (int8) == uint8 zero-point 0. So the network is trained
on [0,1] tensors (RGB, letterboxed with 114/255 gray) and calibration
carries a black and a white frame, which puts MinMax exactly on [0,1].
_verify_input_contract() then refuses to emit a model that misses it -
this failure is silent on device, and stedgeai rejects an unquantised
input outright.

OUTPUT: raw head rows, decoded on the M55 (see yolox_pp.c), because
YOLOX's in-graph decode exports dynamic-shape ops (ScatterND/Expand/
Where) that ST Edge AI cannot shape-infer. Rows are
[dx, dy, dw, dh, obj, cls...] with obj/cls already sigmoided and the
box terms in grid units.
"""
from __future__ import annotations

import io
import json
import math
import queue
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch

import store
import workspace
from vendor.yolox import YOLOX, YOLOPAFPN, YOLOXHead, postprocess
from vendor.yolox.network_blocks import BaseConv
from vendor.yolox.qat import set_qat

PRETRAINED_URL = (
    "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth"
)

# Mirror of the exporters' split rule (server._split_for) so an image
# lands in the same split whether it's exported or trained on.
import hashlib


def _split_for(name: str, train_split: float) -> str:
    h = int(hashlib.sha1(name.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "train" if h < train_split else "val"


@dataclass
class TrainConfig:
    epochs: int = 50
    imgsz: int = 256           # square input, matches the dataset input-shape setting
    batch: int = 16
    lr: float = 0.01           # scaled by batch/64 like upstream
    train_split: float = 0.8
    device: str = "cuda"
    pretrained: bool = True
    conf_thres: float = 0.25   # documented default for the C decoder
    nms_thres: float = 0.45
    # Quantisation-aware fine-tuning after the float run. Measured on
    # VOC2007: post-training quantisation of this depthwise backbone
    # loses 8-17 AP@0.5 points whatever the calibration (per-tensor
    # activation ranges are outlier-dominated; weights quantise for
    # free), so the network is fine-tuned with int8 simulated in the
    # graph and exported straight from that. 0 = PTQ only.
    qat_epochs: int = 8
    # PTQ fallback (qat_epochs=0): percentile calibration on natural
    # images was the best of the calibrators tried (MinMax -17, moving
    # average -11, percentile 99.9 on 200 images -8 points).
    calib_images: int = 200
    calib_percentile: float = 99.9
    seed: int = 0
    warmup_epochs: int = 3
    # Compile the int8 export with ST Edge AI when it is installed and
    # record epochs / weights / pool headroom in report.json. When the
    # compiler is present and rejects the model, the export is refused.
    verify_neuron6: bool = True


@dataclass
class Sample:
    path: Path
    boxes: np.ndarray  # (n, 4) xyxy pixels
    classes: np.ndarray  # (n,)
    split: str
    w: int = 0
    h: int = 0


def _box_xyxy(b: dict) -> tuple[float, float, float, float] | None:
    """Mirror of the export builders' reader: flat x0..y1 keys
    (editedBoxes), a `box` list (V2 auto detections - the resolver's
    canonical output), or a legacy `box_xyxy` list."""
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
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _box_label(b: dict) -> str:
    for k in ("label", "predLabel", "pred_label", "gdLabel", "gd_label"):
        v = b.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip().lower()
    return ""


def load_samples(project_id: str, train_split: float) -> tuple[list[Sample], list[str]]:
    """Labelled images + the class list (dataset tag order)."""
    m = store.load(project_id)
    if not m:
        raise ValueError("dataset not found")
    classes = [t.strip().lower() for t in (m.get("tags") or []) if t and t.strip()]
    if not classes:
        raise ValueError("dataset has no labels")
    cls_idx = {c: i for i, c in enumerate(classes)}
    ds_dir = store.dataset_dir(project_id)
    samples: list[Sample] = []
    for imp in m.get("imports") or []:
        if not isinstance(imp, dict):
            continue
        fname = imp.get("filename")
        if not fname:
            continue
        boxes_src = imp.get("editedBoxes") if imp.get("editedBoxesSet") else (
            imp.get("editedBoxes") or imp.get("detections")
        )
        boxes, cls = [], []
        for b in boxes_src or []:
            if not isinstance(b, dict):
                continue
            rect = _box_xyxy(b)
            label = _box_label(b)
            if rect is None or label not in cls_idx:
                continue
            boxes.append(rect)
            cls.append(cls_idx[label])
        if not boxes:
            continue
        path = ds_dir / "images" / fname
        if not path.is_file():
            continue
        samples.append(
            Sample(
                path=path,
                boxes=np.array(boxes, dtype=np.float32),
                classes=np.array(cls, dtype=np.float32),
                split=_split_for(fname, train_split),
                w=int(imp.get("width") or 0),
                h=int(imp.get("height") or 0),
            )
        )
    return samples, classes


# ── preprocessing ────────────────────────────────────────────────────

def letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    """RGB uint8 -> (size,size,3) with 114 padding. Returns (img, scale, dx, dy)."""
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.full((size, size, 3), 114, dtype=np.uint8)
    dx, dy = (size - nw) // 2, (size - nh) // 2
    out[dy : dy + nh, dx : dx + nw] = resized
    return out, scale, dx, dy


def _dims(s: Sample) -> tuple[int, int]:
    """(h, w) of a sample: the manifest's recorded dims when present,
    else decode once. Val scoring needs these to place ground truth in
    letterbox space."""
    if s.h > 0 and s.w > 0:
        return s.h, s.w
    img = _read_rgb(s.path)
    if img is None:
        return (s.h or 1, s.w or 1)
    s.h, s.w = img.shape[:2]
    return s.h, s.w


def _read_rgb(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _augment(img: np.ndarray, boxes: np.ndarray, cls: np.ndarray
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = img.shape[:2]
    # hflip
    if random.random() < 0.5:
        img = img[:, ::-1].copy()
        if len(boxes):
            boxes = boxes.copy()
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
    # HSV jitter (on RGB via HSV roundtrip)
    if random.random() < 0.8:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.int16)
        hsv[..., 0] = (hsv[..., 0] + random.randint(-8, 8)) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * random.uniform(0.7, 1.3), 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * random.uniform(0.7, 1.3), 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    # scale jitter around centre
    if random.random() < 0.5 and len(boxes):
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
        # [0,1], not [0,255]: see the input contract in the module docstring.
        imgs[bi] = img.astype(np.float32).transpose(2, 0, 1) / 255.0
        n = min(len(boxes), max_objs)
        for j in range(n):
            x0, y0, x1, y1 = boxes[j] * scale
            x0, x1 = x0 + dx, x1 + dx
            y0, y1 = y0 + dy, y1 + dy
            targets[bi, j] = [cls[j], (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0]
    return torch.from_numpy(imgs), torch.from_numpy(targets)


def _batch_iter(samples: list[Sample], order: list[int], batch: int, size: int,
                train: bool, workers: int = 4, lookahead: int = 6):
    """Yield (imgs, targets) batches with decode + augment running on a
    thread pool ahead of the consumer. cv2 releases the GIL, so this
    overlaps the CPU pipeline with the GPU step instead of serialising
    them (VOC2007: 30 s/epoch of loading otherwise)."""
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


# ── evaluation: VOC AP@0.5, no pycocotools ──────────────────────────

def _voc_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def ap50(preds: list[np.ndarray], gts: list[np.ndarray], n_classes: int,
         iou_thr: float = 0.5) -> tuple[float, list[float]]:
    """preds[i]: (n,6) x0,y0,x1,y1,score,cls per image; gts[i]: (m,5) x0..y1,cls."""
    per_class: list[float] = []
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
            continue
        if not rows:
            per_class.append(0.0)
            continue
        rows.sort(key=lambda r: -r[0])
        tp = np.cumsum([r[1] for r in rows])
        fp = np.cumsum([1 - r[1] for r in rows])
        per_class.append(_voc_ap(tp / n_gt, tp / np.maximum(tp + fp, 1e-9)))
    mean = float(np.mean(per_class)) if per_class else 0.0
    return mean, per_class


# Contract constants: uint8 camera bytes re-encoded as int8 must mean [0,1].
CONTRACT_SCALE = 1.0 / 255.0
CONTRACT_ZP = -128


def _decode_raw(raw: np.ndarray, imgsz: int) -> np.ndarray:
    """Apply YOLOX's grid decode to raw head rows, in numpy.

    The graph no longer does this (it exports dynamic-shape ops ST Edge
    AI cannot handle), so the host does it for scoring and yolox_pp.c
    does it on the M55. Row order is stride 8, then 16, then 32."""
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
    """Pin and then CHECK the graph input's quantisation.

    stedgeai is told to re-encode the input as uint8 channel-last; that
    is only correct for an asymmetric [0,1] input (scale 1/255, zp
    -128). A model that misses it either fails to build or, worse,
    runs on scrambled values and detects noise. Refuse to emit one."""
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
        # The input tensor is raw camera bytes, so the only correct
        # parameters are the contract's, whatever calibration saw. What
        # calibration is allowed to tell us is whether the preprocessing
        # fed [0,1] at all: an implied range outside it (a symmetric
        # [-1,1] model, a [0,255] model) means every downstream range was
        # learned on the wrong input and pinning would be the post-hoc
        # rewrite that measurably costs more than it saves - refuse.
        lo, hi = cal_s * (-128 - cal_z), cal_s * (127 - cal_z)
        if lo < -0.02 or hi > 1.02:
            raise ValueError(
                f"export rejected: input calibrated to scale={cal_s:.8f} zp={cal_z} "
                f"(range [{lo:.3f}, {hi:.3f}]), but the Neuro N6 contract needs "
                f"scale={CONTRACT_SCALE:.8f} zp={CONTRACT_ZP} over [0, 1]. The model was "
                "calibrated on the wrong input range - fix the preprocessing rather than "
                "rewriting the finished network.")
        # A sub-range of [0,1] (dim images, percentile clipping): pin.
        for idx, value in ((1, np.float32(CONTRACT_SCALE)), (2, np.int8(CONTRACT_ZP))):
            name = q.input[idx]
            if name in init:
                init[name].CopyFrom(nh.from_array(value, name))
            else:
                consts[name].attribute[0].t.CopyFrom(nh.from_array(value, name))
    else:
        # Nothing quantised the input (e.g. the first op is a reshape):
        # insert the contract pair and rewire the original consumers.
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

    # Re-read and assert, so the check tests the file we actually ship.
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


# ── the run ─────────────────────────────────────────────────────────

class Cancelled(Exception):
    pass


def _build_model(n_classes: int, conv_stem: bool = True) -> YOLOX:
    backbone = YOLOPAFPN(depth=0.33, width=0.25, depthwise=True)
    head = YOLOXHead(num_classes=n_classes, width=0.25, depthwise=True)
    model = YOLOX(backbone, head)
    if conv_stem:
        # YOLOX's Focus stem is a 4-way slice + concat. The Neural-ART
        # compiler maps those Slices as pure-software epochs (measured:
        # 6 of 129), so swap in the stride-2 conv YOLOv5 and ST's own
        # st_yoloxn use instead. Same output shape, so the rest of the
        # COCO warm start still loads; only the stem trains from scratch.
        base_ch = int(0.25 * 64)
        model.backbone.backbone.stem = BaseConv(3, base_ch, ksize=6, stride=2, act="silu")

    def init(m):
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eps, m.momentum = 1e-3, 0.03
    model.apply(init)
    model.head.initialize_biases(1e-2)
    return model


def _load_pretrained(model: YOLOX) -> bool:
    """Warm-start from Megvii's COCO nano checkpoint (skips the
    class-dependent head layers). Cached in <workspace>/weights/yolox/."""
    dest = workspace.weights_dir() / "yolox" / "yolox_nano.pth"
    try:
        if not dest.is_file():
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(".tmp")
            urllib.request.urlretrieve(PRETRAINED_URL, tmp)  # noqa: S310 (pinned URL)
            tmp.replace(dest)
        ckpt = torch.load(dest, map_location="cpu", weights_only=False)
        state = dict(ckpt.get("model", ckpt))
        own = model.state_dict()

        # The COCO checkpoint has YOLOX's Focus stem (4-way slice + 3x3
        # conv on 12 channels). We build a 6x6 stride-2 conv instead, and
        # the two are exactly equivalent: grid cell (i,j) of the Focus
        # path reads original pixels 2i-2..2i+3, which is precisely the
        # 6x6 stride-2 window. Remap rather than drop it, so small
        # datasets keep a warm first layer.
        f_w = state.pop("backbone.backbone.stem.conv.conv.weight", None)
        tgt = "backbone.backbone.stem.conv.weight"
        if f_w is not None and tgt in own and own[tgt].shape[-1] == 6:
            k6 = torch.zeros_like(own[tgt])                 # [out, 3, 6, 6]
            for grp, (py, px) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        k6[:, :, 2 * di + py + 2, 2 * dj + px + 2] =                             f_w[:, grp * 3:(grp + 1) * 3, di + 1, dj + 1]
            # Trained on [0,255]; the graph now takes [0,1], and this is
            # the first layer, so fold the 255 in here. Everything
            # downstream then sees identical activations.
            state[tgt] = k6 * 255.0
        # Focus wrapped its BN one level deeper than a bare BaseConv does.
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
    except Exception:
        return False  # offline / hash mismatch: train from scratch


def train(
    project_id: str,
    cfg: TrainConfig,
    progress: Callable[[dict], None] | None = None,
    cancel=None,
) -> dict:
    t0 = time.time()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    samples, classes = load_samples(project_id, cfg.train_split)
    train_s = [i for i, s in enumerate(samples) if s.split == "train"]
    val_s = [i for i, s in enumerate(samples) if s.split == "val"]
    if len(train_s) < 2:
        raise ValueError(f"need at least 2 labelled training images, have {len(train_s)}")
    if not val_s:  # tiny dataset: fall back to scoring on train
        val_s = list(train_s)

    device = torch.device(cfg.device)
    use_amp = device.type == "cuda"
    if device.type == "cuda" and device.index is not None:
        # Per-thread current device: the vendored head builds its grids
        # with .type(torch.cuda.FloatTensor), which lands on the CURRENT
        # device, so a cuda:1 run needs cuda:1 to be current here.
        torch.cuda.set_device(device)
    model = _build_model(len(classes)).to(device)
    pretrained_used = cfg.pretrained and _load_pretrained(model)

    batch = max(2, min(cfg.batch, len(train_s)))
    lr = cfg.lr * batch / 64
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
        lr=lr, momentum=0.9, nesterov=True,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    steps_per_epoch = max(1, len(train_s) // batch)

    def lr_at(epoch: int, step: int) -> float:
        t = epoch + step / steps_per_epoch
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
            s = samples[si]
            hh, ww = _dims(s)
            scale = min(cfg.imgsz / ww, cfg.imgsz / hh)
            dx = (cfg.imgsz - int(round(ww * scale))) // 2
            dy = (cfg.imgsz - int(round(hh * scale))) // 2
            g = s.boxes * scale
            g[:, [0, 2]] += dx
            g[:, [1, 3]] += dy
            gts.append(np.hstack([g, s.classes[:, None]]))
        net.train()
        return ap50(preds, gts, len(classes))

    best_ap = -1.0
    best_state = None
    for epoch in range(cfg.epochs):
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        order = train_s[:]
        random.shuffle(order)
        # Drop the ragged tail so every step is a full batch (BN stats).
        order = order[: steps_per_epoch * batch]
        epoch_loss = 0.0
        for step, (imgs, targets) in enumerate(
                _batch_iter(samples, order, batch, cfg.imgsz, train=True)):
            for g in opt.param_groups:
                g["lr"] = lr_at(epoch, step)
            imgs, targets = imgs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=use_amp):
                out = model(imgs, targets)
                loss = out["total_loss"]
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            epoch_loss += float(loss.detach())
        mean_ap = None
        if (epoch + 1) % 5 == 0 or epoch + 1 == cfg.epochs:
            mean_ap, _ = evaluate(model)
            if mean_ap >= best_ap:
                best_ap = mean_ap
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if progress:
            progress({
                "index": epoch + 1, "total": cfg.epochs + cfg.qat_epochs,
                "phase": f"loss {epoch_loss / steps_per_epoch:.2f}"
                + (f" · AP50 {mean_ap:.3f}" if mean_ap is not None else ""),
            })
    if best_state is not None:
        model.load_state_dict(best_state)
    fp32_ap, fp32_per_class = evaluate(model)

    # ── export ──────────────────────────────────────────────────────
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_dir = store.dataset_dir(project_id) / "models" / "yolox_nano" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "_", (store.load(project_id) or {}).get("name", "model").lower()).strip("_") or "model"

    # Float reference first: the plain graph, before any fine-tuning.
    # Raw head: the in-graph decode exports ScatterND/Expand/Where, which
    # ST Edge AI cannot shape-infer. yolox_pp.c decodes on the M55.
    model.head.decode_in_inference = False
    fp32_onnx = out_dir / f"{slug}_fp32.onnx"
    _export_onnx(model, fp32_onnx, cfg.imgsz, qat=False)
    model.to(device)
    # The export turned in-graph decoding off; evaluation during the QAT
    # phase needs it back on (raw rows scored as pixel boxes give AP 0).
    model.head.decode_in_inference = True

    # ── quantisation-aware fine-tuning ──────────────────────────────
    qat_ap = None
    if cfg.qat_epochs > 0:
        qat_ap = _qat_finetune(model, samples, train_s, batch, cfg, device, lr,
                               evaluate, progress, cancel)

    torch.save({"model": model.state_dict(), "classes": classes, "imgsz": cfg.imgsz,
                "qat": cfg.qat_epochs > 0},
               out_dir / "best.pt")

    int8_onnx = out_dir / f"{slug}_int8.onnx"
    if cfg.qat_epochs > 0:
        # The QAT graph exports its own Q/DQ pairs; no calibration pass.
        _export_onnx(model, int8_onnx, cfg.imgsz, qat=True)
        _fold_weight_quant(int8_onnx)
        contract = _verify_input_contract(int8_onnx)
    else:
        contract = _ptq_quantize(fp32_onnx, int8_onnx, samples, train_s, cfg)
    int8_ap, int8_per_class = _score_onnx(int8_onnx, samples, val_s, cfg, classes)

    neuron6 = _verify_neuron6(int8_onnx, out_dir, slug) if cfg.verify_neuron6 else {
        "verified": False, "available": None, "reason": "verification disabled"}
    if neuron6.get("error"):
        # The compiler is installed and said no: not a deliverable. Leave
        # the ONNX for debugging, write the reason, emit no bundle.
        (out_dir / "report.json").write_text(json.dumps({
            "run": run_id, "model": "yolox_nano", "exportRejected": True,
            "neuron6": neuron6}, indent=2), "utf-8")
        raise ValueError(f"export rejected by ST Edge AI: {neuron6['error']}")

    _write_bundle(out_dir, slug, classes, cfg, {
        "run": run_id,
        "model": "yolox_nano",
        "images": {"train": len(train_s), "val": len(val_s)},
        "epochs": cfg.epochs,
        "imgsz": cfg.imgsz,
        "pretrained": pretrained_used,
        "head": "raw",
        "input": {**contract, "layout": "NCHW float [0,1]",
                  "note": "stedgeai re-encodes to uint8 channel-last"},
        "ap50_fp32": round(fp32_ap, 4),
        "ap50_int8": round(int8_ap, 4),
        "qat_epochs": cfg.qat_epochs,
        "ap50_qat_sim": (round(qat_ap, 4) if qat_ap is not None else None),
        "per_class_fp32": {c: round(a, 4) for c, a in zip(classes, fp32_per_class)},
        "per_class_int8": {c: round(a, 4) for c, a in zip(classes, int8_per_class)},
        "neuron6": neuron6,
        "minutes": round((time.time() - t0) / 60, 1),
    }, int8_onnx)
    return json.loads((out_dir / "report.json").read_text("utf-8"))


# ── Neuro N6 build verification (PK-1.3) ────────────────────────────

def _verify_neuron6(int8_onnx: Path, out_dir: Path, slug: str) -> dict:
    """Compile the export with ST Edge AI as the core's pragma pipeline
    would. The compiler log and generate report land in <run>/neuron6/."""
    import neuron6_verify

    del slug  # the generated C symbol is discarded; a short name avoids MAX_PATH
    try:
        return neuron6_verify.verify(int8_onnx, out_dir / "neuron6")
    except RuntimeError as e:
        return {"verified": False, "available": True, "error": str(e)}


# ── export helpers ──────────────────────────────────────────────────

def _export_onnx(model: YOLOX, path: Path, imgsz: int, qat: bool) -> None:
    """Trace to ONNX on CPU. With qat=True the fake-quant points export
    as QuantizeLinear/DequantizeLinear pairs (the QDQ pattern ST Edge AI
    accepts) and BatchNorm is already folded by the QAT forward."""
    set_qat(model, qat)
    model.eval().cpu()
    model.head.decode_in_inference = False
    # QAT: activation fake quants export as Q/DQ pairs. The WEIGHT fake
    # quants do not survive the exporter (constant propagation folds
    # Q(const weight) -> DQ back into a float weight), so the conv
    # weights come out float and _fold_weight_quant re-quantises them
    # with the exact per-channel rule the fine-tuning simulated.
    torch.onnx.export(
        model, torch.zeros(1, 3, imgsz, imgsz), str(path), opset_version=17,
        input_names=["images"], output_names=["raw"], dynamo=False,
    )


def _fold_weight_quant(path: Path) -> int:
    """Make a QAT export carry int8 weights the way a PTQ export does.

    1. Every Q/DQ scale and zero-point torch emitted as a Constant node
       becomes an initializer (ST Edge AI resolves them there and dies
       with "'NoneType' object has no attribute 'tolist'" otherwise).
    2. Every Conv whose weight is still a float initializer (the
       exporter folded the weight fake quant away) gets a per-channel
       symmetric int8 initializer + DequantizeLinear in front of it,
       with scale = max|w| / 127 per output channel: exactly what
       qat.fq_weight simulated during fine-tuning.
    Returns the number of convs re-quantised."""
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

    # The exporter de-duplicates identical parameters (three pred-conv
    # biases straight after initialize_biases are identical) into one
    # initializer plus Identity nodes. Resolve those to the source
    # tensor so each conv gets its own quantised copy.
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
                # Bias as int32 with scale = x_scale * w_scale, the way
                # ORT's QDQ output (which the compiler handles) has it.
                src = producer.get(n.input[0])
                if (len(n.input) >= 3 and n.input[2] in init
                        and init[n.input[2]].data_type == onnx.TensorProto.FLOAT
                        and src is not None and src.op_type == "DequantizeLinear"
                        and src.input[1] in init):
                    x_scale = float(np.ravel(nh.to_array(init[src.input[1]]))[0])
                    b_t = init[n.input[2]]
                    b = nh.to_array(b_t)
                    b_scale = (scale * x_scale).astype(np.float32)
                    b_q = np.clip(np.round(b / b_scale), -2**31, 2**31 - 1).astype(np.int32)
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


def _qat_finetune(model: YOLOX, samples, train_s, batch: int, cfg: TrainConfig, device,
                  lr: float, evaluate, progress, cancel) -> float:
    """Fine-tune with int8 simulated throughout the graph (see
    vendor/yolox/qat.py). BatchNorm is folded with frozen statistics;
    the activation observers seed from a pass over training batches,
    then track a slow moving average. Runs in fp32 (no autocast): the
    model is small and fake quant is cheap. Returns the simulated-int8
    AP@0.5 on the val split."""
    from vendor.yolox.qat import set_calibrate, unseeded

    set_qat(model, True)
    steps = max(1, len(train_s) // batch)
    # Seed every observer: the training branch (losses) and the
    # inference branch (the head's output fake quant lives only there).
    set_calibrate(model, True)
    with torch.no_grad():
        seed = train_s[: min(len(train_s), max(batch * 8, cfg.calib_images))]
        model.train()
        for imgs, targets in _batch_iter(samples, seed, batch, cfg.imgsz, train=False):
            model(imgs.to(device), targets.to(device))
        model.eval()
        for imgs, _t in _batch_iter(samples, seed, batch, cfg.imgsz, train=False):
            model(imgs.to(device))
    set_calibrate(model, False)
    missing = unseeded(model)
    if missing:
        raise RuntimeError(f"QAT observers never saw data: {missing[:5]}")
    model.train()
    # Fine-tune at a tenth of the float LR, cosine to 1% over the phase.
    qat_lr = lr * 0.1
    opt = torch.optim.SGD(model.parameters(), lr=qat_lr, momentum=0.9,
                          nesterov=True, weight_decay=5e-4)
    total = cfg.epochs + cfg.qat_epochs
    ap = None
    for qe in range(cfg.qat_epochs):
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        order = train_s[:]
        random.shuffle(order)
        order = order[: steps * batch]
        loss_sum = 0.0
        for step, (imgs, targets) in enumerate(
                _batch_iter(samples, order, batch, cfg.imgsz, train=True)):
            p = (qe + step / steps) / max(1, cfg.qat_epochs)
            for g in opt.param_groups:
                g["lr"] = qat_lr * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            loss = model(imgs, targets)["total_loss"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            # With BatchNorm folded the loss landscape is far sharper than
            # the float run's (gradient norms of 100-200 vs single digits);
            # unclipped, one step at this LR doubles the loss. Measured:
            # clip 5 recovers the float loss within a few dozen steps.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            loss_sum += float(loss.detach())
        if qe + 1 == cfg.qat_epochs or (qe + 1) % 4 == 0:
            ap, _ = evaluate(model)
        if progress:
            progress({
                "index": cfg.epochs + qe + 1, "total": total,
                "phase": f"QAT loss {loss_sum / steps:.2f}"
                + (f" · int8 AP50 {ap:.3f}" if ap is not None else ""),
            })
    if ap is None:
        ap, _ = evaluate(model)
    return ap


# ── int8 PTQ fallback (ONNX Runtime static QDQ) ─────────────────────

def _ptq_quantize(fp32_path: Path, int8_path: Path, samples, train_s, cfg: TrainConfig) -> dict:
    """Post-training quantisation for qat_epochs=0: percentile
    calibration on natural training images (no synthetic extremes: they
    poison every deep range for the sake of the input, which the
    contract check pins anyway). Returns the input contract record."""
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

    quantize_static(
        str(fp32_path), str(int8_path), Reader(),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=CalibrationMethod.Percentile,
        extra_options={"CalibPercentile": cfg.calib_percentile},
    )
    return _verify_input_contract(int8_path)


def _score_onnx(path: Path, samples, val_s, cfg: TrainConfig, classes: list[str]
                ) -> tuple[float, list[float]]:
    """AP@0.5 of an exported (raw-head) ONNX on the val split, through
    the same host decode + NMS the float model is scored with."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    preds, gts = [], []
    for si in val_s:
        imgs, _ = _make_batch(samples, [si], cfg.imgsz, train=False)
        raw = sess.run(None, {"images": imgs.numpy()})[0]
        out = torch.from_numpy(_decode_raw(raw, cfg.imgsz))
        preds.append(_decode_rows(out, 0.01, cfg.nms_thres, len(classes)))
        s = samples[si]
        hh, ww = _dims(s)
        scale = min(cfg.imgsz / ww, cfg.imgsz / hh)
        dx = (cfg.imgsz - int(round(ww * scale))) // 2
        dy = (cfg.imgsz - int(round(hh * scale))) // 2
        g = s.boxes * scale
        g[:, [0, 2]] += dx
        g[:, [1, 3]] += dy
        gts.append(np.hstack([g, s.classes[:, None]]))
    return ap50(preds, gts, len(classes))


# ── the Neuro N6 bundle ─────────────────────────────────────────────

def _write_bundle(out_dir: Path, slug: str, classes: list[str], cfg: TrainConfig,
                  report: dict, int8_onnx: Path) -> None:
    guard = f"{slug.upper()}_LABELS_H"
    labels_h = "\n".join([
        f"#ifndef {guard}",
        f"#define {guard}",
        "/* Generated by PixelKit - YOLOX-Nano trained on your dataset. */",
        f"#define {slug.upper()}_NUM_CLASSES {len(classes)}",
        f"#define {slug.upper()}_INPUT_W {cfg.imgsz}",
        f"#define {slug.upper()}_INPUT_H {cfg.imgsz}",
        f"#define {slug.upper()}_CONF_THRESHOLD {cfg.conf_thres}f",
        f"#define {slug.upper()}_NMS_THRESHOLD {cfg.nms_thres}f",
        f"static const char* const {slug.upper()}_CLASSES[{len(classes)}] = {{",
        *[f'  "{c}",' for c in classes],
        "};",
        f"#endif /* {guard} */",
        "",
    ])
    (out_dir / f"{slug}_labels.h").write_text(labels_h, "utf-8")

    snippet = "\n".join([
        f"// PixelKit export - drop {int8_onnx.name} and {slug}_labels.h into your",
        "// sketch folder, then:",
        "//",
        f'#pragma neuron6 model="{int8_onnx.name}" name={slug}',
        f'#include "{slug}_labels.h"',
        "#include <Models.h>",
        "",
        f"NEURON6_DECLARE_MODEL({slug});",
        "",
        f"// Input: {cfg.imgsz}x{cfg.imgsz}. The camera feeds the NPU directly:",
        "// the model is quantised asymmetrically over [0,1] (scale 1/255,",
        "// zero-point -128), which is exactly what packed RGB888 bytes mean,",
        "// so there is no conversion step in your sketch.",
        "//",
        "// Output: ONE raw head tensor, rows [dx, dy, dw, dh, obj, cls...]",
        "// (obj/cls already sigmoided, box terms in grid units). The",
        "// PostProcess YOLOX decoder does the grid decode, thresholding and",
        "// NMS for you using the values in the header.",
        f"// Pass &NN_Instance_{slug} to Vision as usual.",
        "",
    ])
    (out_dir / f"{slug}_snippet.ino.txt").write_text(snippet, "utf-8")
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), "utf-8")

    zpath = out_dir / "export.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in (int8_onnx, out_dir / f"{slug}_labels.h",
                  out_dir / f"{slug}_snippet.ino.txt", out_dir / "report.json"):
            zf.write(f, f.name)
