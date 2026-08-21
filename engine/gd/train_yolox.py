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
import random
import time
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch

import store
import workspace
from vendor.yolox import YOLOX, YOLOPAFPN, YOLOXHead, postprocess
from vendor.yolox.network_blocks import BaseConv

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
    calib_images: int = 64
    seed: int = 0
    warmup_epochs: int = 3


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
        arr0 = {i.name: nh.to_array(i) for i in g.initializer}
        cal_s, cal_z = float(arr0[q.input[1]]), int(arr0[q.input[2]])
        calibrated = {"scale": cal_s, "zeroPoint": cal_z}
        # Calibration should already BE the contract (we train on [0,1] and
        # calibrate with a black and a white frame). If it is not, the
        # preprocessing has drifted and pinning would be the post-hoc
        # rewrite that measurably costs more accuracy than it saves:
        # refuse instead.
        if cal_z != CONTRACT_ZP or abs(cal_s - CONTRACT_SCALE) > 0.02 * CONTRACT_SCALE:
            raise ValueError(
                f"export rejected: input calibrated to scale={cal_s:.8f} zp={cal_z}, "
                f"but the Neuro N6 contract needs scale={CONTRACT_SCALE:.8f} zp={CONTRACT_ZP}. "
                "The model was calibrated on the wrong input range - fix the "
                "preprocessing rather than rewriting the finished network.")
        # Within tolerance: make it exact.
        init[q.input[1]].CopyFrom(nh.from_array(np.float32(CONTRACT_SCALE), q.input[1]))
        init[q.input[2]].CopyFrom(nh.from_array(np.int8(CONTRACT_ZP), q.input[2]))
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
            img = _read_rgb(s.path)
            hh, ww = (img.shape[:2] if img is not None else (s.h or 1, s.w or 1))
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
        epoch_loss = 0.0
        for step in range(steps_per_epoch):
            idxs = order[step * batch : (step + 1) * batch]
            if not idxs:
                continue
            for g in opt.param_groups:
                g["lr"] = lr_at(epoch, step)
            imgs, targets = _make_batch(samples, idxs, cfg.imgsz, train=True)
            imgs, targets = imgs.to(device), targets.to(device)
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
                "index": epoch + 1, "total": cfg.epochs,
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
    slug = "".join(c if c.isalnum() else "_" for c in (store.load(project_id) or {}).get("name", "model").lower()) or "model"

    torch.save({"model": model.state_dict(), "classes": classes, "imgsz": cfg.imgsz},
               out_dir / "best.pt")

    model.eval().cpu()
    # Raw head: the in-graph decode exports ScatterND/Expand/Where, which
    # ST Edge AI cannot shape-infer. yolox_pp.c decodes on the M55.
    model.head.decode_in_inference = False
    fp32_onnx = out_dir / f"{slug}_fp32.onnx"
    torch.onnx.export(
        model, torch.zeros(1, 3, cfg.imgsz, cfg.imgsz),
        str(fp32_onnx), opset_version=17,
        input_names=["images"], output_names=["raw"], dynamo=False,
    )

    int8_onnx = out_dir / f"{slug}_int8.onnx"
    int8_ap, int8_per_class, contract = _quantize_and_score(
        fp32_onnx, int8_onnx, samples, train_s, val_s, cfg, classes)

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
        "per_class_fp32": {c: round(a, 4) for c, a in zip(classes, fp32_per_class)},
        "per_class_int8": {c: round(a, 4) for c, a in zip(classes, int8_per_class)},
        "minutes": round((time.time() - t0) / 60, 1),
    }, int8_onnx)
    return json.loads((out_dir / "report.json").read_text("utf-8"))


# ── int8 quantization (ONNX Runtime static QDQ) ─────────────────────

def _quantize_and_score(fp32_path: Path, int8_path: Path, samples, train_s, val_s,
                        cfg: TrainConfig, classes: list[str]
                        ) -> tuple[float, list[float], dict]:
    from onnxruntime.quantization import (CalibrationDataReader, QuantFormat,
                                          QuantType, quantize_static)
    import onnxruntime as ort

    calib_idxs = train_s[: cfg.calib_images]
    # A black and a white frame pin MinMax to exactly [0,1], so the
    # calibrated input scale/zp ARE the contract rather than being
    # rewritten into it afterwards (a post-hoc rewrite of a finished
    # network leaves every downstream range mismatched).
    extremes = [np.zeros((1, 3, cfg.imgsz, cfg.imgsz), np.float32),
                np.ones((1, 3, cfg.imgsz, cfg.imgsz), np.float32)]

    class Reader(CalibrationDataReader):
        def __init__(self):
            self._extremes = iter(extremes)
            self._it = iter(calib_idxs)

        def get_next(self):
            e = next(self._extremes, None)
            if e is not None:
                return {"images": e}
            si = next(self._it, None)
            if si is None:
                return None
            imgs, _ = _make_batch(samples, [si], cfg.imgsz, train=False)
            return {"images": imgs.numpy()}

    quantize_static(
        str(fp32_path), str(int8_path), Reader(),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    contract = _verify_input_contract(int8_path)

    sess = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    preds, gts = [], []
    for si in val_s:
        imgs, _ = _make_batch(samples, [si], cfg.imgsz, train=False)
        raw = sess.run(None, {"images": imgs.numpy()})[0]
        out = torch.from_numpy(_decode_raw(raw, cfg.imgsz))
        preds.append(_decode_rows(out, 0.01, cfg.nms_thres, len(classes)))
        s = samples[si]
        img = _read_rgb(s.path)
        hh, ww = (img.shape[:2] if img is not None else (s.h or 1, s.w or 1))
        scale = min(cfg.imgsz / ww, cfg.imgsz / hh)
        dx = (cfg.imgsz - int(round(ww * scale))) // 2
        dy = (cfg.imgsz - int(round(hh * scale))) // 2
        g = s.boxes * scale
        g[:, [0, 2]] += dx
        g[:, [1, 3]] += dy
        gts.append(np.hstack([g, s.classes[:, None]]))
    mean, per_class = ap50(preds, gts, len(classes))
    return mean, per_class, contract


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
