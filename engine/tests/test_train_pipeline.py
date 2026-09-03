"""Training pipeline pieces that need a dataset on disk: the stratified
split, augmentations in the training set, pre-flight checks, batch
sizing and out-of-memory recovery, checkpoint + resume, the calibrated
threshold in the bundle, and labelling with the exported model. CPU
only, 64 px inputs, no COCO download."""
import io
import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image


def _jpeg(w=96, h=64, colour=(200, 30, 30)) -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", (w, h), colour)
    # A brighter block so the picture is not one flat colour.
    for x in range(10, 40):
        for y in range(10, 40):
            img.putpixel((x, y), (250, 250, 250))
    img.save(buf, "JPEG")
    return buf.getvalue()


def _dataset(client, name: str, n: int = 10, labels=("dog", "cat")) -> tuple[str, list[str]]:
    r = client.post("/api/v2/projects", data={"name": name, "labels": json.dumps(list(labels))})
    assert r.status_code == 200, r.text
    pid = r.json()["project_id"]
    iids = []
    for i in range(n):
        r = client.post(f"/api/v2/projects/{pid}/imports/raw",
                        files={"image": (f"img{i:02d}.jpg", _jpeg(colour=(200, 20 * i, 30)), "image/jpeg")})
        assert r.status_code == 200, r.text
        iid = r.json()["import_id"]
        iids.append(iid)
        boxes = [{"id": f"b{i}", "label": labels[i % len(labels)],
                  "x0": 10, "y0": 10, "x1": 40, "y1": 40}]
        if i % 3 == 0:  # a second, other-class box on every third image
            boxes.append({"id": f"c{i}", "label": labels[(i + 1) % len(labels)],
                          "x0": 50, "y0": 20, "x1": 90, "y1": 60})
        r = client.put(f"/api/v2/projects/{pid}/imports/{iid}", json={"editedBoxes": boxes})
        assert r.status_code == 200, r.text
    return pid, iids


# ── split ────────────────────────────────────────────────────────────

def _mk(name, classes, split="train"):
    from train_yolox import Sample

    return Sample(path=Path(name), boxes=np.zeros((len(classes), 4), np.float32),
                  classes=np.array(classes, np.float32), split=split, name=name)


def test_stratified_split_gives_every_class_a_val_image_and_keeps_rare_ones_in_train():
    from train_yolox import stratified_split

    # 20 images of class 0, 3 of class 1, 1 of class 2.
    samples = ([_mk(f"a{i}.jpg", [0]) for i in range(20)]
               + [_mk(f"b{i}.jpg", [1]) for i in range(3)]
               + [_mk("c0.jpg", [2])])
    stratified_split(samples, 0.8)
    val = [s for s in samples if s.split == "val"]
    assert 3 <= sum(1 for s in val if 0 in s.classes) <= 5      # about 20% of 20
    assert sum(1 for s in val if 1 in s.classes) == 1           # 3 images: one goes to val
    assert all(2 not in s.classes for s in val)                 # a single image stays in train
    assert sum(1 for s in samples if s.split == "train" and 1 in s.classes) == 2

    # Deterministic and independent of list order.
    before = {s.name: s.split for s in samples}
    shuffled = list(reversed(samples))
    stratified_split(shuffled, 0.8)
    assert {s.name: s.split for s in shuffled} == before
    # A different seed moves images around but keeps the guarantees.
    stratified_split(samples, 0.8, seed=7)
    assert sum(1 for s in samples if s.split == "val" and 1 in s.classes) == 1


def test_stratified_split_never_strands_a_class_in_val_only():
    from train_yolox import stratified_split

    # Two images share class 1; both also carry a class that wants val.
    samples = [_mk("x.jpg", [0, 1]), _mk("y.jpg", [0, 1]), _mk("z.jpg", [0])] \
        + [_mk(f"w{i}.jpg", [0]) for i in range(6)]
    stratified_split(samples, 0.5)
    train_has_1 = any(s.split == "train" and 1 in s.classes for s in samples)
    assert train_has_1


def test_split_edges():
    from train_yolox import stratified_split

    samples = [_mk(f"{i}.jpg", [0]) for i in range(4)]
    stratified_split(samples, 1.0)
    assert all(s.split == "train" for s in samples)
    stratified_split([], 0.8)


# ── dataset loading: augmentations follow their parent ───────────────

def _write_augmentations(ds_dir: Path, iid: str, parent_w=96, parent_h=64, n=2, label="dog"):
    d = ds_dir / "augmentations" / iid
    d.mkdir(parents=True, exist_ok=True)
    copies = {}
    for k in range(n):
        name = f"{k:03d}.jpg"
        Image.new("RGB", (parent_w, parent_h), (10, 200, 10)).save(d / name, "JPEG")
        copies[name] = [{"label": label, "box": [12, 12, 44, 44], "polys": []}]
    (d / "annotations.json").write_text(json.dumps({"width": parent_w, "height": parent_h, "copies": copies}))


def test_augmentations_join_the_train_split_only(client):
    import store
    from train_yolox import load_samples

    pid, iids = _dataset(client, "Aug Split", n=8)
    m = store.load(pid)
    ds_dir = store.dataset_dir(pid)
    # Mark every import as augmented on disk.
    for imp in m["imports"]:
        imp["n_augmentations"] = 2
        _write_augmentations(ds_dir, imp["id"], label="dog" if "img0" in imp["filename"] else "cat")
    store.save(pid, m)

    samples, classes, info = load_samples(pid, 0.75, use_augmentations=True)
    assert classes == ["dog", "cat"]
    originals = [s for s in samples if s.source == "original"]
    augmented = [s for s in samples if s.source == "augmented"]
    assert len(originals) == 8 and info["originals"] == 8
    train_parents = {s.name for s in originals if s.split == "train"}
    val_parents = {s.name for s in originals if s.split == "val"}
    assert val_parents, "the split should hold something out"
    assert augmented, "augmented copies of train images must be used"
    assert all(s.split == "train" for s in augmented)
    assert all(s.parent in train_parents for s in augmented)
    assert not any(s.parent in val_parents for s in augmented)
    assert info["augmented"] == 2 * len(train_parents)
    assert all(s.path.is_file() and s.w == 96 and s.h == 64 for s in augmented)

    samples2, _c, info2 = load_samples(pid, 0.75, use_augmentations=False)
    assert all(s.source == "original" for s in samples2) and info2["augmented"] == 0

    # A second dataset as the held-out set: everything here trains.
    pid2, _ = _dataset(client, "Held Out", n=4, labels=("cat", "dog"))
    samples3, _c, info3 = load_samples(pid, 0.75, eval_project=pid2)
    assert all(s.split == "train" for s in samples3 if s.source != "eval")
    assert [s.split for s in samples3 if s.source == "eval"] == ["val"] * 4
    assert info3["evalImages"] == 4 and info3["evalMissingClasses"] == []
    with pytest.raises(ValueError):
        load_samples(pid, 0.75, eval_project=pid)


# ── pre-flight ───────────────────────────────────────────────────────

def test_preflight_reports_blockers_and_warnings(client):
    from train_yolox import TrainConfig, preflight

    pid, _ = _dataset(client, "Preflight", n=6)
    cfg = TrainConfig(imgsz=64, train_split=0.8)
    out = preflight(pid, cfg)
    assert out["ok"] is True
    assert out["images"]["train"] + out["images"]["val"] == 6
    names = [c["name"] for c in out["classes"]]
    assert names == ["dog", "cat"]
    joined = " ".join(out["warnings"])
    assert "only" in joined and "training box" in joined   # few examples per class
    # 30x30 boxes on a 96-wide image at 64 px input are 20 px: not small.
    assert "under 8 px" not in joined

    # Objects that vanish at the input size: a 10 px box on a 96 px wide
    # image is under 7 px at a 64 px input.
    from train_yolox import Sample, preflight_samples

    tiny = []
    for i in range(6):
        s = Sample(path=Path(f"t{i}.jpg"), boxes=np.array([[5, 5, 15, 15]], np.float32),
                   classes=np.array([0], np.float32), split="train" if i else "val",
                   w=96, h=64, name=f"t{i}.jpg")
        tiny.append(s)
    small = preflight_samples(tiny, ["dog"], {"originals": 6, "augmented": 0, "unlabelled": 0},
                              TrainConfig(imgsz=64))
    assert any("under 8 px" in w for w in small["warnings"]), small["warnings"]
    assert small["classes"][0]["smallBoxesPct"] == 100
    big = preflight_samples(tiny, ["dog"], {"originals": 6, "augmented": 0, "unlabelled": 0},
                            TrainConfig(imgsz=256))
    assert not any("under 8 px" in w for w in big["warnings"])

    r = client.get(f"/api/v2/projects/{pid}/train/preflight", params={"imgsz": 64, "split": 0.8})
    assert r.status_code == 200 and r.json()["ok"] is True
    assert r.json()["images"]["augmented"] == 0

    # One labelled image: blocked.
    pid1, _ = _dataset(client, "One", n=1)
    out1 = preflight(pid1, cfg)
    assert out1["ok"] is False and "at least 2" in out1["blockers"][0]
    r = client.get(f"/api/v2/projects/{pid1}/train/preflight")
    assert r.status_code == 200 and r.json()["ok"] is False
    assert client.get("/api/v2/projects/nope/train/preflight").status_code == 404


# ── batch sizing and OOM recovery ────────────────────────────────────

def test_auto_batch_on_cpu_and_oom_halving():
    from train_yolox import _auto_batch, _epoch, _is_oom

    assert _auto_batch(torch.device("cpu"), 256, "nano", 300) == 8
    assert _auto_batch(torch.device("cpu"), 256, "nano", 100) == 7   # 15 steps per epoch
    assert _auto_batch(torch.device("cpu"), 256, "nano", 3) == 2
    if torch.cuda.is_available():
        dev = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
        assert _auto_batch(dev, 256, "nano", 240) == 16        # 240 images: 15 steps
        assert _auto_batch(dev, 256, "nano", 4000) in (24, 32, 40, 48, 56, 64)
        assert _auto_batch(dev, 256, "nano", 20) == 2

    calls = []
    oom_cls = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)

    def step_fn(step, imgs, targets):
        calls.append(len(imgs))
        if len(imgs) > 2:
            raise oom_cls("CUDA out of memory. Tried to allocate 1 GiB")
        return 1.0

    from train_yolox import Sample
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "t.png"
        Image.new("RGB", (32, 32), (1, 2, 3)).save(p)
        samples = [Sample(path=p, boxes=np.array([[2, 2, 20, 20]], np.float32),
                          classes=np.array([0], np.float32), split="train", w=32, h=32)
                   for _ in range(8)]
        st = {"batch": 8, "steps": 1}
        notes: list[str] = []
        loss = _epoch(samples, list(range(8)), st, 32, step_fn, None, notes)
    assert st["batch"] == 2 and loss == 1.0
    assert calls[0] == 8 and calls[1] == 4 and calls[2] == 2
    assert len(notes) == 2 and "out of memory at batch 8" in notes[0]

    # An unrelated RuntimeError is not swallowed.
    assert not _is_oom(RuntimeError("shape mismatch"))
    assert _is_oom(RuntimeError("CUDA error: out of memory"))


def test_infer_image_maps_boxes_back_to_the_original_pixels():
    from train_yolox import infer_image

    imgsz, n_classes = 64, 2
    n_rows = (imgsz // 8) ** 2 + (imgsz // 16) ** 2 + (imgsz // 32) ** 2
    raw = np.zeros((1, n_rows, 5 + n_classes), np.float32)
    # Stride-8 grid cell (2, 3): centre ((2+0.5)*8, (3+0.5)*8) = (20, 28),
    # size exp(log 2)*8 = 16 px in letterbox space.
    row = 3 * (imgsz // 8) + 2
    raw[0, row, :4] = [0.5, 0.5, np.log(2.0), np.log(2.0)]
    raw[0, row, 4] = 1.0
    raw[0, row, 5 + 1] = 1.0  # class 1

    class Sess:
        def run(self, _outs, feed):
            assert feed["images"].shape == (1, 3, imgsz, imgsz)
            return [raw]

    img = np.zeros((64, 128, 3), np.uint8)  # 128 wide, 64 tall -> scale 0.5, dy 16
    out = infer_image(Sess(), img, imgsz, n_classes, conf=0.5, nms=0.45)
    assert out.shape == (1, 6)
    x0, y0, x1, y1, score, cls = out[0]
    assert cls == 1 and score == pytest.approx(1.0, abs=1e-4)
    assert (x0, x1) == pytest.approx(((20 - 8) / 0.5, (20 + 8) / 0.5), abs=0.6)
    assert (y0, y1) == pytest.approx(((28 - 8 - 16) / 0.5, (28 + 8 - 16) / 0.5), abs=0.6)


# ── the whole run on CPU: interrupt, resume, bundle, label with it ────

def test_interrupt_resume_and_label_with_the_model(client):
    import store
    from train_yolox import Cancelled, TrainConfig, find_run_dir, train

    pid, iids = _dataset(client, "Resume Run", n=8)
    run_id = "20990101T000000Z"
    cancel = threading.Event()
    seen = []

    def prog(p):
        seen.append(p)
        if p["index"] == 1:
            cancel.set()  # after the first epoch's checkpoint

    cfg = TrainConfig(epochs=2, qat_epochs=0, imgsz=64, batch=2, device="cpu", pretrained=False,
                      verify_neuron6=False, run_id=run_id, checkpoint_every=1, failure_limit=4)
    with pytest.raises(Cancelled):
        train(pid, cfg, prog, cancel)
    run_dir = find_run_dir(pid, run_id)
    assert run_dir is not None and run_dir.parent.name == "yolox_nano"
    assert (run_dir / "ckpt_last.pt").is_file()
    hist = json.loads((run_dir / "history.json").read_text())
    assert hist["status"] == "interrupted" and len(hist["history"]) == 1
    assert seen[0]["total"] == 2 and "eta_s" in seen[0]

    r = client.get(f"/api/v2/projects/{pid}/models")
    assert r.status_code == 200
    (entry,) = r.json()["models"]
    assert entry["status"] == "interrupted" and entry["resumable"] is True
    assert entry["epochsDone"] == 1 and entry["totalEpochs"] == 2 and entry["hasExport"] is False

    # Resume: the request carries only the run id; settings come back
    # from the checkpoint (epochs=2, imgsz=64, no QAT).
    report = train(pid, TrainConfig(resume_run=run_id, device="cpu", verify_neuron6=False))
    assert report["resumed"] is True and report["run"] == run_id
    assert report["epochs"] == 2 and report["imgsz"] == 64 and report["qat_epochs"] == 0
    assert [h["epoch"] for h in report["history"]] == [1, 2]
    assert any("resumed at epoch 2" in n for n in report["notes"])
    assert report["classes"] == ["dog", "cat"]
    assert report["images"]["train"] + report["images"]["val"] == 8
    assert report["split"]["stratified"] is True
    thr = report["threshold"]
    assert 0.01 <= thr["conf"] <= 0.99 and "sweep" in thr
    assert [c["name"] for c in report["classReport"]] == ["dog", "cat"]
    assert len(report["confusion"]) == 3 and report["confusionLabels"][-1] == "background"
    assert isinstance(report["failures"], list)
    assert not (run_dir / "ckpt_last.pt").exists()
    assert json.loads((run_dir / "history.json").read_text())["status"] == "done"
    labels_h = next(run_dir.glob("*_labels.h")).read_text()
    assert f"_CONF_THRESHOLD {thr['conf']:.2f}f" in labels_h
    if thr["calibrated"]:
        assert "Calibrated on the val split" in labels_h

    r = client.get(f"/api/v2/projects/{pid}/models")
    (entry,) = r.json()["models"]
    assert entry["status"] == "done" and entry["hasExport"] is True
    assert entry["threshold"]["conf"] == thr["conf"]
    assert entry["history"] is None  # finished runs serve curves from the detail
    r = client.get(f"/api/v2/projects/{pid}/models/{run_id}")
    assert r.status_code == 200
    detail = r.json()
    assert detail["report"]["history"] == report["history"]
    for f in report["failures"]:
        r = client.get(f"/api/v2/projects/{pid}/models/{run_id}/failures/{f['file']}")
        assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert client.get(f"/api/v2/projects/{pid}/models/{run_id}/failures/..%2Freport.json").status_code in (400, 404)
    assert client.get(f"/api/v2/projects/{pid}/models/{run_id}/failures/99_nope.jpg").status_code == 404
    # Nothing to resume once finished.
    assert client.post(f"/api/v2/projects/{pid}/models/{run_id}/resume", json={}).status_code in (400, 409)

    # Label an unlabelled image with the exported model.
    r = client.post(f"/api/v2/projects/{pid}/imports/raw",
                    files={"image": ("fresh.jpg", _jpeg(colour=(30, 30, 200)), "image/jpeg")})
    fresh = r.json()["import_id"]
    r = client.post("/api/jobs", json={"kind": "label_model", "project": pid,
                                       "params": {"run": run_id, "conf": 0.01}})
    assert r.status_code == 200, r.text
    jid = r.json()["jobId"]
    deadline = time.time() + 120
    while time.time() < deadline:
        j = client.get(f"/api/jobs/{jid}").json()
        if j["status"] in ("done", "failed", "cancelled"):
            break
        time.sleep(0.3)
    assert j["status"] == "done", j
    assert j["result"]["run"] == run_id and "labelled" in j["result"]
    m = store.load(pid)
    e = next(x for x in m["imports"] if x["id"] == fresh)
    assert e["labelled"] is True and e["labelledBy"]["model"] == run_id
    for d in e["detections"]:
        assert d["label"] in ("dog", "cat") and len(d["box"]) == 4 and d["source"] == "pixelkit-model"
        # The editor reads pred_label/gd_score, exports box_xyxy: all present.
        assert d["pred_label"] == d["label"] == d["gd_label"]
        assert d["gd_score"] == d["score"] and d["box_xyxy"] == d["box"]
    # Hand-labelled images were not touched, and a second run finds nothing to do.
    e0 = next(x for x in m["imports"] if x["id"] == iids[0])
    assert e0.get("editedBoxesSet") and len(e0["editedBoxes"]) >= 1
    r = client.post("/api/jobs", json={"kind": "label_model", "project": pid, "params": {"run": run_id}})
    assert r.status_code == 400 and "overwrite" in r.json()["detail"]

    # Start a new run from this one's weights: accepted by the scheduler.
    r = client.post("/api/jobs", json={"kind": "train_yolox", "project": pid,
                                       "params": {"initRun": "20000101T000000Z"}})
    assert r.status_code == 400
    r = client.delete(f"/api/v2/projects/{pid}/models/{run_id}")
    assert r.status_code == 200 and not run_dir.exists()
    assert client.get(f"/api/v2/projects/{pid}/models").json()["models"] == []


def test_train_params_are_clamped(app_module):
    cfg = app_module._train_cfg_from_params({
        "epochs": "9999", "imgsz": 999, "size": "huge", "split": 0.1, "augmentations": "false",
        "initRun": "bad id", "conf": 3, "evalProject": " ",
    }, "cpu")
    assert cfg.epochs == 500 and cfg.imgsz == 256 and cfg.size == "nano"
    assert cfg.train_split == 0.8 and cfg.use_augmentations is False
    odd = app_module._train_cfg_from_params({"epochs": True, "augmentations": "", "imgsz": None}, "cpu")
    assert odd.epochs == 50 and odd.use_augmentations is True and odd.imgsz == 256
    assert cfg.init_run is None and cfg.conf_thres is None and cfg.eval_project is None
    cfg = app_module._train_cfg_from_params({"size": "tiny", "split": "0.9", "conf": "0.4",
                                             "resumeRun": "20990101T000000Z"}, "cuda:1")
    assert cfg.size == "tiny" and cfg.train_split == 0.9 and cfg.conf_thres == 0.4
    assert cfg.resume_run == "20990101T000000Z" and cfg.device == "cuda:1"
    assert app_module._fmt_eta(45) == "45 s" and app_module._fmt_eta(600) == "10 min"
    assert app_module._fmt_eta(2 * 3600 + 5 * 60) == "2 h 05 min"


# ── sweep: one bad file, one bad box, one bad step must not end a run ──

def test_unreadable_images_and_outside_boxes_are_skipped(tmp_path):
    import torch
    from train_yolox import Sample, _build_model, _gt_letterbox, _make_batch, _read_rgb

    empty = tmp_path / "empty.jpg"
    empty.write_bytes(b"")
    assert _read_rgb(empty) is None
    garbage = tmp_path / "garbage.jpg"
    garbage.write_bytes(b"not an image at all")
    assert _read_rgb(garbage) is None
    assert _read_rgb(tmp_path / "missing.jpg") is None

    good = tmp_path / "good.png"
    Image.new("RGB", (96, 64), (10, 20, 30)).save(good)
    # One box inside, one far outside the picture, one that straddles the edge.
    s = Sample(path=good, boxes=np.array([[10, 10, 40, 40], [400, 10, 440, 50], [80, 20, 130, 60]], np.float32),
               classes=np.array([0, 1, 0], np.float32), split="train", w=96, h=64, name="good.png")
    imgs, targets = _make_batch([s, Sample(path=empty, boxes=s.boxes, classes=s.classes, split="train")],
                                [0, 1], size=64, train=False)
    rows = targets[0][targets[0].sum(1) > 0]
    assert len(rows) == 2                       # the outside box is gone
    assert float((rows[:, 1] + rows[:, 3] / 2).max()) <= 64.0 + 1e-3  # x1 clipped to the canvas
    assert targets[1].sum() == 0 and imgs[1].sum() == 0  # unreadable: blank sample, no boxes
    gt = _gt_letterbox(s, 64)
    assert len(gt) == 2 and gt[:, :4].max() <= 64.0
    # The head accepts the batch (this used to raise "selected index k out of range").
    model = _build_model(n_classes=2)
    model.train()
    out = model(imgs, targets)
    assert torch.isfinite(out["total_loss"])


def test_non_finite_steps_are_skipped_and_a_diverged_epoch_raises(tmp_path):
    from train_yolox import Sample, _epoch, _json_safe, _write_json_atomic

    p = tmp_path / "t.png"
    Image.new("RGB", (32, 32), (1, 2, 3)).save(p)
    samples = [Sample(path=p, boxes=np.array([[2, 2, 20, 20]], np.float32),
                      classes=np.array([0], np.float32), split="train", w=32, h=32) for _ in range(4)]
    seen = []

    def step_fn(step, imgs, targets):
        seen.append(step)
        return float("nan") if step == 0 else 2.0

    st = {"batch": 2, "steps": 2}
    notes: list[str] = []
    assert _epoch(samples, [0, 1, 2, 3], st, 32, step_fn, None, notes) == 2.0
    assert notes and "non-finite" in notes[0]
    with pytest.raises(RuntimeError, match="diverged"):
        _epoch(samples, [0, 1, 2, 3], st, 32, lambda *_a: float("inf"), None, [])

    # history.json stays strict JSON whatever the numbers were.
    out = tmp_path / "h.json"
    _write_json_atomic(out, {"loss": float("nan"), "rows": [1.5, float("inf")], "n": np.int64(3)})
    text = out.read_text("utf-8")
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text) == {"loss": None, "rows": [1.5, None], "n": 3}
    assert _json_safe({"a": (np.float32(1.5),)}) == {"a": [1.5]}


def test_atomic_replace_retries_while_the_file_is_open(tmp_path):
    """Windows refuses to rename over a file another handle has open;
    the writer waits it out instead of ending the run."""
    import threading
    import time as _time

    from train_yolox import _write_json_atomic

    target = tmp_path / "history.json"
    _write_json_atomic(target, {"v": 0})
    fh = open(target, "rb")

    def release():
        _time.sleep(0.3)
        fh.close()

    threading.Thread(target=release).start()
    _write_json_atomic(target, {"v": 1})  # would raise PermissionError on Windows without the retry
    assert json.loads(target.read_text("utf-8")) == {"v": 1}
    assert not list(tmp_path.glob("*.tmp"))


def test_preflight_two_labelled_images_are_enough():
    from train_yolox import Sample, TrainConfig, preflight_samples

    def mk(name, split):
        return Sample(path=Path(name), boxes=np.array([[5, 5, 40, 40]], np.float32),
                      classes=np.array([0], np.float32), split=split, w=96, h=64, name=name)

    info = {"originals": 2, "augmented": 0, "unlabelled": 0}
    out = preflight_samples([mk("a.jpg", "train"), mk("b.jpg", "val")], ["dog"], info, TrainConfig(imgsz=64))
    assert out["ok"] is True, out["blockers"]
    # One original plus its augmented copy is still one labelled image.
    aug = mk("aug_a_000.jpg", "train")
    aug.source = "augmented"
    out = preflight_samples([mk("a.jpg", "train"), aug], ["dog"],
                            {"originals": 1, "augmented": 1, "unlabelled": 0}, TrainConfig(imgsz=64))
    assert out["ok"] is False and "have 1" in out["blockers"][0]


def test_label_model_targets_never_touch_hand_edited_images(app_module):
    manifest = {"imports": [
        {"id": "a", "filename": "a.jpg"},                                             # untouched
        {"id": "b", "filename": "b.jpg", "detections": [{"label": "x"}], "labelled": True},  # auto
        {"id": "c", "filename": "c.jpg", "editedBoxesSet": True, "editedBoxes": [{"label": "x"}]},
        {"id": "d", "filename": "d.jpg", "editedBoxesSet": True, "editedBoxes": []},  # cleared by hand
        {"id": "e"},                                                                   # no file
    ]}
    assert [e["id"] for e in app_module._label_model_targets(manifest, False)] == ["a"]
    assert [e["id"] for e in app_module._label_model_targets(manifest, True)] == ["a", "b"]
