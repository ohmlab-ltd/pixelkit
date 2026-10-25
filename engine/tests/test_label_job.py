"""The batch labelling job, with a stand-in SAM 3."""
import io
import json
import threading
import time

import numpy as np
import pytest
from PIL import Image


def _jpeg(colour) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), colour).save(buf, "JPEG")
    return buf.getvalue()


def _wait(client, job_id, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j.get("status") in ("done", "failed", "cancelled"):
            return j
        time.sleep(0.05)
    raise TimeoutError(job_id)


class FakeSam3:
    """segment_labels as the pipeline answers it, recording the colour it
    was shown; `hold` blocks the first call until released."""

    def __init__(self, hold=False, fail=False):
        self.calls = 0
        self.entered, self.release = threading.Event(), threading.Event()
        self.hold, self.fail = hold, fail
        if not hold:
            self.release.set()

    def segment_labels(self, pil, tags, include_crops=True, **kw):
        self.calls += 1
        if self.calls == 1:
            self.entered.set()
            self.release.wait(20)
        if self.fail:
            raise RuntimeError("model is gone")
        r, g, b = np.asarray(pil.convert("RGB")).reshape(-1, 3).mean(0)
        seen = max((("red", r), ("green", g), ("blue", b)), key=lambda kv: kv[1])[0]
        return ([{"box": [0, 0, 10, 10], "gd_label": tags[0], "gd_score": 0.9, "mask": None,
                  "seen": seen}], {"total_ms": 1.0})

    segment_labels_tiled = segment_labels


@pytest.fixture
def sam3(app_module):
    fakes = []

    def install(**kw):
        fakes.append(FakeSam3(**kw))
        app_module.state["charlie"] = fakes[-1]
        return fakes[-1]

    yield install
    for f in fakes:
        f.release.set()
    app_module.state["charlie"] = None


def _dataset(client, name, colours):
    pid = client.post("/api/v2/projects", data={"name": name, "labels": json.dumps(["obj", "other"])}).json()["project_id"]
    ids = {}
    for i, (cname, col) in enumerate(colours):
        r = client.post(f"/api/v2/projects/{pid}/imports/raw",
                        files={"image": (f"{cname}.jpg", _jpeg(col), "image/jpeg")},
                        data={"created_at_ms": str(1_700_000_000_000 - i * 1000)})
        ids[cname] = r.json()["import_id"]
    return pid, ids


def test_an_image_deleted_mid_run_does_not_lend_its_pixels(client, app_module, sam3):
    pid, ids = _dataset(client, "Prefetch", [("red", (255, 0, 0)), ("green", (0, 255, 0)),
                                             ("blue", (0, 0, 255))])
    fake = sam3(hold=True)
    jid = client.post("/api/jobs", json={"project": pid, "kind": "label_charlie", "params": {}}).json()["jobId"]
    assert fake.entered.wait(20)
    time.sleep(0.3)  # the next image's decode has finished
    assert client.delete(f"/api/v2/projects/{pid}/imports/{ids['green']}").status_code == 200
    fake.release.set()
    assert _wait(client, jid)["status"] == "done"
    seen = {e["id"]: [d.get("seen") for d in e.get("detections") or []]
            for e in app_module.load_manifest(pid)["imports"]}
    assert seen[ids["red"]] == ["red"] and seen[ids["blue"]] == ["blue"]


def test_one_label_writer_per_dataset_and_no_rename_under_it(client, sam3):
    pid, _ = _dataset(client, "Writers", [("red", (255, 0, 0))])
    fake = sam3(hold=True)
    jid = client.post("/api/jobs", json={"project": pid, "kind": "label_charlie", "params": {}}).json()["jobId"]
    assert fake.entered.wait(20)
    try:
        r = client.post("/api/jobs", json={"project": pid, "kind": "purge_label", "params": {"label": "obj"}})
        assert r.status_code == 409
        r = client.post(f"/api/projects/{pid}/labels/rename", json={"old_label": "obj", "new_label": "thing"})
        assert r.status_code == 409
    finally:
        fake.release.set()
    assert _wait(client, jid)["status"] == "done"


def test_a_run_where_every_image_fails_is_a_failed_run(client, sam3):
    pid, _ = _dataset(client, "All fail", [("red", (255, 0, 0)), ("blue", (0, 0, 255))])
    sam3(fail=True)
    jid = client.post("/api/jobs", json={"project": pid, "kind": "label_charlie", "params": {}}).json()["jobId"]
    j = _wait(client, jid)
    assert j["status"] == "failed" and "no image could be labelled" in (j.get("error") or "")


def test_a_label_run_logs_its_setup_not_each_images_stages(client, sam3, capsys):
    pid, _ = _dataset(client, "Quiet log", [("red", (255, 0, 0)), ("blue", (0, 0, 255))])
    sam3()
    jid = client.post("/api/jobs", json={"project": pid, "kind": "label_charlie", "params": {}}).json()["jobId"]
    assert _wait(client, jid)["status"] == "done"
    out = capsys.readouterr().out
    assert "2 image(s) to label" in out
    assert not any(s in out for s in ("opening", "waiting for GPU lock", "running SAM3"))


def test_an_ambiguous_detection_takes_the_higher_of_its_top_two_labels(client, app_module, sam3, monkeypatch):
    import v2_dinov2

    # The stand-in DINOv2 embeds a crop by its colour. A reference colour is its
    # label's axis; an import colour's vector is its similarity to bolt, nut, washer.
    vectors = {
        (255, 0, 0): (1, 0, 0), (0, 255, 0): (0, 1, 0), (0, 0, 255): (0, 0, 1),
        (255, 255, 0): (0.9, 0.2, 0.1),
        (255, 0, 255): (0.1, 0.62, 0.58),
        (0, 255, 255): (0.1, 0.6, 0.595),
        (255, 255, 255): (0.1, 0.60001, 0.60004),
    }

    def encode(pil):
        mean = np.asarray(pil.convert("RGB"), dtype=np.float32).reshape(-1, 3).mean(0)
        nearest = min(vectors, key=lambda c: float(np.square(np.subtract(c, mean)).sum()))
        return np.asarray(vectors[nearest], dtype=np.float32)

    monkeypatch.setattr(v2_dinov2, "is_loaded", lambda: True)
    monkeypatch.setattr(v2_dinov2, "encode_image", encode)
    monkeypatch.setattr(v2_dinov2, "encode_images_batch", lambda pils, **_k: np.stack([encode(p) for p in pils]))

    pid = client.post("/api/v2/projects", data={"name": "Close calls",
                                                "labels": json.dumps(["bolt", "nut", "washer"])}).json()["project_id"]
    for label, colour in (("bolt", (255, 0, 0)), ("nut", (0, 255, 0)), ("washer", (0, 0, 255))):
        r = client.post(f"/api/v2/projects/{pid}/references",
                        files={"image": (f"{label}.jpg", _jpeg(colour), "image/jpeg")}, data={"label": label})
        assert r.status_code == 200 and r.json()["embedded"] == 1
    ids = {}
    for name, colour in (("clear", (255, 255, 0)), ("apart", (255, 0, 255)),
                         ("close", (0, 255, 255)), ("tied", (255, 255, 255))):
        r = client.post(f"/api/v2/projects/{pid}/imports/raw",
                        files={"image": (f"{name}.jpg", _jpeg(colour), "image/jpeg")})
        ids[name] = r.json()["import_id"]
    sam3()
    jid = client.post("/api/jobs", json={"project": pid, "kind": "label_charlie", "params": {}}).json()["jobId"]
    assert _wait(client, jid)["status"] == "done"

    dets = {e["id"]: e["detections"] for e in app_module.load_manifest(pid)["imports"]}
    keys = ("pred_label", "pred_source", "ambiguous", "vlm_action", "fused_margin", "rejected",
            "embed_sim_for_label")
    assert {name: tuple(dets[iid][0].get(k) for k in keys) for name, iid in ids.items()} == {
        "clear": ("bolt", "embed", False, None, None, False, 0.9),
        "apart": ("nut", "embed-vlm", False, "tiebreak", 0.04, False, 0.62),
        "close": ("nut", "embed-vlm", True, "tiebreak", 0.005, False, 0.6),
        # Washer scores higher, but at four decimal places it ties with nut, the first label.
        "tied": ("nut", "embed-vlm", True, "tiebreak", 0.0, False, 0.6),
    }


def test_previews_draw_the_users_boxes_once_edited(app_module):
    auto = [{"box": [0, 0, 5, 5], "gd_label": "obj"}]
    mine = [{"id": "b", "label": "obj", "x0": 1, "y0": 1, "x1": 9, "y1": 9}]
    assert app_module._preview_source({"editedBoxes": mine}, auto) == mine
    assert app_module._preview_source({"editedBoxes": [], "editedBoxesSet": True}, auto) == []
    assert app_module._preview_source({"editedBoxes": None}, auto) == auto
    assert app_module._preview_source(None, auto) == auto


def test_an_interrupted_download_is_not_complete(tmp_path, monkeypatch):
    import huggingface_hub

    import models

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: str(tmp_path))
    (tmp_path / "config.json").write_text("{}")
    assert models.is_downloaded("sam3") is False  # only the first small file landed
    (tmp_path / "model.safetensors").write_bytes(b"x")
    assert models.is_downloaded("sam3") is True
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"a": "model-00001.safetensors", "b": "model-00002.safetensors"}}))
    (tmp_path / "model-00001.safetensors").write_bytes(b"x")
    assert models.is_downloaded("sam3") is False  # a shard is missing
    (tmp_path / "model-00002.safetensors").write_bytes(b"x")
    assert models.is_downloaded("sam3") is True


def test_a_drawn_box_takes_what_it_encloses_not_a_neighbour():
    from pipeline_charlie import _bbox_iou, _inside_fraction

    user = [100, 100, 300, 300]
    small_inside = [150, 150, 200, 200]  # a loose box around a small object
    assert _bbox_iou(user, small_inside) < 0.3 and _inside_fraction(small_inside, user) == 1.0
    neighbour = [280, 100, 480, 300]     # only touches the edge of the box
    assert _bbox_iou(user, neighbour) < 0.3 and _inside_fraction(neighbour, user) < 0.8


def test_a_faint_whole_image_mask_does_not_filter_out_the_object():
    from pipeline_charlie import SAM3_THRESHOLD, _keep_relative

    # (box, mask, score, area) at the interactive threshold: the square
    # the user boxed, a speck, and SAM 3's near-whole-image guess.
    square = ([20, 120, 70, 170], None, 0.96, 17_000)
    speck = ([30, 130, 32, 132], None, 0.06, 300)
    frame = ([0, 0, 1008, 756], None, 0.06, 760_000)
    hits = [square, speck, frame]
    # Against the whole-image mask, the square counts as noise.
    assert _keep_relative(hits, 0.05, None) == [frame]
    # Against the largest confident mask it stays; the speck still goes.
    assert _keep_relative(hits, 0.05, SAM3_THRESHOLD) == [square, frame]
    # Nothing confident: the largest overall, as with no threshold.
    faint = [speck, frame]
    assert _keep_relative(faint, 0.05, SAM3_THRESHOLD) == _keep_relative(faint, 0.05, None)
    assert _keep_relative([square], 0.05, SAM3_THRESHOLD) == [square]


def test_queued_embeddings_read_the_picture_when_they_run(app_module, tmp_path):
    # Queued with a path, not the bytes; a picture deleted meanwhile is skipped.
    assert app_module._embed_stored_image("p", "i", tmp_path / "deleted.jpg") is False
