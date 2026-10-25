"""Augmented copies keep every box the dataset holds, inside the picture,
under the current label names."""
import io
import json
import time

from PIL import Image


def _jpeg(w=64, h=48, colour=(200, 30, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), colour).save(buf, "JPEG")
    return buf.getvalue()


def _wait(client, job_id, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j.get("status") in ("done", "failed", "cancelled"):
            return j
        time.sleep(0.2)
    raise TimeoutError(job_id)


def test_clipping_to_the_picture(app_module):
    clip_box, clip_poly = app_module._clip_box_to, app_module._clip_poly_to
    assert clip_box([-10, 5, 30, 50], 40, 40) == [0.0, 5.0, 30.0, 40.0]
    assert clip_box([50, 5, 90, 30], 40, 40) is None          # left the frame
    square = [[-10, 10], [20, 10], [20, 30], [-10, 30]]
    got = clip_poly(square, 40, 40)
    assert min(p[0] for p in got) == 0.0 and max(p[0] for p in got) == 20.0
    assert clip_poly([[50, 50], [60, 50], [60, 60]], 40, 40) == []
    inside = [[1, 1], [9, 1], [9, 9]]
    assert clip_poly(inside, 40, 40) == [[1.0, 1.0], [9.0, 1.0], [9.0, 9.0]]


def test_copies_keep_edited_boxes_and_follow_label_changes(client, app_module):
    r = client.post("/api/v2/projects", data={"name": "Aug labels", "labels": json.dumps(["apple", "pear"])})
    pid = r.json()["project_id"]
    up = client.post(f"/api/v2/projects/{pid}/imports/raw",
                     files={"image": ("a.jpg", _jpeg(), "image/jpeg")}).json()
    iid = up["import_id"]
    boxes = [{"id": "b1", "label": "apple", "x0": 4, "y0": 4, "x1": 30, "y1": 30},
             {"id": "b2", "label": "pear", "x0": 34, "y0": 10, "x1": 60, "y1": 40}]
    assert client.put(f"/api/v2/projects/{pid}/imports/{iid}",
                      json={"editedBoxes": boxes}).status_code == 200

    cfg = {"camera": {"enabled": True, "frequency": "all", "noise": 2}}
    r = client.post(f"/api/v2/projects/{pid}/augment/generate", json={"perImageMode": "1", "config": cfg})
    assert _wait(client, r.json()["job_id"])["status"] == "done"
    anno_path = app_module._augmentations_dir(pid, iid) / "annotations.json"

    def labels():
        copies = json.loads(anno_path.read_text("utf-8"))["copies"]
        return sorted(d["label"] for dets in copies.values() for d in dets)

    # Hand-drawn boxes are x0/y0/x1/y1.
    copies = json.loads(anno_path.read_text("utf-8"))["copies"]
    assert all(len(d["box"]) == 4 for dets in copies.values() for d in dets)
    assert labels() == ["apple", "pear"]

    def bundled():
        return len(app_module._aug_files_to_bundle(pid, app_module.load_manifest(pid)))

    assert bundled() == 1
    r = client.post(f"/api/projects/{pid}/labels/rename", json={"old_label": "apple", "new_label": "plum"})
    assert r.status_code == 200
    assert labels() == ["pear", "plum"]
    assert bundled() == 1, "a rename keeps the copies current"

    r = client.post("/api/jobs", json={"project": pid, "kind": "purge_label", "params": {"label": "pear"}})
    assert _wait(client, r.json()["jobId"])["status"] == "done"
    assert labels() == ["plum"]
    assert bundled() == 1

    # A moved box leaves the copies out of date: exports leave them out.
    moved = [{"id": "b1", "label": "plum", "x0": 8, "y0": 8, "x1": 30, "y1": 30}]
    assert client.put(f"/api/v2/projects/{pid}/imports/{iid}",
                      json={"editedBoxes": moved}).status_code == 200
    assert bundled() == 0


def test_one_generation_at_a_time(client, app_module):
    r = client.post("/api/v2/projects", data={"name": "Aug busy", "labels": json.dumps(["apple"])})
    pid = r.json()["project_id"]
    client.post(f"/api/v2/projects/{pid}/imports/raw", files={"image": ("a.jpg", _jpeg(), "image/jpeg")})
    from jobs import Job

    # A running generation, registered but not queued, so nothing runs.
    busy = Job(id="busyaug", kind="augment_generate", project=pid, params={}, status="running")
    app_module.state["jobs"].jobs[busy.id] = busy
    try:
        r = client.post(f"/api/v2/projects/{pid}/augment/generate", json={"perImageMode": "off", "config": {}})
        assert r.status_code == 409
    finally:
        app_module.state["jobs"].jobs.pop(busy.id, None)


def test_augmentations_and_training_do_not_overlap(client, app_module):
    from jobs import Job

    r = client.post("/api/v2/projects", data={"name": "Aug train", "labels": json.dumps(["apple"])})
    pid = r.json()["project_id"]
    for name in ("a.jpg", "b.jpg"):
        iid = client.post(f"/api/v2/projects/{pid}/imports/raw",
                          files={"image": (name, _jpeg(), "image/jpeg")}).json()["import_id"]
        boxes = [{"id": "b1", "label": "apple", "x0": 4, "y0": 4, "x1": 30, "y1": 30}]
        assert client.put(f"/api/v2/projects/{pid}/imports/{iid}",
                          json={"editedBoxes": boxes}).status_code == 200
    jobs = app_module.state["jobs"].jobs

    # A training run reads the copies every epoch.
    jobs["busytrain"] = Job(id="busytrain", kind="train_yolox", project=pid, params={}, status="running")
    try:
        r = client.post(f"/api/v2/projects/{pid}/augment/generate", json={"perImageMode": "off", "config": {}})
        assert r.status_code == 409
        r = client.delete(f"/api/v2/projects/{pid}/augmentations/{iid}/00.jpg")
        assert r.status_code == 409
    finally:
        jobs.pop("busytrain", None)

    jobs["busyaug"] = Job(id="busyaug", kind="augment_generate", project=pid, params={}, status="running")
    try:
        r = client.post("/api/jobs", json={"kind": "train_yolox", "project": pid, "params": {}})
        assert r.status_code == 409, r.text
    finally:
        jobs.pop("busyaug", None)
