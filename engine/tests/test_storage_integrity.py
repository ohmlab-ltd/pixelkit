"""Writes that cannot lose or leak data: unique temp files and a retrying
rename, moves that leave a dataset in place rather than half-copy it,
crop-dataset resyncs under the child's write lock, and covers that can
only name the dataset's own pictures."""
import io
import json
import os
import threading
import time

import pytest
from PIL import Image

import store


def _jpeg(colour=(200, 30, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (48, 48), colour).save(buf, "JPEG")
    return buf.getvalue()


def _dataset(client, name, n=1):
    pid = client.post("/api/v2/projects", data={"name": name, "labels": json.dumps(["apple"])}).json()["project_id"]
    ups = [client.post(f"/api/v2/projects/{pid}/imports/raw",
                       files={"image": (f"i{i}.jpg", _jpeg((20 * i, 90, 160)), "image/jpeg")}).json()
           for i in range(n)]
    return pid, ups


def test_temp_names_are_unique_and_the_rename_retries(tmp_path, monkeypatch):
    target = tmp_path / "a.json"
    assert store.tmp_path(target) != store.tmp_path(target)
    real = os.replace
    fails = {"n": 2}

    def flaky(src, dst):
        if fails["n"]:
            fails["n"] -= 1
            raise PermissionError(32, "in use")
        real(src, dst)

    monkeypatch.setattr(store.os, "replace", flaky)
    store._atomic_write(target, b"{}")
    assert target.read_bytes() == b"{}" and fails["n"] == 0
    assert list(tmp_path.iterdir()) == [target]


def test_a_move_that_cannot_happen_leaves_the_dataset_whole(client, monkeypatch):
    pid, _ = _dataset(client, "Stays put")
    cid = client.post("/api/containers", json={"name": "Shelf"}).json()["id"]
    before = store.dataset_dir(pid)

    def locked(*_a, **_k):
        raise PermissionError(5, "a file inside is open")

    monkeypatch.setattr(store.os, "rename", locked)
    assert client.post(f"/api/containers/{cid}/datasets/{pid}").status_code == 200
    assert store.dataset_dir(pid) == before and (before / "dataset.json").is_file()
    assert len(list((before / "images").iterdir())) == 1
    # Deleting the project must not delete the dataset still inside it.
    monkeypatch.undo()
    assert client.post(f"/api/containers/{cid}/datasets/{pid}").status_code == 200
    inside = store.dataset_dir(pid)
    monkeypatch.setattr(store.os, "rename", locked)
    assert client.delete(f"/api/containers/{cid}").status_code == 409
    assert (inside / "dataset.json").is_file()


def test_resync_waits_for_the_childs_write_lock(client, app_module, monkeypatch):
    pid, ups = _dataset(client, "Crop parent")
    client.put(f"/api/v2/projects/{pid}/imports/{ups[0]['import_id']}",
               json={"editedBoxes": [{"id": "b", "label": "apple", "x0": 2, "y0": 2, "x1": 40, "y1": 40}]})
    cid = client.post(f"/api/v2/projects/{pid}/derive",
                      data={"name": "Crop child", "labels": json.dumps(["apple"]), "padding": "0",
                            "min_size": "0", "create_project": "false"}).json()["project_id"]
    for _ in range(100):
        crops = app_module.load_manifest(cid).get("imports") or []
        if crops:
            break
        time.sleep(0.1)
    crop = crops[0]

    real = app_module._derived_mod.resync

    def slow(*a, **k):
        time.sleep(1.0)  # a long crop pass
        return real(*a, **k)

    monkeypatch.setattr(app_module._derived_mod, "resync", slow)
    t = threading.Thread(target=app_module._resync_child_from_thread, args=(cid,))
    t.start()
    time.sleep(0.3)
    edit = [{"id": "e", "label": "apple", "x0": 1, "y0": 1, "x1": 20, "y1": 20}]
    assert client.put(f"/api/v2/projects/{cid}/imports/{crop['id']}",
                      json={"editedBoxes": edit}).status_code == 200
    t.join()
    got = client.get(f"/api/v2/projects/{cid}/annotations/{crop['id']}").json()["editedBoxes"]
    assert got == edit  # the pass did not save its stale copy over the edit


def test_the_cover_names_one_of_the_datasets_pictures(client, workspace_dir):
    pid, ups = _dataset(client, "Cover")
    (workspace_dir / "secret.txt").write_text("not yours")
    for bad in ("..\..\secret.txt", "../../secret.txt", "..", "nope.jpg"):
        assert client.put(f"/api/projects/{pid}", json={"cover": bad}).status_code == 400, bad
    assert client.put(f"/api/projects/{pid}", json={"cover": ups[0]["filename"]}).status_code == 200
    assert client.get(f"/api/projects/{pid}/cover_thumb").headers["content-type"] == "image/jpeg"
    # null resets to the default cover (it used to be dropped as "unset").
    assert client.put(f"/api/projects/{pid}", json={"cover": None}).status_code == 200
    assert store.load(pid).get("cover") is None


def test_a_label_colour_can_be_cleared(client):
    pid, _ = _dataset(client, "Colours")
    client.put(f"/api/projects/{pid}", json={"labelColours": {"apple": "#ff0000"}})
    assert store.load(pid)["labelColours"] == {"apple": "#ff0000"}
    client.put(f"/api/projects/{pid}", json={"labelColours": {"apple": ""}})
    assert "apple" not in (store.load(pid).get("labelColours") or {})
