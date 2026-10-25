"""Storage writes and moves that cannot lose or leak data."""
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


def test_duplicates_are_grouped_and_all_but_the_keeper_removed(client, app_module):
    import numpy as np

    pid, _ = _dataset(client, "Dupes")
    same = _jpeg((10, 200, 10))
    ids = [client.post(f"/api/v2/projects/{pid}/imports/raw",
                       files={"image": (f"d{i}.jpg", same if i < 2 else _jpeg((200, 10, 10 * i)), "image/jpeg")}
                       ).json()["import_id"] for i in range(4)]
    dedupe = lambda **kw: client.post(f"/api/v2/projects/{pid}/imports/dedupe", json=kw).json()  # noqa: E731
    (group,) = dedupe(strategy="exact")["groups"]
    assert {group["keep"], *[d["id"] for d in group["drop"]]} == set(ids[:2])
    # Near duplicates by embedding; exact copies are left to the exact pass.
    for iid, vec in zip(ids[1:], ([1, 0, 0], [1, 0, 0.01], [0.99, 0.02, 0])):
        path = app_module._image_embedding_path(pid, iid)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(np.asarray(vec, dtype=np.float32).tobytes())
    (near,) = dedupe(strategy="near")["groups"]
    assert {near["keep"], *[d["id"] for d in near["drop"]]} == set(ids[2:])
    r = dedupe(strategy="exact", mode="commit")
    assert len(r["deleted"]) == 1
    left = {e["id"] for e in app_module.load_manifest(pid)["imports"]}
    assert len(left & set(ids[:2])) == 1


def test_the_cover_names_one_of_the_datasets_pictures(client, workspace_dir):
    pid, ups = _dataset(client, "Cover")
    (workspace_dir / "secret.txt").write_text("not yours")
    for bad in (r"..\..\secret.txt", "../../secret.txt", "..", "nope.jpg"):
        assert client.put(f"/api/projects/{pid}", json={"cover": bad}).status_code == 400, bad
    assert client.put(f"/api/projects/{pid}", json={"cover": ups[0]["filename"]}).status_code == 200
    assert client.get(f"/api/projects/{pid}/cover_thumb").headers["content-type"] == "image/jpeg"
    # null resets to the default cover.
    assert client.put(f"/api/projects/{pid}", json={"cover": None}).status_code == 200
    assert store.load(pid).get("cover") is None


def test_a_label_colour_can_be_cleared(client):
    pid, _ = _dataset(client, "Colours")
    client.put(f"/api/projects/{pid}", json={"labelColours": {"apple": "#ff0000"}})
    assert store.load(pid)["labelColours"] == {"apple": "#ff0000"}
    client.put(f"/api/projects/{pid}", json={"labelColours": {"apple": ""}})
    assert "apple" not in (store.load(pid).get("labelColours") or {})


def test_a_read_racing_a_replace_waits_instead_of_losing_the_dataset(client, monkeypatch):
    pid, _ = _dataset(client, "Read under a replace", n=2)
    real = store.Path.read_bytes
    fails = {"n": 3}

    def busy(self):
        if self.name == "dataset.json" and fails["n"]:
            fails["n"] -= 1
            raise PermissionError(32, "being replaced")
        return real(self)

    monkeypatch.setattr(store.Path, "read_bytes", busy)
    m = store.load(pid)
    assert m and m["id"] == pid and len(m["imports"]) == 2 and fails["n"] == 0


def test_a_read_that_keeps_failing_raises_rather_than_reading_as_empty(client, app_module, monkeypatch):
    pid, _ = _dataset(client, "Unreadable", n=1)
    app_module.invalidate_manifest_cache(pid)
    real = store.Path.read_bytes

    def locked(self):
        if self.name == "dataset.json":
            raise PermissionError(5, "locked")
        return real(self)

    monkeypatch.setattr(store.Path, "read_bytes", locked)
    monkeypatch.setattr(store.time, "sleep", lambda _s: None)
    with pytest.raises(PermissionError):
        store.load(pid)
    with pytest.raises(PermissionError):
        app_module.load_manifest(pid)


def test_a_manifest_without_its_id_is_never_saved_over_a_dataset(client, app_module):
    pid, _ = _dataset(client, "Guarded", n=1)
    before = (store.dataset_dir(pid) / "dataset.json").read_bytes()
    annotations = sorted(p.name for p in (store.dataset_dir(pid) / "annotations").iterdir())
    with pytest.raises(RuntimeError):
        app_module.save_manifest(pid, {})
    assert (store.dataset_dir(pid) / "dataset.json").read_bytes() == before
    assert sorted(p.name for p in (store.dataset_dir(pid) / "annotations").iterdir()) == annotations


def test_missing_blurhashes_are_filled_in_from_the_images_folder(client, app_module):
    pid, ups = _dataset(client, "Blurhash backfill", n=2)
    for _ in range(100):  # the upload computes its blurhash after responding
        m = app_module.load_manifest(pid)
        if all(i.get("blurhash") for i in m["imports"]):
            break
        time.sleep(0.05)
    m = app_module.load_manifest(pid)
    for imp in m["imports"]:
        imp.pop("blurhash", None)
    app_module.save_manifest(pid, m)
    app_module._BLURHASH_CACHE.clear()
    assert client.get(f"/api/v2/projects/{pid}/overview").status_code == 200
    for _ in range(100):
        if all(i.get("blurhash") for i in app_module.load_manifest(pid)["imports"]):
            break
        time.sleep(0.05)
    assert all(i.get("blurhash") for i in app_module.load_manifest(pid)["imports"])


def test_a_reference_filename_cannot_reach_outside_the_references_folder(client, app_module, tmp_path):
    import asyncio

    from fastapi import HTTPException

    pid, _ = _dataset(client, "Reference names")
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(_jpeg())
    for bad in (str(outside), r"..\..\outside.jpg", "../outside.jpg", "C:outside.jpg"):
        with pytest.raises(HTTPException) as e:
            asyncio.run(app_module._reference_image_pil(None, pid, bad, "test"))
        assert e.value.status_code in (400, 403), bad
    refs = store.dataset_dir(pid) / "references"
    refs.mkdir(exist_ok=True)
    (refs / "r.jpg").write_bytes(_jpeg())
    assert asyncio.run(app_module._reference_image_pil(None, pid, "r.jpg", "test")).size == (48, 48)


def test_a_load_that_overlaps_a_save_cannot_make_a_later_save_skip_a_file(monkeypatch):
    """An undo right after an edit reaches disk although a reader was loading meanwhile."""
    import copy

    pid = store.new_id()
    store.create_dataset_dir(pid, "digest race")
    old = [{"x0": 10, "y0": 10, "x1": 50, "y1": 50, "label": "cat"}]
    new = [{"x0": 20, "y0": 20, "x1": 80, "y1": 80, "label": "cat"}]
    m = {"id": pid, "name": "digest race", "tags": ["cat"], "imports": [
        {"id": "x", "filename": "x.jpg", "editedBoxes": old, "editedBoxesSet": True},
        {"id": "y", "filename": "y.jpg", "editedBoxes": old, "editedBoxesSet": True},
    ]}
    store.save(pid, copy.deepcopy(m))
    writer_view = store.load(pid)
    real_read = store.read_bytes
    fired = []

    def reader_meets_a_save(path, attempts=20):
        data = real_read(path, attempts)
        if path.name == "x.json" and not fired:
            fired.append(1)
            edited = copy.deepcopy(writer_view)
            edited["imports"][0]["editedBoxes"] = new
            store.save(pid, edited)
        return data

    monkeypatch.setattr(store, "read_bytes", reader_meets_a_save)
    store.load(pid)
    monkeypatch.setattr(store, "read_bytes", real_read)
    undo = copy.deepcopy(writer_view)
    undo["imports"][0]["editedBoxes"] = old
    store.save(pid, undo)
    assert store.load(pid)["imports"][0]["editedBoxes"] == old


def test_crops_deleted_in_bulk_stay_deleted_after_a_resync(client, app_module):
    """As a single delete does, a bulk delete records each crop's source, so a
    re-sync from the parent does not make it again."""
    pid, ups = _dataset(client, "Bulk crop parent")
    client.put(f"/api/v2/projects/{pid}/imports/{ups[0]['import_id']}",
               json={"editedBoxes": [{"id": "b", "label": "apple", "x0": 2, "y0": 2, "x1": 40, "y1": 40}]})
    cid = client.post(f"/api/v2/projects/{pid}/derive",
                      data={"name": "Bulk crop child", "labels": json.dumps(["apple"]), "padding": "0",
                            "min_size": "0", "create_project": "false"}).json()["project_id"]
    for _ in range(100):
        crops = app_module.load_manifest(cid).get("imports") or []
        if crops:
            break
        time.sleep(0.1)
    assert len(crops) == 1
    r = client.post(f"/api/v2/projects/{cid}/imports/delete_batch", json={"ids": [crops[0]["id"]]})
    assert r.json()["deleted"] == [crops[0]["id"]]
    app_module._resync_child_from_thread(cid)
    assert app_module.load_manifest(cid).get("imports") == []
