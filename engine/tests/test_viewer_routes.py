"""Routes the workspace and the image viewer call: dataset cards, Projects and
their activity, starring a dataset, and segmenting a box that was drawn, moved
or resized."""
import io
import json
import os
import time

import pytest
from PIL import Image


def _jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), (90, 120, 150)).save(buf, "JPEG")
    return buf.getvalue()


def _dataset(client, name, labels=("bolt", "nut")):
    pid = client.post("/api/v2/projects", data={"name": name, "labels": json.dumps(list(labels))}).json()["project_id"]
    r = client.post(f"/api/v2/projects/{pid}/imports/raw",
                    files={"image": ("a.jpg", _jpeg(), "image/jpeg")})
    return pid, r.json()["import_id"]


def _card(client, pid):
    items = client.get("/api/projects?offset=0&limit=500").json()["items"]
    return next(p for p in items if p["id"] == pid), items


def test_a_star_toggles_and_starred_datasets_list_first(client):
    pid, _ = _dataset(client, "Starred")
    r = client.post(f"/api/projects/{pid}/favourite")
    assert r.status_code == 200 and r.json() == {"favouritedByMe": True}
    card, items = _card(client, pid)
    assert card["favouritedByMe"] is True and items[0]["id"] == pid
    r = client.post(f"/api/projects/{pid}/favourite")
    assert r.json() == {"favouritedByMe": False}
    assert _card(client, pid)[0]["favouritedByMe"] is False
    assert client.post("/api/projects/missing0000/favourite").status_code == 404


def test_a_card_built_across_a_save_is_not_taken_as_current(client, app_module, monkeypatch):
    pid = client.post("/api/v2/projects", data={"name": "Starred mid-build", "labels": "[]"}).json()["project_id"]
    deadline = time.time() + 10
    while pid in app_module._SIDECAR_REFRESH_IN_FLIGHT and time.time() < deadline:
        time.sleep(0.05)
    card_path = app_module._workspace_card_sidecar_path(pid)
    app_module._write_workspace_card_sidecar(pid)
    assert card_path.stat().st_mtime_ns == app_module._manifest_mtime_ns(pid)
    real = app_module.load_manifest

    def load_then_star(project_id, copy=True):
        m = real(project_id, copy)
        monkeypatch.setattr(app_module, "load_manifest", real)
        time.sleep(0.05)
        later = real(pid)
        later["favouritedBy"] = [app_module.LOCAL_USER]
        app_module.save_manifest(pid, later)
        return m

    monkeypatch.setattr(app_module, "load_manifest", load_then_star)
    app_module._write_workspace_card_sidecar(pid)
    (card,) = client.get("/api/projects", params={"q": "Starred mid-build", "limit": 5}).json()["items"]
    assert card["favouritedByMe"] is True


def test_routes_of_features_the_app_does_not_have_are_gone(client, app_module):
    pid, _ = _dataset(client, "Gone")
    (fn,) = [e["filename"] for e in app_module.load_manifest(pid)["imports"]]
    assert client.get(f"/api/projects/{pid}/originals/{fn}").status_code == 404
    assert client.post(f"/api/v2/projects/{pid}/augment/object_overlay/segment").status_code == 405
    assert client.delete(f"/api/projects/{pid}/jobs/abc123").status_code == 405


def test_a_deleted_dataset_leaves_the_list(client):
    pid, _ = _dataset(client, "Deleted")
    assert client.delete(f"/api/projects/{pid}").status_code == 200
    assert pid not in [p["id"] for p in client.get("/api/projects").json()]
    assert client.get(f"/api/v2/projects/{pid}/overview").status_code == 404


def test_an_empty_dataset_is_listed(client):
    pid = client.post("/api/v2/projects", data={"name": "Empty", "labels": "[]"}).json()["project_id"]
    assert pid in [p["id"] for p in client.get("/api/projects").json()]


class FakeSam3:
    """segment_box as the pipeline answers it, recording the labels it tried."""

    def __init__(self, mask=True):
        self.mask, self.tried = mask, []

    def segment_box(self, pil, box, labels=None, cache_key=None):
        self.tried.append(list(labels or []))
        if not self.mask:
            return None, {"total_ms": 1.0}
        x0, y0, x1, y1 = box
        poly = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        return {"box": box, "mask": {"polygons": [poly]}, "gd_label": (labels or ["?"])[0]}, {"total_ms": 1.0}


@pytest.fixture
def sam3(app_module):
    def install(**kw):
        app_module.state["charlie"] = FakeSam3(**kw)
        return app_module.state["charlie"]

    yield install
    app_module.state["charlie"] = None


def test_a_moved_box_is_segmented_again(client, sam3):
    pid, iid = _dataset(client, "Segment")
    form = {"box": json.dumps([4, 4, 30, 20]), "project_id": pid, "import_id": iid}
    assert client.post("/api/charlie/imports/segment_box", data=form).status_code == 503
    fake = sam3()
    r = client.post("/api/charlie/imports/segment_box", data={**form, "labels": json.dumps(["nut"])})
    assert r.status_code == 200, r.text
    assert r.json()["mask"]["polygons"][0][0] == [4.0, 4.0] and fake.tried == [["nut"]]
    # Without labels, the dataset's.
    client.post("/api/charlie/imports/segment_box", data=form)
    assert fake.tried[-1] == ["bolt", "nut"]
    sam3(mask=False)
    assert client.post("/api/charlie/imports/segment_box", data=form).status_code == 422
    assert client.post("/api/charlie/imports/segment_box",
                       data={**form, "box": "[1, 2]"}).status_code == 400


def test_reference_masks_reach_the_editor(client):
    """The editor writes back the reference boxes it holds, so it is given
    their masks; without them, opening a dataset erased the stored ones."""
    pid, _ = _dataset(client, "Ref masks")
    poly = [[2.0, 2.0], [20.0, 2.0], [20.0, 20.0], [2.0, 20.0]]
    dets = [{"label": "bolt", "score": 0.9, "box": [2, 2, 20, 20], "mask": {"polygons": [poly]}}]
    r = client.post(f"/api/v2/projects/{pid}/references",
                    files={"image": ("r.jpg", _jpeg(), "image/jpeg")},
                    data={"detections": json.dumps(dets), "label": "bolt"})
    assert r.status_code == 200, r.text
    refs = client.get(f"/api/v2/projects/{pid}/annotations?scope=refs").json()["references"]
    (ref,) = refs.values()
    assert ref["detections"][0]["mask"]["polygons"] == [poly]
    assert "embedding" not in ref["detections"][0]


def test_the_model_badge_follows_the_runs_with_an_export(client, app_module):
    """The workspace card and the Project page show a model while a run of
    the dataset has an export."""
    import store

    pid, _ = _dataset(client, "Badge")
    runs = ["20990102T000000Z", "20990103T000000Z"]
    for run in runs:
        d = store.dataset_dir(pid) / "models" / "yolox_nano" / run
        d.mkdir(parents=True)
        (d / "export.zip").write_bytes(b"PK")

    def badge():
        return _card(client, pid)[0]["hasModel"]

    assert client.delete(f"/api/v2/projects/{pid}/models/{runs[0]}").status_code == 200
    assert badge() is True
    cid = client.post("/api/containers", json={"name": "Badges"}).json()["id"]
    client.post(f"/api/containers/{cid}/datasets/{pid}")
    (row,) = client.get(f"/api/containers/{cid}").json()["datasets"]
    assert row["hasModel"] is True and row["updated"]
    assert client.delete(f"/api/v2/projects/{pid}/models/{runs[1]}").status_code == 200
    assert badge() is False


def test_an_upload_request_is_one_activity_row(client):
    pid, _ = _dataset(client, "Activity")
    cid = client.post("/api/containers", json={"name": "Activity project"}).json()["id"]
    client.post(f"/api/containers/{cid}/datasets/{pid}")
    r = client.post(f"/api/v2/projects/{pid}/imports/raw_batch",
                    files=[("images", (f"{i}.jpg", _jpeg(), "image/jpeg")) for i in range(3)])
    assert r.status_code == 200, r.text
    rows = client.get(f"/api/containers/{cid}/activity").json()["activity"]
    # The dataset's first image, then the batch of three.
    assert [a["n_images"] for a in rows if a["job_kind"] == "upload" and a["dataset"] == pid] == [1, 3]


def test_a_project_cover_is_stored_and_served(client, workspace_dir):
    cid = client.post("/api/containers", json={"name": "Covered"}).json()["id"]
    assert client.get(f"/api/containers/{cid}/cover").status_code == 404
    r = client.post(f"/api/containers/{cid}/cover", files={"file": ("c.jpg", _jpeg(), "image/jpeg")})
    assert r.status_code == 200 and r.json()["cover"] == "cover.jpg"
    assert (workspace_dir / "_storage" / "containers" / cid / "cover.jpg").is_file()
    got = client.get(f"/api/containers/{cid}/cover")
    assert got.status_code == 200 and got.headers["content-type"] == "image/jpeg"
    detail = client.get(f"/api/containers/{cid}").json()
    assert detail["cover"] == "cover.jpg" and detail["cover_updated"] and detail["coverLuma"] >= 0


def test_an_upload_row_holds_what_the_activity_reads(client):
    import audit

    pid, iid = _dataset(client, "Upload row")
    (row,) = [e for e in audit.list_events("job") if e.get("id") == iid]
    assert row["job_kind"] == "upload" and row["project"] == pid and row["n_images"] == 1
    assert "user" not in row and "cost_pence" not in row


def test_activity_lists_the_kinds_the_engine_writes(client):
    import audit

    pid, _ = _dataset(client, "Old events")
    cid = client.post("/api/containers", json={"name": "Old events project"}).json()["id"]
    client.post(f"/api/containers/{cid}/datasets/{pid}")
    for kind in ("container_privacy", "container_max_input", "member_add", "member_remove"):
        audit.add_event(kind, container=cid, member="someone", role="editor")
    # A row of the shape an existing events.db holds.
    audit.add_event("job", id="older-upload", job_kind="upload", project=pid, user="someone",
                    status="done", elapsed_s=0.0, cost_pence=0.0, n_images=2, error=None)
    rows = client.get(f"/api/containers/{cid}/activity").json()["activity"]
    assert {a["kind"] for a in rows} == {"container_create", "dataset_add", "job"}
    assert 2 in [a["n_images"] for a in rows if a["job_kind"] == "upload"]


def test_the_search_matches_names_and_labels(client):
    pid, _ = _dataset(client, "Gearbox", labels=("spur gear",))

    def found(q):
        items = client.get("/api/projects", params={"q": q, "limit": 500}).json()["items"]
        return pid in [p["id"] for p in items]

    assert found("gearb") and found("spur")
    # Every dataset is made under the owner "local", which is not searched.
    assert not found("loc")


def test_files_from_another_version_add_no_fields(client, app_module):
    import containers

    pid, _ = _dataset(client, "Older card")
    deadline = time.time() + 10
    while pid in app_module._SIDECAR_REFRESH_IN_FLIGHT and time.time() < deadline:
        time.sleep(0.05)
    app_module._write_workspace_card_sidecar(pid)
    path = app_module._workspace_card_sidecar_path(pid)
    card = json.loads(path.read_bytes())
    card.update(certified=True, private=True, dataset_health=90, owner="someone",
                createdBy="someone", likedBy=["someone"])
    path.write_text(json.dumps(card))
    stamp = app_module._manifest_disk_mtime(pid) + 1
    os.utime(path, (stamp, stamp))
    (listed,) = client.get("/api/projects", params={"q": "Older card", "limit": 50}).json()["items"]
    assert set(listed) == set(app_module._CARD_KEYS) - {"_cover_subdir"} | {"favouritedByMe", "running", "container"}

    cid = client.post("/api/containers", json={"name": "Older project"}).json()["id"]
    c = containers.load_container(cid)
    c.update(owner="someone", members=[{"username": "someone", "role": "editor"}], private=True,
             max_input_size=640)
    containers.save_container(c)
    hosted = {"owner", "members", "private", "max_input_size"}
    assert not hosted & set(client.get(f"/api/containers/{cid}").json())
    assert not hosted & set(client.patch(f"/api/containers/{cid}", json={"name": "Renamed"}).json())
