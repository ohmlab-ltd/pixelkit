"""Dataset snapshots: create, list, restore (with safety snapshot)."""
import io
import json

from PIL import Image


def _jpeg_bytes(w: int = 64, h: int = 48) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (30, 200, 30)).save(buf, "JPEG")
    return buf.getvalue()


def _dataset_with_box(client, name: str, label: str = "apple"):
    r = client.post(
        "/api/v2/projects",
        data={"name": name, "labels": json.dumps([label])},
    )
    pid = r.json()["project_id"]
    r = client.post(
        f"/api/v2/projects/{pid}/imports/raw",
        files={"image": ("a.jpg", _jpeg_bytes(), "image/jpeg")},
    )
    iid = r.json()["import_id"]
    _put_box(client, pid, iid, label)
    return pid, iid


def _put_box(client, pid: str, iid: str, label: str):
    boxes = [{"id": "b1", "label": label, "x0": 4, "y0": 4, "x1": 40, "y1": 40}]
    r = client.put(
        f"/api/v2/projects/{pid}/imports/{iid}", json={"editedBoxes": boxes}
    )
    assert r.status_code == 200, r.text


def test_snapshot_roundtrip(client):
    pid, iid = _dataset_with_box(client, "Snap RT", "apple")

    r = client.post(f"/api/v2/projects/{pid}/snapshots")
    assert r.status_code == 200, r.text
    snap = r.json()
    assert snap["annotations"] == 1

    # Mutate the annotation, then restore the snapshot.
    _put_box(client, pid, iid, "banana")
    r = client.get(f"/api/v2/projects/{pid}/annotations/{iid}")
    assert r.json()["editedBoxes"][0]["label"] == "banana"

    r = client.post(f"/api/v2/projects/{pid}/snapshots/{snap['id']}/restore")
    assert r.status_code == 200, r.text
    assert r.json()["safetySnapshot"]

    r = client.get(f"/api/v2/projects/{pid}/annotations/{iid}")
    assert r.json()["editedBoxes"][0]["label"] == "apple"

    # Listing shows the manual snapshot plus the pre-restore safety copy.
    r = client.get(f"/api/v2/projects/{pid}/snapshots")
    ids = [s["id"] for s in r.json()["snapshots"]]
    assert snap["id"] in ids and len(ids) >= 2


def test_snapshot_restore_guards(client):
    pid, _ = _dataset_with_box(client, "Snap Guards")
    r = client.post(f"/api/v2/projects/{pid}/snapshots/not-a-valid-id/restore")
    assert r.status_code == 400
    r = client.post(
        f"/api/v2/projects/{pid}/snapshots/20990101T000000Z-deadbeef/restore"
    )
    assert r.status_code == 404
