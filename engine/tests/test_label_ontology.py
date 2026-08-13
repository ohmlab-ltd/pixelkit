"""Project-level label usage + cross-dataset rename/merge."""
import io
import json

from PIL import Image


def _jpeg_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), (30, 30, 200)).save(buf, "JPEG")
    return buf.getvalue()


def _dataset(client, name: str, label: str) -> str:
    r = client.post(
        "/api/v2/projects", data={"name": name, "labels": json.dumps([label])}
    )
    pid = r.json()["project_id"]
    r = client.post(
        f"/api/v2/projects/{pid}/imports/raw",
        files={"image": ("a.jpg", _jpeg_bytes(), "image/jpeg")},
    )
    iid = r.json()["import_id"]
    boxes = [{"id": "b1", "label": label, "x0": 4, "y0": 4, "x1": 40, "y1": 40}]
    r = client.put(
        f"/api/v2/projects/{pid}/imports/{iid}", json={"editedBoxes": boxes}
    )
    assert r.status_code == 200, r.text
    return pid


def test_container_label_usage_and_merge(client):
    r = client.post("/api/containers", json={"name": "Ontology", "private": True})
    cid = r.json()["id"]
    pid_a = _dataset(client, "Bolts A", "bolt")
    pid_b = _dataset(client, "Bolts B", "bolts")
    for pid in (pid_a, pid_b):
        assert client.post(f"/api/containers/{cid}/datasets/{pid}").status_code == 200

    r = client.get(f"/api/containers/{cid}/labels")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["datasets"] == 2
    by_label = {row["label"]: row for row in body["labels"]}
    assert by_label["bolt"]["boxes"] == 1 and by_label["bolt"]["datasets"] == 1
    assert by_label["bolts"]["boxes"] == 1

    # Merge "bolts" into "bolt" across the whole Project.
    r = client.post(
        f"/api/containers/{cid}/labels/rename",
        json={"old_label": "bolts", "new_label": "bolt"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["datasetsTouched"] == 1

    r = client.get(f"/api/containers/{cid}/labels")
    by_label = {row["label"]: row for row in r.json()["labels"]}
    assert "bolts" not in by_label
    assert by_label["bolt"]["boxes"] == 2 and by_label["bolt"]["datasets"] == 2


def test_dataset_rename_covers_v2_records(client):
    # The per-dataset rename must rewrite labels stored inside V2
    # imports[] records (detections/editedBoxes), not just V1 fields.
    pid = _dataset(client, "Rename V2", "cat")
    r = client.post(
        f"/api/projects/{pid}/labels/rename",
        json={"old_label": "cat", "new_label": "kitten"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["renamed"] >= 2  # tag + the edited box

    r = client.get(f"/api/v2/projects/{pid}/initial")
    imports = r.json().get("imports") or r.json().get("items") or []
    # Read the annotation back directly — the box label must be renamed.
    r = client.get(f"/api/v2/projects/{pid}/annotations")
    # fall back to per-import fetch if the bulk route doesn't exist
    if r.status_code == 404:
        assert imports, "no imports listed"
        iid = imports[0]["id"]
        r = client.get(f"/api/v2/projects/{pid}/annotations/{iid}")
    assert r.status_code == 200, r.text
    text = json.dumps(r.json())
    assert "kitten" in text and '"cat"' not in text
