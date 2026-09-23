"""Imports and exports that say what they mean: 16-bit greyscale keeps its
grey, a truncated JPEG is refused, VOC names are escaped, an edit is
visible to the next read, a restore does not bring back deleted pictures,
and the legacy import explains a bad path."""
import io
import json
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image


def _dataset(client, name, labels=("apple",)):
    return client.post("/api/v2/projects", data={"name": name, "labels": json.dumps(list(labels))}).json()["project_id"]


def _upload(client, pid, data, name="a.jpg", ctype="image/jpeg"):
    return client.post(f"/api/v2/projects/{pid}/imports/raw", files={"image": (name, data, ctype)})


def _jpeg(colour=(200, 30, 30), size=(64, 48)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, colour).save(buf, "JPEG", quality=95)
    return buf.getvalue()


def test_16_bit_greyscale_keeps_its_grey(client):
    pid = _dataset(client, "Grey16")
    buf = io.BytesIO()
    Image.fromarray(np.full((40, 60), 32768, dtype=np.uint16)).save(buf, "PNG")
    r = _upload(client, pid, buf.getvalue(), "grey.png", "image/png")
    assert r.status_code == 200
    img = client.get(f"/api/v2/projects/{pid}/imports/{r.json()['filename']}")
    mean = np.asarray(Image.open(io.BytesIO(img.content)).convert("RGB")).mean()
    assert 110 < mean < 145, mean  # mid-grey, not the white convert() clamped it to


def test_a_truncated_jpeg_is_refused(client):
    pid = _dataset(client, "Truncated")
    whole = _jpeg(size=(320, 240))
    r = _upload(client, pid, whole[: len(whole) // 2])
    assert r.status_code == 400 and "truncated" in r.json()["detail"]
    assert _upload(client, pid, whole).status_code == 200


def test_voc_names_are_escaped(client):
    pid = _dataset(client, "Salt", ["salt & pepper"])
    iid = _upload(client, pid, _jpeg()).json()["import_id"]
    client.put(f"/api/v2/projects/{pid}/imports/{iid}",
               json={"editedBoxes": [{"id": "b", "label": "salt & pepper", "x0": 2, "y0": 2, "x1": 30, "y1": 30}]})
    r = client.get(f"/api/projects/{pid}/export", params={"format": "voc"})
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    xml = next(n for n in zf.namelist() if n.endswith(".xml"))
    assert ET.fromstring(zf.read(xml)).find("object/name").text == "salt & pepper"


def test_an_edit_is_visible_to_the_next_read(client):
    pid = _dataset(client, "Fresh")
    iid = _upload(client, pid, _jpeg()).json()["import_id"]
    box = lambda x0: [{"id": "b", "label": "apple", "x0": x0, "y0": 2, "x1": x0 + 20, "y1": 30}]
    client.put(f"/api/v2/projects/{pid}/imports/{iid}", json={"editedBoxes": box(5)})
    client.get(f"/api/v2/projects/{pid}/annotations", params={"scope": "imports"})  # primes the cache
    client.put(f"/api/v2/projects/{pid}/imports/{iid}", json={"editedBoxes": box(30)})
    got = client.get(f"/api/v2/projects/{pid}/annotations", params={"scope": "imports"}).json()
    assert got["imports"][iid]["editedBoxes"][0]["x0"] == 30


def test_a_restore_does_not_bring_back_deleted_pictures(client, app_module):
    pid = _dataset(client, "Restore")
    keep = _upload(client, pid, _jpeg((10, 200, 10))).json()["import_id"]
    gone = _upload(client, pid, _jpeg((10, 10, 200))).json()["import_id"]
    snap = client.post(f"/api/v2/projects/{pid}/snapshots").json()["id"]
    assert client.delete(f"/api/v2/projects/{pid}/imports/{gone}").status_code == 200
    r = client.post(f"/api/v2/projects/{pid}/snapshots/{snap}/restore")
    assert r.status_code == 200 and r.json()["dropped"] == 1
    ids = [e["id"] for e in app_module.load_manifest(pid)["imports"]]
    assert ids == [keep]


def test_the_legacy_import_explains_a_bad_path(client, tmp_path):
    assert client.post("/api/import-legacy", json={"path": "   "}).status_code == 400
    r = client.post("/api/import-legacy", json={"path": str(tmp_path)})
    assert r.status_code == 400 and "no projects/" in r.json()["detail"]
