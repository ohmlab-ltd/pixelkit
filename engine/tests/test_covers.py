"""A dataset's uploaded cover, and the settings that read and reset it."""
import io
import json
import os
import time

from PIL import Image


def _jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (40, 30), (30, 160, 90)).save(buf, "JPEG")
    return buf.getvalue()


def _dataset_with_uploaded_cover(client, name):
    pid = client.post("/api/v2/projects", data={"name": name, "labels": json.dumps(["leaf"])}).json()["project_id"]
    client.post(f"/api/v2/projects/{pid}/imports/raw", files={"image": ("a.jpg", _jpeg(), "image/jpeg")})
    r = client.post(f"/api/projects/{pid}/cover_upload", files={"file": ("c.jpg", _jpeg(), "image/jpeg")})
    assert r.status_code == 200, r.text
    return pid


def _settled(app_module, pid):
    deadline = time.time() + 10
    while pid in app_module._SIDECAR_REFRESH_IN_FLIGHT and time.time() < deadline:
        time.sleep(0.05)


def test_reset_replaces_an_uploaded_cover(client, app_module):
    pid = _dataset_with_uploaded_cover(client, "Reset cover")
    assert client.get(f"/api/v2/projects/{pid}/overview?imports_limit=0").json()["cover_uploaded"] is True
    assert client.put(f"/api/projects/{pid}", json={"cover": None}).status_code == 200
    assert app_module.load_manifest(pid).get("cover_uploaded") is False
    assert client.get(f"/api/v2/projects/{pid}/overview?imports_limit=0").json()["cover_uploaded"] is False
    card = next(p for p in client.get("/api/projects").json() if p["id"] == pid)
    assert card["thumbnail"] != app_module.UPLOADED_COVER_SENTINEL


def test_an_overview_sidecar_without_cover_uploaded_is_rebuilt(client, app_module):
    pid = _dataset_with_uploaded_cover(client, "Old sidecar")
    _settled(app_module, pid)
    app_module._write_overview_sidecar(pid)
    path = app_module._overview_sidecar_path(pid)
    side = json.loads(path.read_bytes())
    del side["cover_uploaded"]
    path.write_text(json.dumps(side))
    stamp = app_module._manifest_disk_mtime(pid) + 1
    os.utime(path, (stamp, stamp))
    body = client.get(f"/api/v2/projects/{pid}/overview?imports_limit=0").json()
    assert body["cover_uploaded"] is True
