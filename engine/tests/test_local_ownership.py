"""Every dataset on this machine is the local user's, whatever owner it was made under."""
import io
import json

from PIL import Image


def _jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 24), (10, 120, 200)).save(buf, "JPEG")
    return buf.getvalue()


def _hosted_dataset(client, app_module, name: str) -> str:
    r = client.post("/api/v2/projects", data={"name": name, "labels": json.dumps(["bolt"])})
    pid = r.json()["project_id"]
    client.post(f"/api/v2/projects/{pid}/imports/raw", files={"image": ("a.jpg", _jpeg(), "image/jpeg")})
    m = app_module.load_manifest(pid)
    m["owner"] = m["createdBy"] = "someone@hosted"
    m["private"] = True
    app_module.save_manifest(pid, m)
    return pid


def test_a_dataset_made_under_another_owner_is_listed_and_can_join_a_project(client, app_module):
    pid = _hosted_dataset(client, app_module, "Hosted import")
    r = client.get("/api/projects", params={"limit": 1000})
    assert r.status_code == 200, r.text
    assert pid in [it["id"] for it in r.json()["items"]]

    r = client.post("/api/containers", json={"name": "Shop floor"})
    assert r.status_code == 200, r.text
    cid = r.json()["id"]
    r = client.post(f"/api/containers/{cid}/datasets/{pid}")
    assert r.status_code == 200, r.text


def test_filter_counts_treat_cleared_boxes_as_unlabelled(app_module):
    counts = app_module._compute_filter_counts([
        {"id": "a", "detections": [{"box": [0, 0, 5, 5]}], "editedBoxes": [], "editedBoxesSet": True},
        {"id": "b", "detections": [{"box": [0, 0, 5, 5]}]},
        {"id": "c", "background": True},
    ], {})
    assert counts["unlabelled"] == 1 and counts["unrated"] == 1 and counts["all"] == 3


def test_only_loopback_host_names_reach_the_api(client):
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/health", headers={"host": "localhost:8001"}).status_code == 200
    # A page that rebinds its own domain to 127.0.0.1 sends that domain.
    assert client.get("/api/health", headers={"host": "attacker.example:8001"}).status_code == 400


def test_a_page_on_another_site_cannot_change_anything(client):
    """A form post from another site reaches the engine whatever CORS says;
    the engine refuses it. The app's own pages, scripts and reads pass."""
    form = {"name": "From elsewhere", "labels": json.dumps(["bolt"])}
    for origin in ("https://attacker.example", "null", "http://localhost.attacker.example"):
        r = client.post("/api/v2/projects", data=form, headers={"origin": origin})
        assert r.status_code == 403, origin
    assert client.post("/api/models/dinov2/load", headers={"origin": "https://attacker.example"}).status_code == 403
    for origin in ("http://127.0.0.1:8001", "http://localhost:3000", None):
        headers = {"origin": origin} if origin else {}
        assert client.post("/api/v2/projects", data=form, headers=headers).status_code == 200, origin
    r = client.get("/api/health", headers={"origin": "https://attacker.example"})
    assert r.status_code == 200 and r.headers["x-content-type-options"] == "nosniff"


def test_a_reference_is_stored_as_the_image_it_is(client, app_module):
    pid = client.post("/api/v2/projects", data={"name": "Refs", "labels": json.dumps(["bolt"])}).json()["project_id"]
    up = lambda name, data: client.post(  # noqa: E731
        f"/api/v2/projects/{pid}/references",
        files={"image": (name, data, "application/octet-stream")}, data={"detections": "[]"})
    # Named .html, but a JPEG: stored under the extension of what it is.
    r = up("page.html", _jpeg())
    assert r.status_code == 200, r.text
    stored = [ref["filename"] for ref in app_module.load_manifest(pid)["references"]]
    assert len(stored) == 1 and stored[0].endswith(".jpg")
    assert up("page.html", b"<script>alert(1)</script>").status_code == 400
    assert len(app_module.load_manifest(pid)["references"]) == 1
