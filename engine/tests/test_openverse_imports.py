"""Openverse imports keep each image's licence, and exports credit it."""
import io
import json
import zipfile

from PIL import Image

URL = "https://live.staticflickr.com/1/pothole.jpg"
RESULT = {
    "id": "a1", "url": URL, "thumbnail": URL, "title": "Pothole on Elm St",
    "creator": "Ann Example", "license": "by-sa", "license_version": "4.0",
    "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
    "attribution": '"Pothole on Elm St" by Ann Example is licensed under CC BY-SA 4.0.',
    "source": "flickr", "foreign_landing_url": "https://www.flickr.com/photos/ann/1",
    "width": 64, "height": 48,
}


def _jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), (90, 90, 90)).save(buf, "JPEG")
    return buf.getvalue()


def test_an_imported_openverse_image_keeps_its_licence_and_exports_credit_it(client, monkeypatch):
    import openverse

    monkeypatch.setattr(openverse, "search_images", lambda q, n, commercial=False: [dict(RESULT)])
    monkeypatch.setattr(openverse, "validate_thumbnails", lambda results: results)
    monkeypatch.setattr(openverse, "download_images_bytes",
                        lambda urls: [(_jpeg(), "image/jpeg") for _ in urls])

    r = client.post("/api/v2/projects", data={"name": "Openverse credit", "labels": json.dumps(["pothole"])})
    pid = r.json()["project_id"]
    r = client.get("/api/openverse/search", params={"q": "pothole", "count": 1, "commercial": True})
    assert r.status_code == 200, r.text
    r = client.post(f"/api/v2/projects/{pid}/imports/from_urls", json={"urls": [URL], "query": "pothole"})
    assert r.status_code == 200, r.text

    ov = client.get(f"/api/v2/projects/{pid}/overview").json()
    assert len(ov["imports"]) == 1
    zf = zipfile.ZipFile(io.BytesIO(client.get(
        f"/api/projects/{pid}/export", params={"format": "coco"}).content))
    credits = zf.read("ATTRIBUTIONS.txt").decode("utf-8")
    assert RESULT["attribution"] in credits
    assert RESULT["license_url"] in credits and RESULT["foreign_landing_url"] in credits
    stored = [n for n in zf.namelist() if n.startswith("images/")]
    assert stored and stored[0].rsplit("/", 1)[1] in credits

    # Without the image files there is nothing to credit.
    zf = zipfile.ZipFile(io.BytesIO(client.get(
        f"/api/projects/{pid}/export", params={"format": "coco", "include_images": False}).content))
    assert "ATTRIBUTIONS.txt" not in zf.namelist()


def test_attributions_cover_copies_and_imports_without_a_recorded_licence(app_module):
    manifest = {"imports": [
        {"filename": "a.jpg", "source": {"kind": "openverse", "url": URL, "title": "T",
                                          "creator": "C", "license": "cc0", "license_version": "1.0"}},
        {"filename": "b.jpg", "source": {"kind": "openverse", "url": "https://x.org/b.jpg"}},
        {"filename": "c.jpg"},
    ]}
    text = app_module._build_attributions(manifest, {"aug_a_00.jpg": "a.jpg", "aug_c_00.jpg": "c.jpg"})
    assert '"T" by C, CC0 1.0' in text
    assert "Changed copies: aug_a_00.jpg" in text and "aug_c_00.jpg" not in text
    assert "b.jpg\n  Licence not recorded" in text and "Source: https://x.org/b.jpg" in text
    assert "c.jpg" not in text
    assert app_module._build_attributions({"imports": [{"filename": "c.jpg"}]}, {}) == ""


def test_a_name_that_resolves_to_loopback_at_connect_time_is_not_fetched(monkeypatch):
    """DNS rebinding: the check sees a public address, the connect's own
    lookup gets 127.0.0.1. No connection is made."""
    import socket

    import openverse

    answers = iter([
        [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))],
        [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))],
    ])
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: next(answers))
    tried = []

    def connect(self, addr):
        tried.append(addr)
        raise OSError("no network in tests")

    monkeypatch.setattr(socket.socket, "connect", connect)
    assert openverse._download_bytes("https://rebind.example/a.jpg") == (None, None)
    assert tried == []
