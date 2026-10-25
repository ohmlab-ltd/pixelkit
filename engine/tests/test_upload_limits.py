"""Upload caps: per-file size, files per batch and Pillow's decompression-bomb ceiling."""
import asyncio
import io

import pytest
from fastapi import HTTPException
from starlette.datastructures import UploadFile


def _upload(data: bytes, size: int | None) -> UploadFile:
    return UploadFile(io.BytesIO(data), size=size, filename="a.png")


def test_an_upload_within_the_cap_is_read(app_module):
    assert asyncio.run(app_module._read_upload(_upload(b"x" * 10, 10))) == b"x" * 10


def test_an_upload_over_the_cap_is_refused(app_module, monkeypatch):
    monkeypatch.setattr(app_module, "MAX_UPLOAD_BYTES_PER_FILE", 16)
    # Refused from its size before a read, or after reading one byte past the cap.
    for size in (17, None):
        with pytest.raises(HTTPException) as exc:
            asyncio.run(app_module._read_upload(_upload(b"x" * 17, size)))
        assert exc.value.status_code == 413


def test_a_batch_over_the_file_count_is_refused(client, monkeypatch, app_module):
    monkeypatch.setattr(app_module, "MAX_FILES_PER_UPLOAD_BATCH", 2)
    pid = client.post("/api/v2/projects", data={"name": "Batch cap", "labels": "[\"a\"]"}).json()["project_id"]
    r = client.post(f"/api/v2/projects/{pid}/imports/raw_batch",
                    files=[("images", (f"{i}.jpg", b"x", "image/jpeg")) for i in range(3)])
    assert r.status_code == 413


def test_upload_caps_constants_stay_bounded(app_module):
    # The caps stay finite, positive and within these ceilings.
    assert 0 < app_module.MAX_UPLOAD_BYTES_PER_FILE <= 100 * 1024 * 1024
    assert 0 < app_module.MAX_FILES_PER_UPLOAD_BATCH <= 100


def test_pillow_decompression_bomb_ceiling_is_set():
    from PIL import Image as PILImage
    # None would disable Pillow's decompression-bomb guard.
    assert PILImage.MAX_IMAGE_PIXELS is not None
    assert PILImage.MAX_IMAGE_PIXELS <= 100_000_000


def test_a_heic_photo_imports_as_a_jpeg(client):
    """HEIC decodes through pi-heif; the stored copy is a JPEG of the same size."""
    import io
    import json
    from pathlib import Path

    pytest.importorskip("pi_heif")
    from PIL import Image as PILImage

    heic = (Path(__file__).parent / "data" / "two-colours.heic").read_bytes()
    r = client.post("/api/v2/projects", data={"name": "Heic", "labels": json.dumps(["x"])})
    pid = r.json()["project_id"]
    r = client.post(f"/api/v2/projects/{pid}/imports/raw",
                    files={"image": ("photo.heic", heic, "image/heic")})
    assert r.status_code == 200, r.text
    assert (r.json()["width"], r.json()["height"]) == (64, 48)
    r = client.get(f"/api/v2/projects/{pid}/imports/{r.json()['filename']}")
    assert r.status_code == 200, r.text
    im = PILImage.open(io.BytesIO(r.content))
    assert im.format == "JPEG" and im.size == (64, 48)
    left, right = im.convert("RGB").getpixel((5, 5)), im.convert("RGB").getpixel((60, 40))
    assert left[2] > 150 > left[0] and right[0] > 150 > right[2]


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
