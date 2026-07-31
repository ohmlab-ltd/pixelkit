"""Regression tests for upload DoS caps (finding S4).

The server caps per-file size, batch file count, and the Pillow
decompression-bomb pixel ceiling so a single request can't exhaust memory
/ disk / GPU before quota or NSFW screening engage. S4 verified these were
already enforced and made no change; these tests lock them in.

Behaviour is exercised through the real `_enforce_upload_caps()` guard (with
the caps monkeypatched small so we don't allocate 100 MB), plus bound checks
on the constants themselves. conftest.py stubs the heavy ML stack.
"""
import pytest
from fastapi import HTTPException


def test_enforce_upload_caps_allows_within_limits(app_module):
    # One small file, comfortably under both caps — must not raise.
    app_module._enforce_upload_caps([("a.png", b"x" * 10, "image/png")])


def test_enforce_upload_caps_rejects_oversize_file(app_module, monkeypatch):
    monkeypatch.setattr(app_module, "MAX_UPLOAD_BYTES_PER_FILE", 16)
    with pytest.raises(HTTPException) as exc:
        app_module._enforce_upload_caps([("big.png", b"x" * 17, "image/png")])
    assert exc.value.status_code == 413


def test_enforce_upload_caps_rejects_too_many_files(app_module, monkeypatch):
    monkeypatch.setattr(app_module, "MAX_FILES_PER_UPLOAD_BATCH", 2)
    blobs = [(f"{i}.png", b"x", "image/png") for i in range(3)]
    with pytest.raises(HTTPException) as exc:
        app_module._enforce_upload_caps(blobs)
    assert exc.value.status_code == 413


def test_upload_caps_constants_stay_bounded(app_module):
    # Regression guard against someone removing or loosening the DoS caps:
    # they must remain finite, positive, and no larger than today's ceiling.
    assert 0 < app_module.MAX_UPLOAD_BYTES_PER_FILE <= 100 * 1024 * 1024
    assert 0 < app_module.MAX_FILES_PER_UPLOAD_BATCH <= 100


def test_pillow_decompression_bomb_ceiling_is_set():
    from PIL import Image as PILImage
    # `None` disables Pillow's decompression-bomb guard entirely — it must
    # stay a finite ceiling.
    assert PILImage.MAX_IMAGE_PIXELS is not None
    assert PILImage.MAX_IMAGE_PIXELS <= 100_000_000


if __name__ == "__main__":  # allow running as a plain script too
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
