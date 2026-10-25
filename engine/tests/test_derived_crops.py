"""Cropped child datasets: each crop's box sits on its object."""
import numpy as np
import pytest
from PIL import Image

import derived


def _parent(tmp_path, box):
    img = np.full((120, 200, 3), 255, np.uint8)
    x0, y0, x1, y1 = (round(v) for v in box)
    img[y0:y1, x0:x1] = 0
    (tmp_path / "parent").mkdir()
    Image.fromarray(img).save(tmp_path / "parent" / "a.png")
    return {"imports": [{"id": "i1", "filename": "a.png", "width": 200, "height": 120,
                         "detections": [{"box": list(box), "label": "bolt", "score": 0.9}]}]}


@pytest.mark.parametrize("crop", [{"fixedSize": 256}, {"minSize": 200}, {"square": True}])
def test_a_crop_keeps_its_box_on_the_object(tmp_path, crop):
    box = (60.6, 40.7, 81.4, 66.3)
    parent = _parent(tmp_path, box)
    child = {"derived": {"labels": ["bolt"], "crop": {"padding": 0.15, **crop}}, "imports": []}
    derived.resync(child, parent, tmp_path / "parent", tmp_path / "child", now_iso="2026-10-08T00:00:00Z")
    (entry,) = child["imports"]
    (det,) = entry["detections"]
    pic = np.asarray(Image.open(tmp_path / "child" / entry["filename"]).convert("L"))
    ys, xs = np.nonzero(pic < 128)
    # The object was drawn on whole pixels, so compare centres: an offset moves them.
    seen = ((xs.min() + xs.max() + 1) / 2, (ys.min() + ys.max() + 1) / 2)
    b = det["box"]
    assert ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) == pytest.approx(seen, abs=1.5)
    assert (entry["width"], entry["height"]) == pic.shape[::-1]


def test_a_crop_keeps_its_pictures_openverse_licence(tmp_path):
    parent = _parent(tmp_path, (60, 40, 81, 66))
    parent["imports"][0]["source"] = {"kind": "openverse", "url": "https://x.org/a.jpg",
                                      "license": "by", "license_version": "2.0"}
    child = {"derived": {"labels": ["bolt"], "crop": {"padding": 0.15}}, "imports": []}
    derived.resync(child, parent, tmp_path / "parent", tmp_path / "child", now_iso="2026-10-08T00:00:00Z")
    (entry,) = child["imports"]
    assert entry["source"] == {"kind": "openverse", "url": "https://x.org/a.jpg",
                               "license": "by", "license_version": "2.0", "cropped": True}
