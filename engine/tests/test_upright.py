"""Images with EXIF orientation are read upright wherever their pixels meet a box."""
import io
import json
import time

import numpy as np
from PIL import Image

from boxes import _box_xyxy


def _rotated_jpeg() -> bytes:
    """Stored 80x40 with orientation 6, shown 40x80: top half red, bottom
    half blue as shown."""
    raw = Image.new("RGB", (80, 40), (255, 0, 0))
    raw.paste((0, 0, 255), (40, 0, 80, 40))  # a quarter turn clockwise puts it at the bottom
    exif = Image.Exif()
    exif[0x0112] = 6
    buf = io.BytesIO()
    raw.save(buf, "JPEG", quality=95, exif=exif.tobytes())
    return buf.getvalue()


def test_helpers_read_the_picture_as_shown(app_module):
    data = _rotated_jpeg()
    im = app_module._open_upright(data)
    assert im.size == (40, 80)
    px = np.asarray(im)
    assert px[10, 20, 0] > 200 and px[70, 20, 2] > 200  # red on top, blue below
    with Image.open(io.BytesIO(data)) as raw:
        assert app_module._upright_size(raw) == (40, 80)


def test_crops_and_previews_of_a_rotated_photo(client, app_module):
    r = client.post("/api/v2/projects", data={"name": "Upright", "labels": json.dumps(["blue"])})
    pid = r.json()["project_id"]
    r = client.post(f"/api/v2/projects/{pid}/imports/raw",
                    files={"image": ("phone.jpg", _rotated_jpeg(), "image/jpeg")})
    up = r.json()
    assert (up["width"], up["height"]) == (40, 80)
    iid = up["import_id"]
    box = {"id": "b", "label": "blue", "x0": 2, "y0": 44, "x1": 38, "y1": 78}
    assert client.put(f"/api/v2/projects/{pid}/imports/{iid}",
                      json={"editedBoxes": [box]}).status_code == 200

    prev = client.get(f"/api/v2/projects/{pid}/imports/{iid}/labelled_preview")
    assert prev.status_code == 200
    assert Image.open(io.BytesIO(prev.content)).size == (40, 80)

    r = client.post(f"/api/v2/projects/{pid}/derive",
                    data={"name": "Upright crops", "labels": json.dumps(["blue"]), "padding": "0",
                          "min_size": "0", "create_project": "false"})
    cid = r.json()["project_id"]
    crops = []
    for _ in range(100):
        crops = app_module.load_manifest(cid).get("imports") or []
        if crops:
            break
        time.sleep(0.1)
    assert crops, "no crop was made"
    img = client.get(f"/api/v2/projects/{cid}/imports/{crops[0]['filename']}")
    mean = np.asarray(Image.open(io.BytesIO(img.content)).convert("RGB")).reshape(-1, 3).mean(0)
    assert mean[2] > 200 and mean[0] < 60, mean  # the blue half, not the red


def test_sam3_and_model_boxes_are_read_as_boxes():
    assert _box_xyxy({"box": [10, 20, 110, 220], "gd_label": "cat"}) == (10.0, 20.0, 110.0, 220.0)
