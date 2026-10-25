"""CVAT, Label Studio and PNG-mask exports: shape and content."""
import io
import json
import time
import zipfile

from PIL import Image


def _jpeg_bytes(w: int = 64, h: int = 48) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (200, 30, 30)).save(buf, "JPEG")
    return buf.getvalue()


def _make_dataset(client, name: str) -> str:
    r = client.post(
        "/api/v2/projects",
        data={"name": name, "labels": json.dumps(["apple"])},
    )
    assert r.status_code == 200, r.text
    pid = r.json()["project_id"]
    r = client.post(
        f"/api/v2/projects/{pid}/imports/raw",
        files={"image": ("a.jpg", _jpeg_bytes(), "image/jpeg")},
    )
    assert r.status_code == 200, r.text
    iid = r.json()["import_id"]
    boxes = [{
        "id": "b1",
        "label": "apple",
        "x0": 4, "y0": 4, "x1": 40, "y1": 40,
        "mask": {"polygons": [[[5, 5], [38, 6], [37, 37], [6, 36]]]},
    }]
    r = client.put(
        f"/api/v2/projects/{pid}/imports/{iid}", json={"editedBoxes": boxes}
    )
    assert r.status_code == 200, r.text
    return pid


def _export_zip(client, pid: str, fmt: str, **params) -> zipfile.ZipFile:
    r = client.get(
        f"/api/projects/{pid}/export", params={"format": fmt, **params}
    )
    assert r.status_code == 200, r.text
    return zipfile.ZipFile(io.BytesIO(r.content))


def test_cvat_export(client):
    pid = _make_dataset(client, "Cvat Fmt")
    zf = _export_zip(client, pid, "cvat")
    names = zf.namelist()
    assert "annotations.xml" in names, names
    # The xml names the content-hash file shipped in images/.
    imgs = [n for n in names if n.startswith("images/") and n.endswith(".jpg")]
    assert len(imgs) == 1, names
    xml = zf.read("annotations.xml").decode()
    assert f'name="{imgs[0].split("/", 1)[1]}"' in xml
    assert 'label="apple"' in xml
    assert "<polygon" in xml  # mask present -> polygon preferred


def test_cvat_boxes_only(client):
    pid = _make_dataset(client, "Cvat Boxes")
    zf = _export_zip(client, pid, "cvat", include_segmentations=False)
    xml = zf.read("annotations.xml").decode()
    assert "<box" in xml and "<polygon" not in xml


def test_labelstudio_export(client):
    pid = _make_dataset(client, "Ls Fmt")
    zf = _export_zip(client, pid, "labelstudio")
    tasks = json.loads(zf.read("tasks.json"))
    assert len(tasks) == 1
    assert tasks[0]["data"]["image"].startswith("images/")
    assert tasks[0]["data"]["image"].endswith(".jpg")
    results = tasks[0]["annotations"][0]["result"]
    assert results, "no results emitted"
    assert results[0]["type"] in ("polygonlabels", "rectanglelabels")
    labels = results[0]["value"].get("polygonlabels") or results[0]["value"].get(
        "rectanglelabels"
    )
    assert labels == ["apple"]
    assert "labeling-config.xml" in zf.namelist()


def test_masks_export(client):
    pid = _make_dataset(client, "Masks Fmt")
    zf = _export_zip(client, pid, "masks")
    names = zf.namelist()
    mask_names = [n for n in names if n.startswith("masks/") and n.endswith(".png")]
    assert len(mask_names) == 1, names
    assert "labels.txt" in names
    assert "1 apple" in zf.read("labels.txt").decode()
    png = Image.open(io.BytesIO(zf.read(mask_names[0])))
    assert png.size == (64, 48)
    assert png.getextrema()[1] == 1  # class index of "apple" painted


def test_masks_requires_segmentations(client):
    pid = _make_dataset(client, "Masks Guard")
    r = client.get(
        f"/api/projects/{pid}/export",
        params={"format": "masks", "include_segmentations": False},
    )
    assert r.status_code == 400


def test_labelstudio_config_escapes_special_labels(client):
    pid = _make_dataset(client, "Escape Fmt")
    # Give the dataset a label with XML-hostile characters.
    r = client.post(
        f"/api/projects/{pid}/labels/rename",
        json={"old_label": "apple", "new_label": 'black & "white"'},
    )
    assert r.status_code == 200, r.text
    zf = _export_zip(client, pid, "labelstudio")
    import xml.etree.ElementTree as ET

    cfg = zf.read("labeling-config.xml").decode()
    root = ET.fromstring(cfg)  # raises on unescaped & or "
    values = [el.get("value") for el in root.iter("Label")]
    assert 'black & "white"' in values


def test_masks_export_covers_unannotated_images(client):
    pid = _make_dataset(client, "Masks All")
    # Second image with no annotations at all.
    r = client.post(
        f"/api/v2/projects/{pid}/imports/raw",
        files={"image": ("empty.jpg", _jpeg_bytes(), "image/jpeg")},
    )
    assert r.status_code == 200
    zf = _export_zip(client, pid, "masks")
    pngs = [n for n in zf.namelist() if n.endswith(".png")]
    assert len(pngs) == 2, pngs  # annotated and unannotated image


def test_labelstudio_results_match_the_config_controls(client):
    import xml.etree.ElementTree as ET

    pid = _make_dataset(client, "Ls Controls")
    for params, kinds in (({}, {"polygonlabels"}), ({"include_segmentations": False}, {"rectanglelabels"})):
        zf = _export_zip(client, pid, "labelstudio", **params)
        cfg = ET.fromstring(zf.read("labeling-config.xml"))
        controls = {el.get("name"): el.tag.lower() for el in cfg if el.tag != "Image"}
        results = [r for t in json.loads(zf.read("tasks.json")) for a in t["annotations"] for r in a["result"]]
        assert {r["type"] for r in results} == kinds
        for r in results:
            assert controls.get(r["from_name"]) == r["type"], (r["from_name"], controls)


def test_augmented_copies_without_boxes_are_in_every_export(client, app_module):
    import xml.etree.ElementTree as ET
    from pathlib import Path

    pid = _make_dataset(client, "Aug Backgrounds")
    r = client.post(
        f"/api/v2/projects/{pid}/imports/raw",
        files={"image": ("empty.jpg", _jpeg_bytes(), "image/jpeg")},
    )
    assert r.status_code == 200, r.text
    bg = r.json()["import_id"]
    # Uploads save their blurhash in the background; edit after those saves.
    for _ in range(250):
        m = app_module.load_manifest(pid)
        if all(e.get("blurhash") for e in m["imports"]):
            break
        time.sleep(0.02)
    entry = next(e for e in m["imports"] if e["id"] == bg)
    entry["n_augmentations"] = 2
    app_module.save_manifest(pid, m)
    out = app_module._augmentations_dir(pid, bg)
    out.mkdir(parents=True)
    for name in ("00.jpg", "01.jpg"):
        (out / name).write_bytes(_jpeg_bytes())
    (out / "annotations.json").write_text(
        json.dumps({"width": 64, "height": 48, "copies": {"00.jpg": [], "01.jpg": []}})
    )
    stem = Path(entry["filename"]).stem
    want = {f"aug_{stem}_00", f"aug_{stem}_01"}

    def stems(names):
        return {Path(n).stem for n in names}

    zf = _export_zip(client, pid, "coco")
    coco = [json.loads(zf.read(n)) for n in zf.namelist() if n.startswith("annotations/")]
    assert want <= stems(i["file_name"] for c in coco for i in c["images"])
    zf = _export_zip(client, pid, "voc")
    assert want <= stems(n for n in zf.namelist() if n.startswith("Annotations/"))
    zf = _export_zip(client, pid, "cvat")
    assert want <= stems(el.get("name") for el in ET.fromstring(zf.read("annotations.xml")).iter("image"))
    zf = _export_zip(client, pid, "labelstudio")
    assert want <= stems(t["data"]["image"] for t in json.loads(zf.read("tasks.json")))
    zf = _export_zip(client, pid, "masks")
    assert want <= stems(n for n in zf.namelist() if n.startswith("masks/"))
    zf = _export_zip(client, pid, "yolo")
    assert want <= stems(n for n in zf.namelist() if n.startswith("images/"))
