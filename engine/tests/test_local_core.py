"""Golden path over the portable storage schema — no models needed.

create project (container) -> create dataset -> upload image -> edit
annotations -> read back (annotations + viewport) -> export -> delete.
Asserts the on-disk shape: per-project folders, dataset.json + annotations/
split, originals alone in images/.
"""
import io
import json
import zipfile

from PIL import Image


def _jpeg_bytes(w=64, h=48, color=(200, 30, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, "JPEG")
    return buf.getvalue()


def test_golden_path(client, workspace_dir):
    # -- container (a "Project" in the UI) -------------------------------
    r = client.post("/api/containers", json={"name": "Fruit Sorting", "private": True})
    assert r.status_code == 200, r.text
    container_id = r.json()["id"]
    proj_folder = workspace_dir / "projects" / "fruit-sorting"
    assert (proj_folder / "project.json").is_file()

    # -- dataset ---------------------------------------------------------
    r = client.post(
        "/api/v2/projects",
        data={"name": "Apples v1", "labels": json.dumps(["apple", "leaf"])},
    )
    assert r.status_code == 200, r.text
    pid = r.json()["project_id"]

    # associate with the container, then the folder moves inside it
    r = client.post(f"/api/containers/{container_id}/datasets/{pid}")
    assert r.status_code == 200, r.text

    # -- upload ----------------------------------------------------------
    r = client.post(
        f"/api/v2/projects/{pid}/imports/raw",
        files={"image": ("apple1.jpg", _jpeg_bytes(), "image/jpeg")},
    )
    assert r.status_code == 200, r.text
    import_id = r.json()["import_id"]

    r = client.get(f"/api/v2/projects/{pid}/initial")
    assert r.status_code == 200, r.text

    # -- on-disk shape ---------------------------------------------------
    ds_dirs = [d for d in workspace_dir.glob("projects/*/*/dataset.json")]
    assert len(ds_dirs) == 1, ds_dirs
    ds = ds_dirs[0].parent
    images = list((ds / "images").iterdir())
    assert len(images) == 1 and images[0].suffix in (".jpg", ".jpeg", ".png"), images
    assert not (ds / "manifest.json").exists()

    # -- edit annotations ------------------------------------------------
    boxes = [{"id": "b1", "label": "apple", "x0": 4, "y0": 4, "x1": 40, "y1": 40}]
    r = client.put(
        f"/api/v2/projects/{pid}/imports/{import_id}",
        json={"editedBoxes": boxes},
    )
    assert r.status_code == 200, r.text

    ann_files = list((ds / "annotations").glob("*.json"))
    assert len(ann_files) == 1, ann_files
    ann = json.loads(ann_files[0].read_text())
    assert ann["editedBoxes"][0]["label"] == "apple"
    # geometry must NOT be duplicated in dataset.json
    dsj = json.loads((ds / "dataset.json").read_text())
    (imp,) = dsj["imports"]
    assert "editedBoxes" not in imp and "detections" not in imp
    assert imp["editedBoxesSet"] is True

    # -- read back -------------------------------------------------------
    r = client.get(f"/api/v2/projects/{pid}/annotations/{import_id}")
    assert r.status_code == 200, r.text
    assert r.json()["editedBoxes"][0]["label"] == "apple"

    r = client.get(f"/api/v3/projects/{pid}/viewport", params={"ids": import_id})
    assert r.status_code == 200, r.text

    # -- export ----------------------------------------------------------
    r = client.get(
        f"/api/projects/{pid}/export",
        params={"format": "yolo", "include_segmentations": False},
    )
    assert r.status_code == 200, r.text
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    names = zf.namelist()
    assert any(n.startswith("labels/") and n.endswith(".txt") for n in names), names
    assert any(n == "data.yaml" for n in names), names

    # -- delete ----------------------------------------------------------
    r = client.delete(f"/api/projects/{pid}")
    assert r.status_code == 200, r.text
    assert not ds.exists()


def test_labelling_endpoints_503_without_models(client):
    r = client.post("/api/charlie/imports/process", data={"labels": "[]"})
    # 422 = request-shape validation firing first; 503 = model-not-loaded.
    # Either way: no crash, no partial work.
    assert r.status_code in (400, 422, 503)
