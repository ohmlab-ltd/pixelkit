"""Background images: checked, with nothing in them to label. Marked in the
viewer or declared by an imported dataset, they count as labelled, the
labelling passes leave them alone, and training learns from them as
negatives. CPU only, 64 px inputs."""
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image


def _jpeg(colour=(200, 30, 30)) -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", (96, 64), colour)
    for x in range(10, 40):
        for y in range(10, 40):
            img.putpixel((x, y), (250, 250, 250))
    img.save(buf, "JPEG")
    return buf.getvalue()


def _dataset(client, name, n_boxed=6, n_background=3):
    pid = client.post("/api/v2/projects",
                      data={"name": name, "labels": json.dumps(["dog"])}).json()["project_id"]
    ids = []
    for i in range(n_boxed + n_background):
        r = client.post(f"/api/v2/projects/{pid}/imports/raw",
                        files={"image": (f"img{i:02d}.jpg", _jpeg((200, 20 * i, 30)), "image/jpeg")})
        iid = r.json()["import_id"]
        ids.append(iid)
        if i < n_boxed:
            body = {"editedBoxes": [{"id": f"b{i}", "label": "dog", "x0": 10, "y0": 10, "x1": 40, "y1": 40}]}
        else:
            body = {"background": True}
        assert client.put(f"/api/v2/projects/{pid}/imports/{iid}", json=body).status_code == 200
    return pid, ids


def _entry(app_module, pid, iid):
    return next(e for e in app_module.load_manifest(pid)["imports"] if e["id"] == iid)


def test_marking_unmarking_and_boxes_clear_the_mark(client, app_module):
    pid, ids = _dataset(client, "Mark background", n_boxed=1, n_background=0)
    iid = ids[0]
    # Marking drops the boxes: a background image carries none.
    assert client.put(f"/api/v2/projects/{pid}/imports/{iid}", json={"background": True}).status_code == 200
    e = _entry(app_module, pid, iid)
    assert e["background"] is True and e["editedBoxes"] == [] and e["editedBoxesSet"] is True
    assert app_module._import_is_labelled(e) and not app_module._import_has_labels(e)
    # An unrelated empty save keeps the mark; boxes clear it.
    client.put(f"/api/v2/projects/{pid}/imports/{iid}", json={"editedBoxes": []})
    assert _entry(app_module, pid, iid).get("background") is True
    box = {"id": "x", "label": "dog", "x0": 1, "y0": 1, "x1": 20, "y1": 20}
    client.put(f"/api/v2/projects/{pid}/imports/{iid}", json={"editedBoxes": [box], "background": False})
    e = _entry(app_module, pid, iid)
    assert "background" not in e and len(e["editedBoxes"]) == 1
    client.put(f"/api/v2/projects/{pid}/imports/{iid}", json={"background": True})
    client.put(f"/api/v2/projects/{pid}/imports/{iid}", json={"editedBoxes": [box]})
    assert "background" not in _entry(app_module, pid, iid)
    # Unmarked without boxes: plain unlabelled again.
    client.put(f"/api/v2/projects/{pid}/imports/{iid}", json={"background": True})
    client.put(f"/api/v2/projects/{pid}/imports/{iid}", json={"background": False})
    e = _entry(app_module, pid, iid)
    assert "background" not in e and not app_module._import_is_labelled(e)


def test_background_images_count_as_labelled_and_labelling_passes_skip_them(client, app_module):
    pid, ids = _dataset(client, "Counts", n_boxed=1, n_background=2)
    ov = client.get(f"/api/v2/projects/{pid}/overview").json()
    flags = {i["id"]: i["background"] for i in ov["imports"]}
    assert flags == {ids[0]: False, ids[1]: True, ids[2]: True}
    assert ov["filter_counts"]["unlabelled"] == 0 and ov["filter_counts"]["unrated"] == 1
    card = app_module._build_workspace_card_payload(pid, app_module.load_manifest(pid))
    assert card["n_labelled"] == 3 and card["n_unlabelled"] == 0
    # Nothing left to label, not even on a relabel of everything but them.
    r = client.post("/api/jobs", json={"project": pid, "kind": "label_charlie", "params": {}})
    assert r.status_code == 400 and "no unlabelled" in r.text
    m = app_module.load_manifest(pid)
    assert [e["id"] for e in app_module._label_model_targets(m, True)] == []
    # A full reset puts them back in the queue.
    assert client.post(f"/api/v2/projects/{pid}/clear_all_annotations").status_code == 200
    assert not any(e.get("background") for e in app_module.load_manifest(pid)["imports"])


def test_an_imported_annotation_listing_no_objects_is_background(client, app_module):
    pid = client.post("/api/v2/projects",
                      data={"name": "Import bg", "labels": json.dumps(["dog"])}).json()["project_id"]
    boxes = [
        json.dumps([]),                                                     # declared empty
        json.dumps([{"label": "dog", "x0": 5, "y0": 5, "x1": 30, "y1": 30}]),
        json.dumps([{"label": "dog", "x0": 500, "y0": 500, "x1": 600, "y1": 600}]),  # all unusable
    ]
    files = [("images", (f"i{i}.jpg", _jpeg(), "image/jpeg")) for i in range(3)]
    r = client.post(f"/api/v2/projects/{pid}/imports/raw_batch", files=files,
                    data={"boxes": boxes, "created_at_ms": ["1", "2", "3"]})
    assert r.status_code == 200, r.text
    by_name = {e["originalFilename"]: e for e in app_module.load_manifest(pid)["imports"]}
    assert by_name["i0.jpg"].get("background") is True
    assert "background" not in by_name["i1.jpg"]
    assert "background" not in by_name["i2.jpg"]  # its boxes were bad, not absent


def test_training_learns_from_background_images(client):
    import store
    from train_yolox import TrainConfig, load_samples, preflight, train

    pid, ids = _dataset(client, "Train bg", n_boxed=6, n_background=4)
    samples, classes, info = load_samples(pid, 0.75)
    assert classes == ["dog"] and info["originals"] == 10 and info["background"] == 4
    assert info["unlabelled"] == 0
    bg = [s for s in samples if not len(s.boxes)]
    assert len(bg) == 4 and all(s.boxes.shape == (0, 4) for s in bg)
    # Split as a class of their own: val also measures the empty pictures.
    assert {s.split for s in bg} == {"train", "val"}
    out = preflight(pid, TrainConfig(imgsz=64, train_split=0.75))
    assert out["ok"] is True and out["images"]["background"] == 4

    # Augmented copies of a background image stay, without boxes.
    m = store.load(pid)
    parent = next(e for e in m["imports"] if e["id"] == ids[-1])
    d = store.dataset_dir(pid) / "augmentations" / parent["id"]
    d.mkdir(parents=True)
    Image.new("RGB", (96, 64), (9, 9, 9)).save(d / "000.jpg", "JPEG")
    (d / "annotations.json").write_text(json.dumps({"width": 96, "height": 64, "copies": {"000.jpg": []}}))
    parent["n_augmentations"] = 1
    store.save(pid, m)
    samples, _, info = load_samples(pid, 0.75)
    parent_split = next(s.split for s in samples if s.import_id == parent["id"] and s.source == "original")
    aug = [s for s in samples if s.source == "augmented"]
    assert len(aug) == (1 if parent_split == "train" else 0)
    assert all(s.boxes.shape == (0, 4) for s in aug)

    report = train(pid, TrainConfig(epochs=1, qat_epochs=0, imgsz=64, batch=4, device="cpu",
                                     pretrained=False, verify_neuron6=False, failure_limit=4,
                                     run_id="20990102T000000Z"))
    assert report["images"]["background"] == 4
    assert len(report["confusion"]) == 2


def test_background_images_alone_teach_nothing():
    from train_yolox import Sample, TrainConfig, preflight_samples

    def mk(name, split):
        return Sample(path=Path(name), boxes=np.zeros((0, 4), np.float32),
                      classes=np.zeros(0, np.float32), split=split, w=96, h=64, name=name)

    out = preflight_samples([mk("a.jpg", "train"), mk("b.jpg", "train"), mk("c.jpg", "val")], ["dog"],
                            {"originals": 3, "augmented": 0, "unlabelled": 0, "background": 3},
                            TrainConfig(imgsz=64))
    assert out["ok"] is False and "background images alone" in out["blockers"][0]
