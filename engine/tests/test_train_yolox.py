"""YOLOX training: fast CPU units (no full training loop in CI)."""
import io
import json

import numpy as np
import pytest
import torch
from PIL import Image


def test_model_forward_and_loss_are_finite():
    from train_yolox import TrainConfig, _build_model, _make_batch, Sample
    import tempfile, pathlib

    model = _build_model(n_classes=2)
    model.train()
    with tempfile.TemporaryDirectory() as td:
        paths = []
        for i in range(2):
            p = pathlib.Path(td) / f"t{i}.png"
            Image.new("RGB", (96, 80), (200, 30 * i, 30)).save(p)
            paths.append(p)
        samples = [
            Sample(path=paths[0], boxes=np.array([[10, 10, 60, 50]], np.float32),
                   classes=np.array([0], np.float32), split="train", w=96, h=80),
            Sample(path=paths[1], boxes=np.array([[5, 5, 40, 40]], np.float32),
                   classes=np.array([1], np.float32), split="train", w=96, h=80),
        ]
        imgs, targets = _make_batch(samples, [0, 1], size=96, train=False)
    out = model(imgs, targets)
    loss = out["total_loss"]
    assert torch.isfinite(loss), out
    loss.backward()  # gradients flow


def test_ap50_perfect_and_empty():
    from train_yolox import ap50

    gt = [np.array([[10, 10, 50, 50, 0]], np.float32)]
    perfect = [np.array([[10, 10, 50, 50, 0.9, 0]], np.float32)]
    mean, per_class = ap50(perfect, gt, n_classes=1)
    assert mean == pytest.approx(1.0)

    none = [np.zeros((0, 6), np.float32)]
    mean, _ = ap50(none, gt, n_classes=1)
    assert mean == pytest.approx(0.0)


def test_letterbox_geometry():
    from train_yolox import letterbox

    img = np.zeros((60, 120, 3), np.uint8)
    out, scale, dx, dy = letterbox(img, 128)
    assert out.shape == (128, 128, 3)
    assert scale == pytest.approx(128 / 120)
    assert dx == 0 and dy == (128 - round(60 * scale)) // 2
    assert out[0, 0, 0] == 114  # padding colour


def _jpeg_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), (30, 60, 200)).save(buf, "JPEG")
    return buf.getvalue()


def test_models_endpoints(client):
    r = client.post(
        "/api/v2/projects", data={"name": "Train API", "labels": json.dumps(["x"])}
    )
    pid = r.json()["project_id"]

    r = client.get(f"/api/v2/projects/{pid}/models")
    assert r.status_code == 200 and r.json() == {"models": []}

    assert client.get(f"/api/v2/projects/{pid}/models/not-a-run/export").status_code == 400
    assert client.get(
        f"/api/v2/projects/{pid}/models/20990101T000000Z/export"
    ).status_code == 404
    assert client.get("/api/v2/projects/nope/models").status_code == 404

    # scheduling training on an unlabelled dataset is refused up front
    r = client.post(
        "/api/jobs", json={"kind": "train_yolox", "project": pid, "params": {}}
    )
    assert r.status_code == 400
    assert "labelled" in r.json()["detail"]
