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


def _toy_qdq_onnx(path, scale, zp, zp_dtype="int8"):
    """Minimal graph: input -> Q -> DQ -> output, with chosen input params."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper as nh

    np_zp = np.int8(zp) if zp_dtype == "int8" else np.uint8(zp)
    init = [nh.from_array(np.float32(scale), "s"), nh.from_array(np_zp, "z")]
    nodes = [helper.make_node("QuantizeLinear", ["images", "s", "z"], ["q"]),
             helper.make_node("DequantizeLinear", ["q", "s", "z"], ["y"])]
    g = helper.make_graph(
        nodes, "toy",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 8, 8])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 8, 8])],
        initializer=init)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(m, str(path))


def test_contract_accepts_and_pins_a_conforming_input(tmp_path):
    from train_yolox import CONTRACT_SCALE, CONTRACT_ZP, _verify_input_contract

    p = tmp_path / "ok.onnx"
    # Slightly off (as real MinMax calibration would be), but within tolerance.
    _toy_qdq_onnx(p, CONTRACT_SCALE * 1.001, CONTRACT_ZP)
    info = _verify_input_contract(p)
    assert info["scale"] == pytest.approx(CONTRACT_SCALE)
    assert info["zeroPoint"] == CONTRACT_ZP
    assert info["range"] == [0.0, 1.0]


def test_contract_refuses_a_symmetric_input(tmp_path):
    """The failure this whole design exists to prevent: a symmetric input
    re-encodes to uint8 zp 128, so raw camera bytes read as noise."""
    from train_yolox import _verify_input_contract

    p = tmp_path / "sym.onnx"
    _toy_qdq_onnx(p, 1.0 / 127.0, 0)
    with pytest.raises(ValueError, match="wrong input range|contract needs"):
        _verify_input_contract(p)


def test_contract_refuses_255_range_calibration(tmp_path):
    """Preprocessing drifting back to [0,255] must fail the export."""
    from train_yolox import _verify_input_contract

    p = tmp_path / "wide.onnx"
    _toy_qdq_onnx(p, 1.0, -128)
    with pytest.raises(ValueError):
        _verify_input_contract(p)


def test_decode_raw_matches_hand_computation():
    """Row 0 is grid (0,0) at stride 8; row 1024 starts the stride-16 level."""
    import numpy as np
    from train_yolox import _decode_raw

    raw = np.zeros((1, 1344, 6), np.float32)
    raw[0, 0, :4] = [0.5, 0.25, 0.0, 0.0]        # stride 8, grid (0,0)
    raw[0, 1024, :4] = [0.0, 0.0, 1.0, 0.0]      # stride 16, grid (0,0)
    d = _decode_raw(raw, 256)
    assert d[0, 0, 0] == pytest.approx(4.0)      # (0.5 + 0) * 8
    assert d[0, 0, 1] == pytest.approx(2.0)      # (0.25 + 0) * 8
    assert d[0, 0, 2] == pytest.approx(8.0)      # exp(0) * 8
    assert d[0, 1024, 2] == pytest.approx(16 * np.e, rel=1e-5)
    # Last stride-8 row is grid (31,31).
    assert d[0, 1023, 0] == pytest.approx(31 * 8)
