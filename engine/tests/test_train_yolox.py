"""YOLOX training units, without a full training loop."""
import io
import json

import numpy as np
import pytest
import torch
from PIL import Image


def test_model_forward_and_loss_are_finite():
    from train_yolox import _build_model, _make_batch, Sample
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
    # Slightly off, as MinMax calibration is, but within tolerance.
    _toy_qdq_onnx(p, CONTRACT_SCALE * 1.001, CONTRACT_ZP)
    info = _verify_input_contract(p)
    assert info["scale"] == pytest.approx(CONTRACT_SCALE)
    assert info["zeroPoint"] == CONTRACT_ZP
    assert info["range"] == [0.0, 1.0]


def test_contract_refuses_a_symmetric_input(tmp_path):
    """A symmetric input is refused: it re-encodes to uint8 zp 128, so raw camera bytes read as noise."""
    from train_yolox import _verify_input_contract

    p = tmp_path / "sym.onnx"
    _toy_qdq_onnx(p, 1.0 / 127.0, 0)
    with pytest.raises(ValueError, match="wrong input range|contract needs"):
        _verify_input_contract(p)


def test_contract_refuses_255_range_calibration(tmp_path):
    """An input calibrated to [0,255] fails the export."""
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


# QAT and the export fold

def test_qat_switch_and_fixed_input_range():
    import torch
    from train_yolox import _build_model, CONTRACT_SCALE, CONTRACT_ZP
    from vendor.yolox.qat import set_qat

    m = _build_model(2)
    n = set_qat(m, True)
    assert n > 100  # one fake-quant point per tensor between ops
    scale, zp = m.fq_in.qparams()
    assert scale == pytest.approx(CONTRACT_SCALE) and zp == CONTRACT_ZP
    # A learned point clamps its range to include zero and is affine.
    fq = m.backbone.backbone.stem.fq
    fq.enabled = True
    fq.train()
    fq(torch.rand(1, 4, 8, 8) + 2.0)  # all positive: lo must still be <= 0
    s, z = fq.qparams()
    assert s > 0 and -128 <= z <= 127
    assert set_qat(m, False) == n


def test_qat_forward_backward_and_export_fold(tmp_path):
    """A QAT model trains, exports to QDQ, and folds to int8 weights, int32 biases and no Constant nodes."""
    import onnx
    import torch
    from train_yolox import (_build_model, _export_onnx, _fold_weight_quant,
                             _verify_input_contract)
    from vendor.yolox.qat import set_qat

    m = _build_model(2)
    set_qat(m, True)
    m.train()
    x = torch.rand(2, 3, 96, 96)
    t = torch.zeros(2, 100, 5)
    t[0, 0] = torch.tensor([1, 40.0, 40.0, 20.0, 20.0])
    with torch.no_grad():
        m(x, t)  # seed the observers
    loss = m(x, t)["total_loss"]
    assert torch.isfinite(loss)
    loss.backward()

    p = tmp_path / "qat_int8.onnx"
    _export_onnx(m, p, 96, qat=True)
    n_conv = _fold_weight_quant(p)
    g = onnx.load(str(p)).graph
    ops = [n.op_type for n in g.node]
    assert n_conv == ops.count("Conv") > 50
    assert "Constant" not in ops and "BatchNormalization" not in ops
    inits = {i.name: i for i in g.initializer}
    convs = [n for n in g.node if n.op_type == "Conv"]
    for c in convs:
        w_dq = next(n for n in g.node if n.output[0] == c.input[1])
        assert w_dq.op_type == "DequantizeLinear"
        assert inits[w_dq.input[0]].data_type == onnx.TensorProto.INT8
        if len(c.input) > 2:
            b_dq = next(n for n in g.node if n.output[0] == c.input[2])
            assert inits[b_dq.input[0]].data_type == onnx.TensorProto.INT32
    info = _verify_input_contract(p)
    assert info["calibrated"] == {"scale": pytest.approx(1 / 255), "zeroPoint": -128}
    # Every Q/DQ parameter is an initializer, as the compiler requires.
    for n in g.node:
        if n.op_type in ("QuantizeLinear", "DequantizeLinear"):
            assert all(i in inits for i in n.input[1:]), n.name


@pytest.mark.skipif(__import__("neuron6_verify").find_stedgeai() is None,
                    reason="ST Edge AI Core not installed")
def test_qat_export_maps_fully_to_the_npu(tmp_path):
    import torch
    import neuron6_verify as nv
    from train_yolox import _build_model, _export_onnx, _fold_weight_quant, _verify_input_contract
    from vendor.yolox.qat import set_qat

    m = _build_model(2)
    set_qat(m, True)
    m.train()
    with torch.no_grad():
        for _ in range(3):
            m(torch.rand(2, 3, 128, 128), torch.zeros(2, 100, 5))
    p = tmp_path / "qat_int8.onnx"
    _export_onnx(m, p, 128, qat=True)
    _fold_weight_quant(p)
    _verify_input_contract(p)
    info = nv.verify(p, tmp_path / "keep")
    assert info["verified"] is True
    assert info["epochsSoftware"] == 0, info
    assert info["epochsHardware"] > 100


def test_slug_is_a_valid_c_identifier():
    from train_yolox import _c_slug, _c_str

    assert _c_slug("3D objects") == "pk_3d_objects"
    assert _c_slug("int") == "pk_int"
    assert _c_slug("Bolts & Nuts!") == "bolts_nuts"
    assert _c_slug("") == "model"
    assert _c_str('say "hi"' + chr(92)) == 'say ' + chr(92) + '"hi' + chr(92) + '"' + chr(92) * 2
    # Arduino caps sketch names at 63 characters; the slug names the sketch.
    long = _c_slug("A very long dataset name about bolts, nuts, washers and every other fastener")
    assert len(long) <= 40 and not long.endswith("_")


def test_export_is_a_sketch_folder_that_runs_the_model(tmp_path):
    """export.zip unzips to <slug>/ holding a complete sketch that runs the model."""
    import zipfile

    from train_yolox import TrainConfig, _write_bundle

    onnx = tmp_path / "bolts_int8.onnx"
    onnx.write_bytes(b"onnx")
    report = {"run": "20260101T000000Z", "ap50_int8": 0.8123,
              "threshold": {"calibrated": True, "f1": 0.8, "precision": 0.9, "recall": 0.7}}
    _write_bundle(tmp_path, "bolts", "Bolts */ & nuts \u00e9", ["bolt", "nut"],
                  TrainConfig(imgsz=320), report, onnx, 0.42)

    names = sorted(zipfile.ZipFile(tmp_path / "export.zip").namelist())
    assert names == ["bolts/LICENSE.txt", "bolts/NOTICE.txt", "bolts/bolts.ino",
                     "bolts/bolts_int8.onnx", "bolts/bolts_labels.h", "bolts/report.json"]
    # A model trained from scratch is still a modified YOLOX network.
    notice = (tmp_path / "NOTICE.txt").read_text("utf-8")
    assert "modified version of YOLOX" in notice and "fine-tuned" not in notice
    licence = (tmp_path / "LICENSE.txt").read_text("utf-8")
    assert "Apache License" in licence and "Megvii Inc." in licence
    ino = (tmp_path / "bolts.ino").read_text("utf-8")
    assert '#pragma neuron6 model="bolts_int8.onnx" name=bolts' in ino
    assert '#include "bolts_labels.h"' in ino
    assert "OV5640 camera;" in ino
    assert "cfg.nn.width  = BOLTS_INPUT_W;" in ino and "cfg.nn.height = BOLTS_INPUT_H;" in ino
    # The board letterboxes the frame with grey 114 bars, as letterbox() does in training.
    assert "cfg.nn.aspect = ASPECT_LETTERBOX;" in ino and "ASPECT_CROP_CENTER" not in ino
    assert "PostProcess_PixelKitYOLOX(confidence, BOLTS_NMS_THRESHOLD," in ino
    # The threshold starts at the calibrated value and is a live slider.
    assert "static float confidence = BOLTS_CONF_THRESHOLD;" in ino
    assert 'n6_tune_float("confidence", &confidence, 0.01f, 0.99f);' in ino
    assert "if (confidence != applied) {" in ino
    assert "BOLTS_CLASSES, BOLTS_NUM_CLASSES);" in ino
    assert "Vision.begin(&NN_Instance_bolts, detector, cfg);" in ino
    assert "nullptr, cfg" not in ino and "Vision.begin(&NN_Instance_bolts, nullptr" not in ino
    assert "AP@0.5 0.812" in ino and "320x320" in ino
    # The dataset name cannot close the header comment, and the sketch is ASCII.
    header = ino.split("*/", 1)[0]
    assert "Bolts * / & nuts ?" in header
    ino.encode("ascii")
    labels_h = (tmp_path / "bolts_labels.h").read_text("utf-8")
    assert "#define BOLTS_INPUT_W 320" in labels_h and "#define BOLTS_CONF_THRESHOLD 0.42f" in labels_h
    assert "/* Generated by PixelKit: YOLOX-Nano trained on the Bolts * / & nuts ? dataset. */" in labels_h
    # An older core, or another board, stops with a message, not a missing name.
    assert "#ifndef NEURON6_CORE_VERSION_AT_LEAST" in ino
    assert "#elif !NEURON6_CORE_VERSION_AT_LEAST(0, 1, 3)" in ino
    assert ino.count('#error "This sketch needs the Neuro N6 core 0.1.3 or later"') == 2
    assert ino.index("NEURON6_CORE_VERSION_AT_LEAST") < ino.index("#include <PostProcess.h>")
    # USB by default; STREAM_OVER_WIFI 1 adds the radio library, an access point and the WiFi transport.
    assert "#define STREAM_OVER_WIFI 0" in ino
    assert "#if STREAM_OVER_WIFI\n#include <MayaW476.h>\n#endif" in ino
    assert ino.index("#define STREAM_OVER_WIFI") < ino.index("#if STREAM_OVER_WIFI")
    assert "MayaW476.beginAP(MayaW476.getHostname(), MayaW476.devicePassword());" in ino
    assert "cfg.transport = TRANSPORT_WIFI_SOFTAP;\n#else" in ino
    assert "cfg.transport = TRANSPORT_USB;\n#endif" in ino
    assert "Serial.println(MayaW476.devicePassword());" in ino
    assert ino.count("#if STREAM_OVER_WIFI") == 3 and ino.count("#endif") == 4
    # LF on every OS: the core checks the files it takes in.
    for f in ("bolts.ino", "bolts_labels.h", "report.json"):
        assert b"\r" not in (tmp_path / f).read_bytes(), f

    # Weights fine-tuned from Megvii's checkpoint go out with its notice.
    _write_bundle(tmp_path, "bolts", "Bolts", ["bolt", "nut"], TrainConfig(imgsz=320),
                  {**report, "pretrained": True}, onnx, 0.42)
    zf = zipfile.ZipFile(tmp_path / "export.zip")
    assert "bolts/NOTICE.txt" in zf.namelist()
    notice = zf.read("bolts/NOTICE.txt").decode("utf-8")
    assert "bolts_int8.onnx" in notice and "Megvii" in notice and "Apache License" in notice
    assert "fine-tuned from Megvii's COCO-pretrained YOLOX-Nano checkpoint" in notice


def test_ap50_per_class_is_positional_when_a_class_has_no_val_gt():
    import numpy as np
    from train_yolox import ap50

    # class 0 never appears in GT; class 1 perfect; class 2 missed.
    gt = [np.array([[10, 10, 50, 50, 1], [60, 60, 90, 90, 2]], np.float32)]
    pr = [np.array([[10, 10, 50, 50, 0.9, 1]], np.float32)]
    mean, per = ap50(pr, gt, n_classes=3)
    assert per[0] is None and per[1] == pytest.approx(1.0) and per[2] == pytest.approx(0.0)
    assert mean == pytest.approx(0.5)


def test_int32_bias_fold_saturates_instead_of_wrapping(tmp_path):
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper as nh
    from train_yolox import _fold_weight_quant

    # input -> Q/DQ -> Conv(w float, b float huge) -> output
    w = np.ones((1, 1, 1, 1), np.float32)
    b = np.array([5e9], np.float32)  # far beyond int32 after / (x_scale * w_scale)
    init = [nh.from_array(np.float32(0.02), "xs"), nh.from_array(np.int8(0), "xz"),
            nh.from_array(w, "W"), nh.from_array(b, "B")]
    nodes = [helper.make_node("QuantizeLinear", ["images", "xs", "xz"], ["q"]),
             helper.make_node("DequantizeLinear", ["q", "xs", "xz"], ["dq"]),
             helper.make_node("Conv", ["dq", "W", "B"], ["y"], name="/c/Conv")]
    g = helper.make_graph(nodes, "t", [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 1, 4, 4])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 4, 4])], initializer=init)
    p = tmp_path / "t.onnx"
    onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)]), str(p))
    assert _fold_weight_quant(p) == 1
    g2 = onnx.load(str(p)).graph
    inits = {i.name: nh.to_array(i) for i in g2.initializer}
    bq = next(v for k, v in inits.items() if k.endswith("B_q"))
    assert bq.dtype == np.int32 and bq[0] == 2**31 - 1  # saturated high, not INT_MIN


def test_board_view_crops_the_centre_square_and_drops_cut_boxes():
    """_center_square moves boxes into the centre square and drops one mostly cut off."""
    import random

    import numpy as np
    from train_yolox import _augment, _center_square

    img = np.zeros((480, 640, 3), np.uint8)
    boxes = np.array([[300, 200, 340, 240],    # in the square
                      [10, 10, 60, 60],        # left of it
                      [40, 300, 100, 360],     # a third in
                      [60, 300, 120, 360]],    # two thirds in
                     np.float32)
    crop, kept, cls = _center_square(img, boxes, np.array([0, 1, 1, 0]))
    assert crop.shape == (480, 480, 3)
    assert kept.tolist() == [[220, 200, 260, 240], [0, 300, 40, 360]] and cls.tolist() == [0, 0]

    # Background images, without boxes, go through every augmentation too.
    for seed in range(50):
        random.seed(seed)
        out, b, c = _augment(img, np.zeros((0, 4), np.float32), np.zeros((0,), np.int64))
        assert b.shape == (0, 4) and len(c) == 0 and out.ndim == 3


def test_class_names_with_control_characters_stay_inside_their_c_strings(tmp_path):
    """A label with a CR or another control character still gives a labels header that compiles."""
    import codecs

    from train_yolox import TrainConfig, _c_str, _write_bundle

    names = ["hex\rnut", "line\nbreak", "tab\tbed", "bell\x07", "a\x017", "del\x7f", "nel\x85",
             "caf" + chr(0xE9), 'say "hi"' + chr(92)]
    for n in names:
        lit = _c_str(n)
        assert lit.isprintable(), lit
        # C reads \ooo, \" and \\ as Python's escape decoder does.
        assert codecs.escape_decode(lit.encode("utf-8"))[0] == n.encode("utf-8"), lit
    # An octal escape takes at most three digits, so the 7 after it stays a 7.
    assert _c_str("a\x017") == "a" + chr(92) + "0017"

    onnx = tmp_path / "pk_int8.onnx"
    onnx.write_bytes(b"onnx")
    _write_bundle(tmp_path, "pk", "Labels", names, TrainConfig(), {"run": "20260101T000000Z"}, onnx, 0.5)
    lines = (tmp_path / "pk_labels.h").read_text("utf-8").split("\n")
    first = lines.index(f"static const char* const PK_CLASSES[{len(names)}] = {{") + 1
    assert lines[first:first + len(names) + 1] == [f'  "{_c_str(n)}",' for n in names] + ["};"]


def test_a_failure_while_zipping_leaves_no_report_and_no_export(tmp_path, monkeypatch):
    """report.json, which lists a run as done, is written once export.zip is complete."""
    import zipfile

    from train_yolox import TrainConfig, _write_bundle

    onnx = tmp_path / "pk_int8.onnx"
    onnx.write_bytes(b"model bytes")
    report = {"run": "20260101T000000Z", "threshold": {"calibrated": False}}
    real_write = zipfile.ZipFile.write

    def locked(self, filename, *a, **k):
        if str(filename).endswith(".onnx"):
            raise PermissionError(13, "The process cannot access the file")
        return real_write(self, filename, *a, **k)

    monkeypatch.setattr(zipfile.ZipFile, "write", locked)
    with pytest.raises(PermissionError):
        _write_bundle(tmp_path, "pk", "Bundle", ["bolt"], TrainConfig(), report, onnx, 0.5)
    assert not (tmp_path / "report.json").exists()
    assert not (tmp_path / "export.zip").exists() and not list(tmp_path.glob("*.tmp"))

    monkeypatch.setattr(zipfile.ZipFile, "write", real_write)
    _write_bundle(tmp_path, "pk", "Bundle", ["bolt"], TrainConfig(), report, onnx, 0.5)
    zf = zipfile.ZipFile(tmp_path / "export.zip")
    assert zf.read("pk/report.json") == (tmp_path / "report.json").read_bytes()
    assert json.loads((tmp_path / "report.json").read_text("utf-8")) == report


# Warm starts: Megvii's COCO checkpoint and earlier runs

class _Response(io.BytesIO):
    """An HTTP response for urlopen: a body and its Content-Length."""

    def __init__(self, body: bytes, length: int | None = None):
        super().__init__(body)
        self.headers = {"Content-Length": str(len(body) if length is None else length)}


class _NotWeights:
    """Pickled into a checkpoint, it needs code to load."""


def _focus_checkpoint(path):
    """A stand-in for yolox_nano.pth: a Focus stem with random weights and BN
    statistics under Megvii's key names. Returns the Focus stem."""
    from vendor.yolox.network_blocks import Focus

    g = torch.Generator().manual_seed(0)
    focus = Focus(3, 16, ksize=3)
    bn = focus.conv.bn
    with torch.no_grad():
        focus.conv.conv.weight.copy_(torch.randn(focus.conv.conv.weight.shape, generator=g) * 0.05)
        bn.weight.uniform_(0.5, 1.5, generator=g)
        bn.bias.normal_(0.0, 0.1, generator=g)
        bn.running_mean.normal_(0.0, 1.0, generator=g)
        bn.running_var.uniform_(0.5, 2.0, generator=g)
    bn.eps = 1e-3  # as _build_model sets it
    state = {f"backbone.backbone.stem.{k}": v for k, v in focus.state_dict().items() if ".fq" not in k}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": state, "start_epoch": 300}, path)
    return focus.eval()


@pytest.fixture
def coco_cache(tmp_path, monkeypatch):
    """The COCO checkpoint cache in tmp_path. urlopen returns
    served["response"]() and counts its calls."""
    import urllib.request

    import workspace

    monkeypatch.setattr(workspace, "weights_dir", lambda: tmp_path)
    served = {"calls": 0, "response": None}

    def urlopen(url, timeout=None):
        served["calls"] += 1
        return served["response"]()

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    return tmp_path / "yolox", served


def test_the_coco_stem_sees_the_channel_order_it_was_trained_on(coco_cache, monkeypatch):
    """The remapped 6x6 stem responds to an RGB [0,1] picture as Megvii's Focus stem does to the BGR [0,255] one."""
    import hashlib

    import train_yolox as ty

    cache, _served = coco_cache
    ckpt = cache / "yolox_nano.pth"
    focus = _focus_checkpoint(ckpt)
    monkeypatch.setitem(ty.MODEL_SPECS["nano"], "sha256", hashlib.sha256(ckpt.read_bytes()).hexdigest())
    model = ty._build_model(2)
    notes = []
    assert ty._load_pretrained(model, notes=notes) is True and notes == []
    stem = model.eval().backbone.backbone.stem
    rgb = torch.randint(0, 256, (1, 3, 64, 64), generator=torch.Generator().manual_seed(1)).float()
    with torch.no_grad():
        got = stem(rgb / 255.0)
        assert torch.allclose(got, focus(rgb.flip(1)), rtol=1e-4, atol=1e-4)
        # Fed RGB, the Focus stem answers differently: the channel order matters.
        assert not torch.allclose(got, focus(rgb), rtol=1e-2, atol=1e-2)


def test_a_download_cut_short_is_not_cached_and_the_next_run_fetches_again(coco_cache):
    import train_yolox as ty

    cache, served = coco_cache
    # Content-Length says 7 MB; the server closes after 1 MB, and read() returns b"".
    served["response"] = lambda: _Response(b"\x80\x02" + b"x" * (1 << 20), length=7 << 20)
    notes = []
    assert ty._load_pretrained(ty._build_model(2), notes=notes) is False
    assert notes == [f"the COCO checkpoint download stopped at {(1 << 20) + 2} of {7 << 20} bytes: "
                     "trained from scratch"]
    assert list(cache.iterdir()) == []  # nothing cached and no temp file left
    assert ty._load_pretrained(ty._build_model(2)) is False and served["calls"] == 2

    # A complete body that is not the release file is not kept either.
    served["response"] = lambda: _Response(b"\x80\x02" + b"x" * 1000)
    notes = []
    assert ty._load_pretrained(ty._build_model(2), notes=notes) is False
    assert "failed its checksum" in notes[0] and list(cache.iterdir()) == []


def test_a_complete_download_is_checked_cached_and_loaded(coco_cache, monkeypatch, tmp_path_factory):
    """Two runs downloading at once each use their own temp file; the cached file is the release's."""
    import hashlib
    import threading

    import train_yolox as ty

    cache, served = coco_cache
    src = tmp_path_factory.mktemp("release") / "yolox_nano.pth"
    _focus_checkpoint(src)
    body = src.read_bytes()
    monkeypatch.setitem(ty.MODEL_SPECS["nano"], "sha256", hashlib.sha256(body).hexdigest())
    both_open = threading.Barrier(2)

    class Slow(_Response):
        def read(self, n=-1):
            chunk = super().read(n)
            if chunk:
                both_open.wait(timeout=30)  # each run holds its temp file open here
            return chunk

    served["response"] = lambda: Slow(body)
    results = []
    runs = [threading.Thread(target=lambda: results.append(ty._load_pretrained(ty._build_model(2))))
            for _ in range(2)]
    for t in runs:
        t.start()
    for t in runs:
        t.join(timeout=60)
    assert results == [True, True] and served["calls"] == 2
    assert [p.name for p in cache.iterdir()] == ["yolox_nano.pth"]
    assert (cache / "yolox_nano.pth").read_bytes() == body
    # The cached file is used from then on.
    assert ty._load_pretrained(ty._build_model(2)) is True and served["calls"] == 2


def test_a_cached_checkpoint_that_fails_its_checksum_or_its_load_is_deleted(coco_cache, monkeypatch):
    import train_yolox as ty

    cache, served = coco_cache

    def offline():
        raise OSError("offline")

    served["response"] = offline
    cached = cache / "yolox_nano.pth"
    cache.mkdir()
    cached.write_bytes(b"not a checkpoint")
    notes = []
    assert ty._load_pretrained(ty._build_model(2), notes=notes) is False
    assert notes == ["the cached COCO checkpoint failed its checksum and was deleted: trained from scratch"]
    assert not cached.exists()
    # The next run tries to download it again.
    notes = []
    assert ty._load_pretrained(ty._build_model(2), notes=notes) is False and served["calls"] == 1
    assert notes == ["COCO checkpoint unavailable (offline?): trained from scratch"]

    # With no hash to check, a file torch cannot load as plain weights is deleted.
    monkeypatch.setitem(ty.MODEL_SPECS["nano"], "sha256", None)
    for content in (b"not a checkpoint", None):
        if content is None:
            torch.save({"model": {}, "extra": _NotWeights()}, cached)
        else:
            cached.write_bytes(content)
        notes = []
        assert ty._load_pretrained(ty._build_model(2), notes=notes) is False
        assert "could not be read and was deleted" in notes[0] and not cached.exists()


def test_a_warm_start_from_a_run_with_other_labels_matches_class_layers_by_name(tmp_path):
    from train_yolox import _build_model, _load_run_weights

    def run_with(classes):
        m = _build_model(len(classes))
        with torch.no_grad():
            for conv in m.head.cls_preds:
                for i in range(len(classes)):
                    conv.weight[i].fill_(float(i + 1))
                    conv.bias[i] = 10.0 * (i + 1)
        d = tmp_path / "yolox_nano" / "-".join(classes)
        d.mkdir(parents=True)
        torch.save({"model": m.state_dict(), "classes": classes}, d / "best.pt")
        return m, d

    old, d = run_with(["cat", "dog", "bird"])
    new = _build_model(2)
    fresh = [conv.bias.detach().clone() for conv in new.head.cls_preds]
    info = _load_run_weights(new, d, ["dog", "fox"])
    assert info["classes"] == ["cat", "dog", "bird"] and info["reusedClasses"] == ["dog"]
    for a, b, f in zip(old.head.cls_preds, new.head.cls_preds, fresh):
        assert torch.equal(b.weight[0], a.weight[1]) and b.bias[0] == a.bias[1]  # dog moved over
        assert b.bias[1] == f[1]                                                 # fox starts fresh
    # Layers that do not depend on the labels come over as before.
    assert torch.equal(new.head.reg_preds[0].weight, old.head.reg_preds[0].weight)

    # The same labels in another order: each channel follows its label.
    old2, d2 = run_with(["cat", "dog"])
    new2 = _build_model(2)
    assert _load_run_weights(new2, d2, ["dog", "cat"])["reusedClasses"] == ["dog", "cat"]
    for a, b in zip(old2.head.cls_preds, new2.head.cls_preds):
        assert torch.equal(b.weight[0], a.weight[1]) and torch.equal(b.weight[1], a.weight[0])
        assert b.bias.tolist() == [20.0, 10.0]
    # The same labels in the same order load unchanged.
    new3 = _build_model(2)
    _load_run_weights(new3, d2, ["cat", "dog"])
    assert all(torch.equal(a.weight, b.weight) and torch.equal(a.bias, b.bias)
               for a, b in zip(old2.head.cls_preds, new3.head.cls_preds))
