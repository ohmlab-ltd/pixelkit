"""ST Edge AI verification: report parsing, tool discovery, and (when the
compiler is installed locally) a real compile of a contract-conforming
export."""
import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

import neuron6_verify as nv

REPORT = """\
ST Edge AI Core v4.0.1-20581 7ed50de05
Epochs details
Total number of epochs                               122
>> pure software (SW) epochs                           0
>> hybrid epochs (using both software and hardware)    3
>> pure hardware (HW or EC) epochs                   119  (implemented in 4 epoch controller blobs/meta epochs)
| epoch ID           |  Type  |   Details    |
| epochs    2 -> 2   |   EC   |              |
| epoch_3            |   SW   |    Slice     |
| epoch_4            |   SW   |    Slice     |
model: macc=271,599,330 weights=3,556,232 activations=-- io=--
Memory usage information  (input/output buffers are included in activations)
\tnpuRAM     [0x34200000 - 0x343C0000]:    448.000 kB /      1.750 MB  ( 25.00 % used) -- weights:          0  B (  0.00 % used)  activations:    448.000 kB ( 25.00 % used)
\tPSRAM      [0x70C00000 - 0x71C00000]:          0  B /     16.000 MB  (  0.00 % used) -- weights:          0  B (  0.00 % used)  activations:          0  B (  0.00 % used)
\tOCTAFLASH  [0x90300000 - 0x91D80000]:    898.188 kB /     26.500 MB  (  3.31 % used) -- weights:    898.188 kB (  3.31 % used)  activations:          0  B (  0.00 % used)
Total:                                             1.346 MB                                  -- weights:    898.188 kB                  activations:    448.000 kB
"""


def test_parse_generate_report_extracts_the_numbers():
    r = nv.parse_generate_report(REPORT)
    assert r["epochsTotal"] == 122
    assert r["epochsSoftware"] == 0
    assert r["epochsHybrid"] == 3
    assert r["epochsHardware"] == 119
    assert r["macc"] == 271_599_330
    assert r["weightsBytes"] == round(898.188 * 1024)
    assert r["poolBytes"] == round(26.5 * 1024 * 1024)
    assert r["activationsBytes"] == 448 * 1024
    assert r["npuRamBytes"] == round(1.75 * 1024 * 1024)
    assert r["softwareOps"] == {"Slice": 2}


def test_parse_generate_report_tolerates_missing_sections():
    assert nv.parse_generate_report("nothing useful here") == {}


def test_weights_pool_comes_from_the_memory_map(tmp_path):
    # 0x90300000..0x91C80000: the pool ends where OTA slot B begins.
    assert nv.weights_pool_bytes() == 26112 * 1024
    assert nv.weights_pool_bytes(tmp_path / "missing.mpool") is None


def _installed_core() -> Path | None:
    """A Neuro N6 core on this machine: NEURON6_CORE, the sketchbook's
    hardware folder, or a Boards Manager install."""
    home = Path.home()
    cands = [Path(os.environ["NEURON6_CORE"])] if os.environ.get("NEURON6_CORE") else []
    cands.append(home / "Documents" / "Arduino" / "hardware" / "Ohmlab" / "stm32n6")
    for a15 in (Path(os.environ.get("LOCALAPPDATA", home)) / "Arduino15",
                home / "Library" / "Arduino15", home / ".arduino15"):
        cands += sorted((a15 / "packages" / "Ohmlab" / "hardware" / "stm32n6").glob("*"), reverse=True)
    return next((c for c in cands if (c / "system" / "npu" / "NeuroN6_MemPool.mpool").is_file()), None)


def test_memory_map_matches_the_installed_core():
    """Exports are compiled against the vendored memory map; a stale copy
    verifies models the board's build places differently (the core has
    moved the PSRAM pool and ended the weights pool at OTA slot B)."""
    core = _installed_core()
    if core is None:
        pytest.skip("no Neuro N6 core installed on this machine")
    theirs = json.loads((core / "system" / "npu" / "NeuroN6_MemPool.mpool").read_text("utf-8"))
    assert json.loads(nv.MPOOL.read_text("utf-8")) == theirs, (
        f"engine/gd/neuron6/NeuroN6_MemPool.mpool differs from {core}; copy the core's")


def test_find_stedgeai_env_override(tmp_path, monkeypatch):
    fake = tmp_path / "stedgeai.exe"
    fake.write_bytes(b"")
    monkeypatch.setenv("NEURON6_STEDGEAI", str(fake))
    assert nv.find_stedgeai() == str(fake)
    monkeypatch.setenv("NEURON6_STEDGEAI", str(tmp_path / "missing.exe"))
    assert nv.find_stedgeai() is None


def test_verify_without_tool_is_soft(tmp_path, monkeypatch):
    monkeypatch.setenv("NEURON6_STEDGEAI", str(tmp_path / "missing.exe"))
    info = nv.verify(tmp_path / "m.onnx", tmp_path / "work")
    assert info == {"verified": False, "available": False, "reason": info["reason"]}
    assert "stedgeai" in info["reason"]


def _contract_int8(path: Path, imgsz: int = 128) -> Path:
    """An untrained YOLOX-Nano, exported and quantised the way a run
    exports it: satisfies the input contract, detects nothing."""
    import numpy as np
    import torch

    from train_yolox import (_build_model, _verify_input_contract)
    from onnxruntime.quantization import (CalibrationDataReader, QuantFormat,
                                          QuantType, quantize_static)

    model = _build_model(2).eval()
    model.head.decode_in_inference = False
    fp32 = path.with_name(path.stem + "_fp32.onnx")
    torch.onnx.export(model, torch.zeros(1, 3, imgsz, imgsz), str(fp32), opset_version=17,
                      input_names=["images"], output_names=["raw"], dynamo=False)

    class R(CalibrationDataReader):
        def __init__(self):
            self.it = iter([np.zeros((1, 3, imgsz, imgsz), np.float32),
                            np.ones((1, 3, imgsz, imgsz), np.float32),
                            np.random.rand(1, 3, imgsz, imgsz).astype(np.float32)])

        def get_next(self):
            x = next(self.it, None)
            return None if x is None else {"images": x}

    quantize_static(str(fp32), str(path), R(), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                    per_channel=True)
    _verify_input_contract(path)
    return path


@pytest.mark.skipif(nv.find_stedgeai() is None, reason="ST Edge AI Core not installed")
def test_real_compile_of_a_contract_export(tmp_path):
    """End to end with the real compiler: an untrained YOLOX-Nano export
    that satisfies the input contract must build for stm32n6 with zero
    pure-software epochs and fit the pool. Slow (~1 min); skipped in CI."""
    int8 = _contract_int8(tmp_path / "m_int8.onnx")
    info = nv.verify(int8, tmp_path / "work", name="t")
    assert info["verified"] is True
    # Structure guard: a Focus stem shows up as Slice software epochs and
    # a decoded head fails to compile at all. Untrained weights with a
    # 3-image calibration can leave a few residual Adds in software
    # (degenerate activation ranges); trained exports measure 0.
    assert "Slice" not in (info.get("softwareOps") or {}), info
    assert info["epochsSoftware"] < 20, info
    assert info["epochsHardware"] > 100
    pool = nv.weights_pool_bytes()
    assert 0 < info["weightsBytes"] < pool
    assert info["poolRemainingBytes"] == pool - info["weightsBytes"]
    assert info["activationsBytes"] > 0


def _arduino_cli() -> str | None:
    """arduino-cli on PATH, or the one the Arduino IDE bundles."""
    found = shutil.which("arduino-cli")
    if found:
        return found
    for base in (Path(os.environ.get("LOCALAPPDATA", "")) / "Programs",
                 Path(os.environ.get("ProgramFiles", r"C:\Program Files"))):
        p = base / "Arduino IDE" / "resources" / "app" / "lib" / "backend" / "resources" / "arduino-cli.exe"
        if p.is_file():
            return str(p)
    mac = Path("/Applications/Arduino IDE.app/Contents/Resources/app/lib/backend/resources/arduino-cli")
    return str(mac) if mac.is_file() else None


def test_exported_sketch_builds_against_the_core(tmp_path):
    """The export's sketch compiles for the board as it ships: the model
    through the core's pragma pipeline, the post-processor and camera API
    against the installed core. Slow (a few minutes the first time);
    skipped without arduino-cli, the Neuro N6 core or ST Edge AI."""
    cli = _arduino_cli()
    if cli is None or nv.find_stedgeai() is None:
        pytest.skip("arduino-cli or ST Edge AI Core not installed")
    boards = subprocess.run([cli, "board", "listall"], capture_output=True, text=True, timeout=300)
    if "Ohmlab:stm32n6:neuron6" not in boards.stdout:
        pytest.skip("the Neuro N6 core is not installed")

    from train_yolox import TrainConfig, _write_bundle

    run = tmp_path / "run"
    run.mkdir()
    int8 = _contract_int8(run / "pk_test_int8.onnx", 192)
    report = {"run": "20260101T000000Z", "ap50_int8": 0.5, "threshold": {"calibrated": False}}
    _write_bundle(run, "pk_test", "Sketch test", ["bolt", "nut"], TrainConfig(imgsz=192),
                  report, int8, 0.5)
    zipfile.ZipFile(run / "export.zip").extractall(tmp_path / "sketches")
    out = subprocess.run([cli, "compile", "--fqbn", "Ohmlab:stm32n6:neuron6",
                          "--build-path", str(tmp_path / "build"),
                          str(tmp_path / "sketches" / "pk_test")],
                         capture_output=True, text=True, timeout=1800)
    assert out.returncode == 0, (out.stdout + out.stderr)[-3000:]
