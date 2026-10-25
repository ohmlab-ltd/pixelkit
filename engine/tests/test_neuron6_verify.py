"""ST Edge AI verification: report parsing, tool discovery and, with the compiler installed, a real compile."""
import json
import os
import shutil
import subprocess
import sys
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


def test_the_memory_map_takes_st_figures_and_the_board_pools(tmp_path):
    from conftest import st_mpool_template

    template = tmp_path / "stm32n6.mpool"
    template.write_text(json.dumps(st_mpool_template()), encoding="utf-8")
    m = nv.build_mpool(template)
    assert m["params"]["param"] == [{"paramname": "max_onchip_sram_size", "value": "1792",
                                     "magnitude": "KBYTES"}]
    pools = {p["fname"]: p for p in m["memory"]["mempools"]}
    assert [p["fname"] for p in m["memory"]["mempools"]] == [
        "AXIFLEXMEM", "AXISRAM1", "AXISRAM2", "AXISRAM3", "xSPI2", "xSPI1"]
    assert pools["xSPI2"]["name"] == "PSRAM" and pools["xSPI2"]["prop"]["read_power"] == 5
    assert pools["xSPI2"]["prop"]["constants_preferred"] == "false"
    assert pools["xSPI1"]["name"] == "OCTAFLASH" and pools["xSPI1"]["prop"]["read_power"] == 6
    assert pools["xSPI1"]["prop"]["byteWidth"] == 2
    assert pools["xSPI1"]["offset"]["value"] == "0x90300000"
    out = tmp_path / "out.mpool"
    out.write_text(json.dumps(m), encoding="utf-8")
    assert nv.weights_pool_bytes(out) == nv.weights_pool_bytes()

    template.write_text(json.dumps(st_mpool_template(pools=("AXIFLEXMEM", "xSPI1"))), encoding="utf-8")
    with pytest.raises(ValueError, match="AXISRAM1"):
        nv.build_mpool(template)


def test_the_template_is_found_in_the_stedgeai_install(tmp_path):
    exe = tmp_path / "4.0" / "Utilities" / "windows" / "stedgeai.exe"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"")
    assert nv.mpool_template(str(exe)) is None
    template = tmp_path.joinpath("4.0", *nv.MPOOL_TEMPLATE)
    template.parent.mkdir(parents=True)
    template.write_text("{}", encoding="utf-8")
    assert nv.mpool_template(str(exe)) == template.resolve()
    assert nv.mpool_template(None) is None


def _installed_core() -> Path | None:
    """A Neuro N6 core on this machine: NEURON6_CORE, the sketchbook's
    hardware folder, or a Boards Manager install."""
    home = Path.home()
    cands = [Path(os.environ["NEURON6_CORE"])] if os.environ.get("NEURON6_CORE") else []
    cands.append(home / "Documents" / "Arduino" / "hardware" / "Ohmlab" / "stm32n6")
    for a15 in (Path(os.environ.get("LOCALAPPDATA", home)) / "Arduino15",
                home / "Library" / "Arduino15", home / ".arduino15"):
        cands += sorted((a15 / "packages" / "Ohmlab" / "hardware" / "stm32n6").glob("*"), reverse=True)
    return next((c for c in cands if (c / "platform.txt").is_file() and (c / "tools").is_dir()), None)


def _core_mpool(core: Path, exe: str) -> dict | None:
    """The memory map the core builds with: printed by its model generator,
    or an older core's system/npu/NeuroN6_MemPool.mpool."""
    gen = core / "tools" / "neuron6_model_gen.py"
    if gen.is_file():
        out = subprocess.run([sys.executable, "-B", str(gen), "--print-mpool", exe],
                             capture_output=True, text=True, timeout=120)
        if out.returncode == 0 and out.stdout.strip():
            return json.loads(out.stdout)
    old = core / "system" / "npu" / "NeuroN6_MemPool.mpool"
    return json.loads(old.read_text("utf-8")) if old.is_file() else None


def test_memory_map_matches_the_installed_core():
    """The memory map built from ST's template matches the installed core's."""
    core = _installed_core()
    exe = nv.find_stedgeai()
    template = nv.mpool_template(exe)
    theirs = _core_mpool(core, exe) if core and template else None
    if theirs is None:
        pytest.skip("needs a Neuro N6 core and ST Edge AI Core on this machine")
    assert nv.build_mpool(template) == theirs, (
        f"neuron6_verify.POOLS differs from the memory map of {core}")


def test_find_stedgeai_env_override(tmp_path, monkeypatch):
    fake = tmp_path / "stedgeai.exe"
    fake.write_bytes(b"")
    monkeypatch.setenv("NEURON6_STEDGEAI", str(fake))
    assert nv.find_stedgeai() == str(fake)
    monkeypatch.setenv("NEURON6_STEDGEAI", str(tmp_path / "missing.exe"))
    assert nv.find_stedgeai() is None


def _ll_aton_header(path: Path, version: tuple) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"#define LL_ATON_VERSION_{k:<6}({v})\n" for k, v in
                            zip(("MAJOR", "MINOR", "MICRO", "DEV"), version)), encoding="utf-8")


def test_find_stedgeai_picks_the_install_the_core_builds_with(tmp_path, monkeypatch):
    """The install whose ll_aton matches the core's, else the newest version by number."""
    root = tmp_path / "STEdgeAI"
    for version in ("4.0", "9.0", "10.0"):
        exe = root / version / "Utilities" / "windows" / "stedgeai.exe"
        exe.parent.mkdir(parents=True)
        exe.write_bytes(b"")
    _ll_aton_header(root.joinpath("4.0", *nv.INSTALL_LL_ATON), (1, 1, 3, 275))
    monkeypatch.delenv("NEURON6_STEDGEAI", raising=False)
    monkeypatch.setattr(nv, "_stedgeai_patterns",
                        lambda: [str(root / "*" / "Utilities" / "windows" / "stedgeai.exe")])
    monkeypatch.setattr(nv.shutil, "which", lambda *_a, **_k: None)

    def picked() -> str:
        return Path(nv.find_stedgeai()).parents[2].name

    monkeypatch.setattr(nv, "core_ll_aton", lambda: None)
    assert picked() == "10.0"  # by name, 9.0 would sort last
    monkeypatch.setattr(nv, "core_ll_aton", lambda: (1, 1, 3, 275))
    assert picked() == "4.0"
    monkeypatch.setattr(nv, "core_ll_aton", lambda: (2, 0, 0, 0))
    assert picked() == "10.0"


def test_install_versions_sort_as_the_core_sorts_them():
    names = ["9.0", "10.0-beta", "10.1", "10.0", "10.0.1"]
    assert sorted(names, key=nv._version_key, reverse=True) == \
        ["10.1", "10.0.1", "10.0", "10.0-beta", "9.0"]


def test_the_cores_ll_aton_comes_from_the_core_the_build_uses(tmp_path, monkeypatch):
    """NEURON6_CORE, else the sketchbook's core, else the newest Boards Manager install."""
    monkeypatch.delenv("NEURON6_CORE", raising=False)
    monkeypatch.setenv("ARDUINO_DIRECTORIES_USER", str(tmp_path / "sketchbook"))
    monkeypatch.setenv("ARDUINO_DIRECTORIES_DATA", str(tmp_path / "data"))
    assert nv.core_ll_aton() is None
    installs = tmp_path / "data" / "packages" / "Ohmlab" / "hardware" / "stm32n6"
    _ll_aton_header(installs.joinpath("0.9.0", *nv.CORE_LL_ATON), (1, 1, 2, 0))
    _ll_aton_header(installs.joinpath("0.10.0", *nv.CORE_LL_ATON), (1, 1, 3, 275))
    assert nv.core_ll_aton() == (1, 1, 3, 275)
    sketchbook = tmp_path / "sketchbook" / "hardware" / "Ohmlab" / "stm32n6"
    _ll_aton_header(sketchbook.joinpath(*nv.CORE_LL_ATON), (1, 1, 1, 0))
    assert nv.core_ll_aton() == (1, 1, 1, 0)
    _ll_aton_header((tmp_path / "pinned").joinpath(*nv.CORE_LL_ATON), (1, 0, 0, 0))
    monkeypatch.setenv("NEURON6_CORE", str(tmp_path / "pinned"))
    assert nv.core_ll_aton() == (1, 0, 0, 0)
    assert nv.ll_aton_version(tmp_path / "missing.h") is None


def test_verify_without_tool_is_soft(tmp_path, monkeypatch):
    monkeypatch.setenv("NEURON6_STEDGEAI", str(tmp_path / "missing.exe"))
    info = nv.verify(tmp_path / "m.onnx", tmp_path / "work")
    assert info == {"verified": False, "available": False, "reason": info["reason"]}
    assert "stedgeai" in info["reason"]


def test_a_compiler_that_cannot_start_leaves_the_export_unverified(tmp_path, fake_stedgeai):
    """A compiler that exits 0xC0000142 or prints nothing is tried twice, then the export is unverified."""
    calls = tmp_path / "calls.txt"
    fake_stedgeai(f'@echo x>>"{calls}"\r\n@exit /b -1073741502\r\n', f'echo x >> "{calls}"\nexit 1\n')
    model = tmp_path / "m.onnx"
    model.write_bytes(b"onnx")
    info = nv.verify(model, tmp_path / "keep")
    assert info["verified"] is False and info["available"] is True
    assert "did not run" in info["reason"] and "not verified" in info["reason"]
    if os.name == "nt":
        assert "0xC0000142" in info["reason"]
    assert calls.read_text().count("x") == 2


def test_a_compiler_that_runs_and_refuses_rejects_the_model(tmp_path, fake_stedgeai):
    fake_stedgeai("@echo E103(CliRuntimeError): ERROR: unsupported layer Foo\r\n@exit /b 1\r\n",
                  'echo "E103(CliRuntimeError): ERROR: unsupported layer Foo"\nexit 1\n')
    model = tmp_path / "m.onnx"
    model.write_bytes(b"onnx")
    with pytest.raises(RuntimeError, match="rejected the model.*unsupported layer Foo"):
        nv.verify(model, tmp_path / "keep")


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
    """An untrained YOLOX-Nano contract export builds for stm32n6 and fits the pool."""
    int8 = _contract_int8(tmp_path / "m_int8.onnx")
    info = nv.verify(int8, tmp_path / "work", name="t")
    assert info["verified"] is True
    # A Focus stem shows as Slice software epochs; untrained weights can leave a few Adds in software.
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
    """The export's sketch compiles against the installed core: as exported,
    with WiFi, and for the TFT carrier with the boxes on its panel."""
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
    sketch = tmp_path / "sketches" / "pk_test"
    out = subprocess.run([cli, "compile", "--fqbn", "Ohmlab:stm32n6:neuron6",
                          "--build-path", str(tmp_path / "build"), str(sketch)],
                         capture_output=True, text=True, timeout=1800)
    assert out.returncode == 0, (out.stdout + out.stderr)[-3000:]

    # The WiFi build.
    ino = sketch / "pk_test.ino"
    ino.write_text(ino.read_text("utf-8").replace("#define STREAM_OVER_WIFI 0",
                                                  "#define STREAM_OVER_WIFI 1"), "utf-8", newline="\n")
    out = subprocess.run([cli, "compile", "--fqbn", "Ohmlab:stm32n6:neuron6",
                          "--build-path", str(tmp_path / "build-wifi"), str(sketch)],
                         capture_output=True, text=True, timeout=1800)
    assert out.returncode == 0, (out.stdout + out.stderr)[-3000:]

    # The TFT carrier build, as its comments describe it.
    text = (ino.read_text("utf-8")
            .replace("#define STREAM_OVER_WIFI 1", "#define STREAM_OVER_WIFI 0")
            .replace("OV5640 camera;", "OV5640 camera(CARRIER_TFT);")
            .replace("cfg.transport = TRANSPORT_USB;", "cfg.transport = TRANSPORT_LCD;"))
    assert "OV5640 camera(CARRIER_TFT);" in text and "cfg.transport = TRANSPORT_LCD;" in text
    ino.write_text(text, "utf-8", newline="\n")
    out = subprocess.run([cli, "compile", "--fqbn", "Ohmlab:stm32n6:neuron6",
                          "--build-path", str(tmp_path / "build-tft"), str(sketch)],
                         capture_output=True, text=True, timeout=1800)
    assert out.returncode == 0, (out.stdout + out.stderr)[-3000:]


def _model(tmp_path) -> Path:
    model = tmp_path / "m.onnx"
    model.write_bytes(b"onnx")
    return model


def test_a_compiler_that_cannot_be_started_leaves_the_export_unverified(tmp_path, fake_stedgeai, monkeypatch):
    fake = fake_stedgeai("@exit /b 0\r\n", "exit 0\n")
    # A file the OS will not run: no exec bit, or not a program on Windows.
    broken = fake.with_name("stedgeai.txt")
    broken.write_text("not a program", encoding="ascii")
    monkeypatch.setenv("NEURON6_STEDGEAI", str(broken))
    info = nv.verify(_model(tmp_path), tmp_path / "keep")
    assert info["verified"] is False and info["available"] is True
    assert "could not start" in info["reason"] and "not verified" in info["reason"]


def test_a_compile_that_runs_too_long_leaves_the_export_unverified(tmp_path, fake_stedgeai, monkeypatch):
    fake_stedgeai("@ping -n 30 127.0.0.1 >nul\r\n", "sleep 30\n")
    monkeypatch.setattr(nv, "COMPILE_TIMEOUT_S", 1)
    info = nv.verify(_model(tmp_path), tmp_path / "keep")
    assert info["verified"] is False and "took more than" in info["reason"]
    assert (tmp_path / "keep" / "stedgeai.log").is_file()


def test_shutting_down_stops_a_running_compile_and_starts_no_more(tmp_path, fake_stedgeai):
    import threading
    import time

    fake_stedgeai("@ping -n 60 127.0.0.1 >nul\r\n", "sleep 60\n")
    out = {}
    t = threading.Thread(target=lambda: out.update(nv.verify(_model(tmp_path), tmp_path / "keep")))
    t.start()
    try:
        for _ in range(100):
            if nv._RUNNING:
                break
            time.sleep(0.05)
        assert nv._RUNNING, "the compile never started"
        t0 = time.time()
        nv.stop_running_compilers()
        t.join(timeout=30)
        assert not t.is_alive() and time.time() - t0 < 30
        assert out["verified"] is False and "closing" in out["reason"]
    finally:
        nv.allow_compilers()
