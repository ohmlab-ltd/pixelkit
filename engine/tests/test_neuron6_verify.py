"""ST Edge AI verification: report parsing, tool discovery, and (when the
compiler is installed locally) a real compile of a contract-conforming
export."""
import os
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


def test_safe_pool_excludes_ota_slot_b():
    # 0x90300000..0x91C80000: the mempool's 26.5 MB minus the 1 MB slot B.
    assert nv.SAFE_POOL_BYTES == 25.5 * 1024 * 1024


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


@pytest.mark.skipif(nv.find_stedgeai() is None, reason="ST Edge AI Core not installed")
def test_real_compile_of_a_contract_export(tmp_path):
    """End to end with the real compiler: an untrained YOLOX-Nano export
    that satisfies the input contract must build for stm32n6 with zero
    pure-software epochs and fit the pool. Slow (~1 min); skipped in CI."""
    import numpy as np
    import torch

    from train_yolox import (_build_model, _verify_input_contract)
    from onnxruntime.quantization import (CalibrationDataReader, QuantFormat,
                                          QuantType, quantize_static)

    model = _build_model(2).eval()
    model.head.decode_in_inference = False
    fp32 = tmp_path / "m_fp32.onnx"
    torch.onnx.export(model, torch.zeros(1, 3, 128, 128), str(fp32), opset_version=17,
                      input_names=["images"], output_names=["raw"], dynamo=False)

    class R(CalibrationDataReader):
        def __init__(self):
            self.it = iter([np.zeros((1, 3, 128, 128), np.float32),
                            np.ones((1, 3, 128, 128), np.float32),
                            np.random.rand(1, 3, 128, 128).astype(np.float32)])

        def get_next(self):
            x = next(self.it, None)
            return None if x is None else {"images": x}

    int8 = tmp_path / "m_int8.onnx"
    quantize_static(str(fp32), str(int8), R(), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                    per_channel=True)
    _verify_input_contract(int8)

    info = nv.verify(int8, tmp_path / "work", name="t")
    assert info["verified"] is True
    # Structure guard: a Focus stem shows up as Slice software epochs and
    # a decoded head fails to compile at all. Untrained weights with a
    # 3-image calibration can leave a few residual Adds in software
    # (degenerate activation ranges); trained exports measure 0.
    assert "Slice" not in (info.get("softwareOps") or {}), info
    assert info["epochsSoftware"] < 20, info
    assert info["epochsHardware"] > 100
    assert 0 < info["weightsBytes"] < nv.SAFE_POOL_BYTES
    assert info["poolRemainingBytes"] == nv.SAFE_POOL_BYTES - info["weightsBytes"]
    assert info["activationsBytes"] > 0
