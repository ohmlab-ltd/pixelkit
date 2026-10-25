"""Engine test fixtures; the environment below is set before any engine import."""
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gd"))

_WS = tempfile.mkdtemp(prefix="pixelkit-test-ws-")
os.environ["PIXELKIT_WORKSPACE"] = _WS
os.environ["PIXELKIT_CONFIG_DIR"] = tempfile.mkdtemp(prefix="pixelkit-test-cfg-")
os.environ["PK_DISABLE_MODELS"] = "1"
# "-1" hides every GPU; an empty value does not on Windows.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.pop("HF_TOKEN", None)


@pytest.fixture(scope="session")
def app_module():
    import server
    return server


@pytest.fixture(scope="session")
def workspace_dir() -> Path:
    return Path(_WS)


@pytest.fixture(scope="session")
def client(app_module):
    from fastapi.testclient import TestClient
    with TestClient(app_module.app, base_url="http://127.0.0.1") as c:
        yield c


def st_mpool_template(pools=("AXIFLEXMEM", "AXISRAM1", "AXISRAM2", "AXISRAM3", "AXISRAM4",
                               "xSPI1", "xSPI2")) -> dict:
    """A stand-in for ST Edge AI's stm32n6.mpool; pool i has read_power i."""
    return {"params": {"param": [{"paramname": "max_onchip_sram_size", "value": "1024",
                                  "magnitude": "KBYTES"}]},
            "memory": {"cacheinfo": [], "mem_file_prefix": "atonbuf",
                       "mempools": [{"fname": f, "name": f.lower(), "fformat": "FORMAT_RAW",
                                     "prop": {"rights": "ACC_WRITE", "byteWidth": 8,
                                              "freqRatio": 1.0, "read_power": i},
                                     "offset": {"value": "0x0", "magnitude": "BYTES"},
                                     "size": {"value": "0", "magnitude": "KBYTES"}}
                                    for i, f in enumerate(pools)]}}


@pytest.fixture
def fake_stedgeai(tmp_path, monkeypatch):
    """fake_stedgeai(windows, posix) installs a stand-in stedgeai script as
    NEURON6_STEDGEAI, with a stand-in stm32n6.mpool in its install."""
    def install(windows: str, posix: str) -> Path:
        import json
        template = tmp_path.joinpath("scripts", "N6_scripts", "my_mpools", "stm32n6.mpool")
        template.parent.mkdir(parents=True, exist_ok=True)
        template.write_text(json.dumps(st_mpool_template()), encoding="utf-8")
        if os.name == "nt":
            path = tmp_path / "stedgeai.bat"
            path.write_text(windows, encoding="ascii")
        else:
            path = tmp_path / "stedgeai"
            path.write_text("#!/bin/sh\n" + posix, encoding="ascii")
            path.chmod(0o755)
        monkeypatch.setenv("NEURON6_STEDGEAI", str(path))
        return path
    return install
