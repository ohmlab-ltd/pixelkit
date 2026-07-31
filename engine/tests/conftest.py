"""Test fixtures for the portable engine.

The workspace is a per-session temp dir (PIXELKIT_WORKSPACE is read once at
import). PK_DISABLE_MODELS keeps every ML load out — these tests cover the
dataset/annotation/storage surface, which must work with zero models.
"""
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gd"))

_WS = tempfile.mkdtemp(prefix="pixelkit-test-ws-")
os.environ["PIXELKIT_WORKSPACE"] = _WS
# Never let tests read or write the user's real app config (HF token!).
os.environ["PIXELKIT_CONFIG_DIR"] = tempfile.mkdtemp(prefix="pixelkit-test-cfg-")
os.environ["PK_DISABLE_MODELS"] = "1"
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
    with TestClient(app_module.app) as c:
        yield c
