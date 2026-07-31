"""Shared pytest fixtures + heavy-dependency stubbing.

The backend's real runtime pulls in a GPU/ML stack (torch, sam2,
transformers, nudenet, GroundingDINO) plus network clients (boto3/R2,
anthropic). None of that is available — or wanted — on a CI box or a
laptop running unit tests. So we install lightweight stand-ins into
`sys.modules` *before* any `gd` module imports them, letting `server.py`
and its siblings import cleanly and run CPU-only.

Real, kept un-stubbed: numpy + Pillow (fast on CPU, and the image code is
worth exercising for real), fastapi/starlette/pydantic (the framework
under test).

This file is auto-loaded by pytest, and the `sys.modules` injection runs
at import time (top of the file) so it's in place before any test module
imports from `gd/`. The one subtlety: torch's `nn.Module` and
`utils.data.Dataset` are given REAL (empty) base classes, because several
modules subclass them at module load and you can't subclass a MagicMock.
"""
from __future__ import annotations

import os
import sys
import types
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ── make gd/ importable (mirror of tests/test_ml_core.py) ──────────────
_HERE = Path(__file__).resolve().parent
_GD = _HERE.parent / "gd"
if str(_GD) not in sys.path:
    sys.path.insert(0, str(_GD))


# ── stub the heavy ML / network stack BEFORE anything imports it ───────
def _fake(name: str) -> MagicMock:
    """A MagicMock module — attribute access auto-creates submodules, so
    `from pkg.sub import thing` resolves as long as `pkg.sub` is also
    registered in sys.modules (see _SUBMODULES below)."""
    return MagicMock(name=name)


def _install_heavy_stubs() -> None:
    # torch: mostly MagicMock, but with concrete attributes the import
    # paths read at load time, and REAL base classes for subclassing.
    torch = _fake("torch")
    torch.cuda.is_available.return_value = False
    torch.cuda.device_count.return_value = 0
    for _dtype in ("float16", "float32", "bfloat16"):
        setattr(torch, _dtype, _dtype)

    class _Module:  # noqa: D401 — stand-in for torch.nn.Module
        pass

    class _Dataset:  # stand-in for torch.utils.data.Dataset
        pass

    nn = types.ModuleType("torch.nn")
    nn.Module = _Module
    utils = types.ModuleType("torch.utils")
    udata = types.ModuleType("torch.utils.data")
    udata.Dataset = _Dataset
    udata.DataLoader = MagicMock(name="DataLoader")
    utils.data = udata
    torch.nn = nn
    torch.utils = utils

    submodules = {
        "torch": torch,
        "torch.nn": nn,
        "torch.utils": utils,
        "torch.utils.data": udata,
        "torchvision": _fake("torchvision"),
        "torchvision.models": _fake("torchvision.models"),
        "torchvision.models.detection": _fake("torchvision.models.detection"),
        "torchvision.ops": _fake("torchvision.ops"),
        "torchvision.transforms": _fake("torchvision.transforms"),
        "torchvision.transforms.functional": _fake("torchvision.transforms.functional"),
        "sam2": _fake("sam2"),
        "sam2.build_sam": _fake("sam2.build_sam"),
        "sam2.sam2_image_predictor": _fake("sam2.sam2_image_predictor"),
        "transformers": _fake("transformers"),
        "transformers.image_utils": _fake("transformers.image_utils"),
        "accelerate": _fake("accelerate"),
        "nudenet": _fake("nudenet"),
        "boto3": _fake("boto3"),
        "botocore": _fake("botocore"),
        "botocore.client": _fake("botocore.client"),
        "anthropic": _fake("anthropic"),
        "cv2": _fake("cv2"),
    }
    for name, mod in submodules.items():
        sys.modules.setdefault(name, mod)


_install_heavy_stubs()


# ── env: a configured backend secret + terminal token by default ───────
@pytest.fixture(autouse=True)
def _auth_env(monkeypatch):
    """Most tests want a configured backend. Tests that need the unset
    case clear these explicitly with monkeypatch.delenv / setenv and (for
    server.py's import-time globals) monkeypatch the module attribute."""
    monkeypatch.setenv("BACKEND_AUTH_SECRET", "test-secret-do-not-use-in-prod")
    monkeypatch.setenv("TERMINAL_TOKEN", "test-terminal-token")
    yield


# ── point the audit log at a throwaway DB (mirror of test_ml_core) ─────
@pytest.fixture(scope="session", autouse=True)
def _isolate_audit_db():
    import audit  # imported here so the gd path is set up first
    tmp = Path(tempfile.mkdtemp(prefix="pk_audit_")) / "audit_test.db"
    audit.DB_PATH = tmp
    audit._conn = None
    yield


# ── shared fixtures for endpoint tests ─────────────────────────────────
@pytest.fixture()
def app_module():
    """The imported FastAPI server module (heavy stack already stubbed)."""
    import server
    return server


@pytest.fixture()
def client(app_module):
    """A TestClient that does NOT enter the lifespan context, so the
    startup model-loading never runs — endpoint logic is exercised with
    the stubbed state. Tests that need seeded models set app_module.state
    entries directly."""
    from fastapi.testclient import TestClient
    return TestClient(app_module.app)


@pytest.fixture()
def bearer():
    """A valid HS256 bearer token for the configured BACKEND_AUTH_SECRET,
    minted via the production signer (gd/auth.py:sign_jwt)."""
    import auth
    return auth.sign_jwt({"sub": "tester", "username": "tester"})
