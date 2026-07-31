"""Regression tests for the terminal-token fail-closed fix (findings
C1/C2).

Before the fix, `TERMINAL_TOKEN` fell back to a hardcoded literal when
the env var was unset, so anyone with the source had admin access. Now
an unset token fails closed (503) and the literal is gone.

Runs under pytest (`python -m pytest tests/test_security.py`) with the
heavy ML stack stubbed by conftest.py.
"""
import importlib

import pytest


# ── resolve_terminal_token: the pure, env-only resolver ────────────────
def test_resolve_terminal_token_returns_value_when_set(monkeypatch):
    import auth
    monkeypatch.setenv("TERMINAL_TOKEN", "  spaced-token  ")
    assert auth.resolve_terminal_token() == "spaced-token"  # trimmed


def test_resolve_terminal_token_none_when_unset(monkeypatch):
    import auth
    monkeypatch.delenv("TERMINAL_TOKEN", raising=False)
    assert auth.resolve_terminal_token() is None


def test_resolve_terminal_token_none_when_blank(monkeypatch):
    import auth
    monkeypatch.setenv("TERMINAL_TOKEN", "   ")
    assert auth.resolve_terminal_token() is None


def test_no_hardcoded_fallback_token_in_source():
    """The leaked literal must not reappear anywhere in gd/server.py."""
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent / "gd" / "server.py").read_text()
    assert "wOL7HeNIlc9ix4klTZGWJi5T6" not in src


# ── endpoint behaviour through the dependency ──────────────────────────
@pytest.fixture()
def terminal_url():
    return "/api/terminal/whoami"


def test_terminal_endpoint_200_with_correct_token(app_module, client, terminal_url):
    app_module.TERMINAL_TOKEN = "test-terminal-token"
    r = client.get(terminal_url, headers={"X-Terminal-Token": "test-terminal-token"})
    assert r.status_code == 200


def test_terminal_endpoint_401_with_wrong_token(app_module, client, terminal_url):
    app_module.TERMINAL_TOKEN = "test-terminal-token"
    r = client.get(terminal_url, headers={"X-Terminal-Token": "nope"})
    assert r.status_code == 401


def test_terminal_endpoint_401_with_missing_token(app_module, client, terminal_url):
    app_module.TERMINAL_TOKEN = "test-terminal-token"
    r = client.get(terminal_url)
    assert r.status_code == 401


def test_terminal_endpoint_503_when_token_unset(app_module, client, terminal_url):
    """Fail-closed: no configured token → 503, never an open door."""
    app_module.TERMINAL_TOKEN = None
    r = client.get(terminal_url, headers={"X-Terminal-Token": "anything"})
    assert r.status_code == 503


if __name__ == "__main__":  # allow running as a plain script too
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
