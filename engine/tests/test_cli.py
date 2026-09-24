"""The launcher CLI's arguments. No engine is started: serve() is
replaced by a recorder."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pixelkit_cli  # noqa: E402


@pytest.mark.parametrize("argv, port", [
    ([], 8001),
    (["serve"], 8001),
    (["--port", "8011"], 8011),
    (["--port", "8011", "serve"], 8011),
    (["serve", "--port", "8012"], 8012),
])
def test_serve_takes_the_port_before_or_after_the_command(monkeypatch, argv, port):
    started = []
    monkeypatch.setattr(pixelkit_cli, "serve", started.append)
    monkeypatch.setattr(sys, "argv", ["pixelkit", *argv])
    pixelkit_cli.main()
    assert started == [port]
