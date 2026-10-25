"""Workspace folder and app config.

The config holds the HF token, so it lives outside the workspace. dir() sets
HF_HOME and must run before transformers is imported.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Callable

_SCHEMA_VERSION = 1


def config_dir() -> Path:
    env = (os.environ.get("PIXELKIT_CONFIG_DIR") or "").strip()
    if env:
        return Path(env).expanduser()
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support" / "PixelKit"
    elif os.name == "nt":
        base = Path(os.environ.get("APPDATA", str(Path.home()))) / "PixelKit"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))) / "pixelkit"
    return base


_CONFIG_LOCK = threading.Lock()


def load_config() -> dict:
    """The config; {} when it is missing or unreadable. A read that meets a
    concurrent replace on Windows is retried."""
    path = config_dir() / "config.json"
    for i in range(20):
        try:
            cfg = json.loads(path.read_text("utf-8"))
            return cfg if isinstance(cfg, dict) else {}
        except PermissionError:
            time.sleep(0.01 * (i + 1))
        except (OSError, ValueError):
            return {}
    return {}


def save_config(cfg: dict) -> None:
    d = config_dir()
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f"config.json.{uuid.uuid4().hex}.tmp"
    # Owner-only from the start: the config holds the HF token.
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(json.dumps(cfg, indent=2))
    os.replace(tmp, d / "config.json")


def update_config(change: Callable[[dict], None]) -> dict:
    """Load, change and save the config as one step; returns the saved config."""
    with _CONFIG_LOCK:
        cfg = load_config()
        change(cfg)
        save_config(cfg)
        return cfg


def _resolve_workspace() -> Path:
    env = (os.environ.get("PIXELKIT_WORKSPACE") or "").strip()
    if env:
        return Path(env).expanduser()
    cfg = load_config()
    if cfg.get("workspace"):
        return Path(cfg["workspace"]).expanduser()
    return Path.home() / "PixelKit"


_WORKSPACE: Path | None = None


def dir() -> Path:
    """The workspace root, created on first access."""
    global _WORKSPACE
    if _WORKSPACE is None:
        ws = _resolve_workspace()
        ws.mkdir(parents=True, exist_ok=True)
        (ws / "projects").mkdir(exist_ok=True)
        (ws / "weights").mkdir(exist_ok=True)
        marker = ws / "workspace.json"
        if not marker.exists():
            tmp = ws / "workspace.json.tmp"
            tmp.write_text(
                json.dumps({"app": "pixelkit", "schemaVersion": _SCHEMA_VERSION}, indent=2),
                "utf-8",
            )
            os.replace(tmp, marker)
        os.environ.setdefault("HF_HOME", str(ws / "weights"))
        _WORKSPACE = ws
    return _WORKSPACE


def projects_dir() -> Path:
    return dir() / "projects"


def weights_dir() -> Path:
    return dir() / "weights"


def set_workspace(path: str) -> Path:
    """Save a new workspace location; it takes effect at the next start."""
    cfg = update_config(lambda c: c.__setitem__("workspace", str(Path(path).expanduser())))
    return Path(cfg["workspace"])
