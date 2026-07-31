"""Workspace resolution for the portable build.

Everything PixelKit stores — projects, datasets, images, annotations, model
weights — lives inside one user-chosen workspace folder. This module resolves
that folder and must be imported BEFORE torch/transformers so HF_HOME points
model downloads into the workspace.

Resolution order:
  1. PIXELKIT_WORKSPACE env var (also how tests isolate themselves)
  2. "workspace" key in the app config file (per-OS config dir)
  3. default: ~/PixelKit

The app config dir (small, machine-local, never inside the workspace — the
workspace may be synced or shared, and config holds the HF token):
  macOS:   ~/Library/Application Support/PixelKit
  Windows: %APPDATA%/PixelKit
  Linux:   $XDG_CONFIG_HOME/pixelkit or ~/.config/pixelkit
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

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


def load_config() -> dict:
    path = config_dir() / "config.json"
    try:
        return json.loads(path.read_text("utf-8"))
    except Exception:
        return {}


def save_config(cfg: dict) -> None:
    d = config_dir()
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "config.json.tmp"
    tmp.write_text(json.dumps(cfg, indent=2), "utf-8")
    os.replace(tmp, d / "config.json")


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
    """The workspace root (created on first access)."""
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
        # Model downloads land in the workspace unless the user has an
        # explicit HF_HOME of their own.
        os.environ.setdefault("HF_HOME", str(ws / "weights"))
        _WORKSPACE = ws
    return _WORKSPACE


def projects_dir() -> Path:
    return dir() / "projects"


def weights_dir() -> Path:
    return dir() / "weights"


def set_workspace(path: str) -> Path:
    """Persist a new workspace location in the app config (takes effect on
    next process start; live re-pointing is not supported)."""
    cfg = load_config()
    cfg["workspace"] = str(Path(path).expanduser())
    save_config(cfg)
    return Path(cfg["workspace"])
