"""PixelKit launcher CLI.

    pixelkit            start the engine (serves the UI at 127.0.0.1:8001)
    pixelkit doctor     check device, weights, workspace, token, UI build
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ENGINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ENGINE_DIR / "gd"))


def _ok(label: str, value: str) -> None:
    print(f"  ✓ {label:<14} {value}")


def _warn(label: str, value: str) -> None:
    print(f"  ! {label:<14} {value}")


def doctor() -> int:
    problems = 0
    print("PixelKit doctor\n")

    v = sys.version_info
    if v >= (3, 11):
        _ok("python", f"{v.major}.{v.minor}.{v.micro}")
    else:
        _warn("python", f"{v.major}.{v.minor} — need 3.11+")
        problems += 1

    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            _ok("device", f"cuda — {name} ({vram:.0f} GB)")
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            _ok("device", "mps — Apple GPU (Metal)")
        else:
            _warn("device", "cpu only — labelling disabled (PK_DEVICE=cpu to force)")
    except Exception as e:
        _warn("torch", f"import failed: {e}")
        problems += 1

    import workspace

    ws = workspace.dir()
    _ok("workspace", str(ws))

    import models

    _ok("free disk", f"{models.free_disk_gb():.0f} GB")
    if models.hf_token():
        _ok("hf token", "configured")
    else:
        _warn("hf token", "not set — needed once for the gated SAM3 weights")
    for name in ("sam3", "dinov2"):
        if models.is_downloaded(name):
            _ok(name, "weights downloaded")
        else:
            _warn(name, "weights not downloaded (setup screen or /api/models)")

    ui = ENGINE_DIR.parent / "ui" / "out"
    if (ui / "index.html").is_file():
        _ok("ui build", str(ui))
    else:
        _warn("ui build", "missing — run `npm run build` in ui/ (engine API still works)")

    print()
    return 1 if problems else 0


def serve(port: int) -> None:
    import uvicorn

    uvicorn.run("server:app", host="127.0.0.1", port=port, reload=False)


def main() -> None:
    ap = argparse.ArgumentParser(prog="pixelkit", description=__doc__)
    ap.add_argument("command", nargs="?", default="serve", choices=["serve", "doctor"])
    ap.add_argument("--port", type=int, default=8001)
    args = ap.parse_args()
    if args.command == "doctor":
        raise SystemExit(doctor())
    print(f"PixelKit engine → http://127.0.0.1:{args.port}")
    serve(args.port)


if __name__ == "__main__":
    main()
