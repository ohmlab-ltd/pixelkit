"""PixelKit launcher CLI.

    pixelkit            start the engine (serves the UI at 127.0.0.1:8001)
    pixelkit doctor     check device, weights, workspace, token, UI build
    pixelkit label <folder> --prompts "a,b"
                        headless batch labelling: make a dataset from a
                        folder of images, auto-label with SAM 3, and
                        optionally export (--export yolo|coco|voc|cvat|
                        labelstudio|masks). Everything lands in the
                        normal workspace, so the app can open it after.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
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


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic", ".avif"}


def label(args) -> int:
    """Headless batch labelling: boots the engine in-process (no HTTP
    port) and drives the same endpoints the app uses, so results are a
    perfectly normal dataset in the workspace."""
    folder = Path(args.folder).expanduser()
    if not folder.is_dir():
        print(f"error: not a folder: {folder}")
        return 2
    prompts = [p.strip() for p in (args.prompts or "").split(",") if p.strip()]
    if not prompts:
        print('error: --prompts is required, e.g. --prompts "bolt,washer"')
        return 2
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
    if not files:
        print(f"error: no images found in {folder}")
        return 2
    if args.device:
        os.environ["PK_DEVICE"] = args.device

    name = args.name or folder.name
    print(f"PixelKit headless labelling")
    print(f"  images:  {len(files)} from {folder}")
    print(f"  prompts: {', '.join(prompts)}")

    from fastapi.testclient import TestClient
    import server

    with TestClient(server.app) as client:
        settings = client.get("/api/settings").json()
        dev = settings.get("device") or "cpu"
        forced_cpu = (
            settings.get("deviceEnvOverride") == "cpu"
            or settings.get("devicePreference") == "cpu"
        )
        if dev == "cpu" and not forced_cpu:
            print("error: no supported GPU detected. Re-run with --device cpu to")
            print("       force (very slow), or use an NVIDIA/Apple-Silicon machine.")
            return 2
        print(f"  device:  {dev}")

        # SAM 3 must be loaded before the job can run. Boot auto-loads
        # cached weights; nudge a load if it's downloaded-but-unloaded.
        deadline = time.time() + args.model_timeout
        nudged = False
        while True:
            sam3 = client.get("/api/models/status").json()["models"]["sam3"]
            if sam3["loaded"]:
                break
            if not sam3["downloaded"]:
                print("error: SAM 3 weights aren't downloaded yet. Open the app once")
                print("       and finish setup (Hugging Face token + download).")
                return 2
            if not nudged:
                client.post("/api/models/sam3/load")
                nudged = True
            if time.time() > deadline:
                print(f"error: SAM 3 didn't finish loading in {args.model_timeout}s")
                return 2
            print("\r  loading SAM 3…", end="", flush=True)
            time.sleep(2)
        print("\r  SAM 3 loaded.   ")

        r = client.post(
            "/api/v2/projects",
            data={"name": name, "labels": json.dumps(prompts)},
        )
        if r.status_code != 200:
            print(f"error: could not create dataset ({r.status_code}): {r.text[:200]}")
            return 1
        pid = r.json()["project_id"]

        uploaded = 0
        for i, p in enumerate(files, 1):
            with p.open("rb") as fh:
                r = client.post(
                    f"/api/v2/projects/{pid}/imports/raw",
                    files={"image": (p.name, fh, "application/octet-stream")},
                )
            if r.status_code == 200:
                uploaded += 1
            else:
                print(f"\n  ! skipped {p.name} ({r.status_code})")
            print(f"\r  uploading {i}/{len(files)}", end="", flush=True)
        print(f"\r  uploaded {uploaded}/{len(files)}   ")
        if uploaded == 0:
            print("error: nothing uploaded")
            return 1

        r = client.post(
            "/api/jobs", json={"kind": "label_charlie", "project": pid, "params": {}}
        )
        if r.status_code != 200:
            print(f"error: could not start labelling ({r.status_code}): {r.text[:200]}")
            return 1
        jid = r.json().get("job_id") or r.json().get("id")

        while True:
            jobs = client.get("/api/jobs/active").json().get("jobs", [])
            mine = next((j for j in jobs if j.get("id") == jid), None)
            if mine is None:
                break
            prog = mine.get("progress") or {}
            idx, total = prog.get("index"), prog.get("total")
            if idx is not None and total:
                print(f"\r  labelling {idx}/{total}", end="", flush=True)
            time.sleep(2)
        print("\r  labelling done.          ")

        if args.export:
            r = client.get(
                f"/api/projects/{pid}/export", params={"format": args.export}
            )
            if r.status_code != 200:
                print(f"error: export failed ({r.status_code}): {r.text[:200]}")
                return 1
            out_dir = Path(args.out or ".").expanduser()
            out_dir.mkdir(parents=True, exist_ok=True)
            out = out_dir / f"{name}-{args.export}.zip"
            out.write_bytes(r.content)
            print(f"  export:  {out}")

        import workspace

        print(f"\nDataset '{name}' ({pid}) is in the workspace: {workspace.dir()}")
        print("Open PixelKit to review and edit the annotations.")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(prog="pixelkit", description=__doc__)
    sub = ap.add_subparsers(dest="command")
    sub.add_parser("serve", help="start the engine (default)").add_argument(
        "--port", type=int, default=8001
    )
    sub.add_parser("doctor", help="environment checks")
    lp = sub.add_parser("label", help="headless batch labelling of an image folder")
    lp.add_argument("folder", help="folder of images to label")
    lp.add_argument("--prompts", "-p", required=True, help='comma-separated concepts, e.g. "bolt,washer"')
    lp.add_argument("--name", help="dataset name (default: folder name)")
    lp.add_argument("--export", choices=["yolo", "coco", "voc", "cvat", "labelstudio", "masks"])
    lp.add_argument("--out", help="directory for the export zip (default: cwd)")
    lp.add_argument("--device", help="force a device (e.g. cpu, cuda:1)")
    lp.add_argument("--model-timeout", type=int, default=600, dest="model_timeout")
    ap.add_argument("--port", type=int, default=8001)
    args = ap.parse_args()

    cmd = args.command or "serve"
    if cmd == "doctor":
        raise SystemExit(doctor())
    if cmd == "label":
        raise SystemExit(label(args))
    print(f"PixelKit engine → http://127.0.0.1:{args.port}")
    serve(args.port)


if __name__ == "__main__":
    main()
