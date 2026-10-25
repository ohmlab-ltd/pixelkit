"""PixelKit launcher CLI.

    pixelkit            start the engine (serves the UI at 127.0.0.1:8001)
    pixelkit doctor     check device, weights, workspace, token, UI build
    pixelkit label <folder> --prompts "a,b"
                        make a workspace dataset from a folder of images,
                        label it with SAM 3, and optionally export it with
                        --export yolo|coco|voc|cvat|labelstudio|masks
    pixelkit train <dataset>
                        train a Neuro N6 model on a workspace dataset
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ENGINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ENGINE_DIR / "gd"))


def _ok(label: str, value: str) -> None:
    print(f"  ok {label:<14} {value}")


def _warn(label: str, value: str) -> None:
    print(f"  !! {label:<14} {value}")


def doctor() -> int:
    problems = 0
    print("PixelKit doctor\n")

    v = sys.version_info
    if v >= (3, 11):
        _ok("python", f"{v.major}.{v.minor}.{v.micro}")
    else:
        _warn("python", f"{v.major}.{v.minor} - need 3.11+")
        problems += 1

    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            _ok("device", f"cuda - {name} ({vram:.0f} GB)")
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            _ok("device", "mps - Apple GPU (Metal)")
        else:
            _warn("device", "cpu only - labelling disabled (PK_DEVICE=cpu to force)")
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
        _warn("hf token", "not set - needed once for the gated SAM3 weights")
    for name in ("sam3", "dinov2"):
        if models.is_downloaded(name):
            _ok(name, "weights downloaded")
        else:
            _warn(name, "weights not downloaded; the engine fetches them when it starts")

    ui = ENGINE_DIR.parent / "ui" / "out"
    if (ui / "index.html").is_file():
        _ok("ui build", str(ui))
    else:
        _warn("ui build", "missing - run `npm run build` in ui/ (engine API still works)")

    print()
    return 1 if problems else 0


def serve(port: int) -> None:
    import uvicorn

    uvicorn.run("server:app", host="127.0.0.1", port=port, reload=False)


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic", ".avif"}


def label(args) -> int:
    """Label a folder headlessly through the app's endpoints, in-process."""
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
    print("PixelKit headless labelling")
    print(f"  images:  {len(files)} from {folder}")
    print(f"  prompts: {', '.join(prompts)}")

    from fastapi.testclient import TestClient
    import server

    with TestClient(server.app, base_url="http://127.0.0.1") as client:
        settings = client.get("/api/settings").json()
        dev = settings.get("device") or "cpu"
        env_dev = settings.get("deviceEnvOverride")
        # PK_DEVICE overrides the configured device preference.
        forced_cpu = env_dev == "cpu" or (not env_dev and settings.get("devicePreference") == "cpu")
        if dev == "cpu" and not forced_cpu:
            print("error: no supported GPU detected. Re-run with --device cpu to")
            print("       force (very slow), or use an NVIDIA/Apple-Silicon machine.")
            return 2
        print(f"  device:  {dev}")

        # The job needs SAM 3 loaded; start a load if the weights are on disk.
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
                r = client.post("/api/models/sam3/load")
                if r.status_code == 409:
                    detail = r.json().get("detail", r.text[:160])
                    print(f"error: {detail}")
                    return 2
                nudged = True
            if time.time() > deadline:
                print(f"error: SAM 3 didn't finish loading in {args.model_timeout}s")
                return 2
            print("\r  loading SAM 3...", end="", flush=True)
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
        # A failed job also leaves /active, so check the final status.
        final = client.get(f"/api/jobs/{jid}")
        body = final.json() if final.status_code == 200 else {}
        status = body.get("status", "unknown")
        if status != "done":
            print(f"\rerror: labelling job ended with status '{status}': {body.get('error') or 'no detail'}")
            return 1
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


def _resolve_dataset(ref: str) -> str | None:
    """Resolve a dataset id, or a unique dataset name in any case, to an id."""
    import store

    ref = (ref or "").strip()
    if not ref:
        return None
    if store.dataset_exists(ref):
        return ref
    hits = []
    for pid in store.iter_dataset_ids():
        if _dataset_name(pid).strip().lower() == ref.lower():
            hits.append(pid)
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        print(f"error: {len(hits)} datasets are called '{ref}'; use the id")
        raise SystemExit(2)
    return None


def _dataset_name(pid: str) -> str:
    """The dataset's name from dataset.json alone; store.load reads every annotation file."""
    import store

    try:
        return str((json.loads((store.dataset_dir(pid) / "dataset.json").read_text("utf-8"))
                    or {}).get("name") or "")
    except (OSError, ValueError, KeyError):
        try:
            return str((store.load(pid) or {}).get("name") or "")
        except Exception:  # noqa: BLE001
            return ""


def train(args) -> int:
    """Run the train_yolox job in-process, as the Train card does."""
    if args.device:
        os.environ["PK_DEVICE"] = args.device

    from fastapi.testclient import TestClient
    import server

    with TestClient(server.app, base_url="http://127.0.0.1") as client:
        pid = _resolve_dataset(args.dataset)
        if pid is None:
            print(f"error: no dataset '{args.dataset}' in the workspace (id or exact name)")
            return 2
        eval_pid = None
        if args.eval_dataset:
            eval_pid = _resolve_dataset(args.eval_dataset)
            if eval_pid is None:
                print(f"error: no held-out dataset '{args.eval_dataset}'")
                return 2
        settings = client.get("/api/settings").json()
        dev = settings.get("device") or "cpu"
        if dev == "cpu" and not args.device:
            print("error: no supported GPU detected. Re-run with --device cpu to force")
            print("       (slow), or use an NVIDIA/Apple-Silicon machine.")
            return 2
        name = _dataset_name(pid) or pid
        print("PixelKit training")
        print(f"  dataset: {name} ({pid})")
        print(f"  device:  {args.device or dev}")

        params: dict = {}
        if args.resume:
            params["resumeRun"] = args.resume
            print(f"  resume:  {args.resume}")
        else:
            params = {
                "epochs": args.epochs, "imgsz": args.input, "size": args.size,
                "split": args.split, "augmentations": not args.no_augmentations,
            }
            if args.from_run:
                params["initRun"] = args.from_run
            if eval_pid:
                params["evalProject"] = eval_pid
            if args.conf is not None:
                params["conf"] = args.conf
            print(f"  model:   YOLOX-{args.size.capitalize()} at {args.input}x{args.input}, "
                  f"{args.epochs} epochs, split {args.split:.2f}"
                  + (", no augmentations" if args.no_augmentations else "")
                  + (f", from run {args.from_run}" if args.from_run else "")
                  + (f", held-out dataset {args.eval_dataset}" if eval_pid else ""))
            r = client.get(f"/api/v2/projects/{pid}/train/preflight", params={
                "imgsz": args.input, "split": args.split, "size": args.size,
                "augmentations": 0 if args.no_augmentations else 1,
                "evalProject": eval_pid or "",
            })
            if r.status_code != 200:
                print(f"error: pre-flight failed ({r.status_code}): {r.text[:200]}")
                return 1
            pf = r.json()
            imgs = pf.get("images") or {}
            if imgs:
                print(f"  images:  {imgs.get('train', 0)} train, {imgs.get('val', 0)} val"
                      + (f", {imgs['augmented']} augmented copies" if imgs.get("augmented") else ""))
            for w in pf.get("warnings") or []:
                print(f"  warning: {w}")
            for b in pf.get("blockers") or []:
                print(f"  blocked: {b}")
            if not pf.get("ok"):
                return 2
        if args.device:
            params["device"] = args.device

        r = client.post("/api/jobs", json={"kind": "train_yolox", "project": pid, "params": params})
        if r.status_code != 200:
            detail = r.json().get("detail", r.text[:200]) if r.headers.get("content-type", "").startswith("application/json") else r.text[:200]
            print(f"error: could not start training ({r.status_code}): {detail}")
            return 1
        jid = r.json().get("jobId") or r.json().get("id")

        last = ""
        j: dict = {}
        status = None
        try:
            while True:
                j = client.get(f"/api/jobs/{jid}").json()
                status = j.get("status")
                prog = j.get("progress") or {}
                idx, total, text = prog.get("index"), prog.get("total"), prog.get("image") or ""
                if status in ("done", "failed", "cancelled", "interrupted"):
                    break
                line = f"  epoch {idx}/{total}  {text}" if idx is not None and total else "  starting"
                if line != last:
                    print(line, flush=True)
                    last = line
                time.sleep(3)
        except KeyboardInterrupt:
            # The cancelled run keeps its last checkpoint for --resume.
            print("\n  cancelling after the current epoch...", flush=True)
            client.delete(f"/api/v2/projects/{pid}/jobs/{jid}")
            deadline = time.time() + 600
            while time.time() < deadline:
                j = client.get(f"/api/jobs/{jid}").json()
                if j.get("status") in ("done", "failed", "cancelled", "interrupted"):
                    break
                time.sleep(2)
            models = client.get(f"/api/v2/projects/{pid}/models").json().get("models") or []
            paused = next((m for m in models if m.get("resumable")), None)
            if paused:
                print(f"  cancelled. Resume with: pixelkit train {args.dataset} --resume {paused['run']}")
            else:
                print("  cancelled.")
            return 130
        if status != "done":
            print(f"error: training ended with status '{status}': {j.get('error') or 'no detail'}")
            return 1

        models = client.get(f"/api/v2/projects/{pid}/models").json().get("models") or []
        result = j.get("result") or {}
        run = next((m for m in models if m.get("run") == result.get("run")), None)
        if run is None:
            done = [m for m in models if m.get("status") == "done"]
            run = done[0] if done else None
        if run is None:
            print("error: training finished but no run was written")
            return 1
        detail = client.get(f"/api/v2/projects/{pid}/models/{run['run']}").json()
        rep = detail.get("report") or {}
        thr = rep.get("threshold") or {}
        n6 = rep.get("neuron6") or {}
        print(f"\n  run:       {run['run']}  ({rep.get('minutes')} min)")
        print(f"  AP@0.5:    float {rep.get('ap50_fp32')}  ->  int8 {rep.get('ap50_int8')}")
        if thr:
            print(f"  threshold: {thr.get('conf')}"
                  + (f"  (F1 {thr.get('f1')}, precision {thr.get('precision')}, recall {thr.get('recall')})"
                     if thr.get("calibrated")
                     else ("  (default: the model found nothing on the val split)"
                           if thr.get("fallback") else "  (as configured)")))
        if n6.get("verified"):
            print(f"  Neuro N6:  {n6.get('epochsSoftware')} software epochs, "
                  f"{(n6.get('weightsBytes') or 0) // 1024} kB weights "
                  f"({n6.get('poolUsedPct')}% of the pool)")
        elif n6.get("error"):
            print(f"  Neuro N6:  REJECTED: {n6['error']}")
        else:
            print("  Neuro N6:  not verified (ST Edge AI not installed)")
        for w in rep.get("warnings") or []:
            print(f"  warning:   {w}")

        import store

        run_dir = store.dataset_dir(pid) / "models" / str(run.get("model") or "yolox_nano") / run["run"]
        print(f"  files:     {run_dir}")
        if args.out:
            out_dir = Path(args.out).expanduser()
            out_dir.mkdir(parents=True, exist_ok=True)
            z = run_dir / "export.zip"
            if z.is_file():
                slug = re.sub(r"[^a-z0-9]+", "-", str(name).lower()).strip("-") or pid
                dest = out_dir / f"{slug}-neuron6-{run['run']}.zip"
                dest.write_bytes(z.read_bytes())
                print(f"  export:    {dest}")
    return 0


def main() -> None:
    # A redirected Windows console is cp1252; unencodable characters are replaced.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(prog="pixelkit", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command")
    # SUPPRESS keeps a --port given before `serve` from being overwritten.
    sub.add_parser("serve", help="start the engine (default)").add_argument(
        "--port", type=int, default=argparse.SUPPRESS
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
    tp = sub.add_parser("train", help="train a Neuro N6 model on a dataset in the workspace")
    tp.add_argument("dataset", help="dataset id or exact name")
    tp.add_argument("--epochs", type=int, default=50)
    tp.add_argument("--input", type=int, default=256, choices=[128, 192, 256, 320, 416],
                    help="square input size (default 256)")
    tp.add_argument("--size", choices=["nano", "tiny"], default="nano", help="YOLOX size (default nano)")
    tp.add_argument("--split", type=float, default=0.8, help="train fraction, 0.5-0.95 (default 0.8)")
    tp.add_argument("--no-augmentations", action="store_true", dest="no_augmentations",
                    help="ignore the dataset's generated augmentations")
    tp.add_argument("--eval-dataset", dest="eval_dataset",
                    help="a second dataset (id or name) used as the held-out set")
    tp.add_argument("--from", dest="from_run", help="warm-start from a previous run id")
    tp.add_argument("--resume", help="continue an interrupted run id (other options are ignored)")
    tp.add_argument("--conf", type=float, help="fix the confidence threshold instead of calibrating it")
    tp.add_argument("--device", help="force a device (e.g. cpu, cuda:1)")
    tp.add_argument("--out", help="directory to copy the N6 export zip into")
    ap.add_argument("--port", type=int, default=8001)
    args = ap.parse_args()

    cmd = args.command or "serve"
    if cmd == "doctor":
        raise SystemExit(doctor())
    if cmd == "label":
        raise SystemExit(label(args))
    if cmd == "train":
        raise SystemExit(train(args))
    print(f"PixelKit engine -> http://127.0.0.1:{args.port}")
    serve(args.port)


if __name__ == "__main__":
    main()
