# PixelKit

Open-source, local-first dataset auto-labelling for computer vision.
Text-prompted detection and segmentation with **SAM 3**, an interactive
annotation editor (boxes, polygon masks, click-to-segment), GPU-accelerated
augmentations, and YOLO / COCO / Pascal VOC export — all running on your own
machine, with every image and annotation stored in a workspace folder you
choose. No accounts, no cloud, no telemetry.

> **Status: pre-release, under active development.** Ported from the
> PixelKit SaaS codebase; the port plan and progress live in
> [`docs/PLAN.md`](docs/PLAN.md).

## Download

Grab the latest installer from
[**Releases**](https://github.com/ohmlab-ltd/pixelkit/releases):

- **Windows x64** — `PixelKit-Setup-<version>.exe` (~80 MB). On first launch
  the app downloads its AI runtime once: Python plus PyTorch (~3.5 GB CUDA
  build on NVIDIA machines, ~300 MB CPU build otherwise) into
  `%LOCALAPPDATA%\PixelKit\runtime`. The installer is currently **unsigned**,
  so SmartScreen will warn: *More info → Run anyway*.
- **macOS (Apple Silicon)** — `PixelKit-<version>.dmg` with the runtime
  bundled. Also unsigned for now: right-click the app → Open, or allow it
  under System Settings → Privacy & Security.
- **Linux** — no packaged build yet; run from source (below).

Model weights download separately on first run into your workspace
(`weights/`): SAM 3 (~1.7 GB, needs a free Hugging Face token — see
[`docs/HF-TOKEN.md`](docs/HF-TOKEN.md)) and DINOv2 (~0.6 GB, automatic).

## Requirements

| Machine | Labelling |
|---|---|
| Windows/Linux + NVIDIA GPU (12 GB VRAM recommended) | Full speed (CUDA) |
| Apple Silicon Mac (16 GB+ unified memory) | Full speed (Metal) |
| No supported GPU (incl. AMD/Intel) | App fully works — editor, datasets, import/export, augmentations. AI labelling is off by default; Settings → Compute → CPU enables it in very-slow mode. |

## Where your data lives

Everything is plain files in one folder you pick on first run (default
`~/PixelKit`): one folder per project, one per dataset, original images
untouched in `images/`, one annotation JSON per image in `annotations/`.
The format is documented in [`docs/WORKSPACE.md`](docs/WORKSPACE.md) —
it's yours to sync, back up, or parse.

Coming from the SaaS? **File → Import Legacy Backup…** converts an old
backup zip into the workspace format (CLI: `python import_legacy.py`).

## Run from source

```bash
# engine — Python 3.12, FastAPI; binds 127.0.0.1:8001
cd engine && pip install -r requirements.txt && python gd/server.py

# ui — served by the engine after a build (or `npm run dev` on :3000)
cd ui && npm install && npm run build

# desktop shell (optional in dev; packaged builds bundle everything)
cd desktop && npm ci && npm start
```

`pip install -e engine` gives you the `pixelkit` launcher and
`pixelkit doctor` environment checks. Device auto-detects cuda → mps → cpu
(`PK_DEVICE` overrides; `PK_DISABLE_MODELS=1` runs the engine with no ML).

## Port status

Phases 0–6 are done: local engine (workspace storage, no auth, split
per-image annotation schema), cuda/mps/cpu device layer, model manager +
HF-token flow, slimmed UI with first-run setup, packaging groundwork and CI.
Phase 8 (standalone desktop app) is in progress — Electron shell with
engine sidecar, desktop-shell UI (see [`docs/DESKTOP-UI.md`](docs/DESKTOP-UI.md)),
slim Windows NSIS installer with first-run runtime bootstrap, update check,
tag-driven release workflow. Remaining before v0.1.0: real-hardware QA
matrix (Phase 7), macOS signing/notarization, Vite/`app://` migration.
The full history is in [`docs/PLAN.md`](docs/PLAN.md) and the git log.

## License

Code: [Apache-2.0](LICENSE). Model weights download separately and carry
their own licenses — see [NOTICE](NOTICE). `facebook/sam3` is gated: you
accept Meta's license on Hugging Face and use your own token.
