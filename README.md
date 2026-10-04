# PixelKit

[![ci](https://github.com/ohmlab-ltd/pixelkit/actions/workflows/ci.yml/badge.svg)](https://github.com/ohmlab-ltd/pixelkit/actions/workflows/ci.yml)
[![release](https://img.shields.io/github/v/release/ohmlab-ltd/pixelkit)](https://github.com/ohmlab-ltd/pixelkit/releases/latest)

Open-source, local-first dataset auto-labelling for computer vision, and
detector training for the OhmLab Neuro N6. Text-prompted detection and
segmentation with **SAM 3**, an interactive annotation editor (boxes, polygon
masks, click-to-segment), GPU-accelerated augmentations, YOLO / COCO / Pascal
VOC export, and YOLOX training with an int8 export ready to build for the
Neuro N6 - all running on your own machine, with every image and annotation
stored in a workspace folder you choose. No accounts, no cloud, no telemetry.

> **Status: pre-release, under active development.**

## Download

Grab the latest installer from
[**Releases**](https://github.com/ohmlab-ltd/pixelkit/releases):

- **Windows x64** - `PixelKit-Setup.exe` (~80 MB) from the latest
  release. On first launch
  the app downloads its AI runtime once: Python plus PyTorch (~3.5 GB CUDA
  build on NVIDIA machines, ~300 MB CPU build otherwise) into
  `%LOCALAPPDATA%\PixelKit\runtime`. The installer is currently **unsigned**,
  so SmartScreen will warn: *More info → Run anyway*.
- **macOS (Apple Silicon)** - `PixelKit-<version>.dmg` with the runtime
  bundled. Also unsigned for now: right-click the app → Open, or allow it
  under System Settings → Privacy & Security.
- **Linux** - no packaged build yet; run from source (below).

Model weights download separately on first run into your workspace
(`weights/`): SAM 3 (~3.4 GB, needs a free Hugging Face token - see
[`docs/HF-TOKEN.md`](docs/HF-TOKEN.md)) and DINOv2 (~1.2 GB, automatic).
The first training run downloads Megvii's COCO-pretrained YOLOX checkpoint
it starts from (~8 MB for Nano).

## Requirements

| Machine | Labelling | Training |
|---|---|---|
| Windows/Linux + NVIDIA GPU (12 GB VRAM recommended) | Full speed (CUDA) | Full speed: a Nano run takes minutes |
| Apple Silicon Mac (16 GB+ unified memory) | Full speed (Metal) | Metal |
| No supported GPU (incl. AMD/Intel) | App fully works - editor, datasets, import/export, augmentations. AI labelling is off by default; Settings → Compute → CPU enables it in very-slow mode. | The same switch, slowly |

## Train for the Neuro N6

On a labelled dataset, **Train for Neuro N6** trains a YOLOX-Nano or
YOLOX-Tiny detector and exports it as an Arduino sketch folder: the int8
model (trained quantisation-aware, so it scores like the float model), a
labels header with a confidence threshold calibrated on the validation
images, and a report (AP@0.5, per-class precision and recall, the images the
model got most wrong, every validation image beside its ground truth). With
ST Edge AI Core installed, every export is also compiled for the board's NPU
and its fit reported. Build the sketch with the Neuro N6 board package
(0.1.3 or later) and ST Edge AI Core, as
[`docs/TRAINING.md`](docs/TRAINING.md#building-it-for-the-board) describes;
Neuro Studio shows the boxes on the live video, with the threshold as a
slider. Details in [`docs/TRAINING.md`](docs/TRAINING.md);
the same runs work headless (`pixelkit train`) and through the local API
([`docs/API.md`](docs/API.md)).

## Where your data lives

Everything is plain files in one folder you pick on first run (default
`~/PixelKit`): one folder per project, one per dataset, original images
untouched in `images/`, one annotation JSON per image in `annotations/`.
The format is documented in [`docs/WORKSPACE.md`](docs/WORKSPACE.md) -
it's yours to sync, back up, or parse.

Coming from the SaaS? **File → Import Legacy Backup…** converts an old
backup zip into the workspace format (CLI: `python import_legacy.py`).

## Run from source

```bash
# engine - Python 3.12, FastAPI; binds 127.0.0.1:8001
cd engine && pip install -r requirements.txt && python gd/server.py

# ui - served by the engine after a build (or `npm run dev` on :3000)
cd ui && npm install && npm run build

# desktop shell (optional in dev; packaged builds bundle everything)
cd desktop && npm ci && npm start
```

`pip install -e engine` gives you the `pixelkit` launcher and
`pixelkit doctor` environment checks. Device auto-detects cuda → mps → cpu
(`PK_DEVICE` overrides; `PK_DISABLE_MODELS=1` runs the engine with no ML).

## Status

Working today: labelling with SAM 3, the annotation editor,
augmentations, dataset import and export, training and the Neuro N6
export, headless labelling and training (`pixelkit label`, `pixelkit
train`) and the local API, and the desktop app with Windows and macOS
installers, a first-run runtime download and an update check. The
desktop-shell UI is documented in [`docs/DESKTOP-UI.md`](docs/DESKTOP-UI.md).
Not yet: signed installers, a Linux package, and training for
classification, segmentation or pose.

## License

Code: [Apache-2.0](LICENSE). Model weights download separately and carry
their own licenses - see [NOTICE](NOTICE). `facebook/sam3` is gated: you
accept Meta's license on Hugging Face and use your own token.
