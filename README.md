# PixelKit

[![ci](https://github.com/ohmlab-ltd/pixelkit/actions/workflows/ci.yml/badge.svg)](https://github.com/ohmlab-ltd/pixelkit/actions/workflows/ci.yml)
[![release](https://img.shields.io/github/v/release/ohmlab-ltd/pixelkit)](https://github.com/ohmlab-ltd/pixelkit/releases/latest)

PixelKit is a desktop application for labelling image datasets and training
object detectors for the Ohm Lab Neuro N6. It runs locally, with no account
and no telemetry, and stores images and annotations as plain files in a
workspace folder.

- Text-prompted detection and segmentation with SAM 3.
- Annotation editor for boxes and polygon masks, with click-to-segment.
- GPU augmentations.
- Import from YOLO, COCO and Pascal VOC; export to those formats and to CVAT,
  Label Studio and mask images.
- Videos and Neuro Studio recordings import as frames.
- YOLOX training with an int8 export for the Neuro N6.

## Download

Installers are on the [Releases](https://github.com/ohmlab-ltd/pixelkit/releases/latest)
page.

| Platform | File | Python runtime |
|---|---|---|
| Windows 10 or later, x64 | `PixelKit-Setup.exe`, 115 MB | Downloaded on first launch into `%LOCALAPPDATA%\PixelKit\runtime`: about 3.5 GB with CUDA on NVIDIA machines, which needs 9 GB free, and 3 GB free otherwise |
| macOS 14 or later, Apple Silicon | `PixelKit-arm64.dmg`, 410 MB | Included |
| Linux | none | Run from source |

The packaged app checks GitHub for a newer release at launch and offers the
release page; `PIXELKIT_NO_UPDATE_CHECK=1` turns the check off.

Model weights download on first run into the workspace's `weights/` folder:
SAM 3, about 3.6 GB, which needs a Hugging Face token
([docs/HF-TOKEN.md](docs/HF-TOKEN.md)), and DINOv2, about 1.3 GB; each
download needs 2 GB free beyond its size. The first training run downloads
Megvii's COCO-pretrained YOLOX checkpoint, 8 MB for Nano and 41 MB for Tiny.

### First launch

The installers are not signed with paid certificates yet, so Windows and
macOS ask you to confirm PixelKit once. After that it opens like any other
app.

On macOS:

1. Open `PixelKit-arm64.dmg` and drag PixelKit to Applications.
2. Open PixelKit from Applications. macOS says it cannot verify the app.
   Close the message with Done, OK or Cancel, whichever it shows, not Move
   to Trash.
3. Open System Settings, then Privacy & Security. Near the bottom it says
   PixelKit was blocked. Click Open Anyway. If the button is not there, open
   PixelKit once more and come back: it shows for about an hour after a
   blocked launch.
4. Confirm with Open Anyway or Open, and with your password or Touch ID if
   asked. PixelKit opens.

If macOS instead says PixelKit "is damaged and can't be opened", run this in
Terminal, then open it again:

```bash
xattr -dr com.apple.quarantine "/Applications/PixelKit.app"
```

On Windows, if "Windows protected your PC" appears when you run
`PixelKit-Setup.exe`, click More info, then Run anyway.

## Requirements

| Machine | Labelling | Training |
|---|---|---|
| NVIDIA GPU, 12 GB VRAM recommended | CUDA | CUDA, minutes for a Nano run |
| Apple Silicon, 16 GB memory or more | Metal | Metal |
| No supported GPU | Off by default. Settings > Compute > CPU enables it, at minutes per image. | CPU, with the same setting |

Without a supported GPU, the editor, datasets, import, export and
augmentations work as usual.

## Training for the Neuro N6

The Train for Neuro N6 card on a dataset's overview trains a YOLOX-Nano or
YOLOX-Tiny detector and exports an Arduino sketch folder with:

- the int8 model, trained quantisation-aware;
- a labels header with a confidence threshold calibrated on the validation
  images;
- `report.json`: AP@0.5, per-class precision and recall, the confusion
  matrix and the threshold sweep;
- `NOTICE.txt` and `LICENSE.txt`, which stay with the model.

The run's Details on the card show the validation images with the most
errors and every validation image beside its ground truth.

With ST Edge AI Core installed, each export is also compiled for the NPU and
its fit is reported. The sketch builds with the Neuro N6 board package 0.1.3
or later and ST Edge AI Core
([docs/TRAINING.md](docs/TRAINING.md#building-for-the-board)). Neuro Studio
shows the boxes on the live video, with the threshold as a slider. Training
also runs from the command line (`pixelkit train`) and through the local API
([docs/API.md](docs/API.md)).

## Workspace

All data is plain files in one workspace folder, `~/PixelKit` unless changed
under Settings, Workspace or by `PIXELKIT_WORKSPACE`: a folder per project
and per dataset, the imported images in `images/`, one annotation JSON per
image in `annotations/`. JPEG files are kept byte for byte; other image
formats are stored as JPEG. The format is specified in
[docs/WORKSPACE.md](docs/WORKSPACE.md).

File > Import Legacy Backup imports a backup zip of an earlier PixelKit
server into the workspace; `python engine/import_legacy.py <folder or zip>`
does the same from the command line.

## Running from source

`installers/install.ps1` on Windows or `installers/install.sh` on macOS and
Linux creates `engine/.venv`, installs PyTorch, with CUDA when an NVIDIA GPU
is found, and the engine, then builds the UI. It needs Python 3.11 or later
and Node.js 22.12 or later.

```bash
# Engine on 127.0.0.1:8001, serving the UI from ui/out
engine/.venv/bin/pixelkit       # Windows: engine\.venv\Scripts\pixelkit.exe

# Desktop shell: uses a running engine, or starts one from engine/.venv
cd desktop && npm ci && npm start
```

The engine looks for `ui/out` when it starts, so the UI is built first.
`npm run dev` in `ui/` serves the UI on port 3000 against the engine.
`pixelkit doctor` checks the environment. The engine uses CUDA, then MPS,
then the CPU; `PK_DEVICE` overrides the device, and `PK_DISABLE_MODELS=1`
runs the engine without models.

## Status

Pre-release. The installers are not signed with paid certificates
([First launch](#first-launch)), there is no Linux package, and training
covers object detection only.

## Documentation

- [Training and Neuro N6 export](docs/TRAINING.md)
- [Local API](docs/API.md)
- [Workspace format](docs/WORKSPACE.md)
- [Hugging Face token](docs/HF-TOKEN.md)
- [Desktop UI](docs/DESKTOP-UI.md)
- [Contributing](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)

## License

Code: [Apache-2.0](LICENSE). Model weights download separately and carry
their own licences, listed in [NOTICE](NOTICE). `facebook/sam3` is gated:
each user accepts Meta's licence on Hugging Face and uses their own token.
Third-party components and their licences are listed in
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).
