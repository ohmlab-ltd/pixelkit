# PixelKit engine

FastAPI server for PixelKit: SAM 3 labelling, a job queue whose progress is
read from `GET /api/jobs/{id}`, GPU augmentations, dataset import and export,
and YOLOX training.
Data is stored in the workspace folder, `~/PixelKit` by default or
`PIXELKIT_WORKSPACE` ([docs/WORKSPACE.md](../docs/WORKSPACE.md)).

Install PyTorch first, from download.pytorch.org with the CUDA index on an
NVIDIA machine; from PyPI, Windows gets a CPU-only build.

```bash
pip install -r requirements.txt
python gd/server.py            # http://127.0.0.1:8001
```

The server binds to localhost and has no authentication. The device is
CUDA, then MPS, then the CPU; `PK_DEVICE` or the Settings device picker
overrides it. Without a supported GPU, every dataset, annotation and export
API works, and the labelling endpoints return 503 unless the CPU is chosen
in Settings.

Command line, after `pip install -e .`:

```bash
pixelkit serve
pixelkit doctor
pixelkit label ./photos --prompts "bolt,washer" --export yolo
pixelkit train <dataset> --epochs 50 --size nano
```

`pixelkit label` also takes `--name`, `--out DIR`, `--device cpu|cuda:N` and
`--model-timeout S`. Training is described in
[docs/TRAINING.md](../docs/TRAINING.md).

`python import_legacy.py <folder or zip>` imports a backup of an earlier
PixelKit server into the workspace.

Tests: `pip install pytest httpx`, then `python -m pytest tests/`.
