# PixelKit engine

Python FastAPI engine for the portable build: SAM 3 auto-labelling
("Pipeline Charlie"), job queue + SSE progress, GPU augmentations,
YOLO/COCO/VOC export — all storing into a local workspace folder
(default `~/PixelKit`, override with `PIXELKIT_WORKSPACE`).

    <workspace>/projects/<project>/<dataset>/
        dataset.json      # metadata + labels + per-image index
        images/           # original images only
        annotations/      # one JSON per image (boxes, masks, labels)

Run:

    pip install -r requirements.txt
    python gd/server.py            # http://127.0.0.1:8001

No auth, no cloud, binds localhost only. Models load on CUDA today
(Metal/MPS lands in the device phase); without a GPU the engine still runs
every dataset/annotation/export API and the labelling endpoints return 503.

Migrate SaaS-era data: `python import_legacy.py <old-dir-or-backup.zip>`.

Tests: `python -m pytest tests/`.
