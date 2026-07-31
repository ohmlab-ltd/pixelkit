# PixelKit engine

Python FastAPI engine: SAM 3 auto-labelling (Pipeline "Charlie"),
job queue + SSE progress, GPU augmentations, dataset export.

See the root README and docs/PLAN.md for the port status. Until Phase 1
lands, parts of this tree still assume the SaaS deployment (R2 storage,
JWT auth, credit gates) and a CUDA device.

Run: `pip install -r requirements.txt && python gd/server.py` (port 8001).
