# PixelKit (portable)

Open-source, local-first dataset auto-labelling for computer vision.
Text-prompted detection and segmentation with **SAM 3**, an interactive
annotation editor (boxes, polygon masks, click-to-segment), GPU-accelerated
augmentations, and YOLO / COCO / Pascal VOC export — all running on your own
machine, with every image and annotation stored in a workspace folder you
choose.

> **Status: pre-release, under active development.** This repo is being ported
> from the PixelKit SaaS codebase. The port plan and current progress are in
> [`docs/PLAN.md`](docs/PLAN.md).

## Targets

- **macOS** (Apple Silicon, Metal/MPS), **Linux** and **Windows** (NVIDIA CUDA)
- **12 GB VRAM** budget (VLM assist optional)
- Single-command install; no accounts, no cloud, no telemetry

## Layout

| Path | What it is |
|---|---|
| `engine/` | Python FastAPI engine: SAM 3 labelling pipeline, jobs + SSE progress, augmentations, export. Serves the UI in the packaged app. |
| `ui/` | Next.js workspace UI: annotation editor, review mode, augmentation designer, dataset stats. |
| `installers/` | Install scripts and packaging (later phase). |
| `docs/` | [`PLAN.md`](docs/PLAN.md) — full architecture + phase plan. |

## Port progress

- [x] **Phase 0 — bootstrap:** fresh-history monorepo; training stack, cloud
      deploy, billing/auth/telemetry SaaS surface and dead code removed;
      UI builds with a static local session.
- [x] **Phase 1 — local engine:** R2 → workspace filesystem storage, auth
      stubbed to a single local user, localhost-only binding, GroundingDINO/
      SAM2/SigLIP/8B-VLM stack removed (SAM3 + DINOv2 + optional 2B VLM),
      Claude/NSFW/credits/telemetry gone. Engine boots with zero config.
- [x] **Phase 2 — workspace schema:** one folder per project, one per
      dataset (`<workspace>/projects/<project>/<dataset>/`), the old
      monolithic manifest split into `dataset.json` + one annotation JSON
      per image, originals alone in `images/`, weights under
      `<workspace>/weights`; `import_legacy.py` migrates SaaS-era data.
      Golden-path test green (create → upload → annotate → export → delete).
- [x] **Phase 3 — device layer:** cuda → mps → cpu auto-detection with a
      `PK_DEVICE` override; SAM3/VLM CUDA-only gates relaxed (fp16 on Metal,
      fp32 + explicit opt-in on CPU); model loads serialized (fixes a
      transformers lazy-import thread race and boot VRAM spikes). Verified
      on Apple Silicon: DINOv2-large runs real inference on MPS; SAM3
      reaches its loader on MPS and awaits only the HF token (Phase 4).
- [x] **Phase 4 — model manager + HF token flow:** `/api/models/*` +
      `/api/settings/*` — per-model downloaded/loaded status, background
      downloads with byte-level progress + disk preflight into
      `<workspace>/weights`, token validation that distinguishes
      invalid-token from license-not-accepted (`facebook/sam3` is gated),
      token stored owner-only in the app config (never the workspace),
      boot auto-loads only already-cached models, VLM load/unload toggle.
      Setup UI screens land in Phase 5.
- [x] **Phase 5 — UI slimming + first-run setup:** pricing/community/
      terminal tabs, ProfileView, plan chip and next-auth removed entirely
      (no login or auth surface anywhere); new SetupWizard (HF token →
      weight downloads with live progress) and SettingsView (workspace,
      device, token, model load/unload) on the Phase 4 endpoints; UI is a
      static export the engine serves itself at 127.0.0.1:8001 with SPA
      deep-link fallback — one process runs the whole product. (Interim
      shape; the shipped product is the Phase 8 application.)
- [ ] Phase 6 — packaging: pinned engine env + weights/bundling groundwork
      the Phase 8 app builds on
- [ ] Phase 7 — QA on real hardware, docs, v0.1.0
- [ ] Phase 8 — **standalone desktop application:** the whole UI converted
      off Next.js into a self-contained JavaScript application (Vite SPA
      bundle loaded from disk inside an Electron shell — the VS Code /
      LM Studio architecture) that owns the engine as a child process.
      Nothing browser-based: no URL, no localhost page, no tabs — a native
      window, real menu bar, native file dialogs, drag-and-drop, and signed
      installers (.dmg / .msi / AppImage) with auto-update. A Flutter
      rewrite is documented in the plan as the fallback if the JS app
      doesn't feel native enough (cost: full editor rewrite, +8–12 weeks).

## Models

All weights download from Hugging Face into `<workspace>/weights` on first
run. `facebook/sam3` is license-gated: you accept Meta's SAM license on
Hugging Face and provide your own access token in the app's setup screen.

## Development (current state)

```bash
# engine — boots anywhere; ML endpoints need CUDA until the MPS phase lands
cd engine && pip install -r requirements.txt && python gd/server.py
# → http://127.0.0.1:8001, workspace auto-created at ~/PixelKit
#   (override with PIXELKIT_WORKSPACE)

# ui
cd ui && npm install && npm run build   # engine then serves ui/out itself
# (or `npm run dev` on :3000 for UI development)

# migrate data from the SaaS deployment
cd engine && python import_legacy.py <old-backend-dir-or-backup.zip>
```

## License

Code: [Apache-2.0](LICENSE). Model weights are downloaded separately and
carry their own licenses — see [NOTICE](NOTICE).
