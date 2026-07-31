"""FastAPI server: named-project pipeline backed by manifest.json.

Run:
    conda activate groundingdino
    python gd/server.py

Project layout:
    projects/<name>/
        images/         input images
        outputs/        annotated jpgs (server-rendered for first pass)
        manifest.json   {tags, thresholds, results, verdicts, editedBoxes, ...}

Endpoints:
    GET    /api/health
    GET    /api/projects                       list summaries
    POST   /api/projects        body: {name}   create empty project
    GET    /api/projects/{project_id}                full manifest
    PUT    /api/projects/{project_id}                merge fields into manifest
    DELETE /api/projects/{project_id}                rm -rf project
    POST   /api/projects/{project_id}/run            multipart run (clears prior results)
    GET    /api/projects/{project_id}/events         SSE progress
    GET    /api/projects/{project_id}/originals/{file}
    GET    /api/projects/{project_id}/files/{file}
"""
import asyncio
import hashlib
import hmac
import io
import json
import os
import uuid as _uuid

# orjson is 3-5× faster than the stdlib json on both parse and dump
# for the large nested manifests this server reads and the large
# annotations payloads it returns. Falls back to stdlib if the
# package isn't installed so the server still boots in environments
# that haven't picked up the new dependency yet.
try:
    import orjson as _orjson
except Exception:
    _orjson = None


def _json_loads(s: str | bytes) -> object:
    if _orjson is not None:
        return _orjson.loads(s.encode() if isinstance(s, str) else s)
    return json.loads(s if isinstance(s, str) else s.decode())

# Load backend/.env (one dir up from gd/) so R2_*, FRONTEND_ORIGINS, etc. work
# without exporting in every terminal. Done before any os.getenv reads below.
from dotenv import load_dotenv  # noqa: E402
from pathlib import Path as _Path  # noqa: E402
load_dotenv(_Path(__file__).resolve().parent.parent / ".env", override=True)

# CUDA allocator: switch to expandable segments BEFORE `import torch`.
# This is the single biggest fix for the long-run-slowdown we kept
# hitting — Qwen-VL allocates differently shaped tensors per image
# crop, and the default caching allocator fragments around those into
# pinned chunks it can't reuse. With expandable segments, those chunks
# grow instead of getting orphaned, so VRAM stays usable across many
# images without periodic empty_cache thrashing.
# https://docs.pytorch.org/docs/stable/notes/cuda.html#environment-variables
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import re
import secrets
import shutil
import tempfile
import time
import zipfile
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import uvicorn
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, ORJSONResponse, RedirectResponse, Response
from PIL import Image as PILImage, ImageOps
# Hard ceiling on the pixel count Pillow will accept for any image
# this process decodes. Without it a 1 KB compressed PNG declaring
# 50 000 × 50 000 dimensions (a "decompression bomb") would expand to
# ~7.5 GB of pixel buffer and OOM the worker. Pillow defaults to a
# warning at ~89 MP; bumping that to a hard cap of 100 MP rejects
# anything larger with PIL.Image.DecompressionBombError so the
# request fails cleanly instead of taking the process with it.
PILImage.MAX_IMAGE_PIXELS = 100_000_000

# Register HEIF/AVIF decoders if the optional plugin is installed, so
# Pillow can decode .heic/.heif/.avif uploads (iPhone photos, modern
# web image formats). Pillow 11+ has native AVIF support; pillow-heif
# covers older Pillow + HEIC. Best-effort: a missing package is a
# no-op — the import path then rejects undecodable formats with a clean
# 400 rather than storing a file nothing downstream can read.
try:
    import pillow_heif as _pillow_heif  # type: ignore
    _pillow_heif.register_heif_opener()
    try:
        _pillow_heif.register_avif_opener()
    except Exception:
        pass
    print("[server] pillow-heif registered (HEIC/AVIF decode enabled)")
except Exception:
    pass

# Per-file size cap. Single biggest legitimate image we've seen is
# ~30 MB (raw 4K JPEG); 100 MB leaves a 3x headroom while keeping a
# single attacker from posting a 5 GB body and filling pending_dir
# (or the request memory) before NSFW screening even runs.
MAX_UPLOAD_BYTES_PER_FILE = 100 * 1024 * 1024
# Cap on items per batch-upload request. A free-tier user could
# otherwise spray 10 000 tiny images in one POST and exhaust the
# label queue + NSFW GPU long before the per-month quota engages.
MAX_FILES_PER_UPLOAD_BATCH = 100


def _enforce_upload_caps(blobs: list[tuple[str, bytes, str | None]]) -> None:
    """Raise 413 if any single file or the batch as a whole exceeds
    the configured caps. Called after the request body has been
    read in fully (FastAPI streams UploadFile, so we know the actual
    size only after `.read()`)."""
    if len(blobs) > MAX_FILES_PER_UPLOAD_BATCH:
        raise HTTPException(
            413,
            f"too many files in one upload ({len(blobs)} > {MAX_FILES_PER_UPLOAD_BATCH})",
        )
    for fn, data, _ in blobs:
        if len(data) > MAX_UPLOAD_BYTES_PER_FILE:
            raise HTTPException(
                413,
                f"file too large: {fn} is {len(data)} bytes "
                f"(max {MAX_UPLOAD_BYTES_PER_FILE})",
            )
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from jobs import JobManager
from storage import LocalStorage, from_env as _storage_from_env
from storage import R2Storage  # alias of LocalStorage; legacy name at ~20 call sites
import audit
import store
from auth import (
    current_user,
    resolve_terminal_token,
    require_user_match,
    require_project_owner as _require_project_owner_factory,
    require_project_read_access as _require_project_read_access_factory,
    require_project_manage as _require_project_manage_factory,
    require_dataset_creator as _require_dataset_creator_factory,
    can_read_project_request,
    request_username,
)
import containers

NSFW_THRESHOLD = 0.3  # referenced by dead guarded branches; gate itself removed
EXPOSED_CLASSES: tuple = ()  # dead-branch stub (NSFW gate removed)


def nsfw_score(*_a, **_k):  # dead-branch stub — state["nsfw"] is always None
    return 0.0, ""


def segment_point(*_a, **_k):  # dead-branch stub — SAM2 removed, guards keep these unreached
    return None


def segment_boxes(*_a, **_k):  # dead-branch stub — SAM2 removed
    return []



def _enforce_nsfw_or_451(
    raw: bytes,
    *,
    label: str = "upload",
    project: str = "",
    file: str = "",
    user: str = "",
) -> None:
    """NSFW gate removed in the portable build — a local user labelling
    their own images doesn't get content-policed."""
    return


# Local filesystem storage rooted in the workspace. The `R2` name survives
# because ~25 call sites read it; it now always resolves.
R2: LocalStorage = _storage_from_env()


def r2_required() -> LocalStorage:
    return R2


def _redirect_to_r2(key: str) -> FileResponse:
    """SaaS build 302'd to a presigned R2 URL. Locally we serve the file
    directly. Name kept — many call sites."""
    try:
        path = R2.resolve(key)
    except FileNotFoundError:
        raise HTTPException(404, "not found")
    if not path.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(path, headers={"Cache-Control": "public, max-age=900"})


def _invalidate_url_cache(key: str) -> None:
    """Presigned-URL cache is gone; kept as a no-op for its call sites."""
    return


# Job/activity events persist to a local SQLite in the workspace.
def add_event(kind: str, **data) -> None:
    try:
        audit.add_event(kind, **data)
    except Exception as e:
        # Audit logging never breaks the request path; just print.
        print(f"[audit] add_event({kind}) failed: {e}")


import threading as _threading  # noqa: E402


def _pil_from_r2(project: str, filename: str) -> PILImage.Image:
    """Load an image from workspace storage, apply EXIF orientation, return RGB."""
    data = r2_required().get_bytes(LocalStorage.image_key(project, filename))
    img = PILImage.open(io.BytesIO(data))
    return ImageOps.exif_transpose(img).convert("RGB")

import workspace  # noqa: E402

ROOT = workspace.dir()
PROJECTS_DIR = workspace.projects_dir()

NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")
RESERVED = {"_jobs"}


state: dict[str, Any] = {"model": None, "segmenter": None, "device": "cpu", "jobs": JobManager()}


# Idle gate for fire-and-forget background work (per-upload whole-image
# embeddings). These embeddings feed ONLY the stats card (2-D variation
# plot, near-duplicate flag, one health factor); labelling never reads
# them. A bulk import queues thousands of them on the GPU, and if the
# user then starts labelling, that backlog fights the label job's SAM3
# calls for the GPU and makes labelling crawl. We pause the backlog
# while any GPU job (label/augment) runs so the job gets the full GPU;
# the embeddings drain afterwards when the GPU is idle. The event is
# created at startup (binds to the loop); helpers no-op until then.
_GPU_JOB_DEPTH = 0


def _pause_bg_embeddings() -> None:
    """Called when a GPU job starts. Clears the idle event so queued
    background embeddings wait instead of starving the job."""
    global _GPU_JOB_DEPTH
    _GPU_JOB_DEPTH += 1
    ev = state.get("gpu_idle")
    if ev is not None:
        ev.clear()


def _resume_bg_embeddings() -> None:
    """Called when a GPU job ends. Re-arms the idle event once the last
    concurrent job has finished so deferred embeddings can drain."""
    global _GPU_JOB_DEPTH
    _GPU_JOB_DEPTH = max(0, _GPU_JOB_DEPTH - 1)
    if _GPU_JOB_DEPTH == 0:
        ev = state.get("gpu_idle")
        if ev is not None:
            ev.set()


def _gpu_job_guarded(runner):
    """Wrap a JobManager runner so background embeddings pause for its
    whole lifetime (even on early return / exception). Used for GPU job
    kinds that don't already pause inline."""
    async def _wrapped(job, emit, cancel_event):
        _pause_bg_embeddings()
        try:
            return await runner(job, emit, cancel_event)
        finally:
            _resume_bg_embeddings()
    return _wrapped


async def _run_bg_embedding(project_id: str, import_id: str, raw_bytes: bytes) -> None:
    """Compute + store the whole-image DINOv2 embedding for the stats
    card (variation plot / near-dup / health), OFF the upload hot path.

    The embed_sem caps how many embeddings get past the idle gate at
    once; the idle gate (cleared while a label/augment job runs) then
    pauses them. Together: a bulk import's thousands of embed tasks can
    NEVER pile onto the GPU lock and starve labelling: at most `embed_sem`
    are ever in flight, and those pause the moment a job starts. They
    drain during idle time. Labelling never reads these, so deferring them
    has no effect on label quality; the stats card just populates lazily."""
    try:
        loop = asyncio.get_running_loop()
        sem = state.get("embed_sem")
        idle = state.get("gpu_idle")
        if sem is not None:
            await sem.acquire()
        try:
            if idle is not None:
                await idle.wait()
            async with state["gpu_lock"].background():
                await loop.run_in_executor(
                    _BG_IMAGE_EXECUTOR,
                    _compute_and_store_image_embedding,
                    project_id, import_id, raw_bytes,
                )
        finally:
            if sem is not None:
                sem.release()
    except Exception as e:
        print(f"[bg-embed] failed for {import_id}: {e}")


def _job_to_audit(job) -> None:
    """Persist a finished job to audit.db so the Terminal can show it after a
    backend restart. Only terminal states are passed in (done/failed/cancelled).

    NOTE: we use `job_kind` rather than `kind` for the audit blob field —
    the row-level `kind` column is already "job" and a kwarg of `kind`
    would clash with the first positional arg of `add_event`."""
    add_event(
        "job",
        id=job.id,
        job_kind=job.kind,
        project=job.project,
        user=job.user,
        status=job.status,
        elapsed_s=round(job.elapsed_s, 2),
        cost_pence=round(job.cost_pence, 5),
        n_images=job.n_images,
        error=job.error,
    )


state["jobs"].on_finish = _job_to_audit


def _load_vlm_into_state() -> None:
    """Load Qwen 2.5-VL 3B (fp16) into VRAM, OR skip entirely when
    a remote VLM worker is configured.

    When `VLM_WORKER_URL` is set in the environment, `vlm_classify` in
    `vlm_validate.py` routes calls over HTTP to that worker, so we
    have no reason to keep the model resident here — that's ~3 GB of
    VRAM freed on the main backend, which is the whole point of the
    worker split (matters on 12 GB cards where the rest of the stack
    barely fits).

    We deliberately don't ping the worker here at startup: a transient
    DNS / worker-restart blip during boot shouldn't keep the main
    backend from coming up. The first real `/classify_box` call will
    surface any worker problems via its own retry/log path.
    """
    worker_url = (os.environ.get("VLM_WORKER_URL") or "").strip()
    if worker_url:
        print(f"[server] VLM offloaded to worker at {worker_url} — local load skipped.")
        state["vlm_model"] = None
        state["vlm_processor"] = None
        return

    from vlm_validate import VLM_MODEL, load_vlm, set_vlm
    print(f"[server] loading VLM {VLM_MODEL} (fp16)...")
    model, processor = load_vlm(state["device"])
    set_vlm(model, processor)
    state["vlm_model"] = model
    state["vlm_processor"] = processor
    print(f"[server] VLM ready on {state['device']}.")


# Master kill-switch for the Label Cascade pipeline (DINOv2/SigLIP
# embeddings, /embeddings/* endpoints, post-auto-label scan,
# relabel-driven refresh). When False, the model never loads, the
# refresh / scan helpers no-op, and the endpoints return empty
# payloads — so the frontend's modal never opens and the
# infrastructure is dormant. Flip back to True to re-enable; all
# the wiring stays in place.
_EMBEDDINGS_ENABLED = False


def _load_dinov2_into_state() -> None:
    """Load the per-segmentation embedding model used by Label
    Cascade. Currently SigLIP 2 base (~370 MB on disk; ~190 MB on
    CUDA in fp16); override via the EMBED_MODEL env var. Cached to
    disk on first run, resident in VRAM after."""
    if not _EMBEDDINGS_ENABLED:
        print("[server] Label Cascade embeddings disabled — skipping model load.")
        return
    try:
        import embeddings as _emb
        _emb.load_dinov2(state["device"])
        _emb.warmup()
    except Exception as e:
        print(f"[server] embedding model load failed: {e} — Label Cascade disabled.")


import heapq as _heapq_module


class PriorityGPUGate:
    """Priority-aware async lock for GPU access — drop-in replacement
    for asyncio.Lock that wakes the highest-priority waiter first when
    the lock releases.

    Two priority levels:
      P_INTERACTIVE = 0   user-facing clicks (segment_box,
                          classify_box, click-to-detect, add-box).
                          Always preempts background work.
      P_JOB         = 10  background runners (augment_generate,
                          label_charlie, image processing). Share
                          the queue FIFO among themselves.

    `async with state["gpu_lock"]:` defaults to P_JOB so every
    existing call site still compiles + behaves the same. Interactive
    endpoints opt in with `async with state["gpu_lock"].interactive():`.
    """
    P_INTERACTIVE = 0
    P_JOB = 10
    # Lowest priority — fire-and-forget work the user isn't waiting on
    # (per-upload whole-image embeddings, stats backfill). Yields the
    # GPU to interactive clicks AND labelling jobs so a burst of
    # background embeddings can never stall a click-to-detect or a
    # labelling pass.
    P_BACKGROUND = 20

    def __init__(self) -> None:
        self._free = True
        self._waiters: "list[tuple[int, int, asyncio.Future]]" = []
        self._counter = 0

    async def _acquire(self, priority: int) -> None:
        # Fast path: gate free AND we wouldn't be jumping a higher-
        # priority queue. Note: with a healthy release loop the
        # waiters list is empty whenever _free is True, but the
        # extra check is cheap defence in depth.
        if self._free and (not self._waiters or priority <= self._waiters[0][0]):
            self._free = False
            return
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        self._counter += 1
        _heapq_module.heappush(self._waiters, (priority, self._counter, fut))
        try:
            await fut
        except asyncio.CancelledError:
            # Caller's request was cancelled while we were queued.
            # Two cases:
            #   • Future not yet set: nothing else has ownership;
            #     just drop our heap entry so the next release
            #     doesn't try to pop a stale slot. heapq doesn't
            #     have a cheap "remove arbitrary" op, so we leave
            #     the slot in place and rely on _release's "skip
            #     done futures" loop to step past us.
            #   • Future already set (we got the gate, now caller
            #     cancels): we must release or the gate is stuck.
            if fut.done() and not fut.cancelled():
                self._release()
            raise

    def _release(self) -> None:
        # Pop heap until we find a live (non-cancelled, not-yet-
        # set) waiter and transfer ownership to it. If we run out,
        # mark the gate free. This loop is what keeps the gate from
        # getting stuck behind a cancelled coroutine's stale heap
        # slot.
        while self._waiters:
            _, _, fut = _heapq_module.heappop(self._waiters)
            if not fut.done():
                fut.set_result(None)
                return
        self._free = True

    # Plain `async with gate:` → P_JOB. Keeps every existing
    # `async with state["gpu_lock"]:` site working without changes.
    async def __aenter__(self) -> "PriorityGPUGate":
        await self._acquire(self.P_JOB)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self._release()

    @asynccontextmanager
    async def at(self, priority: int):
        await self._acquire(priority)
        try:
            yield
        finally:
            self._release()

    @asynccontextmanager
    async def interactive(self):
        """Highest priority — click-to-detect, add-box, segment_box."""
        await self._acquire(self.P_INTERACTIVE)
        try:
            yield
        finally:
            self._release()

    @asynccontextmanager
    async def job(self):
        """Explicit P_JOB CM — same as `async with gate:`, just
        spelled out so call sites can be explicit about intent."""
        await self._acquire(self.P_JOB)
        try:
            yield
        finally:
            self._release()

    @asynccontextmanager
    async def background(self):
        """Lowest priority — fire-and-forget embeddings / backfill.
        Yields to interactive clicks and labelling jobs."""
        await self._acquire(self.P_BACKGROUND)
        try:
            yield
        finally:
            self._release()


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # GPU serialization lock for V2 pipelines. Multiple concurrent
    # /api/v2/references/process or /api/v2/imports/process requests
    # used to all enter `loop.run_in_executor(None, ...)` at once and
    # blow the 12 GB GPU's VRAM budget — each pipeline transiently
    # needs ~2 GB on top of the warm models, so 4 parallel calls
    # CUDA-OOM each other. Holding this lock across the executor
    # call serialises GPU work without queuing on the event loop
    # itself, so non-GPU endpoints stay responsive. Created here
    # so it binds to the running loop.
    state["gpu_lock"] = PriorityGPUGate()
    # Idle gate for background embeddings (see _pause_bg_embeddings).
    # Starts SET (idle) so embeddings run normally when no job is active.
    _gpu_idle = asyncio.Event()
    _gpu_idle.set()
    state["gpu_idle"] = _gpu_idle
    # Concurrency bound for background embeddings. A bulk import spawns
    # one embed task per image; without this they'd ALL clear the idle
    # gate at creation (idle during a pure import) and pile onto the GPU
    # lock, so a later job couldn't hold them back. The semaphore caps
    # how many are past the idle gate at once, so when a job starts the
    # few in-flight finish and the rest block on the (now-cleared) idle
    # gate instead of competing with the job.
    state["embed_sem"] = asyncio.Semaphore(2)

    state["device"] = device
    state["model"] = None       # GroundingDINO removed in the portable build
    state["segmenter"] = None   # SAM2 removed — SAM3 covers interactive segmentation
    state["nsfw"] = None        # NSFW gate removed

    # Model loading policy for the portable build: SAM3 + DINOv2, with the
    # small VLM tiebreak opt-in (VLM_ENABLED=1). Until the Metal/MPS device
    # phase lands, model loads only happen on CUDA; on other devices the
    # engine boots fully for dataset/annotation work and the ML endpoints
    # return 503. PK_DISABLE_MODELS=1 forces that mode anywhere.
    models_disabled = (
        os.environ.get("PK_DISABLE_MODELS", "").lower() in ("1", "true", "yes", "on")
        or device != "cuda"
    )
    if models_disabled:
        print(
            f"[server] models disabled (device={device}"
            + (", PK_DISABLE_MODELS set" if os.environ.get("PK_DISABLE_MODELS") else "")
            + ") — labelling endpoints 503; dataset/annotation APIs fully live."
        )

    if not models_disabled and os.environ.get("VLM_ENABLED", "").lower() in ("1", "true", "yes", "on"):
        await asyncio.get_event_loop().run_in_executor(None, _load_vlm_into_state)

    # DINOv2 for reference-crop embeddings (specific-dataset resolver +
    # near-duplicate scan). Loaded async so a slow first download doesn't
    # block startup.
    async def _load_v2_dino() -> None:
        try:
            import v2_dinov2
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: v2_dinov2.load(state["device"]),
            )
            await asyncio.get_event_loop().run_in_executor(None, v2_dinov2.warmup)
        except Exception as e:
            print(f"[server] v2 DINOv2 load failed: {e}")
    if not models_disabled:
        asyncio.create_task(_load_v2_dino())

    # Pipeline Charlie — SAM3 (gated on HF; needs HF_TOKEN). Loads in a
    # background task so a slow first download or transient HF outage
    # doesn't block server startup.
    async def _load_charlie() -> None:
        if os.environ.get("CHARLIE_DISABLED", "").lower() in ("1", "true", "yes", "on"):
            print("[server] CHARLIE_DISABLED set — pipeline_charlie endpoints will return 503.")
            state["charlie"] = None
            return
        try:
            import pipeline_charlie
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: pipeline_charlie.load_sam3(state["device"]),
            )
            state["charlie"] = pipeline_charlie
            print("[server] pipeline_charlie ready (SAM3).")
        except Exception as e:
            print(f"[server] pipeline_charlie load failed: {e} — /api/charlie/* will return 503.")
            state["charlie"] = None
    if not models_disabled:
        asyncio.create_task(_load_charlie())
    else:
        state["charlie"] = None

    state["jobs"].register_runner("label_charlie", _run_label_charlie_job)
    state["jobs"].register_runner("purge_label", _run_purge_label_job)
    state["jobs"].register_runner("augment_generate", _gpu_job_guarded(_run_augment_generate_job))

    # Restore Terminal job history across restarts. Past completed jobs
    # are stored as kind="job" audit events; pull the most recent batch
    # back into the JobManager so the Terminal lists them after a reboot.
    try:
        past = audit.list_events("job", limit=500)
        loaded = state["jobs"].hydrate_from_audit(past)
        if loaded:
            print(f"[jobs] hydrated {loaded} past jobs from audit.db")
    except Exception as e:
        print(f"[jobs] audit hydrate failed: {e}")

    state["jobs"].start_worker()

    # Worker watchdog: if any of the JobManager's worker tasks exits
    # (shouldn't happen — we hardened it — but be paranoid), revive the
    # missing slots. `start_worker(n=...)` is idempotent: live workers
    # stay, dead/missing slots get respawned.
    async def _worker_watchdog() -> None:
        while True:
            await asyncio.sleep(5)
            wts = state["jobs"].worker_tasks
            live = [t for t in wts if not t.done()]
            if len(live) < len(wts):
                print(f"[jobs] {len(wts) - len(live)} of {len(wts)} workers exited — restarting")
                state["jobs"].start_worker()
    state["_worker_watchdog"] = asyncio.create_task(_worker_watchdog())

    import workspace as _ws
    print(f"[server] workspace: {_ws.dir()}")
    print("[server] ready, job worker started.")
    _log_manifest_cache_capacity()
    yield

app = FastAPI(
    lifespan=lifespan,
    # Disable the auto-generated OpenAPI schema + Swagger / ReDoc UIs.
    # Public scanners hitting our Cloudflare tunnel were pulling the
    # full API surface map from /openapi.json — turning these off
    # makes /openapi.json, /docs, /redoc all 404. If you ever need
    # the docs locally, re-enable conditionally on an env flag.
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

# Localhost-only: the UI is either served by this process or a local dev
# server on :3000. No cloud origins in the portable build.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# ---- helpers ----

def slug(raw: str) -> str:
    s = raw.strip().lower().replace(" ", "-")
    s = re.sub(r"[^a-z0-9_-]+", "", s)
    return s or "project"


def project_dir(project_id: str) -> Path:
    """Resolve a dataset id to its workspace folder (store index)."""
    if project_id in RESERVED or not project_id or "/" in project_id or ".." in project_id:
        raise HTTPException(400, f"invalid project id: {project_id}")
    try:
        return store.dataset_dir(project_id)
    except KeyError:
        raise HTTPException(404, f"unknown project: {project_id}")


def manifest_path(project_id: str) -> Path:
    """Kept for existence checks; the actual manifest is split across
    dataset.json + annotations/ (see store.py)."""
    return project_dir(project_id) / "dataset.json"


# ─── Manifest RAM cache ──────────────────────────────────────────
# Every API request hits load_manifest at least once (often more —
# `/api/projects` walks every project, the per-request handlers
# read-modify-write, etc.). Reading + parsing the JSON off disk on
# each call adds up to a noticeable share of latency, especially on
# the cloudflare-tunnel-fronted host where a manifest read is
# competing with image uploads for IO.
#
# Cache the parsed dict in process memory and serve copies from it
# on subsequent reads. `save_manifest` writes through (disk first,
# then cache) so the cache never serves stale data to other handlers
# in the same process. Returning deep-copies on read keeps mutations
# in the caller's local copy until they explicitly save.
#
# Manifests are tiny (~10-100 KB each). Even a thousand projects
# fits in single-digit MB, well below the safe-bet headroom we
# logged at startup. Skipping eviction is fine until that scales by
# orders of magnitude.
# Built-payload cache. Stores the output of expensive endpoints
# (/overview, /annotations, /dataset-stats — both lite and full)
# keyed by (project_id, name) → (manifest_mtime, payload).
# A handler checks _PAYLOAD_CACHE before doing any work; on a hit
# (cached mtime >= current disk mtime) it returns the cached dict
# immediately. save_manifest doesn't need to touch this — the next
# read picks up the new mtime and misses the cache automatically.
_PAYLOAD_CACHE: dict[tuple[str, str], tuple[float, dict]] = {}
_PAYLOAD_CACHE_LOCK = __import__("threading").Lock()


def _payload_cache_get(project_id: str, name: str, disk_mtime: float) -> dict | None:
    if disk_mtime <= 0:
        return None
    with _PAYLOAD_CACHE_LOCK:
        entry = _PAYLOAD_CACHE.get((project_id, name))
    if entry is None:
        return None
    cached_mtime, payload = entry
    return payload if cached_mtime >= disk_mtime else None


def _payload_cache_get_swr(project_id: str, name: str, disk_mtime: float) -> tuple[dict | None, bool]:
    """Stale-while-revalidate variant. Returns (payload, is_fresh):
      - (None, False)  : no cached value at all
      - (dict, True)   : cache hit, payload is up-to-date
      - (dict, False)  : cache hit but stale (disk has advanced since
                         it was built). Caller should return it
                         immediately for latency AND kick a rebuild in
                         the background so the next read is fresh.

    Used by the read endpoints during an active labelling job, where
    the manifest is being rewritten after every image. Without SWR
    every poll would cache-miss and pay the full 3-5 s rebuild cost.
    """
    if disk_mtime <= 0:
        return None, False
    with _PAYLOAD_CACHE_LOCK:
        entry = _PAYLOAD_CACHE.get((project_id, name))
    if entry is None:
        return None, False
    cached_mtime, payload = entry
    return payload, cached_mtime >= disk_mtime


def _payload_cache_put(project_id: str, name: str, disk_mtime: float, payload: dict) -> None:
    if disk_mtime <= 0:
        return
    with _PAYLOAD_CACHE_LOCK:
        _PAYLOAD_CACHE[(project_id, name)] = (disk_mtime, payload)


# Tracks per-project payload rebuilds that are currently in flight so
# concurrent SWR requests don't all kick off duplicate rebuilds. The
# set is keyed by (project_id, name); a key is added when a rebuild
# starts and removed in the finally block.
_PAYLOAD_REVALIDATE_IN_FLIGHT: set[tuple[str, str]] = set()


async def _payload_revalidate(project_id: str, name: str, builder) -> None:
    """Background rebuild for stale-while-revalidate. `builder` is a
    zero-arg callable that returns the fresh payload. Runs once per
    (project, payload) pair regardless of how many SWR readers fire it."""
    key = (project_id, name)
    with _PAYLOAD_CACHE_LOCK:
        if key in _PAYLOAD_REVALIDATE_IN_FLIGHT:
            return
        _PAYLOAD_REVALIDATE_IN_FLIGHT.add(key)
    try:
        mtime = _manifest_disk_mtime(project_id)
        payload = await asyncio.to_thread(builder)
        _payload_cache_put(project_id, name, mtime, payload)
    except Exception as e:
        print(f"[payload-cache] revalidate failed for {project_id}/{name}: {e}")
    finally:
        with _PAYLOAD_CACHE_LOCK:
            _PAYLOAD_REVALIDATE_IN_FLIGHT.discard(key)


# ─── Read-path sidecars: instant-serve overview + stats ──────────────
# After every manifest write the backend pre-builds:
#   - overview_first100.json  (the first-batch /overview payload)
#   - dataset_stats_lite.json (already exists)
# Reads pull straight from these files — zero compute, zero manifest
# parse on the request path. The labelling job's per-image saves
# trigger an async rebuild that's deduped through
# _SIDECAR_REFRESH_IN_FLIGHT so rapid writes don't pile up rebuilds.

_FAST_OVERVIEW_LIMIT = 100  # must match FE's FIRST_BATCH
_INITIAL_PAYLOAD_LIMIT = 20  # matches FE's /initial first-paint slice
_SIDECAR_REFRESH_IN_FLIGHT: set[str] = set()
# Projects with sidecar refresh suppressed. Used by long-running jobs
# (label_charlie, augment_generate) that save the manifest many times
# in quick succession — without suppression each flush kicks a fresh
# overview + stats + initial + workspace-card rebuild, and on a big
# manifest those rebuilds can each take seconds, piling up faster
# than they drain. The job adds its project_id at start, removes it
# at end, and fires ONE sidecar refresh on completion to capture the
# final state.
_SIDECAR_REFRESH_SUPPRESSED: set[str] = set()


def _overview_sidecar_path(project_id: str) -> Path:
    return project_dir(project_id) / "overview_first100.json"


def _workspace_card_sidecar_path(project_id: str) -> Path:
    """Per-project sidecar for /api/projects' card payload — id, name,
    counters, tags, cover info. Built once on save_manifest so the
    list endpoint just reads N small JSON files instead of opening
    N multi-MB manifests on cold-start."""
    return project_dir(project_id) / "workspace_card_v1.json"


def _build_workspace_card_payload(project_id: str, manifest: dict) -> dict:
    """Compute the per-project card payload from a manifest. Same fields
    list_projects builds in its loop, minus the per-viewer transforms
    (likes/favourites count + isMine flags) which are folded in at
    request time."""
    proj_id = manifest.get("id") or project_id
    display_name = manifest.get("name") or project_id
    proj_owner = manifest.get("owner") or manifest.get("createdBy") or ""

    results = manifest.get("results") or []
    n_labelled = sum(1 for r in results if not r.get("pending"))
    n_unlabelled = sum(1 for r in results if r.get("pending"))
    n_images_v2: int | None = None
    if manifest.get("v2"):
        v2_imports = manifest.get("imports") or []
        n_images_v2 = len(v2_imports)

        def _v2_is_labelled(entry: dict) -> bool:
            if not isinstance(entry, dict):
                return False
            # User edits (click-to-detect, add-box, Clear all) are
            # the authoritative signal once editedBoxesSet flips
            # True — a labelled=False entry that the user manually
            # drew boxes on IS labelled; a labelled=True entry the
            # user emptied via Clear all is NOT.
            if entry.get("editedBoxesSet"):
                edited = entry.get("editedBoxes")
                return isinstance(edited, list) and len(edited) > 0
            flag = entry.get("labelled")
            if flag is True:
                return True
            if flag is False:
                return False
            return bool(entry.get("detections"))
        n_labelled = sum(1 for e in v2_imports if _v2_is_labelled(e))
        n_unlabelled = max(0, n_images_v2 - n_labelled)

    cover = manifest.get("cover")
    v2_refs = manifest.get("references") or []
    v2_imps = manifest.get("imports") or []
    cover_subdir: str | None = None
    # A user-uploaded cover wins over any picked/random one and is served via
    # the cover_thumb endpoint, so we only need a truthy thumbnail marker here
    # (no subdir — it lives at the project root, not in references/imports).
    if manifest.get("cover_uploaded"):
        cover = "cover_upload.jpg"
        cover_subdir = None
    elif cover:
        if any(r.get("filename") == cover for r in v2_refs):
            cover_subdir = "references"
        elif any(i.get("filename") == cover for i in v2_imps):
            cover_subdir = "imports"
        elif any(r.get("image") == cover for r in results):
            cover_subdir = "imports"
        else:
            cover = None
    if not cover:
        import random as _rnd
        seed = _rnd.Random(proj_id)
        ref_files = [r.get("filename") for r in v2_refs if r.get("filename")]
        imp_files = [i.get("filename") for i in v2_imps if i.get("filename")]
        v1_files = [r.get("image") for r in results if r.get("image")]
        if ref_files:
            cover = seed.choice(ref_files)
            cover_subdir = "references"
        elif imp_files:
            cover = seed.choice(imp_files)
            cover_subdir = "imports"
        elif v1_files:
            cover = seed.choice(v1_files)
            cover_subdir = "imports"
        else:
            cover = None
            cover_subdir = None

    n_images_resolved = n_images_v2 if n_images_v2 is not None else len(results)
    return {
        "id": proj_id,
        "name": display_name,
        "owner": proj_owner,
        "createdAt": manifest.get("createdAt"),
        "updatedAt": manifest.get("updatedAt"),
        "n_images": n_images_resolved,
        "n_labelled": n_labelled,
        "n_unlabelled": n_unlabelled,
        "n_references": len(v2_refs) if manifest.get("v2") else 0,
        "tags": collect_tags(manifest),
        "label_aliases": dict(manifest.get("label_aliases") or {}),
        "labelColours": dict(manifest.get("labelColours") or {}),
        "thumbnail": cover,
        "_cover_subdir": cover_subdir,
        "hasModel": bool(manifest.get("hasModel", False)),
        "createdBy": proj_owner,
        "likedBy": list(manifest.get("likedBy") or []),
        "favouritedBy": list(manifest.get("favouritedBy") or []),
        "certified": bool(manifest.get("certified", False)),
        "private": bool(manifest.get("private", False)),
        "v2": bool(manifest.get("v2", False)),
        # Derived ("child") link so the workspace card can show a derived badge.
        "derived": ({"parentProjectId": (manifest.get("derived") or {}).get("parentProjectId"),
                     "parentName": (manifest.get("derived") or {}).get("parentName")}
                    if manifest.get("derived") else None),
        "cover_blurhash": manifest.get("cover_blurhash"),
        "dataset_health": _compute_dataset_health(manifest),
    }


def _write_workspace_card_sidecar(project_id: str) -> None:
    try:
        m = load_manifest(project_id, False)
        if not m:
            return
        payload = _build_workspace_card_payload(project_id, m)
        p = _workspace_card_sidecar_path(project_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        if _orjson is not None:
            tmp.write_bytes(_orjson.dumps(payload))
        else:
            tmp.write_text(json.dumps(payload))
        tmp.replace(p)
    except Exception as e:
        print(f"[workspace-card-sidecar] write failed for {project_id}: {e}")


def _read_workspace_card_sidecar(project_id: str) -> dict | None:
    p = _workspace_card_sidecar_path(project_id)
    if not p.exists():
        return None
    try:
        data = _json_loads(p.read_bytes())
        if isinstance(data, dict):
            return data
    except Exception as e:
        print(f"[workspace-card-sidecar] read failed for {project_id}: {e}")
    return None


_WORKSPACE_CARD_REFRESH_IN_FLIGHT: set[str] = set()


def _kick_workspace_card_refresh(project_id: str) -> None:
    """Lazy-build a workspace_card sidecar for projects that haven't
    been saved since the sidecar mechanism was introduced. Called
    from list_projects' slow-path so the NEXT request hits the fast
    path. Dedup'd so 33 simultaneous slow-path hits during a single
    /api/projects request only enqueue one rebuild per project."""
    if project_id in _WORKSPACE_CARD_REFRESH_IN_FLIGHT:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    _WORKSPACE_CARD_REFRESH_IN_FLIGHT.add(project_id)

    async def _run():
        try:
            await asyncio.to_thread(_write_workspace_card_sidecar, project_id)
        except Exception as e:
            print(f"[workspace-card-refresh] failed for {project_id}: {e}")
        finally:
            _WORKSPACE_CARD_REFRESH_IN_FLIGHT.discard(project_id)
    loop.create_task(_run())


def _initial_sidecar_path(project_id: str) -> Path:
    """The single-round-trip first-paint payload. Combines project
    meta + first-N imports + lite dataset stats into one document so
    the FE can render the gallery + chips + stats card in one fetch."""
    return project_dir(project_id) / "initial_first20.json"


def _write_overview_sidecar(project_id: str) -> None:
    """Compute the first-100 /overview payload and persist atomically.
    Called off the request path via _kick_sidecar_refresh so the
    response itself never waits on this."""
    try:
        m = load_manifest(project_id, False)
        if not m:
            return
        payload = _build_overview_payload(
            project_id, m, imports_limit=_FAST_OVERVIEW_LIMIT, imports_offset=0,
        )
        p = _overview_sidecar_path(project_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        if _orjson is not None:
            tmp.write_bytes(_orjson.dumps(payload))
        else:
            tmp.write_text(json.dumps(payload))
        tmp.replace(p)
    except Exception as e:
        print(f"[overview-sidecar] write failed for {project_id}: {e}")


def _read_overview_sidecar(project_id: str) -> dict | None:
    """Return the persisted first-100 overview, or None when missing /
    unreadable. Allow-stale — the file is good enough until the next
    write replaces it. /overview compute path is the fallback."""
    p = _overview_sidecar_path(project_id)
    if not p.exists():
        return None
    try:
        data = _json_loads(p.read_bytes())
        if isinstance(data, dict):
            return data
    except Exception as e:
        print(f"[overview-sidecar] read failed for {project_id}: {e}")
    return None


def _write_initial_sidecar(project_id: str) -> None:
    """Combine the just-written overview + dataset-stats sidecars into
    one initial-paint document. Cheap because it's just two file reads
    plus a slice — no manifest reload, no recompute. Called from
    _kick_sidecar_refresh AFTER the two upstream sidecars have been
    rewritten."""
    try:
        # Pull the freshly-written overview sidecar (project meta +
        # first 100 imports + refs metadata + dataset_health). Slice
        # imports down to the initial-paint count so the FE doesn't
        # download 5x what it needs for the first viewport.
        overview = _read_overview_sidecar(project_id)
        if overview is None:
            return
        imports = (overview.get("imports") or [])[:_INITIAL_PAYLOAD_LIMIT]
        # Stats-lite sidecar is read fresh from disk — same path the
        # /dataset-stats?lite=true endpoint uses. If it doesn't exist
        # yet (first-ever build), stats fall through to None and the
        # FE re-fetches.
        stats_payload: dict | None = None
        stats_path = _stats_sidecar_path(project_id, True)
        if stats_path.exists():
            try:
                data = _json_loads(stats_path.read_bytes())
                if isinstance(data, dict):
                    stats_payload = data
            except Exception:
                stats_payload = None
        payload = {
            # Project meta — flat, mirrors what /overview returns at
            # the top level so the FE's destructuring keeps working.
            "id": overview.get("id"),
            "name": overview.get("name"),
            "prompt": overview.get("prompt"),
            "tags": overview.get("tags") or [],
            "labelsLastRun": overview.get("labelsLastRun"),
            "settingsLastRun": overview.get("settingsLastRun"),
            "label_aliases": overview.get("label_aliases") or {},
            "labelColours": overview.get("labelColours") or {},
            "cover": overview.get("cover"),
            "cover_blurhash": overview.get("cover_blurhash"),
            "v2": overview.get("v2"),
            "createdAt": overview.get("createdAt"),
            "updatedAt": overview.get("updatedAt"),
            "thresholds": overview.get("thresholds"),
            "vlm_action": overview.get("vlm_action"),
            "synonyms_enabled": overview.get("synonyms_enabled"),
            "private": overview.get("private"),
            "max_input_size": overview.get("max_input_size"),
            # Cached general/specific verdict so the hero badge paints with the
            # first frame instead of after a separate /dataset-type fetch.
            "dataset_type": overview.get("dataset_type"),
            "hasModel": overview.get("hasModel"),
            "owner": overview.get("owner"),
            "createdBy": overview.get("createdBy"),
            "dataset_health": overview.get("dataset_health"),
            # First-paint payload — gallery covers + chip rails + box
            # counts + reference metadata.
            "references": overview.get("references") or [],
            "imports": imports,
            "imports_total": overview.get("imports_total") or 0,
            # Stats card data — counts + label distribution + 3-factor
            # health score. The variation plot (full payload) is fetched
            # later via /dataset-stats when the user expands the card.
            "stats": stats_payload,
        }
        p = _initial_sidecar_path(project_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        if _orjson is not None:
            tmp.write_bytes(_orjson.dumps(payload))
        else:
            tmp.write_text(json.dumps(payload))
        tmp.replace(p)
    except Exception as e:
        print(f"[initial-sidecar] write failed for {project_id}: {e}")


def _read_initial_sidecar(project_id: str) -> dict | None:
    """Return the persisted first-paint document, or None if missing.
    Allow-stale — by the time the user sees this on the next nav, the
    next manifest write will have rewritten it."""
    p = _initial_sidecar_path(project_id)
    if not p.exists():
        return None
    try:
        data = _json_loads(p.read_bytes())
        if isinstance(data, dict):
            return data
    except Exception as e:
        print(f"[initial-sidecar] read failed for {project_id}: {e}")
    return None


def _kick_sidecar_refresh(project_id: str) -> None:
    """Schedule async rebuild of every read-path sidecar for this
    project. Deduped — if a refresh is already in flight, subsequent
    calls are no-ops until it finishes (in which case the new save
    is reflected in the next refresh anyway because save_manifest
    bumps disk mtime which the rebuild reads)."""
    # Suppressed during long-running jobs (label_charlie's per-flush
    # save loop is the worst offender — on a big manifest each
    # rebuild can take seconds, faster than the job is saving). The
    # job will fire one final refresh on completion to capture the
    # end state.
    if project_id in _SIDECAR_REFRESH_SUPPRESSED:
        return
    if project_id in _SIDECAR_REFRESH_IN_FLIGHT:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # save_manifest can run outside an event loop (CLI scripts,
        # tests). Skip silently — sidecars will be (re)built lazily
        # on the next read.
        return
    _SIDECAR_REFRESH_IN_FLIGHT.add(project_id)

    async def _run():
        try:
            # Tiny debounce so a burst of save_manifest calls (e.g. a
            # labelling job iterating its way through a dataset)
            # coalesces into one rebuild rather than firing on each.
            await asyncio.sleep(0.15)
            # Run the rebuilds on the dedicated bg pool, NOT asyncio's
            # default executor. These are multi-second manifest reads;
            # on the default pool they starve request-path work (the
            # FE's /annotations + /overview builds, image serving) —
            # which is what left freshly-labelled tiles showing
            # "Unlabelled" for 10-15s after a job completed while the
            # default pool churned through the completion-time sidecar
            # rebuild. Off the default pool, the FE's post-completion
            # syncAnnotations build runs immediately.
            await loop.run_in_executor(_BG_IMAGE_EXECUTOR, _write_overview_sidecar, project_id)
            await loop.run_in_executor(_BG_IMAGE_EXECUTOR, _persist_dataset_stats, project_id, True)
            # Combined first-paint sidecar — reads the two upstream
            # files (already on disk by now) and merges. Must run
            # AFTER both above so it captures their fresh contents.
            await loop.run_in_executor(_BG_IMAGE_EXECUTOR, _write_initial_sidecar, project_id)
            # Workspace card sidecar — fields /api/projects' loop used
            # to compute by loading the full multi-MB manifest per
            # project. Bake them once here so the list endpoint can
            # serve N projects from N small JSON reads instead.
            await loop.run_in_executor(_BG_IMAGE_EXECUTOR, _write_workspace_card_sidecar, project_id)
        except Exception as e:
            print(f"[sidecar-refresh] failed for {project_id}: {e}")
        finally:
            _SIDECAR_REFRESH_IN_FLIGHT.discard(project_id)

    loop.create_task(_run())


_MANIFEST_CACHE: dict[str, dict] = {}
# Disk mtime when the cache entry was populated. Used to invalidate
# the cache when the manifest file has been written by ANY process
# (a different gunicorn worker, an external editor, etc.) — without
# this check, the cache became a "labels / refs lost on reopen" bug
# when the backend ran with multiple workers, since each worker had
# a private cache that never noticed sibling-worker writes.
_MANIFEST_CACHE_MTIME: dict[str, float] = {}
# Secondary index: project_id → {import_id: entry_dict}. Built and
# kept in sync (under _MANIFEST_CACHE_LOCK) whenever the manifest
# cache is seeded or invalidated. Turns the per-image annotation
# endpoint from an O(n) linear scan into an O(1) dict lookup.
_MANIFEST_IMPORT_INDEX: dict[str, dict[str, dict]] = {}
_MANIFEST_CACHE_LOCK = __import__("threading").Lock()
_manifest_cache_hits = 0
_manifest_cache_misses = 0
_manifest_cache_stale = 0

# Tombstones for recently-deleted projects. delete_project records the
# id here (under _MANIFEST_CACHE_LOCK) BEFORE removing the folder, and
# save_manifest refuses to write for a tombstoned id. Without this, a
# write that races the delete — a debounced label-metadata autosave, a
# like/favourite, or an in-flight job/sidecar flush — would re-run
# save_manifest's `mkdir(parents=True)` and recreate the project
# directory + manifest, resurrecting the deleted project (which then
# reappears in the workspace, sometimes alongside a stale card as a
# duplicate). Bounded FIFO: ids are random UUIDs and never reused, so
# evicting the oldest entry once the cap is hit is safe — any write
# racing that delete is long finished by then.
_DELETED_PROJECT_IDS: "OrderedDict[str, float]" = __import__("collections").OrderedDict()
_DELETED_PROJECT_IDS_MAX = 4096


def _mark_project_deleted(project_id: str) -> None:
    with _MANIFEST_CACHE_LOCK:
        _DELETED_PROJECT_IDS[project_id] = time.time()
        _DELETED_PROJECT_IDS.move_to_end(project_id)
        while len(_DELETED_PROJECT_IDS) > _DELETED_PROJECT_IDS_MAX:
            _DELETED_PROJECT_IDS.popitem(last=False)


def _unmark_project_deleted(project_id: str) -> None:
    """Undo a tombstone — used when a delete fails partway so a project
    that's still on disk doesn't get permanently frozen against writes."""
    with _MANIFEST_CACHE_LOCK:
        _DELETED_PROJECT_IDS.pop(project_id, None)


def _is_project_deleted(project_id: str) -> bool:
    with _MANIFEST_CACHE_LOCK:
        return project_id in _DELETED_PROJECT_IDS


def _manifest_disk_mtime(project_id: str) -> float:
    """Change stamp in seconds, or 0 if missing. store.save() always
    rewrites dataset.json, so its mtime moves on every persisted change —
    the cache validity marker survives the manifest split unchanged."""
    return store.manifest_stamp(project_id)


def load_manifest(project_id: str, copy: bool = True) -> dict:
    """Read the on-disk manifest, going through the in-memory cache when
    fresh. The deep-copy on warm-cache hits is the biggest single cost
    for big projects (~200-500 ms on a 30 MB nested dict) — callers that
    only READ the manifest can pass copy=False and skip it entirely. The
    cache lifetime is the response, not the process, so racing mutations
    aren't a concern in the read-only paths.

    Callers that mutate the manifest MUST keep the default copy=True so
    their edits land in their local copy, not the shared cache. The
    save_manifest path re-seeds the cache with a fresh copy of the
    persisted dict after each write, so concurrent reads on the same
    project still converge to the latest disk truth without the writer
    leaking partial state into the cache.

    A handful of read-only sites (e.g. _kick_blurhash_backfill) do
    monotonically fill in missing fields on the shared dict; that's
    deliberate (the writeback becomes a side-effect of the read) and is
    safe because the writes are idempotent + concurrent-write-tolerant.
    """
    global _manifest_cache_hits, _manifest_cache_misses, _manifest_cache_stale
    import copy as _copy

    disk_mtime = _manifest_disk_mtime(project_id)

    with _MANIFEST_CACHE_LOCK:
        cached = _MANIFEST_CACHE.get(project_id)
        cached_mtime = _MANIFEST_CACHE_MTIME.get(project_id, 0.0)
        if cached is not None and disk_mtime > 0 and disk_mtime <= cached_mtime:
            _manifest_cache_hits += 1
            return _copy.deepcopy(cached) if copy else cached
        if cached is not None and disk_mtime > cached_mtime:
            # Cache is stale (disk was written by someone else —
            # another worker, an external edit, etc.). Drop and
            # re-read.
            _manifest_cache_stale += 1
        _manifest_cache_misses += 1

    try:
        data = store.load(project_id)
    except Exception as e:
        print(f"[manifest-cache] read failed for {project_id}: {e}")
        return {}
    if data is None:
        return {}
    with _MANIFEST_CACHE_LOCK:
        _MANIFEST_CACHE[project_id] = data if not copy else _copy.deepcopy(data)
        _MANIFEST_CACHE_MTIME[project_id] = disk_mtime
        _MANIFEST_IMPORT_INDEX[project_id] = {
            e["id"]: e
            for e in (data.get("imports") or [])
            if isinstance(e, dict) and "id" in e
        }
    # Caller that asked for a copy gets a fresh deepcopy of what we
    # just parsed. The cache also stores a fresh copy in that case
    # (above) so subsequent reads aren't aliased to the caller's
    # dict. When copy=False we share the parsed dict directly — the
    # cache and the caller see the same reference.
    return _copy.deepcopy(data) if copy else data


def save_manifest(project_id: str, manifest: dict, *, cache_by_ref: bool = False) -> None:
    import copy as _copy
    # Refuse to resurrect a deleted project. A write that races
    # delete_project (debounced label-metadata autosave, like/favourite,
    # or an in-flight job/sidecar flush) would otherwise recreate the
    # folder + manifest below via mkdir, bringing the project back.
    if _is_project_deleted(project_id):
        return
    # New datasets get their workspace folder on first save — creation
    # flows (create, duplicate, derive) don't need a separate step.
    if not store.dataset_exists(project_id):
        store.create_dataset_dir(
            project_id,
            str(manifest.get("name") or project_id),
            str(manifest.get("container_id") or "") or None,
        )
    # One-time slim: older label_charlie runs persisted the raw DINOv2
    # (1024-d) + SigLIP (768-d) embedding vectors on every TEST-IMAGE
    # detection. Those vectors are never read back for imports — the
    # resolver re-embeds fresh crops on each (re)label — so they were
    # pure bloat that made big manifests ~10× larger (≈200 MB at 4000
    # imgs). That bloat is what made every whole-manifest write O(n²)
    # past ~400 labels AND made the cached + parsed giant dict drag
    # the load of EVERY other project. New writes strip at the source
    # (see _flush_label_updates); this heals any pre-existing manifest
    # the first time it's saved, then sets a flag so the walk is
    # skipped on every later save (keeping the hot per-edit path free).
    # References (manifest["references"]) keep their embeddings — the
    # resolve step genuinely reads those back.
    if not manifest.get("_imports_embeddings_stripped"):
        for _e in manifest.get("imports") or []:
            if not isinstance(_e, dict):
                continue
            for _d in _e.get("detections") or []:
                if isinstance(_d, dict):
                    for _k in _EMBEDDING_FIELDS:
                        if _k in _d:
                            del _d[_k]
        manifest["_imports_embeddings_stripped"] = True
    # Persist through the split store: dataset.json + per-image
    # annotations/, atomic writes, only-changed annotation files.
    store.save(project_id, manifest)
    # Stat the freshly-replaced file so the cache mtime matches what
    # any other worker will see when they read disk.
    disk_mtime = _manifest_disk_mtime(project_id)
    with _MANIFEST_CACHE_LOCK:
        # Default seeds the cache with a deepcopy so future mutations
        # of the caller's `manifest` variable don't bleed into the
        # cache. `cache_by_ref=True` skips the deepcopy and is safe
        # ONLY when the caller is the last writer of the dict — e.g.
        # a label_charlie flush that's inside the per-project write
        # lock and discards `mm` immediately after save. Skipping
        # the deepcopy saves 200-500ms per call on a 30MB manifest,
        # which adds up to ~30s over a 1000-image labelling job.
        _MANIFEST_CACHE[project_id] = manifest if cache_by_ref else _copy.deepcopy(manifest)
        _MANIFEST_CACHE_MTIME[project_id] = disk_mtime
        _MANIFEST_IMPORT_INDEX[project_id] = {
            e["id"]: e
            for e in (manifest.get("imports") or [])
            if isinstance(e, dict) and "id" in e
        }
    # Kick async sidecar refresh so the next /overview + /dataset-stats
    # reads serve straight from disk with zero compute. Deduped so
    # rapid manifest writes (e.g. a labelling job that saves after
    # every image) don't pile up dozens of parallel rebuilds.
    _kick_sidecar_refresh(project_id)
    # Live one-way sync: if this project is a PARENT of derived (child) projects,
    # re-derive them after edits settle. O(1) + never raises into this hot path.
    _kick_child_resync(project_id)


def invalidate_manifest_cache(project_id: str) -> None:
    """Drop a project from the cache. Used on delete."""
    with _MANIFEST_CACHE_LOCK:
        _MANIFEST_CACHE.pop(project_id, None)
        _MANIFEST_CACHE_MTIME.pop(project_id, None)
        _MANIFEST_IMPORT_INDEX.pop(project_id, None)


# ── derived ("child") projects: one-way parent → child crop sync ──────────
# A child project's dataset is per-detection crops of a parent (see gd/derived.py).
# We keep an in-memory reverse index (parent_id -> child_ids) so the save hook is
# O(1); it's built once in the background on first use. Live re-sync is debounced
# so a burst of parent saves (e.g. a labelling job) coalesces into one pass.
try:  # never let the derived feature break the server's boot
    import derived as _derived_mod  # noqa: E402  (gd/derived.py — no server import, no cycle)
except Exception as _derive_imp_err:
    _derived_mod = None
    print(f"[derived] feature disabled (import failed): {_derive_imp_err}", flush=True)

_DERIVED_INDEX: dict[str, set[str]] = {}
_DERIVED_INDEX_BUILT = False
_DERIVED_BUILDING = False
_DERIVED_LOCK = _threading.Lock()
_DERIVED_TIMERS: dict[str, _threading.Timer] = {}


def _iter_project_ids() -> list[str]:
    try:
        return store.iter_dataset_ids()
    except Exception:
        return []


def _build_derived_index() -> None:
    global _DERIVED_INDEX_BUILT, _DERIVED_BUILDING
    idx: dict[str, set[str]] = {}
    for pid in _iter_project_ids():
        try:
            par = (load_manifest(pid, copy=False).get("derived") or {}).get("parentProjectId")
        except Exception:
            continue
        if par:
            idx.setdefault(par, set()).add(pid)
    with _DERIVED_LOCK:
        _DERIVED_INDEX.clear(); _DERIVED_INDEX.update(idx)
        _DERIVED_INDEX_BUILT = True
        _DERIVED_BUILDING = False


def _children_of(parent_id: str) -> set[str]:
    return set(_DERIVED_INDEX.get(parent_id) or ())


def _register_child(parent_id: str, child_id: str) -> None:
    # Add to the in-memory index. The child manifest is already saved, so a
    # later full build (lazy) still finds it; we don't flip the BUILT flag here
    # (only a full scan may, so other parents' links aren't lost).
    with _DERIVED_LOCK:
        _DERIVED_INDEX.setdefault(parent_id, set()).add(child_id)


def _ensure_derived_index() -> None:
    if not _DERIVED_INDEX_BUILT:
        _build_derived_index()


def _unregister_child(child_id: str) -> None:
    with _DERIVED_LOCK:
        for s in _DERIVED_INDEX.values():
            s.discard(child_id)


def resync_child(child_id: str) -> bool:
    """Re-derive ONE child from its parent's current state. Reads parent, writes
    child only. Returns True on success."""
    if _derived_mod is None:
        return False
    child = load_manifest(child_id)
    parent_id = (child.get("derived") or {}).get("parentProjectId")
    if not parent_id:
        return False
    try:
        parent = load_manifest(parent_id, copy=False)
    except Exception:
        return False
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _derived_mod.resync(child, parent,
                        project_dir(parent_id) / "images",
                        project_dir(child_id) / "images", now_iso=now)
    child["updatedAt"] = now
    save_manifest(child_id, child)
    return True


def resync_children(parent_id: str) -> None:
    for cid in _children_of(parent_id):
        try:
            resync_child(cid)
        except Exception as e:
            print(f"[derived] resync of child {cid} failed: {e}", flush=True)


def _kick_child_resync(parent_id: str) -> None:
    """Debounced background re-sync of a parent's children (~3s after the last
    edit). Fast + swallow-all so it never disturbs save_manifest."""
    global _DERIVED_BUILDING
    try:
        if not _DERIVED_INDEX_BUILT:
            with _DERIVED_LOCK:
                if not _DERIVED_INDEX_BUILT and not _DERIVED_BUILDING:
                    _DERIVED_BUILDING = True
                    _threading.Thread(target=_build_derived_index, daemon=True).start()
            return  # index warming up; live sync engages once it's ready
        if parent_id not in _DERIVED_INDEX or not _DERIVED_INDEX[parent_id]:
            return
        with _DERIVED_LOCK:
            old = _DERIVED_TIMERS.get(parent_id)
            if old:
                old.cancel()
            t = _threading.Timer(3.0, lambda pid=parent_id: resync_children(pid))
            t.daemon = True
            _DERIVED_TIMERS[parent_id] = t
            t.start()
    except Exception:
        pass


# Per-project asyncio lock for the manifest read-modify-write cycle.
# Without this, concurrent uploads to the same project did
# load_manifest → mutate → save_manifest in parallel. Each request
# saw the manifest at upload-start time, appended its own ref, and
# the later save_manifest overwrote the earlier one (most visibly:
# uploading 2 references back-to-back ended with manifest holding
# only the second one). The GPU lock serialises inference but says
# nothing about manifest writes. Use one lock per project rather
# than a global one so projects don't queue behind each other.
_MANIFEST_WRITE_LOCKS: dict[str, asyncio.Lock] = {}
_MANIFEST_WRITE_LOCKS_GUARD = asyncio.Lock()


async def _manifest_write_lock(project_id: str) -> asyncio.Lock:
    """Lazily-allocate per-project Lock. The guard lock makes the
    "is this the first call for this project" check atomic so two
    simultaneous first-callers don't each instantiate their own
    Lock and end up racing each other."""
    async with _MANIFEST_WRITE_LOCKS_GUARD:
        lk = _MANIFEST_WRITE_LOCKS.get(project_id)
        if lk is None:
            lk = asyncio.Lock()
            _MANIFEST_WRITE_LOCKS[project_id] = lk
        return lk


def _find_import_by_idempotency_key(project_id: str, key: str) -> dict | None:
    """Scan the project manifest's imports for a previously-persisted
    entry that carries the given idempotency_key. Returns the
    /imports/raw response shape so the early-probe path in
    v2_upload_import_raw can short-circuit on a retry without
    redoing NSFW + PIL + disk write. Returns None when no match.
    """
    if not key:
        return None
    manifest = load_manifest(project_id, copy=False) or {}
    for entry in manifest.get("imports", []) or []:
        if entry.get("idempotencyKey") == key:
            return {
                "import_id": entry.get("id"),
                "filename": entry.get("filename"),
                "width": int(entry.get("width") or 0),
                "height": int(entry.get("height") or 0),
                "blurhash": entry.get("blurhash"),
                "labelled": bool(entry.get("labelled", False)),
            }
    return None


def _log_manifest_cache_capacity() -> None:
    """One-shot at startup. Prints system RAM headroom + an estimate
    of how many manifests the cache can hold under that headroom.
    Also sizes the served-image RAM cache (LRU bytes-bounded)."""
    try:
        import psutil  # type: ignore
        vm = psutil.virtual_memory()
        total_gb = vm.total / 1e9
        avail_gb = vm.available / 1e9
        cap = int(min(1.0, max(0.1, avail_gb * 0.05)) * 1e9 / 50_000)
        # Image cache budget: 25% of available RAM, capped at 8 GB.
        # On a 16 GB box that's ~4 GB, on a 64 GB box ~8 GB.
        # Sufficient for ~thousands of typical 200 KB JPEGs in RAM
        # so reopens / workspace polls hit the cache nearly always.
        img_budget_bytes = int(min(8.0, max(0.5, avail_gb * 0.25)) * 1e9)
        _set_image_cache_budget(img_budget_bytes)
        print(
            f"[manifest-cache] system RAM: {total_gb:.1f} GB total, "
            f"{avail_gb:.1f} GB available — cap-free cache (target ≤ {cap} manifests)"
        )
        print(
            f"[image-cache] budget: {img_budget_bytes / 1e9:.1f} GB "
            f"(LRU eviction over budget)"
        )
    except Exception:
        # psutil missing or any other error — just don't log the
        # capacity hint. Caches still work; image cache stays at
        # its default 0-byte budget (effectively disabled until
        # explicitly set).
        print("[manifest-cache] enabled (psutil unavailable, skipping capacity probe)")
        # Conservative fallback so the image cache still works
        # without the RAM probe.
        _set_image_cache_budget(1_000_000_000)
        print("[image-cache] budget: 1.0 GB (default — psutil unavailable)")


# Now that load_manifest is defined we can finish wiring the
# project-ownership dependency. `require_project_owner` reads the
# manifest at request time so a deletion / rename of the project
# between token issue and request lands as a 404 not a stale 200.
require_project_owner = _require_project_owner_factory(load_manifest)
# Read-access gate. Public projects readable anonymously; private
# projects readable only by their owner (anyone else gets 404 so the
# UUID's existence can't be probed).
require_project_read_access = _require_project_read_access_factory(load_manifest)
# Manage gate (rename / cover / privacy / members / delete). Standalone dataset
# -> owner; dataset in a Project container -> the container owner only. Applied
# to settings/delete routes (editors get write via require_project_owner but not
# manage).
require_project_manage = _require_project_manage_factory(load_manifest)
# DESTROY a dataset: strictly the dataset's own creator (manifest owner), never
# a Project editor and never even the Project owner.
require_dataset_creator = _require_dataset_creator_factory(load_manifest)



def empty_manifest(name: str, owner: str = "", project_id: str | None = None) -> dict:
    return {
        "id": project_id or _uuid.uuid4().hex,
        "name": name,
        "owner": owner,
        "createdBy": owner,
        "createdAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "updatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "prompt": "",
        "tags": [],
        # Match the frontend's "Normal" preset so new projects open
        # cleanly on Normal mode in the picker — the old (0.25, 0.25,
        # 0.5) defaults don't line up with any named mode and so showed
        # up as a Custom pill on first load, which surprised users.
        "thresholds": {"box": 0.05, "text": 0.15, "nms": 0.7},
        "results": [],
        "verdicts": {},
        "editedBoxes": {},
        "cover": None,
        "hasModel": False,
        "likedBy": [],
        "favouritedBy": [],
        # When true the project is hidden from the public Projects feed.
        # Owner can always see and edit their own private projects.
        "private": False,
        # Per-label colour overrides set in Project settings. Keyed by
        # canonical-lower label; value is a #rrggbb hex string. Missing
        # entries fall back to the stable hash-based palette on the FE.
        "labelColours": {},
    }


def _ensure_random_cover(manifest: dict) -> bool:
    """If no cover is set, pick a random image from `results`.

    Called from the upload paths after new images land so a freshly
    created project gets a varied thumbnail in the Projects grid
    instead of always defaulting to the first uploaded image.
    Returns True when the manifest was modified so the caller knows
    to re-save.
    """
    if manifest.get("cover"):
        return False
    results = manifest.get("results", []) or []
    candidates = [r.get("image") for r in results if r.get("image")]
    if not candidates:
        return False
    import random
    manifest["cover"] = random.choice(candidates)
    return True


def parse_phrase(label):
    if "(" in label and label.endswith(")"):
        text, score = label.rsplit("(", 1)
        try:
            return text.strip(), float(score.rstrip(")"))
        except ValueError:
            pass
    return label, None


def image_size(p: Path) -> dict[str, int]:
    with PILImage.open(p) as img:
        return {"width": img.width, "height": img.height}


# Preference-model code (review checkpoint, logistic-regression trainer,
# CLIP-augmented features) lived here. Removed — VLM validation alone
# now decides accept/reject, and auto-label runs through to completion.


# ---------------------------------------------------------------------------
# Tag synonym expansion (auto prompt-engineering for GroundingDINO).
# ---------------------------------------------------------------------------
# GD recall is meaningfully better when the user's tag is fed alongside
# a few synonyms / near-synonyms ("potholes" + "road damage" + "crack").
# We ask Claude Haiku for the variants once per (project, tag), cache
# them on disk, and on each inference run pass GD the merged list.
# Detected variants are mapped back to the user's canonical tag in
# post so the manifest never sees the synthetic labels.

def _charlie_embed_detections(
    pil: "PILImage.Image",
    detections: list[dict],
) -> list:
    """Embed each Charlie detection's masked (inpainted) crop with DINOv2
    (and SigLIP when loaded), mutating the dicts in place to populate
    `embedding`/`embed_version` and `siglip_embedding`/`siglip_version`.

    Returns the per-detection squares (parallel to `detections`) so the
    caller can run patch-token encoding without rebuilding from polygons.
    Empty list when DINOv2 isn't loaded or there are no detections.

    Matches V2's reference-upload schema exactly so the resolver
    consumes Charlie detections the same way it does V2 detections.
    """
    if not detections:
        return []
    import v2_dinov2 as _v2d
    import v2_siglip as _v2s

    if not _v2d.is_loaded():
        return []

    squares: list = []

    for d in detections:
        box = d.get("box") or [0, 0, 0, 0]
        polys = (d.get("mask") or {}).get("polygons") or []
        try:
            clean = _v2d.inpaint_bbox_crop(pil, box, polys)
            square = _v2d.center_square_crop(clean)
        except Exception:
            # Degenerate box / decode error: use a neutral tile. Its
            # embedding will score poorly against all ref classes and
            # the detection will be rejected by the embed-low rule.
            square = PILImage.new("RGB", (16, 16), (0, 0, 0))
        squares.append(square)

    try:
        d_vecs = _v2d.encode_images_batch(squares)
    except Exception as e:
        print(f"[label-charlie] DINOv2 encode failed: {e}")
        d_vecs = None

    s_vecs = None
    if _v2s.is_loaded():
        try:
            s_vecs = _v2s.encode_images_batch(squares)
        except Exception as e:
            print(f"[label-charlie] SigLIP encode failed: {e}")

    for k, d in enumerate(detections):
        if d_vecs is not None and k < d_vecs.shape[0]:
            d["embedding"] = [round(float(x), 6) for x in d_vecs[k].tolist()]
            d["embed_version"] = _v2d.EMBED_VERSION
        if s_vecs is not None and k < s_vecs.shape[0]:
            d["siglip_embedding"] = [round(float(x), 6) for x in s_vecs[k].tolist()]
            d["siglip_version"] = _v2s.EMBED_VERSION

    return squares


def _charlie_box_iou(box_a, box_b) -> float:
    """Axis-aligned IoU on [x0, y0, x1, y1] boxes. Returns 0.0 on any
    degenerate input (None / wrong arity / negative area)."""
    if not box_a or not box_b or len(box_a) < 4 or len(box_b) < 4:
        return 0.0
    ax0, ay0, ax1, ay1 = float(box_a[0]), float(box_a[1]), float(box_a[2]), float(box_a[3])
    bx0, by0, bx1, by1 = float(box_b[0]), float(box_b[1]), float(box_b[2]), float(box_b[3])
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, (ax1 - ax0) * (ay1 - ay0))
    area_b = max(0.0, (bx1 - bx0) * (by1 - by0))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# Tunables for Charlie's specific path. Each defaults to the value
# we've found works for the alpaca/llama / hare/rabbit fixtures and
# can be overridden per deployment via env vars.
_CHARLIE_AMBIGUOUS_MARGIN = float(os.environ.get("CHARLIE_AMBIGUOUS_MARGIN", "0.08"))
_CHARLIE_VLM_WEIGHT = float(os.environ.get("CHARLIE_VLM_WEIGHT", "0.3"))
_CHARLIE_POST_FUSION_AMBIGUOUS_MARGIN = float(
    os.environ.get("CHARLIE_POST_FUSION_AMBIGUOUS_MARGIN", "0.01")
)

# Broad concept prompts used to recover candidate boxes for SPECIFIC-mode
# labels that SAM3's text prompt can't localise (e.g. "orangutan", or a
# made-up label). SAM3 finds the generic object, then the reference
# resolver assigns the real label. Override via env (comma-separated).
_CHARLIE_GENERIC_FALLBACK_PROMPTS = [
    p.strip() for p in os.environ.get(
        "CHARLIE_GENERIC_FALLBACK_PROMPTS", "object,animal"
    ).split(",") if p.strip()
]


def _charlie_segment_specific_with_fallback(
    charlie,
    pil,
    label_names: list[str],
    *,
    include_crops: bool = False,
    threshold=None,
    mask_threshold=None,
    min_relative_area=None,
    tile_native: bool = False,
    tile_size: int | None = None,
    cancel_check=None,
) -> tuple[list[dict], dict]:
    """SAM3 detection for the SPECIFIC pipeline with a generic recovery
    pass. Runs the label words first; for any label that yields ZERO
    detections, re-runs SAM3 with broad concept prompts
    (_CHARLIE_GENERIC_FALLBACK_PROMPTS) so a candidate box still surfaces.
    The reference resolver downstream assigns the real label to those
    boxes (or rejects them), which is what lets the pipeline detect
    categories whose label word SAM3 doesn't recognise.

    Conservative by design: the fallback only fires for labels that found
    nothing, so well-behaved projects (every label detected) are
    untouched. Generic boxes overlapping an existing detection (IoU > 0.5)
    are dropped, and their `gd_label` is cleared so "object"/"animal"
    never leaks in as a label prior — the references decide.
    """
    # Both passes (real labels + generic fallback) honour the same
    # resolution choice — tiled native crops or the classic 1500px pass.
    def _seg(im, lbls):
        if tile_native:
            return charlie.segment_labels_tiled(
                im, lbls, include_crops,
                threshold=threshold, mask_threshold=mask_threshold,
                min_relative_area=min_relative_area,
                tile_size=tile_size, cancel_check=cancel_check,
            )
        return charlie.segment_labels(
            im, lbls, include_crops,
            threshold=threshold, mask_threshold=mask_threshold,
            min_relative_area=min_relative_area,
        )

    dets, timings = _seg(pil, label_names)
    # If the first pass was cancelled it returns [], which would make every
    # label look "missing" and trigger a wasted generic-fallback sweep on an
    # already-cancelled job. Bail immediately.
    if (timings or {}).get("cancelled") or (cancel_check is not None and cancel_check()):
        return dets, timings
    found = {(d.get("gd_label") or "").strip().lower() for d in dets}
    missing = [l for l in label_names if l.strip().lower() not in found]
    if not missing or not _CHARLIE_GENERIC_FALLBACK_PROMPTS:
        return dets, timings
    try:
        gen_dets, _gt = _seg(pil, _CHARLIE_GENERIC_FALLBACK_PROMPTS)
    except Exception as e:
        print(f"[charlie/generic-fallback] retry failed: {e}")
        return dets, timings
    added = 0
    for g in gen_dets:
        gb = g.get("box") or []
        if len(gb) < 4:
            continue
        if any(_charlie_box_iou(gb, d.get("box") or []) > 0.5 for d in dets):
            continue  # already covered by a real-label detection
        g["gd_label"] = None      # references decide; don't bias on "object"
        g["gd_score"] = None
        dets.append(g)
        added += 1
    if added:
        print(
            f"[charlie/generic-fallback] {added} generic candidate(s) added "
            f"for missing label(s): {missing}"
        )
    return dets, timings


def _charlie_fuse_embed_vlm(
    embed_sims: dict[str, float],
    vlm_label: str | None,
    vlm_score: float | None,
    *,
    weight: float = _CHARLIE_VLM_WEIGHT,
) -> tuple[str | None, float, dict[str, float]]:
    """Combine embedding similarity with the VLM's verdict into a
    fused score per top-2 candidate label.

    For each top-2 label the fused score is `embed_sim + Δ`, where Δ
    is `+weight*vlm_score` if the VLM picked this label, `-weight*vlm_score`
    if the VLM picked the OTHER candidate (an active vote against
    this one), and 0 if the VLM declined or picked outside the
    candidate set. `weight` controls how much the VLM can move things
    relative to the embedding signal — at the default 0.3 a VLM at
    full confidence can flip an embed sim margin of up to ~0.6.

    Returns `(winner_label, fused_margin, fused_scores)`. The margin
    is over the candidate set only — it's the post-fusion equivalent
    of the resolver's `embed_margin` and the caller uses it to decide
    whether the detection is still ambiguous after fusion.
    """
    if not embed_sims:
        return None, 0.0, {}
    candidates = sorted(embed_sims.items(), key=lambda kv: -kv[1])[:2]
    if not candidates:
        return None, 0.0, {}
    cand_lower = {lab.strip().lower(): lab for lab, _ in candidates}
    vlm_lower = (vlm_label or "").strip().lower() if vlm_label else None
    vlm_conf = float(vlm_score) if vlm_score is not None else 0.0

    fused: dict[str, float] = {}
    for lab, sim in candidates:
        lab_lower = lab.strip().lower()
        if vlm_lower == lab_lower:
            delta = +weight * vlm_conf
        elif vlm_lower and vlm_lower in cand_lower:
            delta = -weight * vlm_conf
        else:
            delta = 0.0
        fused[lab] = float(sim) + delta

    sorted_scores = sorted(fused.values(), reverse=True)
    margin = sorted_scores[0] - (sorted_scores[1] if len(sorted_scores) >= 2 else 0.0)
    winner = max(fused, key=fused.get)
    return winner, margin, fused


# Cross-label IoU threshold for Charlie-specific NMS. SAM3 prompts
# each project label individually, so a single physical alpaca often
# comes back as one detection under "alpaca" AND a near-identical one
# under "llama". 0.6 is loose enough to catch sibling-prompt duplicates
# but strict enough to keep two genuinely-distinct objects standing
# close together. Override via CHARLIE_SPECIFIC_NMS_IOU.
_CHARLIE_NMS_IOU = float(os.environ.get("CHARLIE_SPECIFIC_NMS_IOU", "0.6"))

# Containment ratio: when one same-label box is THIS fraction-or-more
# inside another (intersection / smaller_area), suppress regardless
# of IoU. Catches the partial-mask case where SAM3 segments most of
# an object once AND segments a sub-region a second time — the two
# boxes share a label and the smaller is clearly the same object
# viewed at a smaller crop. 0.7 keeps "two glasses overlapping at a
# corner" out but kills "one box inside another" duplicates.
_CHARLIE_NMS_CONTAINMENT = float(os.environ.get("CHARLIE_NMS_CONTAINMENT", "0.7"))


def _charlie_boxes_should_merge(box_a, box_b) -> bool:
    """Same-label dedupe predicate. True when boxes a + b probably
    cover the same physical object — high IoU OR one box mostly
    contained inside the other."""
    if not box_a or not box_b or len(box_a) < 4 or len(box_b) < 4:
        return False
    ax0, ay0, ax1, ay1 = (float(c) for c in box_a[:4])
    bx0, by0, bx1, by1 = (float(c) for c in box_b[:4])
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return False
    area_a = max(0.0, (ax1 - ax0) * (ay1 - ay0))
    area_b = max(0.0, (bx1 - bx0) * (by1 - by0))
    union = area_a + area_b - inter
    iou = inter / union if union > 0 else 0.0
    smaller = min(area_a, area_b)
    containment = inter / smaller if smaller > 0 else 0.0
    return iou >= _CHARLIE_NMS_IOU or containment >= _CHARLIE_NMS_CONTAINMENT


def _charlie_nms_same_label(detections: list[dict]) -> list[dict]:
    """Same-label NMS for Charlie's GENERAL path.

    Two boxes with the same gd_label that overlap heavily are almost
    always the same object detected twice (SAM3 occasionally fires
    duplicate concept matches on the same region). Drop the lower-
    scoring member of each high-IoU same-label pair; keep the rest
    untouched. Different-label boxes are NEVER suppressed here — a
    "person" overlapping a "skateboard" is a real annotation and
    the user picks both.
    """
    if len(detections) <= 1:
        return detections

    def _label(d: dict) -> str:
        # pred_label > gd_label, lowercased. pred_label only fires
        # on specific (resolver-set) and shouldn't differ from
        # gd_label on general — but we look at both for safety.
        lab = (d.get("pred_label") or d.get("gd_label") or "").strip().lower()
        return lab

    def _score(d: dict) -> float:
        # gd_score is SAM3's mask score on the general path; the
        # specific path's embed sim doesn't apply.
        return float(d.get("gd_score") or 0.0)

    order = sorted(
        range(len(detections)),
        key=lambda i: _score(detections[i]),
        reverse=True,
    )
    suppressed: set[int] = set()
    for i in order:
        if i in suppressed:
            continue
        li = _label(detections[i])
        if not li:
            continue
        for j in order:
            if j == i or j in suppressed:
                continue
            if _label(detections[j]) != li:
                continue
            if _charlie_boxes_should_merge(
                detections[i].get("box"), detections[j].get("box"),
            ):
                suppressed.add(j)

    if not suppressed:
        return detections
    print(
        f"[label-charlie] same-label NMS dropped {len(suppressed)} duplicate(s) "
        f"of {len(detections)} (IoU≥{_CHARLIE_NMS_IOU} or containment≥{_CHARLIE_NMS_CONTAINMENT})"
    )
    return [d for i, d in enumerate(detections) if i not in suppressed]


def _charlie_nms_post_resolve(detections: list[dict], iou_thresh: float = _CHARLIE_NMS_IOU) -> list[dict]:
    """Post-resolution NMS for Charlie's specific path.

    SAM3's per-label prompting makes the same physical object appear
    once per sibling label — same box, near-identical mask, possibly
    resolver-assigned to different labels depending on which crop
    landed on top of which reference cluster. Drops the lower-scoring
    member of each high-IoU pair so the gallery shows one detection
    per object, not N (one per project label).

    Tiebreak: embed_sim_for_label first (the actual decision metric),
    gd_score second. Ties on both keep the earlier-listed detection
    so the function is stable.
    """
    if len(detections) <= 1:
        return detections

    def _score(d: dict) -> tuple[float, float]:
        return (
            float(d.get("embed_sim_for_label") or 0.0),
            float(d.get("gd_score") or 0.0),
        )

    # Sort indices by score desc; pop best-first into kept[], skipping
    # any with high IoU vs an already-kept one. Operating on indices
    # keeps the original detections list untouched and makes the
    # stable-order rebuild at the end straightforward.
    order = sorted(
        range(len(detections)),
        key=lambda i: _score(detections[i]),
        reverse=True,
    )
    kept_idx: list[int] = []
    suppressed: set[int] = set()
    for i in order:
        if i in suppressed:
            continue
        kept_idx.append(i)
        for j in order:
            if j == i or j in suppressed:
                continue
            iou = _charlie_box_iou(detections[i].get("box"), detections[j].get("box"))
            if iou >= iou_thresh:
                suppressed.add(j)

    if not suppressed:
        return detections
    keep = set(kept_idx)
    print(
        f"[label-charlie] NMS dropped {len(suppressed)} duplicate(s) of "
        f"{len(detections)} detection(s) (IoU≥{_CHARLIE_NMS_IOU})"
    )
    return [d for i, d in enumerate(detections) if i in keep]


def _charlie_classify_general_sam3(
    image_pil: "PILImage.Image",
    box_xyxy: list[float],
    candidate_labels: list[str],
) -> tuple[dict | None, dict[str, float]]:
    """Run SAM3 once per candidate label and pick the one whose
    detection bbox best overlaps the user's box. Used by Charlie's
    interactive endpoints on GENERAL projects, where the user hasn't
    uploaded references so the embedding pipeline has nothing to
    compare against. SAM3's text-prompted concept segmentation IS
    the label signal for these projects.
    """
    charlie = state.get("charlie")
    if charlie is None:
        return None, {}
    try:
        label, score, timings = charlie.classify_box(image_pil, box_xyxy, candidate_labels)
    except Exception as e:
        print(f"[charlie/classify-general] SAM3 classify failed: {e}")
        return None, {}
    if not label:
        return None, {"sam3_classify_ms": float(timings.get("total_ms", 0.0))}
    verdict = {
        "pred_label": label,
        "pred_source": "vlm",  # closest existing enum value the FE understands
        "embed_nearest_label": label,
        "embed_nearest_sim": float(score) if score is not None else None,
        "embed_sim_for_label": float(score) if score is not None else None,
        "embed_margin": 0.0,
        "rejected": False,
        "reject_reason": None,
        "ambiguous": False,
        "vlm_action": None,
        "sims": {},
    }
    return verdict, {"sam3_classify_ms": round(float(timings.get("total_ms", 0.0)), 1)}


def _charlie_resolve_label_for_box(
    image_pil: "PILImage.Image",
    box_xyxy: list[float],
    mask_polys: list | None,
    project_id: str,
    candidate_labels: list[str],
) -> tuple[dict | None, dict[str, float]]:
    """Embed one box-crop + run the V2 specific resolver against the
    project's reference centroids. Returns (verdict_or_None, timings).

    Shared helper used by Charlie's interactive endpoints
    (detect_point + classify_box) so click-to-detect and add-box
    both go through the same code path the labelling job does. The
    caller decides whether to fire a VLM tiebreak on top.
    """
    import v2_dinov2 as _v2d
    import v2_siglip as _v2s
    if not _v2d.is_loaded():
        return None, {}

    timings: dict[str, float] = {}
    t_embed = time.perf_counter()

    # Inpainted-square crop, same procedure references go through so
    # the cosine comparison stays apples-to-apples.
    clean = _v2d.inpaint_bbox_crop(image_pil, box_xyxy, mask_polys or [])
    square = _v2d.center_square_crop(clean)
    d_vecs = _v2d.encode_images_batch([square])
    emb_raw = (
        [round(float(x), 6) for x in d_vecs[0].tolist()]
        if d_vecs is not None and d_vecs.shape[0] > 0 else []
    )
    emb_siglip_raw: list[float] | None = None
    if _v2s.is_loaded():
        try:
            s_vecs = _v2s.encode_images_batch([square])
            if s_vecs is not None and s_vecs.shape[0] > 0:
                emb_siglip_raw = [round(float(x), 6) for x in s_vecs[0].tolist()]
        except Exception as e:
            print(f"[charlie/resolve] siglip encode failed: {e}")
    timings["embed_ms"] = round((time.perf_counter() - t_embed) * 1000.0, 1)

    if not emb_raw:
        return None, timings

    # Reference embeddings loaded from disk for this project, then scored
    # via _charlie_resolve_from_refs (interactive click-to-detect / add-box
    # path — uses Fisher + patch tokens). NOTE: the labelling JOB
    # (label_charlie) and the public demo deliberately skip Fisher/patches
    # and call _v2_resolve_label_specific directly with raw refs.
    try:
        by_label, by_label_siglip, _dirty = _v2_load_or_backfill_reference_embeddings(project_id)
    except Exception as e:
        print(f"[charlie/resolve] reference load failed: {e}")
        return None, timings
    return _charlie_resolve_from_refs(
        emb_raw, emb_siglip_raw, square,
        by_label, by_label_siglip, candidate_labels,
        project_id=project_id, timings=timings,
    )


def _charlie_resolve_from_refs(
    emb_raw,
    emb_siglip_raw,
    square,
    by_label,
    by_label_siglip,
    candidate_labels,
    *,
    project_id: str | None = None,
    timings: dict | None = None,
    refs_patch: dict | None = None,
    refs_patch_siglip: dict | None = None,
):
    """Score one already-embedded box crop against pre-loaded reference
    embeddings with the V2 specific resolver. Extracted from
    _charlie_resolve_label_for_box so the public demo — which encodes
    references in memory and has no persisted project — runs the
    IDENTICAL resolver the app's labelling job + interactive endpoints
    use: same kNN scoring, same Fisher handling, same LOO class
    thresholds, same patch-token refinement.

    Patch tokens come from disk for a persisted project (project_id set),
    or the caller can pass them in directly (refs_patch / refs_patch_siglip
    keyed by lowercased label) — that's how the demo gets patch-token
    parity without a project on disk. When neither is available the
    resolver falls back to pooled scoring.
    """
    import v2_dinov2 as _v2d
    import v2_siglip as _v2s
    import numpy as _np
    if timings is None:
        timings = {}

    refs_dino_arr = _v2_stack_refs(by_label)
    refs_siglip_arr = _v2_stack_refs(by_label_siglip)
    if not refs_dino_arr:
        return None, timings

    # No Fisher reweighting — match the batch labelling job, which
    # resolves with RAW refs (see _run_label_charlie_job_impl: "Pass raw
    # embeddings directly, no Fisher weighting. Fisher LDA overfits on
    # small reference sets and was causing systematic bias toward one
    # class"). Applying it here made an interactive click on the same crop
    # return a different, biased label than the auto-pass, with no
    # reconciliation. class_thresholds come from the raw refs, as in batch.
    fisher_dino = None
    fisher_siglip = None
    class_thresholds_local = _v2_compute_class_thresholds(refs_dino_arr) or None

    # Caller-provided patches (demo, in-memory) take precedence; else
    # load from disk for a persisted project; else pooled-only.
    if refs_patch is None and refs_patch_siglip is None:
        refs_patch, refs_patch_siglip = {}, {}
        if project_id and _v2_patch_match_enabled():
            try:
                refs_patch, refs_patch_siglip = _v2_load_or_backfill_patch_tokens(project_id)
            except Exception as e:
                print(f"[charlie/resolve] patch refs load failed: {e}")
                refs_patch, refs_patch_siglip = {}, {}
    else:
        refs_patch = refs_patch or {}
        refs_patch_siglip = refs_patch_siglip or {}

    q_patch_dino = (None, None)
    q_patch_siglip = (None, None)
    try:
        if refs_patch and square is not None:
            q_patch_dino = _v2d.encode_image_patches(square)
    except Exception as e:
        print(f"[charlie/resolve] dino patch encode failed: {e}")
    if refs_patch_siglip and square is not None and _v2s.is_loaded():
        try:
            q_patch_siglip = _v2s.encode_image_patches(square)
        except Exception as e:
            print(f"[charlie/resolve] siglip patch encode failed: {e}")

    # Apply Fisher to query for scoring (not stored).
    emb_for_scoring = emb_raw
    emb_siglip_for_scoring = emb_siglip_raw
    if fisher_dino is not None:
        emb_for_scoring = [
            round(float(x), 6) for x in
            _v2_apply_fisher_to_arr(_np.asarray([emb_raw], dtype=_np.float32), fisher_dino)[0].tolist()
        ]
    if fisher_siglip is not None and emb_siglip_raw is not None:
        emb_siglip_for_scoring = [
            round(float(x), 6) for x in
            _v2_apply_fisher_to_arr(_np.asarray([emb_siglip_raw], dtype=_np.float32), fisher_siglip)[0].tolist()
        ]

    label_display_local = {t.lower(): t for t in candidate_labels}
    try:
        verdict = _v2_resolve_label_specific(
            emb_for_scoring,
            None,
            refs_dino_arr,
            label_display_local,
            score_mode="knn",
            gd_score=None,
            embedding_siglip=emb_siglip_for_scoring,
            refs_by_label_siglip=refs_siglip_arr or None,
            class_thresholds=class_thresholds_local,
            query_patch_tokens=q_patch_dino[0],
            query_patch_fg=q_patch_dino[1],
            refs_by_label_patches=refs_patch or None,
            query_patch_tokens_siglip=q_patch_siglip[0],
            query_patch_fg_siglip=q_patch_siglip[1],
            refs_by_label_patches_siglip=refs_patch_siglip or None,
            ambiguous_margin=_CHARLIE_AMBIGUOUS_MARGIN,
        )
    except Exception as e:
        print(f"[charlie/resolve] resolver failed: {e}")
        return None, timings
    return verdict, timings


def _charlie_interactive_ref_label(
    image_pil,
    box_xyxy,
    mask_polys,
    project_id: str,
    candidate_labels: list[str],
) -> tuple[str | None, float | None]:
    """Resolve a label for an interactively-placed region (a SAM2 point /
    box mask) against the project's references — the SAME basis the batch
    labelling job uses. Returns (label, score), or (None, None) when there
    are no usable references (general projects), so the caller falls back
    to the SAM3 / VLM path.

    This is what lets click-to-detect & add-box label categories whose
    word SAM3 can't text-match (e.g. "orangutan", a made-up label): SAM2
    localises the region label-agnostically, then the reference embeddings
    decide the label. The box is drawn deliberately by the user, so even a
    low-similarity verdict returns its best guess (pred / nearest) rather
    than rejecting outright."""
    try:
        verdict, _t = _charlie_resolve_label_for_box(
            image_pil, [float(c) for c in box_xyxy], mask_polys or [],
            project_id, candidate_labels,
        )
    except Exception as e:
        print(f"[charlie/interactive] reference resolve failed: {e}")
        return None, None
    if not verdict:
        return None, None
    label = verdict.get("pred_label") or verdict.get("embed_nearest_label")
    if not label:
        return None, None
    sim = verdict.get("embed_sim_for_label")
    if sim is None:
        sim = verdict.get("embed_nearest_sim")
    return label, (round(float(sim), 4) if sim is not None else None)


async def _charlie_vlm_tiebreak_async(
    verdict: dict,
    image_pil: "PILImage.Image",
    box_xyxy: list[float],
    mask_polys: list | None,
    loop,
) -> tuple[dict, float | None]:
    """Run the VLM tiebreak on an ambiguous verdict, then fuse the
    answer with the embed sims via the confidence-weighted strategy.
    Returns the mutated verdict + the VLM call duration (ms) or None
    if the VLM wasn't available."""
    sims_dict = verdict.get("sims") or {}
    if len(sims_dict) < 2:
        return verdict, None
    try:
        from vlm_validate import vlm_classify as _vlm_classify
    except Exception as e:
        print(f"[charlie/vlm] import failed: {e}")
        return verdict, None
    top2 = sorted(sims_dict.items(), key=lambda kv: -kv[1])[:2]
    top2_labels = [lab for lab, _ in top2]
    t0 = time.perf_counter()
    try:
        async with state["gpu_lock"]:
            v_label, v_score = await loop.run_in_executor(
                None, _vlm_classify, image_pil, box_xyxy, top2_labels, mask_polys,
            )
    except Exception as e:
        print(f"[charlie/vlm] call failed: {e}")
        return verdict, None
    vlm_ms = (time.perf_counter() - t0) * 1000.0

    winner, fused_margin, fused_scores = _charlie_fuse_embed_vlm(
        sims_dict, v_label, v_score,
    )
    verdict["vlm_label"] = v_label
    verdict["vlm_score"] = v_score
    if winner is not None:
        cur_pred = (verdict.get("pred_label") or "").strip().lower()
        winner_lower = winner.strip().lower()
        if winner_lower != cur_pred:
            verdict["pred_label"] = winner
            verdict["pred_source"] = "embed-vlm"
            verdict["vlm_action"] = "tiebreak"
        else:
            verdict["pred_source"] = "embed-vlm"
            verdict["vlm_action"] = (
                "confirm" if v_label and v_label.strip().lower() == cur_pred
                else "tiebreak"
            )
        for kk, vv in sims_dict.items():
            if kk.strip().lower() == winner_lower:
                verdict["embed_sim_for_label"] = round(float(vv), 4)
                break
        verdict["fused_margin"] = round(float(fused_margin), 4)
        verdict["fused_scores"] = {
            k: round(float(v), 4) for k, v in fused_scores.items()
        }
        verdict["ambiguous"] = fused_margin < _CHARLIE_POST_FUSION_AMBIGUOUS_MARGIN
    return verdict, vlm_ms


async def _run_label_charlie_job(job, emit, cancel_event):
    """Thin guard around the V2 Charlie labelling job that ALWAYS lifts
    the per-job sidecar-refresh suppression (even on a mid-job exception)
    and fires one final rebuild. Previously the discard lived inline at
    the end with no try/finally, so any unhandled error mid-job left the
    project suppressed for the process lifetime, freezing its
    overview/stats/initial sidecars. The work is in the _impl below."""
    # Suppress sidecar rebuilds during the job. Each flush would otherwise
    # kick a full overview+stats+initial+workspace rebuild, which on a
    # 1000-image manifest can each take seconds. The finally lifts it and
    # fires one post-job rebuild capturing the final state.
    _SIDECAR_REFRESH_SUPPRESSED.add(job.project)
    # Pause fire-and-forget background embeddings for the duration of the
    # job so a post-import embedding backlog can't steal GPU time from
    # SAM3 and make labelling crawl. They resume + drain when the job ends.
    _pause_bg_embeddings()
    try:
        await _run_label_charlie_job_impl(job, emit, cancel_event)
    finally:
        _resume_bg_embeddings()
        _SIDECAR_REFRESH_SUPPRESSED.discard(job.project)
        try:
            _kick_sidecar_refresh(job.project)
        except Exception as e:
            print(f"[label-charlie] final sidecar refresh failed: {e}")


async def _run_label_charlie_job_impl(job, emit, cancel_event):
    """V2 Charlie labelling job. Iterates the project's manifest for
    every import marked `labelled: False` (or with empty detections,
    for backwards compat with imports that pre-date the flag) and
    runs the SAM3 charlie pipeline against each one. After each
    image, the entry's detections + timings + `labelled` flag are
    written back to the manifest and the labelled_preview is
    re-baked. Emits one progress event per image so the FE job card
    can stream a counter + filename. Sidecar-refresh suppression is
    owned by the wrapper _run_label_charlie_job.
    """
    proj = project_dir(job.project)
    if not proj.exists():
        raise HTTPException(404, "project not found")

    charlie = state.get("charlie")
    if charlie is None:
        raise RuntimeError("pipeline_charlie not loaded — cannot run labelling")

    # Timestamped progress markers so we can attribute slow labelling
    # jobs to specific stages (setup vs SAM3 vs resolver). The user
    # report was "hangs at 0/1 forever on a 1000-image project" — the
    # logs needed to differentiate "setup takes forever" from "first
    # SAM3 call takes forever" from "GPU is busy with another job".
    _t_job_start = time.perf_counter()
    def _stage(label: str) -> None:
        dt = (time.perf_counter() - _t_job_start) * 1000.0
        print(f"[label-charlie] {job.project} +{dt:.0f}ms {label}")

    _stage("setup: load manifest")
    # copy=False — we read tags + iterate imports for the pending
    # filter without mutating. Saves 200-500ms of deepcopy on big
    # manifests at job start.
    manifest = load_manifest(job.project, copy=False) or {}
    tags = list(manifest.get("tags") or [])
    if not tags:
        raise RuntimeError("no labels defined for this project")


    # Per-run SAM3 knobs — sent from the project page's Annotations
    # card. Each value is optional; None means "use the SAM3_* module
    # default in pipeline_charlie." Sanity-clamped so a malformed FE
    # payload can't push thresholds outside [0, 1] (which would either
    # drop every box or run the model with nonsensical inputs).
    sam3_params = (job.params or {})

    def _clamp01(v, default=None):
        try:
            if v is None:
                return default
            f = float(v)
        except (TypeError, ValueError):
            return default
        if f < 0.0 or f > 1.0:
            return default
        return f

    sam3_threshold = _clamp01(sam3_params.get("sam3_threshold"))
    sam3_mask_threshold = _clamp01(sam3_params.get("sam3_mask_threshold"))
    sam3_min_relative_area = _clamp01(sam3_params.get("sam3_min_relative_area"))
    # Native-resolution tiling opt-in ("Downscale vs Tile" on the FE).
    # tile_size None = pipeline default (SAM3_TARGET_LONGEST_EDGE), the
    # only value where per-tile resize is a true no-op.
    charlie_tile_native = bool(sam3_params.get("tile_native"))
    try:
        charlie_tile_size = int(sam3_params.get("tile_size") or 0) or None
    except (TypeError, ValueError):
        charlie_tile_size = None
    # Range-gate the only sam3 param that isn't _clamp01'd: a tiny tile on
    # a 4K frame means 10^5+ SAM3 passes under the GPU lock. The pipeline
    # floors it too; rejecting here keeps garbage out of settingsLastRun.
    if charlie_tile_size is not None and not (256 <= charlie_tile_size <= 8192):
        charlie_tile_size = None

    _stage("setup: classifying dataset type")
    # Dataset-type lookup. Specific projects route detections through
    # V2's embedding-based resolver (SAM3 picks the boxes, reference
    # embeddings pick the labels); general projects keep today's pure
    # SAM3 path where the text prompt IS the label.
    try:
        dt_record = _classify_dataset_type_cached(job.project, list(tags))
        dataset_type = (
            (dt_record.get("type") or "general")
            if isinstance(dt_record, dict) else "general"
        )
    except Exception as e:
        print(f"[label-charlie] dataset_type lookup failed: {e}, defaulting to general")
        dataset_type = "general"

    # Resolver context (specific only). Loaded once for the whole job
    # so we don't re-stack reference embeddings per image.
    refs_by_label_arr: dict = {}
    refs_by_label_siglip_arr: dict = {}
    refs_by_label_patches: dict = {}
    refs_by_label_patches_siglip: dict = {}
    class_thresholds: dict | None = None
    label_display: dict[str, str] = {t.lower(): t for t in tags}
    if dataset_type == "specific":
        try:
            _stage("setup: loading reference embeddings")
            by_label, by_label_siglip, _dirty = _v2_load_or_backfill_reference_embeddings(job.project)
            _stage(f"setup: reference embeddings ready (dino={sum(len(v) for v in by_label.values())} siglip={sum(len(v) for v in by_label_siglip.values())})")
            refs_by_label_arr = _v2_stack_refs(by_label)
            refs_by_label_siglip_arr = _v2_stack_refs(by_label_siglip)

            # Plain LOO thresholds on raw (un-weighted) ref arrays.
            # No Fisher LDA weighting — it overfits on small ref sets
            # (hare/rabbit with <20 refs per class) and amplifies noise
            # dimensions. Patch matching also disabled: plain pooled kNN
            # is more reliable when classes look visually similar.
            # VLM handles the borderline calls instead.
            class_thresholds = _v2_compute_class_thresholds(refs_by_label_arr) or None

            print(
                f"[label-charlie] {job.id} specific resolver ready "
                f"({len(refs_by_label_arr)} dino classes, "
                f"{len(refs_by_label_siglip_arr)} siglip classes, "
                f"no Fisher, no patch matching, VLM margin={_CHARLIE_AMBIGUOUS_MARGIN})"
            )
        except Exception as e:
            print(f"[label-charlie] reference embedding setup failed: {e}")
            refs_by_label_arr = {}
            refs_by_label_siglip_arr = {}
            refs_by_label_patches = {}
            refs_by_label_patches_siglip = {}
            class_thresholds = None
    print(f"[label-charlie] {job.id} dataset_type={dataset_type}")

    imports_dir = proj / "images"
    pending: list[dict] = []
    # `force_relabel` flips the runner from "process only unlabelled
    # images" to "re-process every image". Used by the FE when the
    # user adds a fresh label to an already-fully-labelled dataset
    # so the new tag surfaces on existing images. Auto detections
    # get replaced; user-edited boxes are kept untouched by the
    # per-image save below.
    force_relabel = bool((job.params or {}).get("force_relabel"))
    for entry in (manifest.get("imports") or []):
        if not isinstance(entry, dict):
            continue
        if force_relabel:
            pending.append(entry)
            continue
        # User-edited boxes win as the source of truth — see
        # _v2_is_labelled. A "Clear all" in the viewer leaves
        # labelled=True but flips editedBoxesSet=True with an
        # empty editedBoxes list, which is the user explicitly
        # saying "this image is now unlabelled". Without this
        # branch, clicking Start labelling new images after a
        # Clear all did nothing because the labelled flag was
        # still True from the original pass.
        if entry.get("editedBoxesSet"):
            edited = entry.get("editedBoxes")
            is_unlabelled = not (isinstance(edited, list) and len(edited) > 0)
        else:
            is_unlabelled = (
                entry.get("labelled") is False
                or (entry.get("labelled") is None and not (entry.get("detections") or []))
            )
        if is_unlabelled:
            pending.append(entry)
    # Process pending in the same order the FE gallery shows them
    # (createdAt DESC — newest at top-left, oldest at bottom-right).
    # The FE pre-assigns timestamps in REVERSE drop order within
    # each batch so this DESC iteration walks the grid top-left
    # → bottom-right consistently with how the user sees uploads
    # filling in.
    def _entry_sort_key(e: dict) -> float:
        ca = e.get("createdAt")
        if isinstance(ca, (int, float)):
            return -float(ca)
        if isinstance(ca, str):
            try:
                from datetime import datetime as _dt
                return -_dt.fromisoformat(ca.replace("Z", "+00:00")).timestamp()
            except Exception:
                return 0.0
        return 0.0
    pending.sort(key=_entry_sort_key)
    if not pending:
        await emit("progress", {"index": 0, "total": 0, "image": ""})
        return

    job.n_images = len(pending)
    n_done = 0
    loop = asyncio.get_running_loop()
    _stage(f"loop: {len(pending)} image(s) to label")

    def _load_pil_sync(p: Path) -> "PILImage.Image":
        with PILImage.open(p) as _im:
            return _im.convert("RGB")

    # Batched per-image manifest writes. Each save_manifest call
    # serialises + writes the WHOLE manifest, which becomes the
    # dominant per-iteration cost once a project has more than ~100
    # images persisted (each detection record carries embedding +
    # mask polygons, so the file grows past several MB very fast).
    # Coalesce updates and flush every LABEL_FLUSH_EVERY images;
    # final flush at job-end captures the tail.
    pending_label_updates: list[dict] = []
    # Running tally of import ids the runner has actually committed
    # labels to. Threaded into the auto-augment hook so the post-
    # labelling regen scopes itself to just-labelled images instead
    # of re-augmenting the whole project.
    labelled_import_ids: list[str] = []
    LABEL_FLUSH_EVERY = 10

    async def _flush_label_updates(mark_run: bool = False) -> None:
        if not pending_label_updates:
            return
        updates = list(pending_label_updates)
        lock = await _manifest_write_lock(job.project)
        async with lock:
            # copy=False skips the 200-500ms deepcopy on a 30MB
            # manifest. Safe here because we mutate-then-save_manifest
            # in the same critical section: save_manifest re-seeds the
            # cache with its own deepcopy after the write lands, so
            # any concurrent reader who picks up our mutations
            # transiently sees the post-flush state — which is the
            # state we're about to persist anyway. The per-project
            # _manifest_write_lock above keeps other writers out of
            # the cache while we're mutating.
            mm = load_manifest(job.project, copy=False) or {}
            by_id = {e.get("id"): e for e in (mm.get("imports") or []) if isinstance(e, dict)}
            for upd in updates:
                e = by_id.get(upd["id"])
                if e is None:
                    continue
                # If this image is here because the prior state was
                # "edited boxes explicitly empty" (e.g. user did click-
                # to-detect, then deleted that label — purge_label leaves
                # editedBoxes=[] + editedBoxesSet=True), the new auto
                # detections we're about to write should become the
                # source of truth in the viewer. Without this clear,
                # BoxEditor keeps reading the empty editedBoxes (because
                # editedBoxesSet=True overrides) and the user sees the
                # segmented cover but no boxes/labels inside.
                prior_edited = e.get("editedBoxes")
                if (
                    e.get("editedBoxesSet")
                    and isinstance(prior_edited, list)
                    and len(prior_edited) == 0
                ):
                    e.pop("editedBoxes", None)
                    e.pop("editedBoxesSet", None)
                # Persist the resolved detections WITHOUT the raw
                # embedding vectors. The DINOv2 (1024-d) + SigLIP (768-d)
                # float arrays on each detection are only needed during
                # the resolve step above — which already ran — and a
                # re-label always re-embeds fresh crops (see the
                # _charlie_embed_detections call earlier in the loop), so
                # a stored test-image embedding is never read back. Every
                # other reader of persisted embeddings is a *reference*
                # (manifest["references"], untouched here) or a freshly
                # computed vector. Keeping them on test-image detections
                # ballooned the manifest ~10× on specific projects (~200 MB
                # at 4000 imgs) — which is what turned each flush's
                # whole-manifest write into an O(n²) slog past ~400 labels
                # AND left the bloated cache + sidecar rebuilds dragging
                # every other project's load. Stripping at the source
                # keeps every flush's write small, so labelling stays
                # fast regardless of dataset size. _strip_embedding keeps
                # mask polygons + embed_sims, so the gallery overlay and
                # the pipeline popup still work — it only drops the dead
                # vectors.
                e["detections"] = [
                    _strip_embedding(d) if isinstance(d, dict) else d
                    for d in upd["detections"]
                ]
                e["timings"] = upd["timings"]
                e["labelled"] = True
                # Persisted cachebuster (epoch ms) for the labelled
                # preview. Surfaced in _tile_overview so a cold reopen
                # rebuilds the SAME ?v= the browser cached the segmented
                # bake under — instead of dropping to the bare URL where
                # the pre-label blank preview is still cached.
                e["labelledAt"] = int(time.time() * 1000)
                # Snapshot the tags this entry was searched with so the
                # FE's "fresh label" detection compares against what was
                # actually run, not what happened to produce detections.
                # Without this a label that's prompted-for but never
                # matches anywhere in the dataset would stay flagged
                # "fresh" forever and force_relabel every single click.
                e["labelsRun"] = list(tags)
                if not e.get("width") and upd.get("width"):
                    e["width"] = upd["width"]
                if not e.get("height") and upd.get("height"):
                    e["height"] = upd["height"]
                # Remember which imports we touched so the post-job
                # auto-augment hook can scope its regen to only the
                # just-labelled set.
                if upd["id"]:
                    labelled_import_ids.append(upd["id"])
            # Snapshot the project-level label list + settings this pass
            # ran with, but ONLY when the job actually completes (mark_run).
            # Writing it on every flush meant a CANCELLED force_relabel
            # (which still hits the final flush) marked the new label as
            # fully run after ~10 images, so the FE dropped it from
            # freshLabels and disabled re-triggering. Gating on mark_run
            # keeps "fully run" honest; folded into the same write so a
            # completed job costs no extra save.
            if mark_run:
                mm["labelsLastRun"] = list(tags)
                # settingsLastRun mirrors labelsLastRun. The slider trio is
                # only written when all three are present (a missing value
                # means "use backend default", not a real pick); the tiling
                # choice is written unconditionally — OUTSIDE that gate — so
                # a default-slider run with tiling on still persists it for
                # the FE to re-hydrate.
                slr = dict(mm.get("settingsLastRun") or {})
                if (
                    sam3_threshold is not None
                    and sam3_mask_threshold is not None
                    and sam3_min_relative_area is not None
                ):
                    slr.update({
                        "threshold": sam3_threshold,
                        "mask_threshold": sam3_mask_threshold,
                        "min_relative_area": sam3_min_relative_area,
                    })
                slr["tile_native"] = charlie_tile_native
                if charlie_tile_size:
                    slr["tile_size"] = charlie_tile_size
                else:
                    # Don't let an older run's explicit tile_size linger next
                    # to this run's settings — absent means "backend default".
                    slr.pop("tile_size", None)
                mm["settingsLastRun"] = slr
            mm["updatedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            # cache_by_ref, we hold the only reference to `mm`
            # (load_manifest(copy=False) + our mutations under the
            # per-project write lock) and discard it right after
            # return. Skipping the deepcopy here is the biggest
            # per-flush speedup on big projects.
            # Serialise + write OFF the event loop: on a big labelled
            # project the manifest is multi-MB and orjson.dumps + the
            # file write are synchronous, so doing them inline stalled
            # every other request (the FE's job-progress + overview polls)
            # for the duration of each flush, which read as the page
            # freezing mid-label. save_manifest is already called from
            # worker threads elsewhere, so this is safe; the async write
            # lock still serialises same-project writes.
            await loop.run_in_executor(
                None, lambda: save_manifest(job.project, mm, cache_by_ref=True)
            )
        # Clear only AFTER the write lands, so a save failure (disk full /
        # serialization) can't silently drop these images' detections.
        pending_label_updates.clear()

    # Prefetch: kick off PIL decode for the first valid entry before the
    # loop starts so the GPU never waits for disk I/O on image 0. After
    # that, each iteration starts the next decode before acquiring the
    # GPU lock so PIL's CPU work overlaps GPU inference.
    _prefetch_fut: "asyncio.Future | None" = None
    for _pe in pending:
        _pfn = _pe.get("filename")
        if _pfn and (imports_dir / _pfn).exists():
            _prefetch_fut = loop.run_in_executor(None, _load_pil_sync, imports_dir / _pfn)
            break

    for _pending_idx, entry in enumerate(pending):
        if cancel_event.is_set():
            break
        filename = entry.get("filename")
        import_id = entry.get("id")
        if not filename or not import_id:
            continue
        path = imports_dir / filename
        if not path.exists():
            print(f"[label-charlie] {import_id}: file missing on disk: {path}")
            continue

        _stage(f"image {n_done + 1}/{len(pending)}: opening {filename}")
        # Pre-emit so the FE knows which image is currently in flight
        # (paints a "Labelling…" overlay on the matching tile). The
        # post-emit at the bottom of the loop bumps n_done after the
        # manifest write — the index never goes backwards.
        await emit("progress", {
            "index": n_done,
            "total": len(pending),
            "image": filename,
        })

        # Retrieve the PIL image — either from the prefetch future
        # (started at end of previous iteration / before the loop) or
        # freshly decoded. Running in executor keeps the event loop
        # free during the PIL decode (~20-100 ms for large images).
        try:
            if _prefetch_fut is not None:
                pil = await _prefetch_fut
                _prefetch_fut = None
            else:
                pil = await loop.run_in_executor(None, _load_pil_sync, path)
        except Exception as e:
            print(f"[label-charlie] {import_id}: failed to decode: {e}")
            _prefetch_fut = None
            continue

        # Start decoding the NEXT image NOW, before we touch the GPU,
        # so PIL's CPU work runs concurrently with GPU inference and
        # the event loop doesn't stall on disk I/O at the top of the
        # next iteration.
        _prefetch_fut = None
        for _ni in range(_pending_idx + 1, len(pending)):
            _ne = pending[_ni]
            _nfn = _ne.get("filename")
            if not _nfn or not _ne.get("id"):
                continue
            _np = imports_dir / _nfn
            if _np.exists():
                _prefetch_fut = loop.run_in_executor(None, _load_pil_sync, _np)
                break

        _stage(f"image {n_done + 1}/{len(pending)}: waiting for GPU lock")
        # Acquire the GPU lock ONCE and run both SAM3 and the embedding
        # pass under it. This eliminates the release/re-acquire round-
        # trip between detection and embedding that previously let other
        # jobs interleave (causing OOM on the next acquire) and removes
        # one Python async overhead cycle per image.
        try:
            async with state["gpu_lock"]:
                _stage(f"image {n_done + 1}/{len(pending)}: running SAM3")
                # Specific datasets get the generic-fallback detector so a
                # candidate box still surfaces for labels SAM3 can't localise
                # by name (the references then assign the real label). General
                # datasets keep plain segment_labels — SAM3's text label IS
                # their label, so a generic box would have no meaning.
                if dataset_type == "specific":
                    detections, timings = await loop.run_in_executor(
                        None,
                        lambda: _charlie_segment_specific_with_fallback(
                            charlie, pil, list(tags), include_crops=True,
                            threshold=sam3_threshold,
                            mask_threshold=sam3_mask_threshold,
                            min_relative_area=sam3_min_relative_area,
                            tile_native=charlie_tile_native,
                            tile_size=charlie_tile_size,
                            cancel_check=cancel_event.is_set,
                        ),
                    )
                elif charlie_tile_native:
                    detections, timings = await loop.run_in_executor(
                        None,
                        lambda: charlie.segment_labels_tiled(
                            pil, list(tags), True,
                            threshold=sam3_threshold,
                            mask_threshold=sam3_mask_threshold,
                            min_relative_area=sam3_min_relative_area,
                            tile_size=charlie_tile_size,
                            cancel_check=cancel_event.is_set,
                        ),
                    )
                else:
                    detections, timings = await loop.run_in_executor(
                        None,
                        lambda: charlie.segment_labels(
                            pil, list(tags), True,
                            threshold=sam3_threshold,
                            mask_threshold=sam3_mask_threshold,
                            min_relative_area=sam3_min_relative_area,
                        ),
                    )
                # Specific-dataset path: embed immediately under the same
                # lock so we don't pay a second acquire cycle or risk
                # another job stealing the GPU slot between the two passes.
                # General projects skip this — SAM3's text-prompt assignment
                # is the label source of truth for them.
                if dataset_type == "specific" and detections and refs_by_label_arr:
                    t_embed = time.perf_counter()
                    await loop.run_in_executor(
                        None, _charlie_embed_detections, pil, detections,
                    )
                    timings["embed_ms"] = round(
                        (time.perf_counter() - t_embed) * 1000.0, 1
                    )
        except Exception as e:
            print(f"[label-charlie] {import_id}: GPU inference failed: {e}")
            continue

        # Cancel landed mid-image (tiled runs are ~7× the work, so this is
        # the likely cancel point). The tiled detector returns ([], cancelled)
        # — DON'T queue that as a real result: persisting it would mark the
        # image labelled-with-zero-detections and, on a force_relabel run,
        # wipe its existing boxes. Bail before the append so the image stays
        # exactly as it was and is re-picked next run. Deliberately checks
        # ONLY the sentinel (not cancel_event): an image whose pass COMPLETED
        # just as cancel landed is real work and persists, as it always has.
        if (timings or {}).get("cancelled"):
            break

        if dataset_type == "specific" and detections and refs_by_label_arr:
            t_resolve = time.perf_counter()
            import numpy as _np_resolve
            for k_det, d in enumerate(detections):
                emb_raw = d.get("embedding") or []
                if not emb_raw:
                    # Encoder failed for this crop — leave SAM3's
                    # text-prompt label in place rather than rejecting
                    # outright. The detection will surface in the
                    # gallery; the user can fix it manually if wrong.
                    continue
                emb_siglip_raw = d.get("siglip_embedding") or None

                # Pass raw embeddings directly — no Fisher weighting.
                # Fisher LDA overfits on small reference sets and was
                # causing systematic bias toward one class.
                try:
                    verdict = _v2_resolve_label_specific(
                        emb_raw,
                        d.get("gd_label"),
                        refs_by_label_arr,
                        label_display,
                        score_mode="knn",
                        gd_score=None,
                        embedding_siglip=emb_siglip_raw,
                        refs_by_label_siglip=refs_by_label_siglip_arr or None,
                        class_thresholds=class_thresholds or None,
                        ambiguous_margin=_CHARLIE_AMBIGUOUS_MARGIN,
                    )
                except Exception as e:
                    print(f"[label-charlie] resolver failed for one detection: {e}")
                    continue
                # Merge resolver output into the detection — keys
                # mirror what V2's runner writes so the FE pipeline
                # popup + reject-reason explanations work without any
                # wire-shape changes on Charlie's side.
                for k_v in (
                    "pred_label", "pred_source",
                    "embed_nearest_label", "embed_nearest_sim",
                    "embed_sim_for_label", "embed_margin",
                    "rejected", "reject_reason",
                    "ambiguous", "vlm_action",
                    "siglip_weight",
                    "patch_match_used", "patch_match_used_siglip",
                ):
                    if k_v in verdict:
                        d[k_v] = verdict[k_v]
                if verdict.get("sims") is not None:
                    d["embed_sims"] = verdict["sims"]
                if verdict.get("sims_dino") is not None:
                    d["embed_sims_dino"] = verdict["sims_dino"]
                if verdict.get("sims_siglip") is not None:
                    d["embed_sims_siglip"] = verdict["sims_siglip"]
            timings["resolve_ms"] = round((time.perf_counter() - t_resolve) * 1000.0, 1)

            # VLM tiebreak. Detections the resolver flagged ambiguous
            # (top-1 vs top-2 embed sim within V2_AMBIGUOUS_MARGIN —
            # default 0.005) are coin-flip calls. Restrict Qwen-VL to
            # those two labels and let it pick — same logic V2 uses
            # for its specific path. Confident embed decisions skip
            # the VLM entirely so the per-image cost stays low.
            ambiguous_indices = [
                idx for idx, d in enumerate(detections)
                if d.get("ambiguous") and not d.get("rejected")
            ]
            if ambiguous_indices:
                try:
                    from vlm_validate import vlm_classify as _vlm_classify
                except Exception as e:
                    print(f"[label-charlie] vlm_validate import failed: {e}")
                    _vlm_classify = None
                if _vlm_classify is not None:
                    print(
                        f"[label-charlie] {import_id}: VLM tiebreak on "
                        f"{len(ambiguous_indices)} ambiguous detection(s)"
                    )
                    t_vlm = time.perf_counter()
                    vlm_total_ms = 0.0
                    async with state["gpu_lock"]:
                        for idx in ambiguous_indices:
                            d = detections[idx]
                            sims_dict = d.get("embed_sims") or {}
                            if len(sims_dict) < 2:
                                continue
                            # Top-2 labels by combined sim. Keys are
                            # the project's display labels (preserved
                            # case) so they pass straight to vlm_classify.
                            top2 = sorted(
                                sims_dict.items(), key=lambda kv: -kv[1],
                            )[:2]
                            top2_labels = [lab for lab, _ in top2]
                            bx = d.get("box")
                            mask_obj = d.get("mask")
                            mask_polys = (
                                mask_obj.get("polygons")
                                if isinstance(mask_obj, dict) else None
                            )
                            t_call = time.perf_counter()
                            try:
                                v_label, v_score = await loop.run_in_executor(
                                    None,
                                    _vlm_classify,
                                    pil, bx, top2_labels, mask_polys,
                                )
                            except Exception as e:
                                print(f"[label-charlie] VLM tiebreak failed for box {bx}: {e}")
                                v_label, v_score = None, None
                            vlm_ms = (time.perf_counter() - t_call) * 1000.0
                            vlm_total_ms += vlm_ms
                            d["vlm_label"] = v_label
                            d["vlm_score"] = v_score
                            d["vlm_ms"] = round(vlm_ms, 1)

                            # Confidence-weighted fusion: the VLM
                            # casts a soft vote scaled by its own
                            # confidence rather than overriding the
                            # embedding outright. Lets weak VLM
                            # answers nudge tight embed margins
                            # without strong VLM picks tipping
                            # decisive embed cases.
                            winner, fused_margin, fused_scores = _charlie_fuse_embed_vlm(
                                sims_dict, v_label, v_score,
                            )
                            d["fused_scores"] = {
                                k: round(float(v), 4) for k, v in fused_scores.items()
                            }
                            d["fused_margin"] = round(float(fused_margin), 4)
                            if winner is not None:
                                cur_pred = (d.get("pred_label") or "").strip().lower()
                                winner_lower = winner.strip().lower()
                                if winner_lower != cur_pred:
                                    d["pred_label"] = winner
                                    d["pred_source"] = "embed-vlm"
                                    d["vlm_action"] = "tiebreak"
                                else:
                                    d["pred_source"] = "embed-vlm"
                                    d["vlm_action"] = (
                                        "confirm" if v_label and v_label.strip().lower() == cur_pred
                                        else "tiebreak"
                                    )
                                # Refresh embed_sim_for_label to the
                                # winner's raw similarity.
                                for kk, vv in sims_dict.items():
                                    if kk.strip().lower() == winner_lower:
                                        d["embed_sim_for_label"] = round(float(vv), 4)
                                        break
                                # Re-evaluate ambiguity from the
                                # POST-fusion margin: even after
                                # combining both signals, if the
                                # winner only edges past the runner-
                                # up by less than CHARLIE_POST_FUSION
                                # _AMBIGUOUS_MARGIN we leave the
                                # detection flagged so the user gets
                                # a chance to review it.
                                d["ambiguous"] = (
                                    fused_margin < _CHARLIE_POST_FUSION_AMBIGUOUS_MARGIN
                                )
                    timings["vlm_ms"] = round(vlm_total_ms, 1)
                    timings["vlm_wall_ms"] = round((time.perf_counter() - t_vlm) * 1000.0, 1)
                    print(
                        f"[label-charlie] {import_id}: VLM tiebreak block "
                        f"{vlm_total_ms:.0f} ms across {len(ambiguous_indices)} call(s)"
                    )

            # Cross-label NMS. SAM3 prompted each project label
            # separately, so a single physical alpaca often comes
            # back as one detection under "alpaca" AND a near-duplicate
            # under "llama"; the resolver might assign each one a
            # different final label. Drop the lower-scoring member of
            # each high-IoU pair so the gallery shows one detection
            # per object. Runs after VLM so the dedupe uses post-
            # tiebreak labels + scores.
            t_nms = time.perf_counter()
            detections = _charlie_nms_post_resolve(detections)
            timings["nms_ms"] = round((time.perf_counter() - t_nms) * 1000.0, 1)

            # Specific datasets never hard-reject. Any detection the
            # resolver would have rejected (low ref similarity, signal
            # disagreement, etc.) is surfaced as "unsure" instead — kept
            # with its best-guess label + the amber Unsure chip so the
            # user reviews it rather than silently losing the box. Runs
            # after NMS so overlapping duplicates are already gone (we
            # don't resurrect a rejected duplicate of a kept box).
            for d in detections:
                if d.get("rejected"):
                    d["rejected"] = False
                    d["ambiguous"] = True
                    if not d.get("pred_label"):
                        d["pred_label"] = d.get("embed_nearest_label")

        # General-path NMS. SAM3 is run once per label so the
        # cross-label dedupe the specific path needs doesn't apply
        # here. But the same-label path occasionally fires duplicate
        # concept hits on one object (e.g. two "person" boxes covering
        # the same figure with slightly different masks). Drop the
        # lower-scoring member of each same-label IoU cluster so the
        # gallery shows one detection per object. Skipped on specific
        # since the cross-label NMS above already removes these as a
        # subset.
        if dataset_type != "specific" and detections:
            t_nms_g = time.perf_counter()
            detections = _charlie_nms_same_label(detections)
            # Cross-label NMS for multi-label general projects. SAM3 is
            # prompted once per label so "hare" and "rabbit" each fire on
            # the same physical animal → two near-identical detections.
            # _charlie_nms_post_resolve drops the lower-gd_score member of
            # each high-IoU pair regardless of label, so one detection per
            # object survives. Only needed when >1 label; single-label
            # projects can't have cross-label duplicates.
            if len(tags) > 1:
                # Stricter IoU on the GENERAL path: only merge near-
                # identical cross-label boxes (one object matched by two
                # sibling prompts). Two genuinely distinct overlapping
                # classes (plate+food, picture+frame) sit below 0.9 box
                # IoU and BOTH survive, instead of silently dropping the
                # lower-scoring one (general has no "unsure" fallback).
                detections = _charlie_nms_post_resolve(detections, iou_thresh=0.9)
            timings["nms_ms"] = round((time.perf_counter() - t_nms_g) * 1000.0, 1)

        # Stash the per-image update in memory; we flush every
        # FLUSH_EVERY images to amortise the manifest serialisation
        # cost. Per-image save_manifest used to dominate the loop on
        # a >100-image project because every save rewrote the entire
        # multi-MB manifest. Batching keeps the runner roughly
        # linear in image count instead of quadratic.
        pending_label_updates.append({
            "id": import_id,
            "detections": detections,
            "timings": timings,
            "width": pil.size[0],
            "height": pil.size[1],
        })
        if len(pending_label_updates) >= LABEL_FLUSH_EVERY:
            await _flush_label_updates()

        # Re-bake the labelled preview so the gallery thumb refreshes
        # next time the FE asks for it.
        try:
            loop.run_in_executor(
                None,
                _bake_labelled_preview_sync,
                job.project,
                import_id,
                path,
                list(detections),
                list(tags),
            )
        except Exception as e:
            print(f"[label-charlie] {import_id}: preview bake schedule failed: {e}")

        n_done += 1
        # Post-emit clears `image` so the FE drops the labelling
        # overlay between images instead of leaving it stuck on the
        # last completed tile. The next iteration's pre-emit names
        # the next file in flight.
        await emit("progress", {
            "index": n_done,
            "total": len(pending),
            "image": None,
            "n_detections": len(detections),
        })
        # Yield to the event loop between images. Matches the same
        # pattern in _run_augment_generate_job — without this the
        # labelling runner hogged the loop for the full duration of
        # a 941-image run, blocking concurrent image fetches /
        # click-to-detect calls from other tabs.
        await asyncio.sleep(0)

    # Final flush so any tail-end updates that didn't reach the
    # FLUSH_EVERY threshold still land before the job emits done. Only a
    # run that COMPLETED (not cancelled) marks the label set as fully run
    # (labelsLastRun / settingsLastRun); a cancelled job leaves the FE's
    # freshLabels intact so the user can re-trigger the unfinished label.
    await _flush_label_updates(mark_run=not cancel_event.is_set())

    # Auto-augment-after-labelling hook removed by user request —
    # augmentations now only run when the user explicitly clicks
    # the Update button in the Augmentations card. The previous
    # implementation scheduled augment_generate at the end of every
    # label_charlie run, which tied up the worker (and on big
    # projects could queue behind the labelling job's own flushes),
    # and dragged out the perceived time-to-first-label.

    # Cache invalidation at completion. /annotations + /overview both
    # sit behind a stale-while-revalidate cache; without this the
    # FE's immediate post-completion syncAnnotations call gets the
    # PRE-completion snapshot back (and a background rebuild fires
    # too late to help that response), which leaves freshly-labelled
    # tiles flashing Unlabelled until the user hard-refreshes the
    # page.
    try:
        _invalidate_project_payloads(job.project)
    except Exception as e:
        print(f"[label-charlie] payload invalidation failed: {e}")

    # Sidecar suppression is lifted + the final rebuild fired by the
    # wrapper's finally block (so it runs even if the job raised).
    # _invalidate_project_payloads above already deleted the stale sidecar
    # files; the wrapper's _kick_sidecar_refresh rebuilds them so the next
    # /overview / /initial GET serves from disk, not a request-thread synth.


async def _run_purge_label_job(job, emit, cancel_event):
    """Strip every reference to a single label from a project — across
    all imports' detections + editedBoxes, plus the manifest's tags /
    labelColours / label_aliases / verdicts. Emits per-image progress
    so the FE can render a labelling-style progress card. Survives a
    browser refresh because the JobManager hydrates from the audit
    log on startup; the user reattaches via /jobs/active on mount.

    Per-image work happens INSIDE the per-project manifest write lock
    so concurrent label / augment jobs can't clobber the strip pass.
    """
    proj = project_dir(job.project)
    if not proj.exists():
        raise RuntimeError("project not found")

    target_raw = str((job.params or {}).get("label") or "")
    target_key = target_raw.strip().lower()
    if not target_key:
        raise RuntimeError("label param required")

    write_lock = await _manifest_write_lock(job.project)
    async with write_lock:
        manifest = load_manifest(job.project) or {}
        imports_snapshot = list(manifest.get("imports") or [])
    job.n_images = len(imports_snapshot)
    n_total = len(imports_snapshot)
    job.progress = {
        "index": 0,
        "total": n_total,
        "image": None,
        "phase": "purging",
    }

    def _label_of(d: dict) -> str:
        # Detections store the canonical under any of these keys
        # depending on which pipeline produced them. Charlie writes
        # `gd_label`, the V2 resolver overwrites with `pred_label`,
        # edited boxes use `label`, and the FE camelCase forms can
        # land on disk too if the FE ever serialised before the
        # backend re-write.
        v = (
            d.get("label")
            or d.get("pred_label")
            or d.get("predLabel")
            or d.get("gd_label")
            or d.get("gdLabel")
        )
        return str(v or "").strip().lower()

    n_processed = 0
    n_removed_total = 0
    for imp_snap in imports_snapshot:
        if cancel_event.is_set():
            break
        import_id = imp_snap.get("id")
        filename = imp_snap.get("filename")
        if not import_id:
            n_processed += 1
            continue
        try:
            async with write_lock:
                m_now = load_manifest(job.project) or {}
                target_imp = None
                for entry in (m_now.get("imports") or []):
                    if entry.get("id") == import_id:
                        target_imp = entry
                        break
                if target_imp is not None:
                    dets = list(target_imp.get("detections") or [])
                    new_dets = [d for d in dets if _label_of(d) != target_key]
                    # Only touch editedBoxes if it ALREADY exists on
                    # this import — otherwise we'd persist an empty
                    # list for an image that never had user edits,
                    # which makes the FE treat it as "user cleared
                    # everything" and flag the tile Unlabelled until
                    # a refresh reads the tile from /overview again.
                    had_edited_field = isinstance(target_imp.get("editedBoxes"), list)
                    edited = list(target_imp.get("editedBoxes") or []) if had_edited_field else []
                    new_edited = [
                        b for b in edited
                        if str((b or {}).get("label") or "").strip().lower() != target_key
                    ]
                    removed_dets = len(dets) - len(new_dets)
                    removed_edited = (len(edited) - len(new_edited)) if had_edited_field else 0
                    removed = removed_dets + removed_edited
                    if removed > 0:
                        target_imp["detections"] = new_dets
                        # Only write editedBoxes back when this image
                        # already had user edits. Empty edited result
                        # is fine to persist — that's the user having
                        # explicitly cleared a labelled image — but a
                        # MISSING editedBoxes must stay missing so
                        # the gallery's "is this user-edited?" gate
                        # doesn't misfire.
                        if had_edited_field:
                            target_imp["editedBoxes"] = new_edited
                            # Editedboxes existed → it was already
                            # user-set; keep that flag accurate.
                            target_imp["editedBoxesSet"] = True
                            target_imp["editedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                        m_now["updatedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                        save_manifest(job.project, m_now)
                        # Force the gallery to refetch the labelled
                        # preview JPEG next paint.
                        try:
                            _invalidate_labelled_preview(job.project, import_id)
                        except Exception:
                            pass
                        n_removed_total += removed
        except Exception as e:
            print(f"[purge_label] {import_id}: strip failed: {e}")

        n_processed += 1
        job.progress = {
            "index": n_processed,
            "total": n_total,
            "image": filename,
            "phase": "purging",
        }
        try:
            await emit("progress", {
                "index": n_processed,
                "total": n_total,
                "image": filename,
            })
        except Exception:
            pass
        # Yield so concurrent endpoints stay responsive.
        await asyncio.sleep(0)

    # Final pass — strip the label from the project-level fields so
    # the chip rail + colour map don't reanimate the term on next
    # mount. Done in one lock acquisition.
    async with write_lock:
        m_now = load_manifest(job.project) or {}
        tags = m_now.get("tags") or []
        m_now["tags"] = [t for t in tags if str(t).strip().lower() != target_key]
        colours = m_now.get("labelColours") or {}
        if target_key in colours:
            colours.pop(target_key, None)
            m_now["labelColours"] = colours
        aliases = m_now.get("label_aliases") or {}
        if target_key in aliases:
            aliases.pop(target_key, None)
            m_now["label_aliases"] = aliases
        m_now["updatedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        save_manifest(job.project, m_now)
    try:
        _invalidate_project_payloads(job.project)
    except Exception:
        pass

    print(
        f"[purge_label] {job.project}: stripped label={target_raw!r} "
        f"from {n_processed}/{n_total} images, removed {n_removed_total} entries"
    )


@app.get("/api/health")
async def health():
    return {"ok": True, "device": state["device"], "model_loaded": state["model"] is not None}


def _charlie_reference_detections(image_pil, tags) -> "list[dict] | None":
    """Run the SAM3 (Charlie) pipeline on a reference image and map the
    output to the reference response shape [{label, score, box, mask}].

    This makes reference onboarding use the SAME detector as the project
    import labelling (label_charlie) instead of the old GroundingDINO +
    SAM2 path. Returns None when Charlie isn't loaded so callers fall
    back to GD+SAM2. Boxes/masks are in `image_pil`'s coordinate space;
    the caller scales back if it downsized first.
    """
    charlie = state.get("charlie")
    if charlie is None:
        return None
    dets_raw, _timings = charlie.segment_labels(image_pil, list(tags), include_crops=False)
    # Cross-label NMS. SAM3 prompts every label separately, so one
    # physical object (e.g. a single hare) comes back as overlapping
    # detections under BOTH "hare" and "rabbit". Drop the lower-gd_score
    # member of each high-IoU pair so only the highest-probability label
    # survives per object. Only needed with >1 label.
    if len(list(tags)) > 1 and len(dets_raw) > 1:
        dets_raw = _charlie_nms_post_resolve(dets_raw)
    out: list[dict] = []
    for d in dets_raw:
        box = d.get("box") or []
        if not (isinstance(box, list) and len(box) == 4):
            continue
        out.append({
            # SAM3's text-prompt match IS the label (same as the
            # general label_charlie path — no reference resolver here).
            "label": d.get("gd_label") or d.get("pred_label"),
            "score": round(float(d.get("gd_score") or 0.0), 4),
            "box": [float(c) for c in box],
            "mask": d.get("mask"),
        })
    return out


@app.post("/api/v2/references/process")
async def v2_process_reference(
    image: UploadFile = File(...),
    labels: str = Form(...),
    box_thr: float = Form(0.35),
    text_thr: float = Form(0.25),
    nms_iou: float = Form(0.50),
    force_label: str = Form(""),
):
    """Detect + segment a single reference image, no VLM, no project
    context. Prefers SAM3 (Charlie) — the same detector as the project
    import labelling — and falls back to GroundingDINO + SAM2 when SAM3
    isn't loaded.

    Request: multipart with the image file + a JSON-encoded list of
    label strings + optional thresholds (defaults match the V2
    annotations panel's "Normal" mode; ignored on the SAM3 path).

    Response: `{ width, height, detections: [{ label, score, box, mask }] }`
    where `box` is `[x1, y1, x2, y2]` in pixel coords and `mask` is
    `{ polygons: [[[x, y], ...], ...] }` (or null if no usable mask).
    """
    # Need SAM3 (charlie) OR the GD+SAM2 pair. 503 only if neither path
    # is available.
    if state.get("charlie") is None and (
        state.get("model") is None or state.get("segmenter") is None
    ):
        raise HTTPException(503, "no detection pipeline loaded (SAM3 + GD/SAM2 both unavailable)")

    # Parse the labels JSON.
    try:
        tag_list = json.loads(labels)
        if not isinstance(tag_list, list) or not all(isinstance(t, str) for t in tag_list):
            raise ValueError("labels must be a JSON array of strings")
    except Exception as e:
        raise HTTPException(400, f"invalid labels payload: {e}")
    tags = [t.strip() for t in tag_list if t and t.strip()]
    if not tags:
        # No labels means nothing to detect — fast-path empty result.
        data = await image.read()
        with PILImage.open(io.BytesIO(data)) as pil:
            pil = pil.convert("RGB")
            W, H = pil.size
        return {"width": W, "height": H, "detections": []}

    # Read the upload + open as PIL.
    raw = await image.read()
    if not raw:
        raise HTTPException(400, "empty image upload")

    # NSFW gate — same classifier as import endpoints. Blocks NSFW
    # reference images before any GPU work is done.
    _enforce_nsfw_or_451(raw, label="v2-ref-process", file=image.filename or "")

    try:
        image_pil = PILImage.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"could not decode image: {e}")

    # Cap resolution at 1280px longest edge so GD doesn't spend 5+
    # extra seconds processing a 4K photo. Detections are scaled back
    # to original coords before returning so the FE sees pixel-accurate
    # boxes regardless of whether we downsampled.
    _MAX_REF_DIM = 1280
    W_orig, H_orig = image_pil.size
    scale = 1.0
    if max(W_orig, H_orig) > _MAX_REF_DIM:
        scale = _MAX_REF_DIM / max(W_orig, H_orig)
        new_w = max(1, int(round(W_orig * scale)))
        new_h = max(1, int(round(H_orig * scale)))
        image_pil = image_pil.resize((new_w, new_h), PILImage.LANCZOS)
    W, H = image_pil.size

    loop = asyncio.get_running_loop()

    def _infer():
        try:
            # SAM3 (Charlie) path — same detector as the project's import
            # labelling. Preferred when loaded; falls back to GD+SAM2.
            charlie_dets = _charlie_reference_detections(image_pil, tags)
            if charlie_dets is not None:
                print(f"[v2-ref] SAM3 tags={tags} → {len(charlie_dets)} det(s) size={W}x{H} (orig {W_orig}x{H_orig})")
                if scale != 1.0:
                    for d in charlie_dets:
                        d["box"] = [round(c / scale, 2) for c in d["box"]]
                        mk = d.get("mask")
                        if mk and isinstance(mk, dict):
                            polys = mk.get("polygons") or []
                            d["mask"] = {
                                **mk,
                                "polygons": [
                                    [[pt[0] / scale, pt[1] / scale] for pt in poly]
                                    for poly in polys
                                ],
                            }
                else:
                    for d in charlie_dets:
                        d["box"] = [round(c, 2) for c in d["box"]]
                return charlie_dets
            # GD+SAM2 fallback removed in the portable build — SAM3 is the
            # only reference detector. None here means it isn't loaded yet.
            raise RuntimeError("SAM3 not loaded — reference processing unavailable")
        except Exception as exc:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"v2 reference pipeline failed: {exc}") from exc

    # Interactive GPU priority — the user is watching a progress
    # spinner and reference processing blocks onboarding. Jumps ahead
    # of background label_charlie / augment jobs.
    try:
        async with state["gpu_lock"].interactive():
            detections = await loop.run_in_executor(None, _infer)
    except Exception as exc:
        raise HTTPException(500, f"pipeline error: {exc}")

    # Section-scoped reference: the caller dropped this image into a
    # specific label's section, so the label is known. The detector is
    # used only to localise the object; force every detection onto the
    # section label rather than letting it classify among siblings.
    _force = (force_label or "").strip()
    if _force:
        for d in detections:
            d["label"] = _force

    return {"width": W_orig, "height": H_orig, "detections": detections}


@app.post("/api/v2/projects")
async def v2_create_project(
    name: str = Form(...),
    labels: str = Form(...),
    label_colours: str = Form("{}"),
    is_private: str = Form("false"),
    user: str = Depends(current_user),
):
    """Create a V2 project on the backend with a UUID + manifest. Returns
    `{project_id, name}`. Owner is forced to the authenticated user
    (the frontend can't spoof someone else's account by sending a
    different `owner`). References are uploaded separately via
    /api/v2/projects/{id}/references."""
    try:
        label_list = json.loads(labels)
        if not isinstance(label_list, list):
            raise ValueError("labels must be a JSON array")
    except Exception as e:
        raise HTTPException(400, f"invalid labels payload: {e}")
    tag_list = [str(l).strip() for l in label_list if str(l).strip()]

    # Profanity gate on the project name + every label in one pass.
    # assert_clean raises HTTPException(400) with the offending term;
    # we re-raise with a clearer prefix so the FE error toast tells
    # the user which field tripped.
    from profanity import assert_clean
    try:
        assert_clean(name, field="project name")
    except HTTPException as e:
        raise HTTPException(e.status_code, f"project name: {e.detail}") from None
    for lab in tag_list:
        try:
            assert_clean(lab, field="label")
        except HTTPException as e:
            raise HTTPException(e.status_code, f"label '{lab}': {e.detail}") from None

    colour_map: dict[str, str] = {}
    try:
        raw_colours = json.loads(label_colours) if label_colours else {}
        if isinstance(raw_colours, dict):
            for k, v in raw_colours.items():
                key = str(k).strip().lower()
                val = str(v).strip()
                # Loose hex validation — anything malformed silently
                # drops, the frontend will derive a fallback colour.
                if key and val.startswith("#") and 4 <= len(val) <= 9:
                    colour_map[key] = val
    except Exception:
        colour_map = {}

    project_id = _uuid.uuid4().hex
    proj = store.create_dataset_dir(project_id, name)
    (proj / "references").mkdir(exist_ok=True)

    manifest = empty_manifest(name, owner=user, project_id=project_id)
    manifest["v2"] = True
    manifest["tags"] = tag_list
    manifest["references"] = []
    if colour_map:
        manifest["labelColours"] = colour_map
    # Visibility — onboarding toggle. Same {private: true|false} flag
    # the settings page sets via PUT; baking it in at creation time
    # saves a follow-up round-trip.
    if str(is_private).strip().lower() in ("true", "1", "yes"):
        manifest["private"] = True
    save_manifest(project_id, manifest)
    return {"project_id": project_id, "name": name}


@app.post(
    "/api/v2/projects/{project_id}/derive",
    dependencies=[Depends(require_project_owner)],
)
async def v2_derive_project(
    project_id: str,
    name: str = Form(...),
    labels: str = Form("[]"),
    padding: str = Form("0.15"),
    min_size: str = Form("256"),
    square: str = Form("false"),
    fixed_size: str = Form("0"),
    label_mode: str = Form("inherit"),
    create_project: str = Form("true"),
    user: str = Depends(current_user),
):
    """Create a CHILD project: per-detection crops of this (parent) project for
    the selected labels — one image + one box + one label each. One-way linked:
    the child re-syncs from the parent and never writes back."""
    parent = load_manifest(project_id, copy=False)
    if not parent:
        raise HTTPException(404, "parent project not found")
    # No derivatives of derivatives: a derived (child) dataset can't itself be
    # derived. A parent may have MANY children, but the lineage stays one level
    # deep (a crop-of-a-crop is meaningless and would double-suppress edits).
    if parent.get("derived"):
        raise HTTPException(400, "Cannot create a derived dataset from another derived dataset.")
    try:
        sel = [str(l).strip() for l in json.loads(labels or "[]") if str(l).strip()]
    except Exception:
        raise HTTPException(400, "invalid labels payload")
    from profanity import assert_clean
    try:
        assert_clean(name, field="project name")
    except HTTPException as e:
        raise HTTPException(e.status_code, f"project name: {e.detail}") from None
    try:
        pad = max(0.0, min(1.0, float(padding)))
    except Exception:
        pad = 0.15
    try:
        min_px = max(0, min(1024, int(float(min_size))))
    except Exception:
        min_px = 256
    # Optional fixed crop size: every crop is resized to exactly fixed_px square.
    # 0 = off (crops keep their natural per-detection size).
    try:
        fixed_px = max(0, min(1024, int(float(fixed_size))))
    except Exception:
        fixed_px = 0
    square_mode = str(square).strip().lower() in ("1", "true", "yes", "on")
    # "new" → child starts with a blank label vocabulary (user makes their own);
    # "inherit" (default) → child carries the parent's labels.
    new_labels = str(label_mode).strip().lower() == "new"
    # Whether to group the child (and parent) under a workspace Project
    # (container). Off → the child is created as a standalone dataset.
    make_project = str(create_project).strip().lower() in ("1", "true", "yes", "on")

    child_id = _uuid.uuid4().hex
    proj = store.create_dataset_dir(child_id, name)
    (proj / "references").mkdir(exist_ok=True)
    (proj / "images").mkdir(exist_ok=True)

    child = empty_manifest(name, owner=user, project_id=child_id)
    child["v2"] = True
    # "new" label mode → blank vocabulary (the user creates their own labels);
    # otherwise inherit the selected (or all) parent labels.
    child["tags"] = [] if new_labels else (sel or list(parent.get("tags") or []))
    child["references"] = []
    # If the parent is a SPECIFIC dataset (has reference images), pull its
    # references through so the child is specific too — copy the ref entries
    # (with their detection embeddings) + the image files, keeping only refs
    # relevant to the selected labels. Skipped entirely in "new" label mode:
    # the parent's references are tied to the parent's labels, which we're
    # discarding, so copying them across would be meaningless.
    parent_refs = [] if new_labels else (parent.get("references") or [])
    if parent_refs:
        import shutil as _sh
        lower_sel = {s.lower() for s in sel}
        src_refs = project_dir(project_id) / "references"
        dst_refs = project_dir(child_id) / "references"
        dst_refs.mkdir(parents=True, exist_ok=True)
        kept_refs = []
        for ref in parent_refs:
            if not isinstance(ref, dict):
                continue
            ref_labels = {str(ref.get("label") or "").lower()}
            for d in (ref.get("detections") or []):
                if isinstance(d, dict):
                    ref_labels.add(str(d.get("label") or d.get("predLabel") or d.get("gd_label") or "").lower())
            if lower_sel and not (ref_labels & lower_sel):
                continue  # ref doesn't touch any selected label
            fn = ref.get("filename")
            if fn:
                try:
                    _sh.copy2(src_refs / fn, dst_refs / fn)
                except Exception:
                    continue  # missing source file — skip this ref
            kept_refs.append(dict(ref))
        child["references"] = kept_refs
    # Carry the parent's label colours only when inheriting labels; "new" mode
    # starts with a clean palette.
    pcol = parent.get("labelColours")
    if not new_labels and isinstance(pcol, dict):
        lower_sel = {s.lower() for s in sel}
        child["labelColours"] = {k: v for k, v in pcol.items() if not sel or str(k).lower() in lower_sel}
    # Derived datasets are private by default — they're a working copy of the
    # parent, not something to publish to the community feed.
    child["private"] = True
    child["derived"] = {
        "parentProjectId": project_id,
        "parentName": parent.get("name") or "",
        # `labels` is the SELECTION of parent labels to crop (the sync filter),
        # independent of whether the child keeps or replaces the label text.
        "labels": sel,
        "labelMode": "new" if new_labels else "inherit",
        "crop": {"padding": pad, "minSize": min_px, "square": square_mode, "fixedSize": fixed_px},
        "suppressed": [],
    }
    save_manifest(child_id, child)
    _register_child(project_id, child_id)

    # Group derived datasets under a Project (container) alongside their parent
    # so the workspace keeps related datasets together. If the parent is already
    # in a Project the child joins it; otherwise auto-create a Project named
    # after the parent (privacy = the parent's own, so the parent's visibility is
    # unchanged) and add both. Best-effort: a failure here never blocks the
    # derive itself. Skipped entirely when create_project is off — the child is
    # then left as a standalone dataset (not in any Project).
    if make_project:
        try:
            parent_cid = (parent.get("container_id") or "").strip()
            cont = containers.load_container(parent_cid) if parent_cid else None
            if cont is None:
                cont = containers.create_container(
                    parent.get("name") or name,
                    user,
                    private=bool(parent.get("private", True)),
                )
                add_event("container_create", container=cont["id"], actor=user, name=cont.get("name"))
                # Pull the standalone parent in. Its effective privacy is unchanged
                # because the new Project inherits the parent's own privacy.
                pm = load_manifest(project_id)
                if pm and not (pm.get("container_id") or "").strip():
                    pm["container_id"] = cont["id"]
                    pm["private"] = bool(cont.get("private"))
                    save_manifest(project_id, pm)
                    if project_id not in (cont.get("dataset_ids") or []):
                        cont.setdefault("dataset_ids", []).append(project_id)
                    add_event("dataset_add", container=cont["id"], dataset=project_id, actor=user)
            # Add the child and inherit the Project's privacy.
            cm = load_manifest(child_id)
            if cm:
                cm["container_id"] = cont["id"]
                cm["private"] = bool(cont.get("private"))
                save_manifest(child_id, cm)
            if child_id not in (cont.get("dataset_ids") or []):
                cont.setdefault("dataset_ids", []).append(child_id)
            containers.save_container(cont)
            add_event("dataset_add", container=cont["id"], dataset=child_id, actor=user)
        except Exception as e:
            print(f"[derived] auto-project grouping failed for {child_id}: {e}", flush=True)

    # Crop pass in the background so the request returns immediately; the child
    # appears and fills in as its crops are generated. Errors are logged (not
    # swallowed silently) so a failed crop pass is diagnosable.
    def _initial_crop() -> None:
        try:
            resync_child(child_id)
        except Exception as e:
            print(f"[derived] initial crop pass failed for {child_id}: {e}", flush=True)
    _threading.Thread(target=_initial_crop, daemon=True).start()
    return {"project_id": child_id, "name": name, "parent_project_id": project_id}


@app.post(
    "/api/v2/projects/{project_id}/resync",
    dependencies=[Depends(require_project_owner)],
)
async def v2_resync_child(project_id: str, user: str = Depends(current_user)):
    """Manually re-derive a child project from its parent's current state."""
    m = load_manifest(project_id, copy=False)
    if not (m.get("derived") or {}).get("parentProjectId"):
        raise HTTPException(400, "not a derived (child) project")
    if not resync_child(project_id):
        raise HTTPException(500, "resync failed")
    updated = load_manifest(project_id, copy=False)
    return {"ok": True, "n_images": len(updated.get("imports") or [])}


@app.get(
    "/api/v2/projects/{project_id}/children",
    dependencies=[Depends(require_project_owner)],
)
async def v2_list_children(project_id: str):
    """The derived (child) projects of this parent — for the parent's 'Derived
    datasets' list."""
    _ensure_derived_index()
    out = []
    for cid in _children_of(project_id):
        try:
            m = load_manifest(cid, copy=False)
        except Exception:
            continue
        if (m.get("derived") or {}).get("parentProjectId") != project_id:
            continue
        out.append({
            "project_id": cid,
            "name": m.get("name") or "",
            "labels": (m.get("derived") or {}).get("labels") or [],
            "n_images": len(m.get("imports") or []),
        })
    out.sort(key=lambda c: str(c["name"]).lower())
    return {"children": out}


@app.post(
    "/api/v2/projects/{project_id}/references",
    dependencies=[Depends(require_project_owner)],
)
async def v2_upload_reference(
    project_id: str,
    image: UploadFile = File(...),
    detections: str = Form("[]"),
    width: int = Form(0),
    height: int = Form(0),
    labels: str = Form(""),
    label: str = Form(""),
    box_thr: float = Form(0.35),
    text_thr: float = Form(0.25),
    nms_iou: float = Form(0.50),
):
    """Upload a reference image to a V2 project, run GD+SAM if no
    pre-computed detections were supplied, persist DINOv2 embeddings
    inline on each detection, then append the result to the
    manifest's `references` array.

    Two call modes:

      a) Pre-computed (legacy): caller passes `detections` (already
         from /api/v2/references/process) — server skips inference.
      b) Single-shot: caller passes `labels` (a JSON list of label
         strings) and an empty `detections` — server runs GD+SAM
         inline, replacing the now-deprecated /process round-trip.
         This is what the V2 onboarding does so each reference image
         hits the server exactly once instead of twice.

    In both modes the server embeds each kept detection's bbox via
    DINOv2 and stores the embedding next to the detection so future
    reopens skip the embed-on-every-mount round-trip.
    """
    proj = project_dir(project_id)
    if not proj.exists():
        raise HTTPException(404, "project not found")
    # Existence check only — the actual manifest used for the
    # mutate+save lives inside the manifest_write lock at the
    # bottom of the handler so concurrent uploads don't clobber
    # each other's appends.
    if load_manifest(project_id) is None or load_manifest(project_id) == {}:
        raise HTTPException(404, "manifest not found")

    raw = await image.read()
    if not raw:
        raise HTTPException(400, "empty image upload")

    refs_dir = proj / "references"
    refs_dir.mkdir(exist_ok=True)
    ref_id = _uuid.uuid4().hex
    ext = Path(image.filename or "image").suffix or ".jpg"
    stored_name = f"{ref_id}{ext}"
    (refs_dir / stored_name).write_bytes(raw)

    try:
        det_list = json.loads(detections)
        if not isinstance(det_list, list):
            raise ValueError("detections must be a JSON array")
    except Exception as e:
        raise HTTPException(400, f"invalid detections payload: {e}")

    # Decode the image once so we can both read its dimensions AND
    # crop each box for DINOv2-base embedding. Embeddings are stored
    # inline alongside the detection so the imports endpoint can
    # compute per-label centroids on-the-fly without re-running the
    # encoder on every reference.
    try:
        image_pil = PILImage.open(io.BytesIO(raw)).convert("RGB")
        if not (width and height):
            width, height = image_pil.size
    except Exception:
        image_pil = None

    # Single-shot mode: caller didn't pre-compute detections, so do
    # the inference here. Mirrors /api/v2/references/process inline.
    if not det_list and labels and image_pil is not None:
        try:
            tag_list = json.loads(labels)
            if not isinstance(tag_list, list):
                raise ValueError("labels must be a JSON array")
        except Exception as e:
            raise HTTPException(400, f"invalid labels payload: {e}")
        tags = [str(t).strip() for t in tag_list if str(t).strip()]
        _have_charlie = state.get("charlie") is not None
        _have_gd_sam = state.get("model") is not None and state.get("segmenter") is not None
        if tags and (_have_charlie or _have_gd_sam):
            loop = asyncio.get_running_loop()

            def _detect():
                # Prefer SAM3 (Charlie) so reference onboarding uses the
                # same detector as the project import labelling; fall
                # back to GD+SAM2 when SAM3 isn't loaded.
                charlie_dets = _charlie_reference_detections(image_pil, tags)
                if charlie_dets is not None:
                    for d in charlie_dets:
                        d["box"] = [round(c, 2) for c in d["box"]]
                    print(f"[v2-ref-upload] inline SAM3 tags={tags} → {len(charlie_dets)} det(s)")
                    return charlie_dets
                # GD+SAM2 fallback removed — no detections when SAM3 isn't
                # loaded; the whole-image fallback box below still applies.
                return []

            try:
                async with state["gpu_lock"].interactive():
                    det_list = await loop.run_in_executor(None, _detect)
            except Exception as exc:
                import traceback
                traceback.print_exc()
                raise HTTPException(500, f"inline reference detect failed: {exc}")

    # Section-scoped upload: the reference was dropped into a specific
    # label's section, so the label is known up front. Force every
    # detection onto that label (the detector only localises here), and
    # if nothing was found, fall back to a whole-image box so the
    # reference still gets an embedding and a quality verdict (the
    # "looks like other class" check needs at least one embedded crop).
    section_label = (label or "").strip()
    if section_label:
        for d in det_list:
            d["label"] = section_label
        if not det_list and image_pil is not None:
            W_full, H_full = image_pil.size
            det_list = [{
                "label": section_label,
                "score": 0.0,
                "box": [0.0, 0.0, float(W_full), float(H_full)],
                "mask": None,
            }]

    if image_pil is not None and det_list:
        import v2_dinov2
        import v2_siglip
        if v2_dinov2.is_loaded():
            W, H = image_pil.size
            squares: list[PILImage.Image] = []
            indices: list[int] = []
            for i, d in enumerate(det_list):
                bb = d.get("box") or []
                if not (isinstance(bb, list) and len(bb) == 4):
                    continue
                try:
                    x0 = max(0, int(round(float(bb[0]))))
                    y0 = max(0, int(round(float(bb[1]))))
                    x1 = min(W, int(round(float(bb[2]))))
                    y1 = min(H, int(round(float(bb[3]))))
                except (TypeError, ValueError):
                    continue
                if x1 - x0 < 4 or y1 - y0 < 4:
                    continue
                # Inpaint occluders inside the bbox using SAM's
                # mask polygons so the embedding reflects only the
                # object, not whatever is standing in front of it.
                mask_polys = None
                m = d.get("mask")
                if isinstance(m, dict):
                    mask_polys = m.get("polygons")
                crop = v2_dinov2.inpaint_bbox_crop(image_pil, (x0, y0, x1, y1), mask_polys)
                squares.append(v2_dinov2.center_square_crop(crop))
                indices.append(i)
            if squares:
                vecs = v2_dinov2.encode_images_batch(squares)
                # SigLIP2 ensemble: encode the same crops via SigLIP
                # and store alongside the DINOv2 embedding. The
                # resolver scores both independently and combines.
                sig_vecs = (
                    v2_siglip.encode_images_batch(squares)
                    if v2_siglip.is_loaded() else None
                )
                for k, i in enumerate(indices):
                    det_list[i]["embedding"] = [round(float(x), 6) for x in vecs[k].tolist()]
                    det_list[i]["embed_version"] = v2_dinov2.EMBED_VERSION
                    if sig_vecs is not None and k < sig_vecs.shape[0]:
                        det_list[i]["siglip_embedding"] = [round(float(x), 6) for x in sig_vecs[k].tolist()]
                        det_list[i]["siglip_version"] = v2_siglip.EMBED_VERSION
                print(
                    f"[v2-ref-upload] embedded {len(indices)} of {len(det_list)} detection(s) "
                    f"for ref {ref_id} (dinov{v2_dinov2.EMBED_VERSION}"
                    + (f" + siglipv{v2_siglip.EMBED_VERSION}" if sig_vecs is not None else " — siglip deferred")
                    + ")"
                )
        else:
            print(f"[v2-ref-upload] DINOv2 not loaded yet — embeddings deferred (lazy backfill on first import).")

    # BlurHash placeholder so the FE can render a coloured gradient
    # before the real image bytes stream in. Encoded once at upload
    # time off the file we just saved (5-10 ms). Gracefully None if
    # encode fails; FE then falls back to a flat grey tile.
    blurhash_str = _encode_blurhash_from_path(refs_dir / stored_name)
    ref_entry = {
        "id": ref_id,
        "filename": stored_name,
        "originalFilename": Path(image.filename or "image").name,
        "width": int(width or 0),
        "height": int(height or 0),
        "detections": det_list,
        "label": section_label or None,
        "blurhash": blurhash_str,
        "createdAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    # Atomic append: per-project manifest write lock guards the
    # load → mutate → save sequence so two concurrent /references
    # POSTs can't each load a stale manifest and overwrite each
    # other's append. Reload INSIDE the lock so we always start
    # from the latest persisted state — the manifest variable
    # checked above was just for the existence guard.
    write_lock = await _manifest_write_lock(project_id)
    async with write_lock:
        manifest = load_manifest(project_id) or {}
        was_empty = len(manifest.get("references") or []) == 0
        manifest.setdefault("references", []).append(ref_entry)
        # Seed the project's cover from the first uploaded reference
        # so the workspace gallery has a thumbnail to render.
        if not manifest.get("cover"):
            manifest["cover"] = stored_name
        manifest["updatedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        save_manifest(project_id, manifest)
        n_refs = len(manifest.get("references") or [])
    # Going from zero → some reference images means this is a specific
    # dataset (see _apply_reference_dataset_flip). Done once on the first
    # upload, outside the manifest lock (it writes a separate sidecar).
    if was_empty:
        _apply_reference_dataset_flip(project_id)
    print(
        f"[v2-ref-upload] saved ref {ref_id} for project {project_id} "
        f"— manifest now has {n_refs} reference(s), cover={manifest.get('cover')!r}"
    )
    return {
        "reference_id": ref_id,
        "filename": stored_name,
        "width": width,
        "height": height,
        # Echo detections back so the FE doesn't have to re-fetch the
        # manifest after a single-shot upload to render boxes.
        # Embeddings are stripped from the response payload — they're
        # 1024-dim float arrays that bloat the wire and the FE can
        # hydrate them on next manifest GET if it actually needs them.
        "detections": [
            {k: v for k, v in d.items() if k not in ("embedding",)}
            for d in det_list
        ],
        "embedded": sum(1 for d in det_list if isinstance(d.get("embedding"), list)),
    }


@app.delete(
    "/api/v2/projects/{project_id}/references/{reference_id}",
    dependencies=[Depends(require_project_owner)],
)
async def v2_delete_reference(project_id: str, reference_id: str):
    """Remove a single reference image from the project. Called when the
    user deletes a ref during onboarding — the eager-upload path had
    already POSTed it to the server, so a local-only delete would leave
    an orphaned entry that would resurface on the next /initial hydration."""
    proj = project_dir(project_id)
    if not proj.exists():
        raise HTTPException(404, "project not found")
    write_lock = await _manifest_write_lock(project_id)
    async with write_lock:
        manifest = load_manifest(project_id)
        if not manifest:
            raise HTTPException(404, "manifest not found")
        refs = manifest.get("references") or []
        kept: list[dict] = []
        found = False
        for ref in refs:
            if ref.get("id") == reference_id:
                found = True
                fn = ref.get("filename")
                if fn:
                    try:
                        (proj / "references" / fn).unlink(missing_ok=True)
                    except Exception as exc:
                        print(f"[v2-ref-delete] couldn't unlink {fn}: {exc}")
            else:
                kept.append(ref)
        if not found:
            raise HTTPException(404, "reference not found")
        manifest["references"] = kept
        manifest["updatedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        save_manifest(project_id, manifest)
    return {"ok": True}


class UpdateRefDetectionsIn(BaseModel):
    detections: list[dict]


@app.put(
    "/api/v2/projects/{project_id}/references/{reference_id}",
    dependencies=[Depends(require_project_owner)],
)
async def v2_update_reference(project_id: str, reference_id: str, payload: UpdateRefDetectionsIn):
    """Replace a reference's detections in the manifest and (re-)embed
    any boxes whose `embedding` is missing or whose geometry/mask
    doesn't match the previously stored copy. Used when the user
    edits boxes in the V2 editor and clicks next/prev/close — the
    FE batches the resulting detection list here so the manifest
    stays consistent with what they see and we don't re-encode on
    every reopen.

    Boxes that already carry a valid `embedding` (length matches
    `v2_dinov2.EMBEDDING_DIM`) are kept as-is; new / changed ones
    are run through DINOv2 inline. Single GPU pass per request.
    """
    if not project_dir(project_id).exists():
        raise HTTPException(404, "project not found")
    # Existence check only — the actual manifest used for the
    # mutate+save lives inside the manifest_write lock at the
    # bottom of the handler so a concurrent /references POST
    # can't reset the references array between our load and save.
    snapshot = load_manifest(project_id)
    if not snapshot:
        raise HTTPException(404, "manifest not found")

    refs = snapshot.get("references") or []
    target = next((r for r in refs if r.get("id") == reference_id), None)
    if target is None:
        raise HTTPException(404, "reference not found")

    new_dets = payload.detections or []
    if not isinstance(new_dets, list):
        raise HTTPException(400, "detections must be a list")

    # Validate any text labels (matches V1 PUT's profanity check).
    from profanity import assert_clean
    for d in new_dets:
        if isinstance(d, dict) and isinstance(d.get("label"), str):
            assert_clean(d["label"], field="label")

    # Re-embed boxes that need it. Decode the stored image once.
    refs_dir = project_dir(project_id) / "references"
    img_path = refs_dir / target.get("filename", "")
    image_pil = None
    try:
        if img_path.exists():
            image_pil = PILImage.open(img_path).convert("RGB")
    except Exception:
        image_pil = None

    if image_pil is not None and new_dets:
        import v2_dinov2
        import v2_siglip
        dino_loaded = v2_dinov2.is_loaded()
        siglip_loaded = v2_siglip.is_loaded()
        if dino_loaded:
            W, H = image_pil.size
            embed_dim = v2_dinov2.EMBEDDING_DIM
            siglip_dim = getattr(v2_siglip, "EMBEDDING_DIM", None)
            # Build a unified work list: each detection that's missing
            # EITHER encoder's vector at the current version goes
            # through one shared crop, then we encode whichever
            # encoder(s) need it. Avoids re-cropping the same box
            # twice when both DINOv2 and SigLIP need a refresh.
            squares: list[PILImage.Image] = []
            indices: list[int] = []
            need_dino: list[bool] = []
            need_siglip: list[bool] = []
            for i, d in enumerate(new_dets):
                emb = d.get("embedding")
                emb_v = d.get("embed_version")
                d_have = (
                    isinstance(emb, list)
                    and len(emb) == embed_dim
                    and isinstance(emb_v, int)
                    and emb_v == v2_dinov2.EMBED_VERSION
                )
                s_have = True  # treat as satisfied when SigLIP isn't loaded
                if siglip_loaded:
                    s_emb = d.get("siglip_embedding")
                    s_v = d.get("siglip_version")
                    s_have = (
                        isinstance(s_emb, list)
                        and (siglip_dim is None or len(s_emb) == siglip_dim)
                        and isinstance(s_v, int)
                        and s_v == v2_siglip.EMBED_VERSION
                    )
                if d_have and s_have:
                    continue
                # Old-version embeddings get re-encoded so the manifest
                # stops mixing v3 (224×224 input) and v4 (518×518 input)
                # vectors — the dim is the same but the feature space
                # subtly drifts between input resolutions.
                bb = d.get("box") or []
                if not (isinstance(bb, list) and len(bb) == 4):
                    continue
                try:
                    x0 = max(0, int(round(float(bb[0]))))
                    y0 = max(0, int(round(float(bb[1]))))
                    x1 = min(W, int(round(float(bb[2]))))
                    y1 = min(H, int(round(float(bb[3]))))
                except (TypeError, ValueError):
                    continue
                if x1 - x0 < 4 or y1 - y0 < 4:
                    continue
                mask_polys = None
                m = d.get("mask")
                if isinstance(m, dict):
                    mask_polys = m.get("polygons")
                crop = v2_dinov2.inpaint_bbox_crop(image_pil, (x0, y0, x1, y1), mask_polys)
                squares.append(v2_dinov2.center_square_crop(crop))
                indices.append(i)
                need_dino.append(not d_have)
                need_siglip.append(siglip_loaded and not s_have)
            if squares:
                loop = asyncio.get_running_loop()

                # Encode under the GPU lock — sequential DINOv2 then
                # SigLIP keeps VRAM usage bounded vs. running them in
                # parallel (each model owns its own forward pass on
                # the same device).
                async with state["gpu_lock"]:
                    if any(need_dino):
                        d_vecs = await loop.run_in_executor(
                            None, lambda: v2_dinov2.encode_images_batch(squares)
                        )
                    else:
                        d_vecs = None
                    if any(need_siglip):
                        s_vecs = await loop.run_in_executor(
                            None, lambda: v2_siglip.encode_images_batch(squares)
                        )
                    else:
                        s_vecs = None
                d_count = 0
                s_count = 0
                for k, i in enumerate(indices):
                    if need_dino[k] and d_vecs is not None and k < d_vecs.shape[0]:
                        new_dets[i]["embedding"] = [
                            round(float(x), 6) for x in d_vecs[k].tolist()
                        ]
                        new_dets[i]["embed_version"] = v2_dinov2.EMBED_VERSION
                        d_count += 1
                    if need_siglip[k] and s_vecs is not None and k < s_vecs.shape[0]:
                        new_dets[i]["siglip_embedding"] = [
                            round(float(x), 6) for x in s_vecs[k].tolist()
                        ]
                        new_dets[i]["siglip_version"] = v2_siglip.EMBED_VERSION
                        s_count += 1
                print(
                    f"[v2-ref-update] re-embedded ref {reference_id} — "
                    f"dino={d_count}/{len(new_dets)}, siglip={s_count}/{len(new_dets)}"
                )

    # Atomic update under the per-project manifest write lock.
    # Re-resolve `target` from a fresh manifest copy so any
    # concurrent /references POSTs that landed between our snapshot
    # above and this point are preserved. If the reference was
    # deleted in the interim, return 404 cleanly instead of writing
    # a stale ref entry back.
    write_lock = await _manifest_write_lock(project_id)
    async with write_lock:
        manifest = load_manifest(project_id) or {}
        refs_live = manifest.get("references") or []
        live_target = next((r for r in refs_live if r.get("id") == reference_id), None)
        if live_target is None:
            raise HTTPException(404, "reference no longer exists (concurrent delete?)")
        live_target["detections"] = new_dets
        manifest["updatedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        save_manifest(project_id, manifest)
    n_embedded = sum(1 for d in new_dets if isinstance(d.get("embedding"), list))
    # Diagnostic: print the labels the FE flush actually sent so we
    # can tell whether onboarding label corrections are landing on
    # disk. If the manifest later loads with only one class, this
    # log line confirms whether the PUT received both classes or
    # the FE only flushed one.
    label_summary = {}
    for d in new_dets:
        lab = (d.get("label") or "").strip().lower() or "(blank)"
        label_summary[lab] = label_summary.get(lab, 0) + 1
    print(
        f"[v2-ref-update] saved ref {reference_id} project={project_id} "
        f"labels={label_summary} embedded={n_embedded}/{len(new_dets)}"
    )
    return {
        "reference_id": reference_id,
        "n_detections": len(new_dets),
        "n_embedded": n_embedded,
    }


@app.get(
    "/api/v2/projects/{project_id}/references/{filename}",
    dependencies=[Depends(require_project_read_access)],
)
async def v2_serve_reference(project_id: str, filename: str):
    """Serve a stored reference image, with RAM-cached bytes for
    repeat hits. Path-traversal guarded by resolving the full path
    and verifying it stays inside the project's references dir."""
    proj = project_dir(project_id)
    refs_root = (proj / "references").resolve()
    target = (refs_root / filename).resolve()
    try:
        target.relative_to(refs_root)
    except ValueError:
        raise HTTPException(403, "forbidden")
    return await _serve_cached_image(project_id, "references", filename, target)


# ─── V2 imports persistence ──────────────────────────────────────
# V1 stores every image bytes-and-detections in the manifest under
# `results`. V2 keeps the same idea but in a sibling list `imports`,
# so existing V1 endpoints / migrations can keep treating `results`
# as the canonical V1 list while V2 reads/writes `imports`.

@app.post(
    "/api/v2/projects/{project_id}/imports",
    dependencies=[Depends(require_project_owner)],
)
async def v2_upload_import(
    project_id: str,
    image: UploadFile = File(...),
    detections: str = Form("[]"),
    edited_boxes: str = Form(""),
    timings: str = Form("{}"),
    width: int = Form(0),
    height: int = Form(0),
):
    """Persist an imported image to a V2 project. Image bytes go to
    `projects/<id>/imports/<uuid>.<ext>`; the metadata + the FE-
    computed detections / editedBoxes / timings appended to
    manifest['imports'] so the project can be rehydrated on next
    open without re-running the pipeline."""
    proj = project_dir(project_id)
    if not proj.exists():
        raise HTTPException(404, "project not found")
    if not load_manifest(project_id):
        raise HTTPException(404, "manifest not found")

    raw = await image.read()
    if not raw:
        raise HTTPException(400, "empty image upload")

    imports_dir = proj / "images"
    imports_dir.mkdir(exist_ok=True)
    import_id = _uuid.uuid4().hex
    ext = Path(image.filename or "image").suffix or ".jpg"
    stored_name = f"{import_id}{ext}"
    (imports_dir / stored_name).write_bytes(raw)

    try:
        det_list = json.loads(detections) if detections else []
        if not isinstance(det_list, list):
            raise ValueError("detections must be a JSON array")
    except Exception as e:
        raise HTTPException(400, f"invalid detections payload: {e}")

    edited: list | None = None
    if edited_boxes:
        try:
            parsed = json.loads(edited_boxes)
            if isinstance(parsed, list):
                edited = parsed
        except Exception as e:
            print(f"[v2-import-upload] invalid edited_boxes ignored: {e}")

    try:
        timing_obj = json.loads(timings) if timings else {}
        if not isinstance(timing_obj, dict):
            timing_obj = {}
    except Exception:
        timing_obj = {}

    if not (width and height):
        try:
            image_pil = PILImage.open(io.BytesIO(raw)).convert("RGB")
            width, height = image_pil.size
        except Exception:
            pass

    # BlurHash placeholder for the dataset gallery — same pattern
    # as references: render a colour gradient until the image bytes
    # arrive (lazily, on scroll-into-view).
    blurhash_str = _encode_blurhash_from_path(imports_dir / stored_name)
    entry = {
        "id": import_id,
        "filename": stored_name,
        "originalFilename": Path(image.filename or "image").name,
        "width": int(width or 0),
        "height": int(height or 0),
        "detections": det_list,
        "editedBoxes": edited,
        "timings": timing_obj,
        "blurhash": blurhash_str,
        "createdAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    # Atomic append under the per-project manifest write lock so
    # concurrent /imports POSTs don't overwrite each other (same
    # bug as /references — load_manifest+mutate+save needs to be
    # an atomic critical section).
    write_lock = await _manifest_write_lock(project_id)
    async with write_lock:
        manifest = load_manifest(project_id) or {}
        manifest.setdefault("imports", []).append(entry)
        # Seed the project's cover from the first uploaded import if no
        # cover is set yet. V2 originally relied on references-flow to
        # set this, but Charlie / V3 projects skip references entirely
        # — without this fallback the workspace card stays on a 404
        # forever.
        if not manifest.get("cover"):
            manifest["cover"] = stored_name
        manifest["updatedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        save_manifest(project_id, manifest)
        project_tags_snapshot = list(manifest.get("tags") or [])
    # Bake the labelled preview off-thread so the FE's first
    # /labelled_preview GET is a pure file serve. Fire-and-forget;
    # the lazy GET path renders on demand if the bake hasn't
    # landed yet (slow disk, race) so this is purely a latency
    # optimisation, not a correctness requirement.
    loop = asyncio.get_running_loop()
    loop.run_in_executor(
        None,
        _bake_labelled_preview_sync,
        project_id,
        import_id,
        imports_dir / stored_name,
        list(edited) if isinstance(edited, list) and edited else list(det_list),
        project_tags_snapshot,
    )
    return {"import_id": import_id, "filename": stored_name, "width": width, "height": height}


@app.post(
    "/api/v2/projects/{project_id}/imports/raw",
    dependencies=[Depends(require_project_owner)],
)
async def v2_upload_import_raw(
    project_id: str,
    image: UploadFile = File(...),
    created_at_ms: str = Form(""),
    idempotency_key: str = Form(""),
    user: str = Depends(current_user),
):
    """Persist an imported image WITHOUT running detection. The
    image is added to the project's manifest with empty detections —
    the labelling pass runs later, kicked off by a separate job
    (kind='label_charlie') from the project page.

    Used by the deferred-labelling flow: user uploads first, presses
    Start to label everything in one go.
    """
    proj = project_dir(project_id)
    if not proj.exists():
        raise HTTPException(404, "project not found")
    # The require_project_owner dep already loaded + validated the
    # manifest; the manifest_path existence check below covers the
    # "manifest file disappeared between the dep and here" race
    # without paying for a second load_manifest deepcopy (which on a
    # 30MB / 1000-image project was 300-500ms of dead weight per
    # upload — i.e. why "NSFW processing" felt 10s slower on a big
    # project than a small one).
    if not manifest_path(project_id).exists():
        raise HTTPException(404, "manifest not found")

    # Idempotency probe. The FE generates a stable key per placeholder
    # and resends it on retries; if a previous upload in this project
    # landed with the same key, return that existing record instead
    # of creating a duplicate. Without this, a network blip between
    # request-completion and response-reception (Safari "Load failed",
    # intermediary timeout) triggers the FE catch-block retry and
    # creates two records for one user action. Cheap manifest scan
    # — we re-check under the write lock below to close the TOCTOU
    # gap on concurrent retries.
    idem = (idempotency_key or "").strip()
    if idem:
        existing = _find_import_by_idempotency_key(project_id, idem)
        if existing is not None:
            return existing

    raw = await image.read()
    if not raw:
        print(
            f"[v2-import-raw] 400 empty upload — filename={image.filename!r} "
            f"content_type={image.content_type!r}"
        )
        raise HTTPException(400, "empty image upload")
    # Per-file size cap. FastAPI streams UploadFile so we can only
    # know the actual size after .read(); rejecting here still
    # prevents the much heavier NSFW + PIL + manifest work below.
    if len(raw) > MAX_UPLOAD_BYTES_PER_FILE:
        raise HTTPException(
            413,
            f"image too large ({len(raw)} bytes, max {MAX_UPLOAD_BYTES_PER_FILE})",
        )

    _orig_filename = image.filename or ""
    _loop = asyncio.get_running_loop()

    # Decode header for dimensions + detect format. JPEG sources are
    # stored as-is (fast — no re-encode). Every NON-JPEG source (AVIF,
    # WEBP, PNG, HEIC, CMYK JPEG, …) is FULLY decoded and re-encoded to
    # a baseline RGB JPEG so every downstream consumer handles a single
    # uniform format: the SAM3 labelling pipeline, the labelled_preview
    # renderer, and the raw byte serve. Previously a non-JPEG upload
    # kept its original bytes but was stored under a forced ".jpg" name
    # (format/extension mismatch); the GPU pipeline + preview renderer
    # couldn't always decode it, leaving the gallery tile a white square
    # that never labelled — even though the browser, which decodes the
    # raw bytes itself, still showed the image in the viewer.
    store_bytes = raw
    try:
        with PILImage.open(io.BytesIO(raw)) as _im:
            _fmt = (_im.format or "").upper()
            if _fmt in ("JPEG", "MPO"):
                width, height = _im.size
                try:
                    _orient = _im.getexif().get(0x0112, 1)
                    if _orient in (5, 6, 7, 8):
                        width, height = height, width
                except Exception:
                    pass
            else:
                _norm = ImageOps.exif_transpose(_im).convert("RGB")
                width, height = _norm.size
                _buf = io.BytesIO()
                _norm.save(_buf, "JPEG", quality=90)
                store_bytes = _buf.getvalue()
                print(
                    f"[v2-import-raw] normalised {_fmt or '?'} → JPEG "
                    f"({len(raw)}→{len(store_bytes)} bytes) for {_orig_filename!r}"
                )
    except Exception as _e:
        raise HTTPException(400, f"cannot decode image: {_e}")

    imports_dir = proj / "images"
    imports_dir.mkdir(exist_ok=True)
    import_id = _uuid.uuid4().hex
    # Stored format is always server-decodable: JPEG sources keep their
    # bytes + .jpg/.jpeg extension; everything else was normalised to
    # JPEG above, so it's stored as .jpg.
    if store_bytes is raw:
        ext = Path(_orig_filename or "image").suffix.lower()
        if ext not in (".jpg", ".jpeg"):
            ext = ".jpg"
    else:
        ext = ".jpg"
    stored_name = f"{import_id}{ext}"

    # Build manifest entry before the write so the under-lock
    # idempotency re-check can return it immediately.
    fe_ts_iso: str | None = None
    if created_at_ms:
        try:
            fe_ts_iso = datetime.fromtimestamp(
                float(created_at_ms) / 1000.0, tz=timezone.utc,
            ).isoformat()
        except Exception:
            fe_ts_iso = None

    entry = {
        "id": import_id,
        "filename": stored_name,
        "originalFilename": _orig_filename or stored_name,
        "width": int(width),
        "height": int(height),
        "detections": [],
        "timings": {},
        "blurhash": None,
        "labelled": False,
        "createdAt": fe_ts_iso or datetime.now(timezone.utc).isoformat(),
    }
    if idem:
        entry["idempotencyKey"] = idem

    # NSFW gate — run synchronously BEFORE writing anything, so a blocked
    # image never lands on disk or in the manifest. Previously this ran
    # in a background task AFTER the upload returned 200, which deleted
    # the file out from under a tile the FE had already painted, leaving
    # a white un-labellable ghost. Now a block raises 451 here and the
    # FE's existing 451 handler drops the tile immediately. Uses the
    # strict EXPOSED-only set (see _enforce_nsfw_or_451) on a 640px copy,
    # on the bg pool so the event loop stays free.
    if state.get("nsfw") is not None:
        def _nsfw_enforce():
            _bytes = store_bytes
            try:
                with PILImage.open(io.BytesIO(store_bytes)) as _p:
                    _w, _h = _p.size
                    if max(_w, _h) > 640:
                        _sc = 640.0 / max(_w, _h)
                        _sm = _p.resize(
                            (int(_w * _sc), int(_h * _sc)), PILImage.BILINEAR
                        )
                        _buf = io.BytesIO()
                        _sm.convert("RGB").save(_buf, "JPEG", quality=85)
                        _bytes = _buf.getvalue()
            except Exception:
                pass
            # Raises HTTPException(451) on a block; the await re-raises it
            # into the handler, FastAPI returns 451, FE removes the tile.
            _enforce_nsfw_or_451(
                _bytes, label="v2-import-raw",
                project=project_id, file=_orig_filename, user=user,
            )
        await _loop.run_in_executor(_BG_IMAGE_EXECUTOR, _nsfw_enforce)

    # Write file to disk and update manifest concurrently — neither
    # depends on the other and both are fast I/O. Scheduling the write
    # here (before awaiting anything) means the file lands on disk in
    # parallel with the manifest lock acquisition.
    file_write_fut = _loop.run_in_executor(
        None, (imports_dir / stored_name).write_bytes, store_bytes
    )

    write_lock = await _manifest_write_lock(project_id)
    async with write_lock:
        manifest = load_manifest(project_id, copy=False) or {}
        if idem:
            for existing_entry in manifest.get("imports", []) or []:
                if existing_entry.get("idempotencyKey") == idem:
                    # Race: another request already committed this key.
                    # Cancel our file write (best-effort) and return
                    # the winning entry so the FE stays consistent.
                    file_write_fut.cancel()
                    return {
                        "import_id": existing_entry.get("id"),
                        "filename": existing_entry.get("filename"),
                        "width": int(existing_entry.get("width") or 0),
                        "height": int(existing_entry.get("height") or 0),
                        "blurhash": existing_entry.get("blurhash"),
                        "labelled": bool(existing_entry.get("labelled", False)),
                    }
        manifest.setdefault("imports", []).append(entry)
        if not manifest.get("cover"):
            manifest["cover"] = stored_name
        manifest["updatedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        save_manifest(project_id, manifest, cache_by_ref=True)

    # Ensure file write is done before handing off to background tasks
    # that need to read it. Typically completes before we reach here.
    try:
        await file_write_fut
    except Exception as _fw_err:
        print(f"[v2-import-raw] file write failed for {import_id}: {_fw_err}")

    # Background task: blurhash only. NSFW already ran synchronously
    # above (a blocked image 451s before it ever reaches here), so
    # there's no file/manifest entry to purge — we just compute the
    # BlurHash gradient placeholder and stamp it onto the entry.
    _bg_pid = project_id
    _bg_iid = import_id
    _bg_stored = stored_name
    _bg_dir = imports_dir

    async def _bg_blurhash():
        try:
            def _bh_sync():
                with PILImage.open(_bg_dir / _bg_stored) as _p:
                    return _compute_blurhash_safe(
                        ImageOps.exif_transpose(_p).convert("RGB")
                    )
            bh = await _loop.run_in_executor(_BG_IMAGE_EXECUTOR, _bh_sync)
            if bh:
                _wl = await _manifest_write_lock(_bg_pid)
                async with _wl:
                    _m = load_manifest(_bg_pid, copy=False) or {}
                    for _e in _m.get("imports", []):
                        if _e.get("id") == _bg_iid:
                            _e["blurhash"] = bh
                            break
                    save_manifest(_bg_pid, _m, cache_by_ref=True)
        except Exception:
            pass

    asyncio.create_task(_bg_blurhash())

    # Fire-and-forget whole-image embedding. The stats card uses
    # these for the 2-D variation plot, the near-duplicate flag,
    # and one of the health-score factors. Computed here so the
    # backfill window is "one model pass per upload" rather than a
    # batch sweep later. Failure (model not loaded, decode error)
    # is non-fatal: the stats endpoint backfills missing ones on
    # demand.
    #
    # Routed through the GPU gate at P_BACKGROUND + the dedicated bg
    # executor: a burst of uploads yields the GPU to any click-to-detect
    # (P_INTERACTIVE) or labelling job (P_JOB) instead of storming it,
    # and never ties up a request-path thread. This is what stops the
    # import burst from stalling — embeddings trickle through in the
    # background while uploads return immediately.
    # Whole-image embedding for the stats card, off the hot path. The
    # shared helper bounds concurrency + pauses while a GPU job runs so a
    # bulk-import backlog can't starve labelling of the GPU.
    asyncio.create_task(_run_bg_embedding(project_id, import_id, store_bytes))

    # Audit row for the upload. V2 uploads land directly without
    # going through the job ledger, so the per-user usage rollup
    # (sum_uploaded_images_for_user, which the profile page reads)
    # needs an explicit `job`-shaped event with job_kind="upload"
    # for each successful upload. One audit row per uploaded image
    # keeps the per-image counter accurate.
    try:
        add_event(
            "job",
            id=import_id,
            job_kind="upload",
            project=project_id,
            user=user or "anonymous",
            status="done",
            elapsed_s=0.0,
            cost_pence=0.0,
            n_images=1,
            error=None,
        )
    except Exception as e:
        print(f"[v2-import-raw] upload audit failed for {import_id}: {e}")

    return {
        "import_id": import_id,
        "filename": stored_name,
        "width": int(width),
        "height": int(height),
        "blurhash": None,
        "labelled": False,
    }


def _sanitize_imported_boxes(raw_boxes, width: int, height: int) -> tuple[list[dict], int]:
    """Normalise externally-imported annotation boxes (Pascal VOC / COCO /
    YOLO, converted to absolute pixels client-side) into the canonical
    editable-box shape the editor + exporters consume:
    {id, label, x0, y0, x1, y1, score, mask?} in absolute top-left pixel
    coords. Coerces numbers, repairs reversed corners, clamps to the image,
    and drops degenerate (effectively zero-area) boxes so a later re-export
    can't silently lose them — _box_xyxy rejects any box with x1<=x0 or
    y1<=y0. Returns (clean_boxes, n_dropped)."""
    clean: list[dict] = []
    dropped = 0
    W = float(width or 0)
    H = float(height or 0)
    for i, b in enumerate(raw_boxes if isinstance(raw_boxes, list) else []):
        if not isinstance(b, dict):
            dropped += 1
            continue
        try:
            x0 = float(b.get("x0")); y0 = float(b.get("y0"))
            x1 = float(b.get("x1")); y1 = float(b.get("y1"))
        except (TypeError, ValueError):
            dropped += 1
            continue
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        if W > 0:
            x0 = min(max(x0, 0.0), W)
            x1 = min(max(x1, 0.0), W)
        if H > 0:
            y0 = min(max(y0, 0.0), H)
            y1 = min(max(y1, 0.0), H)
        if x1 - x0 < 1.0 or y1 - y0 < 1.0:
            dropped += 1
            continue
        label = str(b.get("label") or "").strip()
        if not label:
            dropped += 1
            continue
        out = {
            "id": str(b.get("id") or f"imp_{i}"),
            "label": label,
            "x0": round(x0, 2),
            "y0": round(y0, 2),
            "x1": round(x1, 2),
            "y1": round(y1, 2),
            "score": (float(b["score"]) if isinstance(b.get("score"), (int, float)) else None),
        }
        mask = b.get("mask")
        if isinstance(mask, dict) and isinstance(mask.get("polygons"), list):
            out["mask"] = {"polygons": mask["polygons"]}
        clean.append(out)
    return clean, dropped


@app.post(
    "/api/v2/projects/{project_id}/imports/raw_batch",
    dependencies=[Depends(require_project_owner)],
)
async def v2_upload_import_raw_batch(
    project_id: str,
    images: list[UploadFile] = File(...),
    created_at_ms: list[str] = Form(default=[]),
    idempotency_key: list[str] = Form(default=[]),
    boxes: list[str] = Form(default=[]),
    user: str = Depends(current_user),
):
    """Batch sibling of /imports/raw: accept many images in ONE
    multipart request and append them all to the manifest with a
    SINGLE write, instead of a full-manifest rewrite per image.

    When the optional per-image `boxes` field is supplied (a JSON array of
    editable boxes, aligned by index to `images`), the item is stored as an
    ALREADY-LABELLED import: the boxes become its authoritative `editedBoxes`
    (editedBoxesSet=True, labelled=True), so the editor renders + edits them
    and every exporter round-trips them. This powers the "import a labelled
    dataset as a fully-editable project" flow. Items with no `boxes` entry
    behave exactly as a deferred-label upload (the original behaviour).

    Importing thousands of images one-request-per-image made the
    manifest write O(n) per upload (the whole growing manifest is
    re-serialised under the write lock each time) -> O(n^2) overall,
    plus n separate HTTP round-trips, n idempotency scans, AND a second
    full rewrite per image from the background blurhash task. Batching
    collapses the write + round-trip count by the batch size and folds
    the blurhash compute inline (one write, not two), which is the
    dominant cost at thousands-of-images scale.

    Each item is processed independently on the bg image pool; a decode
    error or NSFW block on one image is reported per-item and never
    fails the rest of the batch. Returns {results: [...]} aligned to the
    input order so the FE can map each placeholder tile to its outcome.
    """
    proj = project_dir(project_id)
    if not proj.exists():
        raise HTTPException(404, "project not found")
    if not manifest_path(project_id).exists():
        raise HTTPException(404, "manifest not found")
    if not images:
        raise HTTPException(400, "no images")
    if len(images) > MAX_FILES_PER_UPLOAD_BATCH:
        raise HTTPException(
            413,
            f"too many files in one batch ({len(images)} > {MAX_FILES_PER_UPLOAD_BATCH})",
        )

    imports_dir = proj / "images"
    imports_dir.mkdir(exist_ok=True)
    loop = asyncio.get_running_loop()

    # Read every part's bytes up front (FastAPI streams UploadFile).
    raws: list[bytes] = [await up.read() for up in images]

    # Idempotency: a retried batch resends the same per-image keys.
    # Scan the manifest ONCE (not once per image) and echo any record
    # that already landed.
    existing_by_key: dict[str, dict] = {}
    try:
        _m0 = load_manifest(project_id, copy=False) or {}
        for _e in _m0.get("imports", []) or []:
            _k = _e.get("idempotencyKey")
            if isinstance(_k, str) and _k:
                existing_by_key[_k] = _e
    except Exception:
        existing_by_key = {}

    def _process_one(idx: int) -> dict:
        """Sync worker (bg image pool): decode + normalise, NSFW-gate,
        write the file, encode the blurhash. Builds the manifest entry
        but does NOT touch the manifest (the caller appends all entries
        under a single write lock). On reject/error writes nothing and
        returns a status the FE can surface on the tile."""
        raw = raws[idx]
        orig_filename = images[idx].filename or ""
        idem = idempotency_key[idx].strip() if idx < len(idempotency_key) else ""
        if idem and idem in existing_by_key:
            _e = existing_by_key[idem]
            return {
                "idempotency_key": idem,
                "status": "ok",
                "import_id": _e.get("id"),
                "filename": _e.get("filename"),
                "width": int(_e.get("width") or 0),
                "height": int(_e.get("height") or 0),
                "blurhash": _e.get("blurhash"),
            }
        if not raw:
            return {"idempotency_key": idem, "status": "failed", "error": "empty image upload"}
        if len(raw) > MAX_UPLOAD_BYTES_PER_FILE:
            return {"idempotency_key": idem, "status": "failed", "error": f"image too large ({len(raw)} bytes)"}

        # Decode header + normalise non-JPEG to baseline RGB JPEG, same
        # rules as the single /imports/raw endpoint.
        store_bytes = raw
        try:
            with PILImage.open(io.BytesIO(raw)) as _im:
                _fmt = (_im.format or "").upper()
                if _fmt in ("JPEG", "MPO"):
                    width, height = _im.size
                    try:
                        _orient = _im.getexif().get(0x0112, 1)
                        if _orient in (5, 6, 7, 8):
                            width, height = height, width
                    except Exception:
                        pass
                else:
                    _norm = ImageOps.exif_transpose(_im).convert("RGB")
                    width, height = _norm.size
                    _buf = io.BytesIO()
                    _norm.save(_buf, "JPEG", quality=90)
                    store_bytes = _buf.getvalue()
        except Exception as _e:
            return {"idempotency_key": idem, "status": "failed", "error": f"cannot decode image: {_e}"}

        # Per-item NSFW gate. Unlike the single endpoint (which 451s the
        # whole request), a block here just marks THIS item rejected so
        # the rest of the batch still lands. Strict EXPOSED-only set.
        if state.get("nsfw") is not None:
            try:
                _bytes = store_bytes
                with PILImage.open(io.BytesIO(store_bytes)) as _p:
                    _w, _h = _p.size
                    if max(_w, _h) > 640:
                        _sc = 640.0 / max(_w, _h)
                        _sm = _p.resize((int(_w * _sc), int(_h * _sc)), PILImage.BILINEAR)
                        _b = io.BytesIO()
                        _sm.convert("RGB").save(_b, "JPEG", quality=85)
                        _bytes = _b.getvalue()
                import tempfile as _tempfile
                with _tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as _tmp:
                    _tmp.write(_bytes)
                    _tmp_path = Path(_tmp.name)
                try:
                    _score, _cls = nsfw_score(state["nsfw"], _tmp_path, classes=EXPOSED_CLASSES)
                except Exception:
                    _score, _cls = 0.0, ""
                finally:
                    try:
                        _tmp_path.unlink()
                    except Exception:
                        pass
                if _score >= NSFW_THRESHOLD:
                    print(f"[v2-import-raw-batch] 451 nsfw, score={_score:.3f} class={_cls or '-'}")
                    add_event(
                        "nsfw_block", project=project_id, file=orig_filename,
                        score=round(float(_score), 3), classification=_cls or "", user=user,
                    )
                    return {"idempotency_key": idem, "status": "rejected", "error": "nsfw"}
            except Exception:
                pass

        import_id = _uuid.uuid4().hex
        if store_bytes is raw:
            ext = Path(orig_filename or "image").suffix.lower()
            if ext not in (".jpg", ".jpeg"):
                ext = ".jpg"
        else:
            ext = ".jpg"
        stored_name = f"{import_id}{ext}"
        try:
            (imports_dir / stored_name).write_bytes(store_bytes)
        except Exception as _e:
            return {"idempotency_key": idem, "status": "failed", "error": f"write failed: {_e}"}

        # Blurhash inline so the gallery gets a placeholder immediately
        # AND we avoid the single endpoint's second per-image manifest
        # write from a background blurhash task.
        try:
            with PILImage.open(io.BytesIO(store_bytes)) as _p:
                _bh = _compute_blurhash_safe(ImageOps.exif_transpose(_p).convert("RGB"))
        except Exception:
            _bh = None

        fe_ts_iso: str | None = None
        _cms = created_at_ms[idx] if idx < len(created_at_ms) else ""
        if _cms:
            try:
                fe_ts_iso = datetime.fromtimestamp(float(_cms) / 1000.0, tz=timezone.utc).isoformat()
            except Exception:
                fe_ts_iso = None

        entry = {
            "id": import_id,
            "filename": stored_name,
            "originalFilename": orig_filename or stored_name,
            "width": int(width),
            "height": int(height),
            "detections": [],
            "timings": {},
            "blurhash": _bh,
            "labelled": False,
            "createdAt": fe_ts_iso or datetime.now(timezone.utc).isoformat(),
        }
        if idem:
            entry["idempotencyKey"] = idem

        # Pre-labelled dataset import: when the caller supplies boxes for this
        # item, attach them as the authoritative editable annotation set.
        # editedBoxesSet=True is what makes the read paths (/annotations,
        # /v3/viewport) surface editedBoxes and the editor treat them as the
        # source of truth; labelled=True so the image counts as labelled even
        # when its annotation is intentionally EMPTY (background/negative
        # frames are valid training data). Boxes arrive in the uploaded
        # image's pixel space — the FE scales them to match any downscale.
        _labels: list[str] = []
        _dropped = 0
        if idx < len(boxes):
            # Belt-and-braces: a malformed box payload for one item must never
            # fail the whole batch (the endpoint's contract is per-item
            # isolation). On any error the item still lands as an explicitly
            # empty (background) labelled image rather than crashing the gather.
            try:
                _parsed = json.loads(boxes[idx] or "[]")
                _clean, _dropped = _sanitize_imported_boxes(_parsed, width, height)
                _labels = [b["label"] for b in _clean]
            except Exception as _be:
                print(f"[v2-import-raw-batch] box parse failed for item {idx}: {_be}")
                _clean, _dropped, _labels = [], 0, []
            entry["detections"] = []
            entry["editedBoxes"] = _clean
            entry["editedBoxesSet"] = True
            entry["labelled"] = True

        return {
            "idempotency_key": idem,
            "status": "ok",
            "import_id": import_id,
            "filename": stored_name,
            "width": int(width),
            "height": int(height),
            "blurhash": _bh,
            "dropped_boxes": _dropped,
            "_entry": entry,
            "_store_bytes": store_bytes,
            "_labels": _labels,
        }

    # Decode / NSFW / write / blurhash for every item, concurrently on
    # the bounded bg image pool (3 workers), so the request stays off
    # the event loop and a big batch parallelises across cores.
    results: list[dict] = list(await asyncio.gather(*[
        loop.run_in_executor(_BG_IMAGE_EXECUTOR, _process_one, i)
        for i in range(len(images))
    ]))

    # ONE manifest write for the whole batch's new entries.
    new_entries = [r["_entry"] for r in results if r.get("_entry")]
    if new_entries:
        write_lock = await _manifest_write_lock(project_id)
        async with write_lock:
            mm = load_manifest(project_id, copy=False) or {}
            have_keys = {
                e.get("idempotencyKey")
                for e in (mm.get("imports") or [])
                if e.get("idempotencyKey")
            }
            # Under-lock guard against a concurrent batch that committed
            # the same idempotency key first.
            to_add = [
                e for e in new_entries
                if not (e.get("idempotencyKey") and e.get("idempotencyKey") in have_keys)
            ]
            mm.setdefault("imports", []).extend(to_add)
            # Union any imported class names into the project's label
            # vocabulary (manifest['tags']) so every imported box label is a
            # first-class tag — colour assignment, the label picker, and the
            # YOLO/COCO export category order all key off tags. Normally a
            # no-op (the project is created with the dataset's class list),
            # but it keeps boxes whose label is somehow missing exportable.
            _added_ids = {e.get("id") for e in to_add}
            _existing_tags = {
                t.strip().lower()
                for t in (mm.get("tags") or [])
                if isinstance(t, str)
            }
            for _r in results:
                if _r.get("import_id") not in _added_ids:
                    continue
                for _lbl in (_r.get("_labels") or []):
                    _ll = _lbl.strip().lower()
                    if _ll and _ll not in _existing_tags:
                        mm.setdefault("tags", []).append(_ll)
                        _existing_tags.add(_ll)
            if not mm.get("cover") and to_add:
                mm["cover"] = to_add[0].get("filename")
            mm["updatedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            # Serialise + write off the event loop. On a big import the
            # manifest grows large and orjson.dumps + the file write are
            # synchronous; doing them inline would stall EVERY other
            # request (FE polls, other batches) for the duration of each
            # batch write. save_manifest is already called from worker
            # threads elsewhere (blurhash backfill), so this is safe. The
            # async write lock still serialises same-project writes.
            await loop.run_in_executor(
                None, lambda: save_manifest(project_id, mm, cache_by_ref=True)
            )
        try:
            _invalidate_project_payloads(project_id)
        except Exception as e:
            print(f"[v2-import-raw-batch] payload invalidate failed: {e}")

    # Per-image audit row + fire-and-forget whole-image embedding, same
    # as the single endpoint. Embeddings trickle through the GPU gate at
    # background priority so the import burst never starves a click or
    # labelling job.
    for r in results:
        if r.get("status") != "ok" or not r.get("_entry"):
            continue
        _iid = r["import_id"]
        try:
            add_event(
                "job", id=_iid, job_kind="upload", project=project_id,
                user=user or "anonymous", status="done", elapsed_s=0.0,
                cost_pence=0.0, n_images=1, error=None,
            )
        except Exception as e:
            print(f"[v2-import-raw-batch] upload audit failed for {_iid}: {e}")
        _sb = r.get("_store_bytes")
        if _sb is not None:
            # Off-hot-path embedding via the shared bounded/pausable helper
            # (see _run_bg_embedding) so the batch's embeddings can't pile
            # onto the GPU and starve a subsequent labelling job.
            asyncio.create_task(_run_bg_embedding(project_id, _iid, _sb))

    # Strip internal-only keys before returning.
    out = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
    return {"results": out}


class V2ImportsFromUrlsRequest(BaseModel):
    urls: list[str]
    query: str | None = None


@app.post(
    "/api/v2/projects/{project_id}/imports/from_urls",
    dependencies=[
        Depends(require_project_owner),
    ],
)
async def v2_imports_from_urls(
    project_id: str,
    body: V2ImportsFromUrlsRequest,
    user: str = Depends(current_user),
):
    """V2 mirror of /api/projects/{id}/images_from_urls. Pulls a
    list of Openverse-curated URLs, downloads each, validates,
    NSFW-gates, and appends to manifest["imports"] using the V2
    deferred-labelling shape — same envelope /imports/raw writes,
    so the gallery + label_charlie job pick the new images up
    without any further FE work.

    Returns {added: [import_id], skipped: [url], rejected: [{url,
    reason}]} so the FE can drive a post-import refresh + show the
    user what failed.
    """
    import hashlib
    import uuid as _uuid
    import openverse
    from urllib.parse import urlparse

    proj = project_dir(project_id)
    if not proj.exists():
        raise HTTPException(404, "project not found")
    manifest = load_manifest(project_id)
    if not manifest:
        raise HTTPException(404, "manifest not found")

    raw_urls = [u.strip() for u in (body.urls or []) if isinstance(u, str) and u.strip()]
    # Cap the URL list so a single request can't queue thousands of
    # downloads against the backend's bandwidth/disk. 100 is well
    # above the Openverse panel's typical batch (24 results/page).
    if len(raw_urls) > 100:
        raise HTTPException(400, "too many urls, max 100 per request")
    if not raw_urls:
        raise HTTPException(400, "no urls")
    seen_in_batch: set[str] = set()
    urls: list[str] = []
    for u in raw_urls:
        if u not in seen_in_batch:
            seen_in_batch.add(u)
            urls.append(u)

    raw_q = (body.query or "").strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", raw_q).strip("_")
    if not slug:
        slug = "openverse"
    slug = slug[:48]

    loop = asyncio.get_running_loop()
    blobs: list[tuple[bytes | None, str | None]] = await loop.run_in_executor(
        None, lambda: openverse.download_images_bytes(urls),
    )

    existing_imports = manifest.get("imports") or []
    existing_filenames: set[str] = {
        e.get("filename") for e in existing_imports if e.get("filename")
    }
    existing_hashes: dict[str, str] = {}
    existing_sources: set[str] = set()
    for e in existing_imports:
        h = e.get("hash")
        if isinstance(h, str) and h:
            existing_hashes[h] = e.get("id") or ""
        src = e.get("source")
        if isinstance(src, dict):
            u = src.get("url")
            if isinstance(u, str) and u:
                existing_sources.add(u)

    imports_dir = proj / "images"
    imports_dir.mkdir(exist_ok=True)

    added: list[str] = []
    skipped: list[str] = []
    rejected: list[dict] = []
    new_entries: list[dict] = []

    base_ts = time.time()

    for idx, (url, (data, ctype)) in enumerate(zip(urls, blobs)):
        if url in existing_sources:
            rejected.append({"url": url, "reason": "already_imported"})
            skipped.append(url)
            continue
        if data is None:
            rejected.append({"url": url, "reason": "download_failed"})
            continue
        try:
            with PILImage.open(io.BytesIO(data)) as img:
                img.verify()
        except Exception:
            rejected.append({"url": url, "reason": "not_an_image"})
            continue
        # NSFW gate (skipped when classifier isn't loaded).
        if state.get("nsfw") is not None:
            import tempfile as _tempfile
            with _tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as _tmp:
                _tmp.write(data)
                _tmp_path = Path(_tmp.name)
            try:
                # Strict EXPOSED-only set — same rationale as the
                # /imports/raw gate: the broad COVERED classes
                # false-positive on innocuous dataset/Openverse imagery
                # (fruit, animals, people in PPE) and wrongly reject them.
                score, cls = nsfw_score(state["nsfw"], _tmp_path, classes=EXPOSED_CLASSES)
            except Exception:
                score, cls = 0.0, ""
            finally:
                try:
                    _tmp_path.unlink()
                except Exception:
                    pass
            if score >= NSFW_THRESHOLD:
                rejected.append({"url": url, "reason": "nsfw", "score": round(float(score), 3), "class": cls or None})
                continue
        sha = hashlib.sha256(data).hexdigest()
        if sha in existing_hashes:
            rejected.append({"url": url, "reason": "duplicate"})
            skipped.append(url)
            continue
        # Decode to a normalised RGB image + EXIF transpose, mirror
        # the /imports/raw write path.
        try:
            with PILImage.open(io.BytesIO(data)) as pil:
                pil = ImageOps.exif_transpose(pil)
                pil = pil.convert("RGB")
                width, height = pil.size
                blurhash_str = _compute_blurhash_safe(pil)
        except Exception as e:
            rejected.append({"url": url, "reason": "decode_failed", "detail": str(e)})
            continue

        # Pick a stable, slug-prefixed filename so the import is
        # self-documenting on disk + survives a `ls`.
        ext = ""
        try:
            path = urlparse(url).path
            base = Path(path).name
            if "." in base:
                e_ext = base.rsplit(".", 1)[1].lower()
                if e_ext in ("jpg", "jpeg", "png", "webp"):
                    ext = "." + e_ext
        except Exception:
            pass
        if not ext and ctype:
            guess = ctype.split(";")[0].strip().lower()
            ext = {
                "image/jpeg": ".jpg", "image/jpg": ".jpg",
                "image/png": ".png", "image/webp": ".webp",
            }.get(guess, ".jpg")
        if not ext:
            ext = ".jpg"
        import_id = _uuid.uuid4().hex
        suffix = _uuid.uuid4().hex[:12]
        stored_name = f"{slug}_{suffix}{ext}"
        while stored_name in existing_filenames:
            suffix = _uuid.uuid4().hex[:12]
            stored_name = f"{slug}_{suffix}{ext}"
        existing_filenames.add(stored_name)
        existing_hashes[sha] = import_id
        existing_sources.add(url)

        try:
            (imports_dir / stored_name).write_bytes(data)
        except Exception as e:
            rejected.append({"url": url, "reason": "storage_failed", "detail": str(e)})
            continue

        # Reverse-order timestamps within the batch so the gallery
        # DESC sort puts the first URL at top-left (same trick the
        # raw upload uses for drag-drop).
        ts = base_ts + (len(urls) - idx) * 0.001
        entry = {
            "id": import_id,
            "filename": stored_name,
            "originalFilename": Path(urlparse(url).path).name or stored_name,
            "width": int(width),
            "height": int(height),
            "detections": [],
            "timings": {},
            "blurhash": blurhash_str,
            "hash": sha,
            "source": {"kind": "openverse", "url": url, "query": body.query or ""},
            "createdAt": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
            "labelled": False,
        }
        new_entries.append(entry)
        added.append(import_id)

    if new_entries:
        write_lock = await _manifest_write_lock(project_id)
        async with write_lock:
            mm = load_manifest(project_id) or {}
            mm.setdefault("imports", []).extend(new_entries)
            if not mm.get("cover") and new_entries:
                mm["cover"] = new_entries[0].get("filename")
            mm["updatedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            save_manifest(project_id, mm)
        # Drop the cached /overview + sidecar payloads so the FE's
        # immediate post-import refresh recomputes FRESH instead of being
        # served the pre-import list by the stale-while-revalidate path
        # (which is why the new images only appeared after a manual page
        # reload). Same fix the dedupe / delete mutations use.
        try:
            _invalidate_project_payloads(project_id)
        except Exception as e:
            print(f"[v2-import-from-urls] payload invalidate failed: {e}")
        # One audit row per successfully-added Openverse image so the
        # profile usage panel counts these the same way it counts
        # direct uploads. Parity with the /imports/raw endpoint.
        for import_id in added:
            try:
                add_event(
                    "job",
                    id=import_id,
                    job_kind="upload",
                    project=project_id,
                    user=user or "anonymous",
                    status="done",
                    elapsed_s=0.0,
                    cost_pence=0.0,
                    n_images=1,
                    error=None,
                )
            except Exception as e:
                print(f"[v2-import-from-urls] upload audit failed for {import_id}: {e}")

    return {
        "added": added,
        "skipped": skipped,
        "rejected": rejected,
    }


def _compute_blurhash_safe(pil_image) -> str | None:
    """Best-effort blurhash from a PIL image. Mirrors the encode
    fallbacks used by _encode_blurhash_from_path so we cover all the
    blurhash library version differences in one place. Returns None
    on any error so the upload path never fails because of blurhash;
    FE falls back to a shimmer placeholder when null."""
    try:
        import blurhash as _bh  # type: ignore
        small = pil_image.copy().convert("RGB")
        small.thumbnail((64, 64))
        try:
            return _bh.encode(small, 4, 3)  # type: ignore[arg-type]
        except TypeError:
            import numpy as _np
            arr = _np.asarray(small, dtype=_np.uint8)
            try:
                return _bh.encode(arr, 4, 3)
            except TypeError:
                return _bh.encode(arr, x_components=4, y_components=3)
    except Exception:
        return None


class V2ImportPatch(BaseModel):
    editedBoxes: list | None = None
    detections: list | None = None


@app.put(
    "/api/v2/projects/{project_id}/imports/{import_id}",
    dependencies=[Depends(require_project_owner)],
)
async def v2_update_import(project_id: str, import_id: str, payload: V2ImportPatch):
    """Patch an existing import — typically the user's editedBoxes
    after they've drawn / edited / deleted boxes in the viewer."""
    proj = project_dir(project_id)
    if not proj.exists():
        raise HTTPException(404, "project not found")
    write_lock = await _manifest_write_lock(project_id)
    async with write_lock:
        # copy=False — we mutate + save under the per-project write
        # lock and discard `manifest` right after. cache_by_ref on
        # the save closes the loop, both removing the deepcopies that
        # made each box-drag take 200-500ms longer than necessary on
        # big projects.
        manifest = load_manifest(project_id, copy=False)
        if not manifest:
            raise HTTPException(404, "manifest not found")
        imports = manifest.get("imports") or []
        for i, imp in enumerate(imports):
            if imp.get("id") == import_id:
                if payload.editedBoxes is not None:
                    imp["editedBoxes"] = payload.editedBoxes
                    # Mark this import as user-edited. The annotations
                    # endpoint reads this flag to decide whether to
                    # send `editedBoxes` to the FE — without it, an
                    # explicit delete-all (editedBoxes: []) would be
                    # indistinguishable from "no edits yet" and the
                    # auto detections would re-appear.
                    imp["editedBoxesSet"] = True
                    # Mark the project as "needs re-augment". The PUT
                    # endpoint is hot (every drag fires one), so we
                    # only flag the manifest here and let a debounced
                    # check at the end of the request decide whether
                    # to schedule a fresh augment_generate. Stamping
                    # the timestamp also lets the dataset stats card
                    # detect freshly-edited imports for its health
                    # score.
                    imp["editedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                    # Persisted labelled-preview cachebuster (epoch ms).
                    # The preview is re-baked below; bumping this makes a
                    # cold reopen request the fresh bake instead of a
                    # stale cached one. See _tile_overview's labelledAt.
                    imp["labelledAt"] = int(time.time() * 1000)
                if payload.detections is not None:
                    imp["detections"] = payload.detections
                imports[i] = imp
                manifest["imports"] = imports
                manifest["updatedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                save_manifest(project_id, manifest, cache_by_ref=True)
                # The labelled preview was rendered from the previous
                # detection set — drop it AND queue an immediate re-
                # bake so the next gallery thumb request hits a warm
                # cache. The lazy GET path covers the race between
                # invalidate + bake completion.
                _invalidate_labelled_preview(project_id, import_id)
                fn = imp.get("filename")
                if fn:
                    src = proj / "images" / fn
                    edited = imp.get("editedBoxes")
                    edited_set = bool(imp.get("editedBoxesSet"))
                    # Trust an explicit user-cleared state (set=True,
                    # list=[]) so the preview matches the canvas. Only
                    # fall through to auto detections when the user
                    # hasn't touched this tile.
                    if isinstance(edited, list) and (edited or edited_set):
                        det_for_bake = list(edited)
                    else:
                        det_for_bake = list(imp.get("detections") or [])
                    loop = asyncio.get_running_loop()
                    loop.run_in_executor(
                        None,
                        _bake_labelled_preview_sync,
                        project_id,
                        import_id,
                        src,
                        det_for_bake,
                        list(manifest.get("tags") or []),
                    )
                # Auto-augment-after-edit hook removed by user request.
                # Augmentations only run when the user explicitly
                # clicks the Update button in the Augmentations card.
                return {"ok": True}
    raise HTTPException(404, "import not found")


@app.delete(
    "/api/v2/projects/{project_id}/imports/{import_id}",
    dependencies=[Depends(require_project_owner)],
)
async def v2_delete_import(project_id: str, import_id: str):
    proj = project_dir(project_id)
    if not proj.exists():
        raise HTTPException(404, "project not found")
    write_lock = await _manifest_write_lock(project_id)
    async with write_lock:
        manifest = load_manifest(project_id)
        if not manifest:
            raise HTTPException(404, "manifest not found")
        imports = manifest.get("imports") or []
        kept: list[dict] = []
        found = False
        deleted_detkey: str | None = None
        for imp in imports:
            if imp.get("id") == import_id:
                found = True
                deleted_detkey = (imp.get("derivedFrom") or {}).get("detKey")
                fn = imp.get("filename")
                if fn:
                    try:
                        (proj / "images" / fn).unlink(missing_ok=True)
                    except Exception as e:
                        print(f"[v2-import-delete] couldn't unlink {fn}: {e}")
                _invalidate_labelled_preview(project_id, import_id)
            else:
                kept.append(imp)
        if not found:
            raise HTTPException(404, "import not found")
        manifest["imports"] = kept

        # Child project: tombstone the source detection so a re-sync from the
        # parent won't resurrect a crop the user deleted here. One-way only —
        # this never touches the parent.
        if deleted_detkey and _derived_mod is not None and (manifest.get("derived") or {}).get("parentProjectId"):
            _derived_mod.suppress_detkey(manifest, deleted_detkey)

        # If the deleted import was the project's cover, pick a fresh
        # one from whatever's left so the workspace card doesn't fall
        # back to a placeholder gradient. Priority: a remaining import
        # → a reference → None when the project is genuinely empty.
        cover_filename = manifest.get("cover")
        if cover_filename and not any(
            imp.get("filename") == cover_filename for imp in kept
        ) and not any(
            ref.get("filename") == cover_filename
            for ref in (manifest.get("references") or [])
        ):
            import random as _rnd
            kept_filenames = [
                imp.get("filename") for imp in kept if imp.get("filename")
            ]
            ref_filenames = [
                ref.get("filename")
                for ref in (manifest.get("references") or [])
                if ref.get("filename")
            ]
            new_cover: str | None = None
            if kept_filenames:
                new_cover = _rnd.choice(kept_filenames)
            elif ref_filenames:
                new_cover = _rnd.choice(ref_filenames)
            manifest["cover"] = new_cover
            manifest["cover_blurhash"] = None  # re-derived on next read

        manifest["updatedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        save_manifest(project_id, manifest)
    return {"ok": True}


class DeleteImportsBatchIn(BaseModel):
    ids: list[str]


@app.post(
    "/api/v2/projects/{project_id}/imports/delete_batch",
    dependencies=[Depends(require_project_owner)],
)
async def v2_delete_imports_batch(project_id: str, payload: DeleteImportsBatchIn):
    """Delete many imports in one shot. Held under a single write lock
    so a 50-image bulk delete doesn't fan out 50 separate manifest
    saves + cover-rescue passes (each of which would also serialise
    against any ongoing labelling work)."""
    proj = project_dir(project_id)
    if not proj.exists():
        raise HTTPException(404, "project not found")
    target_ids = {x for x in (payload.ids or []) if isinstance(x, str) and x}
    if not target_ids:
        return {"deleted": [], "not_found": []}
    write_lock = await _manifest_write_lock(project_id)
    async with write_lock:
        manifest = load_manifest(project_id)
        if not manifest:
            raise HTTPException(404, "manifest not found")
        imports = manifest.get("imports") or []
        kept: list[dict] = []
        deleted_ids: list[str] = []
        for imp in imports:
            iid = imp.get("id")
            if iid in target_ids:
                deleted_ids.append(iid)
                fn = imp.get("filename")
                if fn:
                    try:
                        (proj / "images" / fn).unlink(missing_ok=True)
                    except Exception as e:
                        print(f"[v2-import-delete-batch] couldn't unlink {fn}: {e}")
                _invalidate_labelled_preview(project_id, iid)
            else:
                kept.append(imp)
        manifest["imports"] = kept

        # Same cover-rescue logic as the single-import delete — if the
        # cover pointed at a now-deleted file, pick a fresh one from
        # what's left so the workspace card keeps a thumbnail.
        cover_filename = manifest.get("cover")
        if cover_filename and not any(
            imp.get("filename") == cover_filename for imp in kept
        ) and not any(
            ref.get("filename") == cover_filename
            for ref in (manifest.get("references") or [])
        ):
            import random as _rnd
            kept_filenames = [
                imp.get("filename") for imp in kept if imp.get("filename")
            ]
            ref_filenames = [
                ref.get("filename")
                for ref in (manifest.get("references") or [])
                if ref.get("filename")
            ]
            new_cover: str | None = None
            if kept_filenames:
                new_cover = _rnd.choice(kept_filenames)
            elif ref_filenames:
                new_cover = _rnd.choice(ref_filenames)
            manifest["cover"] = new_cover
            manifest["cover_blurhash"] = None

        manifest["updatedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        save_manifest(project_id, manifest)
    not_found = sorted(target_ids - set(deleted_ids))
    return {"deleted": deleted_ids, "not_found": not_found}


def _invalidate_project_payloads(project_id: str) -> None:
    """Drop the in-memory payload cache entries AND the on-disk sidecars
    for a project. Called from any mutation that doesn't naturally fall
    out of the sidecar mtime check — for instance the dedupe commit,
    which deletes imports + bumps manifest mtime but, without this, lets
    the stale-while-revalidate path serve the pre-delete /overview +
    /dataset-stats payloads to the very next request after the user's
    page reload. That manifested as: the deduped count flashing
    briefly, then the workspace card / stats card "reverting" to the
    original count because the stale cache responded faster than the
    background revalidate."""
    with _PAYLOAD_CACHE_LOCK:
        keys = [k for k in _PAYLOAD_CACHE if k[0] == project_id]
        for k in keys:
            _PAYLOAD_CACHE.pop(k, None)
    for path in (
        _overview_sidecar_path(project_id),
        _initial_sidecar_path(project_id),
        _stats_sidecar_path(project_id, True),
        _stats_sidecar_path(project_id, False),
    ):
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass


class DedupeIn(BaseModel):
    """Body for /imports/dedupe. `mode` switches between dry-run and
    destructive. `strategy` picks between exact byte-hash dedup
    (catches literal "same file uploaded twice" cases) and embedding-
    based near-duplicate dedup (catches the common "Roboflow re-export
    of my own dataset" case — same pixels but JPEG-recompressed so byte
    hashes differ). `threshold` is the cosine cutoff for "near" mode;
    default matches the dataset-stats card so the count it reports
    and the modal's group list reference the same source."""

    mode: str = "preview"  # "preview" or "commit"
    strategy: str = "near"  # "exact" or "near"
    threshold: float = 0.95


def _import_file_sha256(project_id: str, filename: str) -> str | None:
    """SHA256 of an import's file bytes. None on missing/unreadable —
    those imports just don't participate in the exact-dedup groups."""
    import hashlib
    p = project_dir(project_id) / "images" / filename
    if not p.exists():
        return None
    try:
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception as e:
        print(f"[dedupe] sha256 failed for {filename}: {e}")
        return None


def _import_keeper_score(imp: dict) -> tuple[int, int, int]:
    """Higher = keep. Ranks by: edited boxes > detections > has filename.
    Used to pick which import survives in a duplicate group."""
    edits = imp.get("editedBoxes") if isinstance(imp.get("editedBoxes"), list) else []
    n_edits = len(edits)
    dets = imp.get("detections") if isinstance(imp.get("detections"), list) else []
    n_dets = sum(1 for d in dets if isinstance(d, dict) and not d.get("rejected"))
    has_filename = 1 if imp.get("filename") else 0
    return (n_edits, n_dets, has_filename)


@app.post(
    "/api/v2/projects/{project_id}/imports/dedupe",
    dependencies=[Depends(require_project_owner)],
)
async def v2_dedupe_imports(project_id: str, payload: DedupeIn):
    """Find (and optionally remove) duplicate imports.

    `strategy=exact` groups by SHA256 of the on-disk bytes — catches
    literal repeat uploads of the same file. Fast but misses JPEG
    re-encodes of the same image.

    `strategy=near` reads the already-computed per-image DINOv2
    embeddings (the same ones the dataset-stats variation plot uses)
    and groups by cosine similarity above `threshold`. Catches the
    common "I dragged in my Roboflow export AND the original raws"
    case where the bytes differ but the pixels are visually the same.

    Each group's keeper is the import with the most edited boxes,
    then most accepted detections, then anything with a filename — so
    we never throw away the labelled copy.

    `mode=preview` returns the groups without touching anything. The
    FE shows them in a confirm modal and then re-posts with
    `mode=commit` to actually delete. Same write-lock + cover-rescue
    path as the bulk-delete endpoint above so an in-flight labelling
    job can't race a dedup.
    """
    proj = project_dir(project_id)
    if not proj.exists():
        raise HTTPException(404, "project not found")
    if payload.mode not in ("preview", "commit"):
        raise HTTPException(400, "mode must be 'preview' or 'commit'")
    if payload.strategy not in ("exact", "near", "auto"):
        raise HTTPException(400, "strategy must be 'exact', 'near', or 'auto'")

    m = await asyncio.to_thread(load_manifest, project_id, False)
    if not m:
        raise HTTPException(404, "manifest not found")
    imports = [imp for imp in (m.get("imports") or []) if isinstance(imp, dict) and imp.get("id")]
    ignored = set(m.get("ignored_near_dups") or [])
    groups: list[dict] = []

    # Unified union-find over all imports. Both byte-hash duplicates AND
    # embedding-near-duplicates feed edges into the same forest, so a
    # group like A==B (same JPEG bytes), B~C (re-encoded visual match)
    # collapses transitively into a single cluster {A,B,C}.
    all_ids = [imp.get("id") for imp in imports]
    idx_of: dict[str, int] = {iid: i for i, iid in enumerate(all_ids) if iid}
    n_all = len(all_ids)
    parent_all = list(range(n_all))

    def find_all(x: int) -> int:
        while parent_all[x] != x:
            parent_all[x] = parent_all[parent_all[x]]
            x = parent_all[x]
        return x

    def union_all(a: int, b: int) -> None:
        ra, rb = find_all(a), find_all(b)
        if ra != rb:
            parent_all[ra] = rb

    # Pass 1: exact byte-hash. Cheap-ish I/O, catches "literally the
    # same JPEG re-uploaded" even when the import is too new to have
    # a DINOv2 embedding on disk yet.
    #
    # We ALSO need the hash buckets for "near" mode: when reporting
    # near-duplicates we want to exclude any image that's already a
    # member of an exact-byte-duplicate group so the same set never
    # surfaces in both the "100% duplicates" and "Near duplicates"
    # tabs in the FE.
    def _hash_pass() -> dict[str, list[int]]:
        buckets: dict[str, list[int]] = {}
        for i, imp in enumerate(imports):
            iid = imp.get("id")
            if not iid or iid in ignored:
                continue
            fn = imp.get("filename")
            if not fn:
                continue
            h = _import_file_sha256(project_id, fn)
            if h is None:
                continue
            buckets.setdefault(h, []).append(i)
        return buckets

    hash_buckets: dict[str, list[int]] = {}
    if payload.strategy in ("exact", "near", "auto"):
        hash_buckets = await asyncio.to_thread(_hash_pass)

    # IDs that appear in an exact-duplicate group of size >= 2.
    # Excluded from the embedding pass below so "near" mode only
    # surfaces clusters that AREN'T already covered by the exact tab.
    exact_dup_ids: set[str] = set()
    for members_idx in hash_buckets.values():
        if len(members_idx) < 2:
            continue
        for k in members_idx:
            iid = all_ids[k] if 0 <= k < n_all else None
            if iid:
                exact_dup_ids.add(iid)

    if payload.strategy in ("exact", "auto"):
        for members_idx in hash_buckets.values():
            if len(members_idx) < 2:
                continue
            for j in range(1, len(members_idx)):
                union_all(members_idx[0], members_idx[j])

    # Pass 2: embedding cosine similarity. Catches re-encoded copies
    # (different bytes, same pixels) which the hash pass misses.
    # Skipped for any import already in an exact-duplicate group so
    # the 100% / near views are mutually exclusive.
    if payload.strategy in ("near", "auto"):
        try:
            import numpy as _np
        except Exception:
            raise HTTPException(500, "numpy not available")

        def _gather_embeddings() -> tuple[list[str], "np.ndarray | None"]:
            local_ids: list[str] = []
            local_vecs: list = []
            for imp in imports:
                iid = imp.get("id")
                if not iid or iid in ignored:
                    continue
                # "near" mode excludes images already accounted for in
                # the exact-byte tab; "auto" merges everything so it
                # keeps the full set.
                if payload.strategy == "near" and iid in exact_dup_ids:
                    continue
                v = _load_image_embedding(project_id, iid)
                if v is None:
                    continue
                local_ids.append(iid)
                local_vecs.append(v)
            if not local_vecs:
                return local_ids, None
            mat = _np.stack(local_vecs, axis=0).astype(_np.float32, copy=False)
            norms = _np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            mat = mat / norms
            return local_ids, mat
        ids_with_embed, mat = await asyncio.to_thread(_gather_embeddings)
        if mat is not None and len(ids_with_embed) >= 2:
            sims = mat @ mat.T
            n_e = len(ids_with_embed)
            for i in range(n_e):
                ai = idx_of.get(ids_with_embed[i])
                if ai is None:
                    continue
                for j in range(i + 1, n_e):
                    if sims[i, j] >= payload.threshold:
                        bi = idx_of.get(ids_with_embed[j])
                        if bi is not None:
                            union_all(ai, bi)

    # Materialise clusters from the unified forest.
    if True:
        clusters: dict[int, list[int]] = {}
        for i in range(n_all):
            iid = all_ids[i]
            if not iid or iid in ignored:
                continue
            clusters.setdefault(find_all(i), []).append(i)
        by_id = {imp.get("id"): imp for imp in imports}
        for members_idx in clusters.values():
            if len(members_idx) < 2:
                continue
            members = [by_id[all_ids[i]] for i in members_idx if all_ids[i] in by_id]
            if len(members) < 2:
                continue
            members.sort(key=_import_keeper_score, reverse=True)
            keeper = members[0]
            keeper_id = keeper.get("id")
            groups.append({
                "keep": keeper_id,
                "keep_filename": keeper.get("filename"),
                "drop": [{"id": x.get("id"), "filename": x.get("filename")} for x in members[1:]],
                # Stable per-group id for FE list-key + bookkeeping.
                # Using the keeper id (which is unique per cluster)
                # avoids the min_sim-collision bug from the previous
                # impl where multiple clusters shared "min_sim=1.000".
                "key": keeper_id or "",
            })

    if payload.mode == "preview":
        return {
            "strategy": payload.strategy,
            "threshold": payload.threshold,
            "groups": groups,
            "drop_count": sum(len(g["drop"]) for g in groups),
            "total_imports": len(imports),
        }

    # commit path — drop everyone in `drop` lists. Use the same write-
    # lock + cover-rescue dance as the bulk-delete endpoint so we
    # don't reinvent the wheel.
    drop_ids = {d["id"] for g in groups for d in g["drop"] if d.get("id")}
    if not drop_ids:
        return {"deleted": [], "groups": groups, "drop_count": 0}
    write_lock = await _manifest_write_lock(project_id)
    async with write_lock:
        manifest = load_manifest(project_id)
        if not manifest:
            raise HTTPException(404, "manifest not found")
        kept_imports: list[dict] = []
        deleted_ids: list[str] = []
        for imp in (manifest.get("imports") or []):
            iid = imp.get("id")
            if iid in drop_ids:
                deleted_ids.append(iid)
                fn = imp.get("filename")
                if fn:
                    try:
                        (proj / "images" / fn).unlink(missing_ok=True)
                    except Exception as e:
                        print(f"[v2-dedupe] couldn't unlink {fn}: {e}")
                _invalidate_labelled_preview(project_id, iid)
            else:
                kept_imports.append(imp)
        manifest["imports"] = kept_imports
        cover_filename = manifest.get("cover")
        if cover_filename and not any(
            imp.get("filename") == cover_filename for imp in kept_imports
        ) and not any(
            ref.get("filename") == cover_filename
            for ref in (manifest.get("references") or [])
        ):
            import random as _rnd
            kept_filenames = [
                imp.get("filename") for imp in kept_imports if imp.get("filename")
            ]
            ref_filenames = [
                ref.get("filename")
                for ref in (manifest.get("references") or [])
                if ref.get("filename")
            ]
            new_cover: str | None = None
            if kept_filenames:
                new_cover = _rnd.choice(kept_filenames)
            elif ref_filenames:
                new_cover = _rnd.choice(ref_filenames)
            manifest["cover"] = new_cover
            manifest["cover_blurhash"] = None
        manifest["updatedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        save_manifest(project_id, manifest)
    # Invalidate cached payloads + sidecars so the FE's next page load
    # gets fresh data. Without this, /overview + /dataset-stats served
    # their PRE-delete cached responses to the very first request after
    # the user's page reload (stale-while-revalidate path), making the
    # count appear to "revert" to the original number for a few seconds
    # before the background revalidate landed.
    _invalidate_project_payloads(project_id)
    return {
        "deleted": deleted_ids,
        "groups": groups,
        "drop_count": len(deleted_ids),
    }


class IgnoreDupsIn(BaseModel):
    """Body for /imports/dedupe/ignore. `ids` are import IDs the user
    accepts as duplicates but wants to keep — the near-duplicate
    detection skips them going forward so the stats card stops
    reporting them and the dedupe review modal no longer surfaces
    them as candidates for deletion."""

    ids: list[str]


@app.post(
    "/api/v2/projects/{project_id}/imports/dedupe/ignore",
    dependencies=[Depends(require_project_owner)],
)
async def v2_ignore_dups(project_id: str, payload: IgnoreDupsIn):
    """Mark import IDs as 'accepted duplicates' so they're no longer
    reported in near-dup detection. Persisted in
    `manifest.ignored_near_dups` (a sorted list of IDs); the stats +
    dedupe endpoints filter any pair where either member is in this
    set."""
    if not project_dir(project_id).exists():
        raise HTTPException(404, "project not found")
    add_ids = [x for x in (payload.ids or []) if isinstance(x, str) and x]
    if not add_ids:
        return {"ignored": [], "added": 0}
    write_lock = await _manifest_write_lock(project_id)
    added = 0
    async with write_lock:
        manifest = load_manifest(project_id)
        if not manifest:
            raise HTTPException(404, "manifest not found")
        existing = set(manifest.get("ignored_near_dups") or [])
        before = len(existing)
        existing.update(add_ids)
        added = len(existing) - before
        manifest["ignored_near_dups"] = sorted(existing)
        manifest["updatedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        save_manifest(project_id, manifest)
    _invalidate_project_payloads(project_id)
    return {"ignored": sorted(existing), "added": added}


async def _serve_import_view(project_id: str, filename: str, src_path: Path, target: int):
    """Serve a downscaled JPEG of an import (longest edge <= `target`), cached on
    disk. Lets the image viewer paint a 4K original almost immediately instead of
    streaming the full file; box/mask coords are unaffected (the FE renders them
    in the ORIGINAL pixel space via imageWidth/imageHeight). Re-rendered when the
    source changes; falls back to the original on any render failure."""
    if not src_path.exists():
        raise HTTPException(404, "image not found")
    import hashlib
    cache_dir = project_dir(project_id) / ".viewcache"
    name_hash = hashlib.sha1(filename.encode("utf-8")).hexdigest()[:12]
    cached_path = cache_dir / f"{name_hash}_{target}.jpg"
    try:
        src_mtime = src_path.stat().st_mtime
    except OSError:
        src_mtime = 0.0
    needs_render = True
    if cached_path.exists():
        try:
            if cached_path.stat().st_mtime >= src_mtime:
                needs_render = False
        except OSError:
            needs_render = True
    if needs_render:
        loop = asyncio.get_running_loop()

        def _render() -> None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            with PILImage.open(src_path) as im:
                # exif_transpose bakes any orientation in so the smaller preview
                # is visually consistent with the full original the browser
                # auto-orients on the next progressive swap.
                im = ImageOps.exif_transpose(im).convert("RGB")
                W, H = im.size
                longest = max(W, H)
                if longest > target:
                    scale = target / float(longest)
                    im = im.resize((max(1, round(W * scale)), max(1, round(H * scale))), PILImage.LANCZOS)
            tmp = cached_path.with_suffix(_unique_tmp_suffix())
            im.save(tmp, format="JPEG", quality=85, optimize=True, progressive=True)
            tmp.replace(cached_path)

        lock = _thumb_render_lock(project_id, f"view_{name_hash}_{target}")
        async with lock:
            try:
                cm = cached_path.stat().st_mtime if cached_path.exists() else 0.0
            except OSError:
                cm = 0.0
            if cm < src_mtime:
                try:
                    await loop.run_in_executor(None, _render)
                except Exception as e:
                    print(f"[import-view] render failed for {project_id}/{filename}: {e}")
                    return await _serve_cached_image(project_id, "imports", filename, src_path)
    return await _serve_cached_image(project_id, "import_view", cached_path.name, cached_path)


@app.get(
    "/api/v2/projects/{project_id}/imports/{filename}",
    dependencies=[Depends(require_project_read_access)],
)
async def v2_serve_import(project_id: str, filename: str, w: int = 0):
    """Serve a stored import image, with RAM-cached bytes for repeat hits.
    Path-traversal guarded. With ?w=N the viewer gets a downscaled display
    variant (longest edge <= N) so 4K originals load fast; ?w=0 (default) serves
    the full original."""
    proj = project_dir(project_id)
    imports_root = (proj / "images").resolve()
    target = (imports_root / filename).resolve()
    try:
        target.relative_to(imports_root)
    except ValueError:
        raise HTTPException(403, "forbidden")
    view_w = max(0, min(4096, int(w or 0)))
    if view_w > 0:
        return await _serve_import_view(project_id, filename, target, view_w)
    return await _serve_cached_image(project_id, "imports", filename, target)


# ─── Augmentation preview ─────────────────────────────────────────
# GPU-accelerated preview pipeline used by the Augmentations tab.
# Loads a single image, applies the requested camera/sensor effects
# in-place on a torch tensor, and returns a JPEG. Nothing is
# persisted — the FE drives this live as the user moves the dials
# so the preview reflects the chain immediately.


class AugmentPreviewIn(BaseModel):
    # Image source: "reference" or "import". Both resolve via the
    # project's /references/ or /imports/ subdir respectively.
    source: str
    filename: str
    # Camera / sensor dials, 0..10 each (floats — the FE picks
    # step 0.1 for finer control). 0 = identity. We map the 0..10
    # range onto perceptually-sensible strengths inside the kernel
    # so the FE can stay simple.
    motion_blur: float = 0
    noise: float = 0
    colour_distortion: float = 0
    chromatic_aberration: float = 0
    bit_depth: float = 0
    # Four newer camera/sensor dials, also 0..10:
    #   lens_distortion → wide-angle barrel warp
    #   pixelation       → nearest-neighbour downsample/upsample
    #   low_resolution   → bilinear downsample/upsample (soft blur)
    #   lens_glare       → off-axis radial highlight overlay
    lens_distortion: float = 0
    pixelation: float = 0
    low_resolution: float = 0
    lens_glare: float = 0
    # Distortion category — geometric warps applied before the
    # sensor chain. Perspective_warp is a 0..10 strength dial.
    # scale_min/max + rot_min/max define a range and the backend
    # samples one (scale, rotation) per render using the seed so
    # the preview is stable across slider moves.
    perspective_warp: float = 0
    scale_min: float = 1.0
    scale_max: float = 1.0
    rot_min: float = 0.0
    rot_max: float = 0.0
    # Random block occlusion. Boxes are placed INSIDE the detection
    # polygons of the chosen import (no effect on references — they
    # don't carry detections). The dial controls how large each box
    # is, as a fraction of the per-detection bounding box.
    block_size: float = 0
    # When true, paint the segmentation polygons' outline on top of
    # the rendered image so the user can see where their detections
    # live in the preview. Honoured for source="import" only.
    show_outlines: bool = False
    # Object-overlay augmentation. overlay_ids reference PNGs
    # previously created via /augment/object_overlay/segment;
    # overlay_scale is the longest-edge size as a fraction of the
    # target image's longest edge (0..1). Each overlay is placed
    # independently with the 50% cap honoured across the cumulative
    # overlay coverage (so two overlays can't sum to >50% on any
    # polygon either). overlay_id stays for backwards compat with
    # an older single-overlay client.
    overlay_id: str | None = None
    overlay_ids: list[str] | None = None
    overlay_scale: float = 0
    # Background randomisation. background_ids reference uploaded
    # images stored under <project>/augment_backgrounds/. When
    # provided AND the source has detections, the preview keeps
    # pixels inside the union of detection polygons unchanged and
    # replaces everything outside with one of the supplied
    # backgrounds (picked at random by the seed).
    background_ids: list[str] | None = None
    # Lighting variation — simulates time-of-day-style scene
    # lighting shifts. 0..10 strength dial; the direction
    # (brighten / darken, warm / cool) is sampled by the seed so
    # different rolls land on different illuminations.
    lighting_strength: float = 0
    # Hue shift — rotates colour hue by a seed-sampled offset
    # proportional to strength. Sits under Distortion in the FE.
    hue_shift: float = 0
    # Optional: seed so the noise pattern is stable across slider
    # moves; the FE generates a seed when it picks a random image
    # and keeps it until the user picks another.
    seed: int | None = None


_AUG_DEVICE: "torch.device | None" = None
# Dedicated single-thread executor for the augment runner. asyncio's
# default ThreadPoolExecutor hops between many threads which makes
# the CUDA context initialise per-thread the first time torch.cuda is
# touched — that re-init takes hundreds of ms and starves the kernel
# of warm caching across images. Pinning all augment work to ONE
# dedicated thread keeps CUDA + the PyTorch caching allocator hot for
# the full duration of a 941-image run while the event loop stays
# free for everything else.
import concurrent.futures as _concurrent_futures
_AUG_EXECUTOR = _concurrent_futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="augment-gpu",
)

# Dedicated pool for non-request-critical per-upload work: NSFW gate,
# blurhash, whole-image embeddings. Keeping these OFF asyncio's default
# executor means a burst of uploads can't starve the request-path I/O
# (file writes, image serving, manifest reads) that the default pool
# serves, which was the root cause of the "6 fast then 10-20s pause"
# import stutter. Sized to the CPU count (capped at 8): the per-image
# import work that dominates here is the NudeNet NSFW check, which is
# CPU-bound ONNX, so 3 workers throttled a bulk import on multi-core
# boxes. GPU embeddings still serialise behind the gpu_lock regardless
# of pool size (only one holds the GPU at a time), so widening the pool
# speeds the CPU decode/NSFW/blurhash path without storming the GPU.
_BG_IMAGE_EXECUTOR = _concurrent_futures.ThreadPoolExecutor(
    max_workers=max(4, min(8, (os.cpu_count() or 4))),
    thread_name_prefix="bg-image",
)


def _aug_device() -> "torch.device":
    """Lazy-pick a device the first time the augment endpoint runs.
    GPU when available; CPU fallback so dev environments without
    CUDA still serve previews (just slower)."""
    global _AUG_DEVICE
    if _AUG_DEVICE is None:
        _AUG_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _AUG_DEVICE


def _aug_apply(
    img_tensor: "torch.Tensor",
    motion_blur: float,
    noise: float,
    colour_distortion: float,
    chromatic_aberration: float,
    bit_depth: float,
    seed: int,
    *,
    lens_distortion: float = 0.0,
    pixelation: float = 0.0,
    low_resolution: float = 0.0,
    lens_glare: float = 0.0,
) -> "torch.Tensor":
    """Apply the camera/sensor augmentation chain. `img_tensor` is
    [C=3, H, W], float, range [0, 1], already on the target device.
    Returns a new tensor on the same device.

    Strengths are 0..10 floats; 0 short-circuits each kernel. Order
    matches the physical capture pipeline:
        lens distortion (optics)
      → chromatic aberration (lens)
      → colour shift
      → motion blur (shutter)
      → noise (sensor)
      → bit-depth quantisation (ADC)
      → low resolution / pixelation (compressed transmission)
      → lens glare (baked-in optical flare overlaid last so it
                    isn't dimmed by the bit-depth quantisation).

    Six of the dials have their effective max capped below 10 — the
    UI dial still goes 0..10 (so the user gets fine-grained control)
    but the value handed to each kernel is scaled by (cap / 10), so
    a UI value of 10 hits the new cap rather than the old 10. Caps
    were chosen after empirically testing the dial range — past the
    cap each effect stopped looking like the real-world artefact and
    started looking destructive."""
    import torch.nn.functional as TF

    # Per-dial cap. Multiply ui by (cap / 10) before each kernel
    # uses it. UI stays linear 0..10.
    motion_blur = max(0.0, min(10.0, float(motion_blur))) * 0.4   # max 4
    noise = max(0.0, min(10.0, float(noise))) * 0.8                # max 8
    bit_depth = max(0.0, min(10.0, float(bit_depth))) * 0.8        # max 8
    pixelation = max(0.0, min(10.0, float(pixelation))) * 0.15     # max 1.5
    low_resolution = max(0.0, min(10.0, float(low_resolution))) * 0.5  # max 5
    lens_glare = max(0.0, min(10.0, float(lens_glare))) * 0.6      # max 6

    x = img_tensor

    # ── Lens distortion ──
    # Wide-angle barrel warp. Dial=10 → strong fisheye; dial<0 would
    # be pincushion (we don't expose negative values in the UI but
    # the kernel handles them so internal callers can opt in). The
    # warp resamples via a polynomial radial map so straight lines
    # near the edges curve outward, matching what a cheap action-cam
    # lens does. Implemented with grid_sample on a normalised radial
    # field — no Python loops, runs on the same device as x.
    if lens_distortion > 0:
        _, H, W = x.shape
        # Wide-angle / fisheye barrel only. The dial used to be
        # bidirectional (negative = pincushion), but pincushion
        # fundamentally needs source content beyond the frame —
        # without it, the negative side either painted black
        # corners or read as a near no-op even under normalisation.
        # The user wanted the dial to be primarily fisheye, so
        # negative values are now treated as 0 and the slider in
        # the UI clamps to 0..+10.
        # Strength 0..+10 → barrel coefficient 0..0.55. Output
        # corner factor = 1 - 2·k1 → 0.45 at full deflection, so
        # the corner pixel samples from source r≈0.45 — a clear,
        # action-cam-style bulge.
        k1 = float(lens_distortion) / 10.0 * 0.55
        ys = torch.linspace(-1.0, 1.0, H, device=x.device, dtype=x.dtype)
        xs = torch.linspace(-1.0, 1.0, W, device=x.device, dtype=x.dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        r2 = grid_x * grid_x + grid_y * grid_y
        factor = 1.0 - k1 * r2
        src_x = grid_x * factor
        src_y = grid_y * factor
        grid = torch.stack([src_x, src_y], dim=-1).unsqueeze(0)
        x = TF.grid_sample(
            x.unsqueeze(0), grid,
            mode="bilinear", padding_mode="zeros", align_corners=True,
        ).squeeze(0)

    # ── Chromatic aberration ──
    # Shift R channel right + B channel left by N px proportional to
    # strength. Replicates the colour fringing you see on cheap lenses.
    if chromatic_aberration > 0:
        max_shift = max(1, int(round(chromatic_aberration * 0.6)))
        # Roll-and-pad: keep tensor size constant, lose `max_shift`
        # px from one edge per channel.
        r = torch.roll(x[0:1], shifts=max_shift, dims=2)
        b = torch.roll(x[2:3], shifts=-max_shift, dims=2)
        # Zero the wrapped column so we don't get a colour bar
        # creeping in from the opposite edge.
        r[..., :max_shift] = x[0:1, ..., :max_shift]
        b[..., -max_shift:] = x[2:3, ..., -max_shift:]
        x = torch.cat([r, x[1:2], b], dim=0)

    # ── Colour distortion ──
    # Brightness + contrast + saturation jitter, deterministic per
    # seed so a static slider doesn't shimmer with random changes.
    if colour_distortion > 0:
        strength = float(colour_distortion) / 10.0  # 0..1
        g = torch.Generator(device="cpu").manual_seed(seed ^ 0xC0CC)
        # All three jitter factors sampled once per render.
        b_jit = float(1.0 + (torch.rand(1, generator=g).item() - 0.5) * strength * 0.6)
        c_jit = float(1.0 + (torch.rand(1, generator=g).item() - 0.5) * strength * 0.6)
        s_jit = float(1.0 + (torch.rand(1, generator=g).item() - 0.5) * strength * 1.0)
        # Brightness — scalar multiply, clamped.
        x = (x * b_jit).clamp(0.0, 1.0)
        # Contrast — pull values toward / away from grey at 0.5.
        mean = x.mean(dim=(1, 2), keepdim=True)
        x = ((x - mean) * c_jit + mean).clamp(0.0, 1.0)
        # Saturation — pull toward / away from luminance grey.
        # Rec. 709 luma weights.
        luma = (0.2126 * x[0] + 0.7152 * x[1] + 0.0722 * x[2]).unsqueeze(0)
        x = (luma + (x - luma) * s_jit).clamp(0.0, 1.0)

    # ── Motion blur ──
    # Depthwise convolution with a horizontal line kernel. Kernel
    # length scales smoothly with the float dial — strength 0
    # short-circuits, strength 10 ends at a 21-px kernel.
    if motion_blur > 0:
        k = 1 + 2 * int(round(float(motion_blur)))
        if k > 1:
            kernel = torch.zeros((3, 1, 1, k), device=x.device, dtype=x.dtype)
            kernel[:, :, 0, :] = 1.0 / k
            x = TF.conv2d(
                x.unsqueeze(0),
                kernel,
                padding=(0, k // 2),
                groups=3,
            ).squeeze(0).clamp(0.0, 1.0)

    # ── Sensor noise ──
    # Additive Gaussian noise. Seeded so a steady slider yields a
    # steady grain pattern — otherwise every preview render
    # twinkles, which looks unstable.
    if noise > 0:
        sigma = (float(noise) / 10.0) * 0.15  # 0..0.15
        g = torch.Generator(device="cpu").manual_seed(seed ^ 0xA15E)
        # Generate on CPU then move so the seed is portable.
        n = torch.randn(x.shape, generator=g, dtype=torch.float32).to(x.device) * sigma
        x = (x + n).clamp(0.0, 1.0)

    # ── Bit-depth quantisation ──
    # Posterise by snapping to a smaller set of levels. Dial maps
    # 0 → 8 bits (identity), 10 → 1 bit (harsh two-level). Levels
    # interpolated smoothly so float steps land between integer
    # bit depths.
    if bit_depth > 0:
        # Effective bits per channel: 8 at dial=0, 1 at dial=10.
        bits = 8.0 - (float(bit_depth) / 10.0) * 7.0
        levels = max(2.0, 2.0 ** bits)
        x = (torch.round(x * (levels - 1.0)) / (levels - 1.0)).clamp(0.0, 1.0)

    # ── Low resolution ──
    # Simulates a low-res capture: downsample with bilinear filtering
    # then upsample back to the original size with bilinear so edges
    # come back smooth-soft. Distinct from pixelation (which uses
    # nearest-neighbour so the blocks are crisp). Dial=10 takes the
    # image down to 1/12 of its original long edge.
    if low_resolution > 0:
        _, H, W = x.shape
        s = float(low_resolution) / 10.0
        long_edge = max(H, W)
        # 0 → 1×, 10 → 1/12×. The +1 prevents divide-by-zero at low
        # strengths; the floor at 8 guarantees the downsample stays a
        # legal tensor size.
        small = max(8, int(round(long_edge / (1.0 + s * 11.0))))
        if H >= W:
            new_h = small
            new_w = max(8, int(round(W * (small / float(H)))))
        else:
            new_w = small
            new_h = max(8, int(round(H * (small / float(W)))))
        if (new_w, new_h) != (W, H):
            x = TF.interpolate(
                x.unsqueeze(0),
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            x = TF.interpolate(
                x, size=(H, W), mode="bilinear", align_corners=False,
            ).squeeze(0).clamp(0.0, 1.0)

    # ── Pixelation ──
    # Like low_resolution but with nearest-neighbour upsample so the
    # blocks stay hard-edged. Reads as deliberate pixel-art rather
    # than a soft-focus low-quality capture.
    if pixelation > 0:
        _, H, W = x.shape
        s = float(pixelation) / 10.0
        long_edge = max(H, W)
        # 0 → 1×, 10 → 1/40× (chunky 25-pixel blocks on a 1024 long
        # edge). Pixelation is more aggressive than low_resolution
        # because the visual signature kicks in faster.
        small = max(4, int(round(long_edge / (1.0 + s * 39.0))))
        if H >= W:
            new_h = small
            new_w = max(4, int(round(W * (small / float(H)))))
        else:
            new_w = small
            new_h = max(4, int(round(H * (small / float(W)))))
        if (new_w, new_h) != (W, H):
            x = TF.interpolate(
                x.unsqueeze(0),
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            x = TF.interpolate(
                x, size=(H, W), mode="nearest",
            ).squeeze(0).clamp(0.0, 1.0)

    # ── Lens glare ──
    # Bright off-axis radial highlights overlaid additively to mimic
    # sunlight or strong point-source bleed through a cheap lens.
    # One or two glares per render, position + tint sampled by the
    # seed so different rolls land differently. Strength scales both
    # the intensity and the apparent size of the highlight halo.
    if lens_glare > 0:
        _, H, W = x.shape
        s = float(lens_glare) / 10.0  # 0..1
        gen = torch.Generator(device="cpu").manual_seed(seed ^ 0x91A1E)
        # Two glares — primary bright, secondary smaller and dimmer.
        # Both placed off-axis (avoid centre) so they read as flare,
        # not a vignette.
        ys = torch.linspace(-1.0, 1.0, H, device=x.device, dtype=x.dtype)
        xs = torch.linspace(-1.0, 1.0, W, device=x.device, dtype=x.dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        accum = torch.zeros_like(x[0])
        # Primary glare
        ang1 = float(torch.rand(1, generator=gen).item()) * 6.2831853
        rad1 = 0.45 + float(torch.rand(1, generator=gen).item()) * 0.4
        cx1 = float(rad1 * torch.cos(torch.tensor(ang1)).item())
        cy1 = float(rad1 * torch.sin(torch.tensor(ang1)).item())
        sigma1 = 0.25 + s * 0.35  # halo size grows with strength
        d2_1 = (gx - cx1) ** 2 + (gy - cy1) ** 2
        accum = accum + torch.exp(-d2_1 / (2.0 * sigma1 * sigma1))
        # Secondary glare — opposite-ish side, smaller, dimmer.
        ang2 = ang1 + 3.1415926 + (float(torch.rand(1, generator=gen).item()) - 0.5) * 1.2
        rad2 = 0.3 + float(torch.rand(1, generator=gen).item()) * 0.3
        cx2 = float(rad2 * torch.cos(torch.tensor(ang2)).item())
        cy2 = float(rad2 * torch.sin(torch.tensor(ang2)).item())
        sigma2 = 0.15 + s * 0.2
        d2_2 = (gx - cx2) ** 2 + (gy - cy2) ** 2
        accum = accum + 0.6 * torch.exp(-d2_2 / (2.0 * sigma2 * sigma2))
        # Warm tint (sun-like). Saturation pulled toward neutral as
        # strength grows so a heavy glare washes the highlights out
        # toward white instead of clipping orange.
        tint = torch.tensor(
            [1.0, 0.92 + 0.08 * s, 0.78 + 0.22 * s],
            device=x.device, dtype=x.dtype,
        ).view(3, 1, 1)
        boost = s * 0.85  # peak additive brightness
        glare = accum.unsqueeze(0) * tint * boost
        x = (x + glare).clamp(0.0, 1.0)

    return x


def _aug_apply_lighting(
    img: "torch.Tensor",
    strength: float,
    seed: int,
) -> "torch.Tensor":
    """Time-of-day-style lighting variation. strength is 0..10;
    the kernel samples one brightness direction, one warmth shift
    and one contrast scale per render from the seed — so different
    rolls land darker / cooler vs brighter / warmer. Pure tensor
    arithmetic on the input tensor's device."""
    if strength <= 0:
        return img
    s = max(0.0, min(1.0, float(strength) / 10.0))
    g = torch.Generator(device="cpu").manual_seed(seed ^ 0xDA77)
    rs = torch.rand(3, generator=g).tolist()
    # Brightness multiplier in [1 - 0.6s, 1 + 0.6s].
    b = 1.0 + (rs[0] - 0.5) * 2.0 * 0.6 * s
    # Contrast scale in [1 - 0.4s, 1 + 0.4s].
    c = 1.0 + (rs[1] - 0.5) * 2.0 * 0.4 * s
    # Warmth shift — pushes R up and B down (or vice versa) by
    # up to ±0.10. Independent of brightness so dark+warm and
    # dark+cool are both reachable.
    w = (rs[2] - 0.5) * 2.0 * 0.10 * s

    x = (img * b).clamp(0.0, 1.0)
    mean = x.mean(dim=(1, 2), keepdim=True)
    x = ((x - mean) * c + mean).clamp(0.0, 1.0)
    # Per-channel warmth (no in-place on a tracked tensor).
    r = (x[0] + w).clamp(0.0, 1.0)
    g_ch = x[1]
    b_ch = (x[2] - w).clamp(0.0, 1.0)
    return torch.stack([r, g_ch, b_ch], dim=0)


def _aug_apply_hue_shift(
    img: "torch.Tensor",
    strength: float,
    seed: int,
) -> "torch.Tensor":
    """Random hue rotation. strength 0..10 → hue factor in
    [-0.5s, 0.5s] (torchvision's adjust_hue convention). Seeded
    so a steady slider doesn't shimmer."""
    if strength <= 0:
        return img
    from torchvision.transforms.functional import adjust_hue as _adjust_hue
    s = max(0.0, min(1.0, float(strength) / 10.0))
    g = torch.Generator(device="cpu").manual_seed(seed ^ 0xC0F2)
    factor = (float(torch.rand(1, generator=g).item()) - 0.5) * 2.0 * (0.5 * s)
    # adjust_hue clamps to [-0.5, 0.5]; clamp defensively.
    factor = max(-0.5, min(0.5, factor))
    out = _adjust_hue(img.unsqueeze(0), float(factor)).squeeze(0)
    return out.clamp(0.0, 1.0)


# ─── Geometric warp helpers ─────────────────────────────────────
# Shared math used by both the live preview kernels and the
# augment_generate runner — the runner needs the same matrices the
# kernels apply so it can warp detection polygons in lock-step
# with the image and save them next to each augmentation copy.


def _perspective_corner_pairs(
    strength: float, W: int, H: int, seed: int,
) -> "tuple[list[list[float]], list[list[float]]] | None":
    """Replicate the sampling _aug_apply_perspective_warp does so
    we can compute the forward homography for polygon warping in
    augment_generate. Returns None when strength is 0 (identity)."""
    if strength <= 0:
        return None
    distortion = max(0.0, min(0.19, (float(strength) / 10.0) * 0.19))
    g = torch.Generator(device="cpu").manual_seed(seed ^ 0x991C)
    rand = torch.rand(8, generator=g).tolist()
    startpoints: list[list[float]] = [
        [0.0, 0.0],
        [float(W - 1), 0.0],
        [float(W - 1), float(H - 1)],
        [0.0, float(H - 1)],
    ]
    endpoints: list[list[float]] = []
    for i, (sx_, sy_) in enumerate(startpoints):
        ex = sx_ + (rand[i * 2] - 0.5) * 2.0 * float(W) * distortion
        ey = sy_ + (rand[i * 2 + 1] - 0.5) * 2.0 * float(H) * distortion
        endpoints.append([ex, ey])
    return startpoints, endpoints


def _solve_homography(
    startpoints: list[list[float]], endpoints: list[list[float]],
) -> "list[list[float]]":
    """Direct linear transform — solve the 8x8 system for the 8
    parameters of a 3x3 homography (last entry fixed at 1) that
    maps each `startpoint` to its matching `endpoint`."""
    import numpy as _np
    A = []
    bvec = []
    for (sx, sy), (ex, ey) in zip(startpoints, endpoints):
        A.append([sx, sy, 1, 0, 0, 0, -ex * sx, -ex * sy])
        bvec.append(ex)
        A.append([0, 0, 0, sx, sy, 1, -ey * sx, -ey * sy])
        bvec.append(ey)
    M = _np.linalg.solve(
        _np.array(A, dtype=_np.float64),
        _np.array(bvec, dtype=_np.float64),
    )
    return [
        [float(M[0]), float(M[1]), float(M[2])],
        [float(M[3]), float(M[4]), float(M[5])],
        [float(M[6]), float(M[7]), 1.0],
    ]


def _scale_rotation_sample_params(
    smin: float, smax: float, rmin: float, rmax: float, seed: int,
) -> "tuple[float, float] | None":
    """Replicate _aug_apply_scale_rotation's sampling so the
    runner can build the same affine matrix for the polygons.
    Returns (scale, rotation_deg) or None for identity."""
    if smin == 1.0 and smax == 1.0 and rmin == 0.0 and rmax == 0.0:
        return None
    g = torch.Generator(device="cpu").manual_seed(seed ^ 0x5C7A)
    rs = torch.rand(2, generator=g).tolist()
    smin_c = max(0.05, min(5.0, float(smin)))
    smax_c = max(smin_c, min(5.0, float(smax)))
    rmin_c = max(-180.0, min(180.0, float(rmin)))
    rmax_c = max(rmin_c, min(180.0, float(rmax)))
    s = smin_c + (smax_c - smin_c) * rs[0]
    r = rmin_c + (rmax_c - rmin_c) * rs[1]
    return s, r


def _scale_rotation_matrix(
    s: float, r_deg: float, W: int, H: int,
) -> "list[list[float]]":
    """Forward affine matrix matching torchvision.transforms.
    functional.affine(image, angle=r, scale=s) — torchvision's
    `angle` is "clockwise direction" on screen, so in y-down image
    coordinates the rotation matrix is [[cos, -sin], [sin, cos]]
    (a point at (x,0) rotates to (cos·x, sin·x), which moves
    visually clockwise as y is screen-down)."""
    import math
    cx = float(W) / 2.0
    cy = float(H) / 2.0
    a = math.radians(float(r_deg))
    cos_a = math.cos(a)
    sin_a = math.sin(a)
    return [
        [s * cos_a, -s * sin_a, cx - s * (cos_a * cx - sin_a * cy)],
        [s * sin_a,  s * cos_a, cy - s * (sin_a * cx + cos_a * cy)],
        [0.0, 0.0, 1.0],
    ]


def _matmul_3x3(A: "list[list[float]]", B: "list[list[float]]") -> "list[list[float]]":
    out = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    for i in range(3):
        for j in range(3):
            out[i][j] = A[i][0] * B[0][j] + A[i][1] * B[1][j] + A[i][2] * B[2][j]
    return out


def _apply_matrix_to_point(M: "list[list[float]]", x: float, y: float) -> "tuple[float, float]":
    w = M[2][0] * x + M[2][1] * y + M[2][2]
    if abs(w) < 1e-9:
        w = 1e-9
    nx = (M[0][0] * x + M[0][1] * y + M[0][2]) / w
    ny = (M[1][0] * x + M[1][1] * y + M[1][2]) / w
    return nx, ny


def _apply_matrix_to_polys(
    polys: "list[list[list[float]]]", M: "list[list[float]] | None",
) -> "list[list[list[float]]]":
    if M is None:
        return polys
    out: list[list[list[float]]] = []
    for poly in polys:
        warped: list[list[float]] = []
        for pt in poly:
            if len(pt) < 2:
                continue
            nx, ny = _apply_matrix_to_point(M, float(pt[0]), float(pt[1]))
            warped.append([nx, ny])
        if warped:
            out.append(warped)
    return out


def _apply_matrix_to_box(
    box: "list[float]", M: "list[list[float]] | None",
) -> "list[float]":
    if M is None or len(box) < 4:
        return list(box)
    corners = [
        (float(box[0]), float(box[1])),
        (float(box[2]), float(box[1])),
        (float(box[2]), float(box[3])),
        (float(box[0]), float(box[3])),
    ]
    xs: list[float] = []
    ys: list[float] = []
    for x, y in corners:
        nx, ny = _apply_matrix_to_point(M, x, y)
        xs.append(nx)
        ys.append(ny)
    return [min(xs), min(ys), max(xs), max(ys)]


def _aug_apply_perspective_warp(
    img: "torch.Tensor",
    strength: float,
    seed: int,
) -> "torch.Tensor":
    """Random perspective warp. `strength` 0..10 → distortion_scale
    0..0.5; the four image corners each get a random offset of up
    to `distortion · dim` in either direction. Runs through
    torchvision's perspective() which builds the homography +
    samples via grid_sample on whatever device the tensor is on."""
    if strength <= 0:
        return img
    from torchvision.transforms.functional import perspective as _tv_perspective
    from torchvision.transforms import InterpolationMode as _IM
    _C, H, W = img.shape
    distortion = max(0.0, min(0.19, (float(strength) / 10.0) * 0.19))
    g = torch.Generator(device="cpu").manual_seed(seed ^ 0x991C)
    rand = torch.rand(8, generator=g).tolist()
    startpoints = [
        [0, 0],
        [W - 1, 0],
        [W - 1, H - 1],
        [0, H - 1],
    ]
    endpoints: list[list[float]] = []
    for i, (sx_, sy_) in enumerate(startpoints):
        ex = float(sx_) + (rand[i * 2] - 0.5) * 2.0 * W * distortion
        ey = float(sy_) + (rand[i * 2 + 1] - 0.5) * 2.0 * H * distortion
        endpoints.append([ex, ey])
    out = _tv_perspective(
        img.unsqueeze(0),
        startpoints,
        endpoints,
        interpolation=_IM.BILINEAR,
        fill=[0.0],
    ).squeeze(0)
    return out.clamp(0.0, 1.0)


def _aug_apply_scale_rotation(
    img: "torch.Tensor",
    scale_min: float,
    scale_max: float,
    rot_min: float,
    rot_max: float,
    seed: int,
) -> "torch.Tensor":
    """Sample one scale in [scale_min, scale_max] and one rotation
    in [rot_min, rot_max] (degrees), apply via torchvision's
    affine() — runs on the input tensor's device."""
    if scale_min == 1.0 and scale_max == 1.0 and rot_min == 0.0 and rot_max == 0.0:
        return img
    from torchvision.transforms.functional import affine as _tv_affine
    from torchvision.transforms import InterpolationMode as _IM
    g = torch.Generator(device="cpu").manual_seed(seed ^ 0x5C7A)
    rs = torch.rand(2, generator=g).tolist()
    smin = max(0.7, min(1.3, float(scale_min)))
    smax = max(smin, min(1.3, float(scale_max)))
    rmin = max(-180.0, min(180.0, float(rot_min)))
    rmax = max(rmin, min(180.0, float(rot_max)))
    s = smin + (smax - smin) * rs[0]
    r = rmin + (rmax - rmin) * rs[1]
    out = _tv_affine(
        img.unsqueeze(0),
        angle=r,
        translate=(0, 0),
        scale=s,
        shear=(0.0, 0.0),
        interpolation=_IM.BILINEAR,
        fill=[0.0],
    ).squeeze(0)
    return out.clamp(0.0, 1.0)


def _rasterise_polygons_to_mask(
    detections: list[dict], H: int, W: int, sx: float, sy: float,
    device: "torch.device",
) -> "torch.Tensor":
    """Rasterise every detection's polygon list into a single binary
    mask, shape [H, W] on `device`. Returns a uint8-style float
    tensor (0.0 / 1.0). Empty when no detection has a polygon."""
    import numpy as _np
    from PIL import Image as _PIL
    from PIL import ImageDraw as _ImageDraw
    canvas = _PIL.new("L", (W, H), 0)
    drawer = _ImageDraw.Draw(canvas)
    for d in detections or []:
        mask = d.get("mask")
        polys = (mask or {}).get("polygons") if isinstance(mask, dict) else None
        if not polys:
            continue
        for poly in polys:
            if not poly or len(poly) < 3:
                continue
            scaled = [(float(x) * sx, float(y) * sy) for (x, y) in poly]
            try:
                drawer.polygon(scaled, fill=255)
            except Exception:
                continue
    arr = _np.asarray(canvas, dtype=_np.float32) / 255.0
    return torch.from_numpy(arr).to(device)


def _aug_apply_block_occlusion(
    img: "torch.Tensor",
    detections: list[dict],
    coverage_pct: float,
    sx: float,
    sy: float,
    seed: int,
) -> "torch.Tensor":
    """Drop one rectangular block per detection polygon, sized so
    its overlap with that polygon is ~`coverage_pct`% of the
    polygon's pixel area. `coverage_pct` is 0..60 (FE dial).

    Every polygon's coverage is a hard ceiling. After placement
    each candidate zap is filtered against EVERY other polygon's
    remaining headroom — so when polygons overlap (a small
    segmentation inside a big one, or two siblings that share
    pixels), no polygon can be pushed past its own pct cap by a
    block placed for a different one. Pixels removed during the
    cap-enforcement pass simply revert to the original image,
    giving the bigger rectangle a "hole" where it would have
    breached an inner polygon's cap."""
    if coverage_pct <= 0:
        return img
    _C, H, W = img.shape
    g = torch.Generator(device="cpu").manual_seed(seed ^ 0xB10C)
    pct = max(0.0, min(60.0, float(coverage_pct))) / 100.0

    # Build (det, poly_mask, area) tuples, drop degenerates, sort by
    # area ascending so the tightest budgets get placed first.
    annotated: list[tuple[dict, "torch.Tensor", float]] = []
    for det in (detections or []):
        polys = (det.get("mask") or {}).get("polygons") if isinstance(det.get("mask"), dict) else None
        if not polys:
            continue
        m = _rasterise_polygons_to_mask([det], H, W, sx, sy, img.device)
        a = float(m.sum().item())
        if a < 4:
            continue
        annotated.append((det, m, a))
    if not annotated:
        return img
    annotated.sort(key=lambda t: t[2])

    n = len(annotated)
    poly_stack = torch.stack([m for (_, m, _) in annotated], dim=0)
    targets_t = torch.tensor(
        [pct * a for (_, _, a) in annotated], device=img.device, dtype=img.dtype,
    )
    zap_mask = torch.zeros((H, W), device=img.device, dtype=img.dtype)

    # Placement pass — one rectangle per polygon, smallest first.
    # Cross-polygon cap enforcement happens in a single batched
    # post-pass below (was previously an N² inner loop with a
    # `.item()` sync on every iteration, which is what was
    # hanging the request for big detection sets).
    for i, (det, poly_mask, _area) in enumerate(annotated):
        # Coverage already inside this polygon from earlier (smaller)
        # blocks → reduces what we still owe.
        current = float((zap_mask * poly_mask).sum().item())
        target_i = pct * _area
        budget_i = target_i - current
        if budget_i < 1.0:
            continue
        side = max(2, int(round(budget_i ** 0.5)))
        bx = det.get("box_xyxy") or det.get("box") or None
        if bx and len(bx) >= 4:
            x0 = max(0, int(float(bx[0]) * sx))
            y0 = max(0, int(float(bx[1]) * sy))
            x1 = min(W, int(float(bx[2]) * sx))
            y1 = min(H, int(float(bx[3]) * sy))
        else:
            ys, xs = torch.nonzero(poly_mask, as_tuple=True)
            if xs.numel() == 0:
                continue
            x0, x1 = int(xs.min().item()), int(xs.max().item()) + 1
            y0, y1 = int(ys.min().item()), int(ys.max().item()) + 1
        bw, bh = max(1, x1 - x0), max(1, y1 - y0)
        kw = min(side, bw)
        kh = min(side, bh)
        ax = x0 + int(torch.randint(0, max(1, bw - kw + 1), (1,), generator=g).item())
        ay = y0 + int(torch.randint(0, max(1, bh - kh + 1), (1,), generator=g).item())
        ax2 = min(W, ax + kw)
        ay2 = min(H, ay + kh)
        if ax2 <= ax or ay2 <= ay:
            continue

        # Add the rectangle directly to zap_mask, clipped to THIS
        # polygon. The post-pass below trims any other polygon that
        # ends up over its cap from the cumulative zap.
        zap_mask[ay:ay2, ax:ax2] = torch.maximum(
            zap_mask[ay:ay2, ax:ax2],
            poly_mask[ay:ay2, ax:ax2],
        )

    # Post-pass — enforce per-polygon caps in one vectorised sweep.
    # Counts every polygon at once, identifies the ones over budget,
    # randomly trims excess pixels from each over-budget polygon's
    # zap. Inner sampling still needs CPU random indices, but each
    # polygon only takes one trim step — way fewer syncs than the
    # old per-detection × per-polygon inner loop.
    counts = (poly_stack * zap_mask.unsqueeze(0)).sum(dim=(1, 2))  # (N,)
    excesses = (counts - targets_t).clamp(min=0)
    # Bring the over-budget table to CPU once; per-polygon trims
    # below are CPU-driven.
    excess_list = excesses.tolist()
    for j in range(n):
        ex = excess_list[j]
        if ex < 1:
            continue
        common = zap_mask * poly_stack[j]
        flat_idx = common.view(-1).nonzero(as_tuple=False).squeeze(-1)
        n_pix = int(flat_idx.numel())
        if n_pix == 0:
            continue
        k = min(int(ex + 0.999), n_pix)
        perm = torch.randperm(n_pix, generator=g)[:k]
        zap_mask.view(-1)[flat_idx[perm]] = 0.0

    if float(zap_mask.sum().item()) <= 0:
        return img

    # Apply: zero the kept channels under the zap, then lift the
    # zapped region slightly off pure black so it reads as
    # "occluded" rather than the segmentation disappearing into the
    # background.
    out = img.clone()
    keep = (1.0 - zap_mask)
    for c in range(3):
        out[c] = out[c] * keep + zap_mask * 0.04
    return out.clamp(0.0, 1.0)


def _aug_apply_segmentation_outlines(
    img: "torch.Tensor",
    detections: list[dict],
    sx: float,
    sy: float,
) -> "torch.Tensor":
    """Draw a thin emerald outline around every detection's polygon
    so the preview shows where the user's segmentation lives. Drawn
    via PIL (fast for the handful of polygons we have) then merged
    back onto the torch tensor with a tight alpha blend."""
    _C, H, W = img.shape
    import numpy as _np
    from PIL import Image as _PIL
    from PIL import ImageDraw as _ImageDraw
    canvas = _PIL.new("RGBA", (W, H), (0, 0, 0, 0))
    drawer = _ImageDraw.Draw(canvas)
    line_w = max(2, int(round(min(W, H) / 320)))  # scales with size
    for d in detections or []:
        mask = d.get("mask")
        polys = (mask or {}).get("polygons") if isinstance(mask, dict) else None
        if not polys:
            continue
        for poly in polys:
            if not poly or len(poly) < 2:
                continue
            scaled = [(float(x) * sx, float(y) * sy) for (x, y) in poly]
            try:
                drawer.line(scaled + [scaled[0]], fill=(52, 211, 153, 235), width=line_w)
            except Exception:
                continue
    arr = _np.asarray(canvas, dtype=_np.float32) / 255.0  # HWC, 4ch
    overlay = torch.from_numpy(arr).to(img.device).permute(2, 0, 1)
    rgb = overlay[:3]
    alpha = overlay[3:4]
    out = img * (1.0 - alpha) + rgb * alpha
    return out.clamp(0.0, 1.0)


def _aug_apply_object_overlay(
    img: "torch.Tensor",
    overlay_path: "Path",
    scale: float,
    detections: list[dict],
    sx: float,
    sy: float,
    seed: int,
    occupied: "torch.Tensor | None" = None,
) -> "tuple[torch.Tensor, torch.Tensor | None]":
    """Composite a previously-segmented transparent PNG onto the
    working image. The overlay is resized so its longest edge is
    `scale × longest_edge(img)`; placement is constrained so the
    overlay's silhouette never covers more than 50% of any existing
    detection polygon.

    Anchor search is vectorised — K candidates are gathered into a
    single (N, K, oh, ow) window stack via advanced indexing on
    GPU, overlaps computed in one tensor op, the best (zero-breach
    or min-breach) anchor picked. Previously each anchor sliced +
    summed inside a Python loop with per-iteration .item() syncs,
    which is what made big overlays take ~10s.

    img: [3, H, W] in [0,1] on some device. Returns (img, occupied)
    where `occupied` is an alpha-coverage accumulator (H, W) so a
    subsequent overlay can account for it when sizing each
    polygon's remaining headroom."""
    _C, H, W = img.shape
    if scale <= 0:
        return img, occupied
    try:
        with PILImage.open(overlay_path) as op:
            overlay_pil = op.convert("RGBA")
    except Exception:
        return img, occupied
    longest = max(W, H)
    target_long = max(8, int(round(longest * float(scale))))
    ow, oh = overlay_pil.size
    if ow <= 0 or oh <= 0:
        return img, occupied
    overlay_long = max(ow, oh)
    if overlay_long > 0:
        s = target_long / float(overlay_long)
        new_w = max(2, int(round(ow * s)))
        new_h = max(2, int(round(oh * s)))
        overlay_pil = overlay_pil.resize((new_w, new_h), PILImage.LANCZOS)
    ow, oh = overlay_pil.size
    if ow > W or oh > H:
        # Overlay larger than the canvas — clamp to canvas and pin
        # to origin (no anchor search possible).
        ow = min(ow, W)
        oh = min(oh, H)

    # Tensorise overlay: RGB + alpha.
    import numpy as _np
    arr = _np.asarray(overlay_pil, dtype=_np.float32) / 255.0  # H,W,4
    device = img.device
    overlay_rgb = torch.from_numpy(arr[..., :3]).permute(2, 0, 1).contiguous().to(device)
    overlay_alpha = torch.from_numpy(arr[..., 3:4]).permute(2, 0, 1).contiguous().to(device)
    if overlay_rgb.shape[1] != oh or overlay_rgb.shape[2] != ow:
        overlay_rgb = overlay_rgb[:, :oh, :ow]
        overlay_alpha = overlay_alpha[:, :oh, :ow]

    # Stack per-detection masks into one (N, H, W) tensor up-front.
    # Each polygon's cap is 50% of its pixel area, minus what
    # earlier overlays in the same batch have already taken from
    # that polygon (so cumulative coverage stays ≤ 50%).
    poly_masks: list["torch.Tensor"] = []
    poly_caps: list[float] = []
    for d in (detections or []):
        polys = (d.get("mask") or {}).get("polygons") if isinstance(d.get("mask"), dict) else None
        if not polys:
            continue
        m = _rasterise_polygons_to_mask([d], H, W, sx, sy, device)
        a = float(m.sum().item())
        if a < 4:
            continue
        cap = 0.5 * a
        if occupied is not None:
            cap -= float((occupied * m).sum().item())
        if cap < 1.0:
            # No room left under this polygon — but we still need to
            # include it in the constraint set so we don't allow ANY
            # new alpha here.
            cap = 0.0
        poly_masks.append(m)
        poly_caps.append(cap)

    g = torch.Generator(device="cpu").manual_seed(seed ^ 0x0BE7)
    max_x = max(0, W - ow)
    max_y = max(0, H - oh)

    if max_x <= 0 and max_y <= 0:
        ax, ay = 0, 0
    elif not poly_masks:
        # No constraints — pick a random anchor in one shot.
        ax = int(torch.randint(0, max(1, max_x + 1), (1,), generator=g).item())
        ay = int(torch.randint(0, max(1, max_y + 1), (1,), generator=g).item())
    else:
        poly_stack = torch.stack(poly_masks, dim=0)
        caps_t = torch.tensor(poly_caps, device=device, dtype=img.dtype)
        N = poly_stack.shape[0]

        # Memory-bounded K: keep the (N, K, oh, ow) gather under
        # ~96 MB. Otherwise drop K accordingly. Worst case K = 4 so
        # we still get SOME search.
        per_window_bytes = oh * ow * 4
        K = min(16, max(4, int(96_000_000 / max(1, N * per_window_bytes))))

        ax_list = torch.randint(0, max(1, max_x + 1), (K,), generator=g).tolist()
        ay_list = torch.randint(0, max(1, max_y + 1), (K,), generator=g).tolist()

        # Build (K, oh, ow) coordinate grids that index into
        # poly_stack. Long dtype required for advanced indexing.
        ay_t = torch.tensor(ay_list, device=device, dtype=torch.long).view(K, 1, 1)
        ax_t = torch.tensor(ax_list, device=device, dtype=torch.long).view(K, 1, 1)
        y_off = torch.arange(oh, device=device, dtype=torch.long).view(1, oh, 1)
        x_off = torch.arange(ow, device=device, dtype=torch.long).view(1, 1, ow)
        y_idx = (ay_t + y_off).expand(K, oh, ow)
        x_idx = (ax_t + x_off).expand(K, oh, ow)

        # poly_stack[:, y_idx, x_idx] broadcasts to (N, K, oh, ow).
        # Single gather; no Python-level per-anchor loop.
        windows = poly_stack[:, y_idx, x_idx]
        alpha_2d = overlay_alpha[0]  # (oh, ow)
        overlaps = (windows * alpha_2d).sum(dim=(2, 3))  # (N, K)
        breaches = (overlaps - caps_t.unsqueeze(1)).clamp(min=0)  # (N, K)
        total_breach = breaches.sum(dim=0)  # (K,)

        valid = total_breach < 0.5
        if bool(valid.any().item()):
            # First zero-breach anchor wins.
            k_pick = int(valid.nonzero(as_tuple=False)[0].item())
        else:
            k_pick = int(total_breach.argmin().item())
        ax = ax_list[k_pick]
        ay = ay_list[k_pick]

    ax2 = min(W, ax + ow)
    ay2 = min(H, ay + oh)
    rx2 = ax2 - ax
    ry2 = ay2 - ay
    if rx2 <= 0 or ry2 <= 0:
        return img, occupied

    out = img.clone()
    a = overlay_alpha[:, :ry2, :rx2]
    rgb = overlay_rgb[:, :ry2, :rx2]
    out[:, ay:ay2, ax:ax2] = out[:, ay:ay2, ax:ax2] * (1.0 - a) + rgb * a
    # Update / start the alpha-occupancy accumulator so a subsequent
    # overlay's anchor search can subtract the area we just took
    # from each polygon's remaining headroom.
    if occupied is None:
        occupied_new = torch.zeros((H, W), device=img.device, dtype=img.dtype)
    else:
        occupied_new = occupied
    region = occupied_new[ay:ay2, ax:ax2]
    occupied_new[ay:ay2, ax:ax2] = torch.maximum(region, a[0])
    return out.clamp(0.0, 1.0), occupied_new


# ─── Object-overlay augmentation ─────────────────────────────────
# User uploads an image + a text label; SAM3 segments the object,
# we cut the foreground onto a transparent PNG and stash it under
# <project>/augment_overlays/<uuid>.png. The augment preview
# endpoint then composites this overlay onto a random dataset
# image, with a constraint that the overlay's silhouette never
# covers more than 50% of any existing detection polygon.


def _overlays_dir(project_id: str) -> "Path":
    return project_dir(project_id) / "augment_overlays"


def _backgrounds_dir(project_id: str) -> "Path":
    return project_dir(project_id) / "augment_backgrounds"


@app.post(
    "/api/v2/projects/{project_id}/augment/background/upload",
    dependencies=[Depends(require_project_owner)],
)
async def v2_augment_background_upload(
    project_id: str,
    image: UploadFile = File(...),
):
    """Persist an uploaded background image under
    augment_backgrounds/<uuid>.jpg. No segmentation — backgrounds
    are used whole, just resized to the target canvas at preview
    time."""
    if not project_dir(project_id).exists():
        raise HTTPException(404, "project not found")
    raw = await image.read()
    if not raw:
        raise HTTPException(400, "empty image upload")
    # NSFW gate — backgrounds get composited into the augmentations
    # pipeline, so they need the same content-safety screen the
    # main /imports/raw endpoint applies.
    _enforce_nsfw_or_451(raw, label="v2-augment-background-upload")
    try:
        src = PILImage.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"could not decode image: {e}")
    # Cap stored size at long edge 1280 so we don't ship a 24 MP
    # raw upload to disk every time. The kernel re-resizes to the
    # working canvas anyway.
    W, H = src.size
    longest = max(W, H)
    if longest > 1280:
        scale = 1280.0 / longest
        src = src.resize((int(W * scale), int(H * scale)), PILImage.LANCZOS)
    bg_id = _uuid.uuid4().hex
    out_dir = _backgrounds_dir(project_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{bg_id}.jpg"
    src.save(out_path, format="JPEG", quality=85, optimize=False, progressive=False)
    return {"background_id": bg_id, "width": src.size[0], "height": src.size[1]}


@app.get(
    "/api/v2/projects/{project_id}/augment/backgrounds/{background_id}",
    dependencies=[Depends(require_project_read_access)],
)
async def v2_augment_background_get(project_id: str, background_id: str):
    safe = "".join(c for c in background_id if c.isalnum())
    if not safe:
        raise HTTPException(400, "bad background id")
    path = _backgrounds_dir(project_id) / f"{safe}.jpg"
    if not path.exists():
        raise HTTPException(404, "background not found")
    return await _serve_cached_image(project_id, "augment_backgrounds", f"{safe}.jpg", path)


@app.delete(
    "/api/v2/projects/{project_id}/augment/backgrounds/{background_id}",
    dependencies=[Depends(require_project_owner)],
)
async def v2_augment_background_delete(project_id: str, background_id: str):
    safe = "".join(c for c in background_id if c.isalnum())
    if not safe:
        raise HTTPException(400, "bad background id")
    path = _backgrounds_dir(project_id) / f"{safe}.jpg"
    if path.exists():
        try:
            path.unlink()
        except Exception as e:
            print(f"[background-delete] {e}")
    return {"ok": True}


# GPU-resident LRU cache of resized background tensors. Keyed by
# (path, target_H, target_W, device-str). Bounded so we don't leak
# VRAM if a user uploads many backgrounds.
from collections import OrderedDict as _BG_OD

_BG_TENSOR_CACHE: "_BG_OD[tuple[str, int, int, str], torch.Tensor]" = _BG_OD()
_BG_TENSOR_CACHE_MAX = 16


def _load_background_resized_gpu(
    path: "Path", H: int, W: int, device: "torch.device",
) -> "torch.Tensor | None":
    """Returns a (3, H, W) tensor of the background image on the
    target device, cached for repeat calls. Image decoded once on
    CPU; resize runs on GPU via F.interpolate so a full bake costs
    a few ms after the cache warms up."""
    key = (str(path), int(H), int(W), str(device))
    hit = _BG_TENSOR_CACHE.get(key)
    if hit is not None:
        _BG_TENSOR_CACHE.move_to_end(key)
        return hit
    try:
        with PILImage.open(path) as bg:
            bg = bg.convert("RGB")
    except Exception as e:
        print(f"[bg-cache] could not open {path}: {e}")
        return None
    import numpy as _np
    arr = _np.asarray(bg, dtype=_np.float32) / 255.0  # HWC
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous().to(device)  # (3, h0, w0)
    import torch.nn.functional as TF
    t = TF.interpolate(t.unsqueeze(0), size=(int(H), int(W)), mode="bilinear", align_corners=False).squeeze(0)
    _BG_TENSOR_CACHE[key] = t
    if len(_BG_TENSOR_CACHE) > _BG_TENSOR_CACHE_MAX:
        _BG_TENSOR_CACHE.popitem(last=False)
    return t


def _aug_apply_background_randomisation(
    img: "torch.Tensor",
    detections: list[dict],
    background_paths: list["Path"],
    sx: float,
    sy: float,
    seed: int,
) -> "torch.Tensor":
    """Replace pixels OUTSIDE the union of detection polygons with
    a randomly-chosen background image (picked by `seed`). Pixels
    inside polygons stay as-is so the user's segmented objects sit
    untouched on top of the new scene.

    GPU path: background is decoded once on CPU, transferred to
    GPU, resized via F.interpolate, then cached so subsequent
    previews skip the disk + transfer cost. The composite itself
    is a single (3, H, W) tensor blend on-device."""
    if not background_paths:
        return img
    _C, H, W = img.shape
    poly_mask = _rasterise_polygons_to_mask(detections, H, W, sx, sy, img.device)
    if float(poly_mask.sum().item()) < 1.0:
        return img
    g = torch.Generator(device="cpu").manual_seed(seed ^ 0xBA66)
    idx = int(torch.randint(0, len(background_paths), (1,), generator=g).item())
    bg_t = _load_background_resized_gpu(background_paths[idx], H, W, img.device)
    if bg_t is None:
        return img
    m = poly_mask.unsqueeze(0)  # (1, H, W)
    return (img * m + bg_t * (1.0 - m)).clamp(0.0, 1.0)


@app.post(
    "/api/v2/projects/{project_id}/augment/object_overlay/segment",
    dependencies=[Depends(require_project_owner)],
)
async def v2_augment_object_overlay_segment(
    project_id: str,
    image: UploadFile = File(...),
    label: str = Form(...),
):
    """Run SAM3 on the uploaded image with the supplied label, cut
    the highest-scoring mask out as a transparent PNG, store under
    augment_overlays/<uuid>.png and return its handle."""
    if not project_dir(project_id).exists():
        raise HTTPException(404, "project not found")
    charlie = state.get("charlie")
    if charlie is None:
        raise HTTPException(503, "SAM3 not loaded")
    clean_label = (label or "").strip()
    if not clean_label:
        raise HTTPException(400, "label must be non-empty")

    raw = await image.read()
    if not raw:
        raise HTTPException(400, "empty image upload")
    # NSFW gate for object overlay uploads (matches /imports/raw +
    # /augment/background/upload). The user pastes a cut-out from
    # an arbitrary photo into the dataset; same content-safety
    # screen applies.
    _enforce_nsfw_or_451(raw, label="v2-augment-object-overlay-segment")
    try:
        src = PILImage.open(io.BytesIO(raw)).convert("RGBA")
    except Exception as e:
        raise HTTPException(400, f"could not decode image: {e}")

    loop = asyncio.get_running_loop()

    def _run() -> tuple[bytes, int, int]:
        # SAM3 expects RGB.
        rgb = src.convert("RGB")
        try:
            detections, _timings = charlie.segment_labels(rgb, [clean_label], include_crops=False)
        except Exception as exc:
            raise RuntimeError(f"SAM3 failed: {exc}") from exc
        if not detections:
            raise RuntimeError("no detections")
        # Pick the highest-scoring detection that has a polygon.
        scored = [
            d for d in detections
            if isinstance(d.get("mask"), dict) and (d["mask"].get("polygons") or [])
        ]
        if not scored:
            raise RuntimeError("no segmentation found")
        scored.sort(key=lambda d: float(d.get("score") or 0.0), reverse=True)
        best = scored[0]
        polys = best["mask"]["polygons"]
        # Rasterise polygons → alpha. Tight crop to the bbox so we
        # don't store the original's whitespace.
        W, H = rgb.size
        from PIL import ImageDraw as _ImageDraw
        alpha = PILImage.new("L", (W, H), 0)
        drawer = _ImageDraw.Draw(alpha)
        for poly in polys:
            if not poly or len(poly) < 3:
                continue
            drawer.polygon([(float(x), float(y)) for (x, y) in poly], fill=255)
        # Apply alpha to the original RGBA, then crop to the alpha
        # bbox (saves bytes + keeps the centre near the image
        # centre for downstream placement).
        rgba = src.copy()
        rgba.putalpha(alpha)
        bbox = alpha.getbbox()
        if bbox:
            rgba = rgba.crop(bbox)
        else:
            raise RuntimeError("empty mask")
        buf = io.BytesIO()
        rgba.save(buf, format="PNG", optimize=False)
        return buf.getvalue(), rgba.size[0], rgba.size[1]

    try:
        async with state["gpu_lock"]:
            data, ow, oh = await loop.run_in_executor(None, _run)
    except Exception as exc:
        raise HTTPException(500, f"object overlay segmentation failed: {exc}")

    # Persist. UUID4 hex used as the filename → also the overlay_id
    # the FE references in subsequent preview requests.
    overlay_id = _uuid.uuid4().hex
    out_dir = _overlays_dir(project_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{overlay_id}.png"
    out_path.write_bytes(data)
    return {
        "overlay_id": overlay_id,
        "width": ow,
        "height": oh,
        "label": clean_label,
    }


# ─── Batch augmentation generation ──────────────────────────────
# Apply the augmentation config to every dataset image, producing
# N copies per image. Run as a background job through the
# existing JobManager so the FE can poll progress + cancel like
# any other batch task. Outputs live under
# <project>/augmentations/<import_id>/<idx>.jpg and are SEPARATE
# from the dataset itself — they're surfaced through the per-tile
# Augmentations viewer but don't enter the labelling pipeline.


def _augmentations_dir(project_id: str, import_id: str = "") -> "Path":
    root = project_dir(project_id) / "augmentations"
    return root / import_id if import_id else root


def _maybe_auto_augment_after_edit(project_id: str, import_id: str | None = None) -> None:
    """Schedule a fresh augment_generate for `project_id` IF:
      • the project has a persisted augmentationConfig (i.e. the user
        has clicked Update on the Augmentations tab at least once)
      • perImageMode isn't "off"
      • no augment_generate job is already running/queued for the
        project (the imports PUT endpoint fires per box drag, so
        this guard stops us queuing dozens of overlapping jobs while
        the user is mid-edit)

    When `import_id` is supplied the regen is scoped to that one
    image — wiping + re-running only its augmentation dir — so a
    single bbox tweak doesn't fan out into a re-augment of the
    whole project. Callers that genuinely want a project-wide
    regen pass `import_id=None`.
    """
    try:
        # copy=False — read-only check of augmentationConfig + owner.
        # Called from PUT /imports/{id}, hot on every box drag, used
        # to pay a 300-500ms deepcopy on big projects.
        manifest = load_manifest(project_id, copy=False) or {}
        cfg = manifest.get("augmentationConfig") or {}
        per_image_mode = str(cfg.get("perImageMode") or "")
        config_dict = cfg.get("config") or {}
        if not config_dict or not per_image_mode:
            return
        if per_image_mode in ("off", "0"):
            return
        # In-flight guard — don't pile up generate jobs while one's
        # already running. Once the active one drains, the next edit
        # will queue a fresh one.
        for j in state["jobs"].jobs.values():
            if j.project != project_id:
                continue
            if j.kind == "augment_generate" and j.status in ("running", "queued"):
                return
        # Wipe the affected images' augmentation dirs so the new run
        # starts from a clean slate. Scoped to a single import when
        # the caller supplied one — leaving every other image's
        # augmentations untouched.
        try:
            aug_root = _augmentations_dir(project_id)
            if import_id:
                target_dir = aug_root / import_id
                if target_dir.exists():
                    import shutil as _shutil
                    _shutil.rmtree(target_dir)
            elif aug_root.exists():
                import shutil as _shutil
                _shutil.rmtree(aug_root)
        except Exception as e:
            print(f"[auto-augment-after-edit] wipe failed for {project_id}: {e}")
        try:
            # copy=False + cache_by_ref — we mutate-then-save the
            # same dict we just read, no concurrent writer in the
            # hot path.
            m_aug = load_manifest(project_id, copy=False) or {}
            mutated = False
            for entry in (m_aug.get("imports") or []):
                if import_id and entry.get("id") != import_id:
                    continue
                if entry.get("n_augmentations"):
                    entry["n_augmentations"] = 0
                    mutated = True
            if mutated:
                save_manifest(project_id, m_aug, cache_by_ref=True)
        except Exception as e:
            print(f"[auto-augment-after-edit] reset counters failed for {project_id}: {e}")
        try:
            # Attribute the regen to the project owner so the terminal
            # feed shows the user's handle, not @system. Fire-and-
            # forget hook fired from /imports PUT — we don't have a
            # session here, so the manifest's owner is the best
            # available signal.
            owner = (
                manifest.get("owner")
                or manifest.get("createdBy")
                or "system"
            )
            params: dict = {"perImageMode": per_image_mode, "config": config_dict}
            n_imgs = len(manifest.get("imports") or [])
            if import_id:
                params["targetImportIds"] = [import_id]
                n_imgs = 1
            state["jobs"].schedule(
                "augment_generate",
                project_id,
                params,
                owner,
                n_images=n_imgs,
            )
            state["jobs"].start_worker()
            print(f"[auto-augment-after-edit] scheduled for {project_id}{' (id=' + import_id + ')' if import_id else ''}")
        except Exception as e:
            print(f"[auto-augment-after-edit] schedule failed for {project_id}: {e}")
    except Exception as e:
        print(f"[auto-augment-after-edit] {project_id} failed: {e}")


# ─── Whole-image embeddings (dataset-stats feature) ─────────────
# Each imported image gets a DINOv2 whole-image embedding stored as
# a raw float32 .bin sidecar under <project>/image_embeddings/. The
# /dataset-stats endpoint folds them into a PCA projection + near-
# duplicate flags + a 0-100 health score. Embeddings are computed
# fire-and-forget on /imports/raw, and the stats endpoint backfills
# anything missing on demand.

def _image_embeddings_dir(project_id: str) -> "Path":
    return project_dir(project_id) / "image_embeddings"


def _image_embedding_path(project_id: str, import_id: str) -> "Path":
    safe = "".join(c for c in import_id if c.isalnum())
    return _image_embeddings_dir(project_id) / f"{safe}.bin"


def _compute_and_store_image_embedding(project_id: str, import_id: str, raw_bytes: bytes) -> bool:
    """Embed one imported image and write the resulting float32 vector
    to its sidecar. Returns True when bytes landed on disk. No-op
    when DINOv2 isn't loaded yet — the stats endpoint retries on
    demand if the sidecar is missing."""
    try:
        import v2_dinov2 as _v2d
        if not _v2d.is_loaded():
            return False
        from PIL import Image as _PIL
        import io as _io
        pil = _PIL.open(_io.BytesIO(raw_bytes)).convert("RGB")
        vec = _v2d.encode_image(pil)  # (D,) L2-normalised float32
        if vec is None or vec.size == 0:
            return False
        out_dir = _image_embeddings_dir(project_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = _image_embedding_path(project_id, import_id)
        # Write atomically — write to temp + rename so a half-baked
        # file never gets read by a concurrent stats request.
        tmp = out_path.with_suffix(".bin.tmp")
        tmp.write_bytes(vec.astype("float32", copy=False).tobytes())
        tmp.replace(out_path)
        return True
    except Exception as e:
        print(f"[v2-img-embed] {import_id} failed: {e}")
        return False


def _load_image_embedding(project_id: str, import_id: str) -> "np.ndarray | None":
    """Read an import's embedding sidecar back into a (D,) float32
    numpy array. Returns None when the file's missing or unreadable —
    the caller treats missing == needs backfill."""
    p = _image_embedding_path(project_id, import_id)
    if not p.exists():
        return None
    try:
        import numpy as _np
        raw = p.read_bytes()
        if not raw or len(raw) % 4 != 0:
            return None
        return _np.frombuffer(raw, dtype=_np.float32)
    except Exception:
        return None


def _backfill_missing_embeddings(project_id: str, imports: list[dict], limit: int = 32) -> int:
    """Compute embeddings for up to `limit` imports that don't yet
    have a sidecar. Called from the stats endpoint when the user
    opens the card on a project whose embeddings predate this
    feature. Returns the count actually written."""
    proj = project_dir(project_id)
    n = 0
    for imp in imports:
        if n >= limit:
            break
        iid = imp.get("id")
        fn = imp.get("filename")
        if not iid or not fn:
            continue
        if _load_image_embedding(project_id, iid) is not None:
            continue
        src = proj / "images" / fn
        if not src.exists():
            continue
        try:
            with open(src, "rb") as f:
                raw = f.read()
            if _compute_and_store_image_embedding(project_id, iid, raw):
                n += 1
        except Exception:
            continue
    return n


def _pca_project_2d(vectors: "np.ndarray") -> "list[list[float]]":
    """PCA → first two principal components for the 2-D variation
    plot. Vectors are L2-normalised on disk so we centre and SVD
    directly. Returns a list of [x, y] floats matching the input
    row order. Falls back to zeros on degenerate inputs (< 2 rows,
    NaNs, etc.)."""
    import numpy as _np
    if vectors is None or len(vectors) < 2:
        return [[0.0, 0.0]] * (0 if vectors is None else len(vectors))
    try:
        X = vectors.astype(_np.float32, copy=False)
        X = X - X.mean(axis=0, keepdims=True)
        # Thin SVD — full_matrices=False so we get just the
        # min(N, D) singular values without paying for the full
        # right-singular-vector matrix.
        U, S, Vt = _np.linalg.svd(X, full_matrices=False)
        # Project onto first 2 PCs and scale to a [-1, 1]-ish range
        # so the FE can render without computing extents itself.
        pc = U[:, :2] * S[:2]
        # Normalise per-axis so the scatter fills the plot regardless
        # of feature variance. Guard against zero-variance axes.
        max_abs = _np.maximum(_np.abs(pc).max(axis=0), 1e-6)
        norm = pc / max_abs
        return [[float(x), float(y)] for x, y in norm]
    except Exception as e:
        print(f"[v2-stats] PCA failed: {e}")
        return [[0.0, 0.0]] * len(vectors)


def _near_duplicate_pairs(vectors: "np.ndarray", ids: list[str], threshold: float = 0.95) -> list[tuple[str, str, float]]:
    """Pairs of import IDs whose cosine similarity exceeds `threshold`.
    Vectors are already L2-normalised so cosine == dot product.
    Limits the count to avoid blowing up on extreme duplicate clusters
    (a 100-image set with all duplicates would otherwise return ~5000
    pairs)."""
    import numpy as _np
    pairs: list[tuple[str, str, float]] = []
    if vectors is None or len(vectors) < 2:
        return pairs
    try:
        # Full pairwise cosine matrix is fine up to a few thousand
        # images (1000² × 4 bytes = 4 MB).
        sims = vectors @ vectors.T
        n = len(vectors)
        # Walk the upper triangle.
        for i in range(n):
            row = sims[i]
            for j in range(i + 1, n):
                s = float(row[j])
                if s >= threshold:
                    pairs.append((ids[i], ids[j], s))
                    if len(pairs) >= 200:
                        return pairs
        return pairs
    except Exception as e:
        print(f"[v2-stats] near-dup detection failed: {e}")
        return pairs


def _compute_dataset_stats_v2(project_id: str, lite: bool = False) -> dict:
    """Build the dataset-stats payload for /api/v2/projects/{id}/dataset-stats.

    Two modes:
      - lite=False (default): counts + label distribution + per-image
        embedding load + PCA + near-duplicate detection + 4-factor health
        blend. Used by the dataset stats card's full expand state.
      - lite=True: counts + label distribution + per-image augmentation
        dir scan, but skips the embedding load, PCA solve, near-duplicate
        pass and uniqueness sub-score. ~10-100× faster on big projects.
        The FE fetches this first to paint the summary row, then upgrades
        to the full payload in the background.

    Health score is a 0-100 blend of four sub-scores in non-lite:
      - Label balance  : 1 - normalised stddev of per-label counts
      - Coverage       : % imports with at least one detection
      - Confidence     : 1 - (unsure detections / total detections)
      - Uniqueness     : 1 - (images flagged near-dup / total imports)
    Lite drops the uniqueness sub-score (which needs embeddings) and
    re-weights the remaining three to 1/3 each so the badge still has
    a meaningful number while the full payload is loading.
    """
    import numpy as _np
    # Read-only pass over the manifest; skip the deepcopy. The lite
    # branch (which is what runs on every project open) walks imports
    # to tally counts, so this is the hottest read path for big
    # projects.
    manifest = load_manifest(project_id, False) or {}
    imports = manifest.get("imports") or []
    # Label distribution across the dataset's CURRENT annotation state
    # (editedBoxes wins when present, otherwise kept detections).
    label_counts: dict[str, int] = {}
    n_detections = 0
    n_unsure = 0
    n_with_dets = 0
    aug_count = 0
    for imp in imports:
        # An explicit empty editedBoxes (user cleared all boxes, with
        # editedBoxesSet) means zero boxes — don't fall back to the auto
        # detections, or the counts + label distribution resurrect the
        # deleted boxes. An empty editedBoxes WITHOUT the flag is a legacy
        # upload seed and still falls back.
        edited = imp.get("editedBoxes")
        if isinstance(edited, list) and (edited or imp.get("editedBoxesSet")):
            dets = edited
            used_edited = True
        else:
            dets = [d for d in (imp.get("detections") or []) if not d.get("rejected")]
            used_edited = False
        if dets:
            n_with_dets += 1
        for d in dets:
            if not isinstance(d, dict):
                continue
            # Cover both naming conventions + both detection sources.
            # editedBoxes carry `label`; auto detections carry
            # `pred_label` (or `gd_label` while the resolver hasn't
            # committed). The FE-converted camelCase variants land
            # here too if a future codepath persists them by
            # mistake. Take the first non-empty string in priority
            # order so we don't miss a label that's only set on one
            # of the fallbacks.
            lab = (
                str(d.get("label") or "").strip()
                or str(d.get("predLabel") or "").strip()
                or str(d.get("pred_label") or "").strip()
                or str(d.get("gdLabel") or "").strip()
                or str(d.get("gd_label") or "").strip()
            )
            if lab:
                label_counts[lab] = label_counts.get(lab, 0) + 1
            n_detections += 1
            if not used_edited and d.get("ambiguous"):
                n_unsure += 1
        aug_count += int(imp.get("n_augmentations") or 0)

    embed_ids: list[str] = []
    embed_vecs: list[_np.ndarray] = []
    if not lite:
        # Backfill missing embeddings (cheap if everything's already on
        # disk, ~30 ms/image when DINOv2 needs to fill a gap). Skipped
        # in lite mode — the variation plot + uniqueness signal don't
        # apply there.
        _backfill_missing_embeddings(project_id, imports, limit=32)

        for imp in imports:
            iid = imp.get("id")
            if not iid:
                continue
            v = _load_image_embedding(project_id, iid)
            if v is None or v.size == 0:
                continue
            embed_ids.append(iid)
            embed_vecs.append(v)

    points: list[dict] = []
    aug_points: list[dict] = []
    near_dup_ids: set[str] = set()
    if embed_vecs:
        mat = _np.stack(embed_vecs, axis=0)
        proj = _pca_project_2d(mat)
        coord_by_id = dict(zip(embed_ids, proj))
        # Deterministic RNG keyed by project id so augmentation
        # scatter is stable across re-fetches (no jitter on every
        # refresh) but varies between projects.
        rng = _np.random.default_rng(abs(hash(project_id)) & 0xFFFFFFFF)
        for imp in imports:
            iid = imp.get("id")
            if not iid or iid not in coord_by_id:
                continue
            edited = imp.get("editedBoxes") if isinstance(imp.get("editedBoxes"), list) else None
            kept = edited if edited else [d for d in (imp.get("detections") or []) if isinstance(d, dict) and not d.get("rejected")]
            primary_lab = None
            for d in kept:
                lab = (
                    str(d.get("label") or "").strip()
                    or str(d.get("predLabel") or "").strip()
                    or str(d.get("pred_label") or "").strip()
                    or str(d.get("gdLabel") or "").strip()
                    or str(d.get("gd_label") or "").strip()
                )
                if lab:
                    primary_lab = lab
                    break
            xy = coord_by_id[iid]
            points.append({
                "id": iid,
                "filename": imp.get("filename"),
                "x": xy[0],
                "y": xy[1],
                "label": primary_lab,
                "n_detections": len(kept),
            })
            # Augmentations: each generated copy gets a point at the
            # source image's PCA coord with a small Gaussian offset
            # so they cluster softly around the parent. Cheaper than
            # embedding every augmentation JPEG (would scale poorly)
            # and visually accurate — augmentations sit near their
            # source in feature space by construction.
            #
            # Source of truth is the augmentations directory on disk
            # rather than the manifest's n_augmentations counter,
            # because the counter can lag (job partially completed,
            # crash mid-run, or a manual file drop). Scan the dir
            # and emit one point per JPEG/PNG found.
            aug_files: list[str] = []
            try:
                aug_subdir = _augmentations_dir(project_id, iid)
                if aug_subdir.exists():
                    for f in sorted(aug_subdir.iterdir()):
                        if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                            aug_files.append(f.name)
            except Exception:
                aug_files = []
            if aug_files:
                jitter = rng.normal(0.0, 0.025, size=(len(aug_files), 2))
                for k, fn in enumerate(aug_files):
                    aug_points.append({
                        "id": f"{iid}__aug_{fn}",
                        "source_id": iid,
                        "filename": fn,
                        "x": float(xy[0]) + float(jitter[k, 0]),
                        "y": float(xy[1]) + float(jitter[k, 1]),
                        "label": primary_lab,
                    })
        dup_pairs = _near_duplicate_pairs(mat, embed_ids, threshold=0.95)
        # Filter out pairs that the user has explicitly ignored via the
        # dedupe review modal — once they confirm "these aren't
        # duplicates", we stop counting them and stop reporting them
        # to the stats card.
        ignored = set(manifest.get("ignored_near_dups") or [])
        for a, b, _ in dup_pairs:
            if a in ignored or b in ignored:
                continue
            near_dup_ids.add(a)
            near_dup_ids.add(b)

    total_imports = len(imports)
    # Health sub-scores — each in [0, 1]. Coverage handles the empty-
    # dataset case (zero divisions) by returning a neutral 1.0 so the
    # score isn't permanently anchored at 0.
    if total_imports == 0:
        score_coverage = 1.0
        score_balance = 1.0
        score_confidence = 1.0
        score_uniqueness = 1.0
    else:
        score_coverage = n_with_dets / total_imports
        if label_counts:
            counts = _np.array(list(label_counts.values()), dtype=_np.float32)
            if counts.sum() > 0 and len(counts) > 1:
                # Coefficient of variation (stddev / mean) clamped to [0,1].
                cv = float(counts.std() / max(counts.mean(), 1e-6))
                score_balance = max(0.0, 1.0 - min(cv, 1.0))
            else:
                # One label or no labels — trivially balanced.
                score_balance = 1.0
        else:
            score_balance = 0.5  # no labels yet, neutral
        score_confidence = 1.0 - (n_unsure / max(n_detections, 1))
        score_uniqueness = 1.0 - (len(near_dup_ids) / max(total_imports, 1))
    # Weighted blend → 0..100. Uniqueness carries the heaviest
    # contribution (40 %) because a dataset packed with near-
    # duplicate images is the single biggest health-risk: the model
    # learns the duplicate's particular framing rather than the
    # underlying concept. Coverage / balance / confidence each carry
    # 20 % so a well-labelled, balanced dataset still rates well
    # without near-perfect uniqueness. In lite mode the uniqueness
    # signal isn't available (no embeddings) so we redistribute its
    # weight across the remaining three sub-scores — the badge is
    # still meaningful while the FE waits for the full payload.
    if lite:
        health = int(round(
            score_balance * 33.34
            + score_coverage * 33.33
            + score_confidence * 33.33
        ))
    else:
        health = int(round(
            score_uniqueness * 40
            + score_balance * 20
            + score_coverage * 20
            + score_confidence * 20
        ))
    health = max(0, min(100, health))

    return {
        "counts": {
            "imports": total_imports,
            "with_detections": n_with_dets,
            "detections": n_detections,
            "unsure_detections": n_unsure,
            "augmentations": aug_count,
            "near_duplicates": len(near_dup_ids),
            "embeddings_ready": len(embed_ids),
        },
        "labels": [
            {"label": k, "count": v}
            for k, v in sorted(label_counts.items(), key=lambda kv: -kv[1])
        ],
        "health": {
            "score": health,
            "factors": {
                "balance": round(score_balance, 3),
                "coverage": round(score_coverage, 3),
                "confidence": round(score_confidence, 3),
                "uniqueness": round(score_uniqueness, 3),
            },
        },
        "variation": {
            "points": points,
            "augmentations": aug_points,
            "near_duplicate_ids": sorted(near_dup_ids),
        },
    }


@app.get(
    "/api/v2/projects/{project_id}/dataset-stats",
    response_class=ORJSONResponse,
    dependencies=[Depends(require_project_read_access)],
)
async def v2_dataset_stats(project_id: str, lite: bool = False):
    """Aggregate dataset metrics + 2-D variation plot + near-duplicate
    flags + 0-100 health score. Used by the project page's stats
    card.

    `lite=true` skips the embedding load + PCA solve + near-duplicate
    pass. Counts, label distribution and a 3-factor health score
    return in milliseconds; the FE fetches the full payload after the
    summary row has painted. The full call's embeddings are read
    from disk sidecars, with backfill for any missing — sub-second for
    typical dataset sizes but multi-second on a 900-image first-touch.

    Wrapped in asyncio.to_thread so the FastAPI event loop stays
    responsive while either mode's sync work runs."""
    if not project_dir(project_id).exists():
        raise HTTPException(404, "project not found")
    disk_mtime = _manifest_disk_mtime(project_id)
    name = "dataset-stats-lite" if lite else "dataset-stats"
    cached, fresh = _payload_cache_get_swr(project_id, name, disk_mtime)
    if cached is not None and fresh:
        return cached
    if cached is not None:
        asyncio.create_task(_payload_revalidate(
            project_id, name,
            lambda: _persist_dataset_stats(project_id, lite),
        ))
        return cached
    # No in-memory cache. Try the on-disk sidecar so cold starts /
    # process restarts don't pay the recompute cost. Allow-stale: if
    # the sidecar exists at all, serve it immediately and kick the
    # recompute in the background. The user sees the previous batch's
    # numbers while the new ones bake. Without this the first request
    # after every manifest write paid the full 2 s compute and the FE
    # had to wait through the spinner.
    disk_payload, fresh = await asyncio.to_thread(
        _read_dataset_stats_sidecar, project_id, lite, disk_mtime,
    )
    if disk_payload is not None:
        _payload_cache_put(project_id, name, disk_mtime if fresh else 0.0, disk_payload)
        if not fresh:
            asyncio.create_task(_payload_revalidate(
                project_id, name,
                lambda: _persist_dataset_stats(project_id, lite),
            ))
        return disk_payload
    payload = await asyncio.to_thread(_persist_dataset_stats, project_id, lite)
    _payload_cache_put(project_id, name, disk_mtime, payload)
    return payload


def _stats_sidecar_path(project_id: str, lite: bool) -> Path:
    """Where the precomputed dataset-stats payload lives. Separate
    files for the lite + full variants so they invalidate independently
    of each other."""
    name = "dataset_stats_lite.json" if lite else "dataset_stats.json"
    return project_dir(project_id) / name


def _read_dataset_stats_sidecar(
    project_id: str, lite: bool, manifest_mtime: float
) -> tuple[dict | None, bool]:
    """Return (payload, fresh). The payload comes back regardless of
    freshness so the caller can serve a stale snapshot immediately
    and kick a background recompute — better than making the user
    wait several seconds for the first request after a manifest write."""
    p = _stats_sidecar_path(project_id, lite)
    if not p.exists():
        return None, False
    try:
        sidecar_mtime = p.stat().st_mtime
        data = _json_loads(p.read_bytes())
        if isinstance(data, dict):
            return data, sidecar_mtime >= manifest_mtime
    except Exception as e:
        print(f"[stats-sidecar] read failed for {project_id}: {e}")
    return None, False


def _persist_dataset_stats(project_id: str, lite: bool) -> dict:
    """Compute the stats payload + write it to disk so the next process
    start / cold cache hit serves instantly from the sidecar. Errors on
    the write side don't bubble — the in-memory cache still has the
    payload, and the next request will retry the disk write."""
    payload = _compute_dataset_stats_v2(project_id, lite)
    try:
        p = _stats_sidecar_path(project_id, lite)
        p.parent.mkdir(parents=True, exist_ok=True)
        # Write atomically — temp file + rename so a crash mid-write
        # can't leave half a stats file on disk that the next read
        # would then parse-fail on.
        tmp = p.with_suffix(".json.tmp")
        if _orjson is not None:
            tmp.write_bytes(_orjson.dumps(payload))
        else:
            tmp.write_text(json.dumps(payload))
        tmp.replace(p)
    except Exception as e:
        print(f"[stats-sidecar] write failed for {project_id}: {e}")
    return payload


def _resolve_n_copies(per_image_mode: str, g: "torch.Generator") -> int:
    """Map the FE perImage state (off / 1 / 2 / 3 / random) to a
    concrete copy count for one image."""
    if per_image_mode in ("off", "0"):
        return 0
    if per_image_mode == "random":
        return int(torch.randint(1, 4, (1,), generator=g).item())
    try:
        return max(0, min(8, int(per_image_mode)))
    except (TypeError, ValueError):
        return 1


def _aug_should_apply(frequency: str, copy_idx: int, total: int, g: "torch.Generator") -> bool:
    """`all` → every copy gets the aug; `random` → roughly half.
    Driven through the generator so reruns are reproducible."""
    if total <= 0:
        return False
    if frequency == "all":
        return True
    # random — flip a coin per copy
    return bool(torch.rand(1, generator=g).item() > 0.5)


async def _run_augment_generate_job(job, emit, cancel_event):
    """Apply the augmentation config to every dataset image,
    saving each generated copy as a JPEG under augmentations/."""
    proj = project_dir(job.project)
    if not proj.exists():
        raise RuntimeError("project not found")
    manifest = load_manifest(job.project) or {}
    imports = manifest.get("imports") or []
    if not imports:
        raise RuntimeError("no dataset images to augment")
    cfg = (job.params or {}).get("config") or {}
    per_image_mode = str((job.params or {}).get("perImageMode") or "1")
    if per_image_mode in ("off", "0"):
        raise RuntimeError("perImage is off — nothing to generate")

    # Optional scope filter. The auto-augment hooks pass a list of
    # import_ids so a regen triggered by labelling 1 image doesn't
    # re-augment the other 940. Empty list means "nothing to do" so
    # we exit cleanly rather than falling through to all imports
    # (which is what the user explicitly didn't want here).
    raw_targets = (job.params or {}).get("targetImportIds")
    if raw_targets is not None:
        target_set = {str(t) for t in raw_targets if t}
        if not target_set:
            await emit("progress", {"index": 0, "total": 0, "image": "", "phase": "augmenting"})
            print(f"[augment_generate] {job.project}: empty target list — nothing to augment")
            return
        imports = [imp for imp in imports if imp.get("id") in target_set]
        if not imports:
            await emit("progress", {"index": 0, "total": 0, "image": "", "phase": "augmenting"})
            print(f"[augment_generate] {job.project}: none of {len(target_set)} targets present in manifest")
            return

    aug_root = _augmentations_dir(job.project)
    aug_root.mkdir(parents=True, exist_ok=True)

    cam = cfg.get("camera") or {}
    dom = cfg.get("domain") or {}
    occ = cfg.get("occlusion") or {}
    dist = cfg.get("distortion") or {}

    # Snap each augmentation's enabled state + frequency + params.
    cam_enabled = bool(cam.get("enabled"))
    cam_freq = str(cam.get("frequency") or "all")
    cam_params = {
        "mb": float(cam.get("motionBlur") or 0),
        "ns": float(cam.get("noise") or 0),
        "cd": float(cam.get("colourDistortion") or 0),
        "ca": float(cam.get("chromaticAberration") or 0),
        "bd": float(cam.get("bitDepth") or 0),
        # Newer dials — same 0..10 scale. Older configs that
        # predate these keys default to 0 (identity).
        "ld": float(cam.get("lensDistortion") or 0),
        "px": float(cam.get("pixelation") or 0),
        "lr": float(cam.get("lowResolution") or 0),
        "lg": float(cam.get("lensGlare") or 0),
    }
    occ_random_block = (occ.get("randomBlock") or {})
    occ_block_enabled = bool(occ.get("enabled")) and bool(occ_random_block.get("enabled"))
    occ_block_freq = str(occ_random_block.get("frequency") or "all")
    # Clamp to 40% — matches the slider cap; older saved configs that
    # carried higher values get pulled down to the new ceiling.
    occ_block_size = max(0.0, min(40.0, float(occ_random_block.get("size") or 0)))

    dist_pw = (dist.get("perspectiveWarp") or {})
    dist_sr = (dist.get("scaleRotation") or {})
    dist_hs = (dist.get("hueShift") or {})
    dist_pw_enabled = bool(dist.get("enabled")) and bool(dist_pw.get("enabled"))
    dist_pw_freq = str(dist_pw.get("frequency") or "all")
    dist_pw_strength = float(dist_pw.get("strength") or 0)
    dist_sr_enabled = bool(dist.get("enabled")) and bool(dist_sr.get("enabled"))
    dist_sr_freq = str(dist_sr.get("frequency") or "all")
    # Scale range clamped to 0.7..1.3 — matches the slider bounds and
    # pulls older saved configs back into the new band before they
    # drive the augmentation kernel.
    dist_sr = {
        "smin": max(0.7, min(1.3, float(dist_sr.get("scaleMin") or 1.0))),
        "smax": max(0.7, min(1.3, float(dist_sr.get("scaleMax") or 1.0))),
        "rmin": float(dist_sr.get("rotMin") or 0.0),
        "rmax": float(dist_sr.get("rotMax") or 0.0),
    }
    dist_hs_enabled = bool(dist.get("enabled")) and bool(dist_hs.get("enabled"))
    dist_hs_freq = str(dist_hs.get("frequency") or "all")
    dist_hs_strength = float(dist_hs.get("strength") or 0)

    dom_bg = (dom.get("backgrounds") or {})
    dom_bg_enabled = bool(dom.get("enabled")) and bool(dom_bg.get("enabled"))
    dom_bg_freq = str(dom_bg.get("frequency") or "all")
    dom_bg_ids = [b.get("id") for b in (dom_bg.get("backgrounds") or []) if isinstance(b, dict) and b.get("id")]
    bg_paths: list[Path] = []
    for bid in dom_bg_ids:
        safe = "".join(c for c in str(bid) if c.isalnum())
        if safe:
            p = _backgrounds_dir(job.project) / f"{safe}.jpg"
            if p.exists():
                bg_paths.append(p)
    dom_lt = (dom.get("lighting") or {})
    dom_lt_enabled = bool(dom.get("enabled")) and bool(dom_lt.get("enabled"))
    dom_lt_freq = str(dom_lt.get("frequency") or "all")
    dom_lt_strength = float(dom_lt.get("strength") or 0)

    job.n_images = len(imports)
    device = _aug_device()
    # Estimated total augmentations the run will produce, used to
    # drive the FE progress bar by augmentations-saved rather than
    # images-processed. Per-image counts come from _resolve_n_copies:
    #   "1"/"2"/"3" → exact, so total is precise
    #   "random"    → 1-3 inclusive, so the average ~2 is a good
    #                 estimate (we cap the displayed index at the
    #                 total so a slight under-estimate doesn't push
    #                 the progress past 100 %).
    def _estimate_per_image() -> int:
        m = (per_image_mode or "").strip().lower()
        if m == "random":
            return 2
        try:
            return max(0, min(8, int(m)))
        except (TypeError, ValueError):
            return 1
    estimated_total_augmentations = len(imports) * _estimate_per_image()
    # Warm CUDA in the augment thread BEFORE the loop starts so the
    # first image doesn't pay the context-init cost. Also re-confirm
    # the device choice — if torch.cuda.is_available() lied at module-
    # load time (cuda lazily initialising), this is our chance to
    # catch it and at least surface the fact in the logs.
    def _warm_cuda_in_thread():
        try:
            if device.type == "cuda":
                torch.cuda.init()
                torch.cuda.synchronize()
            print(
                f"[augment_generate] runner starting on device={device.type}"
                f" cuda_available={torch.cuda.is_available()}"
                f" cuda_device_count={torch.cuda.device_count() if torch.cuda.is_available() else 0}"
            )
        except Exception as e:
            print(f"[augment_generate] CUDA warmup failed: {e}")
    try:
        loop_for_warm = asyncio.get_event_loop()
        await loop_for_warm.run_in_executor(_AUG_EXECUTOR, _warm_cuda_in_thread)
    except Exception:
        pass
    # Surface progress with index=0/total up-front so the FE's poll
    # sees the total immediately and shows the bar at 0/N rather
    # than stuck on "starting…" until the first image completes (on
    # a 941-image dataset that's a 1-2 s wait the user can see).
    # `total` is the ESTIMATED augmentation count, not the image
    # count — the user's progress card must always read in
    # augmentations, never falling back to images.
    job.progress = {
        "index": 0,
        "total": estimated_total_augmentations,
        "image": None,
        "phase": "augmenting",
    }

    # SYNCHRONOUS per-image body. Lifted out of the for-loop and
    # called via asyncio.to_thread so the event loop stays free
    # while torch + PIL + JPEG-save are running. Without this the
    # runner monopolised the loop for the full duration of every
    # image (~0.5-2 s of dense compute) — every other handler
    # (image fetches, /api/projects poll, click-to-detect) was
    # queued behind it. The user reported "the server completely
    # blocks while augmenting".
    def _augment_one_image(i: int, imp: dict) -> int:
        import_id = imp.get("id")
        filename = imp.get("filename")
        if not import_id or not filename:
            return 0
        src_path = proj / "images" / filename
        if not src_path.exists():
            print(f"[augment_generate] src missing for {filename} ({import_id}) — skipping")
            return 0
        out_dir = aug_root / import_id
        if out_dir.exists():
            for f in out_dir.iterdir():
                try:
                    f.unlink(missing_ok=True)
                except Exception:
                    pass
        out_dir.mkdir(parents=True, exist_ok=True)
        g_img = torch.Generator(device="cpu").manual_seed(
            (hash((job.id, import_id)) & 0xFFFFFFFF) | 1,
        )
        n_copies = _resolve_n_copies(per_image_mode, g_img)
        if n_copies <= 0:
            return 0
        try:
            with PILImage.open(src_path) as im:
                im = im.convert("RGB")
                src_W, src_H = im.size
                longest = max(src_W, src_H)
                # Resize on the way in so the working tensor never
                # exceeds 1024 px longest edge — matches the upload
                # ceiling so we don't bloat the dataset with an
                # augmentation that's bigger than its source.
                if longest > 1024:
                    scale = 1024.0 / longest
                    im = im.resize((int(src_W * scale), int(src_H * scale)), PILImage.LANCZOS)
            work_W, work_H = im.size
            sx = work_W / float(src_W) if src_W > 0 else 1.0
            sy = work_H / float(src_H) if src_H > 0 else 1.0
            import numpy as _np
            arr = _np.asarray(im, dtype=_np.float32) / 255.0
            base = torch.from_numpy(arr).permute(2, 0, 1).contiguous().to(device)
        except Exception as e:
            print(f"[augment_generate] load failed for {filename}: {e}")
            return 0
        # Respect an explicit "user cleared all boxes" (editedBoxes == []
        # with editedBoxesSet) instead of falling through to the original
        # auto-detections, which would resurrect the deleted boxes into
        # the augmented copies + export.
        edited = imp.get("editedBoxes")
        if isinstance(edited, list) and (edited or imp.get("editedBoxesSet")):
            dets = edited
        else:
            dets = imp.get("detections") or []
        prepared_dets: list[dict] = []
        for d in dets:
            if not isinstance(d, dict):
                continue
            mask = d.get("mask")
            polys_src = (mask or {}).get("polygons") if isinstance(mask, dict) else None
            polys_work: list[list[list[float]]] = []
            if isinstance(polys_src, list):
                for p in polys_src:
                    if not isinstance(p, list):
                        continue
                    scaled = [
                        [float(pt[0]) * sx, float(pt[1]) * sy]
                        for pt in p
                        if isinstance(pt, (list, tuple)) and len(pt) >= 2
                    ]
                    if len(scaled) >= 3:
                        polys_work.append(scaled)
            bx = d.get("box_xyxy") or d.get("box") or []
            box_work: list[float] = []
            if isinstance(bx, (list, tuple)) and len(bx) >= 4:
                box_work = [
                    float(bx[0]) * sx, float(bx[1]) * sy,
                    float(bx[2]) * sx, float(bx[3]) * sy,
                ]
            label = (
                d.get("label")
                or d.get("predLabel")
                or d.get("pred_label")
                or ""
            )
            prepared_dets.append({"label": str(label), "polys": polys_work, "box": box_work})
        anno_per_copy: dict[str, list[dict]] = {}
        saved_count = 0
        with torch.no_grad():
            for k in range(n_copies):
                if cancel_event.is_set():
                    break
                seed_k = (hash((job.id, import_id, k)) & 0xFFFFFFFF) | 1
                g_k = torch.Generator(device="cpu").manual_seed(seed_k)
                try:
                    out = base
                    cur_matrix: "list[list[float]] | None" = None
                    if dom_bg_enabled and bg_paths and dets and _aug_should_apply(dom_bg_freq, k, n_copies, g_k):
                        out = _aug_apply_background_randomisation(out, dets, bg_paths, sx, sy, seed_k)
                    if occ_block_enabled and dets and _aug_should_apply(occ_block_freq, k, n_copies, g_k):
                        out = _aug_apply_block_occlusion(out, dets, occ_block_size, sx, sy, seed_k)
                    if dist_pw_enabled and _aug_should_apply(dist_pw_freq, k, n_copies, g_k):
                        pairs = _perspective_corner_pairs(dist_pw_strength, work_W, work_H, seed_k)
                        if pairs is not None:
                            H_mat = _solve_homography(*pairs)
                            cur_matrix = H_mat if cur_matrix is None else _matmul_3x3(H_mat, cur_matrix)
                        out = _aug_apply_perspective_warp(out, dist_pw_strength, seed_k)
                    if dist_sr_enabled and _aug_should_apply(dist_sr_freq, k, n_copies, g_k):
                        sr_params = _scale_rotation_sample_params(
                            dist_sr["smin"], dist_sr["smax"], dist_sr["rmin"], dist_sr["rmax"], seed_k,
                        )
                        if sr_params is not None:
                            s_val, r_val = sr_params
                            SR_mat = _scale_rotation_matrix(s_val, r_val, work_W, work_H)
                            cur_matrix = SR_mat if cur_matrix is None else _matmul_3x3(SR_mat, cur_matrix)
                        out = _aug_apply_scale_rotation(out, dist_sr["smin"], dist_sr["smax"], dist_sr["rmin"], dist_sr["rmax"], seed_k)
                    if dist_hs_enabled and _aug_should_apply(dist_hs_freq, k, n_copies, g_k):
                        out = _aug_apply_hue_shift(out, dist_hs_strength, seed_k)
                    if dom_lt_enabled and _aug_should_apply(dom_lt_freq, k, n_copies, g_k):
                        out = _aug_apply_lighting(out, dom_lt_strength, seed_k)
                    if cam_enabled and _aug_should_apply(cam_freq, k, n_copies, g_k):
                        out = _aug_apply(
                            out,
                            cam_params["mb"], cam_params["ns"], cam_params["cd"],
                            cam_params["ca"], cam_params["bd"],
                            seed_k,
                            lens_distortion=cam_params["ld"],
                            pixelation=cam_params["px"],
                            low_resolution=cam_params["lr"],
                            lens_glare=cam_params["lg"],
                        )
                    copy_annos: list[dict] = []
                    for det in prepared_dets:
                        warped_polys = _apply_matrix_to_polys(det["polys"], cur_matrix)
                        warped_box = _apply_matrix_to_box(det["box"], cur_matrix)
                        copy_annos.append({
                            "label": det["label"],
                            "polys": warped_polys,
                            "box": warped_box,
                        })
                    anno_per_copy[f"{k:02d}.jpg"] = copy_annos
                    out_np = (out.clamp(0.0, 1.0) * 255.0 + 0.5).byte().permute(1, 2, 0).cpu().numpy()
                    out_path = out_dir / f"{k:02d}.jpg"
                    # Encode → check file size → re-encode at lower
                    # quality if we're over the 500 KB budget. Mirrors
                    # the FE's resizeForUpload ladder so augmentations
                    # never balloon past the same cap source uploads
                    # are held to. The 1024 px resize above means most
                    # images already land well under 500 KB at
                    # quality=85; the ladder only kicks in on busy
                    # scenes where the entropy is high enough to
                    # need it.
                    out_im = PILImage.fromarray(out_np, mode="RGB")
                    target_bytes = 500 * 1024
                    quality_ladder = (85, 72, 60, 48, 36, 28)
                    saved_ok = False
                    for q in quality_ladder:
                        out_im.save(
                            out_path, format="JPEG", quality=q,
                            optimize=False, progressive=False,
                        )
                        try:
                            if out_path.stat().st_size <= target_bytes:
                                saved_ok = True
                                break
                        except OSError:
                            saved_ok = True
                            break
                    # If we exhausted the ladder and still oversize,
                    # last attempt sits on disk at quality=28 — the
                    # cap is a target, not a hard wall, so we don't
                    # drop the augmentation entirely just because it's
                    # 520 KB.
                    if not saved_ok:
                        pass
                    saved_count += 1
                except Exception as e:
                    print(f"[augment_generate] copy {k} for {filename} ({import_id}) failed: {e}")
        try:
            anno_path = out_dir / "annotations.json"
            with open(anno_path, "w") as f:
                json.dump({
                    "width": int(work_W),
                    "height": int(work_H),
                    "copies": anno_per_copy,
                }, f)
        except Exception as e:
            print(f"[augment_generate] annotations.json write for {import_id} failed: {e}")
        # Mirror the augmented copies + annotations to R2 so a remote
        # training rig (no access to this box's disk) can pull them as
        # extra training samples — see training.collect_augmentation_items.
        # Done FIRE-AND-FORGET on the background executor so these uploads
        # never block image generation or the progress counter (this was
        # making generation crawl + stick at 0/XXX). Disk stays the source
        # of truth for export; the rig only needs them by training time.
        if R2 is not None:
            _anno_bytes: "bytes | None" = None
            try:
                _anno_bytes = anno_path.read_bytes()
            except Exception:
                _anno_bytes = None

            def _upload_aug_to_r2(proj=job.project, iid=import_id,
                                  copy_names=list(anno_per_copy.keys()),
                                  anno_bytes=_anno_bytes, odir=out_dir) -> None:
                try:
                    if anno_bytes is not None:
                        R2.put_bytes(
                            f"projects/{proj}/augmentations/{iid}/annotations.json",
                            anno_bytes, content_type="application/json",
                        )
                    for cn in copy_names:
                        cp = odir / cn
                        if cp.exists():
                            R2.put_bytes(
                                f"projects/{proj}/augmentations/{iid}/{cn}",
                                cp.read_bytes(), content_type="image/jpeg",
                            )
                except Exception as e:
                    print(f"[augment_generate] R2 upload for {iid} failed: {e}")

            try:
                _BG_IMAGE_EXECUTOR.submit(_upload_aug_to_r2)
            except Exception as e:
                print(f"[augment_generate] R2 upload submit failed for {import_id}: {e}")
        # Surface progress on the job object so the FE's
        # /augment/job/active poll picks it up — assigning here means
        # the counter updates in the same tick the thread finishes,
        # no extra await round-trip needed.
        job.progress = {
            "index": i + 1,
            "total": len(imports),
            "image": filename,
            "phase": "augmenting",
        }
        return saved_count

    # Manifest writes are batched — load+rewrite of a multi-MB
    # manifest 941 times was the real reason the runner was slow.
    # Per-image we just stash {id: saved_count} in this dict; every
    # 25 images (and on completion) we flush all stashed updates in
    # one manifest read/write.
    n_aug_updates: dict[str, int] = {}
    FLUSH_EVERY = 25
    # Running total of augmentations saved so far. Drives the FE
    # progress bar — user asked for the counter to track per-
    # augmentation rather than per-image since a single image can
    # produce 1-3 augmented copies.
    saved_total = 0

    def _flush_n_aug_updates_sync() -> None:
        # Sync inner — caller holds the per-project manifest write
        # lock so this read-modify-write can't race a concurrent
        # label_charlie save (now that multiple runners can be in
        # flight at once).
        if not n_aug_updates:
            return
        try:
            m_now = load_manifest(job.project) or {}
            for entry in (m_now.get("imports") or []):
                iid = entry.get("id")
                if iid in n_aug_updates:
                    entry["n_augmentations"] = int(n_aug_updates[iid])
            save_manifest(job.project, m_now)
            n_aug_updates.clear()
        except Exception as e:
            print(f"[augment_generate] manifest batch flush failed: {e}")

    async def _flush_n_aug_updates() -> None:
        if not n_aug_updates:
            return
        wlk = await _manifest_write_lock(job.project)
        async with wlk:
            await asyncio.to_thread(_flush_n_aug_updates_sync)

    loop = asyncio.get_event_loop()
    for i, imp in enumerate(imports):
        if cancel_event.is_set():
            break
        # Per-image body runs on the dedicated _AUG_EXECUTOR thread
        # (one worker, persistent for the process lifetime). CUDA
        # context + caching allocator stay warm across iterations
        # instead of being re-initialised by a different worker each
        # time. Event loop is free for other handlers throughout.
        #
        # state["gpu_lock"] is the same asyncio.Lock every
        # interactive GPU endpoint (segment_box, click-to-detect,
        # classify_box, …) already holds while it works. Acquiring
        # it PER IMAGE rather than for the whole job means a click-
        # to-detect or add-box request from any other user / tab
        # slides in between augment images — at most one image's
        # latency (~0.5-2 s) of wait instead of the full run.
        async with state["gpu_lock"]:
            saved = await loop.run_in_executor(_AUG_EXECUTOR, _augment_one_image, i, imp)
        iid = imp.get("id")
        if iid:
            n_aug_updates[iid] = saved
        saved_total += saved
        # Overwrite the index/total set inside _augment_one_image so
        # the FE sees augmentation count, not image count.
        job.progress = {
            "index": min(saved_total, estimated_total_augmentations),
            "total": max(saved_total, estimated_total_augmentations),
            "image": imp.get("filename"),
            "phase": "augmenting",
        }
        # Batch flush every FLUSH_EVERY images so the FE's dataset
        # gallery picks up the per-tile augmentation icon
        # progressively, but we don't pay the multi-MB manifest
        # rewrite cost on every single image.
        if (i + 1) % FLUSH_EVERY == 0:
            await _flush_n_aug_updates()
        try:
            filename = imp.get("filename")
            await emit("progress", {"i": i + 1, "n": len(imports), "image": filename})
        except Exception:
            pass
        await asyncio.sleep(0)
    # Final flush so any tail-end updates (e.g. images 925-941 when
    # FLUSH_EVERY=25 lands on 925) land before the job finishes.
    if n_aug_updates:
        await _flush_n_aug_updates()


class AugmentGenerateIn(BaseModel):
    perImageMode: str = "1"
    config: dict = {}


@app.post(
    "/api/v2/projects/{project_id}/augment/generate",
    dependencies=[Depends(require_project_owner)],
)
async def v2_augment_generate(project_id: str, payload: AugmentGenerateIn):
    if not project_dir(project_id).exists():
        raise HTTPException(404, "project not found")
    manifest = load_manifest(project_id) or {}
    imports = manifest.get("imports") or []
    if not imports:
        raise HTTPException(400, "no dataset images to augment")
    # perImageMode "off" is the explicit clear path: wipe all
    # augmentations + reset n_augmentations, persist the config,
    # then skip scheduling a generate job. Lets the user "drop
    # everything" without needing to flip per-image to a count and
    # back. Returns job_id: null so the FE knows no job kicked off.
    clear_only = payload.perImageMode in ("off", "0")

    # Persist the config + perImageMode to the manifest so the
    # AugmentationsCard can restore the user's settings on revisit.
    cfg_to_save = {
        "perImageMode": str(payload.perImageMode),
        "config": payload.config or {},
    }
    try:
        write_lock = await _manifest_write_lock(project_id)
        async with write_lock:
            m = load_manifest(project_id) or {}
            m["augmentationConfig"] = cfg_to_save
            m["updatedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            save_manifest(project_id, m)
    except Exception:
        # Not fatal — the job still runs with the payload's config.
        pass

    # Wipe any prior augmentation outputs so the user always sees
    # exactly the set produced by the current settings (the Update
    # button is a "delete & regenerate" action, not "append"). Reset
    # every import's n_augmentations to 0 in the same write so the
    # gallery icon doesn't keep showing for tiles the new run may
    # skip (load error, no detections + only detection-dependent
    # augmentations enabled, etc.).
    aug_root = _augmentations_dir(project_id)
    if aug_root.exists():
        try:
            import shutil as _shutil
            _shutil.rmtree(aug_root)
        except Exception as e:
            print(f"[augment_generate] wipe of {aug_root} failed: {e}")
    try:
        write_lock2 = await _manifest_write_lock(project_id)
        async with write_lock2:
            m = load_manifest(project_id) or {}
            mutated = False
            for entry in (m.get("imports") or []):
                if entry.get("n_augmentations"):
                    entry["n_augmentations"] = 0
                    mutated = True
            if mutated:
                save_manifest(project_id, m)
    except Exception as e:
        print(f"[augment_generate] reset n_augmentations failed: {e}")

    if clear_only:
        # No job to schedule. Manifest is already wiped + reset
        # above; FE polls /overview and sees n_augmentations: 0.
        # Invalidate the payload caches + sidecars so the very next
        # /overview + /dataset-stats fetch from the FE reads from
        # the post-wipe truth instead of serving stale SWR while
        # the background rebuild lands.
        _invalidate_project_payloads(project_id)
        return {"job_id": None, "cleared": True}

    job = state["jobs"].schedule(
        "augment_generate", project_id,
        {"perImageMode": str(payload.perImageMode), "config": payload.config or {}},
        "system",
        n_images=len(imports),
    )
    state["jobs"].start_worker()
    return {"job_id": job.id}


@app.get(
    "/api/v2/projects/{project_id}/augment/config",
    dependencies=[Depends(require_project_read_access)],
)
async def v2_augment_get_config(project_id: str):
    """Return the augmentation config last persisted by Update. Empty
    dict when nothing's saved yet so the FE can default-fill its
    state. No auth gate beyond project existence — config is non-
    sensitive and matches the visibility of the rest of /overview."""
    if not project_dir(project_id).exists():
        raise HTTPException(404, "project not found")
    manifest = load_manifest(project_id) or {}
    return manifest.get("augmentationConfig") or {}


@app.get(
    "/api/v2/projects/{project_id}/augment/job/active",
    dependencies=[Depends(require_project_read_access)],
)
async def v2_augment_active_job(project_id: str):
    """Active augment_generate job for this project, or null."""
    if not project_dir(project_id).exists():
        raise HTTPException(404, "project not found")
    for j in state["jobs"].jobs.values():
        if j.project == project_id and j.kind == "augment_generate" and j.status in ("running", "queued"):
            return j.to_public()
    return None


@app.get(
    "/api/v2/projects/{project_id}/augmentations/{import_id}",
    dependencies=[Depends(require_project_read_access)],
)
async def v2_augment_list(project_id: str, import_id: str):
    """List the generated augmentation filenames for one dataset
    import. Empty list when nothing's been generated yet."""
    if not project_dir(project_id).exists():
        raise HTTPException(404)
    safe = "".join(c for c in import_id if c.isalnum())
    if not safe:
        raise HTTPException(400, "bad import id")
    aug_dir = _augmentations_dir(project_id, safe)
    items: list[str] = []
    if aug_dir.exists():
        for f in sorted(aug_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                items.append(f.name)
    return {"items": items}


@app.get(
    "/api/v2/projects/{project_id}/augmentations/{import_id}/annotations",
    dependencies=[Depends(require_project_read_access)],
)
async def v2_augment_annotations(project_id: str, import_id: str):
    """Per-import warped polygon JSON written by augment_generate.
    Empty when no augmentations exist for the import yet."""
    if not project_dir(project_id).exists():
        raise HTTPException(404)
    safe_id = "".join(c for c in import_id if c.isalnum())
    if not safe_id:
        raise HTTPException(400, "bad import id")
    path = _augmentations_dir(project_id, safe_id) / "annotations.json"
    if not path.exists():
        return {"width": 0, "height": 0, "copies": {}}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {"width": 0, "height": 0, "copies": {}}


@app.get(
    "/api/v2/projects/{project_id}/augmentations/{import_id}/{filename}",
    dependencies=[Depends(require_project_read_access)],
)
async def v2_augment_serve(project_id: str, import_id: str, filename: str):
    if not project_dir(project_id).exists():
        raise HTTPException(404)
    safe_id = "".join(c for c in import_id if c.isalnum())
    # Filename sanitised — alphanumeric, dot, dash, underscore.
    safe_fn = "".join(c for c in filename if c.isalnum() or c in "._-")
    if not safe_id or not safe_fn:
        raise HTTPException(400)
    aug_dir = _augmentations_dir(project_id, safe_id)
    path = (aug_dir / safe_fn).resolve()
    try:
        path.relative_to(aug_dir.resolve())
    except ValueError:
        raise HTTPException(403, "forbidden")
    if not path.exists():
        raise HTTPException(404)
    return await _serve_cached_image(project_id, f"augmentations/{safe_id}", safe_fn, path)


@app.delete(
    "/api/v2/projects/{project_id}/augmentations/{import_id}/{filename}",
    dependencies=[Depends(require_project_owner)],
)
async def v2_augment_delete(project_id: str, import_id: str, filename: str):
    """Delete one generated augmentation JPEG and prune its entry
    from the per-import annotations.json. Decrements the import's
    n_augmentations so the dataset gallery icon hides when the last
    copy is removed. Idempotent — re-deleting a missing file 200s."""
    if not project_dir(project_id).exists():
        raise HTTPException(404, "project not found")
    safe_id = "".join(c for c in import_id if c.isalnum())
    safe_fn = "".join(c for c in filename if c.isalnum() or c in "._-")
    if not safe_id or not safe_fn:
        raise HTTPException(400, "bad path component")
    aug_dir = _augmentations_dir(project_id, safe_id)
    path = (aug_dir / safe_fn).resolve()
    try:
        path.relative_to(aug_dir.resolve())
    except ValueError:
        raise HTTPException(403, "forbidden")
    if path.exists():
        try:
            path.unlink()
        except Exception as e:
            print(f"[augment_delete] unlink failed {path}: {e}")

    # Drop the corresponding entry from annotations.json so the FE
    # hover overlay stops referencing a file that no longer exists.
    anno_path = aug_dir / "annotations.json"
    if anno_path.exists():
        try:
            with open(anno_path) as f:
                data = json.load(f)
            copies = data.get("copies") or {}
            if isinstance(copies, dict) and safe_fn in copies:
                copies.pop(safe_fn, None)
                data["copies"] = copies
                with open(anno_path, "w") as f:
                    json.dump(data, f)
        except Exception as e:
            print(f"[augment_delete] annotations prune failed {anno_path}: {e}")

    # Count remaining augmentations on disk and update the manifest
    # so the dataset gallery icon reflects the new total. We count
    # actual JPEGs (not annotations.json) so a stale annotations
    # file can't keep the icon alive.
    remaining = 0
    if aug_dir.exists():
        for f in aug_dir.iterdir():
            if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                remaining += 1
    try:
        write_lock = await _manifest_write_lock(project_id)
        async with write_lock:
            m = load_manifest(project_id) or {}
            for entry in (m.get("imports") or []):
                if entry.get("id") == safe_id:
                    entry["n_augmentations"] = int(remaining)
                    break
            save_manifest(project_id, m)
    except Exception as e:
        print(f"[augment_delete] manifest update for {safe_id} failed: {e}")
    return {"ok": True, "remaining": remaining}


@app.get(
    "/api/v2/projects/{project_id}/augment/overlays/{overlay_id}",
    dependencies=[Depends(require_project_read_access)],
)
async def v2_augment_overlay_get(project_id: str, overlay_id: str):
    """Serve a previously-segmented overlay PNG for FE preview /
    drag-around UIs. Path-traversal guarded."""
    safe = "".join(c for c in overlay_id if c.isalnum())
    if not safe:
        raise HTTPException(400, "bad overlay id")
    path = _overlays_dir(project_id) / f"{safe}.png"
    if not path.exists():
        raise HTTPException(404, "overlay not found")
    return await _serve_cached_image(project_id, "augment_overlays", f"{safe}.png", path)


@app.post(
    "/api/v2/projects/{project_id}/augment/preview",
    dependencies=[Depends(require_project_owner)],
)
async def v2_augment_preview(project_id: str, payload: AugmentPreviewIn):
    """Render a single image through the camera/sensor augmentation
    chain and return JPEG bytes. Pure compute — nothing persisted."""
    proj = project_dir(project_id)
    if not proj.exists():
        raise HTTPException(404, "project not found")
    src = (payload.source or "").lower()
    if src not in ("reference", "import"):
        raise HTTPException(400, "source must be 'reference' or 'import'")
    subdir = "references" if src == "reference" else "imports"
    target = (proj / subdir / payload.filename).resolve()
    try:
        target.relative_to((proj / subdir).resolve())
    except ValueError:
        raise HTTPException(403, "forbidden")
    if not target.exists():
        raise HTTPException(404, "image not found")

    # Clamp slider values defensively; the FE should already keep
    # them in 0..10 but a malformed request shouldn't crash the
    # kernel.
    def _c(v: float) -> float:
        try:
            return max(0.0, min(10.0, float(v)))
        except (TypeError, ValueError):
            return 0.0

    mb = _c(payload.motion_blur)
    ns = _c(payload.noise)
    cd = _c(payload.colour_distortion)
    ca = _c(payload.chromatic_aberration)
    bd = _c(payload.bit_depth)
    ld = _c(payload.lens_distortion)
    px = _c(payload.pixelation)
    lr = _c(payload.low_resolution)
    lg = _c(payload.lens_glare)
    pw = _c(payload.perspective_warp)
    lt = _c(payload.lighting_strength)
    hu = _c(payload.hue_shift)
    try:
        sr_smin = float(payload.scale_min)
        sr_smax = float(payload.scale_max)
        sr_rmin = float(payload.rot_min)
        sr_rmax = float(payload.rot_max)
    except (TypeError, ValueError):
        sr_smin = sr_smax = 1.0
        sr_rmin = sr_rmax = 0.0
    # Block size is a percentage of segmentation coverage (0..40),
    # not a 0..10 strength like the other dials, so it has its own
    # clamp range. Capped at 40 because anything above starts
    # obscuring more object than is useful for training.
    try:
        bs = max(0.0, min(40.0, float(payload.block_size)))
    except (TypeError, ValueError):
        bs = 0.0
    show_outlines = bool(payload.show_outlines)
    # Object overlay (fraction 0..1; 0 = off). Requires source =
    # "import" since references don't carry detections to constrain
    # placement against.
    try:
        overlay_scale = max(0.0, min(1.0, float(payload.overlay_scale)))
    except (TypeError, ValueError):
        overlay_scale = 0.0
    # Build the list of overlay paths to composite. Accepts both
    # the new overlay_ids list and the legacy single overlay_id.
    overlay_paths: list["Path"] = []
    if overlay_scale > 0 and src == "import":
        raw_ids: list[str] = []
        if payload.overlay_ids:
            raw_ids.extend([x for x in payload.overlay_ids if isinstance(x, str)])
        if payload.overlay_id:
            raw_ids.append(payload.overlay_id)
        seen: set[str] = set()
        for oid in raw_ids:
            oid = (oid or "").strip()
            if not oid or oid in seen:
                continue
            safe = "".join(c for c in oid if c.isalnum())
            if not safe:
                continue
            cand = _overlays_dir(project_id) / f"{safe}.png"
            if cand.exists():
                overlay_paths.append(cand)
                seen.add(oid)
            # Cap at 3 overlays to match the FE slot count.
            if len(overlay_paths) >= 3:
                break

    # Background randomisation paths. Only honoured for source =
    # "import" (refs don't carry detections to define foreground).
    background_paths: list["Path"] = []
    if src == "import" and payload.background_ids:
        for bid in payload.background_ids:
            if not isinstance(bid, str):
                continue
            safe = "".join(c for c in bid if c.isalnum())
            if not safe:
                continue
            cand = _backgrounds_dir(project_id) / f"{safe}.jpg"
            if cand.exists():
                background_paths.append(cand)
    seed = int(payload.seed) if payload.seed is not None else 1234

    # Detections for the named import — loaded whenever the
    # request touches a feature that needs the polygon set.
    # Previously this was gated on bs > 0 / show_outlines only,
    # which meant a background-only preview (no occlusion, no
    # outlines) couldn't build the foreground mask and the
    # background composite silently fell through to the original
    # image.
    needs_detections = (
        bs > 0
        or show_outlines
        or bool(payload.background_ids)
        or bool(payload.overlay_ids)
        or bool(payload.overlay_id)
    )
    detections: list[dict] = []
    if src == "import" and needs_detections:
        manifest = load_manifest(project_id) or {}
        for imp in (manifest.get("imports") or []):
            if imp.get("filename") == payload.filename:
                # Honour an explicit "cleared all boxes" (editedBoxes == []
                # with editedBoxesSet) instead of resurrecting the auto
                # detections into the augmentation preview.
                edited = imp.get("editedBoxes")
                if isinstance(edited, list) and (edited or imp.get("editedBoxesSet")):
                    detections = edited
                else:
                    detections = imp.get("detections") or []
                break

    # Run the whole pipeline off the request thread so the
    # FastAPI event loop isn't blocked by the GPU work.
    loop = asyncio.get_running_loop()

    def _run() -> bytes:
        from io import BytesIO
        with PILImage.open(target) as im:
            im = im.convert("RGB")
            src_W, src_H = im.size
            # Cap the working size so the kernel is cheap; long edge
            # 720 px is plenty for a preview pane and keeps render
            # under ~30 ms on a modest GPU.
            longest = max(src_W, src_H)
            if longest > 720:
                scale = 720.0 / longest
                im = im.resize((int(src_W * scale), int(src_H * scale)), PILImage.LANCZOS)
        work_W, work_H = im.size
        # Pixel-space scale for detection coordinates — they live in
        # the SOURCE image's coordinate system; we shrink them to the
        # working preview size.
        sx = work_W / float(src_W) if src_W > 0 else 1.0
        sy = work_H / float(src_H) if src_H > 0 else 1.0

        device = _aug_device()
        import numpy as _np
        arr = _np.asarray(im, dtype=_np.float32) / 255.0  # HWC
        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous().to(device)
        # Wrap the whole GPU pipeline in no_grad — block occlusion,
        # overlays, and outline rendering all create tensors that
        # otherwise carry an autograd graph for nothing. ~10–30 ms
        # saved on big detection sets.
        with torch.no_grad():
            # Pipeline order — every step builds on the previous so a
            # later distortion can't misalign with a polygon mask
            # baked in earlier. Detection-dependent ops (background
            # swap, block occlusion, object overlay, outlines) run
            # FIRST while the canvas is still in original
            # coordinates; geometric warps then deform the composite;
            # colour shifts + the sensor chain land on top.
            out = t
            if background_paths and detections:
                out = _aug_apply_background_randomisation(out, detections, background_paths, sx, sy, seed)
            if bs > 0 and detections:
                out = _aug_apply_block_occlusion(out, detections, bs, sx, sy, seed)
            if overlay_paths and overlay_scale > 0:
                occupied: "torch.Tensor | None" = None
                for k, op_path in enumerate(overlay_paths):
                    out, occupied = _aug_apply_object_overlay(
                        out, op_path, overlay_scale, detections, sx, sy,
                        seed ^ (0xAB1 * (k + 1)), occupied,
                    )
            if show_outlines and detections:
                out = _aug_apply_segmentation_outlines(out, detections, sx, sy)
            # Geometric warps deform the composite. Polygon-based
            # ops happened above so we don't have to re-warp masks.
            if pw > 0:
                out = _aug_apply_perspective_warp(out, pw, seed)
            if not (sr_smin == 1.0 and sr_smax == 1.0 and sr_rmin == 0.0 and sr_rmax == 0.0):
                out = _aug_apply_scale_rotation(out, sr_smin, sr_smax, sr_rmin, sr_rmax, seed)
            # Colour shifts — applied on the warped composite so
            # they land on the deformed geometry.
            if hu > 0:
                out = _aug_apply_hue_shift(out, hu, seed)
            if lt > 0:
                out = _aug_apply_lighting(out, lt, seed)
            # Sensor chain — final step, the camera "captures" the
            # scene that's already been edited above.
            out = _aug_apply(
                out, mb, ns, cd, ca, bd, seed,
                lens_distortion=ld,
                pixelation=px,
                low_resolution=lr,
                lens_glare=lg,
            )

        out_np = (out.clamp(0.0, 1.0) * 255.0 + 0.5).byte().permute(1, 2, 0).cpu().numpy()
        buf = BytesIO()
        PILImage.fromarray(out_np, mode="RGB").save(buf, format="JPEG", quality=82, optimize=False, progressive=False)
        return buf.getvalue()

    data = await loop.run_in_executor(None, _run)
    from fastapi.responses import Response as _Response
    return _Response(
        content=data,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


@app.get(
    "/api/v2/projects/{project_id}/imports/{import_id}/labelled_preview",
    dependencies=[Depends(require_project_read_access)],
)
async def v2_serve_labelled_preview(project_id: str, import_id: str):
    """Lazy-rendered annotated preview for a single import. ~600 px
    on the long edge, JPEG, dim+blur background with the segmented
    objects bright + per-label colour-tinted. Cached to disk under
    `imports/<id>__lp_v1.jpg` so repeat reads are pure file serves.

    Render-on-first-request keeps /imports/process snappy — we don't
    block the user's drop-zone progress indicator on a PIL render
    that the FE can pull lazily once it scrolls into view.
    """
    proj = project_dir(project_id)
    if not proj.exists():
        raise HTTPException(404, "project not found")
    cached_path = _labelled_preview_path(project_id, import_id)

    # Fast path — preview already on disk.
    if cached_path.exists():
        return await _serve_cached_image(project_id, "imports", cached_path.name, cached_path)

    # O(1) lookup via the import index — avoids a full manifest deepcopy
    # (200-500 ms on large projects) and an O(n) linear scan. Ensures
    # the cache is warm first so the index is populated; copy=False is
    # safe because we only read fields here, never mutate.
    await asyncio.to_thread(load_manifest, project_id, False)
    with _MANIFEST_CACHE_LOCK:
        imp = _MANIFEST_IMPORT_INDEX.get(project_id, {}).get(import_id)
    if not imp:
        raise HTTPException(404, "import not found")
    fn = imp.get("filename")
    if not fn:
        raise HTTPException(404, "import has no stored filename")
    src_path = proj / "images" / fn
    if not src_path.exists():
        raise HTTPException(404, "source image missing on disk")

    # Tags come from the manifest; read with copy=False (index warm after
    # the to_thread call above, so this is just a dict lookup).
    with _MANIFEST_CACHE_LOCK:
        _cached_m = _MANIFEST_CACHE.get(project_id) or {}
    project_labels = list(_cached_m.get("tags") or [])

    # Edited boxes (with masks) win over the raw GD detections — that
    # mirrors what the user sees in the gallery thumb (kept boxes
    # only). Falls back to detections when no edits have been made.
    edited = imp.get("editedBoxes")
    if isinstance(edited, list) and edited:
        det_source = edited
    else:
        det_source = imp.get("detections") or []

    loop = asyncio.get_running_loop()

    def _render() -> bytes:
        with PILImage.open(src_path) as im:
            im = im.convert("RGB")
            preview = _render_labelled_preview(im, det_source, project_labels)
        cached_path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temp file then rename so concurrent requests
        # never read a half-written JPEG. The suffix is unique per
        # call (pid + uuid) so two parallel first-renders for the
        # same import don't both write to the same tmp path and
        # interleave bytes.
        tmp = cached_path.with_suffix(_unique_tmp_suffix())
        preview.save(tmp, format="JPEG", quality=72, optimize=True, progressive=True)
        tmp.replace(cached_path)
        return cached_path.read_bytes()

    # Per-(project, import) lock: only one render at a time, others
    # wait and pick up the freshly-baked file. Without this, parallel
    # first-requests both PIL-render the same image, double the GPU/CPU
    # cost, and one's rename can clobber the other half-written.
    lock = _thumb_render_lock(project_id, f"labelled_preview:{import_id}")
    async with lock:
        # Re-check after acquiring — the previous holder may have
        # done the render already.
        if cached_path.exists():
            return await _serve_cached_image(project_id, "imports", cached_path.name, cached_path)
        try:
            await loop.run_in_executor(None, _render)
        except Exception as e:
            print(f"[labelled-preview] render failed for {project_id}/{import_id}: {e}")
            # Render failed — fall back to the original full-size image
            # so the gallery isn't broken, just slower for this one entry.
            return await _serve_cached_image(project_id, "imports", fn, src_path)

    return await _serve_cached_image(project_id, "imports", cached_path.name, cached_path)


# A user-uploaded dataset cover (distinct from a cover picked from an existing
# import/reference). Stored under a reserved local filename + R2 key and flagged
# on the manifest with `cover_uploaded: true`, which short-circuits the
# cover-thumb + summary logic so it survives image deletes (the cover-rescue
# passes only touch `manifest["cover"]`, never this flag).
UPLOADED_COVER_SENTINEL = "cover_upload.jpg"


def _project_uploaded_cover_key(project_id: str) -> str:
    return f"projects/{project_id}/cover_upload.jpg"


@app.post(
    "/api/projects/{project_id}/cover_upload",
    dependencies=[Depends(require_project_owner)],
)
async def upload_project_cover(project_id: str, file: UploadFile = File(...)):
    """Upload a custom cover image for a dataset. Re-encoded to a capped JPEG,
    stored locally (so cover_thumb renders from it) + in R2 (durable), and
    flagged on the manifest as an uploaded cover so it isn't treated as one of
    the dataset's images and never gets rescued away on image deletes."""
    proj = project_dir(project_id)
    if not proj.exists():
        raise HTTPException(404, "project not found")
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "empty file")
    try:
        with PILImage.open(io.BytesIO(raw)) as im:
            im = im.convert("RGB")
            W, H = im.size
            longest = max(W, H)
            cap = 1600
            if longest > cap:
                scale = cap / longest
                im = im.resize((int(W * scale), int(H * scale)), PILImage.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=85, optimize=True, progressive=True)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(400, "could not read image")
    jpeg = buf.getvalue()
    # Local source for cover_thumb's render path.
    try:
        (proj / UPLOADED_COVER_SENTINEL).write_bytes(jpeg)
    except Exception as e:
        print(f"[cover-upload] local write failed for {project_id}: {e}")
        raise HTTPException(500, "could not store cover")
    # Durable copy in R2 so the cover survives a disk reset (cover_thumb
    # restores it on read if the local file is gone).
    if R2 is not None:
        try:
            R2.put_bytes(_project_uploaded_cover_key(project_id), jpeg, "image/jpeg")
        except Exception as e:
            print(f"[cover-upload] R2 put failed for {project_id}: {e}")
    write_lock = await _manifest_write_lock(project_id)
    async with write_lock:
        manifest = load_manifest(project_id)
        if not manifest:
            raise HTTPException(404, "manifest not found")
        manifest["cover_uploaded"] = True
        manifest["cover_blurhash"] = None  # re-derived (or skipped) on next read
        manifest["updatedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        save_manifest(project_id, manifest)
    _invalidate_project_payloads(project_id)
    return {"ok": True, "cover_uploaded": True}


def _upscale_cover_if_small(im, floor: int):
    """Enlarge a small cover so a full-width hero (or a retina card) never
    browser-upscales a tiny source into a blocky mess. Cheap, CPU-only: LANCZOS
    resample up to `floor` on the longest edge + a light unsharp mask to restore
    crispness. It does NOT invent detail (that needs ML super-resolution) — it
    just makes a small image read clean as a background behind a scrim/title.
    No-op when the image already meets `floor`. Returns the (possibly new) image."""
    from PIL import ImageFilter
    w, h = im.size
    longest = max(w, h)
    if longest <= 0 or longest >= floor:
        return im
    scale = floor / float(longest)
    im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), PILImage.LANCZOS)
    # Mild sharpening to counter the softness Lanczos enlargement introduces.
    return im.filter(ImageFilter.UnsharpMask(radius=1.2, percent=80, threshold=2))


@app.get(
    "/api/projects/{project_id}/cover_thumb",
    dependencies=[Depends(require_project_read_access)],
)
async def serve_cover_thumb(project_id: str, w: int = 480, ai: int = 0):
    """Lazy-rendered ~480 px thumbnail of the project cover. Used by
    the workspace + community card grids so the first paint is small
    bytes, not a full-resolution upload. Cached to
    `<project>/cover_thumb_v1.jpg` and re-rendered when the source
    cover file's mtime changes (cover swap, re-upload).

    Works for both V1 (cover lives in imports/) and V2 (cover lives
    in references/, occasionally imports/) projects — the cached
    filename keeps the project_id only so a cover swap blows the same
    cache entry away regardless of source subdir.
    """
    proj = project_dir(project_id)
    if not proj.exists():
        raise HTTPException(404, "project not found")
    manifest = load_manifest(project_id)
    if not manifest:
        raise HTTPException(404, "manifest not found")

    # Uploaded cover takes precedence over any picked import/reference and is
    # served from its reserved local file (restored from R2 if the disk copy
    # was lost).
    if manifest.get("cover_uploaded"):
        src_path = proj / UPLOADED_COVER_SENTINEL
        if not src_path.exists() and R2 is not None:
            try:
                data = R2.get_bytes(_project_uploaded_cover_key(project_id))
                src_path.write_bytes(data)
            except Exception as e:
                print(f"[cover-thumb] R2 restore failed for {project_id}: {e}")
        if not src_path.exists():
            raise HTTPException(404, "uploaded cover missing")
        cover = UPLOADED_COVER_SENTINEL
    else:
        cover = manifest.get("cover")
        if not cover:
            raise HTTPException(404, "project has no cover")

        # Try references first (V2), fall back to imports (V1 + early-V2
        # cover-from-import flow). The first existing path wins.
        candidates: list[Path] = [
            proj / "references" / cover,
            proj / "images" / cover,
        ]
        src_path = next((p for p in candidates if p.exists()), None)
        if src_path is None:
            raise HTTPException(404, "cover file missing on disk")

    # Cache filename is keyed by an 8-char hash of the source filename
    # so a cover swap (dedupe deleted the old cover and rescued in a
    # different image, the user manually picked a new one, etc.) lands
    # on a FRESH cache slot instead of trying to mtime-compare against
    # the previous cover's bake. Without this key, the old thumb could
    # serve indefinitely if its mtime happened to be later than the
    # new src's mtime — which is exactly how the "cover went white
    # after viewing a project" bug surfaced.
    # Target longest-edge size. Cards request the default ~480 px; the dataset
    # hero requests a larger variant (e.g. 1280) so the full-width banner stays
    # crisp. Clamped so a crafted `w` can't trigger a giant render.
    target = max(240, min(1600, int(w or 480)))
    # AI super-resolution (Real-ESRGAN on the GPU) for the hero, so a small cover
    # gains real detail rather than a soft Lanczos enlarge. Only meaningful when
    # enlarging; cached separately so the AI + plain variants don't clobber.
    use_ai = bool(ai)
    ai_tag = "_ai" if use_ai else ""
    import hashlib
    name_hash = hashlib.sha1(cover.encode("utf-8")).hexdigest()[:8]
    # Target (and AI flag) are part of the cache key so the 480 px card thumb,
    # the 1280 px hero variant and the AI variant cache to separate files
    # instead of clobbering each other.
    cached_path = proj / f"cover_thumb_v2_{name_hash}_{target}{ai_tag}.jpg"
    try:
        src_mtime = src_path.stat().st_mtime
    except OSError:
        src_mtime = 0.0

    # Re-render when the cached thumb is missing or older than the
    # source — covers the cover re-upload case (same filename, fresher
    # bytes).
    needs_render = True
    if cached_path.exists():
        try:
            cached_mtime = cached_path.stat().st_mtime
            if cached_mtime >= src_mtime:
                needs_render = False
        except OSError:
            needs_render = True

    if needs_render:
        loop = asyncio.get_running_loop()

        def _render() -> None:
            with PILImage.open(src_path) as im:
                im = im.convert("RGB")
                W, H = im.size
                longest = max(W, H)
                if longest > target:
                    scale = target / longest
                    im = im.resize((int(W * scale), int(H * scale)), PILImage.LANCZOS)
                elif use_ai and longest < target:
                    # GPU super-resolution: real detail for the full-width hero
                    # instead of a soft enlarge. Falls back to the Lanczos floor
                    # on any failure (no GPU / weights / load error).
                    try:
                        import upscale as _ai_upscale
                        im = _ai_upscale.upscale_to(im, target)
                    except Exception as e:
                        print(f"[cover-thumb] AI upscale failed for {project_id}: {e}; using Lanczos", flush=True)
                        im = _upscale_cover_if_small(im, target)
                else:
                    # Source at/below target: floor small covers up to the target
                    # (Lanczos + light sharpen) so the hero/card never browser-
                    # upscales a tiny image. No-op at the exact size.
                    im = _upscale_cover_if_small(im, target)
            tmp = cached_path.with_suffix(_unique_tmp_suffix())
            im.save(tmp, format="JPEG", quality=78, optimize=True, progressive=True)
            tmp.replace(cached_path)

        # Per-project render lock so two parallel first-requests don't
        # both PIL-render the same cover and stomp on each other's
        # tmp file. The lock also serialises within a single worker —
        # the second request just waits for the first's bake to land,
        # then reads the cached file. Across workers the same project
        # could still double-render once at startup, but tmp suffixes
        # are now unique so the renames don't corrupt each other.
        lock = _thumb_render_lock(project_id, f"cover_thumb_{target}{ai_tag}")
        async with lock:
            # Re-check after acquiring the lock; another waiter may
            # have completed the render in the meantime.
            try:
                cached_mtime = cached_path.stat().st_mtime if cached_path.exists() else 0.0
            except OSError:
                cached_mtime = 0.0
            if cached_mtime < src_mtime:
                try:
                    await loop.run_in_executor(None, _render)
                except Exception as e:
                    print(f"[cover-thumb] render failed for {project_id}: {e}")
                    # Fall back to the original cover so the card still
                    # renders, just at the original size.
                    return await _serve_cached_image(project_id, src_path.parent.name, cover, src_path)

    return await _serve_cached_image(project_id, "cover_thumb", cached_path.name, cached_path)


def _load_reference_image_pil(raw: bytes, project_id: str, filename: str):
    """Decode a reference image for the V2 reference editor. Prefer the
    uploaded bytes (new, unsaved references); otherwise load the saved file
    from disk by project_id + filename. Existing projects can't always
    re-upload the CDN-hosted bytes from the browser (the 'stale reference
    image' 400), so the disk fallback keeps click-to-detect / segment /
    classify working on saved references. Path-traversal guarded like
    v2_serve_reference."""
    if raw:
        try:
            return PILImage.open(io.BytesIO(raw)).convert("RGB")
        except Exception as e:
            raise HTTPException(400, f"could not decode image: {e}")
    fn = (filename or "").strip()
    if project_id and fn:
        refs_root = (project_dir(project_id) / "references").resolve()
        target = (refs_root / fn).resolve()
        try:
            target.relative_to(refs_root)
        except ValueError:
            raise HTTPException(403, "forbidden")
        if not target.exists():
            raise HTTPException(404, f"reference image not found: {fn}")
        try:
            return PILImage.open(target).convert("RGB")
        except Exception as e:
            raise HTTPException(400, f"could not decode reference image: {e}")
    raise HTTPException(400, "no image bytes and no project_id/filename to load from disk")


@app.post("/api/v2/references/embed_crops")
async def v2_embed_crops(
    image: UploadFile = File(...),
    boxes: str = Form(...),
    return_crop: bool = Form(False),
):
    """Crop each box from `image` and embed each crop with DINOv2-base
    (matches `detect_and_crop.py` in the repo root). Embeddings are
    computed in a single batched forward pass — typical 5–20× speedup
    vs encoding one box at a time.

    Request: multipart with the image file + a JSON list of
    `[x0, y0, x1, y1]` boxes in pixel coords. `return_crop=true` to
    also receive a base64 JPEG of each crop (default off — the V2
    reference flow doesn't display crops anymore, so encoding them
    is wasted CPU + bandwidth).

    Response: `{ crops: [{ index, box, embedding: float[D], crop_jpg_b64? }] }`
    where D is the active DINOv2 model's hidden size (1024 for large,
    768 for base — see v2_dinov2.EMBEDDING_DIM).
    """
    import base64
    import v2_dinov2

    if not v2_dinov2.is_loaded():
        raise HTTPException(503, "DINOv2-base not loaded yet (still warming up)")

    try:
        bbs = json.loads(boxes)
        if not isinstance(bbs, list):
            raise ValueError("boxes must be a JSON array")
    except Exception as e:
        raise HTTPException(400, f"invalid boxes payload: {e}")

    raw = await image.read()
    if not raw:
        raise HTTPException(400, "empty image upload")
    try:
        image_pil = PILImage.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"could not decode image: {e}")
    W, H = image_pil.size

    loop = asyncio.get_running_loop()

    def _run():
        # First pass: validate boxes + crop, collecting both the raw
        # crops (for optional JPEG return) and the squared crops (for
        # the model). Skipped boxes get logged but not embedded.
        kept_indices: list[int] = []
        kept_xyxy: list[tuple[int, int, int, int]] = []
        kept_crops: list[PILImage.Image] = []
        kept_squares: list[PILImage.Image] = []
        for i, bb in enumerate(bbs):
            try:
                if not (isinstance(bb, list) and len(bb) == 4):
                    print(f"[v2-embed] box {i} skipped — not a 4-tuple: {bb!r}")
                    continue
                x0 = max(0, int(round(float(bb[0]))))
                y0 = max(0, int(round(float(bb[1]))))
                x1 = min(W, int(round(float(bb[2]))))
                y1 = min(H, int(round(float(bb[3]))))
                if x1 - x0 < 4 or y1 - y0 < 4:
                    print(f"[v2-embed] box {i} skipped — too small after clip: {bb!r} → ({x0},{y0},{x1},{y1}) on {W}x{H}")
                    continue
                crop = image_pil.crop((x0, y0, x1, y1))
                square = v2_dinov2.center_square_crop(crop)
                kept_indices.append(i)
                kept_xyxy.append((x0, y0, x1, y1))
                kept_crops.append(crop)
                kept_squares.append(square)
            except Exception as e:
                print(f"[v2-embed] box {i} crop failed: {e}")

        # Single batched forward pass for ALL boxes in this image.
        vecs = v2_dinov2.encode_images_batch(kept_squares) if kept_squares else None

        results: list[dict] = []
        for k, (i, xyxy, crop) in enumerate(zip(kept_indices, kept_xyxy, kept_crops)):
            entry: dict = {
                "index": i,
                "box": list(xyxy),
                "embedding": [round(float(x), 6) for x in vecs[k].tolist()] if vecs is not None else [],
                # Stamp the version so callers can persist it next to
                # the embedding — centroid-build invalidates anything
                # that doesn't match the current EMBED_VERSION.
                "embed_version": v2_dinov2.EMBED_VERSION,
            }
            if return_crop:
                buf = io.BytesIO()
                crop.convert("RGB").save(buf, format="JPEG", quality=82)
                entry["crop_jpg_b64"] = base64.b64encode(buf.getvalue()).decode("ascii")
            results.append(entry)
        print(f"[v2-embed] {len(results)}/{len(bbs)} boxes embedded for {W}x{H} image (v{v2_dinov2.EMBED_VERSION})")
        return results

    try:
        # Interactive priority — embed_crops is fired by the
        # references editor when the user drops in fresh boxes.
        # Holding behind a background job would feel like the
        # editor froze.
        async with state["gpu_lock"].interactive():
            crops = await loop.run_in_executor(None, _run)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"embed_crops failed: {exc}")

    return {"width": W, "height": H, "crops": crops}


# ─── V2 reference-centroid helpers ─────────────────────────────────────────────
# The imports endpoint needs to decide each detection's final label
# by comparing its DINOv2-base embedding against per-label centroids
# of the project's reference images. These helpers load embeddings
# from the manifest, lazily backfill any references that pre-date
# embedding storage, and expose the centroids as a {label: vec} map.

# Cached result of _v2_load_or_backfill_reference_embeddings, keyed
# by project_id. Value: (manifest_mtime, siglip_active, by_label_dino,
# by_label_siglip). The function is on the hot path for every
# interactive endpoint (click-to-detect, add-box,
# segment_and_classify_box) and the labelling job's setup; on big
# projects the iterate-all-refs + per-call deepcopy was the dominant
# source of latency (200-500ms per call). Hit conditions are strict
# (disk mtime + SigLIP loaded-state both match), so any real change
# busts the cache.
_V2_REFS_CACHE: dict[str, tuple[float, bool, dict, dict]] = {}
_V2_REFS_CACHE_LOCK = __import__("threading").Lock()


def _v2_load_or_backfill_reference_embeddings(project_id: str) -> tuple[dict, dict, bool]:
    """Walk the project's references and gather (label, embedding)
    pairs for every annotated box, for BOTH encoders (DINOv2 and
    SigLIP2). References missing an embedding for either encoder
    (or carrying a stale-version one) get re-encoded inline; the
    manifest is rewritten so the next call is a pure read.

    Returns:
        (by_label_dino, by_label_siglip, dirty)

        by_label_dino   — {label_lower: list[np.ndarray]} of DINOv2 vecs
        by_label_siglip — {label_lower: list[np.ndarray]} of SigLIP vecs
                          (empty dict if SigLIP isn't loaded / disabled)
        dirty           — whether the manifest was rewritten

    Each detection ends up with up to two stored embeddings:
      - "embedding" + "embed_version" (DINOv2 — legacy field name)
      - "siglip_embedding" + "siglip_version" (SigLIP2)
    so the resolver can score against either independently.
    """
    import numpy as _np
    import v2_dinov2 as _v2d
    import v2_siglip as _v2s
    # Steady-state fast path: the function's full body iterates every
    # reference's every detection on every call, which on a 1000-image
    # project with hundreds of refs was the dominant cost for click-
    # to-detect / add-box. Cache the result by manifest mtime — any
    # write to the manifest bumps mtime via save_manifest, busting the
    # cache automatically. Siglip loaded-state is part of the key so
    # the cache also self-recovers when the encoder warms up mid-
    # session.
    disk_mtime = _manifest_disk_mtime(project_id)
    siglip_active = _v2s.is_loaded()
    if disk_mtime > 0:
        with _V2_REFS_CACHE_LOCK:
            cached = _V2_REFS_CACHE.get(project_id)
            if cached is not None:
                cached_mtime, cached_siglip, cached_dino, cached_siglip_dict = cached
                if cached_mtime == disk_mtime and cached_siglip == siglip_active:
                    return cached_dino, cached_siglip_dict, False
    # Cold path: load + iterate + backfill if needed. copy=False
    # because we either (a) don't mutate the manifest (steady state),
    # or (b) mutate and then save_manifest immediately, which re-
    # seeds the cache with a fresh deepcopy. Saves another 200-500ms
    # per cold call on the same big manifest.
    manifest = load_manifest(project_id, copy=False) or {}
    refs = manifest.get("references") or []
    by_label_dino: dict[str, list[_np.ndarray]] = {}
    by_label_siglip: dict[str, list[_np.ndarray]] = {}
    dirty = False
    proj = project_dir(project_id)
    refs_dir = proj / "references"

    siglip_active = _v2s.is_loaded()

    # First pass: classify each detection as DINOv2-fresh / DINOv2-stale
    # and SigLIP-fresh / SigLIP-stale. Fresh embeddings go straight
    # into the by_label dicts. Stale or missing ones are queued for
    # the backfill pass below — keyed by ref so we only decode each
    # source image once even when both encoders need a refresh.
    pending_by_ref: dict[int, list[tuple[int, str, list[float], list | None, bool, bool]]] = {}
    for ri, ref in enumerate(refs):
        for di, d in enumerate(ref.get("detections") or []):
            label = (d.get("label") or "").strip()
            if not label:
                continue
            key = label.lower()

            # DINOv2 freshness check.
            d_emb = d.get("embedding")
            d_emb_v = d.get("embed_version")
            d_fresh = (
                isinstance(d_emb, list) and len(d_emb) > 0
                and d_emb_v == _v2d.EMBED_VERSION
            )
            if d_fresh:
                by_label_dino.setdefault(key, []).append(_np.asarray(d_emb, dtype=_np.float32))

            # SigLIP freshness check (only relevant when the encoder
            # is loaded — otherwise we treat it as "fresh" so we
            # don't queue pointless backfill work).
            if siglip_active:
                s_emb = d.get("siglip_embedding")
                s_emb_v = d.get("siglip_version")
                s_fresh = (
                    isinstance(s_emb, list) and len(s_emb) > 0
                    and s_emb_v == _v2s.EMBED_VERSION
                )
                if s_fresh:
                    by_label_siglip.setdefault(key, []).append(_np.asarray(s_emb, dtype=_np.float32))
            else:
                s_fresh = True

            if d_fresh and s_fresh:
                continue

            box = d.get("box") or []
            if not (isinstance(box, list) and len(box) == 4):
                continue
            mask_polys = None
            m = d.get("mask")
            if isinstance(m, dict):
                mask_polys = m.get("polygons")
            pending_by_ref.setdefault(ri, []).append(
                (di, key, [float(x) for x in box], mask_polys,
                 not d_fresh, siglip_active and not s_fresh),
            )

    if pending_by_ref:
        if not _v2d.is_loaded():
            # Not loaded yet — skip backfill, work with what we've
            # already got. The next import after the model finishes
            # warming will pick these up.
            print("[v2-centroids] DINOv2 not loaded — backfill deferred")
        else:
            for ri, items in pending_by_ref.items():
                ref = refs[ri]
                fn = ref.get("filename")
                if not fn:
                    continue
                p = refs_dir / fn
                if not p.exists():
                    continue
                try:
                    image_pil = PILImage.open(p).convert("RGB")
                except Exception as e:
                    print(f"[v2-centroids] couldn't open {p}: {e}")
                    continue
                W, H = image_pil.size
                squares: list[PILImage.Image] = []
                kept: list[tuple[int, str, bool, bool]] = []
                for di, key, box, mask_polys, need_dino, need_siglip in items:
                    x0 = max(0, int(round(box[0])))
                    y0 = max(0, int(round(box[1])))
                    x1 = min(W, int(round(box[2])))
                    y1 = min(H, int(round(box[3])))
                    if x1 - x0 < 4 or y1 - y0 < 4:
                        continue
                    crop = _v2d.inpaint_bbox_crop(image_pil, (x0, y0, x1, y1), mask_polys)
                    squares.append(_v2d.center_square_crop(crop))
                    kept.append((di, key, need_dino, need_siglip))
                if not squares:
                    continue
                # DINOv2 batched re-encode for any detection that
                # needed a refresh OR that needed SigLIP only — we
                # also store the DINOv2 vector unconditionally since
                # we already have the crop and it's cheap to embed.
                d_vecs = _v2d.encode_images_batch(squares)
                # SigLIP: only run when we have any need_siglip
                # boxes AND the encoder is loaded.
                s_vecs = None
                if siglip_active and any(ns for *_, ns in kept):
                    try:
                        s_vecs = _v2s.encode_images_batch(squares)
                    except Exception as e:
                        print(f"[v2-centroids] siglip backfill failed: {e}")
                        s_vecs = None
                for k, (di, key, need_dino, need_siglip) in enumerate(kept):
                    if need_dino:
                        v = d_vecs[k]
                        refs[ri]["detections"][di]["embedding"] = [round(float(x), 6) for x in v.tolist()]
                        refs[ri]["detections"][di]["embed_version"] = _v2d.EMBED_VERSION
                        by_label_dino.setdefault(key, []).append(_np.asarray(v, dtype=_np.float32))
                        dirty = True
                    if need_siglip and s_vecs is not None and k < s_vecs.shape[0]:
                        sv = s_vecs[k]
                        refs[ri]["detections"][di]["siglip_embedding"] = [round(float(x), 6) for x in sv.tolist()]
                        refs[ri]["detections"][di]["siglip_version"] = _v2s.EMBED_VERSION
                        by_label_siglip.setdefault(key, []).append(_np.asarray(sv, dtype=_np.float32))
                        dirty = True
                print(
                    f"[v2-centroids] backfilled {len(kept)} embedding(s) for ref {ref.get('id')} "
                    f"(dinov{_v2d.EMBED_VERSION}"
                    + (f" + siglipv{_v2s.EMBED_VERSION}" if s_vecs is not None else "")
                    + ")"
                )

    if dirty:
        manifest["references"] = refs
        save_manifest(project_id, manifest)

    # Diagnostic: summarise what we ended up with so a "only one
    # class scored" symptom in the FE popup is traceable to the
    # actual ref-bucket counts. Also surface any detections that
    # had blank labels — those are silently skipped above and could
    # explain a missing class.
    blank_label_count = 0
    for ref in refs:
        for d in ref.get("detections") or []:
            if not (d.get("label") or "").strip():
                blank_label_count += 1
    counts_dino = {k: len(v) for k, v in by_label_dino.items()}
    counts_siglip = {k: len(v) for k, v in by_label_siglip.items()}
    print(
        f"[v2-centroids] project={project_id} dino={counts_dino} siglip={counts_siglip}"
        + (f" blank_label_dets={blank_label_count}" if blank_label_count else "")
    )
    # Seed the cache. After a dirty save the disk mtime is bumped to a
    # new value; cache against THAT so the next read returns straight
    # from memory.
    new_mtime = _manifest_disk_mtime(project_id)
    if new_mtime > 0:
        with _V2_REFS_CACHE_LOCK:
            _V2_REFS_CACHE[project_id] = (new_mtime, siglip_active, by_label_dino, by_label_siglip)
    return by_label_dino, by_label_siglip, dirty


# ─── V2 patch-level matching ──────────────────────────────────────────────────
# DINOv2 patch tokens (1369 per crop at 518×518) carry strong
# locality-aware features that the pooled patch_mean smears together.
# When V2_PATCH_MATCH=on, the resolver swaps the pooled-cosine
# scoring for per-patch matching: each query foreground patch finds
# its nearest match across all reference foreground patches per
# label, and the per-label score is the top-K mean of those
# best matches. Discriminative patches (ear-tip texture, fur pattern)
# vote loudly even when they're a small fraction of the crop.
#
# Storage: per-project NPZ at `projects/<id>/v2_patches_v8.npz`
# keyed by `<ref_id>__<det_idx>__tokens` and `__fg`. Versioned
# filename so EMBED_VERSION bumps invalidate cleanly.

_PATCH_STORE_VERSION = 8
_PATCH_TOP_K_QUERY = int(os.environ.get("V2_PATCH_TOP_K", "10"))


def _v2_patch_store_path(project_id: str) -> Path:
    return project_dir(project_id) / f"v2_patches_v{_PATCH_STORE_VERSION}.npz"


def _v2_patch_match_enabled() -> bool:
    # Default ON. Patch-level matching produces materially wider
    # margins on fine-grained pairs than pooled cosine, so it's the
    # primary scoring path for the specific resolver. Set
    # V2_PATCH_MATCH=off explicitly to fall back to pooled scoring
    # (e.g. for A/B comparison or on hosts where the per-project
    # NPZ store would be too large).
    return os.environ.get("V2_PATCH_MATCH", "on").lower() not in ("0", "off", "false")


def _v2_load_patch_store(project_id: str) -> dict[str, "np.ndarray"]:
    """Load the NPZ patch store for a project. Returns a flat dict
    of `<ref_id>__<det_idx>__tokens` and `<ref_id>__<det_idx>__fg`
    arrays. Empty dict when the file doesn't exist yet."""
    import numpy as _np
    p = _v2_patch_store_path(project_id)
    if not p.exists():
        return {}
    try:
        with _np.load(p) as data:
            return {k: data[k] for k in data.files}
    except Exception as e:
        print(f"[v2-patches] load failed {p}: {e}")
        return {}


def _v2_save_patch_store(project_id: str, store: dict[str, "np.ndarray"]) -> None:
    """Atomic NPZ rewrite. Per-project so cleanup is just unlink.

    The temp filename ends with `.npz` because np.savez_compressed
    silently appends `.npz` to any path that doesn't already have
    that suffix — without this, the written file lives at
    `<tmp>.npz` while we try to rename `<tmp>` (which doesn't
    exist), and the FileNotFoundError bubbles up and torpedoes
    the entire reference loading path.
    """
    import numpy as _np
    p = _v2_patch_store_path(project_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.parent / f"{p.stem}.{os.getpid()}.tmp.npz"
    _np.savez_compressed(tmp, **store)
    tmp.replace(p)


def _v2_load_or_backfill_patch_tokens(
    project_id: str,
) -> tuple[dict[str, list[tuple["np.ndarray", "np.ndarray"]]],
           dict[str, list[tuple["np.ndarray", "np.ndarray"]]]]:
    """Walk the project's references and gather per-detection patch
    tokens grouped by label, for BOTH encoders. Detections without
    a stored patch grid get encoded inline against the same crop
    the pooled embeddings used. Stores the result back to the
    per-project NPZ so subsequent calls are pure reads.

    Returns (dino_by_label, siglip_by_label) where each is
    {label_lower: [(tokens, fg_mask), ...]}.

    Both encoders share the same per-project NPZ; DINOv2 keys are
    `<ref_id>__<det_idx>__tokens` / `__fg`, SigLIP keys are
    `<ref_id>__<det_idx>__siglip_tokens` / `__siglip_fg`. Mixed
    is fine — older NPZs without SigLIP keys backfill SigLIP only
    on the next call without re-running DINOv2 encoding.

    Returns ({}, {}) when V2_PATCH_MATCH is off or DINOv2 isn't
    loaded — callers fall back to pooled scoring.
    """
    import numpy as _np
    if not _v2_patch_match_enabled():
        return {}, {}
    import v2_dinov2 as _v2d
    if not _v2d.is_loaded():
        return {}, {}
    import v2_siglip as _v2s
    siglip_active = _v2s.is_loaded()

    manifest = load_manifest(project_id) or {}
    refs = manifest.get("references") or []
    refs_dir = project_dir(project_id) / "references"
    store = _v2_load_patch_store(project_id)

    by_label_dino: dict[str, list[tuple[_np.ndarray, _np.ndarray]]] = {}
    by_label_siglip: dict[str, list[tuple[_np.ndarray, _np.ndarray]]] = {}
    # (ri, di, label, box, mask_polys, need_dino, need_siglip)
    pending_by_ref: dict[int, list[tuple[int, str, list[float], list | None, bool, bool]]] = {}

    for ri, ref in enumerate(refs):
        for di, d in enumerate(ref.get("detections") or []):
            label = (d.get("label") or "").strip().lower()
            if not label:
                continue
            ref_id = ref.get("id")
            if not ref_id:
                continue

            d_tokens = store.get(f"{ref_id}__{di}__tokens")
            d_fg = store.get(f"{ref_id}__{di}__fg")
            d_have = d_tokens is not None and d_fg is not None and d_tokens.shape[0] > 0
            if d_have:
                by_label_dino.setdefault(label, []).append((d_tokens, d_fg.astype(bool)))

            s_have = False
            if siglip_active:
                s_tokens = store.get(f"{ref_id}__{di}__siglip_tokens")
                s_fg = store.get(f"{ref_id}__{di}__siglip_fg")
                s_have = s_tokens is not None and s_fg is not None and s_tokens.shape[0] > 0
                if s_have:
                    by_label_siglip.setdefault(label, []).append((s_tokens, s_fg.astype(bool)))

            need_dino = not d_have
            need_siglip = siglip_active and not s_have
            if not (need_dino or need_siglip):
                continue

            box = d.get("box") or []
            if not (isinstance(box, list) and len(box) == 4):
                continue
            mask_polys = None
            m = d.get("mask")
            if isinstance(m, dict):
                mask_polys = m.get("polygons")
            pending_by_ref.setdefault(ri, []).append(
                (di, label, [float(x) for x in box], mask_polys, need_dino, need_siglip)
            )

    if pending_by_ref:
        added = 0
        for ri, items in pending_by_ref.items():
            ref = refs[ri]
            ref_id = ref.get("id")
            fn = ref.get("filename")
            if not (ref_id and fn):
                continue
            p = refs_dir / fn
            if not p.exists():
                continue
            try:
                image_pil = PILImage.open(p).convert("RGB")
            except Exception as e:
                print(f"[v2-patches] open failed {p}: {e}")
                continue
            W, H = image_pil.size
            for di, label, box, mask_polys, need_dino, need_siglip in items:
                x0 = max(0, int(round(box[0])))
                y0 = max(0, int(round(box[1])))
                x1 = min(W, int(round(box[2])))
                y1 = min(H, int(round(box[3])))
                if x1 - x0 < 4 or y1 - y0 < 4:
                    continue
                # Same inpainted square for both encoders so the
                # foreground mask + crop framing match between them.
                clean = _v2d.inpaint_bbox_crop(image_pil, (x0, y0, x1, y1), mask_polys)
                square = _v2d.center_square_crop(clean)
                if need_dino:
                    tokens, fg = _v2d.encode_image_patches(square)
                    if tokens.shape[0] > 0:
                        store[f"{ref_id}__{di}__tokens"] = tokens.astype(_np.float16)
                        store[f"{ref_id}__{di}__fg"] = fg
                        by_label_dino.setdefault(label, []).append((tokens, fg))
                        added += 1
                if need_siglip:
                    s_tokens, s_fg = _v2s.encode_image_patches(square)
                    if s_tokens.shape[0] > 0:
                        store[f"{ref_id}__{di}__siglip_tokens"] = s_tokens.astype(_np.float16)
                        store[f"{ref_id}__{di}__siglip_fg"] = s_fg
                        by_label_siglip.setdefault(label, []).append((s_tokens, s_fg))
                        added += 1
        if added > 0:
            try:
                _v2_save_patch_store(project_id, store)
                print(f"[v2-patches] backfilled {added} array(s) for project={project_id}")
            except Exception as e:
                print(f"[v2-patches] save failed (in-memory result still valid): {e}")

    counts_dino = {k: len(v) for k, v in by_label_dino.items()}
    counts_siglip = {k: len(v) for k, v in by_label_siglip.items()}
    print(
        f"[v2-patches] project={project_id} dino={counts_dino} siglip={counts_siglip}"
    )
    return by_label_dino, by_label_siglip


def _v2_score_labels_patchwise(
    q_tokens: "np.ndarray",
    q_fg: "np.ndarray",
    refs_by_label_patches: dict,
    *,
    top_k_query: int = _PATCH_TOP_K_QUERY,
) -> dict[str, float]:
    """Patch-level scoring. For each query foreground patch, find its
    best cosine match across ALL the label's reference foreground
    patches. The label's score is the mean of the top-K query
    patches' best matches.

    Why top-K instead of mean-all: most query patches are body /
    background-ish features that match every label about equally.
    The discriminative work happens in a small subset (ear tips,
    fur boundaries, asymmetric features). Top-K aggregation lets
    those decisive matches dominate without the noise floor of the
    weak ones diluting the score.

    Args:
      q_tokens: (P_q, D) — query patches, L2-normalised
      q_fg:     (P_q,) bool — query foreground mask
      refs_by_label_patches: {label_lower: [(tokens, fg), ...]}
      top_k_query: how many of the strongest query-patch best-matches
                   to mean. Higher = smoother score, lower = more
                   sensitive to localised cues.

    Returns {label_lower: score}, dot-product cosine similarities
    in [0, 1] (higher = more similar).
    """
    import numpy as _np

    # Filter query to its foreground patches. If none, fall back to
    # all patches so we don't return an empty dict on a degenerate
    # SAM mask.
    q = q_tokens[q_fg] if bool(q_fg.any()) else q_tokens
    if q.shape[0] == 0:
        return {}

    out: dict[str, float] = {}
    for label, refs in refs_by_label_patches.items():
        # Concatenate ALL reference foreground patches for this
        # label into one big (sum_n_fg, D) matrix so the per-query-
        # patch best-match is a single .max along axis=1.
        ref_chunks: list[_np.ndarray] = []
        for r_tokens, r_fg in refs:
            r_fg_bool = r_fg.astype(bool) if r_fg.dtype != bool else r_fg
            sel = r_tokens[r_fg_bool] if bool(r_fg_bool.any()) else r_tokens
            if sel.shape[0] > 0:
                # Cast to float32 for the matmul — patch tokens may
                # be stored as float16 to halve disk.
                ref_chunks.append(sel.astype(_np.float32, copy=False))
        if not ref_chunks:
            continue
        R = _np.concatenate(ref_chunks, axis=0)  # (sum_n_fg_ref, D)

        # Per-query-patch best-match: (P_q, D) @ (D, sum_n_fg_ref)
        # → (P_q, sum_n_fg_ref) → max over refs → (P_q,).
        sims_per_q = (q.astype(_np.float32, copy=False) @ R.T).max(axis=1)

        # Top-K query aggregation.
        k = min(top_k_query, int(sims_per_q.shape[0]))
        if k <= 0:
            continue
        top = _np.partition(sims_per_q, -k)[-k:]
        out[label] = float(top.mean())

    return out


# ─── Per-project Fisher reweighting (Level 1 metric adaptation) ──────────────
# For each project we compute a per-dimension weight vector from
# the reference embeddings: `w[i] = sqrt(between_var[i] / within_var[i])`
# where between_var captures how much dim i differs across class
# centroids and within_var captures how much it varies within the
# same class. Applied as `e_adapted = e * w / norm(e * w)`, the
# cosine similarity in the reweighted space puts more weight on
# dimensions that separate THIS project's classes and less on
# dimensions that vary uniformly across all of them.
#
# Practical effect: the embedding "auto-adapts" to the project's
# specific classes. Hare/rabbit gets ear-tip-related dims boosted;
# alpaca/llama gets neck-shape dims boosted. Same idea as classical
# Fisher LDA but works for the 2-class case (where Fisher LDA
# collapses to a single 1-D direction and loses most of the
# original embedding's expressiveness). Diagonal-only — ignores
# cross-dimension correlations, which is fine because DINOv2 features
# are roughly decorrelated by training already.
#
# Cached in-memory keyed on the ref-set fingerprint so the 4-second
# poll doesn't recompute. ~5 ms to compute on a typical 10-ref project.

_FISHER_CACHE: dict[str, "np.ndarray"] = {}


def _v2_fisher_cache_key(by_label_arr: dict, project_id: str | None = None) -> str:
    """Fingerprint of a per-label ref stack so we can detect when a
    project's reference set changed and recompute the Fisher weights.
    Hashes the per-label sample counts plus the first vector of each
    label — cheap, sufficient to invalidate when refs are added /
    removed / re-embedded."""
    import hashlib
    h = hashlib.blake2b(digest_size=16)
    if project_id:
        h.update(project_id.encode())
    for label in sorted(by_label_arr.keys()):
        arr = by_label_arr[label]
        if arr is None or arr.shape[0] == 0:
            continue
        h.update(label.encode())
        h.update(str(arr.shape).encode())
        # First row's first 32 floats — captures any re-embedding.
        h.update(arr[0, :32].tobytes())
    return h.hexdigest()


def _v2_compute_fisher_weights(
    by_label_arr: dict,
    *,
    eps: float = 1e-3,
    cap: float = 5.0,
) -> "np.ndarray | None":
    """Per-dimension Fisher reweighting from per-label reference embeddings.

    Returns a (D,) np.float32 vector of per-dim weights, or None when
    there's not enough data (need ≥2 classes with ≥2 refs each).

    Diagonal Fisher: w[i] = sqrt(between_i / (within_i + eps)) clamped
    at `cap` so a single under-determined within-class variance can't
    blow one dimension's weight out of proportion. Re-normalised so
    sum(w**2) = D (preserves the embedding's scale; cosine sim values
    stay roughly in [0, 1]).

    eps stabilises within-class variance estimates from small ref sets.
    cap=5.0 means the most-discriminative dim can be at most 5× the
    mean weight.
    """
    import numpy as _np
    labels = [k for k, arr in by_label_arr.items() if arr is not None and arr.shape[0] > 0]
    if len(labels) < 2:
        return None
    # Need at least 2 refs per class for a within-class variance
    # estimate. Skip projects where any class has only 1 ref.
    if any(by_label_arr[k].shape[0] < 2 for k in labels):
        return None

    # Compute per-class mean and overall mean.
    class_means: dict[str, _np.ndarray] = {k: by_label_arr[k].mean(axis=0) for k in labels}
    all_arr = _np.concatenate([by_label_arr[k] for k in labels], axis=0)
    overall_mean = all_arr.mean(axis=0)
    D = int(all_arr.shape[1])
    n_total = int(all_arr.shape[0])

    # Per-dim between-class variance: weighted spread of class means
    # around the overall mean. Higher = dimension separates classes.
    between = _np.zeros(D, dtype=_np.float64)
    for k in labels:
        n_k = int(by_label_arr[k].shape[0])
        diff = class_means[k] - overall_mean
        between += n_k * (diff ** 2)
    between /= max(n_total, 1)

    # Per-dim within-class variance: how much each dim wobbles within
    # a single class. Higher = noisier dim, lower = stable feature.
    within = _np.zeros(D, dtype=_np.float64)
    for k in labels:
        arr = by_label_arr[k]
        diff = arr - class_means[k]
        within += (diff ** 2).sum(axis=0)
    within /= max(n_total - len(labels), 1)

    # Fisher score per dim: between / (within + eps), then sqrt for
    # use as a multiplicative weight (since cosine works in linear
    # space, sqrt-weighting scales each dim's contribution to the
    # dot product proportionally to its score).
    score = between / (within + eps)
    # Clamp so a few dims with effectively-zero within-variance don't
    # dominate the embedding entirely.
    score_max = float(score.max())
    if score_max > 0:
        score = score / score_max  # normalise to [0, 1]
    weights = _np.sqrt(score).astype(_np.float32)
    weights = _np.minimum(weights, cap)

    # Re-normalise so the average weight stays at 1 (preserves the
    # embedding's overall magnitude after the element-wise multiply,
    # which keeps cosine sims in their familiar [0, 1] range).
    mean_w = float(weights.mean())
    if mean_w < 1e-6:
        return None
    weights = weights / mean_w
    return weights


def _v2_get_fisher_weights(
    by_label_arr: dict,
    project_id: str | None = None,
) -> "np.ndarray | None":
    """Cached wrapper around _v2_compute_fisher_weights. Re-uses the
    same weights across the 4-second poll cycle and only recomputes
    when the ref set's fingerprint changes (refs added / removed /
    re-embedded)."""
    if not by_label_arr:
        return None
    key = _v2_fisher_cache_key(by_label_arr, project_id)
    cached = _FISHER_CACHE.get(key)
    if cached is not None:
        return cached
    w = _v2_compute_fisher_weights(by_label_arr)
    if w is not None:
        # Cap cache size at 256 entries — projects rarely change refs
        # more than a handful of times so this should never fill up.
        if len(_FISHER_CACHE) > 256:
            _FISHER_CACHE.clear()
        _FISHER_CACHE[key] = w
    return w


def _v2_apply_fisher_to_arr(arr: "np.ndarray", weights: "np.ndarray") -> "np.ndarray":
    """Element-wise weight + per-row L2 renormalise. Works on (D,) or
    (N, D). Output stays L2-normalised so cosine sim still equals dot
    product downstream."""
    import numpy as _np
    if arr.ndim == 1:
        weighted = arr * weights
        n = float(_np.linalg.norm(weighted))
        if n < 1e-8:
            return arr
        return weighted / n
    # (N, D)
    weighted = arr * weights[None, :]
    norms = _np.linalg.norm(weighted, axis=1, keepdims=True)
    norms = _np.maximum(norms, 1e-8)
    return weighted / norms


def _v2_apply_fisher_to_refs(by_label_arr: dict, weights: "np.ndarray") -> dict:
    """Apply Fisher weights to a stacked-refs dict. Returns a new dict
    with the same keys and (N, D) shape per value."""
    return {k: _v2_apply_fisher_to_arr(arr, weights) for k, arr in by_label_arr.items()}


def _v2_compute_class_thresholds(
    by_label_arr: dict,
    *,
    floor: float = 0.20,
    cap: float = 0.40,
    sigma_mult: float = 3.0,
    fallback: float = 0.40,
) -> dict[str, float]:
    """Per-class accept/reject threshold from the leave-one-out
    self-score distribution.

    For each label with ≥3 references we compute the LOO cosine of
    each ref against the centroid of the other refs in its class
    and set:

        threshold = max(floor, min(cap, mean - sigma_mult * std))

    Notably the cap == the global fallback (0.40). The threshold
    can only go DOWN from the default, never up — a tightly-
    clustered class keeps the global 0.40 instead of jumping to
    0.70+. The first version inverted this and ended up rejecting
    every legitimate query because the LOO distribution is in-
    sample optimistic: refs are far closer to their own class's
    centroid than out-of-sample queries are, so a threshold tuned
    to the LOO band rejects normal queries that should pass.
    Loosely-clustered classes (refs span varied poses — "horse":
    mean ≈ 0.55, std ≈ 0.10) drop the threshold to ~0.25 so the
    model isn't punishing legitimate variety. That asymmetry —
    relax for loose, hold the line for tight — is the actual win.

    Classes with <3 refs get `fallback` directly; the LOO
    distribution isn't reliable from 1-2 samples.
    """
    import numpy as _np

    out: dict[str, float] = {}
    for key, arr in by_label_arr.items():
        if arr is None or arr.shape[0] < 3:
            out[key] = fallback
            continue

        # LOO self-scores: (sum_centroid - q) / (n - 1), L2-norm,
        # dot with q. Vectorised so we don't loop per ref.
        n = int(arr.shape[0])
        total = arr.sum(axis=0)  # (D,)
        loo_centroids = (total - arr) / (n - 1)  # (N, D)
        norms = _np.linalg.norm(loo_centroids, axis=1, keepdims=True)
        norms = _np.maximum(norms, 1e-8)
        loo_centroids = loo_centroids / norms
        # Dot each ref with its leave-one-out centroid → (N,)
        self_scores = (loo_centroids * arr).sum(axis=1)

        mean = float(self_scores.mean())
        std = float(self_scores.std())
        threshold = mean - sigma_mult * std
        threshold = max(floor, min(cap, threshold))
        out[key] = float(threshold)

    return out


def _v2_compute_reference_quality(refs: list[dict], by_label_arr: dict) -> dict:
    """Per-detection reference-quality scores via leave-one-out
    cross-validation. For each labelled detection we ask:

      * `self_score`  — cosine sim to the centroid of OTHER refs with
                        the same label. High = this ref looks like the
                        rest of its class. Low = outlier.
      * `other_score` — max cosine sim to the centroids of OTHER labels.
                        High means the ref looks more like a different
                        class than its assigned one.
      * `quality`     — `self_score - other_score`. Above ~0.05 is
                        cleanly separable; near zero or negative means
                        this ref is hurting the centroid.
      * `warning`     — short string when the ref is suspicious:
          - "outlier" when self_score < 0.5 (doesn't even look like
            its own class).
          - "looks like other class" when other_score >= self_score
            (closer to a different class than its assigned one).
          - "only ref for class" when there's no leave-one-out
            comparison possible (1 ref per label gives self_score=None).

    Both q-vector and centroids are L2-normalised, so the dot product
    IS cosine similarity. Uses DINOv2 embeddings (the primary
    discriminator); SigLIP-based quality is left for a follow-up.

    Returns {ref_id: {detection_idx: {self_score, other_score,
    other_label, quality, warning}}}. Refs with no labelled
    detections or no embedding don't appear in the result.
    """
    import numpy as _np

    # Pre-compute (sum, count) per label so the leave-one-out
    # centroid is just (sum - q) / (count - 1). O(N) memory.
    label_sum: dict[str, _np.ndarray] = {}
    label_count: dict[str, int] = {}
    for lab, arr in by_label_arr.items():
        if arr is None or arr.shape[0] == 0:
            continue
        label_sum[lab] = arr.sum(axis=0)
        label_count[lab] = int(arr.shape[0])

    # Full centroids (used for the other-class scoring side — those
    # don't change per detection, so no leave-one-out needed).
    label_centroid: dict[str, _np.ndarray] = {}
    for lab, arr in by_label_arr.items():
        if arr is None or arr.shape[0] == 0:
            continue
        c = arr.mean(axis=0)
        n = float(_np.linalg.norm(c))
        if n < 1e-8:
            continue
        label_centroid[lab] = c / n

    out: dict[str, dict] = {}
    for ref in refs:
        ref_id = ref.get("id")
        if not ref_id:
            continue
        ref_quality: dict[int, dict] = {}
        for di, d in enumerate(ref.get("detections") or []):
            label = (d.get("label") or "").strip().lower()
            if not label:
                continue
            emb = d.get("embedding")
            if not (isinstance(emb, list) and len(emb) > 0):
                continue
            q = _np.asarray(emb, dtype=_np.float32)

            # Leave-one-out self-score.
            self_score: float | None = None
            if label in label_count and label_count[label] > 1:
                loo_sum = label_sum[label] - q
                loo_count = label_count[label] - 1
                loo_centroid = loo_sum / loo_count
                n = float(_np.linalg.norm(loo_centroid))
                if n > 1e-8:
                    self_score = float((loo_centroid / n) @ q)

            # Max sim to OTHER labels' centroids.
            other_score: float | None = None
            other_label: str | None = None
            for ol, oc in label_centroid.items():
                if ol == label:
                    continue
                s = float(oc @ q)
                if other_score is None or s > other_score:
                    other_score = s
                    other_label = ol

            quality = (
                self_score - other_score
                if (self_score is not None and other_score is not None)
                else None
            )

            warning: str | None = None
            if self_score is None:
                warning = "only ref for class"
            elif self_score < 0.5:
                warning = "outlier"
            elif other_score is not None and other_score >= self_score:
                warning = "looks like other class"

            ref_quality[di] = {
                "self_score": round(self_score, 4) if self_score is not None else None,
                "other_score": round(other_score, 4) if other_score is not None else None,
                "other_label": other_label,
                "quality": round(quality, 4) if quality is not None else None,
                "warning": warning,
            }
        if ref_quality:
            out[ref_id] = ref_quality
    return out


@app.get(
    "/api/v2/projects/{project_id}/reference_quality",
    dependencies=[Depends(require_project_owner)],
)
async def v2_reference_quality(project_id: str):
    """Surface per-reference quality scores so the UI can flag
    suspicious entries the user should review. See
    `_v2_compute_reference_quality` for the scoring rules.

    Returns:
        {
          "references": {
            "<ref_id>": {
              "<det_idx>": {
                "self_score": float, "other_score": float,
                "other_label": str, "quality": float,
                "warning": "outlier" | "looks like other class"
                          | "only ref for class" | null
              }, ...
            }, ...
          }
        }

    Implicitly triggers the centroid backfill (loads embeddings;
    re-encodes any at older EMBED_VERSION) so the quality scores
    are always computed against current-version vectors.
    """
    if not project_dir(project_id).exists():
        raise HTTPException(404)
    try:
        by_label, _by_label_siglip, _dirty = _v2_load_or_backfill_reference_embeddings(project_id)
    except Exception as e:
        print(f"[v2-ref-quality] load failed: {e}")
        return {"references": {}}
    arr = _v2_stack_refs(by_label)
    manifest = load_manifest(project_id) or {}
    refs = manifest.get("references") or []
    quality = _v2_compute_reference_quality(refs, arr)
    return {"references": quality}


def _v2_compute_label_centroids(by_label: dict) -> dict[str, list[float]]:
    """Mean-pool per-label embeddings then L2-normalise so cosine ==
    dot product downstream. Returns {label_lower: vec_list}.

    NOTE: superseded by `_v2_score_labels_knn` for the actual resolve
    path — the resolver no longer reduces a label's references to a
    single centroid because that loses multimodal structure (e.g. a
    "horse standing" label with side-view AND front-view references
    has its centroid land between the two clusters and underscores
    queries that match either one cleanly). Kept around in case any
    debugging tool wants the legacy centroid view of the data."""
    import numpy as _np
    out: dict[str, list[float]] = {}
    for key, vecs in by_label.items():
        if not vecs:
            continue
        m = _np.mean(_np.stack(vecs, axis=0), axis=0)
        n = float(_np.linalg.norm(m))
        if n < 1e-8:
            continue
        out[key] = (m / n).astype(_np.float32).tolist()
    return out


# Top-K nearest-neighbour scoring is the V2 default. For each label
# we score the query against every reference embedding and average
# the K highest dot products. This handles multimodal label
# distributions — references that cluster into 2-3 visually distinct
# modes (different poses, angles, scales) — without forcing a single
# averaged centroid that sits between modes.
#
# Fewer than K references → all of them are used (equivalent to
# centroid matching at N=1, mean-of-2 at N=2, etc.).
KNN_TOP_K = 3


def _v2_stack_refs(by_label: dict) -> dict[str, "np.ndarray"]:
    """Pre-stack per-label reference embeddings into (N, D) arrays
    so the resolver can score detections without restacking on every
    call."""
    import numpy as _np
    out: dict[str, _np.ndarray] = {}
    for key, vecs in by_label.items():
        if not vecs:
            continue
        out[key] = _np.stack(
            [_np.asarray(v, dtype=_np.float32) for v in vecs], axis=0,
        )
    return out


def _v2_score_labels_knn(
    q: "np.ndarray",
    refs_by_label_arr: dict,
    k: int = KNN_TOP_K,
) -> dict[str, float]:
    """Top-K kNN scoring across labels. Returns {label_key: score}
    where score is the mean of the K highest cosine sims between q
    and that label's references. Both q and the references are
    L2-normalised in v2_dinov2.encode_*, so the dot product `refs @ q`
    here IS cosine similarity — no extra normalisation needed.

    Cosine is the right metric for high-dim DINOv2 embeddings (1024d
    here) — L2 distance suffers from the curse of dimensionality
    (all points become roughly equidistant) but cosine just measures
    angle, which is what we care about for L2-normalised features.

    Superseded by `_v2_score_labels_knn_weighted` below for the
    actual resolve path. Kept as the simple-mean baseline for quick
    A/B comparison via env var.
    """
    import numpy as _np
    sims: dict[str, float] = {}
    for key, refs in refs_by_label_arr.items():
        if refs is None or refs.shape[0] == 0:
            continue
        all_sims = refs @ q  # (N,)
        kk = min(k, int(all_sims.shape[0]))
        if kk <= 0:
            continue
        # np.partition is O(N) for finding the top-K (no full sort).
        top = _np.partition(all_sims, -kk)[-kk:]
        sims[key] = float(top.mean())
    return sims


# Tiny floor on cosine distance so the inverse-distance weight stays
# finite when a query is essentially identical to a reference. 1e-3
# means a perfect match (sim=1.0, d=0) gets capped at weight=1000 —
# more than enough to dominate over vague matches without blowing up
# the arithmetic on rounding noise.
_KNN_DIST_EPS = 1e-3


def _v2_score_labels_knn_weighted(
    q: "np.ndarray",
    refs_by_label_arr: dict,
    k: int = KNN_TOP_K,
) -> dict[str, float]:
    """Per-label weighted top-K kNN. Returns {label_key: score} where
    score is the weighted mean of the top-K cosine sims for that
    label, weighted by inverse cosine distance (1 / max(1 - sim, eps)).

    The 1/d weighting lets a single very-close reference dominate
    over several vague matches: top-3 of [0.85, 0.50, 0.40] gives
    weighted mean ≈ 0.71 vs simple mean 0.58. Across labels, the
    closest-single-reference label wins, which is typically the
    right answer for fine-grained pairs (one perfect hare beats
    two so-so rabbits).

    Both q and refs are L2-normalised, so dot product is cosine.
    """
    import numpy as _np
    sims: dict[str, float] = {}
    eps = _KNN_DIST_EPS
    for key, refs in refs_by_label_arr.items():
        if refs is None or refs.shape[0] == 0:
            continue
        all_sims = refs @ q  # (N,)
        kk = min(k, int(all_sims.shape[0]))
        if kk <= 0:
            continue
        top = _np.partition(all_sims, -kk)[-kk:]
        dists = _np.maximum(1.0 - top, eps)
        weights = 1.0 / dists
        sims[key] = float((top * weights).sum() / weights.sum())
    return sims


def _v2_filename_labels(filename: str | None, project_labels: list[str]) -> set[str]:
    """Return the lowercased project labels whose tokens appear as a
    contiguous subsequence in the filename. Both filename and label
    are tokenised on non-alphanumeric so "high vis vest" matches
    "horse_high-vis-vest_05.jpg" and "stop sign" matches
    "stop_sign_intersection.png" but not "stopwatch.jpg".
    """
    if not filename or not project_labels:
        return set()
    import re as _re
    base = _re.sub(r"\.[A-Za-z0-9]+$", "", str(filename))
    fname_tokens = _re.findall(r"[a-z0-9]+", base.lower())
    if not fname_tokens:
        return set()
    matched: set[str] = set()
    for label in project_labels:
        label_norm = str(label).strip().lower()
        if not label_norm:
            continue
        label_tokens = _re.findall(r"[a-z0-9]+", label_norm)
        if not label_tokens:
            continue
        L = len(label_tokens)
        for i in range(len(fname_tokens) - L + 1):
            if fname_tokens[i:i + L] == label_tokens:
                matched.add(label_norm)
                break
    return matched


# Margin tolerance for the filename tiebreak. When the gap between
# the top two label sims is smaller than this, the filename's
# explicit mention of one of them gets to nudge that label up by an
# equal amount — enough for a fence-tied second-place to leapfrog
# first place, but small enough that confident decisions are
# untouched.
FILENAME_HINT_TOLERANCE = 0.05


def _v2_apply_filename_hint(
    sims: dict[str, float],
    filename_labels: set[str],
    *,
    tolerance: float = FILENAME_HINT_TOLERANCE,
) -> dict[str, float]:
    """Conditionally bias label sims toward labels mentioned in the
    image filename. Returns a NEW dict (caller-friendly).

    Only fires when:
      - at least one filename-mentioned label is in the sims dict
      - the top-1 / top-2 margin is less than ``tolerance`` (the "on
        the fence" condition the user asked for; ignores cases where
        the embedding is already confident)

    On a hit, every filename-mentioned label gets +``tolerance`` so a
    second-place tied label can leapfrog. Filename labels not in sims
    (e.g., user named the file with a label that isn't in the project
    label set) are ignored.
    """
    if not filename_labels or not sims:
        return sims
    if len(sims) < 2:
        return sims
    relevant = filename_labels & set(sims.keys())
    if not relevant:
        return sims
    sorted_vals = sorted(sims.values(), reverse=True)
    margin = sorted_vals[0] - sorted_vals[1]
    if margin >= tolerance:
        return sims
    return {k: v + (tolerance if k in relevant else 0.0) for k, v in sims.items()}


def _v2_score_labels_centroid(
    q: "np.ndarray",
    refs_by_label_arr: dict,
) -> dict[str, float]:
    """Centroid scoring: reduce each label's references to a single
    L2-normalised mean, then dot the query against it. Sharper
    contrast than kNN for unimodal label distributions (general
    datasets — PPE, vehicles, signage) where every reference for a
    label looks visually similar; the centroid acts as a denoiser
    and produces a clear peak-vs-valley between the right label and
    the wrong ones. Used for general datasets while kNN serves the
    multimodal specific-dataset case."""
    import numpy as _np
    sims: dict[str, float] = {}
    for key, refs in refs_by_label_arr.items():
        if refs is None or refs.shape[0] == 0:
            continue
        c = refs.mean(axis=0)
        n = float(_np.linalg.norm(c))
        if n < 1e-8:
            continue
        c = c / n
        sims[key] = float(c @ q)
    return sims


def _v2_resolve_label(
    embedding: list[float],
    gd_label: str | None,
    refs_by_label: dict,
    label_display: dict[str, str],
    *,
    score_mode: str = "centroid",
    filename_labels: set[str] | None = None,
    gd_score: float | None = None,
    # VLM is informational by default; only consulted on ambiguous
    # cases as a tiebreaker / corroborating signal for relabels.
    # Plain VLM "votes" are NOT aggregated into the decision — the
    # model has been observed picking visually salient occluders
    # over the actual labeled object often enough that we treat its
    # output as a discriminator, not a primary vote.
    vlm_label: str | None = None,
    vlm_score: float | None = None,
    # Reject thresholds.
    reject_gd_threshold: float = 0.15,
    reject_embed_threshold: float = 0.4,
    # Rescue: low-confidence GD with very-high embed -> accept w/ embed.
    gd_low_embed_rescue_min: float = 0.6,
    # Strong-embedding override (rule 0): embed sim ≥ this trumps
    # GD when GD itself is uncertain (< gd_moderate_min). Below the
    # very-strong threshold (rule 0c), so a moderately-confident GD
    # still gets a chance to defend its label via the VLM tiebreak.
    embed_strong_min: float = 0.7,
    # Very-strong embedding override (rule 0c): embed sim ≥ this
    # wins outright over any GD score, no checks, no VLM. Treated
    # as overwhelming visual evidence.
    embed_very_strong_min: float = 0.8,
    # GD-confident lock (rule 0b): GD scoring at or above this
    # threshold is decisive on its own — accept GD's label outright
    # with no VLM, no embed checks. Catches the "person wearing
    # high-vis vest" case where embed confidently picks vest from
    # the contaminated centroid but GD correctly identified the
    # person at 0.5.
    gd_confident_min: float = 0.50,
    # GD-moderate lock (rule 0b'): GD in [gd_moderate_min,
    # gd_confident_min) is accepted ONLY when VLM corroborates
    # GD's label. The soft-accept zone — GD alone isn't decisive
    # but a VLM second opinion locks it in.
    gd_moderate_min: float = 0.40,
    # Confusion reject ceiling (rule 0d): when GD is above the
    # moderate floor AND embed's nearest centroid is a DIFFERENT
    # label with sim below this threshold, neither signal is
    # firmly committed to its pick. The crop is confusing, so
    # reject rather than commit to GD's label.
    confusion_embed_max: float = 0.50,
    # Combined-low reject.
    combined_reject_gd_max: float = 0.25,
    combined_reject_embed_max: float = 0.48,
    # Relabel rule (stricter than the previous "best_sim > gd_sim").
    # All four must hold:
    #   1. best_sim >= relabel_class_min (absolute floor, per-class
    #      knob exposed for future tuning — 0.5 default for now)
    #   2. embed_margin >= relabel_embed_margin (top1 - top2 >= 0.05
    #      so two near-equal centroids don't trigger a flip)
    #   3. best_sim - sim_for_gd_label >= relabel_gd_embed_margin
    #      (top1 must clearly beat GD's centroid by 0.05)
    #   4. gd_score < relabel_gd_low_max  OR  vlm corroborates
    #      (either GD itself wasn't confident, or the VLM agrees
    #      with the embedding's pick — pure embedding override
    #      isn't allowed when GD is strong AND VLM disagrees.)
    # relabel_gd_low_max aligns with ambiguous_gd_max (0.30) — any
    # GD score below the ambiguity threshold is by definition not
    # confident enough to block an embed override. Margins were
    # halved (0.08→0.05 and 0.10→0.05) after a high-vis at embed
    # 0.603 failed to relabel — face_mask's centroid sim sat close
    # enough to high_vis that the older 0.10 gap couldn't open up.
    relabel_class_min: float = 0.5,
    relabel_embed_margin: float = 0.05,
    relabel_gd_embed_margin: float = 0.05,
    relabel_gd_low_max: float = 0.30,
    # Ambiguity flags (display-only). Used by the popup to mark
    # detections worth eyeballing, and as the gate for VLM
    # tiebreaker between GD's label and embed's nearest centroid
    # when the strict relabel rule above didn't fire.
    ambiguous_gd_max: float = 0.30,
    ambiguous_embed_margin_min: float = 0.08,
) -> dict:
    """Apply the V2 relabel/reject rules for one detection.

    Decision tree (first match wins; see param docstrings for the
    exact thresholds):

      0c. Very-strong embed override: embed nearest sim ≥
         embed_very_strong_min (0.80) — accept embed outright,
         beats even a confident GD.
      0d. Confusion reject: gd_score ≥ gd_moderate_min (0.40) AND
         embed-nearest is a DIFFERENT label with sim <
         confusion_embed_max (0.50). Neither signal is firmly
         committed; reject rather than commit to GD's pick.
      0b. GD-confident lock: gd_score ≥ gd_confident_min (0.50) —
         accept GD's label outright. No VLM, no embed checks.
      0b'. GD-moderate lock: gd_score in [gd_moderate_min (0.40),
         gd_confident_min (0.50)) AND VLM agrees with GD — accept
         GD. The soft zone where GD alone isn't decisive.
      0a. Three-way disagreement reject: GD, embed-nearest, and VLM
         each picked a different label AND gd_score is below
         gd_moderate_min (0.40) — no signal corroborates any other
         and GD couldn't defend itself, so reject rather than
         coin-flip. A moderately-confident GD is allowed to keep
         its label even with full disagreement (PPE-wearing
         people: each signal locks onto a different visible item).
      0. Strong-embed override: embed nearest sim ≥ embed_strong_min
         (default 0.7) AND GD < gd_moderate_min (0.40) — trust the
         embedding when GD couldn't defend its own label. Defers
         when GD sits in the moderate zone so rule 0b' gets a shot.
      1. Rescue: gd very low + embedding very confident → accept
         with embed's label.
      2. GD-only reject: gd very low and embedding couldn't rescue.
      3. Combined-low reject: both gd and embed lukewarm.
      4. Strict relabel: embed clears class floor, has clear margin
         over second-best AND over GD's centroid, AND either GD
         was uncertain OR the VLM corroborates embed's pick.
         (This is the only path that overrides GD's label.)
      5. Embed-low reject: final-label centroid sim < 0.40.
      6. VLM tiebreaker: when the case is ambiguous (low gd, or
         gd != embed_best, or top1-top2 margin < 0.08) AND VLM
         picks one of {gd_label, embed_nearest_label} that isn't
         the current pred_label, switch to the VLM-picked option.
         VLM responses outside those two candidates are recorded
         on the response but DO NOT override pred_label.
      7. Otherwise accept with current pred_label.
    """
    import numpy as _np

    # Score per label: centroid for general (sharp peak/valley
    # contrast), kNN for specific (handles multimodal references).
    sims: dict[str, float] = {}
    if embedding and refs_by_label:
        q = _np.asarray(embedding, dtype=_np.float32)
        if score_mode == "knn":
            sims = _v2_score_labels_knn(q, refs_by_label)
        else:
            sims = _v2_score_labels_centroid(q, refs_by_label)

    # Filename hint: when the user named the image with one of the
    # project labels and the top-2 sims are within tolerance, bias
    # toward the filename label. Easy free signal.
    if filename_labels:
        sims = _v2_apply_filename_hint(sims, filename_labels)

    sorted_sims_desc = sorted(sims.values(), reverse=True)
    # Margin is top1 - top2. With <2 labels in sims (e.g. only one
    # label has references uploaded yet) there's no second-best to
    # subtract from, so the margin is undefined — return 0.0 rather
    # than the top1 sim itself, which was misleading the FE into
    # rendering a single sim's value as if it were the margin.
    embed_margin = (
        sorted_sims_desc[0] - sorted_sims_desc[1]
        if len(sorted_sims_desc) >= 2 else
        0.0
    )
    best_key = max(sims, key=sims.get) if sims else None
    best_sim = sims[best_key] if best_key else None

    # Normalised lookups for label comparisons.
    gd_key = (gd_label or "").strip().lower() if gd_label else None
    vlm_key = (vlm_label or "").strip().lower() if vlm_label else None
    gd_sim = sims.get(gd_key) if gd_key else None

    # Default plan: keep GD's label.
    pred_label = gd_label
    pred_source: str | None = "gd" if gd_label else None
    pred_key = gd_key

    # Ambiguity flag — exposed on the response and used to gate the
    # VLM tiebreaker.
    ambiguous = False
    if gd_score is not None and gd_score < ambiguous_gd_max:
        ambiguous = True
    if gd_key and best_key and best_key != gd_key:
        ambiguous = True
    if embed_margin < ambiguous_embed_margin_min:
        ambiguous = True

    def _make(reject_reason: str | None, *, rejected: bool, vlm_action: str | None = None) -> dict:
        final_sim = sims.get(pred_key) if pred_key else None
        return {
            "pred_label": pred_label,
            "pred_source": pred_source,
            "embed_nearest_label": label_display.get(best_key, best_key) if best_key else None,
            "embed_nearest_sim": round(float(best_sim), 4) if best_sim is not None else None,
            "embed_sim_for_label": round(float(final_sim), 4) if final_sim is not None else None,
            "embed_margin": round(float(embed_margin), 4),
            "rejected": rejected,
            "reject_reason": reject_reason,
            "ambiguous": ambiguous,
            "vlm_action": vlm_action,
            "sims": {label_display.get(k, k): round(float(v), 4) for k, v in sims.items()},
        }

    # Rule 0c: very-strong embed override. embed_very_strong_min
    # (default 0.80) wins outright over any GD score, no VLM, no
    # margin checks. Treated as overwhelming visual evidence.
    if (
        best_key is not None and best_sim is not None
        and best_sim >= embed_very_strong_min
    ):
        if best_key != gd_key:
            pred_key = best_key
            pred_label = label_display.get(best_key, best_key)
            pred_source = "embed-strong"
        ambiguous = False
        return _make(None, rejected=False)

    # Rule 0d: confusion reject. GD scored above gd_moderate_min
    # (0.40) but embed picked a DIFFERENT label and even its best
    # centroid sim is below confusion_embed_max (0.50). Neither
    # signal is firmly committed — GD recognised something at
    # moderate confidence, embed couldn't match it cleanly to any
    # of the project's labels. Better to reject than commit.
    if (
        gd_score is not None and gd_score >= gd_moderate_min
        and gd_key is not None
        and best_key is not None and best_sim is not None
        and best_key != gd_key
        and best_sim < confusion_embed_max
    ):
        return _make("confusion", rejected=True, vlm_action="confusion-reject")

    # Rule 0b: GD-confident lock. gd_score ≥ gd_confident_min
    # (default 0.50) — accept GD's label outright. No VLM, no embed
    # checks. Catches the "person wearing high-vis vest" case where
    # embed confidently picks vest from a contaminated centroid but
    # GD correctly identified the person at 0.5+.
    if (
        gd_score is not None and gd_score >= gd_confident_min
        and gd_label
    ):
        ambiguous = False
        return _make(None, rejected=False)

    # Rule 0b': GD-moderate lock with VLM agreement. gd_score in
    # [gd_moderate_min, gd_confident_min) is the soft-accept zone —
    # GD alone isn't decisive, but a VLM second opinion locks it in.
    # Won't fire on the preliminary resolve (no VLM yet) so the box
    # stays ambiguous and the lazy-VLM gate calls Qwen-VL; the
    # second pass then evaluates this rule with vlm_label populated.
    if (
        gd_score is not None
        and gd_moderate_min <= gd_score < gd_confident_min
        and gd_key is not None and vlm_key is not None
        and vlm_key == gd_key
    ):
        ambiguous = False
        return _make(None, rejected=False, vlm_action="confirm")

    # No embedding info — fall back to GD-only gating.
    if not sims:
        rejected_gd = gd_score is not None and gd_score < reject_gd_threshold
        return _make("gd" if rejected_gd else None, rejected=rejected_gd)

    # Rule 0a: three-way disagreement reject. When GD, the nearest
    # centroid, AND the VLM each picked a DIFFERENT label, no signal
    # has corroborating support — committing to any one is a coin
    # flip. Reject outright.
    #
    # Gated to gd_score < gd_moderate_min (0.40): a moderately
    # confident GD is plausibly right even when embed and VLM each
    # pick a different label, because for PPE-wearing people each
    # signal often locks onto a different item visible in the same
    # bbox (vest, helmet, mask, glove) while GD correctly
    # identifies the broader subject (person). Only reject as 3-way
    # disagreement when GD itself didn't clear the moderate floor.
    if (
        gd_key and best_key and vlm_key
        and len({gd_key, best_key, vlm_key}) == 3
        and (gd_score is None or gd_score < gd_moderate_min)
    ):
        return _make("disagree", rejected=True, vlm_action="three-way-disagree")

    # Rule 0: strong-embed override (between embed_strong_min and
    # embed_very_strong_min). Fires when (a) GD already agrees, or
    # (b) GD is uncertain (< gd_moderate_min). When GD sits in the
    # moderate zone we DEFER — leave ambiguous=True so the lazy-VLM
    # gate runs and rule 0b' on the second pass can lock GD's call
    # if VLM corroborates.
    if best_key is not None and best_sim is not None and best_sim >= embed_strong_min:
        if best_key == gd_key:
            ambiguous = False
            return _make(None, rejected=False)
        if gd_score is None or gd_score < gd_moderate_min:
            pred_key = best_key
            pred_label = label_display.get(best_key, best_key)
            pred_source = "embed-strong"
            ambiguous = False
            return _make(None, rejected=False)
        # else: defer to VLM-aware rules below.

    # Rule 1: rescue — gd very low + very confident embedding.
    if (
        gd_score is not None and gd_score < reject_gd_threshold
        and best_key is not None and best_sim is not None
        and best_sim >= gd_low_embed_rescue_min
    ):
        pred_key = best_key
        pred_label = label_display.get(best_key, best_key)
        pred_source = "embed"
        return _make(None, rejected=False)

    # Rule 2: gd very low, no rescue.
    if gd_score is not None and gd_score < reject_gd_threshold:
        return _make("gd", rejected=True)

    # Rule 3: combined low.
    if (
        gd_score is not None and gd_score < combined_reject_gd_max
        and best_sim is not None and best_sim < combined_reject_embed_max
    ):
        return _make("combined", rejected=True)

    # Rule 4: strict relabel. All conditions must hold for the
    # embedding to override GD's label. Pure "best_sim > gd_sim" is
    # NOT enough — the margins guard against near-equal-centroid
    # flips, and the VLM-or-low-GD clause requires external
    # corroboration when GD itself was confident.
    #
    # Hard block: when VLM corroborates GD's label, that GD+VLM
    # consensus overrides any embed-driven relabel attempt — even
    # if gd_score is below relabel_gd_low_max. Without this block
    # the resolver would happily flip a "gd glove + vlm glove"
    # detection to embed's "safety helmet" pick just because gd
    # came in at 0.22 (under 0.25) and the embed margins held up.
    vlm_corroborates_embed = (
        vlm_key is not None and best_key is not None and vlm_key == best_key
    )
    vlm_corroborates_gd = (
        vlm_key is not None and gd_key is not None and vlm_key == gd_key
    )
    can_relabel = (
        best_key is not None and best_sim is not None
        and best_key != gd_key
        and best_sim >= relabel_class_min
        and embed_margin >= relabel_embed_margin
        and (gd_sim is None or (best_sim - gd_sim) >= relabel_gd_embed_margin)
        and (
            (gd_score is not None and gd_score < relabel_gd_low_max)
            or vlm_corroborates_embed
        )
        and not vlm_corroborates_gd
    )
    relabel_action: str | None = None
    if can_relabel:
        pred_key = best_key
        pred_label = label_display.get(best_key, best_key)
        pred_source = "embed-vlm" if vlm_corroborates_embed else "embed"
        relabel_action = "relabel"

    # Rule 5: embed reject for the chosen label.
    final_sim = sims.get(pred_key) if pred_key else None
    if final_sim is not None and final_sim < reject_embed_threshold:
        return _make("embed", rejected=True, vlm_action=relabel_action)

    # Rule 6: VLM tiebreaker on still-ambiguous cases. VLM only
    # tiebreaks BETWEEN gd's label and embed's nearest label —
    # picking a third unrelated label is treated as "VLM disagrees,
    # noted but ignored".
    vlm_action = relabel_action
    if (
        ambiguous
        and not can_relabel  # didn't already trust embedding
        and vlm_key is not None
        and pred_key is not None
    ):
        candidate_keys = {gd_key, best_key} - {None, ""}
        if vlm_key in candidate_keys and vlm_key != pred_key:
            # VLM picks the OTHER candidate -> switch.
            pred_key = vlm_key
            pred_label = label_display.get(vlm_key, vlm_key)
            pred_source = "vlm"
            vlm_action = "tiebreak"
            # Re-check the embed reject for the newly chosen label.
            final_sim = sims.get(pred_key) if pred_key else None
            if final_sim is not None and final_sim < reject_embed_threshold:
                return _make("embed", rejected=True, vlm_action=vlm_action)
        elif vlm_key in candidate_keys and vlm_key == pred_key:
            vlm_action = "confirm"
        else:
            vlm_action = "disagree"

    return _make(None, rejected=False, vlm_action=vlm_action)


def _v2_resolve_label_specific(
    embedding: list[float],
    gd_label: str | None,
    refs_by_label: dict,
    label_display: dict[str, str],
    *,
    score_mode: str = "knn",
    filename_labels: set[str] | None = None,
    gd_score: float | None = None,
    relabel_threshold: float = 0.5,
    reject_embed_threshold: float = 0.4,
    reject_gd_threshold: float = 0.15,
    gd_low_embed_rescue_min: float = 0.6,
    combined_reject_gd_max: float = 0.25,
    combined_reject_embed_max: float = 0.48,
    # SigLIP2 ensemble inputs. When both query embedding and refs
    # are supplied, the resolver scores against each encoder
    # independently and combines the per-label sims as a weighted
    # mean (V2_SIGLIP_WEIGHT, default 0.4). DINOv2 stays the
    # primary signal because foreground-only patch_mean at 518×518
    # already does well on visual structure; SigLIP's contribution
    # is the text-aligned semantic prior.
    embedding_siglip: list[float] | None = None,
    refs_by_label_siglip: dict | None = None,
    siglip_weight: float = float(os.environ.get("V2_SIGLIP_WEIGHT", "0.4")),
    # Per-class accept thresholds derived from each class's leave-
    # one-out self-score distribution (mean - 3σ, capped at the
    # global default 0.40). When supplied, the embed-low reject in
    # rule 5 uses the predicted class's threshold instead of the
    # flat `reject_embed_threshold`. Computed by
    # `_v2_compute_class_thresholds` at the call site so the resolver
    # itself stays stateless.
    class_thresholds: dict[str, float] | None = None,
    # Patch-level matching inputs. When V2_PATCH_MATCH=on AND the
    # query patches + per-label reference patch grids are supplied,
    # the resolver swaps the pooled-cosine scoring for per-patch
    # top-K matching. Captures local distinguishing features (ear
    # texture, asymmetric markings) that the patch_mean smears.
    # DINOv2 and SigLIP patches are scored independently and
    # combined the same way the pooled cosines are (siglip_weight).
    query_patch_tokens: "np.ndarray | None" = None,
    query_patch_fg: "np.ndarray | None" = None,
    refs_by_label_patches: dict | None = None,
    query_patch_tokens_siglip: "np.ndarray | None" = None,
    query_patch_fg_siglip: "np.ndarray | None" = None,
    refs_by_label_patches_siglip: dict | None = None,
    # Margin below which the resolver flags a kept detection as
    # ambiguous (top1 - top2 within this band). Defaults to the env
    # `V2_AMBIGUOUS_MARGIN` (0.005) so V2's behaviour stays unchanged;
    # call sites that want a different threshold (e.g. Charlie's
    # specific path uses 0.013 because SAM3-derived crops sit in a
    # noisier sim distribution) can pass this explicitly.
    ambiguous_margin: float | None = None,
) -> dict:
    """Embed-priority resolver for SPECIFIC datasets.

    Specific datasets (hare vs rabbit; horse standing vs lying; chess
    pieces; plate countries) are fine-grained variants of one base
    concept. GD's open-vocabulary detector can't reliably tell them
    apart from the prompt — its score barely moves between sibling
    labels, AND it carries a strong pretraining bias toward whichever
    label is more common in its training data (rabbits >> hares,
    cats >> servals, etc). The user's reference embeddings are the
    SOLE label discriminator. GD's score is used only to gate "is
    this even a real detection" rejects; its label string is ignored.

    Previously the resolver would defer to GD when the embedding
    margin was small ("tie-break"). That's been removed: the user
    found GD's pretraining prior was always wrong on the cases where
    embedding margins are thin, so any GD intervention hurts more
    than it helps. The reference embeddings always win — even
    razor-thin margins.

    Decision tree (first match wins):
      1. No reference embeddings or empty sims → fall through to
         the gd_score reject so we don't stamp a random label.
      2. gd_score < 0.15 AND embed_best ≥ 0.6 → accept with embed
         (rescue: GD almost missed it but embed is confident).
      3. gd_score < 0.15 → reject "gd" (no rescue available).
      4. gd_score < 0.25 AND embed_best < 0.48 → reject "combined"
         (neither model commits).
      5. final_sim < 0.4 → reject "embed".
      6. otherwise → embed_best as the predicted label.
    """
    import numpy as _np

    sims: dict[str, float] = {}
    sims_dino: dict[str, float] = {}
    sims_siglip: dict[str, float] = {}
    sims_patch: dict[str, float] = {}
    sims_patch_siglip: dict[str, float] = {}

    # Patch-level DINOv2 matching. Replaces the pooled DINOv2 cosine
    # for any label that has both a query patch grid and reference
    # patch grids. Labels missing patch refs (mid-backfill) still
    # fall through to the pooled DINOv2 cosine below.
    use_patch_dino = (
        query_patch_tokens is not None
        and query_patch_fg is not None
        and refs_by_label_patches
    )
    if use_patch_dino:
        try:
            sims_patch = _v2_score_labels_patchwise(
                query_patch_tokens, query_patch_fg, refs_by_label_patches,
            )
        except Exception as e:
            print(f"[v2-resolve-specific] patch (dino) scoring failed: {e}")
            sims_patch = {}

    # Patch-level SigLIP matching. Same idea, parallel encoder.
    # When available it replaces the pooled SigLIP cosine in the
    # final combine step below.
    use_patch_siglip = (
        query_patch_tokens_siglip is not None
        and query_patch_fg_siglip is not None
        and refs_by_label_patches_siglip
    )
    if use_patch_siglip:
        try:
            sims_patch_siglip = _v2_score_labels_patchwise(
                query_patch_tokens_siglip, query_patch_fg_siglip, refs_by_label_patches_siglip,
            )
        except Exception as e:
            print(f"[v2-resolve-specific] patch (siglip) scoring failed: {e}")
            sims_patch_siglip = {}

    if embedding and refs_by_label:
        q = _np.asarray(embedding, dtype=_np.float32)
        if score_mode == "centroid":
            sims_dino = _v2_score_labels_centroid(q, refs_by_label)
        else:
            # Specific datasets: 1/d-weighted top-K kNN. Weighting by
            # inverse cosine distance lets a single very-close
            # reference dominate over several vague matches.
            sims_dino = _v2_score_labels_knn_weighted(q, refs_by_label)

    # If patch matching produced sims, those replace the pooled
    # DINOv2 sims label-by-label. Labels missing from patch_sims
    # still fall back to the pooled value (so a half-backfilled
    # project doesn't lose entire classes during transition).
    if sims_patch:
        merged_dino: dict[str, float] = {}
        for k in set(sims_dino.keys()) | set(sims_patch.keys()):
            merged_dino[k] = sims_patch.get(k, sims_dino.get(k, 0.0))
        sims_dino = merged_dino

    # SigLIP2 second opinion. Independent kNN against the SigLIP
    # reference set; combined with DINOv2's sims as a weighted mean
    # per label. Falls back to DINOv2-only when SigLIP isn't loaded
    # or has no refs for any label. When SigLIP patch tokens are
    # available, the patch top-K replaces the pooled cosine for
    # any label that has both — same fallback shape as the DINOv2
    # patch path above.
    use_siglip = (
        (embedding_siglip is not None and refs_by_label_siglip)
        or sims_patch_siglip
    ) and 0.0 < siglip_weight < 1.0
    if use_siglip:
        if embedding_siglip is not None and refs_by_label_siglip:
            qs = _np.asarray(embedding_siglip, dtype=_np.float32)
            if score_mode == "centroid":
                sims_siglip = _v2_score_labels_centroid(qs, refs_by_label_siglip)
            else:
                sims_siglip = _v2_score_labels_knn_weighted(qs, refs_by_label_siglip)
        if sims_patch_siglip:
            merged_siglip: dict[str, float] = {}
            for k in set(sims_siglip.keys()) | set(sims_patch_siglip.keys()):
                merged_siglip[k] = sims_patch_siglip.get(k, sims_siglip.get(k, 0.0))
            sims_siglip = merged_siglip
        # Combine label-by-label. A label only gets a combined score
        # when both encoders have a sim for it — half-empty pairs
        # fall back to the encoder that does have data, weighted at
        # 1.0 so the half-data label isn't artificially dragged toward
        # zero.
        all_keys = set(sims_dino.keys()) | set(sims_siglip.keys())
        for k in all_keys:
            d_sim = sims_dino.get(k)
            s_sim = sims_siglip.get(k)
            if d_sim is not None and s_sim is not None:
                sims[k] = (1.0 - siglip_weight) * d_sim + siglip_weight * s_sim
            elif d_sim is not None:
                sims[k] = d_sim
            elif s_sim is not None:
                sims[k] = s_sim
    else:
        sims = dict(sims_dino)

    if filename_labels:
        sims = _v2_apply_filename_hint(sims, filename_labels)

    sorted_sims_desc = sorted(sims.values(), reverse=True)
    # Margin is top1 - top2. With <2 labels in sims (e.g. only one
    # label has references uploaded yet) there's no second-best to
    # subtract from, so the margin is undefined — return 0.0 rather
    # than the top1 sim itself, which was misleading the FE into
    # rendering a single sim's value as if it were the margin.
    embed_margin = (
        sorted_sims_desc[0] - sorted_sims_desc[1]
        if len(sorted_sims_desc) >= 2 else
        0.0
    )
    best_key = max(sims, key=sims.get) if sims else None
    best_sim = sims[best_key] if best_key else None

    # Seed the prediction from EMBEDDING'S top label, not GD's.
    # GD's label is intentionally ignored at the label-choice step —
    # see the docstring above for why.
    pred_key = best_key
    pred_label = label_display.get(best_key, best_key) if best_key else None
    pred_source: str | None = "embed" if best_key else None

    # Margin-driven ambiguity flag for the FE. When the embed top-1
    # is within `_ambiguous_margin` of the runner-up the decision is
    # essentially a coin flip and the FE should surface the detection
    # in the "unsure" tray rather than the verified one. The resolver
    # itself still commits to embed_best — this is purely a UI signal
    # so the user can review borderline calls. Threshold tunable via
    # V2_AMBIGUOUS_MARGIN. Computed inside _make so it can also
    # respect the rejected flag (rejected detections already get a
    # red pill — adding "unsure" on top is noisy).
    _ambiguous_margin = (
        float(ambiguous_margin)
        if ambiguous_margin is not None
        else float(os.environ.get("V2_AMBIGUOUS_MARGIN", "0.005"))
    )

    def _make(reject_reason: str | None, *, rejected: bool) -> dict:
        final_sim = sims.get(pred_key) if pred_key else None
        # Only flag ambiguous on KEPT detections — rejected ones
        # already get the "rejected" pill in the FE; doubling up
        # on "unsure" + "rejected" is noisy.
        ambiguous = (
            (not rejected)
            and len(sims) >= 2
            and embed_margin < _ambiguous_margin
        )
        return {
            "pred_label": pred_label,
            "pred_source": pred_source,
            "embed_nearest_label": label_display.get(best_key, best_key) if best_key else None,
            "embed_nearest_sim": round(float(best_sim), 4) if best_sim is not None else None,
            "embed_sim_for_label": round(float(final_sim), 4) if final_sim is not None else None,
            "embed_margin": round(float(embed_margin), 4),
            "rejected": rejected,
            "reject_reason": reject_reason,
            "ambiguous": ambiguous,
            "vlm_action": None,
            "sims": {label_display.get(k, k): round(float(v), 4) for k, v in sims.items()},
            # Per-encoder breakdown for the popup. Lets the user see
            # which encoder's signal carried each detection — handy
            # when DINOv2 and SigLIP disagree on a fine-grained pair.
            "sims_dino": {label_display.get(k, k): round(float(v), 4) for k, v in sims_dino.items()},
            "sims_siglip": {label_display.get(k, k): round(float(v), 4) for k, v in sims_siglip.items()},
            "sims_patch": {label_display.get(k, k): round(float(v), 4) for k, v in sims_patch.items()},
            "sims_patch_siglip": {label_display.get(k, k): round(float(v), 4) for k, v in sims_patch_siglip.items()},
            "siglip_weight": float(siglip_weight) if use_siglip else 0.0,
            "patch_match_used": bool(sims_patch),
            "patch_match_used_siglip": bool(sims_patch_siglip),
        }

    if not sims:
        rejected_gd = gd_score is not None and gd_score < reject_gd_threshold
        # Without embed signal we can't pick a label, so fall back to
        # GD's label here just to surface SOMETHING (it'll be rejected
        # if gd_score is low). This branch only fires for projects
        # with no reference embeddings yet.
        if not rejected_gd and gd_label:
            pred_key = (gd_label or "").strip().lower()
            pred_label = gd_label
            pred_source = "gd"
        return _make("gd" if rejected_gd else None, rejected=rejected_gd)

    # Rule 2: GD-low rescue. GD almost dropped this detection but
    # embed is confident — keep the box, label it from embed.
    if (
        gd_score is not None and gd_score < reject_gd_threshold
        and best_sim is not None
        and best_sim >= gd_low_embed_rescue_min
    ):
        return _make(None, rejected=False)

    # Rule 3: GD very low alone — reject.
    if gd_score is not None and gd_score < reject_gd_threshold:
        return _make("gd", rejected=True)

    # Rule 4: combined-low reject.
    if (
        gd_score is not None and gd_score < combined_reject_gd_max
        and best_sim is not None and best_sim < combined_reject_embed_max
    ):
        return _make("combined", rejected=True)

    # Rule 5: final-label embed-low reject. The prediction is the
    # embed top label; reject if its absolute similarity is too low
    # to be meaningful. (GD tie-break removed — the user found GD's
    # pretraining prior was always wrong on the cases where embed
    # margins were thin, so it dragged decisions in the wrong
    # direction more often than it tied them correctly.)
    final_sim = sims.get(pred_key) if pred_key else None
    # Per-class threshold beats the global default when one is
    # available for the predicted class. A tightly-clustered class
    # ("stop sign", LOO mean 0.85 σ 0.04) raises the bar to ~0.77;
    # a loosely-clustered one ("horse", mean 0.55 σ 0.10) drops it
    # to ~0.35 so legitimate variety isn't rejected. Falls back to
    # `reject_embed_threshold` when class_thresholds is missing or
    # the class has too few refs for a stable distribution.
    threshold_used = reject_embed_threshold
    if class_thresholds is not None and pred_key is not None:
        threshold_used = class_thresholds.get(pred_key, reject_embed_threshold)
    if final_sim is not None and final_sim < threshold_used:
        return _make("embed", rejected=True)

    # Diagnostic: print the sims so we can see what's actually
    # driving each decision when the user reports "rabbits look
    # like hares". One log line per detection — cheap, parsable.
    try:
        sims_str = ", ".join(
            f"{label_display.get(k, k)}={v:.3f}"
            for k, v in sorted(sims.items(), key=lambda kv: -kv[1])
        )
        print(
            f"[v2-resolve-specific] gd={gd_label!r}({gd_score}) "
            f"→ pred={pred_label!r} src={pred_source} "
            f"margin={embed_margin:.3f} thr={threshold_used:.3f} sims=[{sims_str}]"
        )
    except Exception:
        pass

    return _make(None, rejected=False)


def _v2_apply_containment(
    detections: list[dict],
    label_display: dict[str, str],
    *,
    containment_min: float = 0.7,
    size_ratio_max: float = 0.7,
    reject_embed_threshold: float = 0.4,
) -> None:
    """Post-pass that fixes mislabelled containers (mutates in place).

    When a smaller detection is mostly inside a bigger one AND they
    share a label, the bigger box is suspect. Two distinct patterns,
    each with a different fix:

    1. **Duplicate detection** — the bigger box is a looser version
       of the smaller one (same object detected twice). Detected by
       SIGNAL AGREEMENT: GD's label, embedding-nearest centroid, AND
       (if it ran) VLM all picked the same label as the smaller's.
       Action: REJECT the bigger; keep the tighter inner box.

    2. **Compositional mislabel** — the bigger box is genuinely a
       different object (e.g., a person whose bbox embeds close to
       the high-vis vest centroid because the centroid is biased).
       Detected by DISAGREEMENT: at least one of GD / embed-nearest
       / VLM picked a different label. Action: re-resolve the
       bigger box to its next-best centroid that ISN'T the smaller's
       label; reject if no alternative clears the embed floor.

    Different-label pairs are left alone — those are the legitimate
    "vest inside person" detections we want to keep.

    Containment is computed per-pair as
    ``(intersection area) / (smaller box area)``; size_ratio_max
    guards against treating two near-equal boxes as containment.
    """
    n = len(detections)
    fixes: list[str] = []
    for i in range(n):
        big = detections[i]
        if big.get("rejected"):
            continue
        big_label = big.get("pred_label")
        if not big_label:
            continue
        big_box = big.get("box") or []
        if len(big_box) != 4:
            continue
        bx0, by0, bx1, by1 = (float(c) for c in big_box)
        big_area = max((bx1 - bx0) * (by1 - by0), 1.0)
        # Pre-compute big's per-signal labels for the agreement
        # check below (stable across the j-loop).
        big_gd_key = str(big.get("gd_label") or "").strip().lower()
        big_embed_nearest_key = str(big.get("embed_nearest_label") or "").strip().lower()
        big_vlm_key = str(big.get("vlm_label") or "").strip().lower()
        for j in range(n):
            if i == j:
                continue
            small = detections[j]
            if small.get("rejected"):
                continue
            small_label = small.get("pred_label")
            if not small_label:
                continue
            if str(small_label).strip().lower() != str(big_label).strip().lower():
                continue
            small_box = small.get("box") or []
            if len(small_box) != 4:
                continue
            sx0, sy0, sx1, sy1 = (float(c) for c in small_box)
            small_area = max((sx1 - sx0) * (sy1 - sy0), 1.0)
            if small_area >= big_area * size_ratio_max:
                continue
            ix0 = max(sx0, bx0); iy0 = max(sy0, by0)
            ix1 = min(sx1, bx1); iy1 = min(sy1, by1)
            if ix1 <= ix0 or iy1 <= iy0:
                continue
            inter = (ix1 - ix0) * (iy1 - iy0)
            containment = inter / small_area
            if containment < containment_min:
                continue

            # Agreement check: if every signal that fired on the
            # bigger box picked the same label as the smaller's, the
            # pair is the same object detected twice. Two flavours:
            #   - Duplicate: smaller is the tighter precise crop, the
            #     bigger pulls in surrounding pixels. Smaller is the
            #     more confident detection (gd + embed sim higher).
            #   - Partial occlusion: smaller only annotates a fragment
            #     of the object (e.g., a hand covers half a vest),
            #     while the bigger is the correct full annotation.
            #     Bigger is the more confident one in that case.
            # Confidence here = gd_score + embed_sim_for_label. We
            # reject whichever side has the lower combined score.
            # (`small_key` was never assigned in the SaaS build — a latent
            # NameError on this path; the intended value is the smaller
            # box's normalised label, mirroring the sibling NMS pass.)
            small_key = str(small_label).strip().lower()
            all_agree = (
                big_gd_key == small_key
                and big_embed_nearest_key == small_key
                and (not big_vlm_key or big_vlm_key == small_key)
            )
            if all_agree:
                big_score = (
                    float(big.get("gd_score") or 0)
                    + float(big.get("embed_sim_for_label") or 0)
                )
                small_score = (
                    float(small.get("gd_score") or 0)
                    + float(small.get("embed_sim_for_label") or 0)
                )
                if small_score >= big_score:
                    # Smaller more confident (or tied) → duplicate case.
                    big["rejected"] = True
                    big["reject_reason"] = "containment"
                    big["pred_label"] = None
                    big["pred_source"] = "containment"
                    big["vlm_action"] = "containment-duplicate"
                    fixes.append(
                        f"{big_label}→reject(duplicate, conf {big_score:.2f}≤{small_score:.2f})"
                    )
                    break
                # Bigger more confident → partial-occlusion case.
                small["rejected"] = True
                small["reject_reason"] = "containment"
                small["pred_label"] = None
                small["pred_source"] = "containment"
                small["vlm_action"] = "containment-partial"
                fixes.append(
                    f"{big_label}→reject inner(partial, conf {small_score:.2f}<{big_score:.2f})"
                )
                continue

            # Hit. Re-resolve big using sims it already has, but
            # excluding the small's label as a candidate. embed_sims
            # is keyed on display-cased labels so we compare lower.
            small_key = str(small_label).strip().lower()
            sims = big.get("embed_sims") or {}
            alt = {
                k: float(v) for k, v in sims.items()
                if str(k).strip().lower() != small_key
            }
            prev_label = big_label
            if not alt:
                big["rejected"] = True
                big["reject_reason"] = "containment"
                big["pred_label"] = None
                big["pred_source"] = "containment"
                big["vlm_action"] = "containment-reject"
                fixes.append(f"{prev_label}→reject(no-alt)")
                break
            best_alt_disp = max(alt, key=alt.get)
            best_alt_sim = alt[best_alt_disp]
            if best_alt_sim >= reject_embed_threshold:
                big["pred_label"] = best_alt_disp
                big["pred_source"] = "embed-containment"
                big["embed_sim_for_label"] = round(float(best_alt_sim), 4)
                big["vlm_action"] = "containment-relabel"
                fixes.append(f"{prev_label}→{best_alt_disp}")
                # Subsequent j's compare against the NEW label.
                big_label = best_alt_disp
            else:
                big["rejected"] = True
                big["reject_reason"] = "containment"
                big["pred_label"] = None
                big["pred_source"] = "containment"
                big["vlm_action"] = "containment-reject"
                fixes.append(f"{prev_label}→reject(low-alt {best_alt_sim:.2f})")
                break
    if fixes:
        print(f"[v2-import] containment fixes: {len(fixes)} — {', '.join(fixes)}")


def _v2_polygon_components(mask_obj: dict | None, *, secondary_frac: float = 0.10) -> int:
    """Count meaningful disconnected regions in a SAM mask payload.

    SAM's segmentation pipeline already filters polygons under
    MIN_CONTOUR_AREA_FRAC (0.5% of box area) — anything that survives
    is non-trivial. Within that, we treat polygons whose area is
    below `secondary_frac` of the largest as residual noise that
    shouldn't count toward the component total. So:
      - 1 → single connected region (the clean case)
      - 2+ → there's a second entity in the mask (the case we want
             to reject in same-label overlap pairs)
      - 0 → empty / malformed mask
    """
    if not isinstance(mask_obj, dict):
        return 0
    polys = mask_obj.get("polygons") or []
    if not polys:
        return 0
    areas: list[float] = []
    for p in polys:
        if not isinstance(p, list) or len(p) < 3:
            continue
        s = 0.0
        n_pts = len(p)
        for k in range(n_pts):
            try:
                x1, y1 = float(p[k][0]), float(p[k][1])
                x2, y2 = float(p[(k + 1) % n_pts][0]), float(p[(k + 1) % n_pts][1])
            except (TypeError, IndexError, ValueError):
                continue
            s += x1 * y2 - x2 * y1
        areas.append(abs(s) / 2.0)
    if not areas:
        return 0
    max_a = max(areas)
    if max_a <= 0:
        return 0
    return sum(1 for a in areas if a >= max_a * secondary_frac)


def _v2_apply_same_label_overlap(
    detections: list[dict],
    *,
    overlap_iou_min: float = 0.30,
    occlusion_separation_min: float = 0.30,
) -> None:
    """Same-label overlap pass: prefer the cleaner / more confident peer.

    When two same-label boxes overlap (IoU ≥ overlap_iou_min — set
    permissively so a small tolerance still triggers), we drop the
    weaker box. Two-tier discrimination:

    1. Mask quality (primary): if one mask is a single connected
       region (one closed shape) and the other has a secondary
       component (a second entity bleeding in), the multi-component
       box is rejected.
    2. Confidence (fallback): when mask quality doesn't decide
       (both single or both multi), reject the box with the lower
       combined confidence (gd_score + embed_sim_for_label). This
       catches the user's "person wearing high-vis with a hand
       covering half" case — the partial annotation lands lower on
       both signals than the full one.

    Occlusion guard: before either tier runs, we compute the
    Euclidean distance between the two bbox centres normalised by
    the average box side length. When that ratio is >=
    occlusion_separation_min the boxes are at meaningfully
    different positions despite the bbox overlap — likely separate
    objects partially occluding each other (a row of people half
    behind one another) rather than two detections of the same
    object. Skip rejection in that case.

    Equal-confidence ties leave both alone. Different-label overlaps
    are out of scope. Runs AFTER `_v2_apply_containment` so
    containment-rejected boxes don't get reconsidered here.
    """
    n = len(detections)
    fixes: list[str] = []
    components_cache: dict[int, int] = {}

    def _components_for(idx: int) -> int:
        if idx not in components_cache:
            components_cache[idx] = _v2_polygon_components(
                detections[idx].get("mask")
            )
        return components_cache[idx]

    for i in range(n):
        a = detections[i]
        if a.get("rejected"):
            continue
        a_label = a.get("pred_label")
        if not a_label:
            continue
        a_box = a.get("box") or []
        if len(a_box) != 4:
            continue
        ax0, ay0, ax1, ay1 = (float(c) for c in a_box)
        a_area = max((ax1 - ax0) * (ay1 - ay0), 1.0)
        a_label_key = str(a_label).strip().lower()
        for j in range(i + 1, n):
            b = detections[j]
            if b.get("rejected"):
                continue
            b_label = b.get("pred_label")
            if not b_label:
                continue
            if str(b_label).strip().lower() != a_label_key:
                continue
            b_box = b.get("box") or []
            if len(b_box) != 4:
                continue
            bx0, by0, bx1, by1 = (float(c) for c in b_box)
            b_area = max((bx1 - bx0) * (by1 - by0), 1.0)
            ix0 = max(ax0, bx0); iy0 = max(ay0, by0)
            ix1 = min(ax1, bx1); iy1 = min(ay1, by1)
            if ix1 <= ix0 or iy1 <= iy0:
                continue
            inter = (ix1 - ix0) * (iy1 - iy0)
            union = a_area + b_area - inter
            if union <= 0:
                continue
            iou = inter / union
            if iou < overlap_iou_min:
                continue

            # Occlusion guard: bboxes can overlap heavily when two
            # distinct objects partially occlude each other (e.g. a
            # row of people standing half behind one another). Their
            # bbox centres are at meaningfully different positions
            # though, so we use the centre-distance / avg-side ratio
            # as a proxy for "are these the same object or different
            # objects that just happen to share image space".
            a_cx = (ax0 + ax1) / 2.0
            a_cy = (ay0 + ay1) / 2.0
            b_cx = (bx0 + bx1) / 2.0
            b_cy = (by0 + by1) / 2.0
            centre_dist = ((a_cx - b_cx) ** 2 + (a_cy - b_cy) ** 2) ** 0.5
            avg_side = ((a_area ** 0.5) + (b_area ** 0.5)) / 2.0
            if avg_side > 0 and (centre_dist / avg_side) >= occlusion_separation_min:
                continue

            a_n = _components_for(i)
            b_n = _components_for(j)
            reject_a = False
            reject_b = False
            action: str | None = None
            note = ""
            if a_n > 1 and b_n == 1:
                reject_a = True
                action = "overlap-multi-mask"
                note = f"(parts={a_n})→reject vs single-mask peer"
            elif b_n > 1 and a_n == 1:
                reject_b = True
                action = "overlap-multi-mask"
                note = f"(parts={b_n})→reject vs single-mask peer"
            else:
                # Mask quality didn't discriminate (both single or
                # both multi). Compare combined confidence.
                a_score = (
                    float(a.get("gd_score") or 0)
                    + float(a.get("embed_sim_for_label") or 0)
                )
                b_score = (
                    float(b.get("gd_score") or 0)
                    + float(b.get("embed_sim_for_label") or 0)
                )
                if a_score < b_score:
                    reject_a = True
                    action = "overlap-low-conf"
                    note = f"(conf {a_score:.2f}<{b_score:.2f})→reject vs higher-conf peer"
                elif b_score < a_score:
                    reject_b = True
                    action = "overlap-low-conf"
                    note = f"(conf {b_score:.2f}<{a_score:.2f})→reject vs higher-conf peer"
                # else: equal confidence, leave both alone.

            if reject_a:
                a["rejected"] = True
                a["reject_reason"] = "overlap"
                a["pred_label"] = None
                a["pred_source"] = "overlap"
                a["vlm_action"] = action
                fixes.append(f"{a_label}{note}")
                break  # a is gone; advance i
            if reject_b:
                b["rejected"] = True
                b["reject_reason"] = "overlap"
                b["pred_label"] = None
                b["pred_source"] = "overlap"
                b["vlm_action"] = action
                fixes.append(f"{b_label}{note}")
                # don't break: i may pair with another j
    if fixes:
        print(f"[v2-import] same-label overlap fixes: {len(fixes)} — {', '.join(fixes)}")


# ─── Dataset-type classifier (general vs specific) ───────────────
# Distinguishes "general" datasets (visually-distinct categories
# like person/car/pothole — detector dominates the decision) from
# "specific" datasets (fine-grained variants like hare vs rabbit
# or horse-standing vs horse-lying — embeddings against per-label
# centroids do the heavy lifting). Surfaced in the UI so the user
# knows what the pipeline is leaning on, and useful as a tuning
# hook for future per-mode threshold profiles.

def _dataset_type_cache_path(project_id: str) -> Path:
    return project_dir(project_id) / "dataset_type.json"


# Bumped whenever the classifier prompt changes meaningfully so that
# old cached verdicts get re-asked under the new prompt without
# anyone having to delete dataset_type.json by hand. v4: broadened the
# SPECIFIC definition to cover material / look-alike-different-object
# cases (glass vs plastic cup, screw vs threaded standoff).
_DATASET_TYPE_PROMPT_VERSION = 4

# Reason shown when a project's type was set by hand rather than by the
# classifier and no more specific reason was recorded.
_DEFAULT_OVERRIDE_REASON = "Set manually."


def _labels_signature(tags: list[str]) -> str:
    """Stable key used to invalidate the cache when labels change."""
    return ",".join(sorted({str(t).strip().lower() for t in tags if t and str(t).strip()}))


def _read_dataset_type_sidecar(project_id: str) -> dict | None:
    p = _dataset_type_cache_path(project_id)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception as e:
        print(f"[dataset-type] cache read failed for {project_id}: {e}")
        return None


def _write_dataset_type_sidecar(project_id: str, data: dict) -> None:
    p = _dataset_type_cache_path(project_id)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[dataset-type] cache write failed for {project_id}: {e}")


def _classify_dataset_type_cached(project_id: str, tags: list[str]) -> dict:
    """Return {type, reason, labels_signature, prompt_version, source}.

    Resolution order, highest priority first:
      1. A user/reference OVERRIDE persisted in the sidecar
         (`override` field). This wins regardless of the label
         signature — the user explicitly chose it (or added reference
         images), so a later label edit must not silently undo it. The
         label pipeline reads this same resolver, so an override flips
         centroid-vs-kNN scoring too, not just the UI badge.
      2. A fresh cached Claude verdict (matching signature + prompt
         version).
      3. A fresh Claude classification (cached for next time).

    Always returns a dict — silent fallback on any error so the V2
    project page never blocks on Claude. `source` is "manual",
    "references", or "auto" so the UI can explain where the verdict
    came from and offer a "let PixelKit decide" reset."""
    sig = _labels_signature(tags)
    existing = _read_dataset_type_sidecar(project_id)

    if isinstance(existing, dict):
        ov = existing.get("override")
        if ov in ("general", "specific"):
            return {
                "type": ov,
                "reason": existing.get("override_reason") or _DEFAULT_OVERRIDE_REASON,
                "labels_signature": sig,
                "prompt_version": _DATASET_TYPE_PROMPT_VERSION,
                "override": ov,
                "source": existing.get("override_source") or "manual",
            }
        if (
            existing.get("labels_signature") == sig
            and existing.get("prompt_version") == _DATASET_TYPE_PROMPT_VERSION
            and existing.get("type") in ("general", "specific")
        ):
            out = dict(existing)
            out.setdefault("source", "auto")
            return out

    # Portable build: no Claude auto-classification. Datasets default to
    # "general"; adding reference images or the explicit POST /dataset-type
    # override (both handled above) flip it to "specific".
    out = {
        "type": "general",
        "reason": "default (set reference images or choose manually to switch)",
        "labels_signature": sig,
        "prompt_version": _DATASET_TYPE_PROMPT_VERSION,
        "source": "auto",
    }
    _write_dataset_type_sidecar(project_id, out)
    return out


def _resolve_dataset_type_cached_only(project_id: str, tags: list[str]) -> dict | None:
    """Cheap, LLM-free resolve of the dataset type for first-paint use.

    Mirrors the two early-return branches of `_classify_dataset_type_cached`
    (a sticky override, or a fresh cached verdict matching the current label
    signature) but returns None instead of falling through to a Claude call.
    Lets /overview embed the general/specific badge so it paints in the same
    frame as the rest of the hero, without ever blocking the overview build on
    the classifier. A None result means the FE still does its own /dataset-type
    fetch (which may classify)."""
    existing = _read_dataset_type_sidecar(project_id)
    if not isinstance(existing, dict):
        return None
    sig = _labels_signature(tags)
    ov = existing.get("override")
    if ov in ("general", "specific"):
        return {
            "type": ov,
            "reason": existing.get("override_reason") or _DEFAULT_OVERRIDE_REASON,
            "source": existing.get("override_source") or "manual",
        }
    if (
        existing.get("labels_signature") == sig
        and existing.get("prompt_version") == _DATASET_TYPE_PROMPT_VERSION
        and existing.get("type") in ("general", "specific")
    ):
        return {
            "type": existing.get("type"),
            "reason": existing.get("reason") or "",
            "source": existing.get("source") or "auto",
        }
    return None


def _set_dataset_type_override(project_id: str, choice: str, *, source: str, reason: str) -> None:
    """Persist a sticky dataset-type override into the sidecar. `choice`
    is "general" or "specific"; `source` is "manual" or "references".
    Preserves the auto classification fields so picking "auto" later can
    fall back to them without a fresh Claude call when they're still
    valid."""
    data = _read_dataset_type_sidecar(project_id) or {}
    data["override"] = choice
    data["override_source"] = source
    data["override_reason"] = reason
    _write_dataset_type_sidecar(project_id, data)


def _clear_dataset_type_override(project_id: str) -> None:
    """Drop a sticky override so the type reverts to the auto classifier."""
    data = _read_dataset_type_sidecar(project_id)
    if not isinstance(data, dict):
        return
    changed = False
    for k in ("override", "override_source", "override_reason"):
        if k in data:
            del data[k]
            changed = True
    if changed:
        _write_dataset_type_sidecar(project_id, data)


def _apply_reference_dataset_flip(project_id: str) -> None:
    """Adding reference images is a strong signal the dataset is
    SPECIFIC (references only matter when classes look alike). Flip the
    project to specific via a sticky override — but never stomp a choice
    the user made by hand, so someone who deliberately set "general" and
    then uploads a reference keeps their setting."""
    existing = _read_dataset_type_sidecar(project_id)
    if isinstance(existing, dict) and existing.get("override_source") == "manual":
        return
    _set_dataset_type_override(
        project_id,
        "specific",
        source="references",
        reason="You added reference images, so this is treated as a specific dataset.",
    )


@app.get(
    "/api/v2/projects/{project_id}/dataset-type",
    dependencies=[Depends(require_project_read_access)],
)
async def v2_dataset_type(project_id: str):
    """Classify the project's label set as general or specific."""
    if not project_dir(project_id).exists():
        raise HTTPException(404)
    manifest = load_manifest(project_id)
    tags = manifest.get("tags") or []
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, _classify_dataset_type_cached, project_id, list(tags),
    )


# ── AI dataset insight (Claude, token-frugal + cached) ────────────────────────
# One smart, dataset-specific coaching line for the Overview's Insights section.
# Built from a COMPACT numeric summary (no images) and cached by a coarse
# signature so Claude is only hit when the dataset materially changes.

@app.get(
    "/api/v2/projects/{project_id}/access",
    dependencies=[Depends(require_project_read_access)],
)
async def v2_project_access(project_id: str, request: Request, authorization: str = Header(default="")):
    """The caller's effective access to this dataset: write (edit/label/upload)
    and manage. Drives the FE's read-only decision so a Project EDITOR can edit
    a dataset another member created (write comes from Project membership, not
    just dataset ownership)."""
    m = load_manifest(project_id, copy=False)
    if not m:
        raise HTTPException(404, "project not found")
    viewer = request_username(request, authorization)
    acc = containers.dataset_access(m, viewer)
    return {
        "writable": bool(acc["writable"]),
        "manageable": bool(acc["manageable"]),
        "owner": (m.get("owner") or ""),
    }


class DatasetTypeOverrideIn(BaseModel):
    # "general" / "specific" pin the type; "auto" clears the override
    # and hands control back to the classifier.
    type: str


@app.post(
    "/api/v2/projects/{project_id}/dataset-type",
    dependencies=[Depends(require_project_owner)],
)
async def v2_set_dataset_type(project_id: str, payload: DatasetTypeOverrideIn):
    """Let the project owner override the general/specific verdict (or
    reset it to the classifier). The override is sticky and feeds the
    label pipeline — not just the badge — so centroid-vs-kNN scoring
    follows the user's choice. Returns the freshly-resolved verdict."""
    choice = (payload.type or "").strip().lower()
    if choice not in ("general", "specific", "auto"):
        raise HTTPException(400, "type must be 'general', 'specific', or 'auto'")
    manifest = load_manifest(project_id, copy=False)
    tags = list(manifest.get("tags") or [])
    if choice == "auto":
        _clear_dataset_type_override(project_id)
    else:
        _set_dataset_type_override(
            project_id, choice, source="manual", reason=_DEFAULT_OVERRIDE_REASON,
        )
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, _classify_dataset_type_cached, project_id, tags,
    )


class DatasetTypePreviewBody(BaseModel):
    labels: list[str]


@app.get("/api/openverse/search")
async def openverse_search(q: str, count: int = 5, commercial: bool = False):
    """Search Openverse for `q` and return up to `count` (capped 250)
    Creative-Commons-licensed image results. Used by the "Don't have
    images?" panel — small counts (~5) to preview candidates, larger
    counts (up to 250) when the user has confirmed and is pulling the
    full corpus.

    `commercial=true` filters to licences that permit commercial use
    (CC0 / BY / BY-SA / PDM)."""
    import openverse
    query = (q or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="q is required")
    n = max(1, min(250, count))
    loop = asyncio.get_event_loop()
    try:
        results = await loop.run_in_executor(
            None, lambda: openverse.search_images(query, n, commercial=commercial),
        )
    except Exception as e:
        print(f"[openverse] search failed: {e}")
        raise HTTPException(status_code=502, detail=str(e))
    # Server-side thumbnail validation. The frontend probe is
    # CORS-limited (canvas reads fail on most providers, and broken
    # thumbnails sometimes still fire onload), so we add a parallel
    # HEAD-probe pass here that filters out 404s, HTML error pages,
    # and tiny placeholder bytes BEFORE the search response leaves
    # the backend. Runs in a thread pool — typically <2s for a
    # 50-result page on a healthy network.
    try:
        validated = await loop.run_in_executor(
            None, lambda: openverse.validate_thumbnails(results),
        )
        results = validated
    except Exception as e:
        print(f"[openverse] thumbnail validation failed (returning unfiltered): {e}")
    return {"query": query, "results": results, "commercial": commercial}




def collect_tags(manifest: dict) -> list[str]:
    """Union of prompt tags and labels actually present on the
    project's data. Sources, in order of authority:

      1. manifest['tags'] — the explicit project label list (what
         the FE label editor writes via PUT /api/projects/{id}).
      2. V1 editedBoxes (manifest['editedBoxes']) — boxes the user
         drew/relabelled on V1 imports.
      3. V2 references' detections — boxes on uploaded reference
         images with assigned labels (added in the V2 editor).
      4. V2 imports' detections — labels the resolver picked +
         user-confirmed boxes on dataset images.

    The 3rd and 4th are recovery paths for older projects that
    pre-date the FE label-save fix: the FE used to drop
    rename/add/delete edits without persisting them to
    manifest['tags'], so old projects' authoritative tags list is
    sometimes empty even though every reference image has a
    correct label. Pulling from those detections populates the
    workspace card's `tags` from the data the user actually
    labelled.
    """
    seen: set[str] = set()
    ordered: list[str] = []

    for t in manifest.get("tags", []) or []:
        tl = (t or "").strip().lower()
        if tl and tl not in seen:
            seen.add(tl)
            ordered.append(tl)

    extra_labels: set[str] = set()

    # V1 editedBoxes pass.
    edited = manifest.get("editedBoxes", {}) or {}
    for boxes in edited.values():
        if not isinstance(boxes, list):
            continue
        for b in boxes:
            if not isinstance(b, dict):
                continue
            label = (b.get("label") or "").strip()
            if not label or label.lower() == "new":
                continue
            for part in label.split(" + "):
                p = part.strip().lower()
                if p:
                    extra_labels.add(p)

    # V2 references' detections — recovery path for label-save bug.
    for ref in manifest.get("references", []) or []:
        if not isinstance(ref, dict):
            continue
        for d in ref.get("detections", []) or []:
            if not isinstance(d, dict):
                continue
            label = (d.get("label") or "").strip().lower()
            if label and label != "new":
                extra_labels.add(label)

    # V2 imports' detections — same idea.
    for imp in manifest.get("imports", []) or []:
        if not isinstance(imp, dict):
            continue
        for d in imp.get("detections", []) or []:
            if not isinstance(d, dict):
                continue
            label = (d.get("label") or "").strip().lower()
            if label and label != "new":
                extra_labels.add(label)
        # And user-edited boxes on imports.
        for b in imp.get("editedBoxes", []) or []:
            if not isinstance(b, dict):
                continue
            label = (b.get("label") or "").strip().lower()
            if label and label != "new":
                extra_labels.add(label)

    for tl in sorted(extra_labels):
        if tl not in seen:
            seen.add(tl)
            ordered.append(tl)
    return ordered


# Short-lived response cache for /api/projects. Keyed on the request's
# (owner, viewer, offset, limit) tuple; entries expire after a few
# seconds so the FE's 4-second poll cycle doesn't redo the entire
# project-directory walk + manifest load every time. Cache also
# stat-watches every project's manifest mtime so it self-invalidates
# the moment any project changes (upload, edit, delete) — no risk of
# returning stale data to the FE for more than the resolution of the
# stat sweep.

_PROJECTS_RESPONSE_CACHE: dict[tuple, dict] = {}
_PROJECTS_RESPONSE_LOCK = __import__("threading").Lock()
_PROJECTS_RESPONSE_TTL_S = 2.0


def _projects_response_cache_get(key: tuple) -> object | None:
    """Return cached response when fresh AND no project's manifest has
    been written since the cache populated. None on miss / staleness."""
    import time as _t
    with _PROJECTS_RESPONSE_LOCK:
        entry = _PROJECTS_RESPONSE_CACHE.get(key)
        if entry is None:
            return None
        age = _t.time() - entry["ts"]
        if age > _PROJECTS_RESPONSE_TTL_S:
            return None
        # mtime sanity check — a manifest write between the cache
        # populating and now invalidates immediately, regardless of
        # the TTL. Cheap: stat is microseconds per project.
        for pid, mtime in entry["mtimes"].items():
            if _manifest_disk_mtime(pid) > mtime:
                return None
        return entry["body"]


def _projects_response_cache_put(key: tuple, body: object, mtimes: dict[str, float]) -> None:
    import time as _t
    with _PROJECTS_RESPONSE_LOCK:
        # Cap the cache so it doesn't grow unbounded as users hit
        # different (offset, limit) tuples — most usage is the same
        # few keys but a hostile caller could blow it up.
        if len(_PROJECTS_RESPONSE_CACHE) > 64:
            _PROJECTS_RESPONSE_CACHE.clear()
        _PROJECTS_RESPONSE_CACHE[key] = {
            "ts": _t.time(),
            "body": body,
            "mtimes": dict(mtimes),
        }


# In-memory cache for cover blurhashes. Keyed by (project_id, cover_filename)
# so a cover swap on the same project invalidates correctly. Survives until
# restart; ~30 chars per entry so 10K projects = 300 KB. Computed lazily on
# first list_projects call that returns each project — no manifest write,
# no pre-bake step needed.
_BLURHASH_CACHE: dict[tuple[str, str], str] = {}


# Bounded LRU cache for served image bytes. The FE reloads the same
# reference / import / cover thumbnails on every project open and
# every workspace poll; serving from RAM is sub-millisecond vs
# ~10-50 ms hitting disk through OS page-cache (for cold pages) or
# even the SSD itself for cold-after-restart files. With this in
# place a typical project page open feels instant on revisit.
#
# Sizing: ~25% of available RAM at startup, capped at 8 GB. Logged
# at startup alongside the manifest-cache capacity probe.

import collections as _collections


_IMAGE_CACHE: "_collections.OrderedDict[tuple[str, str, str], tuple[bytes, str, float]]" = _collections.OrderedDict()
_IMAGE_CACHE_LOCK = __import__("threading").Lock()
_IMAGE_CACHE_BYTES = 0
_IMAGE_CACHE_BUDGET_BYTES = 0  # set at startup by _log_manifest_cache_capacity


def _set_image_cache_budget(budget_bytes: int) -> None:
    global _IMAGE_CACHE_BUDGET_BYTES
    _IMAGE_CACHE_BUDGET_BYTES = max(0, int(budget_bytes))


def _image_cache_get(project_id: str, subdir: str, filename: str, disk_mtime: float) -> tuple[bytes, str] | None:
    """Return (bytes, content_type) on cache hit OR None on miss /
    stale (mtime later than cached). Promotes the entry to the
    most-recently-used slot on hit."""
    key = (project_id, subdir, filename)
    with _IMAGE_CACHE_LOCK:
        entry = _IMAGE_CACHE.get(key)
        if entry is None:
            return None
        data, ctype, cached_mtime = entry
        if cached_mtime != disk_mtime:
            # File was rewritten; drop and miss.
            global _IMAGE_CACHE_BYTES
            _IMAGE_CACHE_BYTES -= len(data)
            del _IMAGE_CACHE[key]
            return None
        # Promote to MRU end.
        _IMAGE_CACHE.move_to_end(key)
        return data, ctype


def _image_cache_put(project_id: str, subdir: str, filename: str, data: bytes, ctype: str, disk_mtime: float) -> None:
    """Insert into the cache and evict LRU entries until total bytes
    fit under the budget. Skips entries individually larger than
    half the budget so a single 500 MB upload can't blow out the
    whole cache."""
    if _IMAGE_CACHE_BUDGET_BYTES <= 0:
        return
    if len(data) > _IMAGE_CACHE_BUDGET_BYTES // 2:
        return
    key = (project_id, subdir, filename)
    global _IMAGE_CACHE_BYTES
    with _IMAGE_CACHE_LOCK:
        # Replace existing entry's byte count if present.
        old = _IMAGE_CACHE.pop(key, None)
        if old is not None:
            _IMAGE_CACHE_BYTES -= len(old[0])
        _IMAGE_CACHE[key] = (data, ctype, disk_mtime)
        _IMAGE_CACHE_BYTES += len(data)
        # Evict from the LRU end until we're under budget.
        while _IMAGE_CACHE_BYTES > _IMAGE_CACHE_BUDGET_BYTES and _IMAGE_CACHE:
            _, evicted = _IMAGE_CACHE.popitem(last=False)
            _IMAGE_CACHE_BYTES -= len(evicted[0])


_CONTENT_TYPE_BY_EXT = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".avif": "image/avif",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}


async def _serve_cached_image(project_id: str, subdir: str, filename: str, full_path: "Path"):
    """Common path for /api/v2/projects/{id}/{subdir}/{filename} and
    similar V1 endpoints. Returns FastAPI Response with the bytes
    served from the LRU image cache when warm; falls back to disk
    read on miss + populates the cache for next time.

    Sets Cache-Control + Last-Modified + ETag so the browser keeps
    bytes around across navigations. References + imports are
    content-addressed by UUID/hash filename, so the same URL never
    changes its bytes — safe to allow long-lived browser caching.
    The cover-thumb endpoint and labelled-preview endpoint set
    `immutable` via the ?v=<updatedAt> cachebuster the FE appends,
    which is fine: the URL changes when the bytes change.

    Cold reads run on the default executor via asyncio.to_thread so
    a 963-tile gallery first-load doesn't serialise the disk reads on
    the FastAPI event loop. Cache hits return immediately without
    touching disk at all.
    """
    from fastapi.responses import Response, FileResponse
    if not full_path.exists():
        raise HTTPException(404)
    try:
        st = full_path.stat()
        disk_mtime = st.st_mtime
    except OSError:
        disk_mtime = 0.0
    # ETag built from mtime + size — cheap, stable, changes on edit.
    etag = f'W/"{int(disk_mtime)}-{st.st_size}"' if disk_mtime else None
    extra_headers = {
        # 1 hour in shared caches, 1 day in the user's browser.
        # The bytes ARE keyed by filename (UUID) but we leave a
        # validation window so a manual on-disk swap recovers
        # without users having to clear their cache.
        "Cache-Control": "public, max-age=86400, stale-while-revalidate=3600",
    }
    if etag:
        extra_headers["ETag"] = etag
    hit = _image_cache_get(project_id, subdir, filename, disk_mtime)
    ext = full_path.suffix.lower()
    ctype = _CONTENT_TYPE_BY_EXT.get(ext, "application/octet-stream")
    if hit is not None:
        data, cached_ctype = hit
        return Response(content=data, media_type=cached_ctype or ctype, headers=extra_headers)
    # Cache miss — read off-loop so concurrent thumbnail loads don't
    # block each other while one disk read is in flight. The default
    # executor has multiple threads so dozens of reads can run in
    # parallel without choking the event loop.
    try:
        data = await asyncio.to_thread(full_path.read_bytes)
    except Exception as e:
        print(f"[image-cache] read failed for {full_path}: {e}")
        return FileResponse(str(full_path), headers=extra_headers)
    _image_cache_put(project_id, subdir, filename, data, ctype, disk_mtime)
    return Response(content=data, media_type=ctype, headers=extra_headers)


# Same hex palette as frontend/app/v2/OnboardLabelsV2.tsx — keeps
# the labelled-preview tints visually consistent with the chips
# the user sees in the UI. Order matches LABEL_COLOURS exactly so
# `colour_for_label` reproduces the FE's index-based assignment.
_LABEL_RGB: list[tuple[int, int, int]] = [
    (251, 146, 60),   # orange-400
    (96, 165, 250),   # blue-400
    (167, 139, 250),  # violet-400
    (52, 211, 153),   # emerald-400
    (244, 114, 182),  # pink-400
    (250, 204, 21),   # yellow-400
    (45, 212, 191),   # teal-400
    (248, 113, 113),  # red-400
]


def _label_rgb(label: str, project_labels: list[str]) -> tuple[int, int, int]:
    """Pick the RGB tuple a given label should render in. Mirrors
    `colourForLabel` from OnboardLabelsV2.tsx — labels present in the
    project's tag list use the index-into-tags slot; unknown labels
    hash-fall back so they still pick a stable colour."""
    if not label:
        return _LABEL_RGB[0]
    lk = label.strip().lower()
    for i, t in enumerate(project_labels):
        if t.strip().lower() == lk:
            return _LABEL_RGB[i % len(_LABEL_RGB)]
    h = 0
    for ch in label:
        h = (h * 31 + ord(ch)) & 0xFFFFFFFF
    return _LABEL_RGB[h % len(_LABEL_RGB)]


def _vary_label_rgb(
    base_rgb: tuple[int, int, int],
    seed: int,
    *,
    hue_jitter: float = 0.05,   # ±18° on the 360° wheel
    sat_jitter: float = 0.10,
    val_jitter: float = 0.12,
) -> tuple[int, int, int]:
    """Deterministically jitter a base RGB colour in HSV space so
    multiple objects sharing the same label render with subtly
    different shades. The seed drives all three jitter axes through
    independent splash values so the same detection always picks the
    same shade across re-renders.

    Defaults stay small enough that the user still reads the family
    colour at a glance — a hare next to another hare looks like
    "two hares", not "two random objects".
    """
    import colorsys
    r, g, b = base_rgb
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    # Three independent splashes from one seed via Knuth-multiplicative
    # hashing. & 0xFFFFFFFF keeps the constants in 32-bit space so the
    # output is the same across machines.
    s_int = seed & 0xFFFFFFFF
    rnd_h = (((s_int * 2654435761) & 0xFFFFFFFF) / 0xFFFFFFFF) * 2.0 - 1.0
    rnd_s = (((s_int * 40503) & 0xFFFFFFFF) / 0xFFFFFFFF) * 2.0 - 1.0
    rnd_v = (((s_int * 2246822519) & 0xFFFFFFFF) / 0xFFFFFFFF) * 2.0 - 1.0
    h = (h + rnd_h * hue_jitter) % 1.0
    s = max(0.0, min(1.0, s + rnd_s * sat_jitter))
    v = max(0.0, min(1.0, v + rnd_v * val_jitter))
    nr, ng, nb = colorsys.hsv_to_rgb(h, s, v)
    return (int(nr * 255), int(ng * 255), int(nb * 255))


def _polygon_area(polygon: list) -> float:
    """Shoelace area of a polygon given as a list of (x, y) pairs.
    Returns absolute area in pixel². Returns 0 for polygons with
    fewer than 3 vertices."""
    n = len(polygon)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x0, y0 = polygon[i][0], polygon[i][1]
        x1, y1 = polygon[(i + 1) % n][0], polygon[(i + 1) % n][1]
        s += float(x0) * float(y1) - float(x1) * float(y0)
    return abs(s) * 0.5


def _detection_polygon_area(d: dict) -> float:
    """Total mask area of a detection in pixel², summed across all
    polygon contours (handles holes / disjoint pieces consistently
    with the union mask we draw downstream)."""
    m = d.get("mask")
    if not isinstance(m, dict):
        return 0.0
    total = 0.0
    for poly in (m.get("polygons") or []):
        total += _polygon_area(poly)
    return total


def _size_ranked_rgb(rank: int, total: int) -> tuple[int, int, int]:
    """Pick an RGB tuple from a warm→cool ramp keyed on size rank.

    rank=0 is the largest detection, rank=total-1 the smallest. Hue
    sweeps orange (30°) → red (0°/360°) → magenta (310°) → purple
    (270°) → blue (230°), so a five-detection image reads as a clean
    "big-and-warm to small-and-cool" gradient. Larger detections also
    render slightly darker so the eye locks onto them as the primary
    subject; smaller ones get lighter to stay readable against the
    dimmed background.

    A single-detection image just renders dark orange (the "primary"
    end of the ramp) so a one-off looks the same as the largest of a
    multi-detection image."""
    import colorsys
    if total <= 1:
        h, s, v = 30.0 / 360.0, 0.95, 0.65
    else:
        t = rank / (total - 1)              # 0..1, large→small
        h_deg = (30.0 - t * 160.0) % 360.0  # 30 → 230 going via 0
        h = h_deg / 360.0
        s = 0.95 - t * 0.15                 # 0.95 → 0.80
        v = 0.62 + t * 0.23                 # 0.62 (darker) → 0.85
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))


def _random_rgb_from_seed(seed: int) -> tuple[int, int, int]:
    """Pick a saturated, value-high RGB tuple from a seed integer.
    Used by the labelled preview to give each segmentation a distinct
    instance colour, irrespective of label — the colour is just a
    visual separator, not a class indicator. Deterministic for a
    given seed so re-renders match (no flicker between cached and
    fresh previews)."""
    import colorsys
    s_int = seed & 0xFFFFFFFF
    h = (((s_int * 2654435761) & 0xFFFFFFFF) / 0xFFFFFFFF)            # full 0..1
    s = 0.65 + (((s_int * 40503) & 0xFFFFFFFF) / 0xFFFFFFFF) * 0.30    # 0.65..0.95
    v = 0.75 + (((s_int * 2246822519) & 0xFFFFFFFF) / 0xFFFFFFFF) * 0.20  # 0.75..0.95
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))


def _detection_seed(d: dict, idx: int) -> int:
    """Stable per-detection integer used to seed the per-object
    colour jitter. Combines the detection's index with a fingerprint
    of its first polygon point + box corner so a re-ordering of the
    detections array doesn't completely shuffle every object's
    colour, but two distinct detections in the same image always
    pick distinct seeds."""
    seed = (idx + 1) * 2654435761
    box = d.get("box") or []
    if isinstance(box, list) and len(box) >= 4:
        try:
            seed ^= int(round(float(box[0]))) * 73856093
            seed ^= int(round(float(box[1]))) * 19349663
        except (TypeError, ValueError):
            pass
    m = d.get("mask")
    if isinstance(m, dict):
        polys = m.get("polygons") or []
        if polys and len(polys[0]) > 0:
            try:
                p = polys[0][0]
                seed ^= int(round(float(p[0]))) * 83492791
                seed ^= int(round(float(p[1]))) * 1099511628211
            except (TypeError, ValueError, IndexError):
                pass
    return seed & 0xFFFFFFFF


# Label palette mirrored from app/v2/OnboardLabelsV2.tsx so the
# backend's labelled-preview tint + the FE's chip render share one
# palette. Same hex values in the same order, parsed once at import.
_LABEL_PALETTE_HEX = [
    "#fb923c",  # orange-400
    "#60a5fa",  # blue-400
    "#a78bfa",  # violet-400
    "#34d399",  # emerald-400
    "#f472b6",  # pink-400
    "#facc15",  # yellow-400
    "#2dd4bf",  # teal-400
    "#f87171",  # red-400
]


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    s = h.lstrip("#")
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))


_LABEL_PALETTE_RGB: list[tuple[int, int, int]] = [
    _hex_to_rgb(h) for h in _LABEL_PALETTE_HEX
]


def _label_hash_js(s: str) -> int:
    """Reproduce the FE's `_hash` (BoxEditor / OnboardLabelsV2)
    so canonical → palette-slot maps to the same colour on both
    sides. The `| 0` in the JS version forces signed-32-bit
    truncation; we emulate that with mask + branch."""
    h = 0
    for ch in s:
        h = (h * 31 + ord(ch)) & 0xFFFFFFFF
        if h >= 0x80000000:
            h -= 0x100000000
    return h


def _build_label_colour_map(
    all_labels: list[str],
) -> dict[str, tuple[int, int, int]]:
    """Project-scoped palette assignment with collision resolution.
    Mirrors the FE's `buildProjectLabelColourMap` exactly so a label
    rendered on the workspace card, the project view, AND the baked
    labelled-preview JPEG all show the same colour."""
    N = len(_LABEL_PALETTE_RGB)
    taken: set[int] = set()
    out: dict[str, tuple[int, int, int]] = {}
    for raw in all_labels:
        key = (raw or "").strip().lower()
        if not key or key in out:
            continue
        pref = abs(_label_hash_js(key)) % N
        slot = pref
        if pref in taken:
            best_slot = -1
            best_min_dist = -1
            for s in range(N):
                if s in taken:
                    continue
                min_dist = N
                for t in taken:
                    raw_d = abs(s - t)
                    d = min(raw_d, N - raw_d)
                    if d < min_dist:
                        min_dist = d
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_slot = s
            if best_slot >= 0:
                slot = best_slot
            else:
                ss = (pref + 1) % N
                while ss in taken and ss != pref:
                    ss = (ss + 1) % N
                slot = ss
        taken.add(slot)
        out[key] = _LABEL_PALETTE_RGB[slot]
    return out


def _render_labelled_preview(
    image_pil: "PILImage.Image",
    detections: list[dict],
    project_labels: list[str],
    *,
    max_side: int = 600,
    # Tuned to match the demo's segmented look: a darker, more heavily
    # blurred + desaturated backdrop so the bright, label-coloured
    # object cut-outs really pop.
    bg_dim: float = 0.28,        # 0 = pitch black, 1 = no dim
    bg_blur_radius: float = 5.0,
    bg_desat: float = 0.65,      # 0 = no desaturation, 1 = greyscale
    tint_strength: float = 0.34, # 0 = no tint, 1 = solid label colour
) -> "PILImage.Image":
    """Bake a small annotated preview where the segmented objects are
    bright with a slight label-colour tint and the background is
    darkened + softly blurred. Returns an RGB PIL image suitable for
    JPEG encoding.

    Detections without a polygon mask are skipped — there's nothing to
    composite. Rejected detections are skipped too so the cutouts only
    reflect labels the user kept.

    Falls back to a plain downsized copy when no kept detection
    contributes a polygon (e.g. an unlabelled image dropped into the
    dataset). The FE then still gets a small preview instead of the
    full-resolution original.
    """
    from PIL import ImageDraw, ImageFilter
    W, H = image_pil.size
    longest = max(W, H)
    scale = max_side / longest if longest > max_side else 1.0
    if scale < 1.0:
        small = image_pil.resize((int(W * scale), int(H * scale)), PILImage.LANCZOS)
    else:
        small = image_pil.copy()
    base = small.convert("RGB")

    kept = [
        d for d in detections
        if isinstance(d, dict)
        and not d.get("rejected")
        and isinstance(d.get("mask"), dict)
        and (d.get("mask") or {}).get("polygons")
    ]
    if not kept:
        return base

    # Union mask used to composite bright foreground over dim bg.
    union_mask = PILImage.new("L", base.size, 0)
    drw = ImageDraw.Draw(union_mask)
    for d in kept:
        for polygon in (d["mask"].get("polygons") or []):
            if len(polygon) < 3:
                continue
            pts = [(int(p[0] * scale), int(p[1] * scale)) for p in polygon]
            drw.polygon(pts, fill=255)

    # Darken, lightly desaturate, and softly blur the background so
    # the eye locks onto the bright cutouts. Desaturate FIRST (mix
    # toward greyscale) then darken+blur — keeps the colour pull on
    # the foreground without making the bg feel washed out. The
    # foreground composite uses the original `base` so the kept
    # objects stay full-saturation regardless of what we do here.
    bg_layer = base
    if bg_desat > 0:
        # Convert to luminance, then back to RGB so we can blend.
        # PIL's "L" → "RGB" gives a true greyscale frame; partial
        # blend = partial desaturation.
        grey = base.convert("L").convert("RGB")
        bg_layer = PILImage.blend(base, grey, bg_desat)
    black = PILImage.new("RGB", base.size, (0, 0, 0))
    dimmed = PILImage.blend(bg_layer, black, 1.0 - bg_dim)
    if bg_blur_radius > 0:
        dimmed = dimmed.filter(ImageFilter.GaussianBlur(radius=bg_blur_radius))
    feather = union_mask.filter(ImageFilter.GaussianBlur(radius=1.0))
    composite = PILImage.composite(base, dimmed, feather)

    # Per-detection tint, label-coloured. Each detection picks up
    # its label's project-palette colour so the baked preview
    # matches the chips the user sees on the workspace card +
    # project view. Falls back to a neutral grey when a detection
    # has no label (rare — pre-resolver / SAM-only crops). Old
    # cached previews are not re-baked; the new colour scheme only
    # applies to fresh bakes (cache file path version stays at v3).
    if tint_strength > 0:
        label_colour_map = _build_label_colour_map(project_labels)
        for d in kept:
            lab = (
                d.get("pred_label")
                or d.get("gd_label")
                or ""
            )
            colour = label_colour_map.get(lab.strip().lower(), (200, 200, 200))
            obj_mask = PILImage.new("L", base.size, 0)
            drw_obj = ImageDraw.Draw(obj_mask)
            for polygon in (d["mask"].get("polygons") or []):
                if len(polygon) < 3:
                    continue
                pts = [(int(p[0] * scale), int(p[1] * scale)) for p in polygon]
                drw_obj.polygon(pts, fill=255)
            obj_mask = obj_mask.filter(ImageFilter.GaussianBlur(radius=1.0))
            tint_layer = PILImage.new("RGB", base.size, colour)
            tinted = PILImage.blend(composite, tint_layer, tint_strength)
            composite = PILImage.composite(tinted, composite, obj_mask)

    return composite


def _labelled_preview_path(project_id: str, import_id: str) -> "Path":
    """Cache location for the rendered labelled preview. Keyed by
    import_id so renames / re-uploads of the underlying file don't
    collide. Versioned filename so future tint / dim tweaks invalidate
    the whole project's previews on next request."""
    d = project_dir(project_id) / "thumbs"
    d.mkdir(exist_ok=True)
    return d / f"{import_id}__lp_v4.jpg"


def _invalidate_labelled_preview(project_id: str, import_id: str) -> None:
    """Delete the cached labelled preview for one import. Called on
    PUT (label edits) and DELETE so the next /labelled_preview GET
    re-renders from the latest detections."""
    try:
        _labelled_preview_path(project_id, import_id).unlink(missing_ok=True)
    except Exception as e:
        print(f"[labelled-preview] invalidate failed for {project_id}/{import_id}: {e}")


# Per-(project, kind) lock registry so concurrent first-renders of the
# same thumb don't stomp on each other's tmp files. asyncio.Lock keyed
# on (project_id, kind) where kind is e.g. "labelled_preview:<import_id>"
# or "cover_thumb". Created lazily; lives for the process lifetime
# (handful of bytes per project, fine).
_THUMB_RENDER_LOCKS: dict[tuple[str, str], asyncio.Lock] = {}


def _thumb_render_lock(project_id: str, kind: str) -> asyncio.Lock:
    key = (project_id, kind)
    lock = _THUMB_RENDER_LOCKS.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _THUMB_RENDER_LOCKS[key] = lock
    return lock


def _unique_tmp_suffix() -> str:
    """Per-call tmp suffix — pid alone collides when the same worker
    process fields two simultaneous first-renders of the same thumb,
    overwriting each other's bytes. _uuid + pid keeps the suffix
    stable per call AND unique."""
    return f".{os.getpid()}.{_uuid.uuid4().hex[:8]}.tmp.jpg"


def _bake_labelled_preview_sync(
    project_id: str,
    import_id: str,
    src_path: "Path",
    detections: list[dict],
    project_labels: list[str],
) -> None:
    """Render the labelled preview JPEG and persist it to disk. Runs
    off the request thread (loop.run_in_executor). Never raises —
    bake failures fall back to the lazy GET path automatically.

    Called eagerly after upload + edit so the first /labelled_preview
    request is a pure file serve with no per-pixel work — the user's
    drop-zone progress indicator gets to update at network speed.
    """
    cached_path = _labelled_preview_path(project_id, import_id)
    try:
        with PILImage.open(src_path) as im:
            im = im.convert("RGB")
            preview = _render_labelled_preview(im, detections, project_labels)
        cached_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cached_path.with_suffix(_unique_tmp_suffix())
        preview.save(tmp, format="JPEG", quality=72, optimize=True, progressive=True)
        tmp.replace(cached_path)
    except Exception as e:
        print(f"[labelled-preview] bake failed for {project_id}/{import_id}: {e}")


def _encode_blurhash_from_path(path: "Path") -> str | None:
    """Generic blurhash encoder. Reads + thumbnails + encodes any
    image file at `path`. Returns the ~30-char hash string or None
    on any failure (missing file, decode error, encode error)."""
    if not path.exists():
        return None
    try:
        import blurhash as _bh
        from PIL import Image as _PIL
        with _PIL.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((64, 64))
            try:
                return _bh.encode(im, 4, 3)
            except TypeError:
                import numpy as _np
                arr = _np.asarray(im, dtype=_np.uint8)
                try:
                    return _bh.encode(arr, 4, 3)
                except TypeError:
                    return _bh.encode(arr, x_components=4, y_components=3)
    except Exception as e:
        print(f"[blurhash] encode failed for {path}: {e}")
        return None


def _blurhash_backfill_async(project_id: str, items: list[tuple[str, str]]) -> None:
    """Encode missing blurhashes off the request thread and persist
    them back to the manifest. Items is a list of (subdir, filename)
    tuples — we walk them, encode each via the on-disk path, write
    results to the in-memory cache and the manifest. Never raises;
    logs failures so the next read just re-tries.

    Called fire-and-forget from get_project so the GET response
    isn't blocked on PIL encodes. The next manifest read picks up
    whatever this loop has finished writing.
    """
    if not items:
        return
    encoded: list[tuple[str, str, str]] = []  # (subdir, filename, hash)
    for subdir, fn in items:
        try:
            path = project_dir(project_id) / subdir / fn
            h = _encode_blurhash_from_path(path)
            if h:
                _BLURHASH_CACHE[(project_id, f"{subdir}/{fn}")] = h
                encoded.append((subdir, fn, h))
        except Exception as e:
            print(f"[blurhash-async] encode failed for {project_id}/{subdir}/{fn}: {e}")
    if not encoded:
        return
    # Persist to manifest under the per-project write lock so a
    # concurrent reference / import POST doesn't get clobbered.
    # _manifest_write_lock returns an asyncio.Lock — but we're on a
    # worker thread here. Use the sync lock pattern: load → mutate →
    # save in a tight critical section, treating concurrent edits
    # as best-effort (worst case the next backfill round picks up
    # any drift).
    try:
        manifest = load_manifest(project_id) or {}
        by_subdir = {"references": "references", "imports": "imports"}
        for subdir, fn, h in encoded:
            for entry in manifest.get(by_subdir.get(subdir, subdir)) or []:
                if isinstance(entry, dict) and entry.get("filename") == fn and not entry.get("blurhash"):
                    entry["blurhash"] = h
        save_manifest(project_id, manifest)
        print(f"[blurhash-async] backfilled {len(encoded)} hash(es) for {project_id}")
    except Exception as e:
        print(f"[blurhash-async] persist failed for {project_id}: {e}")


def _cached_blurhash(project_id: str, filename: str, subdir: str) -> str | None:
    """Return a cached blurhash for `<project_id>/<subdir>/<filename>`,
    encoding it on first miss. Used for references and imports — same
    cache as the cover hash so a file used as both gets a single
    encode. Cache key namespaces by subdir so file names that
    collide between references/ and imports/ don't share entries."""
    if not filename:
        return None
    key = (project_id, f"{subdir}/{filename}")
    cached = _BLURHASH_CACHE.get(key)
    if cached is not None:
        return cached
    path = project_dir(project_id) / subdir / filename
    h = _encode_blurhash_from_path(path)
    if h is not None:
        _BLURHASH_CACHE[key] = h
    return h


def _compute_cover_blurhash(project_id: str, cover_filename: str, *, is_v2: bool) -> str | None:
    """Return a 4×3-component BlurHash string for the project's cover.

    BlurHash (https://blurha.sh) packs a 30-ish-char string that the FE
    decodes into a small color gradient. Renders instantly while the
    real cover image streams over the network — same trick Wolt /
    Unsplash use to avoid blank tiles on slow connections.

    Cached in `_BLURHASH_CACHE` so subsequent /api/projects polls don't
    re-encode. Returns None when the cover file is missing or the
    encoder fails — caller treats None as "no placeholder, fall back to
    plain background".
    """
    if not cover_filename:
        return None
    key = (project_id, cover_filename)
    cached = _BLURHASH_CACHE.get(key)
    if cached is not None:
        return cached

    subdir = "references" if is_v2 else "imports"
    img_path = project_dir(project_id) / subdir / cover_filename
    if not img_path.exists():
        return None

    try:
        import blurhash as _bh
        from PIL import Image as _PIL
        with _PIL.open(img_path) as im:
            im = im.convert("RGB")
            # Encode at low resolution — blurhash stretches a few
            # components into a smooth gradient, so feeding it a
            # 64-px thumbnail is more than enough and keeps the
            # encode under ~5 ms per image.
            im.thumbnail((64, 64))
            # The PyPI `blurhash` package's `encode` accepts a PIL
            # image OR a file path and takes (x, y) components as
            # POSITIONAL args. Earlier code passed them as kwargs,
            # which the C wrapper rejects with TypeError. Try the
            # PIL-direct positional call first; fall back to numpy
            # in case a different blurhash flavour is installed.
            try:
                hash_str = _bh.encode(im, 4, 3)
            except TypeError:
                import numpy as _np
                arr = _np.asarray(im, dtype=_np.uint8)
                try:
                    hash_str = _bh.encode(arr, 4, 3)
                except TypeError:
                    # Last-resort: kwargs (the halcy/blurhash-python
                    # package). Worst case all three fail and we
                    # bail to None below.
                    hash_str = _bh.encode(arr, x_components=4, y_components=3)
    except Exception as e:
        print(f"[blurhash] encode failed for {project_id}/{cover_filename}: {e}")
        return None

    _BLURHASH_CACHE[key] = hash_str
    return hash_str


@app.get("/api/projects")
async def list_projects(
    owner: str | None = None,
    viewer: str | None = None,
    offset: int = 0,
    limit: int | None = None,
    sort: str | None = None,
    q: str | None = None,
):
    """Pass `?owner=<username>` to scope to a single user's workspaces.
    Without it, returns every project (community feed). `?viewer=<username>`
    populates `likedByMe` per project so the heart button knows its state.

    Pagination: pass `?limit=N&offset=K` and the response shape changes
    from `list[item]` to `{total: int, items: list[item], offset, limit}`
    — the public projects page uses this so the first 12 cards render
    instantly while the rest stream in via infinite scroll. Without
    `limit` the response stays a plain list (back-compat with the
    workspace's existing 4-second poll).

    Each item carries a `cover_blurhash` string when a cover is set
    and the cover file is on disk. The FE decodes it client-side
    into a small colour gradient that fills the card slot
    immediately, then crossfades the real image in once it loads."""
    # Response cache check — the FE polls this every 4 s and the
    # answer rarely changes within that window. The cache also
    # stat-watches every contributing manifest so an upload / edit
    # invalidates immediately, keeping staleness within the
    # cache-fill window (~ms) regardless of TTL.
    q_norm = (q or "").strip().lower()
    _cache_key = (owner or "", viewer or "", int(offset), -1 if limit is None else int(limit), sort or "", q_norm)
    _cached = _projects_response_cache_get(_cache_key)
    if _cached is not None:
        return _cached

    # Snapshot the JobManager once so we can flag projects with a job
    # currently queued or running. Used by the frontend to render the
    # "In progress" badge — independent of whether some images are
    # already labelled (the "Partial" / "Unlabelled" badges).
    running_projects: set[str] = set()
    try:
        for j in state["jobs"].jobs.values():
            if j.status in ("queued", "running") and j.project:
                running_projects.add(j.project)
    except Exception as e:
        print(f"[list_projects] failed to read job state: {e}")
    items = []
    # Reverse map dataset_id -> {id, name, private} for every Project (container).
    # Built once up-front so the privacy filter below can treat a dataset as
    # private when its container is private even if the manifest's own flag
    # drifted (failed write / legacy data), and so each card can carry a
    # clickable Project chip. Fails open to {} (chip absent, manifest flag still
    # enforced) on error.
    try:
        _ds_container = containers.dataset_container_index()
    except Exception as e:
        print(f"[list_projects] container index failed: {e}")
        _ds_container = {}

    def _eff_private(ds_id: str, own_private: bool) -> bool:
        info = _ds_container.get(str(ds_id))
        return bool(own_private) or (bool(info.get("private")) if info else False)

    # Tracks the on-disk mtime of every manifest we touched while
    # building this response, so the response cache can later check
    # those files and invalidate the moment any of them changes.
    cache_mtimes: dict[str, float] = {}
    for _pid in _iter_project_ids():
        manifest_mtime = _manifest_disk_mtime(_pid)
        cache_mtimes[_pid] = manifest_mtime
        # Fast path: the per-project card sidecar carries all the
        # static-per-write card fields. Skips the multi-MB manifest
        # read entirely when it's fresh — the workspace's 4 s poll
        # used to hit `load_manifest` 33+ times on every tick.
        card = _read_workspace_card_sidecar(_pid)
        if card is not None:
            try:
                card_mtime = _workspace_card_sidecar_path(_pid).stat().st_mtime
            except OSError:
                card_mtime = 0.0
            if card_mtime >= manifest_mtime:
                proj_id = card.get("id") or _pid
                proj_owner = card.get("owner") or ""
                if owner is not None and proj_owner != owner:
                    continue
                if _eff_private(proj_id, bool(card.get("private"))) and proj_owner and viewer != proj_owner:
                    continue
                if owner is None and (card.get("n_images") or 0) <= 0:
                    continue
                liked_by = card.get("likedBy") or []
                fav_by = card.get("favouritedBy") or []
                items.append({
                    **{k: v for k, v in card.items() if k not in ("likedBy", "favouritedBy")},
                    "likes": len(liked_by),
                    "likedByMe": bool(viewer and viewer in liked_by),
                    "favourites": len(fav_by),
                    "favouritedByMe": bool(viewer and viewer in fav_by),
                    "running": proj_id in running_projects,
                })
                continue
        # Slow path: sidecar missing or stale, fall back to a full
        # manifest read. Same loop body as before. Kick an async
        # rebuild so the next request hits the fast path — bulk-
        # warming via _kick_sidecar_refresh would also write
        # overview + initial sidecars we don't need here, so do the
        # workspace-card one directly. Dedup via a per-project guard.
        _kick_workspace_card_refresh(_pid)
        # Read-only consumer — skip the deepcopy. The async sidecar
        # write (kicked above) is the only mutation downstream and it
        # works from its own load_manifest call.
        manifest = load_manifest(_pid, copy=False)
        proj_id = manifest.get("id") or _pid
        display_name = manifest.get("name") or _pid
        proj_owner = manifest.get("owner") or manifest.get("createdBy") or ""
        if owner is not None and proj_owner != owner:
            continue
        # Private projects are hidden from anyone except their owner. Owners
        # still see them in the community feed if they happen to land there.
        # A dataset in a private Project counts as private even if its own flag
        # drifted (see _eff_private).
        if _eff_private(proj_id, bool(manifest.get("private"))) and proj_owner and viewer != proj_owner:
            continue
        results = manifest.get("results", []) or []
        n_labelled = sum(1 for r in results if not r.get("pending"))
        n_unlabelled = sum(1 for r in results if r.get("pending"))
        # V2 projects keep dataset images in `imports`, not `results`,
        # so the workspace card's "Images" count reads zero on V2
        # without this. For V2 we also derive labelled / unlabelled
        # from each entry's explicit `labelled` flag (set by the
        # label_charlie job) — falling back to whether the entry
        # has any detections for older imports that pre-date the flag.
        if manifest.get("v2"):
            v2_imports = manifest.get("imports") or []
            n_images_v2 = len(v2_imports)

            def _v2_is_labelled(entry: dict) -> bool:
                if not isinstance(entry, dict):
                    return False
                # See the inner copy above (search this file for
                # editedBoxesSet) for the rationale.
                if entry.get("editedBoxesSet"):
                    edited = entry.get("editedBoxes")
                    return isinstance(edited, list) and len(edited) > 0
                flag = entry.get("labelled")
                if flag is True:
                    return True
                if flag is False:
                    return False
                return bool(entry.get("detections"))

            n_labelled = sum(1 for e in v2_imports if _v2_is_labelled(e))
            n_unlabelled = max(0, n_images_v2 - n_labelled)
        else:
            n_images_v2 = None
        cover = manifest.get("cover")
        v2_refs = manifest.get("references") or []
        v2_imps = manifest.get("imports") or []
        cover_subdir: str | None = None
        if cover:
            # Identify which subdir the manifest's stored cover came
            # from so we know where to look for blurhash + serve.
            if any(r.get("filename") == cover for r in v2_refs):
                cover_subdir = "references"
            elif any(i.get("filename") == cover for i in v2_imps):
                cover_subdir = "imports"
            elif any(r.get("image") == cover for r in results):
                cover_subdir = "imports"  # V1 stores under imports/
            else:
                cover = None  # cover refs a file that no longer exists
        if not cover:
            # Pick a random cover so each card stays varied / fresh
            # rather than always showing the first uploaded image.
            # Seed the random pick on `proj_id` so the same project
            # gets the SAME cover across calls (and across pagination
            # pages) — without that, the workspace card thumbnail
            # would shuffle every 4 s poll and look broken.
            #
            # Priority: V2 references first (the user's curated set);
            # fall through to V2 imports (dataset images), then V1
            # results, then None.
            import random as _rnd
            seed = _rnd.Random(proj_id)
            ref_files = [r.get("filename") for r in v2_refs if r.get("filename")]
            imp_files = [i.get("filename") for i in v2_imps if i.get("filename")]
            v1_files = [r.get("image") for r in results if r.get("image")]
            if ref_files:
                cover = seed.choice(ref_files)
                cover_subdir = "references"
            elif imp_files:
                cover = seed.choice(imp_files)
                cover_subdir = "imports"
            elif v1_files:
                cover = seed.choice(v1_files)
                cover_subdir = "imports"
            else:
                cover = None
                cover_subdir = None
        # Empty-project filter for the PUBLIC feed only. The FE used
        # to drop these client-side which made `total` lie ("Showing
        # 5 of 10" when the other 5 are filtered-out blanks). Apply
        # at the source so the count matches what users actually
        # see. Owner-scoped queries (workspace) keep showing empty
        # projects so the user can keep working on them.
        n_images_resolved = n_images_v2 if n_images_v2 is not None else len(results)
        if owner is None and n_images_resolved <= 0:
            continue
        liked_by = manifest.get("likedBy") or []
        fav_by = manifest.get("favouritedBy") or []
        items.append({
            "id": proj_id,
            "name": display_name,
            "owner": proj_owner,
            "createdAt": manifest.get("createdAt"),
            "updatedAt": manifest.get("updatedAt"),
            "n_images": n_images_resolved,
            "n_labelled": n_labelled,
            "n_unlabelled": n_unlabelled,
            "n_references": len(v2_refs) if manifest.get("v2") else 0,
            "tags": collect_tags(manifest),
            # Display alias map (canonical_lower → renamed display)
            # — surfaced so the workspace + public cards can render
            # a label rename instantly without each one having to
            # fetch the full manifest. Empty dict if the user hasn't
            # renamed any labels yet.
            "label_aliases": dict(manifest.get("label_aliases") or {}),
            # Per-label colour overrides — surfaced on the list so the
            # workspace + public cards can paint chips in the user's
            # chosen colours without each card fetching the full
            # manifest. Empty dict when no overrides set.
            "labelColours": dict(manifest.get("labelColours") or {}),
            "thumbnail": cover,
            "hasModel": bool(manifest.get("hasModel", False)),
            "createdBy": proj_owner,
            "likes": len(liked_by),
            "likedByMe": bool(viewer and viewer in liked_by),
            "favourites": len(fav_by),
            "favouritedByMe": bool(viewer and viewer in fav_by),
            "certified": bool(manifest.get("certified", False)),
            "private": bool(manifest.get("private", False)),
            "derived": ({"parentProjectId": (manifest.get("derived") or {}).get("parentProjectId"),
                         "parentName": (manifest.get("derived") or {}).get("parentName")}
                        if manifest.get("derived") else None),
            "running": proj_id in running_projects,
            # Surfaced so the workspace can dispatch to the V2 page
            # for V2 projects without having to fetch each manifest.
            "v2": bool(manifest.get("v2", False)),
            # Cover blurhash filled in below so we only encode for
            # the slice the caller actually pages into. Default null.
            "cover_blurhash": None,
            # Internal: which subdir under projects/<id>/ the cover
            # lives in. Used by the blurhash backfill loop below;
            # popped before sending to the FE.
            "_cover_subdir": cover_subdir,
            # Backend-canonical dataset health so the FE doesn't
            # have to compute it from labels + refs (which were
            # both arriving asynchronously and producing a
            # "Bad → Okay → Good" flicker on first paint).
            "dataset_health": _compute_dataset_health(manifest),
        })
    # Attach each dataset's Project (container) so the workspace card can render
    # a clickable Project chip (index built once before the loop). None when the
    # dataset is not in any project.
    for it in items:
        info = _ds_container.get(str(it.get("id")))
        it["container"] = {"id": info["id"], "name": info["name"]} if info else None

    # Search filter — applied before sort + pagination so `total`
    # reflects matching count and infinite-scroll pages stay stable.
    # Matches as a case-insensitive substring across name, owner /
    # createdBy, raw tags, and the renamed display values from
    # label_aliases. Keeps the search server-side so it covers every
    # project, not just the page the FE has loaded.
    if q_norm:
        def _matches(it: dict) -> bool:
            haystack: list[str] = []
            for key in ("name", "owner", "createdBy"):
                v = it.get(key)
                if v:
                    haystack.append(str(v).lower())
            for t in (it.get("tags") or []):
                if t:
                    haystack.append(str(t).lower())
            for disp in (it.get("label_aliases") or {}).values():
                if disp:
                    haystack.append(str(disp).lower())
            return any(q_norm in h for h in haystack)
        items = [it for it in items if _matches(it)]

    # Sort modes — server-side so pagination boundaries stay stable
    # as the user scrolls. Python's sort is stable, so we layer the
    # sorts: first by the primary criterion, then a final pass that
    # pins viewer's favourites to the head. Without server-side
    # sorting the FE would re-sort each fetched page on top of the
    # cumulative list, shuffling later-page projects (more likes,
    # higher trending score) up into earlier slots.
    sort_mode = (sort or "trending").strip().lower()
    if sort_mode in ("most_liked", "most-liked", "likes"):
        items.sort(key=lambda x: (x.get("likes") or 0), reverse=True)
    elif sort_mode == "newest":
        items.sort(key=lambda x: x.get("createdAt") or "", reverse=True)
    else:
        # Trending: likes desc with updatedAt as tiebreaker.
        items.sort(key=lambda x: x.get("updatedAt") or "", reverse=True)
        items.sort(key=lambda x: (x.get("likes") or 0), reverse=True)
    if viewer:
        items.sort(key=lambda x: 0 if x.get("favouritedByMe") else 1)

    total = len(items)
    paginate = limit is not None and limit > 0
    if paginate:
        off = max(0, int(offset))
        lim = max(0, int(limit))
        page = items[off:off + lim]
    else:
        page = items

    # Encode blurhash only for the slice the caller is about to render.
    # 5-10 ms per project on first encode, then cached in
    # _BLURHASH_CACHE for the rest of the process lifetime — so the
    # 4-second workspace poll only pays the encode cost once per
    # cover-image change.
    for it in page:
        cover = it.get("thumbnail")
        subdir = it.get("_cover_subdir")
        if cover and subdir:
            try:
                it["cover_blurhash"] = _cached_blurhash(it["id"], cover, subdir)
            except Exception as e:
                print(f"[list_projects] blurhash skipped for {it['id']}: {e}")
    # Strip internal-only fields before serialising — _cover_subdir
    # was just to route the blurhash lookup, FE doesn't need it.
    for it in page:
        it.pop("_cover_subdir", None)

    if paginate:
        body = {"total": total, "items": page, "offset": off, "limit": lim}
    else:
        body = page
    # Stash for the next caller — TTL plus mtime watch keeps it
    # both fast and correct.
    _projects_response_cache_put(_cache_key, body, cache_mtimes)
    return body


class CreateProjectIn(BaseModel):
    name: str
    owner: str | None = None


@app.post("/api/projects")
async def create_project(payload: CreateProjectIn, user: str = Depends(current_user)):
    """Mint a fresh UUID for the project's identity (folder name + R2 prefix).
    The user-supplied name is stored in the manifest as a freeform display
    label — duplicate names across users (or even within one user) are fine.
    Owner is forced to the authenticated user — clients can't spoof the
    `owner` field by sending someone else's username in the payload."""
    from profanity import assert_clean

    display_name = (payload.name or "").strip() or "Untitled project"
    assert_clean(display_name, field="project name")
    project_id = _uuid.uuid4().hex
    while store.dataset_exists(project_id):
        project_id = _uuid.uuid4().hex
    save_manifest(
        project_id,
        empty_manifest(display_name, owner=user, project_id=project_id),
    )
    return {"id": project_id, "name": display_name}


# ============================================================================
# Project containers (teams). A "Project" (UI term) is a CONTAINER of datasets
# + members (gd/containers.py). Routes here are the CRUD + membership + dataset-
# membership surface. Access: read = member-or-public, manage = container owner.
# Member-add emails are sent FRONTEND-side (the backend has no user-email store;
# emails live in the NextAuth Postgres). add_event() rows feed the Phase-2
# activity feed and always record the acting user (originator, not owner).
# ============================================================================

def _cascade_container_privacy(container: dict) -> None:
    """Push the container's privacy onto every dataset in it (datasets in a
    Project can't have a different privacy to the Project)."""
    priv = bool(container.get("private"))
    for did in (container.get("dataset_ids") or []):
        m = load_manifest(did)
        if m and bool(m.get("private")) != priv:
            m["private"] = priv
            save_manifest(did, m)


def _cascade_container_max_input(container: dict) -> None:
    """Push the container's max input image size onto every dataset in it so the
    upload resize uses the Project's ceiling (read off the dataset manifest)."""
    size = containers.clamp_max_input(container.get("max_input_size"))
    for did in (container.get("dataset_ids") or []):
        m = load_manifest(did)
        if m and int(m.get("max_input_size") or 0) != size:
            m["max_input_size"] = size
            save_manifest(did, m)


def _container_card(c: dict) -> dict:
    """Compact shape for the workspace grid: cover, owner, last-updated, count."""
    return {
        "id": c.get("id"),
        "name": c.get("name"),
        "owner": c.get("owner"),
        "private": bool(c.get("private")),
        "cover": c.get("cover"),
        "max_input_size": containers.clamp_max_input(c.get("max_input_size")),
        "n_datasets": len(c.get("dataset_ids") or []),
        "n_members": len(c.get("members") or []),
        "updated": c.get("updated"),
        "created": c.get("created"),
    }


def _container_detail(c: dict, username: str | None) -> dict:
    """Full Project page payload: members, datasets, my role. Models + activity
    are layered on in Phase 2/3."""
    datasets = []
    for did in (c.get("dataset_ids") or []):
        m = load_manifest(did, copy=False) or {}
        datasets.append({
            "id": did,
            "name": m.get("name") or did,
            "cover": m.get("cover"),
            "private": bool(m.get("private")),
            "n_images": len(m.get("imports") or m.get("results") or []),
            "hasModel": bool((m.get("model") or {}).get("weights")),
            "updated": m.get("updated"),
            # Creator handle — drives the FE's per-dataset delete permissions
            # (only the creator can destroy; the Project owner can detach).
            "owner": (m.get("owner") or ""),
            # True when this dataset is a derived (cropped child) of another, so
            # the Project page can mark it with the derived branch icon.
            "derived": bool(m.get("derived")),
        })
    return {
        **c,
        "my_role": containers.member_role(c, username or ""),
        "datasets": datasets,
    }


class ContainerCreateIn(BaseModel):
    name: str
    private: bool = True


@app.post("/api/containers")
async def container_create(payload: ContainerCreateIn, user: str = Depends(current_user)):
    from profanity import assert_clean
    name = (payload.name or "").strip() or "Untitled project"
    assert_clean(name, field="project name")
    c = containers.create_container(name, owner=user, private=bool(payload.private))
    add_event("container_create", container=c["id"], actor=user, name=name)
    return c


@app.get("/api/containers")
async def container_list(user: str = Depends(current_user)):
    cards = [_container_card(c) for c in containers.list_containers_for_user(user)]
    cards.sort(key=lambda c: (c.get("updated") or ""), reverse=True)
    return {"containers": cards}


# Declared BEFORE /api/containers/{container_id} so the static "public" segment
# isn't captured as a container id.
@app.get("/api/containers/public")
async def container_list_public():
    """Public (non-private) Projects for the Community carousel. No auth, since
    the Community feed is viewable by anyone. Only Projects with at least one
    dataset are returned so the carousel has nothing empty in it."""
    cards = [_container_card(c) for c in containers.list_public_containers()]
    cards = [c for c in cards if (c.get("n_datasets") or 0) > 0]
    cards.sort(key=lambda c: (c.get("updated") or ""), reverse=True)
    return {"containers": cards}


@app.get("/api/containers/{container_id}")
async def container_get(container_id: str, request: Request, authorization: str = Header(default="")):
    c = containers.load_container(container_id)
    if not c:
        raise HTTPException(404, "project not found")
    username = request_username(request, authorization)
    if not containers.can_read(c, username):
        # 404 (not 403) so a private container's existence can't be probed.
        raise HTTPException(404, "project not found")
    # Lazy-backfill the cover's bottom-band luminance for covers uploaded before
    # we started storing it, so the Project page can pick a black/white title
    # from the actual image (not just the theme). Computed once; doesn't bump
    # `updated` so viewing a Project never reorders the workspace.
    if c.get("cover") and c.get("coverLuma") is None:
        try:
            from PIL import ImageStat as _ImageStat
            data = r2_required().get_bytes(_container_cover_key(container_id))
            with PILImage.open(io.BytesIO(data)) as _im:
                _im = _im.convert("RGB")
                _w, _h = _im.size
                _band = _im.crop((0, int(_h * 0.5), _w, _h)).convert("L")
                c["coverLuma"] = round(_ImageStat.Stat(_band).mean[0])
        except Exception as e:
            print(f"[container cover-luma backfill] {container_id}: {e}")
            c["coverLuma"] = 0  # mark done so we don't retry on every request
        try:
            containers.save_container(c, bump_updated=False)
        except Exception:
            pass
    return _container_detail(c, username)


class ContainerPatchIn(BaseModel):
    name: str | None = None
    private: bool | None = None
    cover: str | None = None
    max_input_size: int | None = None


@app.patch("/api/containers/{container_id}")
async def container_patch(container_id: str, payload: ContainerPatchIn, user: str = Depends(current_user)):
    c = containers.load_container(container_id)
    if not c:
        raise HTTPException(404, "project not found")
    if not containers.can_manage(c, user):
        raise HTTPException(403, "requires owner")
    if payload.name is not None:
        from profanity import assert_clean
        nm = (payload.name or "").strip() or "Untitled project"
        assert_clean(nm, field="project name")
        c["name"] = nm[:120]
    if payload.cover is not None:
        c["cover"] = payload.cover or None
        c["cover_updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if payload.private is not None and bool(payload.private) != bool(c.get("private")):
        c["private"] = bool(payload.private)
        _cascade_container_privacy(c)  # datasets inherit the new privacy
        add_event("container_privacy", container=container_id, actor=user, private=bool(payload.private))
    if payload.max_input_size is not None:
        new_size = containers.clamp_max_input(payload.max_input_size)
        if new_size != int(c.get("max_input_size") or containers.MAX_INPUT_DEFAULT):
            c["max_input_size"] = new_size
            _cascade_container_max_input(c)  # datasets inherit the new ceiling
            add_event("container_max_input", container=container_id, actor=user, max_input_size=new_size)
    containers.save_container(c)
    return c


@app.delete("/api/containers/{container_id}")
async def container_delete(container_id: str, user: str = Depends(current_user)):
    c = containers.load_container(container_id)
    if not c:
        raise HTTPException(404, "project not found")
    if not containers.can_manage(c, user):
        raise HTTPException(403, "requires owner")
    # Detach datasets (they survive as standalone) rather than delete them.
    for did in (c.get("dataset_ids") or []):
        m = load_manifest(did)
        if m and (m.get("container_id") or "") == container_id:
            m["container_id"] = ""
            save_manifest(did, m)
    containers.delete_container(container_id)
    add_event("container_delete", container=container_id, actor=user)
    return {"ok": True}


class ContainerMemberIn(BaseModel):
    username: str
    role: str = "editor"


@app.post("/api/containers/{container_id}/members")
async def container_add_member(container_id: str, payload: ContainerMemberIn, user: str = Depends(current_user)):
    c = containers.load_container(container_id)
    if not c:
        raise HTTPException(404, "project not found")
    if not containers.can_manage(c, user):
        raise HTTPException(403, "requires owner")
    target = (payload.username or "").strip().lower()
    if not target:
        raise HTTPException(400, "username required")
    role = payload.role if payload.role in containers.ROLES else containers.ROLE_VIEWER
    containers.set_member(c, target, role)
    containers.save_container(c)
    add_event("member_add", container=container_id, actor=user, member=target, role=role)
    return c


@app.delete("/api/containers/{container_id}/members/{username}")
async def container_remove_member(container_id: str, username: str, user: str = Depends(current_user)):
    c = containers.load_container(container_id)
    if not c:
        raise HTTPException(404, "project not found")
    if not containers.can_manage(c, user):
        raise HTTPException(403, "requires owner")
    containers.remove_member(c, username)
    containers.save_container(c)
    add_event("member_remove", container=container_id, actor=user, member=(username or "").strip().lower())
    return c


@app.post("/api/containers/{container_id}/datasets/{dataset_id}")
async def container_add_dataset(container_id: str, dataset_id: str, user: str = Depends(current_user)):
    c = containers.load_container(container_id)
    if not c:
        raise HTTPException(404, "project not found")
    # Any editor (or owner) of the Project can add a dataset — but only one they
    # own (you can't pull someone else's dataset into a Project).
    if not containers.can_write(c, user):
        raise HTTPException(403, "requires editor")
    m = load_manifest(dataset_id)
    if not m:
        raise HTTPException(404, "dataset not found")
    if (m.get("owner") or "").strip().lower() != user.lower():
        raise HTTPException(403, "not your dataset")
    m["container_id"] = container_id
    m["private"] = bool(c.get("private"))  # inherit Project privacy
    m["max_input_size"] = containers.clamp_max_input(c.get("max_input_size"))  # inherit Project ceiling
    save_manifest(dataset_id, m)
    if dataset_id not in (c.get("dataset_ids") or []):
        c.setdefault("dataset_ids", []).append(dataset_id)
        containers.save_container(c)
    # Keep the workspace tree mirroring the logical nesting: the dataset
    # folder physically moves into the project's folder (cosmetic only —
    # identity lives in the JSONs, so a failed move is harmless).
    try:
        store.move_dataset(dataset_id, container_id)
    except Exception as e:
        print(f"[containers] folder move failed for {dataset_id}: {e}")
    add_event("dataset_add", container=container_id, dataset=dataset_id, actor=user)
    return {"ok": True}


@app.delete("/api/containers/{container_id}/datasets/{dataset_id}")
async def container_remove_dataset(container_id: str, dataset_id: str, user: str = Depends(current_user)):
    c = containers.load_container(container_id)
    if not c:
        raise HTTPException(404, "project not found")
    m = load_manifest(dataset_id)
    ds_owner = (m.get("owner") or "").strip().lower() if m else ""
    # Detach is non-destructive (the dataset survives as standalone), so the
    # Project owner may remove any dataset to organise the Project, and an editor
    # may remove their OWN dataset — but an editor can't pull out someone else's.
    if not (containers.can_manage(c, user) or (ds_owner and ds_owner == (user or "").strip().lower())):
        raise HTTPException(403, "requires the Project owner or the dataset's creator")
    if m and (m.get("container_id") or "") == container_id:
        m["container_id"] = ""
        save_manifest(dataset_id, m)
    c["dataset_ids"] = [d for d in (c.get("dataset_ids") or []) if d != dataset_id]
    containers.save_container(c)
    try:
        store.move_dataset(dataset_id, None)
    except Exception as e:
        print(f"[containers] folder move failed for {dataset_id}: {e}")
    add_event("dataset_remove", container=container_id, dataset=dataset_id, actor=user)
    return {"ok": True}


def _container_cover_key(container_id: str) -> str:
    return f"containers/{container_id}/cover.jpg"


@app.post("/api/containers/{container_id}/cover")
async def container_upload_cover(
    container_id: str,
    file: UploadFile = File(...),
    user: str = Depends(current_user),
):
    """Upload/replace the Project cover (owner-only). Normalised to JPEG and
    stored in R2; served by GET .../cover under the container read gate."""
    c = containers.load_container(container_id)
    if not c:
        raise HTTPException(404, "project not found")
    if not containers.can_manage(c, user):
        raise HTTPException(403, "requires owner")
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES_PER_FILE:
        raise HTTPException(413, "cover too large")
    try:
        img = PILImage.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img).convert("RGB")
    except PILImage.DecompressionBombError:
        raise HTTPException(413, "image dimensions exceed the maximum allowed")
    except Exception:
        raise HTTPException(400, "not an image")
    img.thumbnail((1600, 1600), PILImage.LANCZOS)
    # Floor small uploads up to a minimum size so the full-width Project hero
    # never browser-upscales a tiny cover into a blocky mess (cheap Lanczos +
    # light sharpen; a no-op once the upload is already >= the floor).
    img = _upscale_cover_if_small(img, 1280)
    # Average luminance (0-255) of the bottom band, where the Project page
    # renders the title over the cover. The FE uses it to pick a black or white
    # title (and a matching scrim) that stands out on this particular cover.
    try:
        from PIL import ImageStat as _ImageStat
        _w, _h = img.size
        _band = img.crop((0, int(_h * 0.5), _w, _h)).convert("L")
        c["coverLuma"] = round(_ImageStat.Stat(_band).mean[0])
    except Exception:
        c["coverLuma"] = 0
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    r2_required().put_bytes(_container_cover_key(container_id), buf.getvalue(), "image/jpeg")
    c["cover"] = "cover.jpg"
    # Dedicated cover version: the R2 key is fixed (cover.jpg), so the FE can't
    # tell a re-upload from a cache hit. Bump this only on a cover change so the
    # Project page busts its cover cache HERE without reloading (flickering) the
    # hero on every unrelated settings save.
    c["cover_updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    containers.save_container(c)
    return {"cover": "cover.jpg", "coverLuma": c.get("coverLuma", 0), "cover_updated": c["cover_updated"]}


@app.get("/api/containers/{container_id}/cover")
async def container_serve_cover(container_id: str, request: Request, authorization: str = Header(default="")):
    """Serve the Project cover via an R2 presigned redirect, under the container
    read gate (public container -> anyone; private -> members; <img> auths via
    the pk_auth cookie)."""
    c = containers.load_container(container_id)
    if not c:
        raise HTTPException(404, "project not found")
    if not containers.can_read(c, request_username(request, authorization)):
        raise HTTPException(404, "project not found")
    if not c.get("cover"):
        raise HTTPException(404, "no cover")
    return _redirect_to_r2(_container_cover_key(container_id))


@app.get("/api/containers/{container_id}/activity")
async def container_activity(container_id: str, request: Request, authorization: str = Header(default=""), limit: int = 100):
    """Recent activity across all datasets in the Project (uploads, labelling,
    training, member changes, ...). Read gate = member-or-public."""
    c = containers.load_container(container_id)
    if not c:
        raise HTTPException(404, "project not found")
    if not containers.can_read(c, request_username(request, authorization)):
        raise HTTPException(404, "project not found")
    import audit
    limit = max(1, min(int(limit or 100), 300))
    return {"activity": audit.list_activity(container_id, c.get("dataset_ids") or [], limit=limit)}


def _compute_dataset_health(manifest: dict) -> dict:
    """Backend-canonical dataset health so the FE doesn't have to
    recompute it from labels + refs after both have loaded — that
    looked janky because the badge would change from "Bad" to
    "Okay" to "Good" as data trickled in. Same logic the FE used
    to run, lifted here so the value lands on first paint.
    """
    tags = collect_tags(manifest)
    refs = manifest.get("references") or []
    if not tags:
        return {
            "level": "bad",
            "label": "Bad",
            "reason": "No labels defined — add labels so the model knows what to detect.",
        }
    if not refs:
        return {
            "level": "okay",
            "label": "Okay",
            "reason": "Labels defined but no reference images uploaded yet. Add references to improve detection accuracy.",
        }
    n = len(refs)
    return {
        "level": "good",
        "label": "Good",
        "reason": f"{n} reference image{'' if n == 1 else 's'} uploaded. Annotate at least 5 examples per label in the Label tab to complete the dataset.",
    }


# ─── Lightweight overview / annotations split ────────────────────────────────
# The single-manifest GET used to bundle metadata + every detection's mask
# polygon + every detection's 1024-dim DINOv2 + SigLIP embedding. On a
# moderate project that's a multi-MB JSON payload — devastating over a
# tunnel or slow link, regardless of how fast the backend renders it.
#
# These two endpoints split the response by what the FE actually needs at
# each phase:
#
#   /overview     — render-critical only: project metadata, per-tile
#                   { id, filename, blurhash, w, h, n_detections,
#                     label_set }. Loads in tens of milliseconds. Drives
#                   the gallery placeholder grid + chip rail.
#   /annotations  — detections + editedBoxes, embeddings stripped (the
#                   resolver never sends them to the FE — they're for
#                   server-side scoring only). Pulled lazily after the
#                   overview has painted.
#
# Embeddings stay in the on-disk manifest for the resolver's use. The
# legacy GET /api/projects/{id} also stays, returning the full manifest,
# so existing callers (V1 ProjectView, scripts, debug tools) keep working.

def _kick_blurhash_backfill(project_id: str, m: dict) -> None:
    """Surface in-memory cached blurhashes onto the manifest dict + queue
    any missing encodes for an off-thread backfill. Shared by /overview,
    /annotations and the legacy GET so they all benefit from the same
    cache + persistence path."""
    needs: list[tuple[str, str]] = []
    for ref in m.get("references") or []:
        if isinstance(ref, dict) and not ref.get("blurhash"):
            fn = ref.get("filename")
            if not fn:
                continue
            cached = _BLURHASH_CACHE.get((project_id, f"references/{fn}"))
            if cached:
                ref["blurhash"] = cached
            else:
                needs.append(("references", fn))
    for imp in m.get("imports") or []:
        if isinstance(imp, dict) and not imp.get("blurhash"):
            fn = imp.get("filename")
            if not fn:
                continue
            cached = _BLURHASH_CACHE.get((project_id, f"imports/{fn}"))
            if cached:
                imp["blurhash"] = cached
            else:
                needs.append(("imports", fn))
    if needs:
        # _build_overview_payload runs in asyncio.to_thread (no running loop),
        # so get_running_loop() raises there. Guard it: an entry missing a
        # blurhash (e.g. a derived-project crop) must not 500 the overview —
        # just skip the async backfill when there's no loop to schedule on.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            loop.run_in_executor(None, _blurhash_backfill_async, project_id, needs)


def _sort_imports_desc(entries: list[dict]) -> list[dict]:
    """Return entries sorted by createdAt DESC, with missing-createdAt
    entries sinking to the bottom. Matches the FE's
    compareImportedMediaDesc so the first-batch slice the FE renders
    is already in final paint order — fixes the "gallery rearranges
    while loading then settles" flicker on big projects."""
    def key(e: dict) -> tuple[int, float, str]:
        # (has_no_ts, -ms_for_desc, id_for_stable_tiebreak).
        # Python's sorted(..., reverse=False) on this tuple yields:
        # has_no_ts ascending (0 first), then -ms ascending (= ms
        # descending), then id ascending.
        raw = e.get("createdAt")
        ms: float | None = None
        if isinstance(raw, (int, float)):
            ms = float(raw)
        elif isinstance(raw, str) and raw:
            try:
                ms = float(raw)
            except ValueError:
                try:
                    from datetime import datetime
                    ms = datetime.fromisoformat(
                        raw.replace("Z", "+00:00")
                    ).timestamp() * 1000.0
                except Exception:
                    ms = None
        if ms is None:
            return (1, 0.0, e.get("id") or "")
        return (0, -ms, e.get("id") or "")
    return sorted([e for e in entries if isinstance(e, dict)], key=key)


def _tile_overview(entries: list[dict]) -> list[dict]:
    """Reduce a list of reference / import dicts down to render-critical
    metadata only. Detections and editedBoxes drop out entirely; we
    surface aggregate counts + the unique label set so the gallery thumb
    can show "5 boxes · hare, rabbit" without pulling the actual
    geometry."""
    out: list[dict] = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        dets = e.get("detections") or []
        edited = e.get("editedBoxes") if isinstance(e.get("editedBoxes"), list) else None
        edited_set = bool(e.get("editedBoxesSet"))
        # editedBoxes wins for the chip count when the user has touched
        # them — including the explicit empty case (delete-all). Without
        # the editedBoxesSet branch a user who deletes every box on a
        # previously auto-labelled tile would still see the auto chips,
        # which looks like the delete didn't take.
        if edited is not None and edited_set:
            counted = edited
            label_pool = [str(b.get("label") or "") for b in edited if isinstance(b, dict)]
        elif edited:
            counted = edited
            label_pool = [str(b.get("label") or "") for b in edited if isinstance(b, dict)]
        else:
            counted = [d for d in dets if isinstance(d, dict) and not d.get("rejected")]
            # Label priority: vlm_label (post-VLM refinement) > pred_label
            # (resolver output) > label (legacy field) > gd_label (raw
            # Grounding DINO label). Without the gd_label fallback,
            # detections that haven't been through the resolver yet
            # surface label_set=[] here, which strips the chip rail off
            # every gallery tile even though the boxes do have labels.
            label_pool = [
                str(
                    d.get("vlm_label")
                    or d.get("pred_label")
                    or d.get("label")
                    or d.get("gd_label")
                    or ""
                )
                for d in counted
            ]
        labels: list[str] = []
        seen: set[str] = set()
        for lab in label_pool:
            k = lab.strip().lower()
            if not k or k in seen:
                continue
            seen.add(k)
            labels.append(lab)
        # source carries the origin metadata for URL-imported entries
        # (kind="openverse", url=...). The FE uses it to filter the
        # Openverse search results — anything already in the project
        # gets dropped from the preview so users don't accidentally
        # re-add. None for drag-dropped entries.
        src = e.get("source") if isinstance(e.get("source"), dict) else None
        # Compact per-label counts. Replaces the FE's per-tile
        # synthetic-detection allocations (one fake ImportDetection
        # per detection just to drive the chip rail). At ~25 dets
        # per image x 9000 images that was a 225k-object pile of
        # garbage on the heap; the chip rail now reads counts
        # directly from this dict. label_set kept on the wire for
        # back-compat with older FE builds.
        label_stats_pool: dict[str, int] = {}
        for lab in label_pool:
            k = lab.strip()
            if not k:
                continue
            label_stats_pool[k] = label_stats_pool.get(k, 0) + 1
        out.append({
            "id": e.get("id"),
            "filename": e.get("filename"),
            "originalFilename": e.get("originalFilename"),
            "width": e.get("width"),
            "height": e.get("height"),
            "blurhash": e.get("blurhash"),
            "createdAt": e.get("createdAt"),
            "n_detections": len(counted),
            "label_set": labels,
            # Per-label count map. FE reads this directly to render
            # the chip rail with no synthetic detection objects.
            "label_stats": label_stats_pool,
            "has_edits": bool(edited),
            "source": src,
            # Persisted labelling timestamp (epoch ms). Drives the FE's
            # labelled_preview cachebuster (?v=). It MUST be persisted +
            # surfaced here: the FE-only Date.now() it used to set was
            # lost on cold re-open, so the preview URL fell back to the
            # bare path and the browser served the stale BLANK preview
            # cached before the image was labelled — the "segmented
            # cover vanishes when you reopen the project" bug.
            "labelledAt": e.get("labelledAt"),
            # Number of augmentation copies persisted for this
            # entry. Set by the augment_generate job after each
            # image; the FE gates the per-tile Augmentations icon
            # on it being > 0.
            "n_augmentations": int(e.get("n_augmentations") or 0),
            # For a derived ("new labels") crop: the parent's original label,
            # surfaced so the gallery + viewer can show it as a reference while
            # the user assigns their own fresh labels. None on normal imports.
            "derivedLabel": ((e.get("derivedFrom") or {}).get("label") or None)
            if isinstance(e.get("derivedFrom"), dict) else None,
        })
    return out


_EMBEDDING_FIELDS = (
    "embedding",
    "siglip_embedding",
    # Some older code paths persisted the SigLIP embedding under the
    # reversed name. Cover both so the strip is complete regardless
    # of which writer ran.
    "embedding_siglip",
    "embed_version",
    "siglip_version",
)

# Fields that are heavy (mask polygons are the worst — hundreds of
# (x, y) pairs per detection; base64-encoded crops are ~15 KB each)
# AND not needed for gallery-tile rendering. /annotations strips
# these by default; the per-image /annotations/{import_id} endpoint
# returns them for the viewer + pipeline popup.
_HEAVY_DETECTION_FIELDS = (
    "mask",
    "polygons",
    # Base64-encoded JPEG crop of the detection's bounding region.
    # Used by the pipeline popup to show "what the model saw" tiles.
    # ~15 KB per detection — by far the largest contributor to wire
    # size in /annotations on big projects (a 9000-detection project
    # was shipping 130 MB of base64 crops alone).
    "crop_jpg_b64",
    # The per-label sim dicts can each carry one float per project
    # label — for a 10-label project that's still 10× more bytes per
    # detection than the predLabel itself. Pipeline popup needs them
    # but the gallery doesn't.
    "embed_sims",
    "embed_sims_dino",
    "embed_sims_siglip",
)


def _strip_embedding(d: dict) -> dict:
    """Return a shallow copy of `d` minus the four embedding fields.
    Roughly 2× faster than the dict-comprehension form because dict()
    is a single C-level copy, then we pop the four known keys instead
    of iterating every field looking for matches."""
    slim = dict(d)
    for k in _EMBEDDING_FIELDS:
        slim.pop(k, None)
    return slim


def _strip_embedding_and_heavy(d: dict) -> dict:
    """Same as _strip_embedding but also drops mask polygons + per-
    label similarity dicts. Used by the bulk /annotations endpoint so
    its response stays small enough to ship over the wire quickly.
    Viewer-side /annotations/{import_id} reads the full record."""
    slim = dict(d)
    for k in _EMBEDDING_FIELDS:
        slim.pop(k, None)
    for k in _HEAVY_DETECTION_FIELDS:
        slim.pop(k, None)
    return slim


def _strip_edited_box(b: dict) -> dict:
    """editedBoxes carry their own mask polygons (the user-drawn or
    backend-segmented outline for each box). The gallery doesn't need
    them — only the viewer does, and the viewer fetches the per-image
    /annotations/{import_id} which doesn't go through this strip."""
    slim = dict(b)
    slim.pop("mask", None)
    slim.pop("polygons", None)
    return slim


def _detection_annotations(entries: list[dict], *, include_edits: bool, include_heavy: bool = False) -> dict[str, dict]:
    """Strip embeddings out of every detection so the annotations payload
    is ~10× smaller than the raw manifest. Keeps mask polygons by default
    (the FE needs them to render box outlines + the labelled-preview
    overlay) — set include_heavy=False to also strip masks +
    per-label similarity dicts on detections AND editedBoxes for the
    bulk gallery-hydration endpoint."""
    strip = _strip_embedding if include_heavy else _strip_embedding_and_heavy
    out: dict[str, dict] = {}
    for e in entries:
        if not isinstance(e, dict):
            continue
        eid = e.get("id")
        if not eid:
            continue
        slim_dets = [
            strip(d)
            for d in (e.get("detections") or [])
            if isinstance(d, dict)
        ]
        rec: dict = {"detections": slim_dets}
        if include_edits:
            edited = e.get("editedBoxes")
            # Only surface editedBoxes once the user has actually
            # touched them. Older imports were seeded with an empty
            # editedBoxes:[] at upload time which the FE then mistook
            # for "user has zero boxes" and used to mask out the auto
            # detections. The `editedBoxesSet` flag (written by the
            # PUT /imports/{id} handler) is the authoritative signal.
            if isinstance(edited, list) and e.get("editedBoxesSet"):
                rec["editedBoxes"] = (
                    edited if include_heavy
                    else [_strip_edited_box(b) for b in edited if isinstance(b, dict)]
                )
            timings = e.get("timings")
            if isinstance(timings, dict):
                rec["timings"] = timings
        out[eid] = rec
    return out


def _build_overview_payload(
    project_id: str,
    m: dict,
    imports_limit: int | None = None,
    imports_offset: int = 0,
) -> dict:
    """Sync overview-builder. Runs inside asyncio.to_thread so the
    JSON load + deepcopy + dataset-health computation doesn't pin the
    FastAPI event loop when a labelling job is also in flight.

    `imports_limit` slices the imports list before _tile_overview
    iterates — the per-tile reduction (n_detections + label_set) is
    the dominant cost on a 963-image project (~600 ms). With a limit
    of 20 the slice cost drops to ~12 ms, so the FE can paint the
    first viewport almost instantly and fetch the rest in the
    background. `imports_total` is always set so the FE knows how
    many follow-up batches it needs."""
    _kick_blurhash_backfill(project_id, m)
    # Sort by createdAt DESC so the first-batch slice matches the FE's
    # gallery order. Without this the FE shows the first 100 manifest-
    # order entries, then re-sorts when the remainder lands —
    # producing the visible "rearrange then settle" flicker the user
    # reported. With the sort moved here every slice is already in
    # final paint order.
    all_imports = _sort_imports_desc(m.get("imports") or [])
    if imports_limit is not None:
        sliced_imports = all_imports[imports_offset : imports_offset + imports_limit]
    else:
        sliced_imports = all_imports
    # Project-level snapshot of the labels the last label_charlie job
    # ran with. Used by the FE's freshLabels heuristic to distinguish
    # "this tag was searched but never matched" from "this tag was
    # added after the last run." Backfill with current tags when the
    # field is missing AND at least one import is labelled, so legacy
    # projects don't trip the force_relabel false positive on the very
    # first click after this code ships.
    labels_last_run = m.get("labelsLastRun")
    if labels_last_run is None:
        has_labelled = any(
            isinstance(e, dict) and e.get("labelled") is True
            for e in (m.get("imports") or [])
        )
        if has_labelled:
            labels_last_run = list(m.get("tags") or [])
    return {
        "id": m.get("id") or project_id,
        "name": m.get("name"),
        "prompt": m.get("prompt"),
        "tags": m.get("tags") or [],
        "labelsLastRun": labels_last_run,
        "settingsLastRun": m.get("settingsLastRun"),
        "label_aliases": m.get("label_aliases") or {},
        "labelColours": m.get("labelColours") or {},
        # Per-image verdicts from the fast-review modal (good/bad/
        # unsure). Surfaced here so the V2 project page can hydrate
        # the verdict filter + review state on first paint without a
        # separate full-manifest round-trip.
        "verdicts": m.get("verdicts") or {},
        "cover": m.get("cover"),
        "cover_blurhash": m.get("cover_blurhash"),
        "v2": bool(m.get("v2")),
        "createdAt": m.get("createdAt"),
        "updatedAt": m.get("updatedAt"),
        "thresholds": m.get("thresholds"),
        "vlm_action": m.get("vlm_action"),
        "synonyms_enabled": m.get("synonyms_enabled"),
        "private": m.get("private"),
        # Max upload long-edge (px). The FE caps client-side resize to this, so
        # it MUST reach the FE or 4K-Project datasets silently fall back to the
        # 1500 default and uploads get downscaled. Inherited from the Project.
        "max_input_size": containers.clamp_max_input(m.get("max_input_size")),
        # Cached general/specific verdict (LLM-free resolve) so the hero badge
        # paints in the same frame as the rest of /overview instead of popping
        # in 100-1000 ms later after a separate /dataset-type round-trip. Null
        # when only a fresh Claude classification could answer — the FE then
        # falls back to its own /dataset-type fetch.
        "dataset_type": _resolve_dataset_type_cached_only(
            project_id, list(m.get("tags") or [])
        ),
        # Derived ("child") link: present only on cropped child projects so the
        # FE can show a derived badge + a link back to the parent.
        "derived": m.get("derived") or None,
        "hasModel": m.get("hasModel"),
        # Owner / createdBy let the FE drop the slow legacy
        # /api/projects/{id} round-trip it used to make on mount just
        # to populate the project-meta cache with these fields.
        "owner": m.get("owner"),
        "createdBy": m.get("createdBy"),
        "dataset_health": _compute_dataset_health(m),
        "references": _tile_overview(m.get("references") or []),
        "imports": _tile_overview(sliced_imports),
        "imports_total": len(all_imports),
        "imports_offset": imports_offset,
        # Pre-computed gallery-filter chip counts. Scans the FULL
        # imports list (not just the sliced page) so the FE doesn't
        # have to wait for all batches to land before the ALL / GOOD /
        # BAD / UNLABELLED chips show their final numbers. Mirrors
        # the same logic the FE's per-item iteration uses (see
        # DatasetGallery's filterCounts useMemo) so the values match
        # the moment the imports list catches up.
        "filter_counts": _compute_filter_counts(
            all_imports, m.get("verdicts") or {},
        ),
    }


def _compute_filter_counts(
    all_imports: list[dict], verdicts: dict,
) -> dict[str, int]:
    """Authoritative chip counts across the full imports list. The
    FE will display these directly so the chips read their final
    values on first paint instead of climbing 20 → 100 → N as the
    imports list streams in batches."""
    counts = {"all": 0, "unlabelled": 0, "unrated": 0, "good": 0, "bad": 0, "unsure": 0}
    for e in all_imports:
        if not isinstance(e, dict):
            continue
        counts["all"] += 1
        has_dets = bool(e.get("detections")) or bool(e.get("editedBoxes"))
        if not has_dets:
            counts["unlabelled"] += 1
        eid = e.get("id")
        v = verdicts.get(eid) if eid else None
        if eid and not v and has_dets:
            counts["unrated"] += 1
        if v == "good":
            counts["good"] += 1
        elif v == "bad":
            counts["bad"] += 1
        elif v == "unsure":
            counts["unsure"] += 1
    return counts


@app.get(
    "/api/v2/projects/{project_id}/initial",
    response_class=ORJSONResponse,
    dependencies=[Depends(require_project_read_access)],
)
async def get_project_initial(
    project_id: str,
    response: Response,
    n: int = _INITIAL_PAYLOAD_LIMIT,
):
    """Single-round-trip first-paint payload — project meta, first-N
    import tiles (covers + chips + box counts) and the lite stats card
    data, all in one document.

    Served straight from the on-disk initial_first20.json sidecar that
    save_manifest's _kick_sidecar_refresh writes after every change.
    Zero compute, zero manifest parse on the request thread, and only
    one HTTP round-trip from the FE — so the gallery + chip rail + stats
    badge all render in one paint instead of three staggered fetches.

    If the sidecar is missing (first-touch / cold deploy) we synthesise
    it from /overview + /dataset-stats(lite) on the spot and kick a
    rebuild for next time. `n` slices the imports list before send so a
    caller asking for 5 doesn't get back 20.
    """
    if not project_dir(project_id).exists():
        raise HTTPException(404)
    # Bypass the browser's HTTP cache. After a destructive action
    # (dedupe commit, ignore, manual delete) the FE hits
    # window.location.reload() — which respects the disk cache by
    # default, and without this header the browser handed back the
    # pre-delete /initial response on the very first paint, so the
    # stats card + gallery rendered with stale numbers until the
    # /dataset-stats?lite (cache:no-store) follow-up landed and
    # overwrote them. apiFetch's in-flight dedup is independent of
    # the HTTP cache so co-fired duplicate /initial calls still
    # coalesce into one round-trip.
    response.headers["Cache-Control"] = "no-store, must-revalidate"
    n = max(1, min(int(n), _INITIAL_PAYLOAD_LIMIT))
    side = await asyncio.to_thread(_read_initial_sidecar, project_id)
    if side is not None:
        # Stale-detection: compare sidecar mtime to manifest mtime. If
        # the sidecar predates the manifest (or any backend code-path
        # that rebuilds it), kick an async refresh AND fall through to
        # the compute path so the FIRST request after a server upgrade
        # gets fresh data instead of one more stale-snapshot served.
        # Race-safe: the sidecar can be deleted between the read above and
        # this stat (concurrent invalidate / refresh), which would raise
        # FileNotFoundError → unhandled 500. Treat a vanished sidecar as
        # "fall through to inline compute".
        try:
            sidecar_mtime = _initial_sidecar_path(project_id).stat().st_mtime
        except OSError:
            sidecar_mtime = None
        manifest_mtime = _manifest_disk_mtime(project_id)
        if sidecar_mtime is not None and sidecar_mtime >= manifest_mtime:
            # Fresh enough — serve it AND kick a background refresh so
            # any out-of-band changes (label fallback rules, format
            # tweaks) propagate by the next visit.
            _kick_sidecar_refresh(project_id)
            imps = (side.get("imports") or [])[:n]
            return {**side, "imports": imps}
        # Stale sidecar — fall through to the inline compute below so
        # the user doesn't see a one-version-old snapshot.
    # Cold path: no sidecar on disk yet. Build it inline so the first
    # visit still gets data, AND kick the async refresh so subsequent
    # visits hit the fast path. This is the same cost as /overview +
    # /dataset-stats?lite combined, so we're not regressing — just not
    # winning until the sidecar lands.
    _kick_sidecar_refresh(project_id)
    m = await asyncio.to_thread(load_manifest, project_id, False)
    if not m:
        raise HTTPException(404)
    overview_payload = await asyncio.to_thread(
        _build_overview_payload, project_id, m,
        _FAST_OVERVIEW_LIMIT, 0,
    )
    # Best-effort lite stats — same compute the /dataset-stats?lite=true
    # endpoint runs. Returns counts + label distribution + 3-factor
    # health, fast on small projects, sub-second on big ones.
    stats_payload: dict | None
    try:
        stats_payload = await asyncio.to_thread(
            _compute_dataset_stats_v2, project_id, True,
        )
    except Exception as e:
        print(f"[initial cold] stats compute failed for {project_id}: {e}")
        stats_payload = None
    return {
        "id": overview_payload.get("id"),
        "name": overview_payload.get("name"),
        "prompt": overview_payload.get("prompt"),
        "tags": overview_payload.get("tags") or [],
        "labelsLastRun": overview_payload.get("labelsLastRun"),
        "settingsLastRun": overview_payload.get("settingsLastRun"),
        "label_aliases": overview_payload.get("label_aliases") or {},
        "labelColours": overview_payload.get("labelColours") or {},
        "cover": overview_payload.get("cover"),
        "cover_blurhash": overview_payload.get("cover_blurhash"),
        "v2": overview_payload.get("v2"),
        "createdAt": overview_payload.get("createdAt"),
        "updatedAt": overview_payload.get("updatedAt"),
        "thresholds": overview_payload.get("thresholds"),
        "vlm_action": overview_payload.get("vlm_action"),
        "synonyms_enabled": overview_payload.get("synonyms_enabled"),
        "private": overview_payload.get("private"),
        "max_input_size": overview_payload.get("max_input_size"),
        "dataset_type": overview_payload.get("dataset_type"),
        "hasModel": overview_payload.get("hasModel"),
        "owner": overview_payload.get("owner"),
        "createdBy": overview_payload.get("createdBy"),
        "dataset_health": overview_payload.get("dataset_health"),
        "references": overview_payload.get("references") or [],
        "imports": (overview_payload.get("imports") or [])[:n],
        "imports_total": overview_payload.get("imports_total") or 0,
        "stats": stats_payload,
    }


@app.get(
    "/api/v2/projects/{project_id}/overview",
    response_class=ORJSONResponse,
    dependencies=[Depends(require_project_read_access)],
)
async def get_project_overview(
    project_id: str,
    imports_limit: int | None = None,
    imports_offset: int = 0,
):
    """Render-critical project metadata. No detections, no embeddings,
    no mask polygons. The heavy bits (manifest deepcopy + tile reduction
    + dataset-health roll-up) run on the default executor so concurrent
    requests aren't queued behind each other on the event loop when a
    labelling job is hogging GPU + CPU.

    Reads through the shared cache without deepcopying — _kick_blurhash_backfill
    writes a missing blurhash back on the cached dict (idempotent), the
    rest is pure read. Built payloads are cached by manifest mtime so
    repeat polls inside an unchanged window are sub-millisecond.

    `?imports_limit=20&imports_offset=0` returns just the first 20
    import tiles instead of the whole gallery. The FE uses this to
    paint the first viewport in ~50 ms, then fetches the rest in the
    background. `imports_total` in the response is the unsliced count.
    """
    if not project_dir(project_id).exists():
        raise HTTPException(404)
    # Fast path: the FE's first-batch request matches the persisted
    # sidecar exactly, so we serve it straight from disk — zero
    # compute, zero manifest parse on the request thread. Allow-stale
    # because the sidecar gets rebuilt by save_manifest's async hook
    # after each write; serving the previous batch's numbers for a
    # second is preferable to making the user wait through a compute.
    if (
        imports_offset == 0
        and imports_limit is not None
        and imports_limit <= _FAST_OVERVIEW_LIMIT
    ):
        side = await asyncio.to_thread(_read_overview_sidecar, project_id)
        if side is not None:
            # Re-stat for the freshness check. The sidecar can be deleted
            # between the read above and this stat (a concurrent
            # _invalidate_project_payloads / sidecar refresh / label job),
            # so a raw .stat() races and raises FileNotFoundError → an
            # unhandled 500 that even drops the CORS header. Treat a
            # vanished/unreadable sidecar as "can't use the fast path" and
            # fall through to the inline compute below.
            try:
                sidecar_mtime = _overview_sidecar_path(project_id).stat().st_mtime
            except OSError:
                sidecar_mtime = None
            manifest_mtime = _manifest_disk_mtime(project_id)
            if sidecar_mtime is not None and sidecar_mtime >= manifest_mtime:
                # Fresh — serve it AND kick an async refresh so
                # out-of-band changes (label fallback rules, schema
                # tweaks) land by the next visit.
                _kick_sidecar_refresh(project_id)
                # Sidecar holds the full first-100; respect the caller's
                # limit so a request for 20 doesn't get back 100 entries.
                imps = (side.get("imports") or [])[:imports_limit]
                return {**side, "imports": imps, "imports_offset": 0}
            # Stale — fall through to inline compute so this request
            # gets fresh data, and kick a refresh for future visits.
            _kick_sidecar_refresh(project_id)
        else:
            # Sidecar missing — kick a build and fall through to the
            # synchronous compute below so the FIRST visit still works.
            _kick_sidecar_refresh(project_id)
    disk_mtime = _manifest_disk_mtime(project_id)
    # Pre-warm the dataset-stats sidecar in the background on every
    # /overview hit. By the time the FE's DatasetStatsCard fetches
    # /dataset-stats?lite=true (a few hundred ms after /overview lands)
    # the sidecar is already on disk and the response is a quick file
    # read. Only kicks one rebuild per project at a time thanks to the
    # _PAYLOAD_REVALIDATE_IN_FLIGHT dedupe inside _payload_revalidate.
    asyncio.create_task(_payload_revalidate(
        project_id, "dataset-stats-lite",
        lambda: _persist_dataset_stats(project_id, True),
    ))
    # Cache key folds in the slice so different pages don't share an
    # entry. A fresh cache hit for the same slice returns instantly.
    cache_name = f"overview:{imports_offset}:{imports_limit}"
    cached, fresh = _payload_cache_get_swr(project_id, cache_name, disk_mtime)
    if cached is not None and fresh:
        return cached
    if cached is not None:
        # Stale-while-revalidate: serve the previous payload now,
        # rebuild in the background so the next request is fresh.
        def _build():
            m2 = load_manifest(project_id, False)
            return _build_overview_payload(project_id, m2, imports_limit, imports_offset) if m2 else {}
        asyncio.create_task(_payload_revalidate(project_id, cache_name, _build))
        return cached
    m = await asyncio.to_thread(load_manifest, project_id, False)
    if not m:
        return m
    payload = await asyncio.to_thread(
        _build_overview_payload, project_id, m, imports_limit, imports_offset,
    )
    _payload_cache_put(project_id, cache_name, disk_mtime, payload)
    return payload


def _build_annotations_payload(m: dict) -> dict:
    """Bulk gallery-hydration payload — heavy mask polygons + per-label
    sim dicts are stripped here so the wire size stays manageable on a
    963-image / 9000-detection project. The viewer fetches the full
    record per-image via /annotations/{import_id} when the user opens
    a tile."""
    return {
        "references": _detection_annotations(
            m.get("references") or [], include_edits=False, include_heavy=False,
        ),
        "imports": _detection_annotations(
            m.get("imports") or [], include_edits=True, include_heavy=False,
        ),
    }


@app.get(
    "/api/v3/projects/{project_id}/viewport",
    dependencies=[Depends(require_project_read_access)],
)
async def v3_viewport_batch(project_id: str, ids: str = "", request: Request = None):  # type: ignore[assignment]
    """Batched per-image annotations for a comma-separated list of
    import ids. Replaces N parallel /annotations/{import_id} calls
    from the viewer's neighbour-prefetch with a single round-trip —
    cycling between gallery tiles paints from the cached batch
    instead of hitting the network per arrow press.

    Returns the same shape /annotations/{import_id} does for each
    requested id, under `imports[id]`. Unknown ids are simply
    omitted from the response so the FE can safely request more
    than exists.

    The endpoint is intentionally simple: full geometry only (no
    adaptive levels). Adaptive geometry / RDP-simplified polygons
    are a follow-up; this single batched call already collapses
    the most expensive part of the viewer's request profile.
    """
    if not project_dir(project_id).exists():
        raise HTTPException(404)
    raw_ids = [s.strip() for s in (ids or "").split(",") if s.strip()]
    if not raw_ids:
        return {"imports": {}}
    # Cap to keep one bad request from scanning the manifest
    # repeatedly with a giant id list.
    if len(raw_ids) > 64:
        raw_ids = raw_ids[:64]
    requested = set(raw_ids)
    await asyncio.to_thread(load_manifest, project_id, False)
    with _MANIFEST_CACHE_LOCK:
        idx = _MANIFEST_IMPORT_INDEX.get(project_id, {})
        entries = {eid: idx[eid] for eid in requested if eid in idx}
    out: dict[str, dict] = {}
    for eid, e in entries.items():
        out[eid] = {
            "detections": [
                _strip_embedding(d)
                for d in (e.get("detections") or [])
                if isinstance(d, dict)
            ],
            "editedBoxes": (
                e.get("editedBoxes")
                if isinstance(e.get("editedBoxes"), list) and e.get("editedBoxesSet")
                else None
            ),
            "timings": e.get("timings") if isinstance(e.get("timings"), dict) else None,
        }
    payload = {"imports": out}
    # P6 binary wire: when the client opts in via Accept header,
    # return the same dict serialized with msgpack instead of JSON.
    # ~30-50% smaller on mask-heavy responses; parse is faster too
    # since we skip the string→number coercion. Falls back to ORJSON
    # when the header isn't present so all existing callers keep
    # working unchanged.
    accept = (request.headers.get("accept") or "") if request is not None else ""
    if "application/msgpack" in accept.lower():
        try:
            import msgpack
            return Response(
                content=msgpack.packb(payload, use_bin_type=True),
                media_type="application/msgpack",
            )
        except Exception as e:
            print(f"[v3-viewport] msgpack pack failed, falling back to JSON: {e}")
    return ORJSONResponse(payload)


@app.post(
    "/api/v3/projects/{project_id}/dedupe-imports",
    dependencies=[Depends(require_project_owner)],
)
async def v3_dedupe_imports(project_id: str, apply: bool = False):
    """One-shot pass that groups manifest imports by SHA-256 of
    their on-disk image bytes and within each group keeps the
    entry most likely to be the user's intended record. The rest
    are deleted (manifest row + image file + cached labelled
    preview).

    Default is DRY-RUN: returns the proposed deletions and a
    summary, no writes. Pass `?apply=1` to actually delete.

    Keeper-selection: highest detection count wins; on tie, the
    entry whose originalFilename does NOT look like a UUID-named
    upload-retry fallback (e.g. ``3f6db17ccb...jpg``) wins.

    Built specifically to clean up duplicates created by the
    upload-retry-without-idempotency-key race the FE fixed
    elsewhere — that fix prevents future dupes; this scrubber
    cleans up the ones already on disk.
    """
    import hashlib
    import re
    proj = project_dir(project_id)
    if not proj.exists():
        raise HTTPException(404, "project not found")
    imports_dir = proj / "images"
    if not imports_dir.exists():
        return {"applied": False, "would_delete": 0, "duplicates": []}
    manifest = await asyncio.to_thread(load_manifest, project_id, True)
    if not manifest:
        raise HTTPException(404, "manifest not found")
    entries = manifest.get("imports") or []

    def _hash_one(entry: dict) -> tuple[str, str | None]:
        eid = entry.get("id", "")
        fn = entry.get("filename")
        if not fn:
            return eid, None
        path = imports_dir / fn
        if not path.exists():
            return eid, None
        try:
            h = hashlib.sha256()
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(1024 * 64), b""):
                    h.update(chunk)
            return eid, h.hexdigest()
        except Exception as e:
            print(f"[dedupe] hash failed for {fn}: {e}")
            return eid, None

    hashes: dict[str, str] = {}
    for entry in entries:
        eid, hsh = await asyncio.to_thread(_hash_one, entry)
        if hsh is not None:
            hashes[eid] = hsh

    groups: dict[str, list[dict]] = {}
    for entry in entries:
        eid = entry.get("id", "")
        hsh = hashes.get(eid)
        if hsh is None:
            continue
        groups.setdefault(hsh, []).append(entry)

    uuid_like = re.compile(r"^[0-9a-f]{32}\.(?:jpg|jpeg|png|webp)$", re.IGNORECASE)

    def _score(entry: dict) -> tuple[int, int]:
        # Higher wins. Tuple is compared lexicographically.
        det_count = len(entry.get("detections") or [])
        orig = entry.get("originalFilename") or entry.get("filename") or ""
        is_uuid = 1 if uuid_like.match(orig) else 0
        # det_count desc, then non-UUID-named preferred (so -is_uuid).
        return (det_count, -is_uuid)

    losers: list[dict] = []
    losers_ids: set[str] = set()
    for hsh, group in groups.items():
        if len(group) < 2:
            continue
        sorted_group = sorted(group, key=_score, reverse=True)
        keeper = sorted_group[0]
        for entry in sorted_group[1:]:
            losers_ids.add(entry.get("id", ""))
            losers.append({
                "id": entry.get("id"),
                "filename": entry.get("filename"),
                "originalFilename": entry.get("originalFilename"),
                "detection_count": len(entry.get("detections") or []),
                "kept_id": keeper.get("id"),
                "kept_filename": keeper.get("filename"),
                "sha256": hsh,
            })

    if not apply or not losers:
        return {
            "applied": False,
            "would_delete": len(losers),
            "duplicates": losers,
        }

    write_lock = await _manifest_write_lock(project_id)
    deleted: list[str] = []
    async with write_lock:
        manifest = load_manifest(project_id)
        if not manifest:
            raise HTTPException(404, "manifest not found")
        kept: list[dict] = []
        for imp in (manifest.get("imports") or []):
            if imp.get("id") in losers_ids:
                fn = imp.get("filename")
                if fn:
                    try:
                        (imports_dir / fn).unlink(missing_ok=True)
                    except Exception as e:
                        print(f"[dedupe] unlink failed {fn}: {e}")
                _invalidate_labelled_preview(project_id, imp.get("id", ""))
                deleted.append(imp.get("id"))
            else:
                kept.append(imp)
        manifest["imports"] = kept
        # Cover-rescue if we deleted the file that was the cover.
        cover_filename = manifest.get("cover")
        if cover_filename and not any(
            imp.get("filename") == cover_filename for imp in kept
        ) and not any(
            ref.get("filename") == cover_filename
            for ref in (manifest.get("references") or [])
        ):
            import random as _rnd
            kept_filenames = [
                imp.get("filename") for imp in kept if imp.get("filename")
            ]
            if kept_filenames:
                manifest["cover"] = _rnd.choice(kept_filenames)
                manifest["cover_blurhash"] = None
        manifest["updatedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        save_manifest(project_id, manifest)
    return {
        "applied": True,
        "count": len(deleted),
        "deleted": deleted,
    }


@app.get(
    "/api/v2/projects/{project_id}/annotations/{import_id}",
    response_class=ORJSONResponse,
    dependencies=[Depends(require_project_read_access)],
)
async def get_project_annotation_for_import(project_id: str, import_id: str):
    """Full detection record for a single import (mask polygons + per-
    label sim dicts included). Fetched by the viewer when the user
    opens a tile — bulk /annotations skips these heavy fields to keep
    its wire size down. Bypasses the payload cache: per-image data is
    tiny enough that the cache layer doesn't pay off, and the viewer
    expects fresh data after any in-flight edit."""
    if not project_dir(project_id).exists():
        raise HTTPException(404)
    # Ensure cache + index are fresh, then do an O(1) id lookup
    # instead of scanning all imports linearly.
    await asyncio.to_thread(load_manifest, project_id, False)
    with _MANIFEST_CACHE_LOCK:
        e = _MANIFEST_IMPORT_INDEX.get(project_id, {}).get(import_id)
    if not e:
        raise HTTPException(404, "import not found")
    return {
        "detections": [
            _strip_embedding(d)
            for d in (e.get("detections") or [])
            if isinstance(d, dict)
        ],
        "editedBoxes": (
            e.get("editedBoxes")
            if isinstance(e.get("editedBoxes"), list) and e.get("editedBoxesSet")
            else None
        ),
        "timings": e.get("timings") if isinstance(e.get("timings"), dict) else None,
    }


def _build_annotations_payload_scoped(m: dict, scope: str) -> dict:
    """Same as _build_annotations_payload but lets the caller drop the
    refs or imports half. `scope=refs` is the gallery's preferred call
    — the imports' real geometry isn't needed for the chip rail (the
    placeholder synth from /overview is correct) and the viewer reads
    the per-image endpoint for full geometry on demand."""
    out: dict = {}
    if scope in ("all", "refs"):
        out["references"] = _detection_annotations(
            m.get("references") or [], include_edits=False, include_heavy=False,
        )
    else:
        out["references"] = {}
    if scope in ("all", "imports"):
        out["imports"] = _detection_annotations(
            m.get("imports") or [], include_edits=True, include_heavy=False,
        )
    else:
        out["imports"] = {}
    return out


@app.get(
    "/api/v2/projects/{project_id}/annotations",
    response_class=ORJSONResponse,
    dependencies=[Depends(require_project_read_access)],
)
async def get_project_annotations(project_id: str, scope: str = "all"):
    """Per-image detections + editedBoxes WITHOUT embeddings. Pulled by
    the FE after the overview has painted; powers the BoxEditor + the
    pipeline popup. Embeddings are server-side only — the resolver
    reads them off the on-disk manifest, the FE never sees them.

    `scope=refs` returns ONLY the references' detection geometry — the
    gallery uses placeholder detections synthesised from /overview's
    n_detections + label_set, so the bulk imports payload (~270 KB on
    a 9000-detection project) is dead weight there. The viewer fetches
    real per-image data through /annotations/{import_id} when the user
    opens a tile.

    Same off-loop pattern as /overview: the manifest load + per-entry
    embedding strip is done on the executor so other read endpoints
    don't queue behind it.
    """
    if not project_dir(project_id).exists():
        raise HTTPException(404)
    scope_norm = scope if scope in ("all", "refs", "imports") else "all"
    cache_key = "annotations" if scope_norm == "all" else f"annotations:{scope_norm}"
    disk_mtime = _manifest_disk_mtime(project_id)
    cached, fresh = _payload_cache_get_swr(project_id, cache_key, disk_mtime)
    if cached is not None and fresh:
        return cached
    if cached is not None:
        def _build_scoped():
            m2 = load_manifest(project_id, False)
            return _build_annotations_payload_scoped(m2, scope_norm) if m2 else {}
        asyncio.create_task(_payload_revalidate(project_id, cache_key, _build_scoped))
        return cached
    # Read-only — pass copy=False to skip the manifest deepcopy. The
    # builder only reads from m to produce a brand-new payload dict.
    m = await asyncio.to_thread(load_manifest, project_id, False)
    if not m:
        return m
    payload = await asyncio.to_thread(_build_annotations_payload_scoped, m, scope_norm)
    _payload_cache_put(project_id, cache_key, disk_mtime, payload)
    return payload


@app.get(
    "/api/projects/{project_id}",
    dependencies=[Depends(require_project_read_access)],
)
async def get_project(project_id: str):
    if not project_dir(project_id).exists():
        raise HTTPException(404)
    m = load_manifest(project_id)
    if not m:
        return m
    # Surface any blurhashes already in the in-memory cache without
    # re-encoding from disk (which is what made the GET handler slow
    # — 50 encodes × ~10 ms ≈ 500 ms blocking the response). Cache
    # hits are O(1); misses fall through and get scheduled for a
    # background backfill below.
    needs_backfill: list[tuple[str, str, dict]] = []
    for ref in m.get("references") or []:
        if isinstance(ref, dict) and not ref.get("blurhash"):
            fn = ref.get("filename")
            if not fn:
                continue
            cached = _BLURHASH_CACHE.get((project_id, f"references/{fn}"))
            if cached:
                ref["blurhash"] = cached
            else:
                needs_backfill.append(("references", fn, ref))
    for imp in m.get("imports") or []:
        if isinstance(imp, dict) and not imp.get("blurhash"):
            fn = imp.get("filename")
            if not fn:
                continue
            cached = _BLURHASH_CACHE.get((project_id, f"imports/{fn}"))
            if cached:
                imp["blurhash"] = cached
            else:
                needs_backfill.append(("imports", fn, imp))

    # Schedule any missing encodes to run on the executor pool so the
    # GET response returns IMMEDIATELY with whatever's already cached.
    # Next manifest read (the user re-opens the project, or a poll
    # tick) picks up the freshly encoded hashes — meanwhile the FE
    # falls back to its animated gradient placeholder, no blank tiles.
    if needs_backfill:
        loop = asyncio.get_running_loop()
        loop.run_in_executor(
            None,
            _blurhash_backfill_async,
            project_id,
            [(subdir, fn) for subdir, fn, _ref in needs_backfill],
        )
    # Dataset health lands directly in the response so the FE can
    # render the badge on first paint without waiting for labels +
    # refs to both hydrate.
    m["dataset_health"] = _compute_dataset_health(m)
    # Resolve the upload size ceiling (inherited from the dataset's Project, or
    # the default) so the FE always has a concrete value to cap uploads to.
    m["max_input_size"] = containers.clamp_max_input(m.get("max_input_size"))
    if m.get("v2"):
        n_refs = len(m.get("references") or [])
        n_imps = len(m.get("imports") or [])
        print(
            f"[v2-manifest-get] {project_id} → references={n_refs}, "
            f"imports={n_imps}, cover={m.get('cover')!r}"
        )
    return m


class UpdateProjectIn(BaseModel):
    tags: list[str] | None = None
    prompt: str | None = None
    thresholds: dict[str, float] | None = None
    vlm_action: str | None = None
    synonyms_enabled: bool | None = None
    verdicts: dict[str, str] | None = None
    editedBoxes: dict[str, list[dict]] | None = None
    cover: str | None = None
    hasModel: bool | None = None
    private: bool | None = None
    # Display-only aliases for canonical labels: {canonical_lower:
    # display_name}. Backend stores + serves them through; nothing on
    # the server consumes the alias for scoring / persistence — the
    # canonical key remains the source of truth on every detection,
    # ref, and import. Lets the UI rename a label without rewriting
    # every box. None means "leave existing aliases untouched".
    label_aliases: dict[str, str] | None = None
    # Per-label colour overrides, keyed by canonical-lower label →
    # #rrggbb. Same store-and-forward shape as label_aliases — the
    # backend doesn't render these, the FE composes them on top of
    # the stable hash-based palette. None = leave existing untouched.
    labelColours: dict[str, str] | None = None


@app.put(
    "/api/projects/{project_id}",
    dependencies=[Depends(require_project_owner)],
)
async def update_project(project_id: str, payload: UpdateProjectIn):
    if not project_dir(project_id).exists():
        raise HTTPException(404)
    from profanity import assert_clean

    manifest = load_manifest(project_id)
    data = payload.model_dump(exclude_none=True)

    # Validate every persisted text input. Tags + box labels feed back
    # into the model prompt and the user-facing UI, so banned terms
    # would otherwise leak through both.
    new_tags = data.get("tags")
    if isinstance(new_tags, list):
        for t in new_tags:
            if isinstance(t, str):
                assert_clean(t, field="label")
    new_edited_boxes = data.get("editedBoxes")
    if isinstance(new_edited_boxes, dict):
        for boxes in new_edited_boxes.values():
            if not isinstance(boxes, list):
                continue
            for b in boxes:
                if isinstance(b, dict) and isinstance(b.get("label"), str):
                    assert_clean(b["label"], field="label")

    # Sanitise label colour overrides: drop anything that isn't a
    # plain #rrggbb hex string. Keeps the manifest schema tight and
    # avoids stray HTML / colour functions making it into FE styles.
    new_label_colours = data.get("labelColours")
    if isinstance(new_label_colours, dict):
        clean: dict[str, str] = {}
        for k, v in new_label_colours.items():
            if not isinstance(k, str) or not isinstance(v, str):
                continue
            vs = v.strip()
            if len(vs) == 7 and vs.startswith("#") and all(c in "0123456789abcdefABCDEF" for c in vs[1:]):
                clean[k.strip().lower()] = "#" + vs[1:].lower()
        # Merge into existing rather than replace, so the FE can PUT
        # only the deltas (e.g. when a user edits a single label).
        merged = dict(manifest.get("labelColours") or {})
        merged.update(clean)
        # An explicit empty string would clear an override — strip
        # any keys with an empty value out of the merged map.
        for k, v in list(clean.items()):
            if v == "" or v == "#":
                merged.pop(k, None)
        data["labelColours"] = merged

    # Capture which images had their editedBoxes change so we can re-bake
    # the preview JPEGs after the manifest is saved. Compare per-image to
    # avoid re-baking on no-op saves (the FE does a debounced PUT every
    # ~500ms when other fields change).
    rebake_images: list[str] = []
    new_edited = data.get("editedBoxes")
    if isinstance(new_edited, dict):
        old_edited = manifest.get("editedBoxes") or {}
        for img_name, new_boxes in new_edited.items():
            if old_edited.get(img_name) != new_boxes:
                rebake_images.append(img_name)

    manifest.update(data)
    # Picking a dataset image (or reference) as the cover supersedes any
    # previously-uploaded custom cover, so clear the flag and let the picked
    # filename drive cover_thumb again.
    if data.get("cover"):
        manifest["cover_uploaded"] = False
    manifest["updatedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    save_manifest(project_id, manifest)
    # Drop the SWR payload cache + on-disk sidecars so the very next
    # /overview read computes from the freshly-saved manifest instead
    # of serving a pre-PUT snapshot. Without this a fast page refresh
    # right after a label-colour pick reads the still-cached payload
    # back (sidecar refresh is debounced 150 ms; the SWR cache holds
    # the previous mtime), and the user sees their old colour.
    _invalidate_project_payloads(project_id)

    # Re-bake the preview JPEG for each affected image, off the request
    # thread so the PUT returns fast. Failures are logged but don't fail
    # the save — the preview can always be regenerated later.
    if rebake_images and R2 is not None:
        loop = asyncio.get_running_loop()
        loop.run_in_executor(None, _rebake_previews_sync, project_id, rebake_images)
        # Refresh DINOv2 embeddings for the same set of images so the
        # similar-label search stays in sync with manual edits. The
        # incremental refresh skips unchanged boxes, so this is cheap
        # on label-only edits and only re-encodes when geometry or
        # mask actually changed.
        loop.run_in_executor(None, _refresh_embeddings_sync, project_id, rebake_images)

    return {"ok": True}


def _rebake_previews_sync(project: str, image_names: list[str]) -> None:
    """Re-render the preview JPEG for each image. Called off the request
    thread; never raises. Picks up masks from both auto-detections and
    user-edited boxes so manual segmentations show up immediately."""
    if R2 is None:
        return
    try:
        manifest = load_manifest(project)
    except Exception as e:
        print(f"[rebake] {project}: load_manifest failed: {e}")
        return
    edited_all = manifest.get("editedBoxes") or {}
    results_by_image = {r.get("image"): r for r in (manifest.get("results") or [])}

    for img_name in image_names:
        try:
            r = results_by_image.get(img_name)
            masks: list[dict | None] = []
            if r:
                for d in r.get("detections") or []:
                    if d.get("mask"):
                        masks.append(d["mask"])
            for eb in edited_all.get(img_name, []) or []:
                if eb.get("mask"):
                    masks.append(eb["mask"])

            data = R2.get_bytes(R2Storage.image_key(project, img_name))
            image = PILImage.open(io.BytesIO(data))
            buf = io.BytesIO()
            from preview import draw_preview as _draw_preview
            _draw_preview(image.copy(), masks).save(
                buf, format="JPEG", quality=72, optimize=True, progressive=True,
            )
            stem = Path(img_name).stem
            annotated_name = f"{stem}_annotated.jpg"
            output_key = R2Storage.output_key(project, annotated_name)
            R2.put_bytes(output_key, buf.getvalue(), content_type="image/jpeg")
            _invalidate_url_cache(output_key)
        except Exception as e:
            print(f"[rebake] {project}/{img_name} failed: {e}")


# ---------------------------------------------------------------------------
# Per-segmentation embeddings — DINOv2 features for similar-label discovery.
# ---------------------------------------------------------------------------

def _collect_segmentation_inputs(manifest: dict, *, image_names: set[str] | None = None):
    """Yield `(image, box_id, label, box_xyxy, polygons)` for every
    box in the manifest that has a usable mask. Pulls from both
    `editedBoxes` (manual edits, the user-facing source of truth) and
    auto detections so freshly-run inference is also covered."""
    edited_all = manifest.get("editedBoxes") or {}
    for img_name, boxes in edited_all.items():
        if image_names is not None and img_name not in image_names:
            continue
        if not isinstance(boxes, list):
            continue
        for b in boxes:
            if not isinstance(b, dict):
                continue
            mask = b.get("mask") or {}
            polys = mask.get("polygons") if isinstance(mask, dict) else None
            if not polys:
                continue
            try:
                box = [float(b.get("x0", 0)), float(b.get("y0", 0)),
                       float(b.get("x1", 0)), float(b.get("y1", 0))]
            except Exception:
                continue
            yield (
                str(img_name),
                str(b.get("id") or ""),
                (b.get("label") or "").strip(),
                box,
                polys,
            )

    # Don't yield detections for images that already have editedBoxes —
    # editedBoxes are the user-facing source of truth and the embedding
    # store would otherwise carry a duplicate row per physical box.
    images_with_edits = {img for img, boxes in edited_all.items() if boxes}
    for r in (manifest.get("results") or []):
        img_name = r.get("image")
        if not img_name:
            continue
        if image_names is not None and img_name not in image_names:
            continue
        if img_name in images_with_edits:
            continue
        for d in r.get("detections") or []:
            if not isinstance(d, dict):
                continue
            mask = d.get("mask") or {}
            polys = mask.get("polygons") if isinstance(mask, dict) else None
            if not polys:
                continue
            box = d.get("box_xyxy") or []
            if len(box) != 4:
                continue
            try:
                box = [float(v) for v in box]
            except Exception:
                continue
            # Use a deterministic synthetic id so re-encoding the same
            # auto detection updates the existing row instead of
            # appending a duplicate.
            box_id = f"auto:{img_name}:{round(box[0],1)}:{round(box[1],1)}:{round(box[2],1)}:{round(box[3],1)}"
            yield (
                str(img_name),
                box_id,
                (d.get("label") or "").strip().split(" (")[0],
                box,
                polys,
            )


def _refresh_embeddings_sync(project_id: str, image_names: list[str] | None = None) -> dict:
    """Compute / update DINOv2 embeddings for the project. Off the
    request thread; never raises. When `image_names` is None, walks
    the entire manifest; otherwise only the listed images.

    Removes embedding rows for boxes that no longer exist."""
    if not _EMBEDDINGS_ENABLED:
        return {"computed": 0, "removed": 0, "skipped": "embeddings_disabled"}
    try:
        import embeddings as _emb
        if not _emb.is_loaded():
            return {"computed": 0, "removed": 0, "skipped": "dinov2_not_loaded"}

        manifest = load_manifest(project_id) or empty_manifest(project_id)
        proj_dir = project_dir(project_id)
        target_images = set(image_names) if image_names else None

        # Collect every (box_id, ...) currently present so we can
        # diff against the saved store afterwards and drop dead rows.
        present_ids: set[str] = set()
        targets: list[tuple[str, str, str, list[float], list]] = []
        for tup in _collect_segmentation_inputs(manifest, image_names=target_images):
            img_name, box_id, label, box, polys = tup
            if not box_id:
                continue
            present_ids.add(box_id)
            targets.append(tup)

        # Match against the existing store. Skip rows that match
        # exactly — same box_id, same label, same box geometry, same
        # polygon vertex count — so unchanged boxes don't trigger an
        # encoder pass.
        existing_arr, existing_meta = _emb.load_store(proj_dir)
        existing_by_id = {str(m.get("box_id")): (i, m) for i, m in enumerate(existing_meta)}

        rows_to_upsert: list[tuple[dict, "np.ndarray"]] = []
        # Lazy image loader — one PIL.Image per image, decoded once
        # even if there are many boxes on it.
        image_cache: dict[str, "PILImage.Image"] = {}

        from embeddings import ENCODER_VERSION as _ENCODER_VERSION

        for img_name, box_id, label, box, polys in targets:
            existing = existing_by_id.get(box_id)
            box_sig = [round(v, 1) for v in box]
            poly_sig = sum(len(p) for p in polys)
            if existing is not None:
                _idx, m = existing
                same_box = (m.get("box_xyxy") or []) == box_sig
                same_poly = m.get("poly_count") == poly_sig
                same_label = m.get("label") == label
                has_size = "size_frac" in (m or {})
                # Encoder version gate: when the encode pipeline
                # changes (e.g. dropping the grey-background mask
                # isolation in v2), every legacy row needs to be
                # re-encoded under the new transform — otherwise
                # similarity search compares apples to oranges.
                same_encoder = m.get("encoder_version") == _ENCODER_VERSION
                if same_box and same_poly and same_label and has_size and same_encoder:
                    continue  # unchanged — skip the encode

            image_pil = image_cache.get(img_name)
            if image_pil is None:
                try:
                    image_pil = _pil_from_r2(project_id, img_name)
                except Exception as e:
                    print(f"[embeddings] {project_id}/{img_name} image fetch failed: {e}")
                    continue
                image_cache[img_name] = image_pil
            vec = _emb.encode_segmentation(image_pil, polys, box)
            if vec is None:
                continue
            # Record the box's size relative to the image so the
            # similarity search can weight by scale as well as
            # appearance — two genuinely-similar objects at very
            # different scales used to score below threshold and
            # never appear; the combined score in find_similar
            # handles that.
            iw, ih = image_pil.size
            box_w = max(1.0, box[2] - box[0])
            box_h = max(1.0, box[3] - box[1])
            size_frac = float((box_w * box_h) / max(1.0, float(iw * ih)))
            rows_to_upsert.append((
                {
                    "box_id": box_id,
                    "image": img_name,
                    "label": label,
                    "box_xyxy": box_sig,
                    "poly_count": poly_sig,
                    "size_frac": round(size_frac, 6),
                    "encoder_version": _ENCODER_VERSION,
                },
                vec,
            ))

        if rows_to_upsert:
            _emb.upsert_rows(proj_dir, rows_to_upsert)

        # Drop rows whose box no longer exists.
        if target_images is None:
            stale = [str(m.get("box_id")) for m in existing_meta if str(m.get("box_id")) not in present_ids]
        else:
            stale = [
                str(m.get("box_id"))
                for m in existing_meta
                if m.get("image") in target_images and str(m.get("box_id")) not in present_ids
            ]
        removed = _emb.remove_rows(proj_dir, stale) if stale else 0

        return {"computed": len(rows_to_upsert), "removed": removed}
    except Exception as e:
        print(f"[embeddings] refresh failed for {project_id}: {e}")
        return {"computed": 0, "removed": 0, "error": str(e)}


@app.delete(
    "/api/projects/{project_id}",
    # DESTROY is creator-only: a Project editor (or even the Project owner) can't
    # permanently delete a dataset someone else made. They can detach it from the
    # Project via DELETE /api/containers/{id}/datasets/{id} instead.
    dependencies=[Depends(require_dataset_creator)],
)
async def delete_project(project_id: str):
    p = project_dir(project_id)
    if not p.exists():
        raise HTTPException(404)
    # If this dataset lives in a Project (container), drop it from that
    # container's dataset list so destroying it doesn't leave a dangling id.
    try:
        _m = load_manifest(project_id, copy=False) or {}
        _owning_cid = (_m.get("container_id") or "").strip()
        if _owning_cid:
            _c = containers.load_container(_owning_cid)
            if _c and project_id in (_c.get("dataset_ids") or []):
                _c["dataset_ids"] = [d for d in _c["dataset_ids"] if d != project_id]
                containers.save_container(_c, bump_updated=False)
    except Exception as e:
        print(f"[delete_project] container prune failed for {project_id}: {e}")
    # Tombstone the id BEFORE any teardown so a write that races this
    # delete (a debounced label-metadata autosave, a like/favourite, or
    # an in-flight job/sidecar flush) can't recreate the folder via
    # save_manifest after rmtree and resurrect the project.
    _mark_project_deleted(project_id)
    removed = False
    try:
        # Cancel any in-flight or queued jobs for this project before nuking
        # the folder. Without this, a stuck/hung runner that didn't honour the
        # earlier cancel keeps the job in the JobManager as running/queued —
        # later /jobs/active probes for this id (or a re-created project that
        # somehow shares the id) can re-attach to it.
        for j in list(state["jobs"].jobs.values()):
            if j.project != project_id:
                continue
            if j.status in ("queued", "running"):
                state["jobs"].cancel(j.id)
                # Stronger than cancel(): force the status terminal even if the
                # runner is hung and never checks cancel_event. We never want
                # a deleted project's job lingering in the listing.
                j.status = "cancelled"
                if j.finished_at is None:
                    j.finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        shutil.rmtree(p)
        removed = True
        invalidate_manifest_cache(project_id)
        if R2 is not None:
            R2.delete_prefix(R2Storage.project_prefix(project_id))
    except Exception:
        # If we never removed the folder, lift the tombstone so the
        # still-present project isn't permanently frozen against writes.
        # If rmtree already succeeded, keep the tombstone (the project is
        # gone; we just want to block resurrection) and surface the error.
        if not removed:
            _unmark_project_deleted(project_id)
        raise
    return {"ok": True}


class DuplicateProjectIn(BaseModel):
    # Optional name for the copy; defaults to "Copy of <source name>".
    name: str | None = None


@app.post(
    "/api/projects/{project_id}/duplicate",
    # Owner-gated: you duplicate your OWN datasets (the workspace card action).
    # Same gate as rename/settings — not the stricter creator-only delete gate.
    dependencies=[Depends(require_project_owner)],
)
async def duplicate_project(
    project_id: str,
    payload: DuplicateProjectIn | None = None,
    user: str = Depends(current_user),
):
    """Deep-copy a dataset into a brand-new project owned by the caller.

    A project is fully contained in its on-disk folder (manifest.json +
    imports/ image bytes + references/ + augmentations/ + image_embeddings/
    + cache sidecars) plus, for legacy V1 datasets, an R2 prefix of image/
    output bytes. We byte-copy both, then rewrite the new manifest's identity
    fields. Every filename inside the manifest is relative to those dirs, and
    per-image/per-reference ids are project-scoped, so nothing else needs
    rewriting. The copy is a standalone dataset — it does not inherit the
    source's Project (container) membership or derived-parent linkage."""
    src = project_dir(project_id)
    if not src.exists() or not manifest_path(project_id).exists():
        raise HTTPException(404, "project not found")
    src_manifest = load_manifest(project_id)
    if not src_manifest:
        raise HTTPException(404, "no manifest")

    # Fresh id; retry on the astronomically unlikely collision / tombstone.
    new_id = _uuid.uuid4().hex
    for _ in range(5):
        if not store.dataset_exists(new_id) and not _is_project_deleted(new_id):
            break
        new_id = _uuid.uuid4().hex

    name = (payload.name.strip() if (payload and payload.name and payload.name.strip()) else "")
    if not name:
        name = f"Copy of {src_manifest.get('name') or 'dataset'}"
    from profanity import assert_clean
    try:
        assert_clean(name, field="project name")
    except HTTPException:
        # A custom name that trips the gate falls back to a safe default
        # rather than failing the whole (potentially multi-GB) copy.
        name = f"Copy of {src_manifest.get('name') or 'dataset'}"

    # Reserve the destination folder (copytree requires it not to exist).
    dst = store.reserve_dataset_dir(new_id, name)

    loop = asyncio.get_running_loop()

    def _copy() -> None:
        # Whole-folder byte copy. On a local SSD this is fast even for a
        # multi-GB dataset; run off the event loop so concurrent requests
        # aren't blocked. copytree creates dst (which must not pre-exist).
        shutil.copytree(src, dst)
        # Drop copied cache sidecars (workspace_card, overview/initial, etc.)
        # so save_manifest regenerates them for the NEW id instead of
        # serving the source's cached payloads. The manifest is the only
        # top-level JSON that is source-of-truth, not a derivable cache.
        try:
            for f in dst.glob("*.json"):
                if f.name != "dataset.json":
                    f.unlink()
        except Exception as e:
            print(f"[duplicate] sidecar cleanup failed for {new_id}: {e}")
        # (SaaS build also server-side-copied the R2 prefix here; locally the
        # copytree above already carried every byte.)

    try:
        await loop.run_in_executor(None, _copy)
    except Exception as e:
        # Best-effort cleanup of a half-written copy so a failed duplicate
        # doesn't leave an orphan folder in the workspace.
        try:
            if dst.exists():
                shutil.rmtree(dst)
        except Exception:
            pass
        print(f"[duplicate] copy failed {project_id} -> {new_id}: {e}")
        raise HTTPException(500, "duplicate failed")

    # Rewrite identity on the copied manifest. Load the freshly-copied one so
    # we carry every field (tags, imports, references, labelColours, tiling,
    # settingsLastRun, …) and only override what must change.
    m = load_manifest(new_id) or dict(src_manifest)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    m["id"] = new_id
    m["name"] = name
    m["owner"] = user
    m["createdBy"] = user
    m["createdAt"] = now
    m["updatedAt"] = now
    m["likedBy"] = []
    m["favouritedBy"] = []
    # Standalone copy — not part of the source's Project, not a derived child.
    m.pop("container_id", None)
    m.pop("derived", None)
    save_manifest(new_id, m)
    try:
        add_event("project_duplicate", project=new_id, source=project_id, user=user or "anonymous")
    except Exception:
        pass
    return {"id": new_id, "name": name}


class RenameIn(BaseModel):
    name: str


@app.post(
    "/api/projects/{project_id}/rename",
    dependencies=[Depends(require_project_owner)],
)
async def rename_project(project_id: str, payload: RenameIn):
    """Update the manifest's display name. The folder + R2 prefix stay
    addressed by the project's UUID, so renames cost only a manifest write —
    no R2 moves, no broken links, two users can have the same display name."""
    from profanity import assert_clean

    new_name = (payload.name or "").strip()
    if not new_name:
        raise HTTPException(400, "name required")
    assert_clean(new_name, field="project name")
    if not project_dir(project_id).exists():
        raise HTTPException(404)
    manifest = load_manifest(project_id)
    manifest["name"] = new_name
    manifest["updatedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    save_manifest(project_id, manifest)
    return {"id": project_id, "name": new_name}


@app.post(
    "/api/projects/{project_id}/images",
    dependencies=[
        Depends(require_project_owner),
        # Uploads burn credits (1 credit = 800 uploaded images). Gate
        # at the request boundary so an over-cap user can't keep
        # filling their storage column or the NSFW GPU queue.
    ],
)
async def add_images(project_id: str, images: list[UploadFile] = File(...), user: str | None = None):
    """Two-phase upload: file receive → NSFW gate → R2 put + manifest.

    Phase 1 (`upload` job): per-file dedup + image validation, then stage
    raw bytes to a temp folder on disk. Fast — no GPU, no R2.

    Phase 2 (`nsfw_check` job): NSFW classifier on each staged file. If it
    passes, push to R2 + add to manifest as pending. If it fails, log the
    block + drop the temp file. Surfaces as its own job in the Terminal so
    operators can see GPU spend on safety checks separately from raw upload.
    """
    import hashlib

    r2 = r2_required()
    proj = project_dir(project_id)
    if not proj.exists():
        raise HTTPException(404)
    if not images:
        raise HTTPException(400, "no images")

    # Reject obviously-abusive batch shapes (too many files at once).
    # The per-file size check fires inside the read loop below so we
    # can reject the FIRST oversize file without buffering the whole
    # batch into memory.
    if len(images) > MAX_FILES_PER_UPLOAD_BATCH:
        raise HTTPException(
            413,
            f"too many files in one upload ({len(images)} > {MAX_FILES_PER_UPLOAD_BATCH})",
        )

    # Read all uploaded bytes upfront — UploadFile streams are tied to the
    # request lifetime, so we capture them before run_inline kicks off.
    file_blobs: list[tuple[str, bytes, str | None]] = []
    for f in images:
        fn = Path(f.filename or "image").name
        data = await f.read()
        if len(data) > MAX_UPLOAD_BYTES_PER_FILE:
            raise HTTPException(
                413,
                f"file too large: {fn} is {len(data)} bytes "
                f"(max {MAX_UPLOAD_BYTES_PER_FILE})",
            )
        file_blobs.append((fn, data, f.content_type))

    pending_dir = proj / "_pending"
    pending_dir.mkdir(parents=True, exist_ok=True)

    # Records passed from the upload runner to the NSFW runner. Each entry
    # is (filename, sha, temp_path_on_disk, content_type, size_dict).
    staged: list[tuple[str, str, Path, str | None, dict]] = []
    upload_skipped: list[str] = []
    upload_rejected: list[dict] = []

    async def upload_runner(job, emit, cancel_event):
        await emit("status", {"phase": "running", "total": len(file_blobs)})
        loop = asyncio.get_running_loop()
        manifest = load_manifest(project_id) or empty_manifest(project_id)
        results = manifest.get("results", []) or []
        existing_names = {r.get("image") for r in results}
        existing_hashes: dict[str, str] = {}
        for r in results:
            h = r.get("hash")
            if isinstance(h, str) and h:
                existing_hashes[h] = r.get("image") or ""

        # The per-file work (SHA, PIL open, disk write) is all synchronous
        # IO + CPU; do it on a thread so the asyncio loop stays responsive
        # for everything else the server is serving (heartbeat, project
        # lookups, the FE's manifest poll, etc.).
        def stage_one(fn: str, data: bytes, ctype: str | None):
            if fn in existing_names:
                return ("skipped", fn)
            sha = hashlib.sha256(data).hexdigest()
            if sha in existing_hashes:
                return ("duplicate", fn, sha, existing_hashes[sha])
            try:
                with PILImage.open(io.BytesIO(data)) as img:
                    size = {"width": img.width, "height": img.height}
            except Exception:
                return ("skipped", fn)
            tmp_path = pending_dir / fn
            tmp_path.write_bytes(data)
            return ("staged", fn, sha, tmp_path, ctype, size)

        for i, (fn, data, ctype) in enumerate(file_blobs, 1):
            if cancel_event.is_set():
                break
            await emit("progress", {"index": i, "total": len(file_blobs), "image": fn})
            outcome = await loop.run_in_executor(None, stage_one, fn, data, ctype)
            kind = outcome[0]
            if kind == "skipped":
                upload_skipped.append(outcome[1])
                continue
            if kind == "duplicate":
                _, fn_, sha_, dup_of = outcome
                upload_rejected.append({
                    "file": fn_, "reason": "duplicate",
                    "duplicate_of": dup_of,
                })
                continue
            # staged
            _, fn_, sha_, tmp_path, ctype_, size = outcome
            staged.append((fn_, sha_, tmp_path, ctype_, size))
            existing_names.add(fn_)
            existing_hashes[sha_] = fn_
            await emit("result", {"index": i, "image": fn_, "staged": True})

        return {"staged": [s[0] for s in staged], "skipped": upload_skipped, "duplicates": upload_rejected}

    upload_summary = await state["jobs"].run_inline(
        kind="upload",
        project=project_id,
        params={"count": len(file_blobs)},
        user=user or "anonymous",
        runner=upload_runner,
        n_images=len(file_blobs),
    )

    # If nothing made it past stage 1, skip the NSFW phase entirely so we
    # don't spawn a no-op job in the Terminal.
    added: list[str] = []
    nsfw_rejected: list[dict] = []

    if staged:
        async def nsfw_runner(job, emit, cancel_event):
            await emit("status", {"phase": "running", "total": len(staged)})
            loop = asyncio.get_running_loop()

            # Sync per-file work pushed to a worker thread. NSFW classifier
            # is GPU-bound and R2 PUT is network-bound — both block the
            # asyncio loop if run inline. We still write the manifest from
            # the async side to avoid concurrent writers stepping on it.
            def check_and_upload(fn: str, sha: str, tmp_path: Path, ctype: str | None):
                try:
                    score, cls = nsfw_score(state["nsfw"], tmp_path)
                except Exception as e:
                    print(f"[nsfw] check failed for {fn}: {e}")
                    score, cls = 0.0, ""
                verdict = "BLOCK" if score >= NSFW_THRESHOLD else "ok"
                print(f"[nsfw] {verdict} {fn} score={score:.3f} class={cls or '-'}")
                if score >= NSFW_THRESHOLD:
                    return ("blocked", fn, round(score, 3), cls)
                data = tmp_path.read_bytes()
                r2.put_bytes(R2Storage.image_key(project_id, fn), data, content_type=ctype or None)
                return ("ok", fn, sha)

            for i, (fn, sha, tmp_path, ctype, size) in enumerate(staged, 1):
                if cancel_event.is_set():
                    break
                await emit("progress", {"index": i, "total": len(staged), "image": fn})
                try:
                    outcome = await loop.run_in_executor(
                        None, check_and_upload, fn, sha, tmp_path, ctype,
                    )
                    if outcome[0] == "blocked":
                        _, fn_, score, cls = outcome
                        nsfw_rejected.append({
                            "file": fn_, "reason": "nsfw",
                            "score": score, "class": cls,
                        })
                        add_event(
                            "nsfw_block",
                            project=project_id, file=fn_,
                            score=score, classification=cls,
                            user=job.user,
                        )
                        continue

                    _, fn_, sha_ = outcome
                    manifest = load_manifest(project_id) or empty_manifest(project_id)
                    results = manifest.get("results", []) or []
                    new_result = {
                        "image": fn_,
                        "annotated": None,
                        "size": size,
                        "detections": [],
                        "pending": True,
                        "hash": sha_,
                    }
                    results.append(new_result)
                    manifest["results"] = results
                    # Randomise the project's cover the first time
                    # an image lands. Subsequent adds keep whatever
                    # was picked (or the user's manual override).
                    _ensure_random_cover(manifest)
                    manifest["updatedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                    save_manifest(project_id, manifest)
                    added.append(fn_)
                    await emit("result", {"index": i, "result": new_result})
                finally:
                    # Clean up the temp file no matter what — we don't
                    # want NSFW-blocked content sitting on disk either.
                    try:
                        tmp_path.unlink(missing_ok=True)
                    except Exception:
                        pass

            return {"added": added, "rejected": nsfw_rejected}

        await state["jobs"].run_inline(
            kind="nsfw_check",
            project=project_id,
            params={"count": len(staged)},
            user=user or "anonymous",
            runner=nsfw_runner,
            n_images=len(staged),
        )

    # Wipe any leftover temp files (cancelled job, errored runner) so the
    # _pending dir doesn't grow unbounded.
    try:
        for f in pending_dir.iterdir():
            try:
                f.unlink()
            except Exception:
                pass
    except FileNotFoundError:
        pass

    rejected_combined = upload_rejected + nsfw_rejected
    return {
        "added": added,
        "skipped": upload_skipped,
        "rejected": rejected_combined,
    }


class ImagesFromUrlsRequest(BaseModel):
    urls: list[str]
    # The search term that produced these URLs. Used as the filename
    # prefix so imported images self-document their origin.
    query: str | None = None


@app.post(
    "/api/projects/{project_id}/images_from_urls",
    dependencies=[
        Depends(require_project_owner),
    ],
)
async def add_images_from_urls(project_id: str, body: ImagesFromUrlsRequest, user: str | None = None):
    """Pull a list of image URLs (typically from an Openverse search)
    straight into the project. Single-phase, single job — no NSFW gate
    (these come from a curated CC-licensed source) and no GroundingDINO
    pre-pass (the user already vetted each thumbnail by hand).

    Per URL: download bytes → validate as PIL-decodable image → SHA
    dedupe vs the manifest → unique filename → R2 PUT → manifest entry.
    Returns the same shape as `add_images` (added / skipped / rejected)
    so the frontend can reuse its post-upload refresh path."""
    import hashlib
    import uuid as _uuid
    import openverse
    from urllib.parse import urlparse

    r2 = r2_required()
    proj = project_dir(project_id)
    if not proj.exists():
        raise HTTPException(404)
    raw_urls = [u.strip() for u in (body.urls or []) if isinstance(u, str) and u.strip()]
    # Cap the URL list so a single request can't queue thousands of
    # downloads against the backend's bandwidth/disk. 100 is well
    # above the Openverse panel's typical batch (24 results/page).
    if len(raw_urls) > 100:
        raise HTTPException(400, "too many urls, max 100 per request")
    if not raw_urls:
        raise HTTPException(400, "no urls")
    # Dedupe within the batch — Openverse pagination occasionally
    # surfaces the same URL on multiple pages, and a client-side bug
    # could double-submit. First-occurrence wins so order is stable.
    seen_in_batch: set[str] = set()
    urls: list[str] = []
    for u in raw_urls:
        if u not in seen_in_batch:
            seen_in_batch.add(u)
            urls.append(u)

    # Slug the search term into a filename-safe prefix. Empty / non-
    # alphanumeric-only queries fall back to "openverse" so the saved
    # filenames stay legible.
    raw_q = (body.query or "").strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", raw_q).strip("_")
    if not slug:
        slug = "openverse"
    # Cap length so a stray giant query doesn't blow past sane filename
    # limits. R2 keys + manifest entries stay short.
    slug = slug[:48]

    added: list[str] = []
    skipped: list[str] = []
    rejected: list[dict] = []

    async def runner(job, emit, cancel_event):
        await emit("status", {"phase": "running", "total": len(urls)})
        loop = asyncio.get_running_loop()

        # Download in parallel (network-bound, low CPU) on the executor
        # pool so the asyncio loop stays responsive.
        blobs: list[tuple[bytes | None, str | None]] = await loop.run_in_executor(
            None, lambda: openverse.download_images_bytes(urls),
        )

        manifest = load_manifest(project_id) or empty_manifest(project_id)
        results = manifest.get("results", []) or []
        existing_names: set[str] = {r.get("image") for r in results if r.get("image")}
        existing_hashes: dict[str, str] = {}
        # Set of URLs already imported into this project — driven by
        # the `source.url` field stamped onto each manifest entry by
        # this same handler. Lets us reject re-import attempts before
        # we even hit the network, both for the request-level cache
        # the user asked for and as a safety net against any client
        # that double-sends an already-imported list.
        existing_source_urls: set[str] = set()
        for r in results:
            h = r.get("hash")
            if isinstance(h, str) and h:
                existing_hashes[h] = r.get("image") or ""
            src = r.get("source")
            if isinstance(src, dict):
                u = src.get("url")
                if isinstance(u, str) and u:
                    existing_source_urls.add(u)

        # Pick a unique filename. Format: <query-slug>_<12-hex-uuid>.<ext>
        # The query slug self-documents what the image came from, the
        # 12-hex tail (~48 bits of entropy) is collision-proof for any
        # plausible single-project import scale.
        def _filename_for(url: str, _sha: str, ctype: str | None) -> str:
            ext = ""
            try:
                path = urlparse(url).path
                base = Path(path).name
                if "." in base:
                    e = base.rsplit(".", 1)[1].lower()
                    if e in ("jpg", "jpeg", "png", "webp", "gif", "bmp"):
                        ext = "." + e
            except Exception:
                pass
            if not ext and ctype:
                # image/jpeg → .jpg, image/png → .png
                guess = ctype.split(";")[0].strip().lower()
                mapping = {
                    "image/jpeg": ".jpg",
                    "image/jpg": ".jpg",
                    "image/png": ".png",
                    "image/webp": ".webp",
                    "image/gif": ".gif",
                    "image/bmp": ".bmp",
                }
                ext = mapping.get(guess, "")
            if not ext:
                ext = ".jpg"
            # Loop until we land on a non-colliding name. Practically
            # one shot — 48 bits of entropy means even at 100 imports
            # the collision odds are sub-billionth — but keep the loop
            # in case someone retries the same URL set repeatedly.
            while True:
                suffix = _uuid.uuid4().hex[:12]
                fn = f"{slug}_{suffix}{ext}"
                if fn not in existing_names:
                    return fn

        for i, (url, (data, ctype)) in enumerate(zip(urls, blobs), 1):
            if cancel_event.is_set():
                break
            await emit("progress", {"index": i, "total": len(urls), "url": url})

            if url in existing_source_urls:
                rejected.append({"url": url, "reason": "already_imported"})
                skipped.append(url)
                await emit("result", {"index": i, "url": url, "ok": False, "reason": "already_imported"})
                continue

            if data is None:
                rejected.append({"url": url, "reason": "download_failed"})
                await emit("result", {"index": i, "url": url, "ok": False, "reason": "download_failed"})
                continue

            # Validate the bytes are actually a decodable image. Catches
            # 200-OK HTML error pages that Openverse occasionally serves
            # when the upstream provider has rotated the URL.
            try:
                with PILImage.open(io.BytesIO(data)) as img:
                    img.verify()
            except Exception:
                rejected.append({"url": url, "reason": "not_an_image"})
                await emit("result", {"index": i, "url": url, "ok": False, "reason": "not_an_image"})
                continue

            # Reject blank / placeholder / 1×1-pixel images at the
            # source. Provider URLs sometimes round-trip a tiny stub
            # or a uniform black tile with a 200 OK — `img.verify()`
            # passes them, but they're useless to label and clutter
            # the project. Frontend probe catches most thumbnails;
            # this catches what the full-resolution URL serves up.
            try:
                from image_utils import is_blank_image_bytes
                blank, reason = is_blank_image_bytes(data)
            except Exception as e:
                print(f"[import_urls] blank-image check failed for {url}: {e}")
                blank, reason = False, None
            if blank:
                rejected.append({"url": url, "reason": "blank_image", "detail": reason})
                await emit("result", {"index": i, "url": url, "ok": False, "reason": "blank_image"})
                continue

            # Detect + crop a uniform border (Openverse occasionally
            # serves stock photos with white/black mats around the
            # actual content). Returns original bytes unchanged when
            # there's nothing to crop, so the no-border path is just
            # one decode + one np.median + one variance check.
            try:
                from image_utils import maybe_crop_border_bytes
                data, size = await loop.run_in_executor(
                    None, lambda: maybe_crop_border_bytes(data, ctype),
                )
            except Exception as e:
                print(f"[import_urls] border-crop failed for {url}: {e}")
                # Fall back to original bytes + measured size.
                with PILImage.open(io.BytesIO(data)) as img:
                    size = {"width": img.width, "height": img.height}

            sha = hashlib.sha256(data).hexdigest()
            if sha in existing_hashes:
                rejected.append({"url": url, "reason": "duplicate", "duplicate_of": existing_hashes[sha]})
                skipped.append(url)
                await emit("result", {"index": i, "url": url, "ok": False, "reason": "duplicate"})
                continue

            fn = _filename_for(url, sha, ctype)
            try:
                await loop.run_in_executor(
                    None,
                    lambda: r2.put_bytes(R2Storage.image_key(project_id, fn), data, content_type=ctype or None),
                )
            except Exception as e:
                print(f"[import_urls] R2 put failed for {fn}: {e}")
                rejected.append({"url": url, "reason": "storage_failed"})
                await emit("result", {"index": i, "url": url, "ok": False, "reason": "storage_failed"})
                continue

            new_result = {
                "image": fn,
                "annotated": None,
                "size": size,
                "detections": [],
                "pending": True,
                "hash": sha,
                "source": {"kind": "openverse", "url": url},
            }
            results.append(new_result)
            manifest["results"] = results
            # Randomise the cover on first image-add. No-op once a
            # cover has been picked or the user explicitly set one.
            _ensure_random_cover(manifest)
            manifest["updatedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            save_manifest(project_id, manifest)
            existing_names.add(fn)
            existing_hashes[sha] = fn
            existing_source_urls.add(url)
            added.append(fn)
            await emit("result", {"index": i, "url": url, "ok": True, "image": fn})

        return {"added": added, "skipped": skipped, "rejected": rejected}

    await state["jobs"].run_inline(
        kind="import_urls",
        project=project_id,
        params={"count": len(urls), "source": "openverse"},
        user=user or "anonymous",
        runner=runner,
        n_images=len(urls),
    )

    return {"added": added, "skipped": skipped, "rejected": rejected}


class RenameLabelRequest(BaseModel):
    old_label: str
    new_label: str


@app.post(
    "/api/projects/{project_id}/labels/rename",
    dependencies=[Depends(require_project_owner)],
)
async def rename_label(project_id: str, body: RenameLabelRequest):
    """Rename a label across the entire project. Updates the project's
    tag list, every detection across all images, and every edited box
    so the change pulls through to the auto-label prompt, the existing
    annotated detections, and the user's manual edits in one shot."""
    proj = project_dir(project_id)
    if not proj.exists():
        raise HTTPException(404)
    from profanity import assert_clean

    old = (body.old_label or "").strip()
    new = (body.new_label or "").strip()
    if not old or not new:
        raise HTTPException(400, "old_label and new_label are required")
    assert_clean(new, field="label")
    if old.lower() == new.lower():
        # Trivial — same name. Nothing to do.
        return {"ok": True, "renamed": 0}

    manifest = load_manifest(project_id)
    if not manifest:
        raise HTTPException(404)

    renamed = 0

    # Project tag list — case-insensitive match, dedupe so renaming
    # to an existing tag merges rather than duplicates.
    tags = manifest.get("tags", []) or []
    out_tags: list[str] = []
    for t in tags:
        s = (t or "").strip()
        if s.lower() == old.lower():
            if new not in out_tags:
                out_tags.append(new)
            renamed += 1
        else:
            if s not in out_tags:
                out_tags.append(s)
    manifest["tags"] = out_tags

    # Detections on every image.
    for r in manifest.get("results", []) or []:
        for det in r.get("detections", []) or []:
            label = (det.get("label") or "").strip()
            if label.lower() == old.lower():
                det["label"] = new
                renamed += 1

    # User-edited boxes — same rename applies. Boxes carry the label
    # that drives the right-hand list inside the image editor, so
    # this is what makes the rename "pull through" to the UI.
    edited = manifest.get("editedBoxes", {}) or {}
    for boxes in edited.values():
        if not isinstance(boxes, list):
            continue
        for b in boxes:
            if not isinstance(b, dict):
                continue
            label = (b.get("label") or "").strip()
            if label.lower() == old.lower():
                b["label"] = new
                renamed += 1

    manifest["updatedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    save_manifest(project_id, manifest)

    # (synonyms cache removed with the GroundingDINO pipeline)

    return {"ok": True, "renamed": renamed, "tags": out_tags}


@app.post(
    "/api/projects/{project_id}/images/{filename}/clear_labels",
    dependencies=[Depends(require_project_owner)],
)
async def clear_image_labels(project_id: str, filename: str):
    """Reset a single image to its unlabelled state — drops all
    detections + edited boxes, removes the rendered annotated preview
    from R2, clears the verdict, and flips pending=True so the image
    rejoins the auto-label queue. Other images aren't affected."""
    proj = project_dir(project_id)
    if not proj.exists():
        raise HTTPException(404)
    fn = Path(filename).name

    manifest = load_manifest(project_id)
    if not manifest:
        raise HTTPException(404)

    results = manifest.get("results", []) or []
    target = None
    for r in results:
        if r.get("image") == fn:
            target = r
            break
    if target is None:
        raise HTTPException(404, f"image {fn} not in manifest")

    # Annotated preview — delete the R2 object so the file doesn't
    # linger orphaned. Best-effort: a stale preview is recoverable
    # from the manifest reset; a failed delete shouldn't block the
    # state change.
    annotated = target.get("annotated")
    if annotated and R2 is not None:
        try:
            R2.delete(R2Storage.output_key(project_id, annotated))
        except Exception as e:
            print(f"[clear_labels] failed to delete annotated {annotated}: {e}")

    target["detections"] = []
    target["annotated"] = None
    target["pending"] = True
    # Drop any stored validation/verdict-on-detection data on the
    # image too — those were tied to the old detections.
    target.pop("missingObjects", None)
    target.pop("missingReason", None)

    edited = manifest.get("editedBoxes", {}) or {}
    if fn in edited:
        del edited[fn]
        manifest["editedBoxes"] = edited
    verdicts = manifest.get("verdicts", {}) or {}
    if fn in verdicts:
        del verdicts[fn]
        manifest["verdicts"] = verdicts

    manifest["updatedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    save_manifest(project_id, manifest)
    return {"ok": True, "image": fn}


@app.post(
    "/api/v2/projects/{project_id}/clear_all_annotations",
    dependencies=[Depends(require_project_owner)],
)
async def clear_all_annotations(project_id: str):
    """Wipe every detection + editedBoxes off every import in the
    project, drop the labelled-preview JPEGs, flip pending=True so
    everything re-enters the auto-label queue. Project-level tags /
    labelColours / label_aliases are preserved — only the per-image
    annotation data is reset."""
    proj = project_dir(project_id)
    if not proj.exists():
        raise HTTPException(404)

    lock = await _manifest_write_lock(project_id)
    async with lock:
        manifest = load_manifest(project_id) or {}
        imports = manifest.get("imports") or []
        cleared = 0
        for entry in imports:
            if not isinstance(entry, dict):
                continue
            import_id = entry.get("id")
            entry["detections"] = []
            entry["timings"] = {}
            entry["labelled"] = False
            entry["pending"] = True
            entry.pop("editedBoxes", None)
            entry.pop("editedBoxesSet", None)
            entry.pop("editedAt", None)
            entry.pop("missingObjects", None)
            entry.pop("missingReason", None)
            entry.pop("verdict", None)
            if import_id:
                try:
                    _invalidate_labelled_preview(project_id, import_id)
                except Exception:
                    pass
                cleared += 1
        # v1-era project-level dicts keyed by filename — wipe in full.
        if "editedBoxes" in manifest:
            manifest["editedBoxes"] = {}
        if "verdicts" in manifest:
            manifest["verdicts"] = {}
        manifest["updatedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        save_manifest(project_id, manifest)

    try:
        _invalidate_project_payloads(project_id)
    except Exception:
        pass
    return {"ok": True, "cleared": cleared}


@app.delete(
    "/api/projects/{project_id}/images/{filename}",
    dependencies=[Depends(require_project_owner)],
)
async def delete_image(project_id: str, filename: str):
    proj = project_dir(project_id)
    if not proj.exists():
        raise HTTPException(404)
    fn = Path(filename).name

    if R2 is not None:
        R2.delete(R2Storage.image_key(project_id, fn))
        annotated_name = f"{Path(fn).stem}_annotated.jpg"
        R2.delete(R2Storage.output_key(project_id, annotated_name))

    manifest = load_manifest(project_id)
    manifest["results"] = [r for r in manifest.get("results", []) or [] if r.get("image") != fn]
    if "verdicts" in manifest and isinstance(manifest["verdicts"], dict):
        manifest["verdicts"].pop(fn, None)
    if "editedBoxes" in manifest and isinstance(manifest["editedBoxes"], dict):
        manifest["editedBoxes"].pop(fn, None)
    if manifest.get("cover") == fn:
        manifest["cover"] = None
    manifest["updatedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    save_manifest(project_id, manifest)
    return {"ok": True}


class ScheduleJobIn(BaseModel):
    project: str
    kind: str  # "label" | "segment"
    user: str | None = None
    params: dict = {}


@app.post("/api/jobs")
async def schedule_job(
    payload: ScheduleJobIn,
    request: Request,
    user: str = Depends(current_user),
):
    """Queue a job. Returns the job id; subscribe to /api/jobs/{id}/events
    for live progress. Requires WRITE access to the target project — the
    dataset's own creator, OR (for a dataset in a Project) any editor/owner of
    that Project, so a Project editor can run auto-labelling / training on a
    dataset a teammate created."""
    if payload.kind not in ("label_charlie", "purge_label"):
        raise HTTPException(400, f"unknown kind: {payload.kind}")
    proj = project_dir(payload.project)
    if not proj.exists():
        raise HTTPException(404, "project not found")
    # copy=False — we only read owner + counts here. Deepcopying a
    # 30MB manifest just to schedule a job was ~300-500ms of dead
    # weight before the label_charlie job runner even picks up.
    project_manifest = load_manifest(payload.project, copy=False) or {}
    proj_owner = (project_manifest.get("owner") or "").strip().lower()
    # Write access via Project membership (editor+) OR dataset ownership; legacy
    # unowned + demo ("anonymous") datasets stay open as before.
    writable = containers.dataset_access(project_manifest, user)["writable"]
    if not (writable or not proj_owner or proj_owner == "anonymous"):
        raise HTTPException(403, "not your project")
    # Credit gate. purge_label is a cleanup operation (removes labels,
    # doesn't run inference) so it doesn't burn credits and stays open
    # even when a user is over cap; everything else schedules real GPU
    # work and must be gated.
    if payload.kind in ("label", "label_lite", "label_charlie", "segment", "train"):
        # Run in a thread so the event loop isn't blocked while
        # _user_usage_counters iterates all project manifests on a
        # cache miss (can take 2-5s on a multi-project account,
        # causing the FE "Starting…" state to linger).
        pass  # credit gate removed (portable build)

    n_images = 0
    # Reuse `project_manifest` (already loaded copy=False above) for
    # every n_images count — no second deepcopy of the same dict.
    if payload.kind == "label":
        manifest = project_manifest
        n_images = sum(1 for r in (manifest.get("results", []) or []) if r.get("pending"))
        if n_images == 0:
            raise HTTPException(400, "no pending images")
    elif payload.kind == "label_lite":
        manifest = project_manifest
        all_images = [r.get("image") for r in (manifest.get("results", []) or []) if r.get("image")]
        requested = payload.params.get("images")
        if requested:
            n_images = sum(1 for n in all_images if n in set(requested))
        else:
            n_images = len(all_images)
        if n_images == 0:
            raise HTTPException(400, "no images to process")
    elif payload.kind == "train":
        manifest = project_manifest
        edited = manifest.get("editedBoxes") or {}
        n_images = sum(1 for v in edited.values() if isinstance(v, list) and v)
        if n_images == 0:
            raise HTTPException(400, "no labelled images yet — label some first")
    elif payload.kind == "label_charlie":
        manifest = project_manifest
        force_relabel = bool((payload.params or {}).get("force_relabel"))
        if force_relabel:
            # Re-label everything — counts every import.
            n_images = sum(
                1 for e in (manifest.get("imports") or []) if isinstance(e, dict)
            )
        else:
            # Mirror the runner's unlabelled filter (see
            # _run_label_charlie_job around line 3275). Counts
            # editedBoxesSet=True with empty editedBoxes as
            # unlabelled too — fixes a bug where "Clear all" on a
            # tile left labelled=True and the schedule endpoint
            # 400'd "no unlabelled images" even though the user had
            # just emptied one.
            def _is_unlabelled(e: dict) -> bool:
                if e.get("editedBoxesSet"):
                    edited = e.get("editedBoxes")
                    return not (isinstance(edited, list) and len(edited) > 0)
                return (
                    e.get("labelled") is False
                    or (e.get("labelled") is None and not (e.get("detections") or []))
                )
            n_images = sum(
                1
                for e in (manifest.get("imports") or [])
                if isinstance(e, dict) and _is_unlabelled(e)
            )
        if n_images == 0:
            raise HTTPException(
                400,
                "no images to process — drop some onto the dataset first"
                if force_relabel
                else "no unlabelled images to process",
            )

    job = state["jobs"].schedule(
        kind=payload.kind,
        project=payload.project,
        params=payload.params,
        user=payload.user or "anonymous",
        n_images=n_images,
    )
    return {"jobId": job.id, **job.to_public()}


@app.delete(
    "/api/v2/projects/{project_id}/jobs/{job_id}",
    dependencies=[Depends(require_project_owner)],
)
async def v2_cancel_project_job(project_id: str, job_id: str):
    """Owner-scoped job cancel. Used by the project page's job-card
    X button so a user can cancel their own labelling / augment /
    upload runs without holding the admin terminal token.

    Refuses to cancel a job that belongs to a different project,
    so a malicious owner-of-A can't cancel jobs-of-B by guessing
    a job id."""
    job = state["jobs"].jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    if job.project != project_id:
        raise HTTPException(404, "job not found")
    if not state["jobs"].cancel(job_id):
        raise HTTPException(409, "job already finished")
    return {"ok": True}


@app.get("/api/jobs/{job_id}/events")
async def job_events(request: Request, job_id: str, authorization: str = Header(default="")):
    job = state["jobs"].jobs.get(job_id)
    if not job:
        raise HTTPException(404)
    # Access control: this SSE streams detection/result events, so the job's
    # project must be readable by the caller. EventSource can't set an
    # Authorization header, so auth rides the pk_auth cookie (read inside
    # can_read_project_request). Private project + non-owner -> 404 (no oracle).
    _job_proj = (getattr(job, "project", "") or "")
    if _job_proj:
        _m = load_manifest(_job_proj, copy=False)
        if not can_read_project_request(request, authorization, _m):
            raise HTTPException(404)
    queue = await state["jobs"].subscribe(job_id)

    async def gen():
        try:
            while True:
                ev = await queue.get()
                yield ev
                if ev["event"] in ("done", "failed", "cancelled", "complete"):
                    break
        finally:
            state["jobs"].unsubscribe(job_id, queue)

    return EventSourceResponse(gen())


# Legacy endpoint kept as a thin shim so any old client still works.
@app.get(
    "/api/projects/{project_id}/jobs/active",
    dependencies=[Depends(require_project_read_access)],
)
async def project_active_job(project_id: str):
    """Return the running or queued job for this project (label/segment), or
    null. Lets the project view re-attach to a job that's still running after
    the user navigated away and back."""
    proj = project_dir(project_id)
    if not proj.exists():
        raise HTTPException(404, "project not found")
    candidates = [
        j for j in state["jobs"].jobs.values()
        if j.project == project_id
        and j.kind in ("label", "label_charlie", "purge_label", "segment")
        and j.status in ("running", "queued")
    ]
    if not candidates:
        return None
    # Prefer running over queued, then most recent.
    candidates.sort(key=lambda j: (0 if j.status == "running" else 1, j.queued_at), reverse=False)
    candidates.sort(key=lambda j: j.queued_at, reverse=True)
    candidates.sort(key=lambda j: 0 if j.status == "running" else 1)
    return candidates[0].to_public()


@app.get(
    "/api/projects/{project_id}/jobs/{job_id}",
    dependencies=[Depends(require_project_owner)],
)
async def get_project_job(project_id: str, job_id: str):
    """Per-project job status. Returns the same shape as
    /api/jobs/{job_id} but scoped so a project owner can poll for
    progress without needing the terminal token."""
    job = state["jobs"].jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    if job.project != project_id:
        raise HTTPException(403, "job does not belong to this project")
    return job.to_public()


@app.delete(
    "/api/projects/{project_id}/jobs/{job_id}",
    dependencies=[Depends(require_project_owner)],
)
async def cancel_project_job(project_id: str, job_id: str):
    """Cancel a running or queued job — scoped to this project so the action
    can't escape into another project's queue."""
    job = state["jobs"].jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    if job.project != project_id:
        raise HTTPException(403, "job does not belong to this project")
    ok = state["jobs"].cancel(job_id)
    return {"cancelled": ok, "id": job_id, "status": job.status}


@app.get(
    "/api/projects/{project_id}/originals/{filename}",
    dependencies=[Depends(require_project_read_access)],
)
async def project_original(project_id: str, filename: str):
    """302 to a short-lived R2 presigned URL — bytes flow browser ↔ R2,
    backend just serves the redirect header. URL is cached for ~50 min so
    repeat hits are CPU-free; the response carries Cache-Control so the
    browser stops asking entirely for 5 min and serves stale up to 55."""
    fn = Path(filename).name
    return _redirect_to_r2(R2Storage.image_key(project_id, fn))


@app.get(
    "/api/projects/{project_id}/files/{filename}",
    dependencies=[Depends(require_project_read_access)],
)
async def project_annotated(project_id: str, filename: str):
    fn = Path(filename).name
    return _redirect_to_r2(R2Storage.output_key(project_id, fn))


# ---- Dataset export -----------------------------------------------------
# One endpoint, format chosen via query string. Always returns a downloadable
# response with a sensible filename. Annotations are read from manifest.editedBoxes
# (the source of truth for labelled state — manual edits + auto detections);
# detections-only exports would miss the user's corrections.

def _safe_slug(s: str) -> str:
    """Aggressively strip an arbitrary string down to `[a-zA-Z0-9_-]+`.
    Used for filenames in Content-Disposition headers, ZIP entry
    names, and other places where a malicious project / label name
    could otherwise inject quotes, path separators, or header
    terminators. The regex replaces every non-ascii-safe character
    with `-`, so the output is guaranteed to be header-safe."""
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", (s or "export").strip()) or "export"


def _image_index(manifest: dict) -> dict[str, dict]:
    """Unified per-image view across V1 (results[] + top-level editedBoxes)
    and V2 (imports[] with per-entry width/height/editedBoxes).

    Returns {filename: {"width": W, "height": H, "boxes": [...]}}.
    For V2 imports we prefer `editedBoxes` when set, falling back to
    `detections` so projects that haven't had manual edits still
    export their auto-labels."""
    out: dict[str, dict] = {}

    # V1 path: results[] holds dimensions, editedBoxes (top-level) holds boxes.
    for r in (manifest.get("results") or []):
        img = r.get("image")
        sz = r.get("size") or {}
        if not img:
            continue
        out[img] = {
            "width": int(sz.get("width") or 0),
            "height": int(sz.get("height") or 0),
            "boxes": [],
        }
    for img_name, boxes in (manifest.get("editedBoxes") or {}).items():
        if not isinstance(boxes, list):
            continue
        slot = out.setdefault(img_name, {"width": 0, "height": 0, "boxes": []})
        slot["boxes"].extend(b for b in boxes if isinstance(b, dict))

    # V2 path: each import carries its own width/height/editedBoxes/detections.
    for entry in (manifest.get("imports") or []):
        if not isinstance(entry, dict):
            continue
        fname = entry.get("filename")
        if not fname:
            continue
        slot = out.setdefault(fname, {"width": 0, "height": 0, "boxes": []})
        # Dimensions: prefer the import's own values; only fill in if the
        # results-derived ones were missing.
        w = int(entry.get("width") or 0)
        h = int(entry.get("height") or 0)
        if w and not slot["width"]:
            slot["width"] = w
        if h and not slot["height"]:
            slot["height"] = h
        # Boxes: editedBoxes wins when explicitly set (even if empty —
        # editedBoxesSet=True means "user blanked it deliberately"),
        # otherwise fall back to auto detections. The detections path
        # drops anything the resolver flagged `rejected` so confidence-
        # gate / containment / overlap rejects don't sneak back into
        # the export (matches every other consumer of /detections).
        edited = entry.get("editedBoxes")
        if isinstance(edited, list) and (entry.get("editedBoxesSet") or edited):
            slot["boxes"].extend(b for b in edited if isinstance(b, dict))
        else:
            dets = entry.get("detections") or []
            if isinstance(dets, list):
                slot["boxes"].extend(
                    b for b in dets
                    if isinstance(b, dict) and not b.get("rejected")
                )

    return out


def _project_box_iter(manifest: dict):
    """Yield (image_filename, (W, H), box_dict) for every labelled box,
    unified across V1 and V2 manifest shapes."""
    idx = _image_index(manifest)
    for img_name, info in idx.items():
        sz = (info["width"], info["height"])
        for b in info["boxes"]:
            yield img_name, sz, b


def _image_names_list(manifest: dict) -> list[str]:
    """Every original image filename in the project, V1 + V2 combined."""
    return list(_image_index(manifest).keys())


# Box size classification — matches the frontend exactly (BOX_FAIL_PX=12,
# BOX_WARN_PX=24). Letterbox scaling: same factor on both axes, so the
# box's worst-side bottleneck is preserved.
_BOX_FAIL_PX = 12
_BOX_WARN_PX = 24


def _parse_input_shape(s: str) -> tuple[int, int]:
    try:
        w, h = (s or "").lower().split("x")
        wi, hi = int(w), int(h)
        if wi <= 0 or hi <= 0:
            raise ValueError
        return wi, hi
    except Exception:
        return 256, 256


def _scaled_min_side(x0: float, y0: float, x1: float, y1: float,
                     img_w: int, img_h: int,
                     in_w: int, in_h: int) -> float:
    if not img_w or not img_h:
        return float("inf")
    s = min(in_w / img_w, in_h / img_h)
    return min(abs(x1 - x0), abs(y1 - y0)) * s


def _box_status(min_side: float) -> str:
    if min_side < _BOX_FAIL_PX:
        return "fail"
    if min_side < _BOX_WARN_PX:
        return "warn"
    return "ok"


def _keep_by_size(x0: float, y0: float, x1: float, y1: float,
                  img_w: int, img_h: int,
                  in_w: int, in_h: int,
                  exclude_red: bool, exclude_orange: bool) -> bool:
    """Apply the user-selected red/orange exclusions to a single box.
    Mirrors the FE's status colouring under the same letterbox scaling."""
    status = _box_status(_scaled_min_side(x0, y0, x1, y1, img_w, img_h, in_w, in_h))
    if status == "fail" and exclude_red:
        return False
    if status == "warn" and exclude_orange:
        return False
    return True


def _iter_augmentations(project_id: str, manifest: dict):
    """Yield (export_filename, parent_filename, canvas_w, canvas_h, box_dict)
    for every augmented copy that exists on disk + has annotations.

    Export filename pattern is `aug_<orig_stem>_<k>.jpg` so the trained
    model sees augmentation lineage at a glance and the user can tell
    augs apart from originals in the bundle.

    Box dicts are adapted from the per-copy annotation shape (label +
    polys + box) into the same x0/y0/x1/y1 + label + mask.polygons
    shape the rest of the export pipeline expects, so the existing
    builders need no aug-specific branches."""
    if not manifest:
        return
    aug_root = _augmentations_dir(project_id)
    if not aug_root.exists():
        return
    for entry in (manifest.get("imports") or []):
        import_id = entry.get("id")
        parent = entry.get("filename") or ""
        n_aug = int(entry.get("n_augmentations") or 0)
        if not import_id or not parent or n_aug <= 0:
            continue
        anno_path = aug_root / import_id / "annotations.json"
        if not anno_path.exists():
            continue
        try:
            data = json.loads(anno_path.read_text())
        except Exception:
            continue
        canvas_w = int(data.get("width") or 0)
        canvas_h = int(data.get("height") or 0)
        copies = data.get("copies") or {}
        parent_stem = Path(parent).stem
        for copy_name, dets in copies.items():
            if not isinstance(dets, list):
                continue
            # Strip the `.jpg` and rebuild as `aug_<orig_stem>_<k>.jpg`
            # — the `k` index is the leading zero-padded number the
            # generator already chose.
            k = Path(copy_name).stem
            export_name = f"aug_{parent_stem}_{k}.jpg"
            # Confirm the JPEG exists on disk before yielding annotations
            # that reference it — drops half-written copies.
            jpeg_path = aug_root / import_id / copy_name
            if not jpeg_path.exists():
                continue
            for det in dets:
                if not isinstance(det, dict):
                    continue
                box = det.get("box") or []
                if not (isinstance(box, (list, tuple)) and len(box) == 4):
                    continue
                adapted = {
                    "x0": float(box[0]),
                    "y0": float(box[1]),
                    "x1": float(box[2]),
                    "y1": float(box[3]),
                    "label": det.get("label") or "",
                }
                polys = det.get("polys") or []
                if polys:
                    adapted["mask"] = {"polygons": polys}
                yield export_name, parent, canvas_w, canvas_h, adapted, import_id, copy_name


def _aug_files_to_bundle(project_id: str, manifest: dict) -> list[tuple[str, Path]]:
    """Return [(export_name, source_path)] for every augmented JPEG that has
    surviving annotations. Used by _add_images_to_zip when augmentations
    are being shipped alongside originals."""
    if not manifest:
        return []
    aug_root = _augmentations_dir(project_id)
    if not aug_root.exists():
        return []
    out: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for entry in (manifest.get("imports") or []):
        import_id = entry.get("id")
        parent = entry.get("filename") or ""
        n_aug = int(entry.get("n_augmentations") or 0)
        if not import_id or not parent or n_aug <= 0:
            continue
        anno_path = aug_root / import_id / "annotations.json"
        if not anno_path.exists():
            continue
        try:
            data = json.loads(anno_path.read_text())
        except Exception:
            continue
        parent_stem = Path(parent).stem
        for copy_name in (data.get("copies") or {}).keys():
            jpeg_path = aug_root / import_id / copy_name
            if not jpeg_path.exists():
                continue
            k = Path(copy_name).stem
            export_name = f"aug_{parent_stem}_{k}.jpg"
            if export_name in seen:
                continue
            seen.add(export_name)
            out.append((export_name, jpeg_path))
    return out


def _build_readme(manifest: dict) -> str:
    """README written into every export zip. Names the project, who made
    it, when it was exported, and where PixelKit lives. Kept plain-text
    so it sits cleanly next to YOLO classes.txt, COCO annotations.json,
    or VOC ImageSets without surprising any tooling."""
    name = manifest.get("name") or "Untitled project"
    owner = manifest.get("owner") or manifest.get("createdBy") or "unknown"
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return (
        "PixelKit dataset export\n"
        "=======================\n"
        "\n"
        f"Project: {name}\n"
        f"Created by: {owner}\n"
        f"Exported at: {now_iso} (UTC)\n"
        "\n"
        "Built with PixelKit — https://pixelkit.ai\n"
    )


def _box_xyxy(b: dict) -> tuple[float, float, float, float] | None:
    """Extract (x0, y0, x1, y1) from whichever shape the box dict uses.

    Three flavours need to be supported:
      • V2 editedBoxes / aug detections: flat x0/y0/x1/y1 floats.
      • V2 auto detections: a `box` list of four floats. This is the
        canonical resolver output and was previously missed by export,
        which is why YOLO/COCO came back with zero labels on any V2
        import the user hadn't manually edited.
      • Legacy: `box_xyxy` list of four floats.
    """
    if all(k in b for k in ("x0", "y0", "x1", "y1")):
        x0, y0, x1, y1 = float(b["x0"]), float(b["y0"]), float(b["x1"]), float(b["y1"])
    elif isinstance(b.get("box_xyxy"), (list, tuple)) and len(b["box_xyxy"]) == 4:
        x0, y0, x1, y1 = (float(v) for v in b["box_xyxy"])
    elif isinstance(b.get("box"), (list, tuple)) and len(b["box"]) == 4:
        x0, y0, x1, y1 = (float(v) for v in b["box"])
    else:
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _box_label(b: dict) -> str:
    """Resolve a box's label across the naming conventions in play.

    Fallback order, matching what the FE viewer does
    (predLabel || gdLabel):
      • `label` — set on editedBoxes and augmentation entries.
      • `predLabel` / `pred_label` — set on V2 auto detections after
        the specific-dataset embedding resolver runs.
      • `gd_label` / `gdLabel` / `gd_variant` — set on V2 auto
        detections that *don't* go through the resolver (general
        datasets skip it entirely and trust SAM 3's text-prompt
        assignment). Missing this fallback was why general-dataset
        YOLO/COCO exports came back with zero label files even
        after the box-shape fix landed — every detection had a box,
        but `_box_label` resolved to "" so the category lookup
        silently skipped every one."""
    for key in (
        "label",
        "predLabel", "pred_label",
        "gd_label", "gdLabel", "gd_variant",
    ):
        v = b.get(key)
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return ""


def _categories(manifest: dict) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    # Project tags first so the index is stable across re-exports.
    for t in (manifest.get("tags") or []):
        tl = (t or "").strip().lower()
        if tl and tl not in seen:
            seen.add(tl)
            out.append(tl)
    # Then any extra labels from boxes that aren't in the tag list.
    for _img, _sz, b in _project_box_iter(manifest):
        lbl = _box_label(b).lower()
        if lbl and lbl not in seen:
            seen.add(lbl)
            out.append(lbl)
    return out


def _load_image_bytes(project_id: str, img_name: str) -> bytes | None:
    """Resolve an original image's bytes. V2 projects keep imports on the
    local filesystem; V1 projects keep them in R2. Try the cheap local
    path first and fall back to R2 — silently returns None if neither
    has the file so the caller can skip the entry."""
    local_path = project_dir(project_id) / "images" / img_name
    if local_path.exists():
        try:
            return local_path.read_bytes()
        except Exception as e:
            print(f"[export] local read failed for {img_name}: {e}")
    try:
        return r2_required().get_bytes(R2Storage.image_key(project_id, img_name))
    except Exception as e:
        print(f"[export] r2 read failed for {img_name}: {e}")
        return None


def _zip_image_entry(zf, arcname: str, data: bytes) -> None:
    zf.writestr(arcname, data)


def _split_for(parent_name: str, train_split: float) -> str:
    """Stable train/val assignment for an image. Augmentations pass the
    parent image's filename so every copy of one source lands in the
    same split — avoids val leakage from augmentations of a train image.

    train_split is the fraction that goes to train; the rest goes to val.
    sha1(filename)/0xFFFFFFFF gives a uniform [0,1) bucket independent
    of file order, so re-runs and re-exports place each file the same way."""
    if train_split >= 0.999:
        return "train"
    if train_split <= 0.001:
        return "val"
    digest = hashlib.sha1((parent_name or "").encode("utf-8")).hexdigest()[:8]
    bucket = int(digest, 16) / 0xFFFFFFFF
    return "train" if bucket < train_split else "val"


def _build_coco(
    manifest: dict,
    *,
    project_id: str,
    include_boxes: bool,
    include_segmentations: bool,
    exclude_red: bool,
    exclude_orange: bool,
    input_shape: str,
    with_augmentations: bool,
    train_split: float,
) -> dict[str, dict]:
    """Build per-split COCO dicts. Returns {"train": {...}, "val": {...}}
    (val key omitted when empty so we don't write a stub file). Each
    image's split is decided by _split_for() on its parent filename so
    augmentations follow their source image."""
    cats = _categories(manifest)
    cat_index = {c: i + 1 for i, c in enumerate(cats)}  # COCO category ids are 1-indexed
    in_w, in_h = _parse_input_shape(input_shape)

    splits = ("train", "val")
    images_by_split: dict[str, list[dict]] = {s: [] for s in splits}
    anns_by_split: dict[str, list[dict]] = {s: [] for s in splits}
    id_by_file: dict[str, tuple[str, int]] = {}  # file_name -> (split, image_id)
    next_ann_id: dict[str, int] = {s: 1 for s in splits}

    def _register_image(split: str, file_name: str, w: int, h: int) -> int:
        new_id = len(images_by_split[split]) + 1
        images_by_split[split].append({
            "id": new_id,
            "file_name": file_name,
            "width": int(w),
            "height": int(h),
        })
        id_by_file[file_name] = (split, new_id)
        return new_id

    def _append_ann(file_name: str, img_w: int, img_h: int, b: dict) -> None:
        slot = id_by_file.get(file_name)
        if slot is None:
            return
        split, image_id = slot
        rect = _box_xyxy(b)
        if rect is None:
            return
        x0, y0, x1, y1 = rect
        if not _keep_by_size(x0, y0, x1, y1, img_w, img_h, in_w, in_h,
                             exclude_red, exclude_orange):
            return
        label = _box_label(b).lower()
        cat_id = cat_index.get(label)
        if not cat_id:
            return
        polys = (b.get("mask") or {}).get("polygons") or []
        seg: list[list[float]] = []
        if polys:
            for poly in polys:
                flat: list[float] = []
                for pt in poly:
                    if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                        flat.extend([float(pt[0]), float(pt[1])])
                if flat:
                    seg.append([round(v, 1) for v in flat])
        if not include_boxes and not include_segmentations:
            return
        if include_segmentations and not include_boxes and not seg:
            return
        w, h = x1 - x0, y1 - y0
        ann: dict = {
            "id": next_ann_id[split],
            "image_id": image_id,
            "category_id": cat_id,
            "area": round(w * h, 2),
            "iscrowd": 0,
        }
        if include_boxes:
            ann["bbox"] = [round(x0, 2), round(y0, 2), round(w, 2), round(h, 2)]
        if include_segmentations and seg:
            ann["segmentation"] = seg
        anns_by_split[split].append(ann)
        next_ann_id[split] += 1

    # Register originals first so the image_id-by-name lookup is
    # populated before annotations are walked. Pulls the unified V1+V2
    # view so V2 projects (imports[]-based) and V1 projects
    # (results[]-based) both produce a populated images[] section.
    img_idx = _image_index(manifest)
    for img_name, info in img_idx.items():
        w, h = info["width"], info["height"]
        if not w or not h:
            continue
        _register_image(_split_for(img_name, train_split), img_name, w, h)

    # Register augmentations: split by *parent* filename so all copies of
    # one source land together.
    if with_augmentations:
        seen_aug: set[str] = set()
        for export_name, parent, cw, ch, _det, _iid, _cn in _iter_augmentations(project_id, manifest):
            if export_name in seen_aug:
                continue
            seen_aug.add(export_name)
            _register_image(_split_for(parent, train_split), export_name, int(cw), int(ch))

    # Now walk annotations and bucket them by their image's pre-decided split.
    for img_name, sz, b in _project_box_iter(manifest):
        W, H = sz
        if not W or not H:
            continue
        _append_ann(img_name, W, H, b)

    if with_augmentations:
        for export_name, _parent, cw, ch, det, _iid, _cn in _iter_augmentations(project_id, manifest):
            _append_ann(export_name, int(cw), int(ch), det)

    info = {
        "description": manifest.get("name") or "PixelKit export",
        "version": "1.0",
        "exporter": "pixelkit.ai",
    }
    categories = [
        {"id": i, "name": name, "supercategory": "object"}
        for name, i in cat_index.items()
    ]
    out: dict[str, dict] = {}
    for s in splits:
        if not images_by_split[s] and s == "val":
            continue
        out[s] = {
            "info": info,
            "licenses": [],
            "images": images_by_split[s],
            "annotations": anns_by_split[s],
            "categories": categories,
        }
    return out


def _build_yolo(
    manifest: dict,
    *,
    project_id: str,
    include_boxes: bool,
    include_segmentations: bool,
    exclude_red: bool,
    exclude_orange: bool,
    input_shape: str,
    with_augmentations: bool,
    train_split: float,
) -> tuple[dict[tuple[str, str], str], list[str]]:
    """Build YOLO .txt files keyed by (split, txt_name).

    When include_segmentations is true and a box carries polygon data
    the line follows the Ultralytics YOLOv8-seg convention:
        <class_id> x1 y1 x2 y2 x3 y3 ...   (normalised 0..1)
    Detection-only lines are the usual:
        <class_id> cx cy w h               (normalised 0..1)

    Augmentations inherit their parent image's split so val never
    contains augmented copies of train images."""
    cats = _categories(manifest)
    cat_index = {c: i for i, c in enumerate(cats)}  # YOLO is 0-indexed
    by_image: dict[tuple[str, str], list[str]] = {}
    in_w, in_h = _parse_input_shape(input_shape)

    def _emit(split: str, txt_name: str, img_w: int, img_h: int, b: dict) -> None:
        if not img_w or not img_h:
            return
        rect = _box_xyxy(b)
        if rect is None:
            return
        x0, y0, x1, y1 = rect
        if not _keep_by_size(x0, y0, x1, y1, img_w, img_h, in_w, in_h,
                             exclude_red, exclude_orange):
            return
        label = _box_label(b).lower()
        cls = cat_index.get(label)
        if cls is None:
            return
        polys = (b.get("mask") or {}).get("polygons") or []
        key = (split, txt_name)
        emitted = False
        if include_segmentations and polys:
            for poly in polys:
                flat: list[str] = []
                for pt in poly:
                    if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                        flat.append(f"{float(pt[0]) / img_w:.6f}")
                        flat.append(f"{float(pt[1]) / img_h:.6f}")
                if len(flat) >= 6:  # need at least three (x,y) pairs
                    by_image.setdefault(key, []).append(f"{cls} " + " ".join(flat))
                    emitted = True
        if include_boxes and not emitted:
            cx = (x0 + x1) / 2 / img_w
            cy = (y0 + y1) / 2 / img_h
            bw = (x1 - x0) / img_w
            bh = (y1 - y0) / img_h
            by_image.setdefault(key, []).append(
                f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
            )

    # Originals
    for img_name, sz, b in _project_box_iter(manifest):
        W, H = sz
        split = _split_for(img_name, train_split)
        _emit(split, Path(img_name).stem + ".txt", W, H, b)

    # Augmentations follow their parent's split so the val set never sees
    # an augmented copy of a train image.
    if with_augmentations:
        for export_name, parent, cw, ch, det, _iid, _cn in _iter_augmentations(project_id, manifest):
            split = _split_for(parent, train_split)
            _emit(split, Path(export_name).stem + ".txt", int(cw), int(ch), det)

    return ({k: "\n".join(v) + "\n" for k, v in by_image.items()}, cats)


def _build_voc_xml(
    img_name: str,
    W: int,
    H: int,
    boxes: list[dict],
    *,
    exclude_red: bool,
    exclude_orange: bool,
    input_shape: str,
) -> str:
    """Pascal VOC 2012-style annotation XML. VOC is bbox-only (no polygon
    spec), so segmentations are silently ignored here — the include_seg
    toggle on the FE doesn't change VOC output."""
    in_w, in_h = _parse_input_shape(input_shape)
    obj_blocks = []
    for b in boxes:
        rect = _box_xyxy(b)
        if rect is None:
            continue
        x0, y0, x1, y1 = rect
        if not _keep_by_size(x0, y0, x1, y1, W, H, in_w, in_h,
                             exclude_red, exclude_orange):
            continue
        label = _box_label(b).lower()
        if not label:
            continue
        obj_blocks.append(
            "  <object>\n"
            f"    <name>{label}</name>\n"
            "    <pose>Unspecified</pose>\n"
            "    <truncated>0</truncated>\n"
            "    <difficult>0</difficult>\n"
            "    <bndbox>\n"
            f"      <xmin>{int(round(x0))}</xmin>\n"
            f"      <ymin>{int(round(y0))}</ymin>\n"
            f"      <xmax>{int(round(x1))}</xmax>\n"
            f"      <ymax>{int(round(y1))}</ymax>\n"
            "    </bndbox>\n"
            "  </object>"
        )
    return (
        "<annotation>\n"
        f"  <folder>images</folder>\n"
        f"  <filename>{img_name}</filename>\n"
        "  <size>\n"
        f"    <width>{W}</width>\n"
        f"    <height>{H}</height>\n"
        "    <depth>3</depth>\n"
        "  </size>\n"
        "  <segmented>0</segmented>\n"
        + ("\n".join(obj_blocks) + "\n" if obj_blocks else "")
        + "</annotation>\n"
    )


@app.get(
    "/api/projects/{project_id}/export",
    dependencies=[Depends(require_project_read_access)],
)
async def export_project(
    project_id: str,
    format: str,
    include_images: bool = True,
    include_boxes: bool = True,
    include_segmentations: bool = True,
    exclude_red: bool = True,
    exclude_orange: bool = False,
    input_shape: str = "256x256",
    train_split: float = 0.8,
):
    """Build a downloadable export of the project's labels.

    Supported `format` values: `yolo`, `coco`, `voc`. Everything else 400s.

    Box content is gated by `include_boxes` / `include_segmentations`. At
    least one must be true. VOC is bbox-only so an `include_boxes=false`
    VOC request 400s.

    Size-class exclusions match the FE's letterbox-scaled minimum-side
    classifier (< 12 px = red, 12-24 px = orange) computed against
    `input_shape` (e.g. `320x320`). Red is dropped by default; orange
    is kept by default.

    `train_split` (0.0..1.0) controls the train/val split. Per-file
    decision is deterministic (sha1(filename)) so every re-export of
    the same project places each image in the same set. Augmentations
    follow their parent image's split.

    Layout per format (training-ready):
      - YOLO:  images/{train,val}/*.jpg + labels/{train,val}/*.txt + data.yaml
      - COCO:  annotations/instances_{train,val}.json + images/{train,val}/*.jpg
      - VOC:   JPEGImages/*.jpg + Annotations/*.xml + ImageSets/Main/{train,val}.txt

    Augmentations on disk are bundled when `include_images=true`,
    renamed `aug_<orig_stem>_<k>.jpg`. Every archive carries a README.txt
    with project name, owner and export timestamp."""
    if not project_dir(project_id).exists():
        raise HTTPException(404, "project not found")
    manifest = load_manifest(project_id)
    if not manifest:
        raise HTTPException(404, "no manifest")

    if format not in ("yolo", "coco", "voc"):
        raise HTTPException(400, f"unsupported format: {format}")
    if not include_boxes and not include_segmentations:
        raise HTTPException(400, "at least one of include_boxes / include_segmentations must be true")
    if format == "voc" and not include_boxes:
        raise HTTPException(400, "Pascal VOC only supports bounding boxes; include_boxes must be true")
    train_split = max(0.0, min(1.0, float(train_split)))

    base = _safe_slug(manifest.get("name") or project_id)
    image_names = _image_names_list(manifest)
    aug_files = _aug_files_to_bundle(project_id, manifest) if include_images else []
    aug_parent_by_export = _aug_parent_lookup(project_id, manifest) if include_images else {}
    with_augs = bool(aug_files)
    readme = _build_readme(manifest)

    def _write_images(zf, *, train_subdir: str, val_subdir: str) -> None:
        # Originals — try local disk first (V2 layout), then R2 (V1). Skip
        # silently if neither has the bytes.
        for img_name in image_names:
            data = _load_image_bytes(project_id, img_name)
            if data is None:
                continue
            split = _split_for(img_name, train_split)
            sub = train_subdir if split == "train" else val_subdir
            zf.writestr(f"{sub}/{img_name}" if sub else img_name, data)
        # Augmentations from local disk under projects/<id>/augmentations/.
        for export_name, src in aug_files:
            try:
                parent = aug_parent_by_export.get(export_name, export_name)
                split = _split_for(parent, train_split)
                sub = train_subdir if split == "train" else val_subdir
                zf.writestr(f"{sub}/{export_name}" if sub else export_name, src.read_bytes())
            except Exception as e:
                print(f"[export] skipped aug {export_name}: {e}")

    if format == "coco":
        coco_by_split = _build_coco(
            manifest,
            project_id=project_id,
            include_boxes=include_boxes,
            include_segmentations=include_segmentations,
            exclude_red=exclude_red,
            exclude_orange=exclude_orange,
            input_shape=input_shape,
            with_augmentations=with_augs,
            train_split=train_split,
        )
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            zf.writestr("README.txt", readme)
            for split, coco in coco_by_split.items():
                zf.writestr(f"annotations/instances_{split}.json", json.dumps(coco, indent=2))
            if include_images:
                _write_images(zf, train_subdir="images/train", val_subdir="images/val")
        return Response(
            buf.getvalue(),
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{base}-coco.zip"'},
        )

    if format == "yolo":
        files, cats = _build_yolo(
            manifest,
            project_id=project_id,
            include_boxes=include_boxes,
            include_segmentations=include_segmentations,
            exclude_red=exclude_red,
            exclude_orange=exclude_orange,
            input_shape=input_shape,
            with_augmentations=with_augs,
            train_split=train_split,
        )
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            zf.writestr("README.txt", readme)
            zf.writestr("classes.txt", "\n".join(cats) + "\n")
            # data.yaml in the Ultralytics shape — `path` is the dataset
            # root (the zip's contents once extracted), train/val point
            # at the two image subdirs. The YOLO loader looks for a
            # parallel labels/ tree by replacing `images/` with `labels/`
            # in each image path, which matches the layout below.
            zf.writestr("data.yaml",
                f"path: .\n"
                f"train: images/train\n"
                f"val: images/val\n"
                f"nc: {len(cats)}\n"
                f"names: [{', '.join(repr(c) for c in cats)}]\n",
            )
            for (split, fname), content in files.items():
                zf.writestr(f"labels/{split}/{fname}", content)
            if include_images:
                _write_images(zf, train_subdir="images/train", val_subdir="images/val")
        return Response(
            buf.getvalue(),
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{base}-yolo.zip"'},
        )

    # format == "voc". VOC uses ImageSets/Main/{train,val}.txt to record
    # the split; all JPEGs sit in JPEGImages/ regardless. Annotation
    # XMLs sit in Annotations/. Uses the V1+V2 unified image index so
    # imports[]-only V2 projects emit XMLs (older code only walked
    # results[]/editedBoxes which V2 doesn't populate at the top level).
    img_idx = _image_index(manifest)
    train_stems: list[str] = []
    val_stems: list[str] = []
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.writestr("README.txt", readme)
        for img_name, info in img_idx.items():
            W, H = info["width"], info["height"]
            if not W or not H:
                continue
            xml = _build_voc_xml(
                img_name, W, H, info["boxes"],
                exclude_red=exclude_red,
                exclude_orange=exclude_orange,
                input_shape=input_shape,
            )
            stem = Path(img_name).stem
            zf.writestr(f"Annotations/{stem}.xml", xml)
            (train_stems if _split_for(img_name, train_split) == "train" else val_stems).append(stem)
        if with_augs:
            grouped: dict[str, tuple[int, int, str, list[dict]]] = {}
            for export_name, parent, cw, ch, det, _iid, _cn in _iter_augmentations(project_id, manifest):
                slot = grouped.setdefault(export_name, (int(cw), int(ch), parent, []))
                slot[3].append(det)
            for export_name, (cw, ch, parent, dets) in grouped.items():
                xml = _build_voc_xml(
                    export_name, cw, ch, dets,
                    exclude_red=exclude_red,
                    exclude_orange=exclude_orange,
                    input_shape=input_shape,
                )
                stem = Path(export_name).stem
                zf.writestr(f"Annotations/{stem}.xml", xml)
                (train_stems if _split_for(parent, train_split) == "train" else val_stems).append(stem)
        zf.writestr("ImageSets/Main/train.txt", "\n".join(train_stems) + ("\n" if train_stems else ""))
        zf.writestr("ImageSets/Main/val.txt", "\n".join(val_stems) + ("\n" if val_stems else ""))
        if include_images:
            # VOC puts every JPEG in a flat JPEGImages/ regardless of split.
            _write_images(zf, train_subdir="JPEGImages", val_subdir="JPEGImages")
    return Response(
        buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{base}-voc.zip"'},
    )


def _aug_parent_lookup(project_id: str, manifest: dict) -> dict[str, str]:
    """Map each export-name aug back to its parent image filename so the
    train/val split decider can route it consistently with the rest of
    its lineage. Built once per request to avoid re-walking the
    augmentations dir during image bundling."""
    if not manifest:
        return {}
    out: dict[str, str] = {}
    aug_root = _augmentations_dir(project_id)
    if not aug_root.exists():
        return out
    for entry in (manifest.get("imports") or []):
        import_id = entry.get("id")
        parent = entry.get("filename") or ""
        if not import_id or not parent:
            continue
        anno_path = aug_root / import_id / "annotations.json"
        if not anno_path.exists():
            continue
        try:
            data = json.loads(anno_path.read_text())
        except Exception:
            continue
        parent_stem = Path(parent).stem
        for copy_name in (data.get("copies") or {}).keys():
            k = Path(copy_name).stem
            out[f"aug_{parent_stem}_{k}.jpg"] = parent
    return out


# ───────────────────────────────────────────────────────────────────────
# Pipeline Charlie endpoints — drop-in shape compatible with the V2
# /api/v2/imports/* routes so the FE can target the new pipeline by
# changing the URL prefix only. Charlie is SAM3-only for now: no
# GroundingDINO, no SAM2, no VLM, no DINOv2/SigLIP. Reference flows
# (/api/v2/references/*) intentionally have no Charlie counterpart —
# Charlie's resolver pipeline isn't built yet.
# ───────────────────────────────────────────────────────────────────────


@app.post("/api/charlie/imports/process")
async def charlie_imports_process(
    image: UploadFile = File(...),
    labels: str = Form(...),
    project_id: str = Form(""),
):
    """SAM3 promptable concept segmentation on an uploaded image.

    Inputs match V2's /api/v2/imports/process:
      - image:   the file to process
      - labels:  JSON array of label strings
      - project_id: accepted for shape parity, currently unused (no
                    reference resolver in Charlie yet)

    Response shape mirrors V2's /api/v2/imports/process so the FE can
    iterate `detections` without restructuring. Per-detection fields
    that V2 fills via DINOv2/SigLIP/VLM are returned empty/null:
        embedding=[], embedding_siglip=[], vlm_label=null, vlm_score=null

    Debug info for the pipeline popup goes in `timings` and `pipeline`.
    """
    charlie = state.get("charlie")
    if charlie is None:
        raise HTTPException(503, "pipeline_charlie not loaded (SAM3 unavailable)")

    try:
        tag_list = json.loads(labels)
        if not isinstance(tag_list, list) or not all(isinstance(t, str) for t in tag_list):
            raise ValueError("labels must be a JSON array of strings")
    except Exception as e:
        raise HTTPException(400, f"invalid labels payload: {e}")
    tags = [t.strip() for t in tag_list if t and t.strip()]

    raw = await image.read()
    if not raw:
        raise HTTPException(400, "empty image upload")
    try:
        image_pil = PILImage.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"could not decode image: {e}")
    W, H = image_pil.size

    if not tags:
        return {
            "pipeline": "charlie",
            "width": W, "height": H,
            "prompt_tags": [],
            "timings": {"sam3_predict_ms": 0.0, "sam3_post_ms": 0.0,
                         "encode_crops_ms": 0.0, "total_ms": 0.0},
            "detections": [],
        }

    loop = asyncio.get_running_loop()

    def _infer():
        return charlie.segment_labels(image_pil, tags, include_crops=True)

    try:
        async with state["gpu_lock"]:
            detections, timings = await loop.run_in_executor(None, _infer)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"charlie pipeline error: {exc}")
    finally:
        import gc as _gc
        _gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(
        f"[charlie] /imports/process W={W} H={H} prompts={tags} "
        f"hits={len(detections)} sam3={timings['sam3_predict_ms']:.0f}ms "
        f"post={timings['sam3_post_ms']:.0f}ms total={timings['total_ms']:.0f}ms"
    )

    return {
        "pipeline": "charlie",
        "width": W, "height": H,
        "prompt_tags": tags,
        "timings": timings,
        "detections": detections,
    }


async def _charlie_load_image_pil(
    image: UploadFile | None,
    project_id: str,
    import_id: str,
    *,
    endpoint_tag: str,
) -> "PILImage.Image":
    """Resolve the source image bytes for a Charlie interactive
    endpoint, preferring the on-disk copy when the caller supplied
    `project_id` + `import_id`. Avoids round-tripping the image
    through the browser (and the CORS-blocked fetch that was making
    click-to-detect unreachable from www.pixelkit.ai).

    Raises HTTPException(400) on bad input and HTTPException(404)
    when the requested import doesn't exist on disk.
    """
    # By-id path: load from disk. No browser fetch involved.
    if project_id and import_id:
        proj = project_dir(project_id)
        if not proj.exists():
            raise HTTPException(404, "project not found")
        # copy=False — we only read filename + scan for one entry.
        # Deepcopying the full 30MB manifest just to find one ID was
        # the single biggest per-call hit on big projects (200-500ms
        # PER click-to-detect / add-box / segment_box / classify_box
        # call). The manifest cache stays correct because we never
        # mutate.
        manifest = load_manifest(project_id, copy=False) or {}
        imp = next(
            (m for m in (manifest.get("imports") or []) if m.get("id") == import_id),
            None,
        )
        if not imp:
            raise HTTPException(404, "import not found")
        fn = imp.get("filename")
        if not fn:
            raise HTTPException(404, "import has no stored filename")
        src_path = proj / "images" / fn
        if not src_path.exists():
            raise HTTPException(404, "source image missing on disk")
        try:
            with PILImage.open(src_path) as im:
                return ImageOps.exif_transpose(im).convert("RGB")
        except Exception as e:
            print(f"[{endpoint_tag}] 400 disk decode failed for {src_path}: {e}")
            raise HTTPException(400, f"could not decode stored image: {e}")

    # Upload path: legacy / no-project flows.
    if image is None:
        raise HTTPException(
            400,
            "either (project_id + import_id) or an image file must be provided",
        )
    raw = await image.read()
    if not raw:
        print(
            f"[{endpoint_tag}] 400 empty upload — filename={image.filename!r} "
            f"content_type={image.content_type!r}"
        )
        raise HTTPException(400, "empty image upload")
    try:
        with PILImage.open(io.BytesIO(raw)) as im:
            return ImageOps.exif_transpose(im).convert("RGB")
    except Exception as e:
        print(
            f"[{endpoint_tag}] 400 decode failed — "
            f"filename={image.filename!r} size_bytes={len(raw)} "
            f"head_hex={raw[:16].hex()} err={type(e).__name__}: {e}"
        )
        raise HTTPException(400, f"could not decode image: {e}")


async def _detect_point_unified(
    image_pil,
    point_xy,
    *,
    candidate_labels,
    project_id,
    dataset_type_hint=None,
    allow_sam3=True,
    allow_reject=False,
    is_labelled=False,
):
    """One click-to-detect implementation shared by the charlie, demo, and
    (optionally) v2 endpoints, so every surface behaves identically.

    Flow:
      STAGE 1  SAM3 text-concept first try, only on unlabelled + SAM3-loaded +
               candidate-bearing tiles. On a hit it returns SAM3's concept label.
      STAGE 2  SAM2 point segmentation as the universal floor: it returns a mask
               under any valid click, so a click never dead-ends. The ONLY
               legitimate 422 is SAM2 finding no mask (a true-background click).
      STAGE 3  Label the SAM2 mask. Projects with reference images use the
               embedding resolver (specific -> knn, general -> centroid); other
               cases use the VLM to pick from the candidate labels. The VLM is
               ALWAYS given the mask so the background is suppressed on the crop.
      STAGE 4  Terminal, never 422: a reject (labelled tiles only), else a
               best-guess label (first candidate), else label=None.

    Returns the charlie-shaped envelope.
    """
    # Portable build: SAM2 is gone — SAM3 is both the concept detector and
    # the interactive floor. 503 only when nothing is loaded at all.
    if state.get("charlie") is None and state.get("segmenter") is None:
        raise HTTPException(503, "segmentation model not loaded")
    loop = asyncio.get_running_loop()
    W, H = image_pil.size
    px = min(max(float(point_xy[0]), 0.0), float(W - 1))
    py = min(max(float(point_xy[1]), 0.0), float(H - 1))
    labels = [str(t).strip() for t in (candidate_labels or []) if str(t).strip()]

    # STAGE 1 — SAM3 text-concept first try (eligible tiles only).
    if allow_sam3 and labels and state.get("charlie") is not None and not is_labelled:
        charlie = state["charlie"]

        def _run_sam3():
            return charlie.segment_point(image_pil, [px, py], labels)

        try:
            async with state["gpu_lock"].interactive():
                detection, sam3_timings = await loop.run_in_executor(None, _run_sam3)
        except Exception as exc:  # noqa: BLE001 — any SAM3 failure falls back to SAM2
            print(f"[detect_point/sam3] error, falling back to SAM2: {exc}")
            detection, sam3_timings = None, {}
        if detection is not None:
            box_xyxy = [float(c) for c in detection["box"]]
            mask_polys = (
                detection["mask"]["polygons"]
                if isinstance(detection.get("mask"), dict) else []
            )
            label = detection.get("gd_label") or None
            score = float(detection["gd_score"]) if detection.get("gd_score") is not None else None
            return {
                "pipeline": "charlie", "width": W, "height": H,
                "timings": {**sam3_timings, "route": "sam3"},
                "route": "sam3",
                "box_xyxy": box_xyxy,
                "mask": {"polygons": mask_polys},
                "mask_score": score,
                "label": label, "score": score,
                "rejected": False, "reject_reason": None,
            }

    # STAGE 2 — SAM2 floor removed in the portable build. When SAM3 found
    # nothing above, there is no universal-segmenter fallback: report a
    # clean miss instead of NameError'ing into a 500.
    if state.get("segmenter") is None:
        raise HTTPException(422, "no mask found at that point")

    def _run_sam2():
        return segment_point(state, image_pil, [px, py])

    t0 = time.perf_counter()
    try:
        async with state["gpu_lock"].interactive():
            seg = await loop.run_in_executor(None, _run_sam2)
    except Exception as exc:
        raise HTTPException(500, f"SAM2 segment failed: {exc}")
    sam2_ms = (time.perf_counter() - t0) * 1000.0
    if seg is None:
        raise HTTPException(422, "no mask found at that point")
    sam2_box = [float(c) for c in seg["box_xyxy"]]
    sam2_polys = seg.get("polygons") or []
    sam2_score = float(seg["score"]) if seg.get("score") is not None else None
    base = {
        "pipeline": "charlie", "width": W, "height": H,
        "box_xyxy": sam2_box, "mask": {"polygons": sam2_polys}, "mask_score": sam2_score,
    }

    def _env(label, score, route, *, rejected=False, reject_reason=None, vlm_ms=0.0):
        return {
            **base,
            "timings": {"sam2_ms": sam2_ms, "vlm_ms": vlm_ms, "total_ms": sam2_ms + vlm_ms, "route": route},
            "route": route,
            "label": label, "score": score,
            "rejected": rejected, "reject_reason": reject_reason,
        }

    # STAGE 3 — label the SAM2 mask. Load references for the project (if any).
    refs_by_label_arr = None
    refs_by_label_siglip_arr = None
    proj_tags = []
    label_display = {}
    refs_present = False
    if project_id:
        try:
            manifest = load_manifest(project_id, copy=False) or {}
            proj_tags = manifest.get("tags") or []
            label_display = {str(t).lower().strip(): str(t) for t in proj_tags}
            by_label, by_label_siglip, _dirty = await loop.run_in_executor(
                None, _v2_load_or_backfill_reference_embeddings, project_id,
            )
            refs_by_label_arr = _v2_stack_refs(by_label)
            refs_by_label_siglip_arr = _v2_stack_refs(by_label_siglip)
            refs_present = bool(refs_by_label_arr)
        except Exception as e:  # noqa: BLE001
            print(f"[detect_point] reference load failed: {e}")
            refs_present = False

    embed_reject = False

    # TIER A — reference-embedding resolver (projects with references).
    if refs_present:
        try:
            import v2_dinov2
            if not v2_dinov2.is_loaded():
                raise RuntimeError("DINOv2 not loaded")
            x0 = max(0, int(round(sam2_box[0]))); y0 = max(0, int(round(sam2_box[1])))
            x1 = min(W, int(round(sam2_box[2]))); y1 = min(H, int(round(sam2_box[3])))
            if x1 - x0 >= 4 and y1 - y0 >= 4:
                if dataset_type_hint:
                    dataset_type = dataset_type_hint
                else:
                    try:
                        dt = _classify_dataset_type_cached(project_id, list(proj_tags))
                        dataset_type = dt.get("type") if isinstance(dt, dict) else "general"
                    except Exception:
                        dataset_type = "general"
                score_mode = "knn" if dataset_type == "specific" else "centroid"

                def _encode():
                    import v2_dinov2 as _v2d
                    import v2_siglip as _v2s
                    clean = _v2d.inpaint_bbox_crop(image_pil, (x0, y0, x1, y1), sam2_polys)
                    square = _v2d.center_square_crop(clean)
                    vecs = _v2d.encode_images_batch([square])
                    if vecs is None or vecs.shape[0] == 0:
                        return None, None
                    e = [round(float(x), 6) for x in vecs[0].tolist()]
                    es = None
                    if _v2s.is_loaded():
                        try:
                            sv = _v2s.encode_images_batch([square])
                            if sv is not None and sv.shape[0] > 0:
                                es = [round(float(x), 6) for x in sv[0].tolist()]
                        except Exception:
                            es = None
                    return e, es

                async with state["gpu_lock"].interactive():
                    emb, emb_siglip = await loop.run_in_executor(None, _encode)
                if emb is not None:
                    class_thresholds = _v2_compute_class_thresholds(refs_by_label_arr)
                    verdict = _v2_resolve_label_specific(
                        emb,
                        None,
                        refs_by_label_arr,
                        label_display,
                        score_mode=score_mode,
                        gd_score=None,
                        embedding_siglip=emb_siglip,
                        refs_by_label_siglip=refs_by_label_siglip_arr or None,
                        class_thresholds=class_thresholds,
                    )
                    if verdict.get("rejected"):
                        embed_reject = True
                    elif verdict.get("pred_label"):
                        return _env(verdict.get("pred_label"), verdict.get("embed_sim_for_label"), "sam2+embed")
        except Exception as e:  # noqa: BLE001 — embed failure degrades to the VLM
            print(f"[detect_point] embed resolve failed, falling back to VLM: {e}")

    # TIER B — VLM labeller (always passes the SAM mask so background is suppressed).
    if labels:
        v_label = v_score = None
        t1 = time.perf_counter()
        try:
            from vlm_validate import vlm_classify
            v_label, v_score = await loop.run_in_executor(
                None, vlm_classify, image_pil, sam2_box, labels, sam2_polys,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[detect_point] vlm_classify failed: {e}")
        vlm_ms = (time.perf_counter() - t1) * 1000.0
        if v_label:
            return _env(v_label, float(v_score) if v_score is not None else None, "sam2+vlm", vlm_ms=vlm_ms)

    # STAGE 4 — terminal (never 422).
    if allow_reject and embed_reject:
        return _env(None, None, "sam2+embed", rejected=True, reject_reason="embed")
    if labels:
        return _env(labels[0], sam2_score, "sam2+vlm")
    return _env(None, None, "sam2")


@app.post("/api/charlie/imports/detect_point")
async def charlie_imports_detect_point(
    image: UploadFile | None = File(None),
    point: str = Form(...),
    project_id: str = Form(""),
    import_id: str = Form(""),
    labels: str = Form(""),
):
    """Click-to-detect with dual pipeline routing.

    - Image has NOT been auto-labelled yet → SAM3 text-prompt path
      (vision encoded once, then N text-only forward passes). Fast
      and accurate when the candidate labels match the image content.
    - Image HAS been auto-labelled → SAM2 (point prompt) + VLM
      label-classify the crop. The user has already accepted a
      prior labelling pass, so SAM3 retrying the same text prompts
      typically just re-finds what's already on the tile; SAM2 gives
      a fresh mask under the click and the VLM picks the label.

    Falls back to SAM3 for legacy uploads with no project/import
    context (no manifest entry to check the labelled flag against).

    Returns box_xyxy + mask + label + score, the shape BoxEditor's
    onPointDetect handler expects.
    """
    try:
        pt = json.loads(point)
        if not (isinstance(pt, list) and len(pt) == 2):
            raise ValueError("point must be [x, y]")
        px, py = float(pt[0]), float(pt[1])
    except Exception as e:
        print(f"[charlie/detect_point] 400 invalid point — raw={point!r} err={e}")
        raise HTTPException(400, f"invalid point payload: {e}")

    candidate_labels: list[str] = []
    if labels:
        try:
            parsed = json.loads(labels)
            if isinstance(parsed, list):
                candidate_labels = [str(t).strip() for t in parsed if str(t).strip()]
        except Exception:
            pass
    if not candidate_labels and project_id:
        try:
            m = load_manifest(project_id, copy=False) or {}
            candidate_labels = [str(t).strip() for t in (m.get("tags") or []) if str(t).strip()]
        except Exception:
            pass

    # Decide the route: SAM3 for never-labelled tiles, SAM2+VLM for
    # already-labelled ones. Mirrors _v2_is_labelled so the routing
    # matches the dataset's badge state.
    use_sam2_vlm = False
    if project_id and import_id:
        try:
            m_manifest = load_manifest(project_id, copy=False) or {}
            for entry in (m_manifest.get("imports") or []):
                if isinstance(entry, dict) and entry.get("id") == import_id:
                    if entry.get("editedBoxesSet"):
                        edited = entry.get("editedBoxes")
                        use_sam2_vlm = isinstance(edited, list) and len(edited) > 0
                    else:
                        flag = entry.get("labelled")
                        if flag is True:
                            use_sam2_vlm = True
                        elif flag is False:
                            use_sam2_vlm = False
                        else:
                            use_sam2_vlm = bool(entry.get("detections"))
                    break
        except Exception as e:
            print(f"[charlie/detect_point] manifest lookup failed, defaulting to SAM3: {e}")

    image_pil = await _charlie_load_image_pil(
        image, project_id, import_id, endpoint_tag="charlie/detect_point",
    )
    W, H = image_pil.size

    loop = asyncio.get_running_loop()

    # SPECIFIC projects: SAM2 point (label-agnostic mask) + reference
    # resolver — the same basis as the batch job, and the only click-to-
    # detect path that labels objects whose word SAM3 can't text-match
    # (e.g. "orangutan", a made-up label). For these projects the
    # references are the label source of truth, so we use this regardless
    # of the labelled/unlabelled routing (which only helps general
    # projects). Falls through to the SAM3 / VLM logic when there are no
    # usable references.
    if project_id and state.get("segmenter") is not None:
        try:
            dt = await loop.run_in_executor(
                None, _classify_dataset_type_cached, project_id, candidate_labels,
            )
            is_specific = isinstance(dt, dict) and dt.get("type") == "specific"
        except Exception as e:
            print(f"[charlie/detect_point] dataset-type lookup failed: {e}")
            is_specific = False
        if is_specific:
            def _run_ref():
                seg = segment_point(state, image_pil, [px, py])
                if seg is None:
                    return None
                mp = seg.get("polygons") or []
                box = [float(c) for c in seg["box_xyxy"]]
                lbl, sc = _charlie_interactive_ref_label(
                    image_pil, box, mp, project_id, candidate_labels,
                )
                return box, mp, lbl, sc, seg.get("score")
            t0 = time.perf_counter()
            try:
                async with state["gpu_lock"].interactive():
                    ref_res = await loop.run_in_executor(None, _run_ref)
            except Exception as exc:
                print(f"[charlie/detect_point] ref path failed: {exc}")
                ref_res = None
            if ref_res is not None and ref_res[2]:
                box, mp, lbl, sc, mask_score = ref_res
                ms = (time.perf_counter() - t0) * 1000.0
                print(
                    f"[charlie/detect_point/sam2+refs] W={W} H={H} "
                    f"point=({px:.0f},{py:.0f}) label={lbl!r} ms={ms:.0f}"
                )
                return {
                    "pipeline": "charlie",
                    "width": W, "height": H,
                    "timings": {"total_ms": ms, "route": "sam2+refs"},
                    "box_xyxy": box,
                    "mask": {"polygons": mp},
                    "label": lbl,
                    "score": (sc if sc is not None else mask_score),
                    "rejected": False,
                    "reject_reason": None,
                }
            # else fall through to the SAM3 / VLM routing below.

    if use_sam2_vlm:
        # SAM2 point prompt — label-agnostic mask under the click.
        # Cheap (~80-300ms on GPU), runs inside the interactive gate
        # so it doesn't queue behind the label-charlie batch job.
        if state.get("segmenter") is None:
            print("[charlie/detect_point] SAM2 unavailable; falling through to SAM3")
            use_sam2_vlm = False
        else:
            def _run_sam2():
                return segment_point(state, image_pil, [px, py])
            t0 = time.perf_counter()
            try:
                async with state["gpu_lock"].interactive():
                    seg = await loop.run_in_executor(None, _run_sam2)
            except Exception as exc:
                print(f"[charlie/detect_point] SAM2 raised: {exc}")
                raise HTTPException(500, f"SAM2 segment failed: {exc}")
            sam2_ms = (time.perf_counter() - t0) * 1000.0
            if seg is None:
                raise HTTPException(422, "no mask found at that point")
            sam2_box = [float(c) for c in seg["box_xyxy"]]
            sam2_mask_polys = seg.get("polygons") or []
            sam2_score = float(seg.get("score")) if seg.get("score") is not None else None

            # VLM picks a label from the candidates. Remote worker
            # path (VLM_WORKER_URL) is the fast one; local fallback
            # is slower but keeps the feature working in dev.
            vlm_label: str | None = None
            vlm_score: float | None = None
            t1 = time.perf_counter()
            try:
                from vlm_validate import vlm_classify as _vlm_classify
            except Exception as e:
                print(f"[charlie/detect_point] vlm_validate import failed: {e}")
                _vlm_classify = None
            if _vlm_classify is not None:
                try:
                    v_label, v_score = await loop.run_in_executor(
                        None,
                        _vlm_classify,
                        image_pil, sam2_box, candidate_labels, sam2_mask_polys,
                    )
                    vlm_label = v_label
                    if v_score is not None:
                        vlm_score = float(v_score)
                except Exception as e:
                    print(f"[charlie/detect_point] VLM call raised: {e}")
            vlm_ms = (time.perf_counter() - t1) * 1000.0
            # Last-resort label: first candidate, so the user gets a
            # placed box they can rename in one click rather than a
            # silent 422.
            if not vlm_label and candidate_labels:
                vlm_label = candidate_labels[0]
            score_out = vlm_score if vlm_score is not None else sam2_score
            print(
                f"[charlie/detect_point/sam2+vlm] W={W} H={H} point=({px:.0f},{py:.0f}) "
                f"label={vlm_label!r} sam2_ms={sam2_ms:.0f} vlm_ms={vlm_ms:.0f}"
            )
            return {
                "pipeline": "charlie",
                "width": W, "height": H,
                "timings": {
                    "sam2_ms": sam2_ms,
                    "vlm_ms": vlm_ms,
                    "total_ms": sam2_ms + vlm_ms,
                    "route": "sam2+vlm",
                },
                "box_xyxy": sam2_box,
                "mask": {"polygons": sam2_mask_polys},
                "label": vlm_label,
                "score": score_out,
                "rejected": False,
                "reject_reason": None,
            }

    # SAM3 path: unlabelled tiles, or labelled tiles where SAM2 was
    # unavailable. (`charlie` was never bound in this handler in the SaaS
    # build — a latent NameError on this route, fixed here.)
    charlie = state.get("charlie")
    if charlie is None:
        raise HTTPException(503, "SAM3 not loaded")

    def _run_sam3():
        return charlie.segment_point(image_pil, [px, py], candidate_labels)

    try:
        async with state["gpu_lock"].interactive():
            detection, sam3_timings = await loop.run_in_executor(None, _run_sam3)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"charlie detect_point error: {exc}")

    if detection is None:
        # SAM3 found no concept under the click — fall back to the SAM2
        # floor (+ reference/VLM label) so the click still places a box
        # instead of dead-ending with a 422.
        return await _detect_point_unified(
            image_pil, [px, py],
            candidate_labels=candidate_labels,
            project_id=(project_id or None),
            dataset_type_hint=None,
            allow_sam3=False,
            allow_reject=use_sam2_vlm,
            is_labelled=use_sam2_vlm,
        )

    box_xyxy = [float(c) for c in detection["box"]]
    mask_polys = (
        detection["mask"]["polygons"]
        if isinstance(detection.get("mask"), dict) else []
    )
    label = detection.get("gd_label") or None
    score = float(detection["gd_score"]) if detection.get("gd_score") is not None else None
    print(
        f"[charlie/detect_point/sam3] W={W} H={H} point=({px:.0f},{py:.0f}) "
        f"label={label!r} score={score} total={sam3_timings.get('total_ms', 0):.0f}ms"
    )
    return {
        "pipeline": "charlie",
        "width": W, "height": H,
        "timings": {**sam3_timings, "route": "sam3"},
        "box_xyxy": box_xyxy,
        "mask": {"polygons": mask_polys},
        "label": label,
        "score": score,
        "rejected": False,
        "reject_reason": None,
    }


@app.post("/api/charlie/imports/segment_box")
async def charlie_imports_segment_box(
    image: UploadFile | None = File(None),
    box: str = Form(...),
    labels: str = Form(""),
    project_id: str = Form(""),
    import_id: str = Form(""),
):
    """Box-prompted segmentation via SAM3 (portable build; SAM2 removed).

    SAM3's input_boxes channel isn't a true "segment inside this box"
    prompt, so pipeline_charlie.segment_box runs text-prompted SAM3 per
    candidate label and picks the detection whose bbox best matches the
    user's box. Source image is loaded from disk via import_id whenever
    possible; falls back to an `image` upload for unsaved imports.
    """
    charlie = state.get("charlie")
    if charlie is None:
        raise HTTPException(503, "SAM3 not loaded — segment_box unavailable")
    try:
        coords = json.loads(box)
        if not (isinstance(coords, list) and len(coords) == 4):
            raise ValueError("box must be [x0, y0, x1, y1]")
    except Exception as e:
        raise HTTPException(400, f"invalid box payload: {e}")

    image_pil = await _charlie_load_image_pil(
        image, project_id, import_id, endpoint_tag="charlie/segment_box",
    )

    candidate_labels: list[str] = []
    try:
        if labels:
            parsed = json.loads(labels)
            if isinstance(parsed, list):
                candidate_labels = [str(t).strip() for t in parsed if str(t).strip()]
    except Exception:
        candidate_labels = []
    if not candidate_labels and project_id:
        try:
            candidate_labels = list(
                (load_manifest(project_id, copy=False) or {}).get("tags") or []
            )
        except Exception:
            candidate_labels = []

    loop = asyncio.get_running_loop()

    def _infer():
        t0 = time.perf_counter()
        detection, _timings = charlie.segment_box(
            image_pil, [float(c) for c in coords], candidate_labels or None,
        )
        ms = (time.perf_counter() - t0) * 1000.0
        return detection, ms

    try:
        # Interactive — Charlie pipeline's add-box / drag-box. User
        # is in the BoxEditor waiting on the mask render.
        async with state["gpu_lock"].interactive():
            detection, sam3_ms = await loop.run_in_executor(None, _infer)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"charlie segment_box error: {exc}")

    mask_payload = (detection or {}).get("mask")
    if not mask_payload:
        raise HTTPException(422, "no mask found in box")

    return {
        "pipeline": "charlie",
        "timings": {"sam3_predict_ms": round(sam3_ms, 1), "total_ms": round(sam3_ms, 1)},
        "box_xyxy": [float(c) for c in coords],
        "mask": mask_payload,
        "mask_score": None,
    }


@app.post("/api/charlie/imports/classify_box")
async def charlie_imports_classify_box(
    image: UploadFile | None = File(None),
    box: str = Form(...),
    labels: str = Form(...),
    project_id: str = Form(""),
    import_id: str = Form(""),
):
    """Embedding-based classification of a user-drawn bbox.

    SAM3 box-match mask (best effort) → DINOv2 → reference-centroid
    resolver → optional VLM tiebreak (specific only) via the same
    confidence-weighted fusion the labelling job uses. SAM2 removed in
    the portable build; classification degrades gracefully to a plain
    box crop when no mask is available.

    Source image is loaded from disk via import_id when available
    (skips the FE-side fetch entirely); falls back to an `image`
    upload for unsaved imports.

    Returns {label, score, verdict} — same shape as detect_point
    so the FE pipeline popup renders the same way.
    """
    import v2_dinov2 as _v2d
    if not _v2d.is_loaded():
        raise HTTPException(503, "DINOv2 not loaded yet")

    try:
        coords = json.loads(box)
        if not (isinstance(coords, list) and len(coords) == 4):
            raise ValueError("box must be [x0, y0, x1, y1]")
        coords_f = [float(c) for c in coords]
    except Exception as e:
        raise HTTPException(400, f"invalid box payload: {e}")
    try:
        candidate_list = json.loads(labels)
        if not isinstance(candidate_list, list) or not all(isinstance(t, str) for t in candidate_list):
            raise ValueError("labels must be a JSON array of strings")
    except Exception as e:
        raise HTTPException(400, f"invalid labels payload: {e}")

    if not project_id:
        return {
            "pipeline": "charlie",
            "label": None, "score": None,
            "reason": "project_id missing — no reference centroids to resolve against",
        }

    image_pil = await _charlie_load_image_pil(
        image, project_id, import_id, endpoint_tag="charlie/classify_box",
    )

    loop = asyncio.get_running_loop()
    timings: dict[str, float] = {}

    # Best-effort mask for the inpaint crop via SAM3's box matcher.
    # No mask (SAM3 unloaded / no match) → plain box crop downstream.
    def _segment():
        t0 = time.perf_counter()
        charlie = state.get("charlie")
        detection = None
        if charlie is not None:
            try:
                detection, _t = charlie.segment_box(image_pil, coords_f, candidate_list or None)
            except Exception as e:
                print(f"[charlie/classify_box] sam3 mask failed: {e}")
        ms = (time.perf_counter() - t0) * 1000.0
        return ((detection or {}).get("mask") or None), ms

    try:
        # Interactive — Charlie pipeline's classify_box (called from
        # the BoxEditor's relabel + add-box flows).
        async with state["gpu_lock"].interactive():
            mask_payload, sam2_ms = await loop.run_in_executor(None, _segment)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"classify_box segment failed: {exc}")
    timings["sam3_mask_ms"] = round(sam2_ms, 1)

    mask_polys = (
        mask_payload.get("polygons")
        if isinstance(mask_payload, dict) else None
    )

    # Dataset-type branch: specific → embedding resolver; general →
    # SAM3 text-prompted classification.
    try:
        dt = _classify_dataset_type_cached(project_id, list(candidate_list))
        dataset_type = (
            (dt.get("type") or "general")
            if isinstance(dt, dict) else "general"
        )
    except Exception:
        dataset_type = "general"
    try:
        if dataset_type == "specific":
            # Same classify_box request — interactive throughout.
            async with state["gpu_lock"].interactive():
                verdict, embed_timings = await loop.run_in_executor(
                    None,
                    _charlie_resolve_label_for_box,
                    image_pil, coords_f, mask_polys, project_id, candidate_list,
                )
            timings.update(embed_timings)
        else:
            async with state["gpu_lock"].interactive():
                verdict, sam3_timings = await loop.run_in_executor(
                    None,
                    _charlie_classify_general_sam3,
                    image_pil, coords_f, candidate_list,
                )
            timings.update(sam3_timings)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"charlie classify_box error: {exc}")

    # VLM tiebreak on ambiguous detections (specific only — the
    # resolver only sets `ambiguous` when it was called with the
    # specific path's threshold).
    if (
        verdict
        and verdict.get("ambiguous")
        and not verdict.get("rejected")
    ):
        verdict, vlm_ms = await _charlie_vlm_tiebreak_async(
            verdict, image_pil, coords_f, mask_polys, loop,
        )
        if vlm_ms is not None:
            timings["vlm_ms"] = round(vlm_ms, 1)

    timings["total_ms"] = round(
        sum(v for v in timings.values() if isinstance(v, (int, float))), 1,
    )
    return {
        "pipeline": "charlie",
        "timings": timings,
        "label": (verdict.get("pred_label") if verdict else None),
        "score": (verdict.get("embed_sim_for_label") if verdict else None),
        "verdict": verdict,
    }


@app.post("/api/charlie/imports/segment_and_classify_box")
async def charlie_imports_segment_and_classify_box(
    image: UploadFile | None = File(None),
    box: str = Form(...),
    labels: str = Form(...),
    project_id: str = Form(""),
    import_id: str = Form(""),
):
    """One-shot add-box — pure SAM3.

    Single SAM3 round-trip with cached vision: one vision encode +
    N text-only forwards (one per candidate label). The label whose
    text-prompted mask has the highest IoU with the user-drawn box
    wins. Mirrors the click-to-detect endpoint above and matches the
    user's "make it as fast as possible, just SAM3" brief.

    Previous flow chained SAM2 (box-prompt mask) → DINOv2 + SigLIP
    embedding → reference-centroid resolver → optional VLM tiebreak.
    That's still what the batch labelling job uses; the interactive
    add-box gesture just needs a label fast.

    Returns box_xyxy, mask, label, score — same shape BoxEditor's
    onAddBoxDetect handler expects.
    """
    charlie = state.get("charlie")
    if charlie is None:
        raise HTTPException(503, "SAM3 not loaded — add-box unavailable")

    try:
        coords = json.loads(box)
        if not (isinstance(coords, list) and len(coords) == 4):
            raise ValueError("box must be [x0, y0, x1, y1]")
        coords_f = [float(c) for c in coords]
    except Exception as e:
        raise HTTPException(400, f"invalid box payload: {e}")
    try:
        candidate_list = json.loads(labels)
        if not isinstance(candidate_list, list) or not all(isinstance(t, str) for t in candidate_list):
            raise ValueError("labels must be a JSON array of strings")
    except Exception as e:
        raise HTTPException(400, f"invalid labels payload: {e}")

    if not candidate_list:
        raise HTTPException(400, "no candidate labels to match against")

    image_pil = await _charlie_load_image_pil(
        image, project_id, import_id, endpoint_tag="charlie/segment_and_classify_box",
    )

    loop = asyncio.get_running_loop()

    # SPECIFIC projects: SAM2 box-prompt (label-agnostic geometry) + the
    # reference resolver — same basis as the batch job. This is the only
    # add-box path that labels a box around an object whose label SAM3
    # can't text-match (e.g. "orangutan", a made-up label). SAM3's
    # text-IoU shortcut below (general projects) returns 422 for those.
    is_specific = False
    if project_id and state.get("segmenter") is not None:
        try:
            dt = await loop.run_in_executor(
                None, _classify_dataset_type_cached, project_id, candidate_list,
            )
            is_specific = isinstance(dt, dict) and dt.get("type") == "specific"
        except Exception as e:
            print(f"[charlie/segment_and_classify_box] dataset-type lookup failed: {e}")
    if is_specific:
        def _run_ref():
            masks = segment_boxes(state, image_pil, [coords_f])
            mp = (masks[0] or {}).get("polygons") if (masks and masks[0]) else None
            if mp is None:
                return None
            lbl, sc = _charlie_interactive_ref_label(
                image_pil, coords_f, mp, project_id, candidate_list,
            )
            return mp, lbl, sc
        try:
            async with state["gpu_lock"].interactive():
                ref_res = await loop.run_in_executor(None, _run_ref)
        except Exception as exc:
            print(f"[charlie/segment_and_classify_box] ref path failed: {exc}")
            ref_res = None
        if ref_res is not None and ref_res[1]:
            mp, lbl, sc = ref_res
            return {
                "pipeline": "charlie",
                "timings": {},
                "box_xyxy": coords_f,
                "mask": {"polygons": mp or []},
                "label": lbl,
                "score": sc,
            }
        # else fall through to the SAM3 text-IoU path below.

    def _run():
        return charlie.segment_box(image_pil, coords_f, candidate_list)

    try:
        async with state["gpu_lock"].interactive():
            detection, sam3_timings = await loop.run_in_executor(None, _run)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"charlie segment_and_classify_box error: {exc}")

    if detection is None:
        raise HTTPException(422, "no mask found in box")

    out_box = [float(c) for c in detection["box"]]
    mask_polys = (
        detection["mask"]["polygons"]
        if isinstance(detection.get("mask"), dict) else []
    )
    label = detection.get("gd_label") or None
    score = float(detection["gd_score"]) if detection.get("gd_score") is not None else None
    return {
        "pipeline": "charlie",
        "timings": sam3_timings,
        "box_xyxy": out_box,
        "mask": {"polygons": mask_polys},
        "label": label,
        "score": score,
    }




# ── Reference-editor interactive tools (portable build: SAM3-backed) ─────────
# The SaaS build served these with SAM2 (segment) + GroundingDINO (classify).
# Rewritten as thin wrappers over pipeline_charlie + _detect_point_unified so
# the reference editor keeps its click/box tools on the SAM3-only stack.

async def _reference_image_pil(
    image: UploadFile | None, project_id: str, filename: str, endpoint: str
) -> "PILImage.Image":
    if image is not None:
        raw = await image.read()
        if raw:
            img = PILImage.open(io.BytesIO(raw))
            return ImageOps.exif_transpose(img).convert("RGB")
    if project_id and filename and "/" not in filename and ".." not in filename:
        p = project_dir(project_id) / "references" / filename
        if p.is_file():
            img = PILImage.open(p)
            return ImageOps.exif_transpose(img).convert("RGB")
    raise HTTPException(400, f"{endpoint}: no image supplied")


def _parse_labels_json(labels: str) -> list[str]:
    try:
        parsed = json.loads(labels) if labels else []
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(t).strip() for t in parsed if str(t).strip()]


@app.post("/api/v2/references/segment_box")
async def v2_reference_segment_box(
    image: UploadFile | None = File(None),
    box: str = Form(...),
    labels: str = Form(""),
    project_id: str = Form(""),
    filename: str = Form(""),
):
    charlie = state.get("charlie")
    if charlie is None:
        raise HTTPException(503, "SAM3 not loaded")
    try:
        coords = [float(c) for c in json.loads(box)]
        assert len(coords) == 4
    except Exception:
        raise HTTPException(400, "box must be [x0, y0, x1, y1]")
    image_pil = await _reference_image_pil(image, project_id, filename, "references/segment_box")
    candidate = _parse_labels_json(labels)
    if not candidate and project_id:
        candidate = list((load_manifest(project_id, copy=False) or {}).get("tags") or [])
    loop = asyncio.get_running_loop()

    def _run():
        detection, _t = charlie.segment_box(image_pil, coords, candidate or None)
        return detection

    try:
        async with state["gpu_lock"].interactive():
            detection = await loop.run_in_executor(None, _run)
    except Exception as exc:
        raise HTTPException(500, f"reference segment_box failed: {exc}")
    mask = (detection or {}).get("mask")
    if not mask:
        raise HTTPException(422, "no mask found in box")
    return {"mask": mask, "box_xyxy": coords}


@app.post("/api/v2/references/classify_box")
async def v2_reference_classify_box(
    image: UploadFile | None = File(None),
    box: str = Form(...),
    labels: str = Form("[]"),
    project_id: str = Form(""),
    filename: str = Form(""),
):
    charlie = state.get("charlie")
    if charlie is None:
        raise HTTPException(503, "SAM3 not loaded")
    try:
        coords = [float(c) for c in json.loads(box)]
        assert len(coords) == 4
    except Exception:
        raise HTTPException(400, "box must be [x0, y0, x1, y1]")
    candidate = _parse_labels_json(labels)
    if not candidate:
        raise HTTPException(400, "labels must be a non-empty JSON array")
    image_pil = await _reference_image_pil(image, project_id, filename, "references/classify_box")
    loop = asyncio.get_running_loop()

    def _run():
        return charlie.classify_box(image_pil, coords, candidate)

    try:
        async with state["gpu_lock"].interactive():
            label, score, _t = await loop.run_in_executor(None, _run)
    except Exception as exc:
        raise HTTPException(500, f"reference classify_box failed: {exc}")
    return {"label": label, "score": score}


@app.post("/api/v2/references/detect_point")
async def v2_reference_detect_point(
    image: UploadFile | None = File(None),
    point: str = Form(...),
    labels: str = Form(""),
    force_label: str = Form(""),
    project_id: str = Form(""),
    filename: str = Form(""),
):
    try:
        pt = [float(c) for c in json.loads(point)]
        assert len(pt) == 2
    except Exception:
        raise HTTPException(400, "point must be [x, y]")
    image_pil = await _reference_image_pil(image, project_id, filename, "references/detect_point")
    candidate = _parse_labels_json(labels)
    if not candidate and project_id:
        candidate = list((load_manifest(project_id, copy=False) or {}).get("tags") or [])
    result = await _detect_point_unified(
        image_pil,
        pt,
        candidate_labels=candidate,
        project_id=project_id or "",
        allow_sam3=True,
    )
    fl = (force_label or "").strip()
    if fl and isinstance(result, dict):
        result["label"] = fl
    return result


if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8001, reload=False)
