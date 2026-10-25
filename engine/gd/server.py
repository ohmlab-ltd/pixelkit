"""The PixelKit engine: a FastAPI app over the workspace's datasets.

Run:
    python gd/server.py
"""
import asyncio
import hashlib
import io
import json
import os
import sys
import uuid as _uuid

# UTF-8 with replacement, so a non-cp1252 character in a log line cannot
# raise UnicodeEncodeError on a Windows console. Runs before any print.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    import orjson as _orjson
except Exception:
    _orjson = None


def _json_loads(s: str | bytes) -> object:
    if _orjson is not None:
        return _orjson.loads(s.encode() if isinstance(s, str) else s)
    return json.loads(s if isinstance(s, str) else s.decode())

# Loads the .env one level above gd/, before any os.getenv read.
from dotenv import load_dotenv  # noqa: E402
from pathlib import Path as _Path  # noqa: E402
load_dotenv(_Path(__file__).resolve().parent.parent / ".env", override=True)

# Set before torch is imported. Expandable segments keep the CUDA allocator
# from fragmenting VRAM; MPS ops without a kernel fall back to the CPU.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import re
import secrets


# Must match engine/pyproject.toml, desktop/package.json and the release tag.
PK_VERSION = "0.4.1"


def _configured_device() -> str:
    """Configured device, or "" for auto-detect. PK_DEVICE overrides the config key."""
    env = (os.environ.get("PK_DEVICE") or "").strip().lower()
    if env:
        return env
    try:
        import workspace as _ws
        cfg = (_ws.load_config().get("device") or "").strip().lower()
    except Exception:
        return ""
    if cfg in ("cuda", "mps", "cpu") or re.fullmatch(r"cuda:\d+", cfg):
        return cfg
    return ""


def _resolve_device() -> str:
    """_configured_device() when available, else cuda > mps > cpu."""
    want = _configured_device()
    has_mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    if want:
        if re.fullmatch(r"cuda(:\d+)?", want) and torch.cuda.is_available():
            idx = int(want.split(":")[1]) if ":" in want else 0
            if idx < torch.cuda.device_count():
                return want
        elif want == "mps" and has_mps:
            return "mps"
        elif want == "cpu":
            return "cpu"
        print(f"[server] device {want!r} not available on this machine; auto-detecting.")
    if torch.cuda.is_available():
        return "cuda"
    if has_mps:
        return "mps"
    return "cpu"
import shutil
import time
import zipfile
from collections import OrderedDict
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, Response
from PIL import Image as PILImage, ImageOps
from xml.sax.saxutils import escape as _xml_escape
# Larger images raise DecompressionBombError.
PILImage.MAX_IMAGE_PIXELS = 100_000_000

# HEIC/HEIF decoding when the optional pi-heif package is installed. Pillow
# decodes AVIF itself.
try:
    import pi_heif as _pi_heif  # type: ignore
    _pi_heif.register_heif_opener()
    print("[server] pi-heif registered (HEIC decode enabled)")
except Exception:
    pass

MAX_UPLOAD_BYTES_PER_FILE = 100 * 1024 * 1024
MAX_FILES_PER_UPLOAD_BATCH = 100


class FastJSONResponse(Response):
    """JSON through orjson, for the large payloads; numpy values and
    non-string keys as FastAPI's deprecated ORJSONResponse wrote them."""

    media_type = "application/json"

    def render(self, content) -> bytes:
        if _orjson is None:
            return json.dumps(content, default=str).encode()
        return _orjson.dumps(content, option=_orjson.OPT_NON_STR_KEYS | _orjson.OPT_SERIALIZE_NUMPY)


async def _read_upload(upload: UploadFile, what: str = "image") -> bytes:
    """An upload's bytes; 413 above MAX_UPLOAD_BYTES_PER_FILE, refused before
    a larger file is read into memory."""
    size = getattr(upload, "size", None)
    if size is not None and size > MAX_UPLOAD_BYTES_PER_FILE:
        raise HTTPException(413, f"{what} too large ({size} bytes, max {MAX_UPLOAD_BYTES_PER_FILE})")
    raw = await upload.read(MAX_UPLOAD_BYTES_PER_FILE + 1)
    if len(raw) > MAX_UPLOAD_BYTES_PER_FILE:
        raise HTTPException(413, f"{what} too large (max {MAX_UPLOAD_BYTES_PER_FILE} bytes)")
    return raw


# Formats a browser shows, stored as uploaded under their own extension.
_SHOWN_FORMATS = {"JPEG": ".jpg", "MPO": ".jpg", "PNG": ".png", "WEBP": ".webp",
                  "GIF": ".gif", "BMP": ".bmp", "AVIF": ".avif"}


def _stored_image(raw: bytes) -> tuple[bytes, str]:
    """The bytes and extension to store for an uploaded image: a format a
    browser shows as uploaded, any other as an upright RGB JPEG. 400 when the
    bytes are not an image."""
    try:
        with PILImage.open(io.BytesIO(raw)) as im:
            fmt = (im.format or "").upper()
            if fmt in _SHOWN_FORMATS:
                im.verify()
                return raw, _SHOWN_FORMATS[fmt]
            buf = io.BytesIO()
            _to_rgb(ImageOps.exif_transpose(im)).save(buf, "JPEG", quality=90)
            return buf.getvalue(), ".jpg"
    except Exception as e:
        raise HTTPException(400, f"cannot decode image: {e}")
from pydantic import BaseModel

from jobs import JobManager
import audit
import aug_source
import store
import models as models_mgr
import workspace
import containers


# Job/activity events persist to a local SQLite in the workspace.
def add_event(kind: str, **data) -> None:
    try:
        audit.add_event(kind, **data)
    except Exception as e:
        print(f"[audit] add_event({kind}) failed: {e}")


import threading as _threading  # noqa: E402


def _to_rgb(im: PILImage.Image) -> PILImage.Image:
    """RGB for any mode. 16- and 32-bit greyscale is scaled to 8 bits
    first, since convert() clamps it."""
    if im.mode in ("I;16", "I;16B", "I;16L", "I;16N", "I"):
        a = np.asarray(im, dtype=np.float64)
        top = 65535.0 if (im.mode != "I" or a.max() > 255) else 255.0
        return PILImage.fromarray(np.clip(a * (255.0 / top), 0, 255).astype(np.uint8), "L").convert("RGB")
    return im.convert("RGB")


def _check_jpeg_whole(raw: bytes) -> None:
    """Raise when a JPEG stops early. The 1/8-scale draft decode still reads every scan."""
    with PILImage.open(io.BytesIO(raw)) as im:
        im.draft("RGB", (max(1, im.width // 8), max(1, im.height // 8)))
        im.load()


def _open_upright(src: "Path | str | bytes") -> PILImage.Image:
    """Decode with EXIF orientation applied, as RGB. Boxes and stored sizes
    refer to this upright picture."""
    with PILImage.open(io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else src) as im:
        return _to_rgb(ImageOps.exif_transpose(im))


def _upright_size(im: PILImage.Image) -> tuple[int, int]:
    """(width, height) after EXIF orientation, without decoding."""
    try:
        orientation = im.getexif().get(0x0112, 1)
    except Exception:
        orientation = 1
    return (im.height, im.width) if orientation in (5, 6, 7, 8) else (im.width, im.height)


state: dict[str, Any] = {"device": "cpu", "jobs": JobManager()}


# Running GPU jobs. Background embeddings wait on state["gpu_idle"], which
# is clear while this is nonzero and is created at startup.
_GPU_JOB_DEPTH = 0


def _pause_bg_embeddings() -> None:
    """Called when a GPU job starts."""
    global _GPU_JOB_DEPTH
    _GPU_JOB_DEPTH += 1
    ev = state.get("gpu_idle")
    if ev is not None:
        ev.clear()


def _resume_bg_embeddings() -> None:
    """Called when a GPU job ends."""
    global _GPU_JOB_DEPTH
    _GPU_JOB_DEPTH = max(0, _GPU_JOB_DEPTH - 1)
    if _GPU_JOB_DEPTH == 0:
        ev = state.get("gpu_idle")
        if ev is not None:
            ev.set()


def _gpu_job_guarded(runner):
    """Wrap a JobManager runner so background embeddings pause while it runs."""
    async def _wrapped(job, emit, cancel_event):
        _pause_bg_embeddings()
        try:
            return await runner(job, emit, cancel_event)
        finally:
            _resume_bg_embeddings()
    return _wrapped


def _embed_stored_image(project_id: str, import_id: str, path: "Path") -> bool:
    """Read the stored picture only when its turn comes, so queued uploads
    are not held in memory."""
    try:
        data = path.read_bytes()
    except OSError:
        return False  # deleted while it waited
    return _compute_and_store_image_embedding(project_id, import_id, data)


async def _run_bg_embedding(project_id: str, import_id: str, path: "Path") -> None:
    """Compute and store the whole-image DINOv2 embedding for the dataset stats.
    At most embed_sem run at once, and they wait while a GPU job runs."""
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
                    _embed_stored_image,
                    project_id, import_id, path,
                )
        finally:
            if sem is not None:
                sem.release()
    except Exception as e:
        print(f"[bg-embed] failed for {import_id}: {e}")


def _job_to_audit(job) -> None:
    """Record a finished job in the event log. The field is `job_kind` because
    `kind` is add_event's first argument."""
    add_event(
        "job",
        id=job.id,
        job_kind=job.kind,
        project=job.project,
        status=job.status,
        elapsed_s=round(job.elapsed_s, 2),
        n_images=job.n_images,
        error=job.error,
    )


state["jobs"].on_finish = _job_to_audit


import heapq as _heapq_module


class PriorityGPUGate:
    """Async GPU lock that wakes the most urgent waiter first, FIFO within a
    priority: P_INTERACTIVE for clicks via .interactive(), P_JOB for job
    runners by default, P_BACKGROUND for fire-and-forget work via .background()."""
    P_INTERACTIVE = 0
    P_JOB = 10
    P_BACKGROUND = 20

    def __init__(self) -> None:
        self._free = True
        self._waiters: "list[tuple[int, int, asyncio.Future]]" = []
        self._counter = 0

    async def _acquire(self, priority: int) -> None:
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
            # Cancelled after being handed the gate: release it. A slot left
            # in the heap is skipped by _release.
            if fut.done() and not fut.cancelled():
                self._release()
            raise

    def _release(self) -> None:
        while self._waiters:
            _, _, fut = _heapq_module.heappop(self._waiters)
            if not fut.done():
                fut.set_result(None)
                return
        self._free = True

    def contended(self, priority: int = P_JOB) -> bool:
        """True while work at `priority` or more urgent waits. Safe from any thread."""
        return any(p <= priority and not f.done() for p, _, f in list(self._waiters))

    async def hand_over(self, priority: int = P_JOB) -> None:
        """For a long-running holder: let waiters at `priority` or more
        urgent go first, then queue to take the gate back."""
        if not self.contended(priority):
            return
        self._release()
        await self._acquire(priority)

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
        await self._acquire(self.P_INTERACTIVE)
        try:
            yield
        finally:
            self._release()

    @asynccontextmanager
    async def job(self):
        await self._acquire(self.P_JOB)
        try:
            yield
        finally:
            self._release()

    @asynccontextmanager
    async def background(self):
        await self._acquire(self.P_BACKGROUND)
        try:
            yield
        finally:
            self._release()


# Model loads for lifespan and /api/models/{name}/load, serialised by
# state["model_load_lock"]: transformers' lazy imports are not thread-safe.

async def _bg_load_dino() -> None:
    try:
        import v2_dinov2
        async with state["model_load_lock"]:
            if v2_dinov2.is_loaded():
                return
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: v2_dinov2.load(state["device"]),
            )
            await asyncio.get_event_loop().run_in_executor(None, v2_dinov2.warmup)
    except Exception as e:
        print(f"[server] v2 DINOv2 load failed: {e}")


async def _bg_load_charlie() -> None:
    try:
        import pipeline_charlie
        async with state["model_load_lock"]:
            if not pipeline_charlie.is_loaded():
                await asyncio.get_event_loop().run_in_executor(
                    None, lambda: pipeline_charlie.load_sam3(state["device"]),
                )
        state["charlie"] = pipeline_charlie
        print("[server] SAM 3 ready.")
    except Exception as e:
        print(f"[server] SAM 3 failed to load: {e}")
        state["charlie"] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    import neuron6_verify
    neuron6_verify.allow_compilers()
    device = _resolve_device()
    # Crop-dataset resync threads reach the per-dataset write locks through this loop.
    state["loop"] = asyncio.get_running_loop()

    # Serialises GPU work so concurrent pipelines cannot exhaust VRAM.
    state["gpu_lock"] = PriorityGPUGate()
    # Set while no GPU job runs.
    _gpu_idle = asyncio.Event()
    _gpu_idle.set()
    state["gpu_idle"] = _gpu_idle
    # Background embeddings past the idle gate at once.
    state["embed_sem"] = asyncio.Semaphore(2)

    state["device"] = device

    # SAM3 and DINOv2 load on cuda and mps, and on cpu only when chosen
    # explicitly. Otherwise, or with PK_DISABLE_MODELS=1, the ML endpoints return 503.
    cpu_forced = _configured_device() == "cpu"
    models_disabled = (
        os.environ.get("PK_DISABLE_MODELS", "").lower() in ("1", "true", "yes", "on")
        or (device == "cpu" and not cpu_forced)
    )
    if models_disabled:
        print(
            f"[server] models disabled (device={device}"
            + (", PK_DISABLE_MODELS set" if os.environ.get("PK_DISABLE_MODELS") else "")
            + "): labelling is off, everything else works."
        )
    elif device == "cpu":
        print("[server] device set to CPU: models load on the CPU and labelling is very slow.")


    # Downloads run in background tasks, so startup never waits on one.
    state["model_load_lock"] = asyncio.Lock()
    models_mgr.apply_token_env()
    sam3_off = models_disabled or os.environ.get("CHARLIE_DISABLED", "").lower() in ("1", "true", "yes", "on")
    # A token saved later downloads and loads SAM 3 under the same rule.
    state["sam3_off"] = sam3_off

    if not models_disabled:
        if models_mgr.is_downloaded("dinov2"):
            asyncio.create_task(_bg_load_dino())
        else:
            # DINOv2 is ungated: no token needed.
            async def _auto_dino() -> None:
                print("[server] DINOv2 weights missing; downloading...")
                await asyncio.to_thread(models_mgr.start_download, "dinov2")
                while True:
                    await asyncio.sleep(5)
                    if models_mgr.is_downloaded("dinov2"):
                        await _bg_load_dino()
                        return
                    with models_mgr._DL_LOCK:
                        rec = dict(models_mgr._DOWNLOADS.get("dinov2") or {})
                    if rec.get("status") == "error":
                        print(f"[server] DINOv2 auto-download failed: {rec.get('error')}")
                        return
            asyncio.create_task(_auto_dino())
        if sam3_off:
            print("[server] CHARLIE_DISABLED is set: SAM 3 is not loaded.")
            state["charlie"] = None
        elif models_mgr.is_downloaded("sam3"):
            asyncio.create_task(_bg_load_charlie())
        elif models_mgr.hf_token():
            # With a token on file, SAM3 downloads in the background and loads when done.
            state["charlie"] = None

            async def _auto_sam3() -> None:
                print("[server] SAM 3 weights missing; downloading with the saved token...")
                await asyncio.to_thread(models_mgr.start_download, "sam3")
                while True:
                    await asyncio.sleep(5)
                    with models_mgr._DL_LOCK:
                        rec = dict(models_mgr._DOWNLOADS.get("sam3") or {})
                    if rec.get("status") == "error":
                        print(f"[server] SAM 3 download failed: {rec.get('error')}")
                        return
                    if rec.get("status") == "done":
                        break
                await _bg_load_charlie()
            asyncio.create_task(_auto_sam3())
        else:
            state["charlie"] = None
            print(
                "[server] SAM 3 needs a Hugging Face token: add it in Settings, "
                "then SAM 3 downloads and loads"
            )
    else:
        state["charlie"] = None

    state["jobs"].register_runner("label_charlie", _run_label_charlie_job)
    state["jobs"].register_runner("train_yolox", _run_train_yolox_job)
    state["jobs"].register_runner("label_model", _run_label_model_job)
    state["jobs"].register_runner("purge_label", _run_purge_label_job)
    state["jobs"].register_runner("augment_generate", _gpu_job_guarded(_run_augment_generate_job))

    # Finished jobs from the event log, so a poll after a restart finds them.
    try:
        past = audit.list_events("job", limit=500)
        loaded = state["jobs"].hydrate_from_audit(past)
        if loaded:
            print(f"[jobs] hydrated {loaded} past jobs from events.db")
    except Exception as e:
        print(f"[jobs] audit hydrate failed: {e}")

    state["jobs"].start_worker()

    # Respawns exited worker tasks; start_worker() keeps live ones.
    async def _worker_watchdog() -> None:
        while True:
            await asyncio.sleep(5)
            wts = state["jobs"].worker_tasks
            live = [t for t in wts if not t.done()]
            if len(live) < len(wts):
                print(f"[jobs] {len(wts) - len(live)} of {len(wts)} workers exited; restarting")
                state["jobs"].start_worker()
    state["_worker_watchdog"] = asyncio.create_task(_worker_watchdog())

    print(f"[server] workspace: {workspace.dir()}")
    print("[server] ready, job worker started.")
    _size_image_cache()
    yield
    # A compile runs in its own process group, which the desktop's stop
    # signal for the engine does not reach.
    neuron6_verify.stop_running_compilers()

app = FastAPI(
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

# Localhost origins only: this process or a dev server on :3000.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)
# Only the loopback names: a web page that rebinds its own domain to
# 127.0.0.1 sends that domain as Host and is refused.
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])

_LOCAL_ORIGIN = re.compile(r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$")


class _LocalPagesOnly:
    """Refuses a POST, PUT, PATCH or DELETE that a page on another site sent:
    CORS stops such a page reading the answer, not a form post or a POST
    without a body reaching the engine. Requests without an Origin, from
    scripts and the CLI, pass. Every response carries nosniff."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        if scope["method"] not in ("GET", "HEAD", "OPTIONS"):
            origin = dict(scope["headers"]).get(b"origin")
            if origin is not None and not _LOCAL_ORIGIN.match(origin.decode("latin-1")):
                refused = FastJSONResponse({"detail": "cross-site request refused"}, status_code=403)
                await refused(scope, receive, send)
                return

        async def send_nosniff(message):
            if message["type"] == "http.response.start":
                message["headers"] = [*message.get("headers", []),
                                      (b"x-content-type-options", b"nosniff")]
            await send(message)

        await self.app(scope, receive, send_nosniff)


app.add_middleware(_LocalPagesOnly)


def project_dir(project_id: str) -> Path:
    """Resolve a dataset id to its workspace folder."""
    if not project_id or "/" in project_id or ".." in project_id:
        raise HTTPException(400, f"invalid project id: {project_id}")
    try:
        return store.dataset_dir(project_id)
    except KeyError:
        raise HTTPException(404, f"unknown project: {project_id}")


def manifest_path(project_id: str) -> Path:
    """For existence checks: the manifest is split across dataset.json and annotations/."""
    return project_dir(project_id) / "dataset.json"


# Built payloads of /overview, /annotations and /dataset-stats:
# (project_id, name) -> (manifest_mtime, payload). Fresh while its mtime
# is at least the manifest's.
_PAYLOAD_CACHE: dict[tuple[str, str], tuple[float, dict]] = {}
_PAYLOAD_CACHE_LOCK = __import__("threading").Lock()


def _payload_cache_get_swr(project_id: str, name: str, disk_mtime: float) -> tuple[dict | None, bool]:
    """Stale-while-revalidate read: (None, False) when absent, (payload, True)
    when fresh, (payload, False) when stale and due a background rebuild."""
    if disk_mtime <= 0:
        return None, False
    with _PAYLOAD_CACHE_LOCK:
        entry = _PAYLOAD_CACHE.get((project_id, name))
    if entry is None:
        return None, False
    cached_mtime, payload = entry
    return payload, cached_mtime >= disk_mtime


# Per-project drop count. A rebuild started before a drop does not store its result.
_PAYLOAD_DROPS: dict[str, int] = {}


def _drop_annotation_payloads(project_id: str) -> None:
    """Forget the cached annotations after an edit, so the editor never gets a stale copy."""
    with _PAYLOAD_CACHE_LOCK:
        _PAYLOAD_DROPS[project_id] = _PAYLOAD_DROPS.get(project_id, 0) + 1
        for name in ("annotations", "annotations:imports"):
            _PAYLOAD_CACHE.pop((project_id, name), None)


def _payload_cache_put(project_id: str, name: str, disk_mtime: float, payload: dict) -> None:
    if disk_mtime <= 0:
        return
    with _PAYLOAD_CACHE_LOCK:
        _PAYLOAD_CACHE[(project_id, name)] = (disk_mtime, payload)


# (project_id, name) rebuilds in flight.
_PAYLOAD_REVALIDATE_IN_FLIGHT: set[tuple[str, str]] = set()


async def _payload_revalidate(project_id: str, name: str, builder) -> None:
    """Background rebuild for stale-while-revalidate, at most one per key.
    `builder` takes no arguments and returns the payload."""
    key = (project_id, name)
    with _PAYLOAD_CACHE_LOCK:
        if key in _PAYLOAD_REVALIDATE_IN_FLIGHT:
            return
        _PAYLOAD_REVALIDATE_IN_FLIGHT.add(key)
        drops = _PAYLOAD_DROPS.get(project_id, 0)
    try:
        mtime = _manifest_disk_mtime(project_id)
        payload = await asyncio.to_thread(builder)
        with _PAYLOAD_CACHE_LOCK:
            dropped_since = _PAYLOAD_DROPS.get(project_id, 0) != drops
        if not dropped_since:
            _payload_cache_put(project_id, name, mtime, payload)
    except Exception as e:
        print(f"[payload-cache] revalidate failed for {project_id}/{name}: {e}")
    finally:
        with _PAYLOAD_CACHE_LOCK:
            _PAYLOAD_REVALIDATE_IN_FLIGHT.discard(key)


# Read-path sidecars, rebuilt after each manifest write and served as files.

_FAST_OVERVIEW_LIMIT = 100  # must match the UI's FIRST_BATCH
_INITIAL_PAYLOAD_LIMIT = 20  # the UI's /initial first-paint slice
_SIDECAR_REFRESH_IN_FLIGHT: set[str] = set()
# Projects whose sidecar refresh is suppressed while a long job saves
# repeatedly. The job refreshes once when it ends.
_SIDECAR_REFRESH_SUPPRESSED: set[str] = set()


def _overview_sidecar_path(project_id: str) -> Path:
    return project_dir(project_id) / "overview_first100.json"


def _workspace_card_sidecar_path(project_id: str) -> Path:
    """Sidecar holding the /api/projects card payload, built on save_manifest."""
    return project_dir(project_id) / "workspace_card_v1.json"


def _import_has_labels(entry: dict) -> bool:
    """Whether an image has labels: its edited boxes once edited, else the
    labelled flag, else its detections."""
    if entry.get("editedBoxesSet"):
        edited = entry.get("editedBoxes")
        return isinstance(edited, list) and len(edited) > 0
    flag = entry.get("labelled")
    if isinstance(flag, bool):
        return flag
    return bool(entry.get("detections"))


def _import_is_labelled(entry: dict) -> bool:
    """Has labels, or is marked background: checked, with nothing to label."""
    return isinstance(entry, dict) and (bool(entry.get("background")) or _import_has_labels(entry))


# The card fields GET /api/projects returns; a sidecar written by another version can hold more.
_CARD_KEYS = ("id", "name", "createdAt", "updatedAt", "n_images", "n_labelled", "n_unlabelled",
              "n_references", "tags", "label_aliases", "labelColours", "thumbnail", "_cover_subdir",
              "hasModel", "v2", "derived", "cover_blurhash")


def _build_workspace_card_payload(project_id: str, manifest: dict) -> dict:
    """Per-project card payload; favouritedByMe, running and container are added at request time."""
    proj_id = manifest.get("id") or project_id
    display_name = manifest.get("name") or project_id

    results = manifest.get("results") or []
    n_labelled = sum(1 for r in results if not r.get("pending"))
    n_unlabelled = sum(1 for r in results if r.get("pending"))
    n_images_v2: int | None = None
    if manifest.get("v2"):
        v2_imports = manifest.get("imports") or []
        n_images_v2 = len(v2_imports)
        n_labelled = sum(1 for e in v2_imports if _import_is_labelled(e))
        n_unlabelled = max(0, n_images_v2 - n_labelled)

    cover = manifest.get("cover")
    v2_refs = manifest.get("references") or []
    v2_imps = manifest.get("imports") or []
    cover_subdir: str | None = None
    # An uploaded cover wins. It lives at the project root and is served by cover_thumb.
    if manifest.get("cover_uploaded"):
        cover = "cover_upload.jpg"
        cover_subdir = None
    elif cover:
        if any(r.get("filename") == cover for r in v2_refs):
            cover_subdir = "references"
        elif any(i.get("filename") == cover for i in v2_imps):
            cover_subdir = "images"
        elif any(r.get("image") == cover for r in results):
            cover_subdir = "images"
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
            cover_subdir = "images"
        elif v1_files:
            cover = seed.choice(v1_files)
            cover_subdir = "images"
        else:
            cover = None
            cover_subdir = None

    n_images_resolved = n_images_v2 if n_images_v2 is not None else len(results)
    return {
        "id": proj_id,
        "name": display_name,
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
        "favouritedBy": list(manifest.get("favouritedBy") or []),
        "v2": bool(manifest.get("v2", False)),
        "derived": ({"parentProjectId": (manifest.get("derived") or {}).get("parentProjectId"),
                     "parentName": (manifest.get("derived") or {}).get("parentName")}
                    if manifest.get("derived") else None),
        "cover_blurhash": manifest.get("cover_blurhash"),
    }


def _manifest_mtime_ns(project_id: str) -> int:
    """dataset.json's file time in ns, or 0 when it is missing."""
    try:
        return manifest_path(project_id).stat().st_mtime_ns
    except (HTTPException, OSError):
        return 0


def _write_sidecar(path: Path, payload: dict, stamp_ns: int) -> None:
    """Write a sidecar atomically with the file time `stamp_ns`, the manifest's
    when the payload was built: a save during the build leaves it stale."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = store.tmp_path(path)
    tmp.write_bytes(_orjson.dumps(payload) if _orjson is not None else json.dumps(payload).encode())
    os.utime(tmp, ns=(stamp_ns, stamp_ns))
    store.replace(tmp, path)


def _write_workspace_card_sidecar(project_id: str) -> None:
    try:
        stamp = _manifest_mtime_ns(project_id)
        m = load_manifest(project_id, False)
        if not m:
            return
        _write_sidecar(_workspace_card_sidecar_path(project_id),
                       _build_workspace_card_payload(project_id, m), stamp)
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
    """Rebuild a missing or stale card sidecar in the background, one at a
    time per project. Called from list_projects."""
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
    """First-paint payload: project meta, the first imports and lite dataset stats."""
    return project_dir(project_id) / "initial_first20.json"


def _write_overview_sidecar(project_id: str) -> None:
    """Write the first-100 /overview payload. Called from _kick_sidecar_refresh."""
    try:
        stamp = _manifest_mtime_ns(project_id)
        m = load_manifest(project_id, False)
        if not m:
            return
        payload = _build_overview_payload(
            project_id, m, imports_limit=_FAST_OVERVIEW_LIMIT, imports_offset=0,
        )
        _write_sidecar(_overview_sidecar_path(project_id), payload, stamp)
    except Exception as e:
        print(f"[overview-sidecar] write failed for {project_id}: {e}")


def _read_overview_sidecar(project_id: str) -> dict | None:
    """The stored first-100 overview, possibly stale, or None when missing or unreadable."""
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
    """Combine the overview and dataset-stats sidecars into the first-paint
    document, stamped with the older of their file times. Runs after both are written."""
    try:
        stats_path = _stats_sidecar_path(project_id, True)
        stamps = [p.stat().st_mtime_ns for p in (_overview_sidecar_path(project_id), stats_path)
                  if p.exists()]
        overview = _read_overview_sidecar(project_id)
        if overview is None or not stamps:
            return
        imports = (overview.get("imports") or [])[:_INITIAL_PAYLOAD_LIMIT]
        stats_payload: dict | None = None
        if stats_path.exists():
            try:
                data = _json_loads(stats_path.read_bytes())
                if isinstance(data, dict):
                    stats_payload = data
            except Exception:
                stats_payload = None
        payload = {
            # Same top-level fields as /overview.
            "id": overview.get("id"),
            "name": overview.get("name"),
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
            "dataset_type": overview.get("dataset_type"),
            "hasModel": overview.get("hasModel"),
            "references": overview.get("references") or [],
            "imports": imports,
            "imports_total": overview.get("imports_total") or 0,
            # Lite stats; the full payload comes from /dataset-stats.
            "stats": stats_payload,
        }
        _write_sidecar(_initial_sidecar_path(project_id), payload, min(stamps))
    except Exception as e:
        print(f"[initial-sidecar] write failed for {project_id}: {e}")


def _read_initial_sidecar(project_id: str) -> dict | None:
    """The stored first-paint document, possibly stale, or None if missing."""
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
    """Schedule a rebuild of the project's read-path sidecars. A no-op while
    one is in flight or the project is suppressed."""
    if project_id in _SIDECAR_REFRESH_SUPPRESSED:
        return
    if project_id in _SIDECAR_REFRESH_IN_FLIGHT:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No event loop: the sidecars rebuild on a later read.
        return
    _SIDECAR_REFRESH_IN_FLIGHT.add(project_id)

    async def _run():
        try:
            # Debounce, so a burst of saves makes one rebuild.
            await asyncio.sleep(0.15)
            # The background pool, so these multi-second rebuilds do not
            # starve request-path work on the default executor.
            await loop.run_in_executor(_BG_IMAGE_EXECUTOR, _write_overview_sidecar, project_id)
            await loop.run_in_executor(_BG_IMAGE_EXECUTOR, _persist_dataset_stats, project_id, True)
            # Must run after the two sidecars above.
            await loop.run_in_executor(_BG_IMAGE_EXECUTOR, _write_initial_sidecar, project_id)
            await loop.run_in_executor(_BG_IMAGE_EXECUTOR, _write_workspace_card_sidecar, project_id)
        except Exception as e:
            print(f"[sidecar-refresh] failed for {project_id}: {e}")
        finally:
            _SIDECAR_REFRESH_IN_FLIGHT.discard(project_id)

    loop.create_task(_run())


_MANIFEST_CACHE: dict[str, dict] = {}
# Disk mtime when the entry was cached. A newer file, written by any
# process, invalidates the entry.
_MANIFEST_CACHE_MTIME: dict[str, float] = {}
# project_id -> {import_id: entry}, kept in step with the cache under
# _MANIFEST_CACHE_LOCK.
_MANIFEST_IMPORT_INDEX: dict[str, dict[str, dict]] = {}
_MANIFEST_CACHE_LOCK = __import__("threading").Lock()

# Recently deleted ids. delete_project adds one before removing the folder
# and save_manifest refuses to write it, so a racing write cannot recreate
# the project. Ids are never reused, so FIFO eviction is safe.
_DELETED_PROJECT_IDS: OrderedDict[str, float] = OrderedDict()
_DELETED_PROJECT_IDS_MAX = 4096


def _mark_project_deleted(project_id: str) -> None:
    _projects_changed()
    with _MANIFEST_CACHE_LOCK:
        _DELETED_PROJECT_IDS[project_id] = time.time()
        _DELETED_PROJECT_IDS.move_to_end(project_id)
        while len(_DELETED_PROJECT_IDS) > _DELETED_PROJECT_IDS_MAX:
            _DELETED_PROJECT_IDS.popitem(last=False)


def _unmark_project_deleted(project_id: str) -> None:
    """Undo a tombstone when a delete fails partway."""
    with _MANIFEST_CACHE_LOCK:
        _DELETED_PROJECT_IDS.pop(project_id, None)


def _is_project_deleted(project_id: str) -> bool:
    with _MANIFEST_CACHE_LOCK:
        return project_id in _DELETED_PROJECT_IDS


def _manifest_disk_mtime(project_id: str) -> float:
    """Change stamp in seconds, or 0 if missing. store.save() rewrites
    dataset.json on every change."""
    return store.manifest_stamp(project_id)


def load_manifest(project_id: str, copy: bool = True) -> dict:
    """Read the manifest through the in-memory cache. copy=False returns the
    shared cached dict, so a caller that mutates must keep copy=True."""
    import copy as _copy

    disk_mtime = _manifest_disk_mtime(project_id)

    with _MANIFEST_CACHE_LOCK:
        cached = _MANIFEST_CACHE.get(project_id)
        cached_mtime = _MANIFEST_CACHE_MTIME.get(project_id, 0.0)
        if cached is not None and disk_mtime > 0 and disk_mtime <= cached_mtime:
            return _copy.deepcopy(cached) if copy else cached

    # A read error propagates: an empty manifest returned in its place could be
    # saved back over the dataset.
    data = store.load(project_id)
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
    return _copy.deepcopy(data) if copy else data


def save_manifest(project_id: str, manifest: dict, *, cache_by_ref: bool = False) -> None:
    import copy as _copy
    if _is_project_deleted(project_id):
        return
    # A manifest without its id (an empty one from a failed read, say) would
    # overwrite the dataset and delete its annotation files.
    if store.dataset_exists(project_id) and manifest.get("id") != project_id:
        raise RuntimeError(f"refusing to save dataset {project_id}: the manifest has no matching id")
    # A new dataset gets its folder on first save.
    if not store.dataset_exists(project_id):
        store.create_dataset_dir(
            project_id,
            str(manifest.get("name") or project_id),
            str(manifest.get("container_id") or "") or None,
        )
    # Import detections carry no embedding vectors, since nothing reads them
    # back. References keep theirs.
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
    # dataset.json and the changed per-image annotation files, written atomically.
    store.save(project_id, manifest)
    _projects_changed()
    disk_mtime = _manifest_disk_mtime(project_id)
    with _MANIFEST_CACHE_LOCK:
        # cache_by_ref skips the deepcopy: only for a caller that discards
        # the dict after saving.
        _MANIFEST_CACHE[project_id] = manifest if cache_by_ref else _copy.deepcopy(manifest)
        _MANIFEST_CACHE_MTIME[project_id] = disk_mtime
        _MANIFEST_IMPORT_INDEX[project_id] = {
            e["id"]: e
            for e in (manifest.get("imports") or [])
            if isinstance(e, dict) and "id" in e
        }
    _kick_sidecar_refresh(project_id)
    # Re-derives any child projects once edits settle.
    _kick_child_resync(project_id)


def invalidate_manifest_cache(project_id: str) -> None:
    """Drop a project from the cache. Used on delete."""
    with _MANIFEST_CACHE_LOCK:
        _MANIFEST_CACHE.pop(project_id, None)
        _MANIFEST_CACHE_MTIME.pop(project_id, None)
        _MANIFEST_IMPORT_INDEX.pop(project_id, None)


# Derived projects: a child holds per-detection crops of a parent, synced one
# way. _DERIVED_INDEX maps parent_id -> child_ids and is built on first use.
try:
    import derived as _derived_mod  # noqa: E402
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
    # Only a full scan sets _DERIVED_INDEX_BUILT.
    with _DERIVED_LOCK:
        _DERIVED_INDEX.setdefault(parent_id, set()).add(child_id)


def _ensure_derived_index() -> None:
    if not _DERIVED_INDEX_BUILT:
        _build_derived_index()


def resync_child(child_id: str) -> bool:
    """Re-derive one child from its parent, writing the child only. True on success."""
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


async def _resync_child_locked(child_id: str) -> bool:
    """resync_child under the child's write lock, with the crop work in a thread."""
    lock = await _manifest_write_lock(child_id)
    async with lock:
        return await asyncio.to_thread(resync_child, child_id)


def _resync_child_from_thread(child_id: str) -> bool:
    """_resync_child_locked for the timer and background threads."""
    loop = state.get("loop")
    if loop is None or loop.is_closed():
        return resync_child(child_id)
    return asyncio.run_coroutine_threadsafe(_resync_child_locked(child_id), loop).result()


def resync_children(parent_id: str) -> None:
    for cid in _children_of(parent_id):
        try:
            _resync_child_from_thread(cid)
        except Exception as e:
            print(f"[derived] resync of child {cid} failed: {e}", flush=True)


def _kick_child_resync(parent_id: str) -> None:
    """Re-sync a parent's children 3 s after its last edit. Never raises."""
    global _DERIVED_BUILDING
    try:
        if not _DERIVED_INDEX_BUILT:
            with _DERIVED_LOCK:
                if not _DERIVED_INDEX_BUILT and not _DERIVED_BUILDING:
                    _DERIVED_BUILDING = True
                    _threading.Thread(target=_build_derived_index, daemon=True).start()
            return  # index still building
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


# Per-project lock around the manifest read-modify-write cycle.
_MANIFEST_WRITE_LOCKS: dict[str, asyncio.Lock] = {}
_MANIFEST_WRITE_LOCKS_GUARD = asyncio.Lock()


async def _manifest_write_lock(project_id: str) -> asyncio.Lock:
    """The project's write lock, created on first use."""
    async with _MANIFEST_WRITE_LOCKS_GUARD:
        lk = _MANIFEST_WRITE_LOCKS.get(project_id)
        if lk is None:
            lk = asyncio.Lock()
            _MANIFEST_WRITE_LOCKS[project_id] = lk
        return lk


def _find_import_by_idempotency_key(project_id: str, key: str) -> dict | None:
    """The import carrying `key`, in the /imports/raw response shape, or None."""
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


def _size_image_cache() -> None:
    """Size the served-image cache at startup: 25% of the available RAM,
    between 0.5 and 8 GB, or 1 GB without psutil."""
    try:
        import psutil  # type: ignore
        avail_gb = psutil.virtual_memory().available / 1e9
        budget = int(min(8.0, max(0.5, avail_gb * 0.25)) * 1e9)
    except Exception:
        budget = 1_000_000_000
    _set_image_cache_budget(budget)
    print(f"[image-cache] budget: {budget / 1e9:.1f} GB")


def empty_manifest(name: str, project_id: str | None = None) -> dict:
    return {
        "id": project_id or _uuid.uuid4().hex,
        "name": name,
        "createdAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "updatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tags": [],
        "results": [],
        "verdicts": {},
        "editedBoxes": {},
        "cover": None,
        "hasModel": False,
        "favouritedBy": [],
        # Lower-case label -> #rrggbb colour override.
        "labelColours": {},
    }


def _replace_missing_cover(manifest: dict) -> None:
    """A cover that names no import or reference is replaced by a random
    import, else a reference, else None."""
    cover = manifest.get("cover")
    if not cover:
        return
    imports = [e.get("filename") for e in manifest.get("imports") or []
               if isinstance(e, dict) and e.get("filename")]
    refs = [e.get("filename") for e in manifest.get("references") or []
            if isinstance(e, dict) and e.get("filename")]
    if cover in imports or cover in refs:
        return
    import random
    manifest["cover"] = random.choice(imports or refs) if (imports or refs) else None
    manifest["cover_blurhash"] = None  # re-derived on next read


def _charlie_embed_detections(
    pil: "PILImage.Image",
    detections: list[dict],
) -> list:
    """Embed each detection's inpainted crop with DINOv2, setting `embedding`
    and `embed_version` in place. Returns the square crops, parallel to
    `detections`, or [] when DINOv2 is not loaded."""
    if not detections:
        return []
    import v2_dinov2 as _v2d

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
            # Neutral tile for a degenerate box; the embed-low rule rejects it.
            square = PILImage.new("RGB", (16, 16), (0, 0, 0))
        squares.append(square)

    try:
        d_vecs = _v2d.encode_images_batch(squares)
    except Exception as e:
        print(f"[label-charlie] DINOv2 encode failed: {e}")
        d_vecs = None

    for k, d in enumerate(detections):
        if d_vecs is not None and k < d_vecs.shape[0]:
            d["embedding"] = [round(float(x), 6) for x in d_vecs[k].tolist()]
            d["embed_version"] = _v2d.EMBED_VERSION

    return squares


def _charlie_box_iou(box_a, box_b) -> float:
    """IoU of two [x0, y0, x1, y1] boxes; 0.0 for degenerate input."""
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


# Tunables for Charlie's specific path, overridable by env vars.
_CHARLIE_AMBIGUOUS_MARGIN = float(os.environ.get("CHARLIE_AMBIGUOUS_MARGIN", "0.08"))
_CHARLIE_POST_FUSION_AMBIGUOUS_MARGIN = float(
    os.environ.get("CHARLIE_POST_FUSION_AMBIGUOUS_MARGIN", "0.01")
)

# Broad prompts for specific-mode labels SAM3 cannot localise. The
# reference resolver assigns the real label.
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
    """SAM3 detection for the specific pipeline. When a label finds nothing,
    a second pass runs the generic fallback prompts; its boxes overlapping a
    detection at IoU > 0.5 are dropped and the rest carry no gd_label."""
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
    # A cancelled pass returns [], which would look like every label missing.
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
        g["gd_label"] = None      # the references decide
        g["gd_variant"] = None    # every label reader falls back to this key
        g["gd_score"] = None
        g["generic"] = True
        dets.append(g)
        added += 1
    if added:
        print(
            f"[charlie/generic-fallback] {added} generic candidate(s) added "
            f"for missing label(s): {missing}"
        )
    return dets, timings


def _charlie_tiebreak(d: dict, sims: dict[str, float]) -> None:
    """Give an ambiguous verdict or detection the top label of `sims`, the
    first on a tie, in place. It stays ambiguous when the top two are closer
    than _CHARLIE_POST_FUSION_AMBIGUOUS_MARGIN."""
    top2 = {lab: float(sim) for lab, sim in sorted(sims.items(), key=lambda kv: -kv[1])[:2]}
    first, second = sorted(top2.values(), reverse=True)
    winner = max(top2, key=top2.get)
    d["pred_label"] = winner
    d["pred_source"] = "embed-vlm"
    d["vlm_action"] = "tiebreak"
    d["embed_sim_for_label"] = round(top2[winner], 4)
    d["fused_scores"] = {lab: round(sim, 4) for lab, sim in top2.items()}
    d["fused_margin"] = round(first - second, 4)
    d["ambiguous"] = first - second < _CHARLIE_POST_FUSION_AMBIGUOUS_MARGIN


# NMS IoU threshold. SAM3 prompts each label separately, so one object can
# come back under two labels.
_CHARLIE_NMS_IOU = float(os.environ.get("CHARLIE_SPECIFIC_NMS_IOU", "0.6"))

# Same-label boxes whose intersection / smaller area reaches this merge
# regardless of IoU.
_CHARLIE_NMS_CONTAINMENT = float(os.environ.get("CHARLIE_NMS_CONTAINMENT", "0.7"))


def _charlie_boxes_should_merge(box_a, box_b) -> bool:
    """True when two same-label boxes likely cover one object: high IoU or containment."""
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
    """Same-label NMS for Charlie's general path: drops the lower-scoring box
    of each overlapping same-label pair. Different labels are never suppressed."""
    if len(detections) <= 1:
        return detections

    def _label(d: dict) -> str:
        lab = (d.get("pred_label") or d.get("gd_label") or "").strip().lower()
        return lab

    def _score(d: dict) -> float:
        # SAM3's mask score.
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
        f"of {len(detections)} (IoU >= {_CHARLIE_NMS_IOU} or containment >= {_CHARLIE_NMS_CONTAINMENT})"
    )
    return [d for i, d in enumerate(detections) if i not in suppressed]


def _charlie_nms_post_resolve(detections: list[dict], iou_thresh: float = _CHARLIE_NMS_IOU) -> list[dict]:
    """Post-resolution NMS for Charlie's specific path, across labels.
    Ranked by embed_sim_for_label, then gd_score; ties keep list order."""
    if len(detections) <= 1:
        return detections

    def _score(d: dict) -> tuple[float, float]:
        return (
            float(d.get("embed_sim_for_label") or 0.0),
            float(d.get("gd_score") or 0.0),
        )

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
        f"{len(detections)} detection(s) (IoU >= {_CHARLIE_NMS_IOU})"
    )
    return [d for i, d in enumerate(detections) if i in keep]


def _charlie_classify_general_sam3(
    image_pil: "PILImage.Image",
    box_xyxy: list[float],
    candidate_labels: list[str],
) -> tuple[dict | None, dict[str, float]]:
    """Run SAM3 once per candidate label and pick the one whose detection
    best overlaps the box. For general projects, which have no references."""
    charlie = state.get("charlie")
    if charlie is None:
        return None, {}
    try:
        label, score, timings = charlie.classify_box(image_pil, box_xyxy, candidate_labels)
    except Exception as e:
        print(f"[charlie/classify-general] SAM 3 classify failed: {e}")
        return None, {}
    if not label:
        return None, {"sam3_classify_ms": float(timings.get("total_ms", 0.0))}
    verdict = {
        "pred_label": label,
        "pred_source": "vlm",  # the nearest value the UI knows
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
    """Embed one box crop and run the specific resolver against the project's
    reference centroids. Returns (verdict or None, timings). Used by
    classify_box."""
    import v2_dinov2 as _v2d
    if not _v2d.is_loaded():
        return None, {}

    timings: dict[str, float] = {}
    t_embed = time.perf_counter()

    # The same crop procedure as the references.
    clean = _v2d.inpaint_bbox_crop(image_pil, box_xyxy, mask_polys or [])
    square = _v2d.center_square_crop(clean)
    d_vecs = _v2d.encode_images_batch([square])
    emb_raw = (
        [round(float(x), 6) for x in d_vecs[0].tolist()]
        if d_vecs is not None and d_vecs.shape[0] > 0 else []
    )
    timings["embed_ms"] = round((time.perf_counter() - t_embed) * 1000.0, 1)

    if not emb_raw:
        return None, timings

    try:
        by_label = _v2_load_or_backfill_reference_embeddings(project_id)
    except Exception as e:
        print(f"[charlie/resolve] reference load failed: {e}")
        return None, timings
    return _charlie_resolve_from_refs(
        emb_raw, square, by_label, candidate_labels,
        project_id=project_id, timings=timings,
    )


def _charlie_resolve_from_refs(
    emb_raw,
    square,
    by_label,
    candidate_labels,
    *,
    project_id: str | None = None,
    timings: dict | None = None,
    refs_patch: dict | None = None,
):
    """Score one embedded box crop against loaded reference embeddings with
    the specific resolver. Patch tokens come from refs_patch, keyed by
    lower-case label, else from disk when project_id is set; without them
    scoring is pooled."""
    import v2_dinov2 as _v2d
    if timings is None:
        timings = {}

    refs_dino_arr = _v2_stack_refs(by_label)
    if not refs_dino_arr:
        return None, timings

    class_thresholds_local = _v2_compute_class_thresholds(refs_dino_arr) or None

    if refs_patch is None:
        refs_patch = {}
        if project_id and _v2_patch_match_enabled():
            try:
                refs_patch = _v2_load_or_backfill_patch_tokens(project_id)
            except Exception as e:
                print(f"[charlie/resolve] patch refs load failed: {e}")
                refs_patch = {}

    q_patch_dino = (None, None)
    try:
        if refs_patch and square is not None:
            q_patch_dino = _v2d.encode_image_patches(square)
    except Exception as e:
        print(f"[charlie/resolve] dino patch encode failed: {e}")

    label_display_local = {t.lower(): t for t in candidate_labels}
    try:
        verdict = _v2_resolve_label_specific(
            emb_raw,
            None,
            refs_dino_arr,
            label_display_local,
            class_thresholds=class_thresholds_local,
            query_patch_tokens=q_patch_dino[0],
            query_patch_fg=q_patch_dino[1],
            refs_by_label_patches=refs_patch or None,
            ambiguous_margin=_CHARLIE_AMBIGUOUS_MARGIN,
        )
    except Exception as e:
        print(f"[charlie/resolve] resolver failed: {e}")
        return None, timings
    return verdict, timings


async def _run_label_charlie_job(job, emit, cancel_event):
    """Run _run_label_charlie_job_impl with sidecar refresh suppressed and
    background embeddings paused, then refresh the sidecars once."""
    _SIDECAR_REFRESH_SUPPRESSED.add(job.project)
    _pause_bg_embeddings()
    try:
        return await _run_label_charlie_job_impl(job, emit, cancel_event)
    finally:
        _resume_bg_embeddings()
        _SIDECAR_REFRESH_SUPPRESSED.discard(job.project)
        try:
            _kick_sidecar_refresh(job.project)
        except Exception as e:
            print(f"[label-charlie] final sidecar refresh failed: {e}")


async def _run_label_charlie_job_impl(job, emit, cancel_event):
    """Label each unlabelled import with the SAM3 Charlie pipeline, writing
    its detections back and emitting one progress event per image."""
    proj = project_dir(job.project)
    if not proj.exists():
        raise HTTPException(404, "dataset not found")

    charlie = state.get("charlie")
    if charlie is None:
        raise RuntimeError("SAM 3 is not loaded, so labelling cannot run")

    _t_job_start = time.perf_counter()
    def _stage(label: str) -> None:
        dt = (time.perf_counter() - _t_job_start) * 1000.0
        print(f"[label-charlie] {job.project} +{dt:.0f}ms {label}")

    _stage("setup: load manifest")
    # Read only, so copy=False.
    manifest = load_manifest(job.project, copy=False) or {}
    tags = list(manifest.get("tags") or [])
    if not tags:
        raise RuntimeError("no labels defined for this project")


    # Optional per-run SAM3 settings. None, or a value outside [0, 1], uses
    # the pipeline_charlie default.
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
    # tile_size None is the pipeline default, SAM3_TARGET_LONGEST_EDGE.
    charlie_tile_native = bool(sam3_params.get("tile_native"))
    try:
        charlie_tile_size = int(sam3_params.get("tile_size") or 0) or None
    except (TypeError, ValueError):
        charlie_tile_size = None
    if charlie_tile_size is not None and not (256 <= charlie_tile_size <= 8192):
        charlie_tile_size = None

    _stage("setup: classifying dataset type")
    # Specific: SAM3 finds boxes and reference embeddings pick labels.
    # General: the SAM3 prompt is the label.
    dataset_type = _dataset_type(job.project)["type"]

    # Resolver context, specific only, loaded once per job.
    refs_by_label_arr: dict = {}
    class_thresholds: dict | None = None
    label_display: dict[str, str] = {t.lower(): t for t in tags}
    if dataset_type == "specific":
        try:
            _stage("setup: loading reference embeddings")
            # A backfill runs DINOv2, so this runs in a thread under the GPU gate.
            async with state["gpu_lock"]:
                by_label = await asyncio.to_thread(
                    _v2_load_or_backfill_reference_embeddings, job.project)
            # Drops references of deleted labels.
            by_label = {k: v for k, v in by_label.items() if str(k).lower() in label_display}
            _stage(f"setup: reference embeddings ready ({sum(len(v) for v in by_label.values())})")
            refs_by_label_arr = _v2_stack_refs(by_label)

            # Leave-one-out thresholds; the job scores pooled embeddings, without patch matching.
            class_thresholds = _v2_compute_class_thresholds(refs_by_label_arr) or None

            print(
                f"[label-charlie] {job.id} specific resolver ready "
                f"({len(refs_by_label_arr)} classes, "
                f"ambiguous margin={_CHARLIE_AMBIGUOUS_MARGIN})"
            )
        except Exception as e:
            print(f"[label-charlie] reference embedding setup failed: {e}")
            refs_by_label_arr = {}
            class_thresholds = None
    print(f"[label-charlie] {job.id} dataset_type={dataset_type}")

    imports_dir = proj / "images"
    pending: list[dict] = []
    # force_relabel reprocesses every image. Edited boxes are kept.
    force_relabel = bool((job.params or {}).get("force_relabel"))
    for entry in (manifest.get("imports") or []):
        # Background images are skipped, even on a relabel.
        if not isinstance(entry, dict) or entry.get("background"):
            continue
        if force_relabel or not _import_has_labels(entry):
            pending.append(entry)
    # Newest first, the gallery's order.
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
    n_failed = 0
    first_error = ""
    loop = asyncio.get_running_loop()
    _stage(f"loop: {len(pending)} image(s) to label")

    def _load_pil_sync(p: Path) -> "PILImage.Image":
        return _open_upright(p)

    # Updates are batched and flushed every LABEL_FLUSH_EVERY images.
    pending_label_updates: list[dict] = []
    labelled_import_ids: list[str] = []
    LABEL_FLUSH_EVERY = 10

    async def _flush_label_updates(mark_run: bool = False) -> None:
        if not pending_label_updates:
            return
        updates = list(pending_label_updates)
        lock = await _manifest_write_lock(job.project)
        async with lock:
            # copy=False is safe: the mutation and the save happen under the write lock.
            mm = load_manifest(job.project, copy=False) or {}
            by_id = {e.get("id"): e for e in (mm.get("imports") or []) if isinstance(e, dict)}
            for upd in updates:
                e = by_id.get(upd["id"])
                if e is None:
                    continue
                # An explicitly empty edit would hide the new detections, so it is cleared.
                prior_edited = e.get("editedBoxes")
                if (
                    e.get("editedBoxesSet")
                    and isinstance(prior_edited, list)
                    and len(prior_edited) == 0
                ):
                    e.pop("editedBoxes", None)
                    e.pop("editedBoxesSet", None)
                # Stored without embedding vectors: a relabel re-embeds fresh crops.
                e["detections"] = [
                    _strip_embedding(d) if isinstance(d, dict) else d
                    for d in upd["detections"]
                ]
                e["timings"] = upd["timings"]
                e["labelled"] = True
                # Epoch ms; the labelled preview's ?v= cachebuster.
                e["labelledAt"] = int(time.time() * 1000)
                # Tags this entry was searched with, for the UI's fresh-label check.
                e["labelsRun"] = list(tags)
                if not e.get("width") and upd.get("width"):
                    e["width"] = upd["width"]
                if not e.get("height") and upd.get("height"):
                    e["height"] = upd["height"]
                if upd["id"]:
                    labelled_import_ids.append(upd["id"])
            # Only a completed job records labelsLastRun and settingsLastRun;
            # a cancelled one still flushes.
            if mark_run:
                mm["labelsLastRun"] = list(tags)
                # The thresholds are stored only when all three are set;
                # tile_native always.
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
                    # Absent means the default.
                    slr.pop("tile_size", None)
                mm["settingsLastRun"] = slr
            mm["updatedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            # Off the event loop, since the multi-MB write is synchronous.
            # cache_by_ref is safe: mm is discarded after the save.
            await loop.run_in_executor(
                None, lambda: save_manifest(job.project, mm, cache_by_ref=True)
            )
        # Cleared only after the write, so a failed save keeps the updates.
        pending_label_updates.clear()

    # Decodes run one image ahead, keyed by import id so a skipped image's
    # pixels are never used for the next one.
    _prefetch: "tuple[str, asyncio.Future] | None" = None
    for _pe in pending:
        _pfn = _pe.get("filename")
        if _pfn and _pe.get("id") and (imports_dir / _pfn).exists():
            _prefetch = (_pe["id"], loop.run_in_executor(None, _load_pil_sync, imports_dir / _pfn))
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

        # The image in flight; index counts finished images.
        await emit("progress", {
            "index": n_done,
            "total": len(pending),
            "image": filename,
        })

        try:
            if _prefetch is not None and _prefetch[0] == import_id:
                pil = await _prefetch[1]
            else:
                if _prefetch is not None:
                    _prefetch[1].cancel()  # decoded for an image that was skipped
                pil = await loop.run_in_executor(None, _load_pil_sync, path)
        except Exception as e:
            print(f"[label-charlie] {import_id}: failed to decode: {e}")
            n_failed += 1
            first_error = first_error or f"could not read {filename}: {e}"
            _prefetch = None
            continue

        # Start the next decode before taking the GPU lock.
        _prefetch = None
        for _ni in range(_pending_idx + 1, len(pending)):
            _ne = pending[_ni]
            _nfn = _ne.get("filename")
            if not _nfn or not _ne.get("id"):
                continue
            _np = imports_dir / _nfn
            if _np.exists():
                _prefetch = (_ne["id"], loop.run_in_executor(None, _load_pil_sync, _np))
                break

        # SAM3 and the embedding pass share one hold of the GPU lock.
        try:
            async with state["gpu_lock"]:
                # Only specific datasets use the generic fallback: in a general
                # one the SAM3 prompt is the label.
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
            n_failed += 1
            first_error = first_error or f"labelling {filename} failed: {e}"
            continue

        # A cancelled pass returns no detections: stop without saving, so the
        # image stays as it was. A pass that completed as the cancel landed saves.
        if (timings or {}).get("cancelled"):
            break

        if dataset_type == "specific" and detections and refs_by_label_arr:
            t_resolve = time.perf_counter()
            for k_det, d in enumerate(detections):
                emb_raw = d.get("embedding") or []
                if not emb_raw:
                    # No embedding: SAM3's label stays.
                    continue
                try:
                    verdict = _v2_resolve_label_specific(
                        emb_raw,
                        d.get("gd_label"),
                        refs_by_label_arr,
                        label_display,
                        class_thresholds=class_thresholds or None,
                        ambiguous_margin=_CHARLIE_AMBIGUOUS_MARGIN,
                    )
                except Exception as e:
                    print(f"[label-charlie] resolver failed for one detection: {e}")
                    continue
                # The verdict's fields a detection keeps.
                for k_v in (
                    "pred_label", "pred_source",
                    "embed_nearest_label", "embed_nearest_sim",
                    "embed_sim_for_label", "embed_margin",
                    "rejected", "reject_reason",
                    "ambiguous", "vlm_action",
                    "patch_match_used",
                ):
                    if k_v in verdict:
                        d[k_v] = verdict[k_v]
                if verdict.get("sims") is not None:
                    d["embed_sims"] = verdict["sims"]
                if verdict.get("sims_dino") is not None:
                    d["embed_sims_dino"] = verdict["sims_dino"]
            timings["resolve_ms"] = round((time.perf_counter() - t_resolve) * 1000.0, 1)

            for d in detections:
                if d.get("ambiguous"):
                    _charlie_tiebreak(d, d["embed_sims"])

            # Cross-label NMS, after the tiebreak so it sees the final labels.
            t_nms = time.perf_counter()
            detections = _charlie_nms_post_resolve(detections)
            timings["nms_ms"] = round((time.perf_counter() - t_nms) * 1000.0, 1)

            # Specific datasets never reject: a rejected detection is kept
            # as ambiguous. Runs after NMS, so no rejected duplicate returns.
            for d in detections:
                if d.get("rejected"):
                    d["rejected"] = False
                    d["ambiguous"] = True
                    if not d.get("pred_label"):
                        d["pred_label"] = d.get("embed_nearest_label")

        # A generic box with no resolved label has nothing to store.
        detections = [d for d in detections if not (d.get("generic") and not d.get("pred_label"))]

        if dataset_type != "specific" and detections:
            t_nms_g = time.perf_counter()
            detections = _charlie_nms_same_label(detections)
            if len(tags) > 1:
                # Cross-label NMS at IoU 0.9 only, so distinct overlapping
                # classes both survive.
                detections = _charlie_nms_post_resolve(detections, iou_thresh=0.9)
            timings["nms_ms"] = round((time.perf_counter() - t_nms_g) * 1000.0, 1)

        pending_label_updates.append({
            "id": import_id,
            "detections": detections,
            "timings": timings,
            "width": pil.size[0],
            "height": pil.size[1],
        })
        if len(pending_label_updates) >= LABEL_FLUSH_EVERY:
            await _flush_label_updates()

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
        # image None: nothing in flight between images.
        await emit("progress", {
            "index": n_done,
            "total": len(pending),
            "image": None,
            "n_detections": len(detections),
        })
        # Lets other requests run between images.
        await asyncio.sleep(0)

    await _flush_label_updates(mark_run=not cancel_event.is_set())

    # The cached payloads would otherwise serve the pre-completion snapshot.
    try:
        _invalidate_project_payloads(job.project)
    except Exception as e:
        print(f"[label-charlie] payload invalidation failed: {e}")

    # A run in which every image failed does not report done.
    if n_failed and not n_done and not cancel_event.is_set():
        raise RuntimeError(f"no image could be labelled ({n_failed} failed): {first_error}")
    return {"labelled": n_done, "failed": n_failed}


_TRAIN_IMGSZ = (128, 192, 256, 320, 416)


_TRAIN_SIZES = ("nano", "tiny")


def _fmt_eta(seconds: float) -> str:
    s = int(round(seconds))
    if s < 90:
        return f"{max(1, s)} s"
    m = int(round(s / 60))
    if m < 90:
        return f"{m} min"
    return f"{m // 60} h {m % 60:02d} min"


def _train_cfg_from_params(params: dict, device: str):
    """Job params -> TrainConfig, each field clamped to what the UI offers.
    On resume the checkpoint's own settings win."""
    import train_yolox as _ty

    def _int(k, d, lo, hi):
        raw = params.get(k, d)
        if isinstance(raw, bool) or raw in (None, ""):
            raw = d
        try:
            v = int(raw)
        except (TypeError, ValueError):
            v = d
        return max(lo, min(hi, v))

    epochs = _int("epochs", 50, 1, 500)
    imgsz = _int("imgsz", 256, 32, 4096)
    if imgsz not in _TRAIN_IMGSZ:
        imgsz = 256
    size = str(params.get("size") or "nano").strip().lower()
    if size not in _TRAIN_SIZES:
        size = "nano"
    try:
        split = float(params.get("split", 0.8))
    except (TypeError, ValueError):
        split = 0.8
    if not (0.5 <= split <= 0.95):
        split = 0.8
    aug = params.get("augmentations", True)
    if aug is None or (isinstance(aug, str) and not aug.strip()):
        aug = True
    elif isinstance(aug, str):
        aug = aug.strip().lower() not in ("0", "false", "no", "off")
    else:
        aug = bool(aug)

    def _run_id(k):
        v = str(params.get(k) or "").strip()
        return v if _ty.RUN_ID_RE.fullmatch(v) else None

    conf = params.get("conf")
    try:
        conf = float(conf) if conf not in (None, "") else None
    except (TypeError, ValueError):
        conf = None
    if conf is not None and not (0.01 <= conf <= 0.99):
        conf = None
    eval_project = str(params.get("evalProject") or "").strip() or None
    return _ty.TrainConfig(
        epochs=epochs, imgsz=imgsz, size=size, train_split=split, device=device,
        use_augmentations=aug, init_run=_run_id("initRun"), eval_project=eval_project,
        resume_run=_run_id("resumeRun"), conf_thres=conf,
    )


def _second_gpu(engine_dev: str) -> str | None:
    """The CUDA device with the most free memory other than the labelling one, or None."""
    try:
        if not engine_dev.startswith("cuda") or torch.cuda.device_count() < 2:
            return None
        own = int(engine_dev.split(":")[1]) if ":" in engine_dev else 0
        best, best_free = None, -1
        for i in range(torch.cuda.device_count()):
            if i == own:
                continue
            try:
                free, _total = torch.cuda.mem_get_info(i)
            except Exception:  # noqa: BLE001
                continue
            if free > best_free:
                best, best_free = i, free
        return f"cuda:{best}" if best is not None else None
    except Exception:  # noqa: BLE001
        return None


def _train_job_active(project_id: str) -> bool:
    return any(
        j.project == project_id and j.kind == "train_yolox" and j.status in ("queued", "running")
        for j in state["jobs"].jobs.values()
    )


def _label_model_targets(manifest: dict, overwrite: bool) -> list[dict]:
    """Imports a trained model may label: never hand-edited ones, and
    without overwrite only those with no boxes."""
    out = []
    for e in manifest.get("imports") or []:
        if not isinstance(e, dict) or not e.get("filename"):
            continue
        if e.get("editedBoxesSet"):
            continue
        has_auto = bool(e.get("detections")) or e.get("labelled") is True
        if has_auto and not overwrite:
            continue
        out.append(e)
    return out


async def _run_train_yolox_job(job, emit, cancel_event):
    """YOLOX-Nano training and Neuro N6 export. On the labelling device it
    holds the GPU gate and hands it over between steps; on another device
    it trains outside the gate."""
    import train_yolox as _ty

    params = job.params or {}

    def _norm_dev(s: str) -> str:
        # "cuda" and "cuda:0" are the same card.
        try:
            d = torch.device(s)
        except (RuntimeError, TypeError, ValueError):
            return s
        if d.type == "cuda":
            return f"cuda:{d.index if d.index is not None else 0}"
        return d.type

    engine_dev = _norm_dev(str(state.get("device") or "cpu"))
    train_dev = _norm_dev(str(params.get("device") or "").strip() or _second_gpu(engine_dev) or engine_dev)
    if train_dev == "cpu" and _configured_device() != "cpu":
        raise RuntimeError(
            "no GPU available for training (switch Compute to CPU in Settings to force)"
        )
    cfg = _train_cfg_from_params(params, train_dev)
    loop = asyncio.get_running_loop()

    def prog(p: dict) -> None:
        phase = p.get("phase") or ""
        eta = p.get("eta_s")
        if isinstance(eta, (int, float)) and eta > 0:
            phase = f"{phase} · {_fmt_eta(eta)} left" if phase else f"{_fmt_eta(eta)} left"
        fut = asyncio.run_coroutine_threadsafe(
            emit("progress", {
                "index": p.get("index"),
                "total": p.get("total"),
                "image": phase,
            }),
            loop,
        )
        try:
            fut.result(timeout=5)
        except Exception:
            pass  # progress is best-effort

    gate = state["gpu_lock"]
    shared = train_dev == engine_dev

    def gpu_turn() -> None:
        # Called by the trainer's thread between two units of GPU work.
        if not gate.contended():
            return
        fut = asyncio.run_coroutine_threadsafe(gate.hand_over(), loop)
        while True:
            try:
                fut.result(timeout=1)
                return
            except TimeoutError:
                if not loop.is_running():  # the engine is shutting down
                    raise _ty.Cancelled() from None

    async def _go():
        return await asyncio.to_thread(_ty.train, job.project, cfg, prog, cancel_event,
                                       gpu_turn if shared else None)

    try:
        if shared:
            async with gate:
                report = await _go()
        else:
            report = await _go()
    except _ty.Cancelled:
        return {"cancelled": True}
    print(f"[train-yolox] {job.project}: AP50 fp32={report.get('ap50_fp32')} "
          f"int8={report.get('ap50_int8')} run={report.get('run')}")
    await _sync_has_model(job.project)
    return {
        "run": report.get("run"), "model": report.get("model"), "size": report.get("size"),
        "ap50_fp32": report.get("ap50_fp32"), "ap50_int8": report.get("ap50_int8"),
        "threshold": (report.get("threshold") or {}).get("conf"),
    }


def _has_exported_run(project_id: str) -> bool:
    """Whether any of the dataset's training runs has an export."""
    try:
        return any((project_dir(project_id) / "models").glob("*/*/export.zip"))
    except (OSError, KeyError):
        return False


async def _sync_has_model(project_id: str) -> None:
    """Sets the dataset's hasModel to whether a run has an export. A dataset
    deleted meanwhile is left alone; the run it trained is not failed."""
    try:
        want = await asyncio.to_thread(_has_exported_run, project_id)
        lock = await _manifest_write_lock(project_id)
        async with lock:
            m = load_manifest(project_id)
            if not m or bool(m.get("hasModel")) == want:
                return
            m["hasModel"] = want
            save_manifest(project_id, m)
    except Exception as e:  # noqa: BLE001
        print(f"[train-yolox] {project_id}: model badge not updated: {e}")


async def _run_label_model_job(job, emit, cancel_event):
    """Pre-label images with a model trained on this dataset: the run's int8
    ONNX on the CPU, at its calibrated threshold unless params.conf
    overrides it. Hand-edited images are never touched; auto-labelled ones
    only with params.overwrite."""
    _SIDECAR_REFRESH_SUPPRESSED.add(job.project)
    try:
        return await _run_label_model_job_impl(job, emit, cancel_event)
    finally:
        _SIDECAR_REFRESH_SUPPRESSED.discard(job.project)
        try:
            _kick_sidecar_refresh(job.project)
        except Exception as e:  # noqa: BLE001
            print(f"[label-model] final sidecar refresh failed: {e}")


async def _run_label_model_job_impl(job, emit, cancel_event):
    import train_yolox as _ty

    params = job.params or {}
    run_id = str(params.get("run") or "").strip()
    run_dir = _ty.find_run_dir(job.project, run_id)
    if run_dir is None:
        raise RuntimeError("model run not found")
    try:
        report = _json_loads((run_dir / "report.json").read_bytes())
    except Exception:
        raise RuntimeError("that run has no report (training did not finish)")
    onnx_path = next(iter(sorted(run_dir.glob("*_int8.onnx"))), None)
    if onnx_path is None or report.get("exportRejected"):
        raise RuntimeError("that run has no exported model")
    classes = [str(c) for c in (report.get("classes") or [])]
    if not classes:
        classes = list((report.get("per_class_int8") or {}).keys())
    if not classes:
        raise RuntimeError("the run's report lists no classes")
    try:
        imgsz = int(report.get("imgsz") or 256)
    except (TypeError, ValueError):
        imgsz = 256
    conf_default = float((report.get("threshold") or {}).get("conf") or 0.25)
    try:
        conf = float(params.get("conf")) if params.get("conf") not in (None, "") else conf_default
    except (TypeError, ValueError):
        conf = conf_default
    conf = max(0.01, min(0.99, conf))
    nms = 0.45
    overwrite = bool(params.get("overwrite"))

    manifest = load_manifest(job.project, copy=False) or {}
    targets = _label_model_targets(manifest, overwrite)
    ids = [(str(e.get("id") or ""), str(e.get("filename") or "")) for e in targets]
    job.n_images = len(ids)
    if not ids:
        return {"labelled": 0, "run": run_id, "conf": conf}
    sess = await asyncio.to_thread(_ty.open_onnx, onnx_path)
    images_root = project_dir(job.project) / "images"
    pending: list[dict] = []
    done = 0
    now_tags = [t for t in classes]

    async def flush() -> None:
        nonlocal pending
        if not pending:
            return
        updates, pending = pending, []
        lock = await _manifest_write_lock(job.project)
        async with lock:
            mm = load_manifest(job.project, copy=False) or {}
            by_id = {e.get("id"): e for e in (mm.get("imports") or []) if isinstance(e, dict)}
            tags = [str(t) for t in (mm.get("tags") or [])]
            lowered = {t.strip().lower() for t in tags}
            for c in classes:
                if c not in lowered:
                    tags.append(c)
                    lowered.add(c)
            mm["tags"] = tags
            mm["updatedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            for upd in updates:
                e = by_id.get(upd["id"])
                if e is None or e.get("editedBoxesSet"):
                    continue  # edited by hand since the job started
                e["detections"] = upd["detections"]
                e["labelled"] = True
                e["labelledAt"] = int(time.time() * 1000)
                e["labelsRun"] = list(now_tags)
                e["labelledBy"] = {"model": run_id, "conf": conf}
                if not e.get("width") and upd.get("width"):
                    e["width"] = upd["width"]
                if not e.get("height") and upd.get("height"):
                    e["height"] = upd["height"]
            await asyncio.get_running_loop().run_in_executor(
                None, lambda: save_manifest(job.project, mm, cache_by_ref=True))

    def _infer(path):
        img = _ty._read_rgb(path)
        if img is None:
            return None
        rows = _ty.infer_image(sess, img, imgsz, len(classes), conf, nms)
        return img.shape[1], img.shape[0], rows

    for i, (iid, fname) in enumerate(ids):
        if cancel_event.is_set():
            break
        res = await asyncio.to_thread(_infer, images_root / fname)
        if res is None:
            continue
        w, h, rows = res
        dets = []
        for r in rows:
            ci = int(r[5])
            if not (0 <= ci < len(classes)):
                continue
            name = classes[ci]
            sc = round(float(r[4]), 4)
            bx = [round(float(v), 2) for v in r[:4]]
            # The editor reads pred_label, gd_label and gd_score; exports and
            # the trainer read label, box_xyxy and box.
            dets.append({
                "id": f"pm_{secrets.token_hex(4)}",
                "label": name, "pred_label": name, "gd_label": name,
                "score": sc, "gd_score": sc,
                "box": bx, "box_xyxy": bx,
                "mask": None,
                "source": "pixelkit-model",
                "modelRun": run_id,
            })
        pending.append({"id": iid, "detections": dets, "width": w, "height": h})
        done += 1
        await emit("progress", {"index": i + 1, "total": len(ids), "image": fname})
        if len(pending) >= 10:
            await flush()
    await flush()
    _invalidate_project_payloads(job.project)
    return {"labelled": done, "run": run_id, "conf": conf, "cancelled": cancel_event.is_set()}


async def _run_purge_label_job(job, emit, cancel_event):
    """Remove one label from a project: each import's detections and
    editedBoxes, under the write lock per image, then the project-level fields."""
    proj = project_dir(job.project)
    if not proj.exists():
        raise RuntimeError("dataset not found")

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
        # The key depends on which pipeline wrote the detection.
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
                    # A missing editedBoxes stays missing: an empty list
                    # means the image was cleared by hand.
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
                        if had_edited_field:
                            target_imp["editedBoxes"] = new_edited
                            target_imp["editedBoxesSet"] = True
                            target_imp["editedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                        m_now["updatedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                        save_manifest(job.project, m_now)
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

    # Project-level fields, under one lock hold.
    async with write_lock:
        m_now = load_manifest(job.project) or {}
        tags = m_now.get("tags") or []
        m_now["tags"] = [t for t in tags if str(t).strip().lower() != target_key]
        # A reference only about this label is dropped; its picture stays
        # on disk for snapshot restores.
        kept_refs = []
        for ref in m_now.get("references") or []:
            if not isinstance(ref, dict):
                continue
            dets = [d for d in (ref.get("detections") or [])
                    if not (isinstance(d, dict) and _label_of(d) == target_key)]
            if str(ref.get("label") or "").strip().lower() == target_key:
                if not dets:
                    continue
                ref["label"] = _box_label(dets[0])
            ref["detections"] = dets
            kept_refs.append(ref)
        if "references" in m_now:
            m_now["references"] = kept_refs
        _relabel_augmentations(job.project, lambda lab: None if lab.strip().lower() == target_key else lab)
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
    # VRAM in GB on cuda; None elsewhere or on a driver error.
    vram = None
    dev = state.get("device") or ""
    if dev.startswith("cuda") and torch.cuda.is_available():
        try:
            free_b, total_b = torch.cuda.mem_get_info(torch.device(dev))
            vram = {
                "usedGb": round((total_b - free_b) / 1024**3, 1),
                "totalGb": round(total_b / 1024**3, 1),
            }
        except Exception:
            vram = None
    return {
        "ok": True,
        # The desktop shell adopts a server on this port only with this marker.
        "app": "pixelkit",
        "version": PK_VERSION,
        "device": state["device"],
        "vram": vram,
        "model_loaded": state.get("charlie") is not None,
    }


def _charlie_reference_detections(image_pil, tags) -> "list[dict] | None":
    """Run SAM3 on a reference image. Returns [{label, score, box, mask}] in
    image_pil's coordinates, or None when SAM3 is not loaded."""
    charlie = state.get("charlie")
    if charlie is None:
        return None
    dets_raw, _timings = charlie.segment_labels(image_pil, list(tags), include_crops=False)
    # Cross-label NMS: one object can match several label prompts.
    if len(list(tags)) > 1 and len(dets_raw) > 1:
        dets_raw = _charlie_nms_post_resolve(dets_raw)
    out: list[dict] = []
    for d in dets_raw:
        box = d.get("box") or []
        if not (isinstance(box, list) and len(box) == 4):
            continue
        out.append({
            # The SAM3 prompt is the label; no reference resolver.
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
    force_label: str = Form(""),
):
    """Detect and segment one reference image with SAM3; returns {width, height, detections}."""
    if state.get("charlie") is None:
        raise HTTPException(503, "SAM 3 is not loaded")

    try:
        tag_list = json.loads(labels)
        if not isinstance(tag_list, list) or not all(isinstance(t, str) for t in tag_list):
            raise ValueError("labels must be a JSON array of strings")
    except Exception as e:
        raise HTTPException(400, f"invalid labels payload: {e}")
    tags = [t.strip() for t in tag_list if t and t.strip()]
    if not tags:
        data = await _read_upload(image)
        with PILImage.open(io.BytesIO(data)) as pil:
            W, H = _upright_size(pil)
        return {"width": W, "height": H, "detections": []}

    raw = await _read_upload(image)
    if not raw:
        raise HTTPException(400, "empty image upload")


    try:
        image_pil = _open_upright(raw)
    except Exception as e:
        raise HTTPException(400, f"could not decode image: {e}")

    # Detection runs at most 1280 px on the long edge; results are scaled
    # back to the original size.
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
            charlie_dets = _charlie_reference_detections(image_pil, tags)
            if charlie_dets is not None:
                print(f"[v2-ref] SAM 3 tags={tags} -> {len(charlie_dets)} det(s) size={W}x{H} (orig {W_orig}x{H_orig})")
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
            # None: SAM3 is not loaded yet.
            raise RuntimeError("SAM 3 is not loaded")
        except Exception as exc:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"v2 reference pipeline failed: {exc}") from exc

    # Interactive priority: onboarding waits on this.
    try:
        async with state["gpu_lock"].interactive():
            detections = await loop.run_in_executor(None, _infer)
    except Exception as exc:
        raise HTTPException(500, f"detection failed: {exc}")

    # force_label: the image was dropped into one label's section, so every
    # detection takes that label.
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
):
    """Create a dataset; returns {project_id, name}."""
    try:
        label_list = json.loads(labels)
        if not isinstance(label_list, list):
            raise ValueError("labels must be a JSON array")
    except Exception as e:
        raise HTTPException(400, f"invalid labels payload: {e}")
    tag_list = [str(l).strip() for l in label_list if str(l).strip()]

    colour_map: dict[str, str] = {}
    try:
        raw_colours = json.loads(label_colours) if label_colours else {}
        if isinstance(raw_colours, dict):
            for k, v in raw_colours.items():
                key = str(k).strip().lower()
                val = str(v).strip()
                # Malformed colours are dropped.
                if key and val.startswith("#") and 4 <= len(val) <= 9:
                    colour_map[key] = val
    except Exception:
        colour_map = {}

    project_id = _uuid.uuid4().hex
    proj = store.create_dataset_dir(project_id, name)
    (proj / "references").mkdir(exist_ok=True)

    manifest = empty_manifest(name, project_id=project_id)
    manifest["v2"] = True
    manifest["tags"] = tag_list
    manifest["references"] = []
    if colour_map:
        manifest["labelColours"] = colour_map
    save_manifest(project_id, manifest)
    return {"project_id": project_id, "name": name}


@app.post("/api/v2/projects/{project_id}/derive")
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
):
    """Create a child project of per-detection crops, synced one way from this project."""
    parent = load_manifest(project_id, copy=False)
    if not parent:
        raise HTTPException(404, "parent project not found")
    if parent.get("derived"):
        raise HTTPException(400, "Cannot create a derived dataset from another derived dataset.")
    try:
        sel = [str(l).strip() for l in json.loads(labels or "[]") if str(l).strip()]
    except Exception:
        raise HTTPException(400, "invalid labels payload")
    try:
        pad = max(0.0, min(1.0, float(padding)))
    except Exception:
        pad = 0.15
    try:
        min_px = max(0, min(1024, int(float(min_size))))
    except Exception:
        min_px = 256
    # Square crop size in px; 0 keeps each crop's own size.
    try:
        fixed_px = max(0, min(1024, int(float(fixed_size))))
    except Exception:
        fixed_px = 0
    square_mode = str(square).strip().lower() in ("1", "true", "yes", "on")
    # "new": the child starts with no labels; "inherit": the parent's.
    new_labels = str(label_mode).strip().lower() == "new"
    # Group the child and parent under a workspace Project.
    make_project = str(create_project).strip().lower() in ("1", "true", "yes", "on")

    child_id = _uuid.uuid4().hex
    proj = store.create_dataset_dir(child_id, name)
    (proj / "references").mkdir(exist_ok=True)
    (proj / "images").mkdir(exist_ok=True)

    child = empty_manifest(name, project_id=child_id)
    child["v2"] = True
    child["tags"] = [] if new_labels else (sel or list(parent.get("tags") or []))
    child["references"] = []
    # The parent's references for the selected labels are copied with their
    # files, except in "new" label mode.
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
                    continue  # missing source file
            kept_refs.append(dict(ref))
        child["references"] = kept_refs
    pcol = parent.get("labelColours")
    if not new_labels and isinstance(pcol, dict):
        lower_sel = {s.lower() for s in sel}
        child["labelColours"] = {k: v for k, v in pcol.items() if not sel or str(k).lower() in lower_sel}
    child["derived"] = {
        "parentProjectId": project_id,
        "parentName": parent.get("name") or "",
        # The parent labels to crop, whatever labelMode is.
        "labels": sel,
        "labelMode": "new" if new_labels else "inherit",
        "crop": {"padding": pad, "minSize": min_px, "square": square_mode, "fixedSize": fixed_px},
        "suppressed": [],
    }
    save_manifest(child_id, child)
    _register_child(project_id, child_id)

    # The child joins the parent's Project, or a new one named after the parent.
    # A failure here does not fail the derive.
    if make_project:
        try:
            parent_cid = (parent.get("container_id") or "").strip()
            cont = containers.load_container(parent_cid) if parent_cid else None
            if cont is None:
                cont = containers.create_container(parent.get("name") or name)
                add_event("container_create", container=cont["id"], name=cont.get("name"))
                pm = load_manifest(project_id)
                if pm and not (pm.get("container_id") or "").strip():
                    pm["container_id"] = cont["id"]
                    save_manifest(project_id, pm)
                    if project_id not in (cont.get("dataset_ids") or []):
                        cont.setdefault("dataset_ids", []).append(project_id)
                    add_event("dataset_add", container=cont["id"], dataset=project_id)
            cm = load_manifest(child_id)
            if cm:
                cm["container_id"] = cont["id"]
                save_manifest(child_id, cm)
            if child_id not in (cont.get("dataset_ids") or []):
                cont.setdefault("dataset_ids", []).append(child_id)
            containers.save_container(cont)
            add_event("dataset_add", container=cont["id"], dataset=child_id)
        except Exception as e:
            print(f"[derived] auto-project grouping failed for {child_id}: {e}", flush=True)

    # The first crop pass runs in the background.
    def _initial_crop() -> None:
        try:
            _resync_child_from_thread(child_id)
        except Exception as e:
            print(f"[derived] initial crop pass failed for {child_id}: {e}", flush=True)
    _threading.Thread(target=_initial_crop, daemon=True).start()
    return {"project_id": child_id, "name": name, "parent_project_id": project_id}


@app.post("/api/v2/projects/{project_id}/resync")
async def v2_resync_child(project_id: str):
    """Manually re-derive a child project from its parent's current state."""
    m = load_manifest(project_id, copy=False)
    if not (m.get("derived") or {}).get("parentProjectId"):
        raise HTTPException(400, "not a derived (child) project")
    if not await _resync_child_locked(project_id):
        raise HTTPException(500, "resync failed")
    updated = load_manifest(project_id, copy=False)
    return {"ok": True, "n_images": len(updated.get("imports") or [])}


@app.get("/api/v2/projects/{project_id}/children")
async def v2_list_children(project_id: str):
    """The derived projects of this parent."""
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


@app.post("/api/v2/projects/{project_id}/references")
async def v2_upload_reference(
    project_id: str,
    image: UploadFile = File(...),
    detections: str = Form("[]"),
    width: int = Form(0),
    height: int = Form(0),
    labels: str = Form(""),
    label: str = Form(""),
):
    """Store a reference image with embedded detections; detects with SAM3 when none are given."""
    proj = project_dir(project_id)
    if not proj.exists():
        raise HTTPException(404, "dataset not found")
    # Existence check only; the write happens under the lock below.
    if load_manifest(project_id) is None or load_manifest(project_id) == {}:
        raise HTTPException(404, "dataset not found")

    raw = await _read_upload(image)
    if not raw:
        raise HTTPException(400, "empty image upload")
    raw, ext = await asyncio.to_thread(_stored_image, raw)

    refs_dir = proj / "references"
    refs_dir.mkdir(exist_ok=True)
    ref_id = _uuid.uuid4().hex
    stored_name = f"{ref_id}{ext}"
    (refs_dir / stored_name).write_bytes(raw)

    try:
        det_list = json.loads(detections)
        if not isinstance(det_list, list):
            raise ValueError("detections must be a JSON array")
    except Exception as e:
        raise HTTPException(400, f"invalid detections payload: {e}")

    try:
        image_pil = _open_upright(raw)
        if not (width and height):
            width, height = image_pil.size
    except Exception:
        image_pil = None

    # No detections supplied: detect here.
    if not det_list and labels and image_pil is not None:
        try:
            tag_list = json.loads(labels)
            if not isinstance(tag_list, list):
                raise ValueError("labels must be a JSON array")
        except Exception as e:
            raise HTTPException(400, f"invalid labels payload: {e}")
        tags = [str(t).strip() for t in tag_list if str(t).strip()]
        if tags and state.get("charlie") is not None:
            loop = asyncio.get_running_loop()

            def _detect():
                charlie_dets = _charlie_reference_detections(image_pil, tags)
                if charlie_dets is not None:
                    for d in charlie_dets:
                        d["box"] = [round(c, 2) for c in d["box"]]
                    print(f"[v2-ref-upload] inline SAM 3 tags={tags} -> {len(charlie_dets)} det(s)")
                    return charlie_dets
                # SAM3 not loaded: the whole-image box below still applies.
                return []

            try:
                async with state["gpu_lock"].interactive():
                    det_list = await loop.run_in_executor(None, _detect)
            except Exception as exc:
                import traceback
                traceback.print_exc()
                raise HTTPException(500, f"inline reference detect failed: {exc}")

    # A section label applies to every detection; with none found the whole
    # image becomes one box, so the reference still gets an embedding.
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
                # Occluders inside the box are inpainted from the mask polygons.
                mask_polys = None
                m = d.get("mask")
                if isinstance(m, dict):
                    mask_polys = m.get("polygons")
                crop = v2_dinov2.inpaint_bbox_crop(image_pil, (x0, y0, x1, y1), mask_polys)
                squares.append(v2_dinov2.center_square_crop(crop))
                indices.append(i)
            if squares:
                # In a thread under the GPU gate, off the event loop.
                async with state["gpu_lock"].interactive():
                    vecs = await asyncio.get_running_loop().run_in_executor(
                        None, lambda: v2_dinov2.encode_images_batch(squares))
                for k, i in enumerate(indices):
                    det_list[i]["embedding"] = [round(float(x), 6) for x in vecs[k].tolist()]
                    det_list[i]["embed_version"] = v2_dinov2.EMBED_VERSION
                print(
                    f"[v2-ref-upload] embedded {len(indices)} of {len(det_list)} detection(s) "
                    f"for ref {ref_id} (dinov{v2_dinov2.EMBED_VERSION})"
                )
        else:
            print("[v2-ref-upload] DINOv2 is not loaded yet; the embeddings are made on first use.")

    # None when encoding fails.
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
    # Reloaded under the write lock, so concurrent appends are kept.
    write_lock = await _manifest_write_lock(project_id)
    async with write_lock:
        manifest = load_manifest(project_id) or {}
        manifest.setdefault("references", []).append(ref_entry)
        if not manifest.get("cover"):
            manifest["cover"] = stored_name
        manifest["updatedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        save_manifest(project_id, manifest)
        n_refs = len(manifest.get("references") or [])
    print(
        f"[v2-ref-upload] saved ref {ref_id} for project {project_id} "
        f"({n_refs} reference(s), cover={manifest.get('cover')!r})"
    )
    return {
        "reference_id": ref_id,
        "filename": stored_name,
        "width": width,
        "height": height,
        # Without the DINOv2 embeddings.
        "detections": [
            {k: v for k, v in d.items() if k not in ("embedding",)}
            for d in det_list
        ],
        "embedded": sum(1 for d in det_list if isinstance(d.get("embedding"), list)),
    }


@app.delete("/api/v2/projects/{project_id}/references/{reference_id}")
async def v2_delete_reference(project_id: str, reference_id: str):
    """Remove one reference image and its file."""
    proj = project_dir(project_id)
    if not proj.exists():
        raise HTTPException(404, "dataset not found")
    write_lock = await _manifest_write_lock(project_id)
    async with write_lock:
        manifest = load_manifest(project_id)
        if not manifest:
            raise HTTPException(404, "dataset not found")
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
        _replace_missing_cover(manifest)
        manifest["updatedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        save_manifest(project_id, manifest)
    return {"ok": True}


class UpdateRefDetectionsIn(BaseModel):
    detections: list[dict]


@app.put("/api/v2/projects/{project_id}/references/{reference_id}")
async def v2_update_reference(project_id: str, reference_id: str, payload: UpdateRefDetectionsIn):
    """Replace a reference's detections, embedding boxes without a current embedding."""
    if not project_dir(project_id).exists():
        raise HTTPException(404, "dataset not found")
    # Existence check only; the write happens under the lock below.
    snapshot = load_manifest(project_id)
    if not snapshot:
        raise HTTPException(404, "dataset not found")

    refs = snapshot.get("references") or []
    target = next((r for r in refs if r.get("id") == reference_id), None)
    if target is None:
        raise HTTPException(404, "reference not found")

    new_dets = payload.detections or []
    if not isinstance(new_dets, list):
        raise HTTPException(400, "detections must be a list")

    refs_dir = project_dir(project_id) / "references"
    img_path = refs_dir / target.get("filename", "")
    image_pil = None
    try:
        if img_path.exists():
            image_pil = _open_upright(img_path)
    except Exception:
        image_pil = None

    if image_pil is not None and new_dets:
        import v2_dinov2
        if v2_dinov2.is_loaded():
            W, H = image_pil.size
            embed_dim = v2_dinov2.EMBEDDING_DIM
            # One crop per box without a current vector.
            squares: list[PILImage.Image] = []
            indices: list[int] = []
            for i, d in enumerate(new_dets):
                emb = d.get("embedding")
                emb_v = d.get("embed_version")
                # Another embed_version has the same dim but a different feature space.
                if (
                    isinstance(emb, list)
                    and len(emb) == embed_dim
                    and isinstance(emb_v, int)
                    and emb_v == v2_dinov2.EMBED_VERSION
                ):
                    continue
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
            if squares:
                loop = asyncio.get_running_loop()
                async with state["gpu_lock"]:
                    d_vecs = await loop.run_in_executor(
                        None, lambda: v2_dinov2.encode_images_batch(squares)
                    )
                for k, i in enumerate(indices):
                    new_dets[i]["embedding"] = [
                        round(float(x), 6) for x in d_vecs[k].tolist()
                    ]
                    new_dets[i]["embed_version"] = v2_dinov2.EMBED_VERSION
                print(
                    f"[v2-ref-update] re-embedded ref {reference_id}: "
                    f"{len(indices)}/{len(new_dets)}"
                )

    # Re-resolved under the write lock; a reference deleted meanwhile is a 404.
    write_lock = await _manifest_write_lock(project_id)
    async with write_lock:
        manifest = load_manifest(project_id) or {}
        refs_live = manifest.get("references") or []
        live_target = next((r for r in refs_live if r.get("id") == reference_id), None)
        if live_target is None:
            raise HTTPException(404, "reference not found")
        live_target["detections"] = new_dets
        manifest["updatedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        save_manifest(project_id, manifest)
    n_embedded = sum(1 for d in new_dets if isinstance(d.get("embedding"), list))
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


@app.get("/api/v2/projects/{project_id}/references/{filename}")
async def v2_serve_reference(project_id: str, filename: str):
    """Serve a stored reference image; paths outside references/ are refused."""
    proj = project_dir(project_id)
    refs_root = (proj / "references").resolve()
    target = (refs_root / filename).resolve()
    try:
        target.relative_to(refs_root)
    except ValueError:
        raise HTTPException(403, "forbidden")
    return await _serve_cached_image(project_id, "references", filename, target)


@app.post("/api/v2/projects/{project_id}/imports/raw")
async def v2_upload_import_raw(
    project_id: str,
    image: UploadFile = File(...),
    created_at_ms: str = Form(""),
    idempotency_key: str = Form(""),
):
    """Store an imported image without detection; a label_charlie job labels it later."""
    proj = project_dir(project_id)
    if not proj.exists():
        raise HTTPException(404, "dataset not found")
    if not manifest_path(project_id).exists():
        raise HTTPException(404, "dataset not found")

    # A retry with the same idempotency_key gets the existing record. Checked
    # again under the write lock below.
    idem = (idempotency_key or "").strip()
    if idem:
        existing = _find_import_by_idempotency_key(project_id, idem)
        if existing is not None:
            return existing

    raw = await _read_upload(image)
    if not raw:
        print(
            f"[v2-import-raw] 400 empty upload: filename={image.filename!r} "
            f"content_type={image.content_type!r}"
        )
        raise HTTPException(400, "empty image upload")

    _orig_filename = image.filename or ""
    _loop = asyncio.get_running_loop()

    # JPEG is stored as uploaded; any other format is decoded and stored as RGB JPEG.
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
                _norm = _to_rgb(ImageOps.exif_transpose(_im))
                width, height = _norm.size
                _buf = io.BytesIO()
                _norm.save(_buf, "JPEG", quality=90)
                store_bytes = _buf.getvalue()
                print(
                    f"[v2-import-raw] normalised {_fmt or '?'} to JPEG "
                    f"({len(raw)} -> {len(store_bytes)} bytes) for {_orig_filename!r}"
                )
    except Exception as _e:
        raise HTTPException(400, f"cannot decode image: {_e}")
    if store_bytes is raw:
        try:
            await asyncio.to_thread(_check_jpeg_whole, raw)
        except Exception as _e:
            raise HTTPException(400, f"damaged or truncated image: {_e}")

    imports_dir = proj / "images"
    imports_dir.mkdir(exist_ok=True)
    import_id = _uuid.uuid4().hex
    if store_bytes is raw:
        ext = Path(_orig_filename or "image").suffix.lower()
        if ext not in (".jpg", ".jpeg"):
            ext = ".jpg"
    else:
        ext = ".jpg"
    stored_name = f"{import_id}{ext}"

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


    # The file write runs alongside the manifest update.
    file_write_fut = _loop.run_in_executor(
        None, (imports_dir / stored_name).write_bytes, store_bytes
    )

    write_lock = await _manifest_write_lock(project_id)
    async with write_lock:
        manifest = load_manifest(project_id, copy=False) or {}
        if idem:
            for existing_entry in manifest.get("imports", []) or []:
                if existing_entry.get("idempotencyKey") == idem:
                    # Another request committed this key first.
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

    # The background tasks below read the file.
    try:
        await file_write_fut
    except Exception as _fw_err:
        print(f"[v2-import-raw] file write failed for {import_id}: {_fw_err}")

    # BlurHash, computed in the background and stored on the entry.
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

    # Whole-image embedding for the dataset stats, in the background.
    asyncio.create_task(_run_bg_embedding(project_id, import_id, imports_dir / stored_name))

    # One activity row per upload request.
    add_event("job", id=import_id, job_kind="upload", project=project_id, status="done", n_images=1)

    return {
        "import_id": import_id,
        "filename": stored_name,
        "width": int(width),
        "height": int(height),
        "blurhash": None,
        "labelled": False,
    }


def _sanitize_imported_boxes(raw_boxes, width: int, height: int) -> tuple[list[dict], int]:
    """Normalise imported boxes, already in absolute pixels, to {id, label,
    x0, y0, x1, y1, score, mask?}: reversed corners are swapped, boxes
    clamped to the image and boxes under 1 px dropped. Returns
    (clean_boxes, n_dropped)."""
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


@app.post("/api/v2/projects/{project_id}/imports/raw_batch")
async def v2_upload_import_raw_batch(
    project_id: str,
    images: list[UploadFile] = File(...),
    created_at_ms: list[str] = Form(default=[]),
    idempotency_key: list[str] = Form(default=[]),
    boxes: list[str] = Form(default=[]),
):
    """Batch /imports/raw with one manifest write; items with `boxes` are stored labelled."""
    proj = project_dir(project_id)
    if not proj.exists():
        raise HTTPException(404, "dataset not found")
    if not manifest_path(project_id).exists():
        raise HTTPException(404, "dataset not found")
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

    raws: list[bytes] = [await up.read(MAX_UPLOAD_BYTES_PER_FILE + 1) for up in images]

    # An item whose idempotency key already landed gets the existing record.
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
        """Decode, store and blurhash one item. Returns its result with the
        manifest entry; the caller does the manifest write."""
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
            return {"idempotency_key": idem, "status": "failed",
                    "error": f"image too large (max {MAX_UPLOAD_BYTES_PER_FILE} bytes)"}

        # The format rules of /imports/raw.
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
                    _norm = _to_rgb(ImageOps.exif_transpose(_im))
                    width, height = _norm.size
                    _buf = io.BytesIO()
                    _norm.save(_buf, "JPEG", quality=90)
                    store_bytes = _buf.getvalue()
            if store_bytes is raw:
                _check_jpeg_whole(raw)
        except Exception as _e:
            return {"idempotency_key": idem, "status": "failed", "error": f"cannot decode image: {_e}"}


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

        # Inline, so the batch needs one manifest write.
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

        # Supplied boxes, in the uploaded image's pixels, become the item's
        # editedBoxes, and it counts as labelled even when they are empty.
        _labels: list[str] = []
        _dropped = 0
        if idx < len(boxes):
            # A malformed payload leaves the item labelled with no boxes.
            try:
                _parsed = json.loads(boxes[idx] or "[]")
                _clean, _dropped = _sanitize_imported_boxes(_parsed, width, height)
                _labels = [b["label"] for b in _clean]
            except Exception as _be:
                print(f"[v2-import-raw-batch] box parse failed for item {idx}: {_be}")
                _parsed, _clean, _dropped, _labels = None, [], 0, []
            entry["detections"] = []
            entry["editedBoxes"] = _clean
            entry["editedBoxesSet"] = True
            entry["labelled"] = True
            # An annotation that lists no objects declares a background
            # frame; one whose boxes were all unusable does not.
            if _parsed == []:
                entry["background"] = True

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
            "_labels": _labels,
        }

    results: list[dict] = list(await asyncio.gather(*[
        loop.run_in_executor(_BG_IMAGE_EXECUTOR, _process_one, i)
        for i in range(len(images))
    ]))

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
            # Skips keys a concurrent batch committed first.
            to_add = [
                e for e in new_entries
                if not (e.get("idempotencyKey") and e.get("idempotencyKey") in have_keys)
            ]
            mm.setdefault("imports", []).extend(to_add)
            # Imported box labels join the project's tags, which exports key off.
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
            # Off the event loop, since the write is synchronous.
            await loop.run_in_executor(
                None, lambda: save_manifest(project_id, mm, cache_by_ref=True)
            )
        try:
            _invalidate_project_payloads(project_id)
        except Exception as e:
            print(f"[v2-import-raw-batch] payload invalidate failed: {e}")

    # One activity row for the request, and a background embedding per image.
    stored = [r for r in results if r.get("status") == "ok" and r.get("_entry")]
    if stored:
        add_event("job", id=stored[0]["import_id"], job_kind="upload", project=project_id,
                  status="done", n_images=len(stored))
    for r in stored:
        asyncio.create_task(_run_bg_embedding(project_id, r["import_id"], project_dir(project_id) / "images" / r["filename"]))

    out = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
    return {"results": out}


class V2ImportsFromUrlsRequest(BaseModel):
    urls: list[str]
    query: str | None = None


@app.post("/api/v2/projects/{project_id}/imports/from_urls")
async def v2_imports_from_urls(
    project_id: str,
    body: V2ImportsFromUrlsRequest,
):
    """Download Openverse URLs as unlabelled imports; returns {added, skipped, rejected}."""
    import hashlib
    import uuid as _uuid
    import openverse
    from urllib.parse import urlparse

    proj = project_dir(project_id)
    if not proj.exists():
        raise HTTPException(404, "dataset not found")
    manifest = load_manifest(project_id)
    if not manifest:
        raise HTTPException(404, "dataset not found")

    raw_urls = [u.strip() for u in (body.urls or []) if isinstance(u, str) and u.strip()]
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
        sha = hashlib.sha256(data).hexdigest()
        if sha in existing_hashes:
            rejected.append({"url": url, "reason": "duplicate"})
            skipped.append(url)
            continue
        try:
            with PILImage.open(io.BytesIO(data)) as pil:
                pil = ImageOps.exif_transpose(pil)
                pil = pil.convert("RGB")
                width, height = pil.size
                blurhash_str = _compute_blurhash_safe(pil)
        except Exception as e:
            rejected.append({"url": url, "reason": "decode_failed", "detail": str(e)})
            continue

        # Stored as <query slug>_<random hex><ext>.
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

        # Descending timestamps, so the first URL sorts first in the gallery.
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
            "source": {"kind": "openverse", "url": url, "query": body.query or "",
                       **_OPENVERSE_META.get(url, {})},
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
        # Otherwise the cached payloads serve the pre-import list.
        try:
            _invalidate_project_payloads(project_id)
        except Exception as e:
            print(f"[v2-import-from-urls] payload invalidate failed: {e}")
        # One activity row for the request.
        if added:
            add_event("job", id=added[0], job_kind="upload", project=project_id,
                      status="done", n_images=len(added))

    return {
        "added": added,
        "skipped": skipped,
        "rejected": rejected,
    }


def _compute_blurhash_safe(pil_image) -> str | None:
    """BlurHash of a PIL image, or None on any error."""
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
    # True marks the image background: no boxes, skipped by labelling, a
    # training negative. Saving boxes onto it unmarks it.
    background: bool | None = None


@app.put("/api/v2/projects/{project_id}/imports/{import_id}")
async def v2_update_import(project_id: str, import_id: str, payload: V2ImportPatch):
    """Patch an import's editedBoxes, detections or background flag."""
    proj = project_dir(project_id)
    if not proj.exists():
        raise HTTPException(404, "dataset not found")
    write_lock = await _manifest_write_lock(project_id)
    async with write_lock:
        # copy=False is safe: mutated and saved under the write lock.
        manifest = load_manifest(project_id, copy=False)
        if not manifest:
            raise HTTPException(404, "dataset not found")
        imports = manifest.get("imports") or []
        edited_boxes = [] if payload.background else payload.editedBoxes
        for i, imp in enumerate(imports):
            if imp.get("id") == import_id:
                if payload.background:
                    imp["background"] = True
                elif payload.background is False or edited_boxes:
                    imp.pop("background", None)
                if edited_boxes is not None:
                    imp["editedBoxes"] = edited_boxes
                    # Tells an explicit delete-all apart from no edits.
                    imp["editedBoxesSet"] = True
                    imp["editedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                    # Epoch ms; the labelled preview's cachebuster.
                    imp["labelledAt"] = int(time.time() * 1000)
                if payload.detections is not None:
                    imp["detections"] = payload.detections
                imports[i] = imp
                manifest["imports"] = imports
                manifest["updatedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                save_manifest(project_id, manifest, cache_by_ref=True)
                _drop_annotation_payloads(project_id)
                # Re-baked now; the GET renders on demand until the bake lands.
                _invalidate_labelled_preview(project_id, import_id)
                fn = imp.get("filename")
                if fn:
                    src = proj / "images" / fn
                    edited = imp.get("editedBoxes")
                    edited_set = bool(imp.get("editedBoxesSet"))
                    # An explicitly cleared edit bakes an empty preview.
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
                return {"ok": True}
    raise HTTPException(404, "import not found")


def _remove_imports(project_id: str, manifest: dict, ids: set[str], tag: str) -> list[str]:
    """Drops the imports in `ids` from `manifest`, with their image files and
    previews, and moves a cover that named one. In a derived dataset each
    crop's source detection is suppressed, so a re-sync does not restore it.
    Returns the ids removed; the caller holds the write lock and saves."""
    proj = project_dir(project_id)
    derived = _derived_mod is not None and bool((manifest.get("derived") or {}).get("parentProjectId"))
    kept: list[dict] = []
    removed: list[str] = []
    for imp in manifest.get("imports") or []:
        iid = imp.get("id")
        if iid not in ids:
            kept.append(imp)
            continue
        removed.append(iid)
        fn = imp.get("filename")
        if fn:
            try:
                (proj / "images" / fn).unlink(missing_ok=True)
            except Exception as e:
                print(f"[{tag}] couldn't unlink {fn}: {e}")
        _invalidate_labelled_preview(project_id, iid)
        detkey = (imp.get("derivedFrom") or {}).get("detKey")
        if derived and detkey:
            _derived_mod.suppress_detkey(manifest, detkey)
    manifest["imports"] = kept
    _replace_missing_cover(manifest)
    return removed


@app.delete("/api/v2/projects/{project_id}/imports/{import_id}")
async def v2_delete_import(project_id: str, import_id: str):
    proj = project_dir(project_id)
    if not proj.exists():
        raise HTTPException(404, "dataset not found")
    write_lock = await _manifest_write_lock(project_id)
    async with write_lock:
        manifest = load_manifest(project_id)
        if not manifest:
            raise HTTPException(404, "dataset not found")
        if not _remove_imports(project_id, manifest, {import_id}, "v2-import-delete"):
            raise HTTPException(404, "import not found")
        manifest["updatedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        save_manifest(project_id, manifest)
    return {"ok": True}


class DeleteImportsBatchIn(BaseModel):
    ids: list[str]


@app.post("/api/v2/projects/{project_id}/imports/delete_batch")
async def v2_delete_imports_batch(project_id: str, payload: DeleteImportsBatchIn):
    """Delete many imports with one manifest save under one lock hold."""
    proj = project_dir(project_id)
    if not proj.exists():
        raise HTTPException(404, "dataset not found")
    target_ids = {x for x in (payload.ids or []) if isinstance(x, str) and x}
    if not target_ids:
        return {"deleted": [], "not_found": []}
    write_lock = await _manifest_write_lock(project_id)
    async with write_lock:
        manifest = load_manifest(project_id)
        if not manifest:
            raise HTTPException(404, "dataset not found")
        deleted_ids = _remove_imports(project_id, manifest, target_ids, "v2-import-delete-batch")
        manifest["updatedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        save_manifest(project_id, manifest)
    not_found = sorted(target_ids - set(deleted_ids))
    return {"deleted": deleted_ids, "not_found": not_found}


def _invalidate_project_payloads(project_id: str) -> None:
    """Drop the project's cached payloads and on-disk sidecars."""
    with _PAYLOAD_CACHE_LOCK:
        _PAYLOAD_DROPS[project_id] = _PAYLOAD_DROPS.get(project_id, 0) + 1
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
    """Body for /imports/dedupe. `threshold` is the cosine cutoff for near
    matches; the dataset stats use 0.95."""

    mode: str = "preview"  # "preview" or "commit"
    strategy: str = "near"  # "exact", "near" or "auto"
    threshold: float = 0.95


def _import_file_sha256(project_id: str, filename: str) -> str | None:
    """SHA256 of an import's file, or None when missing or unreadable."""
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
    """Rank of a duplicate group's survivor: edited boxes, then detections, then a filename."""
    edits = imp.get("editedBoxes") if isinstance(imp.get("editedBoxes"), list) else []
    n_edits = len(edits)
    dets = imp.get("detections") if isinstance(imp.get("detections"), list) else []
    n_dets = sum(1 for d in dets if isinstance(d, dict) and not d.get("rejected"))
    has_filename = 1 if imp.get("filename") else 0
    return (n_edits, n_dets, has_filename)


@app.post("/api/v2/projects/{project_id}/imports/dedupe")
async def v2_dedupe_imports(project_id: str, payload: DedupeIn):
    """Group duplicate imports by file hash or embedding; mode=commit deletes all but each keeper."""
    proj = project_dir(project_id)
    if not proj.exists():
        raise HTTPException(404, "dataset not found")
    if payload.mode not in ("preview", "commit"):
        raise HTTPException(400, "mode must be 'preview' or 'commit'")
    if payload.strategy not in ("exact", "near", "auto"):
        raise HTTPException(400, "strategy must be 'exact', 'near', or 'auto'")

    m = await asyncio.to_thread(load_manifest, project_id, False)
    if not m:
        raise HTTPException(404, "dataset not found")
    imports = [imp for imp in (m.get("imports") or []) if isinstance(imp, dict) and imp.get("id")]
    ignored = set(m.get("ignored_near_dups") or [])
    groups: list[dict] = []

    # Union-find over all imports: hash and embedding matches join groups transitively.
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

    # Hash buckets. Near mode also uses them to leave exact duplicates out.
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

    hash_buckets = await asyncio.to_thread(_hash_pass)

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

    # Near duplicates by embedding cosine similarity.
    if payload.strategy in ("near", "auto"):
        def _gather_embeddings() -> tuple[list[str], "np.ndarray | None"]:
            local_ids: list[str] = []
            local_vecs: list = []
            for imp in imports:
                iid = imp.get("id")
                if not iid or iid in ignored:
                    continue
                if payload.strategy == "near" and iid in exact_dup_ids:
                    continue
                v = _load_image_embedding(project_id, iid)
                if v is None:
                    continue
                local_ids.append(iid)
                local_vecs.append(v)
            if not local_vecs:
                return local_ids, None
            mat = np.stack(local_vecs, axis=0).astype(np.float32, copy=False)
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
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
            # The keeper id, unique per group.
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

    # Commit: delete every group's drop list, replacing the cover as the bulk delete does.
    drop_ids = {d["id"] for g in groups for d in g["drop"] if d.get("id")}
    if not drop_ids:
        return {"deleted": [], "groups": groups, "drop_count": 0}
    write_lock = await _manifest_write_lock(project_id)
    async with write_lock:
        manifest = load_manifest(project_id)
        if not manifest:
            raise HTTPException(404, "dataset not found")
        deleted_ids = _remove_imports(project_id, manifest, drop_ids, "v2-dedupe")
        manifest["updatedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        save_manifest(project_id, manifest)
    _invalidate_project_payloads(project_id)
    return {
        "deleted": deleted_ids,
        "groups": groups,
        "drop_count": len(deleted_ids),
    }


class IgnoreDupsIn(BaseModel):
    """Body for /imports/dedupe/ignore: import IDs kept as accepted duplicates."""

    ids: list[str]


@app.post("/api/v2/projects/{project_id}/imports/dedupe/ignore")
async def v2_ignore_dups(project_id: str, payload: IgnoreDupsIn):
    """Mark imports as accepted duplicates, kept in manifest.ignored_near_dups."""
    if not project_dir(project_id).exists():
        raise HTTPException(404, "dataset not found")
    add_ids = [x for x in (payload.ids or []) if isinstance(x, str) and x]
    if not add_ids:
        return {"ignored": [], "added": 0}
    write_lock = await _manifest_write_lock(project_id)
    added = 0
    async with write_lock:
        manifest = load_manifest(project_id)
        if not manifest:
            raise HTTPException(404, "dataset not found")
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
    """Serve a JPEG of an import at most `target` px on the long edge, cached on
    disk. Boxes stay in the original's pixels. The original is served if
    rendering fails."""
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
                # Orientation baked in, as the browser shows the original.
                im = ImageOps.exif_transpose(im).convert("RGB")
                W, H = im.size
                longest = max(W, H)
                if longest > target:
                    scale = target / float(longest)
                    im = im.resize((max(1, round(W * scale)), max(1, round(H * scale))), PILImage.LANCZOS)
            tmp = cached_path.with_suffix(_unique_tmp_suffix())
            im.save(tmp, format="JPEG", quality=85, optimize=True, progressive=True)
            store.replace(tmp, cached_path)

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


@app.get("/api/v2/projects/{project_id}/imports/{filename}")
async def v2_serve_import(project_id: str, filename: str, w: int = 0):
    """Serve a stored import image; ?w=N serves a copy at most N px on the long edge."""
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


# Augmentation preview: the effects applied to one image on the GPU,
# returned as a JPEG. Nothing is stored.


class AugmentPreviewIn(BaseModel):
    source: str  # "reference" or "import"
    filename: str
    # Camera and sensor dials, 0..10; 0 is identity.
    motion_blur: float = 0
    noise: float = 0
    colour_distortion: float = 0
    chromatic_aberration: float = 0
    bit_depth: float = 0
    #   lens_distortion  barrel warp
    #   pixelation       nearest-neighbour down and up sampling
    #   low_resolution   bilinear down and up sampling
    #   lens_glare       off-axis radial highlight
    lens_distortion: float = 0
    pixelation: float = 0
    low_resolution: float = 0
    lens_glare: float = 0
    # Geometric warps, applied before the sensor chain. One scale and one
    # rotation are sampled from these ranges per seed.
    perspective_warp: float = 0
    scale_min: float = 1.0
    scale_max: float = 1.0
    rot_min: float = 0.0
    rot_max: float = 0.0
    # Occlusion block size, a fraction of each detection's box. Imports only.
    block_size: float = 0
    # Draws the detection outlines. Imports only.
    show_outlines: bool = False
    # Images under <project>/augment_backgrounds/ that replace everything
    # outside the detection polygons.
    background_ids: list[str] | None = None
    # 0..10; the lighting direction is sampled from the seed.
    lighting_strength: float = 0
    # Hue offset scaled by strength, sampled from the seed.
    hue_shift: float = 0
    # Keeps the random pattern stable across slider moves.
    seed: int | None = None


_AUG_DEVICE: "torch.device | None" = None
# One thread for all augment work keeps the CUDA context and caching allocator warm.
import concurrent.futures as _concurrent_futures
_AUG_EXECUTOR = _concurrent_futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="augment-gpu",
)

# Background per-upload work, kept off the default executor so request-path
# I/O is not starved.
_BG_IMAGE_EXECUTOR = _concurrent_futures.ThreadPoolExecutor(
    max_workers=max(4, min(8, (os.cpu_count() or 4))),
    thread_name_prefix="bg-image",
)


def _aug_device() -> "torch.device":
    """Device for augmentations, picked on first use. Unlike the models they also run on CPU."""
    global _AUG_DEVICE
    if _AUG_DEVICE is None:
        # state["device"] is set by lifespan before any request runs.
        _AUG_DEVICE = torch.device(state.get("device") or _resolve_device())
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
    """Apply the camera and sensor chain to a [3, H, W] float tensor in
    [0, 1], on its device. Strengths are 0..10 and 0 skips a stage. Order:
    lens distortion, chromatic aberration, colour, motion blur, noise,
    bit depth, low resolution, pixelation, lens glare."""
    import torch.nn.functional as TF

    # The UI's 0..10 scaled to each dial's maximum.
    motion_blur = max(0.0, min(10.0, float(motion_blur))) * 0.4   # max 4
    noise = max(0.0, min(10.0, float(noise))) * 0.8                # max 8
    bit_depth = max(0.0, min(10.0, float(bit_depth))) * 0.8        # max 8
    pixelation = max(0.0, min(10.0, float(pixelation))) * 0.15     # max 1.5
    low_resolution = max(0.0, min(10.0, float(low_resolution))) * 0.5  # max 5
    lens_glare = max(0.0, min(10.0, float(lens_glare))) * 0.6      # max 6

    x = img_tensor

    # Lens distortion: barrel. The radius rho is in pixels over the half-diagonal,
    # so the corners sit at 1; output radius rho shows the source at
    # rho * (1 + k rho^2) / (1 + k), which keeps the corners and never folds.
    if lens_distortion > 0:
        _, H, W = x.shape
        k = _lens_k1(lens_distortion)
        cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
        ys = torch.linspace(-1.0, 1.0, H, device=x.device, dtype=x.dtype)
        xs = torch.linspace(-1.0, 1.0, W, device=x.device, dtype=x.dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        rho2 = ((grid_x * cx) ** 2 + (grid_y * cy) ** 2) / (cx * cx + cy * cy)
        factor = (1.0 + k * rho2) / (1.0 + k)
        grid = torch.stack([grid_x * factor, grid_y * factor], dim=-1).unsqueeze(0)
        x = TF.grid_sample(
            x.unsqueeze(0), grid,
            mode="bilinear", padding_mode="border", align_corners=True,
        ).squeeze(0)

    # Chromatic aberration: R shifts right and B left by max_shift px.
    if chromatic_aberration > 0:
        max_shift = max(1, int(round(chromatic_aberration * 0.6)))
        r = torch.roll(x[0:1], shifts=max_shift, dims=2)
        b = torch.roll(x[2:3], shifts=-max_shift, dims=2)
        # The wrapped columns keep their own channel's original values.
        r[..., :max_shift] = x[0:1, ..., :max_shift]
        b[..., -max_shift:] = x[2:3, ..., -max_shift:]
        x = torch.cat([r, x[1:2], b], dim=0)

    # Colour: brightness, contrast and saturation jitter, seeded.
    if colour_distortion > 0:
        strength = float(colour_distortion) / 10.0  # 0..1
        g = torch.Generator(device="cpu").manual_seed(seed ^ 0xC0CC)
        b_jit = float(1.0 + (torch.rand(1, generator=g).item() - 0.5) * strength * 0.6)
        c_jit = float(1.0 + (torch.rand(1, generator=g).item() - 0.5) * strength * 0.6)
        s_jit = float(1.0 + (torch.rand(1, generator=g).item() - 0.5) * strength * 1.0)
        x = (x * b_jit).clamp(0.0, 1.0)
        # Contrast about each channel's mean.
        mean = x.mean(dim=(1, 2), keepdim=True)
        x = ((x - mean) * c_jit + mean).clamp(0.0, 1.0)
        # Saturation about Rec. 709 luma.
        luma = (0.2126 * x[0] + 0.7152 * x[1] + 0.0722 * x[2]).unsqueeze(0)
        x = (luma + (x - luma) * s_jit).clamp(0.0, 1.0)

    # Motion blur: a horizontal box kernel, 1 + 2 x round(strength) px.
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

    # Sensor noise: additive Gaussian, seeded.
    if noise > 0:
        sigma = (float(noise) / 10.0) * 0.15  # 0..0.12 after the cap
        g = torch.Generator(device="cpu").manual_seed(seed ^ 0xA15E)
        # Generated on the CPU, so the seed gives the same noise on any device.
        n = torch.randn(x.shape, generator=g, dtype=torch.float32).to(x.device) * sigma
        x = (x + n).clamp(0.0, 1.0)

    # Bit depth: 2^bits levels per channel, 8 bits at 0, fractional bits allowed.
    if bit_depth > 0:
        bits = 8.0 - (float(bit_depth) / 10.0) * 7.0
        levels = max(2.0, 2.0 ** bits)
        x = (torch.round(x * (levels - 1.0)) / (levels - 1.0)).clamp(0.0, 1.0)

    # Low resolution: bilinear down and up sampling.
    if low_resolution > 0:
        _, H, W = x.shape
        s = float(low_resolution) / 10.0
        long_edge = max(H, W)
        # The 8 px floor keeps the downsample a legal size.
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

    # Pixelation: as low resolution, with a nearest-neighbour upsample.
    if pixelation > 0:
        _, H, W = x.shape
        s = float(pixelation) / 10.0
        long_edge = max(H, W)
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

    # Lens glare: two additive off-axis radial highlights, placed by the seed.
    if lens_glare > 0:
        _, H, W = x.shape
        s = float(lens_glare) / 10.0  # 0..1
        gen = torch.Generator(device="cpu").manual_seed(seed ^ 0x91A1E)
        ys = torch.linspace(-1.0, 1.0, H, device=x.device, dtype=x.dtype)
        xs = torch.linspace(-1.0, 1.0, W, device=x.device, dtype=x.dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        accum = torch.zeros_like(x[0])
        ang1 = float(torch.rand(1, generator=gen).item()) * 6.2831853
        rad1 = 0.45 + float(torch.rand(1, generator=gen).item()) * 0.4
        cx1 = float(rad1 * torch.cos(torch.tensor(ang1)).item())
        cy1 = float(rad1 * torch.sin(torch.tensor(ang1)).item())
        sigma1 = 0.25 + s * 0.35
        d2_1 = (gx - cx1) ** 2 + (gy - cy1) ** 2
        accum = accum + torch.exp(-d2_1 / (2.0 * sigma1 * sigma1))
        # The second glare: roughly opposite, smaller and dimmer.
        ang2 = ang1 + 3.1415926 + (float(torch.rand(1, generator=gen).item()) - 0.5) * 1.2
        rad2 = 0.3 + float(torch.rand(1, generator=gen).item()) * 0.3
        cx2 = float(rad2 * torch.cos(torch.tensor(ang2)).item())
        cy2 = float(rad2 * torch.sin(torch.tensor(ang2)).item())
        sigma2 = 0.15 + s * 0.2
        d2_2 = (gx - cx2) ** 2 + (gy - cy2) ** 2
        accum = accum + 0.6 * torch.exp(-d2_2 / (2.0 * sigma2 * sigma2))
        # Warm tint, nearer white at higher strength.
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
    """Lighting variation, strength 0..10: brightness, contrast and warmth sampled from the seed."""
    if strength <= 0:
        return img
    s = max(0.0, min(1.0, float(strength) / 10.0))
    g = torch.Generator(device="cpu").manual_seed(seed ^ 0xDA77)
    rs = torch.rand(3, generator=g).tolist()
    # Brightness multiplier in [1 - 0.6s, 1 + 0.6s].
    b = 1.0 + (rs[0] - 0.5) * 2.0 * 0.6 * s
    # Contrast scale in [1 - 0.4s, 1 + 0.4s].
    c = 1.0 + (rs[1] - 0.5) * 2.0 * 0.4 * s
    # Warmth: R up and B down, or the reverse, by up to 0.10.
    w = (rs[2] - 0.5) * 2.0 * 0.10 * s

    x = (img * b).clamp(0.0, 1.0)
    mean = x.mean(dim=(1, 2), keepdim=True)
    x = ((x - mean) * c + mean).clamp(0.0, 1.0)
    # Not in place: the tensor may be tracked.
    r = (x[0] + w).clamp(0.0, 1.0)
    g_ch = x[1]
    b_ch = (x[2] - w).clamp(0.0, 1.0)
    return torch.stack([r, g_ch, b_ch], dim=0)


def _aug_apply_hue_shift(
    img: "torch.Tensor",
    strength: float,
    seed: int,
) -> "torch.Tensor":
    """Seeded hue rotation: strength 0..10 gives an adjust_hue factor in [-0.5s, 0.5s]."""
    if strength <= 0:
        return img
    from torchvision.transforms.functional import adjust_hue as _adjust_hue
    s = max(0.0, min(1.0, float(strength) / 10.0))
    g = torch.Generator(device="cpu").manual_seed(seed ^ 0xC0F2)
    factor = (float(torch.rand(1, generator=g).item()) - 0.5) * 2.0 * (0.5 * s)
    # adjust_hue accepts [-0.5, 0.5].
    factor = max(-0.5, min(0.5, factor))
    out = _adjust_hue(img.unsqueeze(0), float(factor)).squeeze(0)
    return out.clamp(0.0, 1.0)


# Geometric warp helpers, shared by the preview and augment_generate so
# polygons warp with the image.


def _perspective_corner_pairs(
    strength: float, W: int, H: int, seed: int,
) -> "tuple[list[list[float]], list[list[float]]] | None":
    """The corner pairs _aug_apply_perspective_warp samples, for warping
    polygons. None at strength 0."""
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
    """3x3 homography, last entry 1, mapping each startpoint to its endpoint."""
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
    """The (scale, rotation_deg) _aug_apply_scale_rotation samples, or None for identity."""
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
    """Forward affine matching torchvision's affine(angle=r, scale=s). Its
    angle is clockwise on screen, so in y-down coordinates the rotation
    is [[cos, -sin], [sin, cos]]."""
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


def _lens_k1(strength: float) -> float:
    """Barrel coefficient for the lens dial: 0..10 -> 0..0.6, the centre's zoom less one."""
    return max(0.0, min(10.0, float(strength))) / 10.0 * 0.6


def _lens_forward_point(
    x: float, y: float, lens: "tuple[int, int, float]",
) -> "tuple[float, float, bool]":
    """Where the barrel warp carries source pixel (x, y), lens = (W, H, k), and
    whether that lands inside the picture. Solves r (1 + k r^2) = rho (1 + k)."""
    W, H, k = lens
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    dx, dy = x - cx, y - cy
    half_diag = (cx * cx + cy * cy) ** 0.5
    rho = (dx * dx + dy * dy) ** 0.5 / half_diag
    if k <= 0 or rho == 0.0:
        return x, y, True
    target = rho * (1.0 + k)
    r = min(rho, 1.0)
    for _ in range(50):
        step = (r * (1.0 + k * r * r) - target) / (1.0 + 3.0 * k * r * r)
        r -= step
        if abs(step) < 1e-12:
            break
    ox, oy = cx + dx * r / rho, cy + dy * r / rho
    eps = 1e-6
    shown = -eps <= ox <= W - 1 + eps and -eps <= oy <= H - 1 + eps
    return ox, oy, shown


def _warp_points(
    pts: "list[tuple[float, float]]", M: "list[list[float]] | None",
    lens: "tuple[int, int, float] | None",
) -> "list[tuple[float, float]]":
    """Points through M, then the lens warp; [] when the lens shows none
    of them."""
    if M is not None:
        pts = [_apply_matrix_to_point(M, x, y) for x, y in pts]
    if not lens:
        return pts
    moved = [_lens_forward_point(x, y, lens) for x, y in pts]
    return [(x, y) for x, y, _ in moved] if any(s for _, _, s in moved) else []


def _apply_matrix_to_polys(
    polys: "list[list[list[float]]]", M: "list[list[float]] | None",
    lens: "tuple[int, int, float] | None" = None,
) -> "list[list[list[float]]]":
    """Polygons through M, then the lens warp. A polygon the lens shows
    nothing of is dropped."""
    if M is None and not lens:
        return polys
    out: list[list[list[float]]] = []
    for poly in polys:
        pts = [(float(pt[0]), float(pt[1])) for pt in poly if len(pt) >= 2]
        warped = [[x, y] for x, y in _warp_points(pts, M, lens)]
        if warped:
            out.append(warped)
    return out


def _apply_matrix_to_box(
    box: "list[float]", M: "list[list[float]] | None",
    lens: "tuple[int, int, float] | None" = None,
) -> "list[float] | None":
    """Extent of a box through M and then the lens warp, which bends its
    edges, so the outline is sampled. None when the lens shows none of it."""
    if len(box) < 4 or (M is None and not lens):
        return list(box)
    x0, y0, x1, y1 = (float(v) for v in box[:4])
    n = 8 if lens else 1
    ts = [i / n for i in range(n + 1)]
    outline = ([(x0 + (x1 - x0) * t, y0) for t in ts] + [(x1, y0 + (y1 - y0) * t) for t in ts]
               + [(x0 + (x1 - x0) * t, y1) for t in ts] + [(x0, y0 + (y1 - y0) * t) for t in ts])
    pts = _warp_points(outline, M, lens)
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return [min(xs), min(ys), max(xs), max(ys)]


def _clip_box_to(box: "list[float]", w: float, h: float) -> "list[float] | None":
    """A box clipped to a w x h picture, or None when under 2 px remains on either axis."""
    x0, y0 = max(0.0, float(box[0])), max(0.0, float(box[1]))
    x1, y1 = min(float(w), float(box[2])), min(float(h), float(box[3]))
    return [x0, y0, x1, y1] if x1 - x0 >= 2.0 and y1 - y0 >= 2.0 else None


def _clip_poly_to(poly: "list[list[float]]", w: float, h: float) -> "list[list[float]]":
    """A polygon clipped to a w x h picture by Sutherland-Hodgman; [] when nothing is left."""
    def cut(points, inside, cross):
        out = []
        for i, cur in enumerate(points):
            prev = points[i - 1]
            if inside(cur):
                if not inside(prev):
                    out.append(cross(prev, cur))
                out.append(cur)
            elif inside(prev):
                out.append(cross(prev, cur))
        return out

    def at_x(x):
        return lambda a, b: [x, a[1] + (x - a[0]) * (b[1] - a[1]) / (b[0] - a[0])]

    def at_y(y):
        return lambda a, b: [a[0] + (y - a[1]) * (b[0] - a[0]) / (b[1] - a[1]), y]

    pts = [[float(p[0]), float(p[1])] for p in poly if isinstance(p, (list, tuple)) and len(p) >= 2]
    for inside, cross in ((lambda p: p[0] >= 0.0, at_x(0.0)), (lambda p: p[0] <= w, at_x(float(w))),
                          (lambda p: p[1] >= 0.0, at_y(0.0)), (lambda p: p[1] <= h, at_y(float(h)))):
        if len(pts) < 3:
            return []
        pts = cut(pts, inside, cross)
    return pts if len(pts) >= 3 else []


def _relabel_augmentations(project_id: str, relabel: Callable[[str], str | None]) -> int:
    """Apply `relabel` to every augmented copy's boxes; it returns the new
    name, the same name, or None to drop the box. Returns the boxes changed."""
    root = _augmentations_dir(project_id)
    changed = 0
    for path in sorted(root.glob("*/annotations.json")) if root.exists() else []:
        try:
            data = json.loads(path.read_text("utf-8"))
        except Exception:
            continue
        n_before = changed
        copies = data.get("copies") or {}
        for name, dets in copies.items():
            if not isinstance(dets, list):
                continue
            kept = []
            for d in dets:
                if not isinstance(d, dict):
                    continue
                label = str(d.get("label") or "")
                new = relabel(label)
                if new != label:
                    changed += 1
                if new is not None:
                    d["label"] = new
                    kept.append(d)
            copies[name] = kept
        if aug_source.relabel_source(data, relabel) or changed != n_before:
            store._atomic_write(path, json.dumps(data).encode("utf-8"))
    return changed


def _aug_apply_perspective_warp(
    img: "torch.Tensor",
    strength: float,
    seed: int,
) -> "torch.Tensor":
    """Random perspective warp: strength 0..10 moves each corner by up to
    0..0.19 of the image size in either direction."""
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
    """One scale in [scale_min, scale_max] and one rotation in degrees in
    [rot_min, rot_max], applied with torchvision's affine()."""
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
    """Every detection's polygons as one [H, W] float mask of 0.0 and 1.0 on `device`."""
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
    """One block per detection polygon, covering about `coverage_pct`%
    (0..60) of it. No polygon's coverage passes that cap, even where
    polygons overlap."""
    if coverage_pct <= 0:
        return img
    _C, H, W = img.shape
    g = torch.Generator(device="cpu").manual_seed(seed ^ 0xB10C)
    pct = max(0.0, min(60.0, float(coverage_pct))) / 100.0

    # Smallest polygons first, so the tightest budgets are placed first.
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

    for i, (det, poly_mask, _area) in enumerate(annotated):
        # Earlier blocks count toward this polygon's budget.
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

        # Clipped to this polygon. The pass below trims any polygon over its cap.
        zap_mask[ay:ay2, ax:ax2] = torch.maximum(
            zap_mask[ay:ay2, ax:ax2],
            poly_mask[ay:ay2, ax:ax2],
        )

    counts = (poly_stack * zap_mask.unsqueeze(0)).sum(dim=(1, 2))  # (N,)
    excesses = (counts - targets_t).clamp(min=0)
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

    # Occluded pixels are set to 0.04, near black.
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
    """Draw a thin emerald outline around every detection polygon."""
    _C, H, W = img.shape
    import numpy as _np
    from PIL import Image as _PIL
    from PIL import ImageDraw as _ImageDraw
    canvas = _PIL.new("RGBA", (W, H), (0, 0, 0, 0))
    drawer = _ImageDraw.Draw(canvas)
    line_w = max(2, int(round(min(W, H) / 320)))
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


def _backgrounds_dir(project_id: str) -> "Path":
    return project_dir(project_id) / "augment_backgrounds"


@app.post("/api/v2/projects/{project_id}/augment/background/upload")
async def v2_augment_background_upload(
    project_id: str,
    image: UploadFile = File(...),
):
    """Store a background image as augment_backgrounds/<uuid>.jpg; it is used whole."""
    if not project_dir(project_id).exists():
        raise HTTPException(404, "dataset not found")
    raw = await _read_upload(image)
    if not raw:
        raise HTTPException(400, "empty image upload")
    try:
        src = _open_upright(raw)
    except Exception as e:
        raise HTTPException(400, f"could not decode image: {e}")
    # Stored at most 1280 px on the long edge.
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


@app.get("/api/v2/projects/{project_id}/augment/backgrounds/{background_id}")
async def v2_augment_background_get(project_id: str, background_id: str):
    safe = "".join(c for c in background_id if c.isalnum())
    if not safe:
        raise HTTPException(400, "bad background id")
    path = _backgrounds_dir(project_id) / f"{safe}.jpg"
    if not path.exists():
        raise HTTPException(404, "background not found")
    return await _serve_cached_image(project_id, "augment_backgrounds", f"{safe}.jpg", path)


@app.delete("/api/v2/projects/{project_id}/augment/backgrounds/{background_id}")
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


# LRU of resized background tensors on the device, keyed by (path, H, W, device).
from collections import OrderedDict as _BG_OD

_BG_TENSOR_CACHE: "_BG_OD[tuple[str, int, int, str], torch.Tensor]" = _BG_OD()
_BG_TENSOR_CACHE_MAX = 16


def _load_background_resized_gpu(
    path: "Path", H: int, W: int, device: "torch.device",
) -> "torch.Tensor | None":
    """A cached (3, H, W) tensor of the background on `device`, or None."""
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
    """Replace pixels outside the detection polygons with a background chosen by `seed`."""
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


# Batch augmentation: N copies per image under
# <project>/augmentations/<import_id>/, outside the dataset and labelling.


def _augmentations_dir(project_id: str, import_id: str = "") -> "Path":
    root = project_dir(project_id) / "augmentations"
    return root / import_id if import_id else root


# Whole-image DINOv2 embeddings for dataset stats: raw float32 .bin files
# under <project>/image_embeddings/, computed after upload and backfilled on demand.

def _image_embeddings_dir(project_id: str) -> "Path":
    return project_dir(project_id) / "image_embeddings"


def _image_embedding_path(project_id: str, import_id: str) -> "Path":
    safe = "".join(c for c in import_id if c.isalnum())
    return _image_embeddings_dir(project_id) / f"{safe}.bin"


def _compute_and_store_image_embedding(project_id: str, import_id: str, raw_bytes: bytes) -> bool:
    """Embed one image and write its float32 vector. True when written;
    False when DINOv2 is not loaded."""
    try:
        import v2_dinov2 as _v2d
        if not _v2d.is_loaded():
            return False
        pil = _open_upright(raw_bytes)
        vec = _v2d.encode_image(pil)  # (D,) L2-normalised float32
        if vec is None or vec.size == 0:
            return False
        out_dir = _image_embeddings_dir(project_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = _image_embedding_path(project_id, import_id)
        # Atomic, so a concurrent reader never sees a partial file.
        tmp = store.tmp_path(out_path)
        tmp.write_bytes(vec.astype("float32", copy=False).tobytes())
        store.replace(tmp, out_path)
        return True
    except Exception as e:
        print(f"[v2-img-embed] {import_id} failed: {e}")
        return False


def _load_image_embedding(project_id: str, import_id: str) -> "np.ndarray | None":
    """An import's embedding as a (D,) float32 array, or None when missing or unreadable."""
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
    """Embed up to `limit` imports that have none; returns how many were written."""
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
    """[x, y] on the first two principal components per row, each axis
    scaled to [-1, 1]. Zeros for fewer than 2 rows or on failure."""
    import numpy as _np
    if vectors is None or len(vectors) < 2:
        return [[0.0, 0.0]] * (0 if vectors is None else len(vectors))
    try:
        X = vectors.astype(_np.float32, copy=False)
        X = X - X.mean(axis=0, keepdims=True)
        U, S, Vt = _np.linalg.svd(X, full_matrices=False)
        pc = U[:, :2] * S[:2]
        max_abs = _np.maximum(_np.abs(pc).max(axis=0), 1e-6)
        norm = pc / max_abs
        return [[float(x), float(y)] for x, y in norm]
    except Exception as e:
        print(f"[v2-stats] PCA failed: {e}")
        return [[0.0, 0.0]] * len(vectors)


def _near_duplicate_pairs(vectors: "np.ndarray", ids: list[str], threshold: float = 0.95) -> list[tuple[str, str, float]]:
    """Up to 200 pairs of import IDs with cosine similarity at least
    `threshold`. The vectors are L2-normalised, so cosine is the dot product."""
    pairs: list[tuple[str, str, float]] = []
    if vectors is None or len(vectors) < 2:
        return pairs
    try:
        sims = vectors @ vectors.T
        n = len(vectors)
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
    """Dataset-stats payload. lite skips the embeddings, PCA, near-duplicate
    pass and the uniqueness factor.

    Health is 0..100 from factors in [0, 1]:
      balance     1 - coefficient of variation of per-label counts
      coverage    imports with a detection or marked background / imports
      confidence  1 - unsure detections / detections
      uniqueness  1 - near-duplicate images / imports
    """
    import numpy as _np
    # Read only, so copy=False.
    manifest = load_manifest(project_id, False) or {}
    imports = manifest.get("imports") or []
    label_counts: dict[str, int] = {}
    n_detections = 0
    n_unsure = 0
    n_with_dets = 0
    n_background = 0
    aug_count = 0
    for imp in imports:
        # Edited boxes win, an explicit empty edit included; an empty list
        # without editedBoxesSet falls back to the detections.
        edited = imp.get("editedBoxes")
        if isinstance(edited, list) and (edited or imp.get("editedBoxesSet")):
            dets = edited
            used_edited = True
        else:
            dets = [d for d in (imp.get("detections") or []) if not d.get("rejected")]
            used_edited = False
        if dets:
            n_with_dets += 1
        elif imp.get("background"):
            n_background += 1  # checked and empty: counts as annotated
        for d in dets:
            if not isinstance(d, dict):
                continue
            # The first non-empty label key.
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
        # Jitter seeded per project.
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
            # One point per augmentation file, jittered around its source.
            # The directory is read because n_augmentations can lag.
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
        # Pairs with an ignored member are not counted.
        ignored = set(manifest.get("ignored_near_dups") or [])
        for a, b, _ in dup_pairs:
            if a in ignored or b in ignored:
                continue
            near_dup_ids.add(a)
            near_dup_ids.add(b)

    total_imports = len(imports)
    # An empty dataset scores 1.0 on each factor.
    if total_imports == 0:
        score_coverage = 1.0
        score_balance = 1.0
        score_confidence = 1.0
        score_uniqueness = 1.0
    else:
        score_coverage = (n_with_dets + n_background) / total_imports
        if label_counts:
            counts = _np.array(list(label_counts.values()), dtype=_np.float32)
            if counts.sum() > 0 and len(counts) > 1:
                # Coefficient of variation, clamped to [0, 1].
                cv = float(counts.std() / max(counts.mean(), 1e-6))
                score_balance = max(0.0, 1.0 - min(cv, 1.0))
            else:
                score_balance = 1.0
        else:
            score_balance = 0.5  # no labels yet
        score_confidence = 1.0 - (n_unsure / max(n_detections, 1))
        score_uniqueness = 1.0 - (len(near_dup_ids) / max(total_imports, 1))
    # Lite has no uniqueness factor and weighs the other three equally.
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
            "background": n_background,
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


@app.get("/api/v2/projects/{project_id}/dataset-stats", response_class=FastJSONResponse)
async def v2_dataset_stats(project_id: str, lite: bool = False):
    """Dataset counts, labels, variation plot and health score; lite=true skips the embeddings."""
    if not project_dir(project_id).exists():
        raise HTTPException(404, "dataset not found")
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
    # Else the on-disk sidecar, served even when stale while a rebuild runs.
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
    """Stored dataset-stats payload; lite and full have separate files."""
    name = "dataset_stats_lite.json" if lite else "dataset_stats.json"
    return project_dir(project_id) / name


def _read_dataset_stats_sidecar(
    project_id: str, lite: bool, manifest_mtime: float
) -> tuple[dict | None, bool]:
    """(payload, fresh). A stale payload is still returned."""
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
    """Compute the stats payload and write its sidecar. A failed write is logged, not raised."""
    stamp = _manifest_mtime_ns(project_id)
    payload = _compute_dataset_stats_v2(project_id, lite)
    try:
        _write_sidecar(_stats_sidecar_path(project_id, lite), payload, stamp)
    except Exception as e:
        print(f"[stats-sidecar] write failed for {project_id}: {e}")
    return payload


def _resolve_n_copies(per_image_mode: str, g: "torch.Generator") -> int:
    """Copies for one image: off, a count up to 8, or random 1..3."""
    if per_image_mode in ("off", "0"):
        return 0
    if per_image_mode == "random":
        return int(torch.randint(1, 4, (1,), generator=g).item())
    try:
        return max(0, min(8, int(per_image_mode)))
    except (TypeError, ValueError):
        return 1


def _aug_should_apply(frequency: str, g: "torch.Generator") -> bool:
    """Whether a copy gets the augmentation: always for "all", about half for "random"."""
    if frequency == "all":
        return True
    return bool(torch.rand(1, generator=g).item() > 0.5)


async def _run_augment_generate_job(job, emit, cancel_event):
    """Apply the augmentation config to every dataset image,
    saving each generated copy as a JPEG under augmentations/."""
    proj = project_dir(job.project)
    if not proj.exists():
        raise RuntimeError("dataset not found")
    manifest = load_manifest(job.project) or {}
    imports = manifest.get("imports") or []
    if not imports:
        raise RuntimeError("no dataset images to augment")
    cfg = (job.params or {}).get("config") or {}
    per_image_mode = str((job.params or {}).get("perImageMode") or "1")
    if per_image_mode in ("off", "0"):
        raise RuntimeError("augmentations per image is off; nothing to generate")

    # targetImportIds limits the run; an empty list does nothing.
    raw_targets = (job.params or {}).get("targetImportIds")
    if raw_targets is not None:
        target_set = {str(t) for t in raw_targets if t}
        if not target_set:
            await emit("progress", {"index": 0, "total": 0, "image": "", "phase": "augmenting"})
            print(f"[augment_generate] {job.project}: no images to augment")
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

    cam_enabled = bool(cam.get("enabled"))
    cam_freq = str(cam.get("frequency") or "all")
    cam_params = {
        "mb": float(cam.get("motionBlur") or 0),
        "ns": float(cam.get("noise") or 0),
        "cd": float(cam.get("colourDistortion") or 0),
        "ca": float(cam.get("chromaticAberration") or 0),
        "bd": float(cam.get("bitDepth") or 0),
        # A missing key is 0, the identity.
        "ld": float(cam.get("lensDistortion") or 0),
        "px": float(cam.get("pixelation") or 0),
        "lr": float(cam.get("lowResolution") or 0),
        "lg": float(cam.get("lensGlare") or 0),
    }
    occ_random_block = (occ.get("randomBlock") or {})
    occ_block_enabled = bool(occ.get("enabled")) and bool(occ_random_block.get("enabled"))
    occ_block_freq = str(occ_random_block.get("frequency") or "all")
    # The slider's 40% cap.
    occ_block_size = max(0.0, min(40.0, float(occ_random_block.get("size") or 0)))

    dist_pw = (dist.get("perspectiveWarp") or {})
    dist_sr = (dist.get("scaleRotation") or {})
    dist_hs = (dist.get("hueShift") or {})
    dist_pw_enabled = bool(dist.get("enabled")) and bool(dist_pw.get("enabled"))
    dist_pw_freq = str(dist_pw.get("frequency") or "all")
    dist_pw_strength = float(dist_pw.get("strength") or 0)
    dist_sr_enabled = bool(dist.get("enabled")) and bool(dist_sr.get("enabled"))
    dist_sr_freq = str(dist_sr.get("frequency") or "all")
    # The slider's 0.7..1.3 scale range.
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
    # Expected copies, for the progress total; random counts as 2.
    def _estimate_per_image() -> int:
        m = (per_image_mode or "").strip().lower()
        if m == "random":
            return 2
        try:
            return max(0, min(8, int(m)))
        except (TypeError, ValueError):
            return 1
    estimated_total_augmentations = len(imports) * _estimate_per_image()
    # Initialise the device in the augment thread before the loop.
    def _warm_cuda_in_thread():
        try:
            if device.type == "cuda":
                torch.cuda.init()
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()
            print(f"[augment_generate] runner starting on device={device.type}")
        except Exception as e:
            print(f"[augment_generate] device warmup failed: {e}")
    try:
        loop_for_warm = asyncio.get_event_loop()
        await loop_for_warm.run_in_executor(_AUG_EXECUTOR, _warm_cuda_in_thread)
    except Exception:
        pass
    # total counts augmentations, not images.
    job.progress = {
        "index": 0,
        "total": estimated_total_augmentations,
        "image": None,
        "phase": "augmenting",
    }

    # Synchronous; runs off the event loop.
    def _augment_one_image(i: int, imp: dict) -> int:
        import_id = imp.get("id")
        filename = imp.get("filename")
        if not import_id or not filename:
            return 0
        src_path = proj / "images" / filename
        if not src_path.exists():
            print(f"[augment_generate] src missing for {filename} ({import_id}), skipped")
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
                im = ImageOps.exif_transpose(im).convert("RGB")
                src_W, src_H = im.size
                longest = max(src_W, src_H)
                # Worked at most 1024 px on the long edge.
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
        # Edited boxes once edited (an explicit empty edit means none), else
        # the detections the labeller did not reject, as exports read them.
        edited = imp.get("editedBoxes")
        if isinstance(edited, list) and (edited or imp.get("editedBoxesSet")):
            dets = edited
        else:
            dets = [d for d in imp.get("detections") or []
                    if not (isinstance(d, dict) and d.get("rejected"))]
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
            # Hand-drawn and imported boxes are x0/y0/x1/y1; auto labels use `box`.
            xyxy = _box_xyxy(d)
            box_work: list[float] = []
            if xyxy is not None:
                box_work = [xyxy[0] * sx, xyxy[1] * sy, xyxy[2] * sx, xyxy[3] * sy]
            label = _box_label(d)
            if not label or not (box_work or polys_work):
                continue
            prepared_dets.append({"label": label, "polys": polys_work, "box": box_work})
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
                    # Lens distortion applies after the matrices; annotations follow it.
                    lens: "tuple[int, int, float] | None" = None
                    if dom_bg_enabled and bg_paths and dets and _aug_should_apply(dom_bg_freq, g_k):
                        out = _aug_apply_background_randomisation(out, dets, bg_paths, sx, sy, seed_k)
                    if occ_block_enabled and dets and _aug_should_apply(occ_block_freq, g_k):
                        out = _aug_apply_block_occlusion(out, dets, occ_block_size, sx, sy, seed_k)
                    if dist_pw_enabled and _aug_should_apply(dist_pw_freq, g_k):
                        pairs = _perspective_corner_pairs(dist_pw_strength, work_W, work_H, seed_k)
                        if pairs is not None:
                            H_mat = _solve_homography(*pairs)
                            cur_matrix = H_mat if cur_matrix is None else _matmul_3x3(H_mat, cur_matrix)
                        out = _aug_apply_perspective_warp(out, dist_pw_strength, seed_k)
                    if dist_sr_enabled and _aug_should_apply(dist_sr_freq, g_k):
                        sr_params = _scale_rotation_sample_params(
                            dist_sr["smin"], dist_sr["smax"], dist_sr["rmin"], dist_sr["rmax"], seed_k,
                        )
                        if sr_params is not None:
                            s_val, r_val = sr_params
                            SR_mat = _scale_rotation_matrix(s_val, r_val, work_W, work_H)
                            cur_matrix = SR_mat if cur_matrix is None else _matmul_3x3(SR_mat, cur_matrix)
                        out = _aug_apply_scale_rotation(out, dist_sr["smin"], dist_sr["smax"], dist_sr["rmin"], dist_sr["rmax"], seed_k)
                    if dist_hs_enabled and _aug_should_apply(dist_hs_freq, g_k):
                        out = _aug_apply_hue_shift(out, dist_hs_strength, seed_k)
                    if dom_lt_enabled and _aug_should_apply(dom_lt_freq, g_k):
                        out = _aug_apply_lighting(out, dom_lt_strength, seed_k)
                    if cam_enabled and _aug_should_apply(cam_freq, g_k):
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
                        if _lens_k1(cam_params["ld"]) > 0:
                            lens = (work_W, work_H, _lens_k1(cam_params["ld"]))
                    copy_annos: list[dict] = []
                    for det in prepared_dets:
                        # Clipped: a warp can move an object partly or wholly out of frame.
                        warped_polys = [q for q in (_clip_poly_to(p, work_W, work_H)
                                                    for p in _apply_matrix_to_polys(det["polys"], cur_matrix, lens)) if q]
                        moved = _apply_matrix_to_box(det["box"], cur_matrix, lens) if det["box"] else None
                        warped_box = _clip_box_to(moved, work_W, work_H) if moved else None
                        if warped_box is None and not warped_polys:
                            continue
                        if warped_box is None:  # a mask without a box: its extent
                            xs = [pt[0] for q in warped_polys for pt in q]
                            ys = [pt[1] for q in warped_polys for pt in q]
                            warped_box = _clip_box_to([min(xs), min(ys), max(xs), max(ys)], work_W, work_H)
                        copy_annos.append({
                            "label": det["label"],
                            "polys": warped_polys,
                            "box": warped_box or [],
                        })
                    anno_per_copy[f"{k:02d}.jpg"] = copy_annos
                    out_np = (out.clamp(0.0, 1.0) * 255.0 + 0.5).byte().permute(1, 2, 0).cpu().numpy()
                    out_path = out_dir / f"{k:02d}.jpg"
                    # Lower JPEG quality until the file is at most 500 KB.
                    out_im = PILImage.fromarray(out_np, mode="RGB")
                    target_bytes = 500 * 1024
                    quality_ladder = (85, 72, 60, 48, 36, 28)
                    # The quality-28 file stays even when over the target.
                    for q in quality_ladder:
                        out_im.save(
                            out_path, format="JPEG", quality=q,
                            optimize=False, progressive=False,
                        )
                        try:
                            if out_path.stat().st_size <= target_bytes:
                                break
                        except OSError:
                            break
                    saved_count += 1
                except Exception as e:
                    print(f"[augment_generate] copy {k} for {filename} ({import_id}) failed: {e}")
        try:
            store._atomic_write(out_dir / "annotations.json", json.dumps({
                "width": int(work_W),
                "height": int(work_H),
                "copies": anno_per_copy,
                "source": aug_source.source_boxes(imp),
            }).encode("utf-8"))
        except Exception as e:
            print(f"[augment_generate] annotations.json write for {import_id} failed: {e}")
        job.progress = {
            "index": i + 1,
            "total": len(imports),
            "image": filename,
            "phase": "augmenting",
        }
        return saved_count

    # n_augmentations updates, flushed every FLUSH_EVERY images and at the end.
    n_aug_updates: dict[str, int] = {}
    FLUSH_EVERY = 25
    # Augmentations saved so far, the progress index.
    saved_total = 0

    def _flush_n_aug_updates_sync() -> None:
        # The caller holds the project's write lock.
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
        # The GPU lock is taken per image, so interactive requests run between images.
        async with state["gpu_lock"]:
            saved = await loop.run_in_executor(_AUG_EXECUTOR, _augment_one_image, i, imp)
        iid = imp.get("id")
        if iid:
            n_aug_updates[iid] = saved
        saved_total += saved
        # Counted in augmentations, not images.
        job.progress = {
            "index": min(saved_total, estimated_total_augmentations),
            "total": max(saved_total, estimated_total_augmentations),
            "image": imp.get("filename"),
            "phase": "augmenting",
        }
        if (i + 1) % FLUSH_EVERY == 0:
            await _flush_n_aug_updates()
        await asyncio.sleep(0)
    if n_aug_updates:
        await _flush_n_aug_updates()


class AugmentGenerateIn(BaseModel):
    perImageMode: str = "1"
    config: dict = {}


# Jobs that rewrite a dataset's labels. Each flushes from its own snapshot,
# so two at once, or a rename during one, would overwrite each other.
_LABEL_WRITERS = ("label_charlie", "label_model", "purge_label")


def _label_job_active(project_id: str) -> bool:
    return any(j.project == project_id and j.kind in _LABEL_WRITERS
               and j.status in ("queued", "running") for j in state["jobs"].jobs.values())


def _dataset_job_active(project_id: str) -> bool:
    """Any job queued or running on this dataset."""
    return any(j.project == project_id and j.status in ("queued", "running")
               for j in state["jobs"].jobs.values())


def _augment_job_active(project_id: str) -> bool:
    return any(j.project == project_id and j.kind == "augment_generate"
               and j.status in ("queued", "running") for j in state["jobs"].jobs.values())


@app.post("/api/v2/projects/{project_id}/augment/generate")
async def v2_augment_generate(project_id: str, payload: AugmentGenerateIn):
    if not project_dir(project_id).exists():
        raise HTTPException(404, "dataset not found")
    # Clearing or regenerating wipes the folder a running generation writes into,
    # and the copies a training run reads every epoch.
    if _augment_job_active(project_id):
        raise HTTPException(409, "augmentations are being generated; wait for that run or cancel it")
    if _train_job_active(project_id):
        raise HTTPException(409, "a training run is using this dataset's augmentations; change them when it finishes")
    manifest = load_manifest(project_id) or {}
    imports = manifest.get("imports") or []
    if not imports:
        raise HTTPException(400, "no dataset images to augment")
    # perImageMode off only clears; no job is scheduled and job_id is None.
    clear_only = payload.perImageMode in ("off", "0")

    # Saved so the UI can restore the settings.
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
        # The job runs with the payload's config regardless.
        pass

    # Earlier outputs are removed and n_augmentations reset.
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
        _invalidate_project_payloads(project_id)
        return {"job_id": None, "cleared": True}

    job = state["jobs"].schedule(
        "augment_generate", project_id,
        {"perImageMode": str(payload.perImageMode), "config": payload.config or {}},
        n_images=len(imports),
    )
    state["jobs"].start_worker()
    return {"job_id": job.id}


@app.get("/api/v2/projects/{project_id}/augment/config")
async def v2_augment_get_config(project_id: str):
    """The saved augmentation config, or {}."""
    if not project_dir(project_id).exists():
        raise HTTPException(404, "dataset not found")
    manifest = load_manifest(project_id) or {}
    return manifest.get("augmentationConfig") or {}


@app.get("/api/v2/projects/{project_id}/augment/job/active")
async def v2_augment_active_job(project_id: str):
    """Active augment_generate job for this project, or null."""
    if not project_dir(project_id).exists():
        raise HTTPException(404, "dataset not found")
    for j in state["jobs"].jobs.values():
        if j.project == project_id and j.kind == "augment_generate" and j.status in ("running", "queued"):
            return j.to_public()
    return None


@app.get("/api/v2/projects/{project_id}/augmentations/{import_id}")
async def v2_augment_list(project_id: str, import_id: str):
    """Augmentation filenames for one import."""
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


@app.get("/api/v2/projects/{project_id}/augmentations/{import_id}/annotations")
async def v2_augment_annotations(project_id: str, import_id: str):
    """The import's annotations.json from augment_generate, or an empty one."""
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


@app.get("/api/v2/projects/{project_id}/augmentations/{import_id}/{filename}")
async def v2_augment_serve(project_id: str, import_id: str, filename: str):
    if not project_dir(project_id).exists():
        raise HTTPException(404)
    safe_id = "".join(c for c in import_id if c.isalnum())
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


@app.delete("/api/v2/projects/{project_id}/augmentations/{import_id}/{filename}")
async def v2_augment_delete(project_id: str, import_id: str, filename: str):
    """Delete one augmentation and its annotations entry, updating
    n_augmentations. Deleting a missing file succeeds."""
    if not project_dir(project_id).exists():
        raise HTTPException(404, "dataset not found")
    if _train_job_active(project_id):
        raise HTTPException(409, "a training run is using this dataset's augmentations; change them when it finishes")
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

    # Counted from the image files on disk.
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


@app.post("/api/v2/projects/{project_id}/augment/preview")
async def v2_augment_preview(project_id: str, payload: AugmentPreviewIn):
    """Render one image through the augmentation chain as a JPEG; nothing is stored."""
    proj = project_dir(project_id)
    if not proj.exists():
        raise HTTPException(404, "dataset not found")
    src = (payload.source or "").lower()
    if src not in ("reference", "import"):
        raise HTTPException(400, "source must be 'reference' or 'import'")
    subdir = "references" if src == "reference" else "images"
    target = (proj / subdir / payload.filename).resolve()
    try:
        target.relative_to((proj / subdir).resolve())
    except ValueError:
        raise HTTPException(403, "forbidden")
    if not target.exists():
        raise HTTPException(404, "image not found")

    # Dials clamped to 0..10.
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
    # Block size is a coverage percentage, 0..40.
    try:
        bs = max(0.0, min(40.0, float(payload.block_size)))
    except (TypeError, ValueError):
        bs = 0.0
    show_outlines = bool(payload.show_outlines)

    # Imports only: the detections define the foreground.
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

    # Loaded whenever a feature needs the polygons.
    needs_detections = bs > 0 or show_outlines or bool(payload.background_ids)
    detections: list[dict] = []
    if src == "import" and needs_detections:
        manifest = load_manifest(project_id) or {}
        for imp in (manifest.get("imports") or []):
            if imp.get("filename") == payload.filename:
                # An explicit empty edit means no boxes.
                edited = imp.get("editedBoxes")
                if isinstance(edited, list) and (edited or imp.get("editedBoxesSet")):
                    detections = edited
                else:
                    detections = imp.get("detections") or []
                break

    loop = asyncio.get_running_loop()

    def _run() -> bytes:
        from io import BytesIO
        with PILImage.open(target) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            src_W, src_H = im.size
            # Rendered at most 720 px on the long edge.
            longest = max(src_W, src_H)
            if longest > 720:
                scale = 720.0 / longest
                im = im.resize((int(src_W * scale), int(src_H * scale)), PILImage.LANCZOS)
        work_W, work_H = im.size
        # Detections are in source pixels.
        sx = work_W / float(src_W) if src_W > 0 else 1.0
        sy = work_H / float(src_H) if src_H > 0 else 1.0

        device = _aug_device()
        import numpy as _np
        arr = _np.asarray(im, dtype=_np.float32) / 255.0  # HWC
        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous().to(device)
        with torch.no_grad():
            # Detection-based steps run first, in source coordinates; the
            # warps, colour and the sensor chain follow.
            out = t
            if background_paths and detections:
                out = _aug_apply_background_randomisation(out, detections, background_paths, sx, sy, seed)
            if bs > 0 and detections:
                out = _aug_apply_block_occlusion(out, detections, bs, sx, sy, seed)
            if show_outlines and detections:
                out = _aug_apply_segmentation_outlines(out, detections, sx, sy)
            if pw > 0:
                out = _aug_apply_perspective_warp(out, pw, seed)
            if not (sr_smin == 1.0 and sr_smax == 1.0 and sr_rmin == 0.0 and sr_rmax == 0.0):
                out = _aug_apply_scale_rotation(out, sr_smin, sr_smax, sr_rmin, sr_rmax, seed)
            if hu > 0:
                out = _aug_apply_hue_shift(out, hu, seed)
            if lt > 0:
                out = _aug_apply_lighting(out, lt, seed)
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


@app.get("/api/v2/projects/{project_id}/imports/{import_id}/labelled_preview")
async def v2_serve_labelled_preview(project_id: str, import_id: str):
    """Annotated preview of one import, about 600 px on the long edge,
    rendered on first request and cached on disk."""
    proj = project_dir(project_id)
    if not proj.exists():
        raise HTTPException(404, "dataset not found")
    cached_path = _labelled_preview_path(project_id, import_id)

    if cached_path.exists():
        return await _serve_cached_image(project_id, "imports", cached_path.name, cached_path)

    # Loading warms the cache; the import index then gives the entry.
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

    with _MANIFEST_CACHE_LOCK:
        _cached_m = _MANIFEST_CACHE.get(project_id) or {}
    project_labels = list(_cached_m.get("tags") or [])

    # Edited boxes win over detections.
    det_source = _preview_source(imp, imp.get("detections") or [])

    loop = asyncio.get_running_loop()

    def _render() -> None:
        with PILImage.open(src_path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            preview = _render_labelled_preview(im, det_source, project_labels)
        cached_path.parent.mkdir(parents=True, exist_ok=True)
        # A unique temp file, then a rename, so no reader sees a partial JPEG.
        tmp = cached_path.with_suffix(_unique_tmp_suffix())
        preview.save(tmp, format="JPEG", quality=72, optimize=True, progressive=True)
        store.replace(tmp, cached_path)

    # One render per import at a time; waiters reuse the result.
    lock = _thumb_render_lock(project_id, f"labelled_preview:{import_id}")
    async with lock:
        if cached_path.exists():
            return await _serve_cached_image(project_id, "imports", cached_path.name, cached_path)
        try:
            await loop.run_in_executor(None, _render)
        except Exception as e:
            print(f"[labelled-preview] render failed for {project_id}/{import_id}: {e}")
            # The original is served if rendering fails.
            return await _serve_cached_image(project_id, "imports", fn, src_path)

    return await _serve_cached_image(project_id, "imports", cached_path.name, cached_path)


# File name of an uploaded cover, marked by the manifest flag cover_uploaded.
# Cover replacement on image deletes changes only manifest["cover"].
UPLOADED_COVER_SENTINEL = "cover_upload.jpg"


@app.post("/api/projects/{project_id}/cover_upload")
async def upload_project_cover(project_id: str, file: UploadFile = File(...)):
    """Upload a dataset cover, stored as a JPEG of at most 1600 px and flagged cover_uploaded."""
    proj = project_dir(project_id)
    if not proj.exists():
        raise HTTPException(404, "dataset not found")
    raw = await _read_upload(file, "cover")
    if not raw:
        raise HTTPException(400, "empty file")
    try:
        with PILImage.open(io.BytesIO(raw)) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
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
    try:
        (proj / UPLOADED_COVER_SENTINEL).write_bytes(jpeg)
    except Exception as e:
        print(f"[cover-upload] local write failed for {project_id}: {e}")
        raise HTTPException(500, "could not store cover")
    write_lock = await _manifest_write_lock(project_id)
    async with write_lock:
        manifest = load_manifest(project_id)
        if not manifest:
            raise HTTPException(404, "dataset not found")
        manifest["cover_uploaded"] = True
        manifest["cover_blurhash"] = None  # re-derived on next read
        manifest["updatedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        save_manifest(project_id, manifest)
    _invalidate_project_payloads(project_id)
    return {"ok": True, "cover_uploaded": True}


def _upscale_cover_if_small(im, floor: int):
    """Enlarge a cover under `floor` px on the long edge, with LANCZOS and a
    light unsharp mask. Returns the image."""
    from PIL import ImageFilter
    w, h = im.size
    longest = max(w, h)
    if longest <= 0 or longest >= floor:
        return im
    scale = floor / float(longest)
    im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), PILImage.LANCZOS)
    return im.filter(ImageFilter.UnsharpMask(radius=1.2, percent=80, threshold=2))


def _cover_source(proj: Path, cover: str) -> Path | None:
    """The cover's file: references/ first, then images/."""
    return next((p for p in (proj / "references" / cover, proj / "images" / cover) if p.is_file()), None)


async def _repair_missing_cover(project_id: str) -> str | None:
    """Point a cover whose file is gone at an image, else a reference, that
    exists on disk; returns the cover, or None when there is no picture."""
    lock = await _manifest_write_lock(project_id)
    async with lock:
        m = load_manifest(project_id)
        if not m:
            return None
        proj = project_dir(project_id)
        cover = m.get("cover")
        if cover and _cover_source(proj, cover):
            return cover
        names = [e.get("filename") for key in ("imports", "references") for e in m.get(key) or []
                 if isinstance(e, dict) and e.get("filename") and e["filename"] == Path(e["filename"]).name]
        new = next((n for n in names if _cover_source(proj, n)), None)
        m["cover"] = new
        m["cover_blurhash"] = None
        save_manifest(project_id, m)
        return new


@app.get("/api/projects/{project_id}/cover_thumb")
async def serve_cover_thumb(project_id: str, w: int = 480):
    """Cover thumbnail `w` px on the long edge, 240..1600, cached on disk."""
    proj = project_dir(project_id)
    if not proj.exists():
        raise HTTPException(404, "dataset not found")
    manifest = load_manifest(project_id)
    if not manifest:
        raise HTTPException(404, "dataset not found")

    # An uploaded cover wins.
    if manifest.get("cover_uploaded"):
        src_path = proj / UPLOADED_COVER_SENTINEL
        if not src_path.exists():
            raise HTTPException(404, "uploaded cover missing")
        cover = UPLOADED_COVER_SENTINEL
    else:
        cover = manifest.get("cover")
        if not cover:
            raise HTTPException(404, "project has no cover")
        if cover != Path(cover).name or cover in (".", ".."):
            raise HTTPException(404, "project has no cover")

        src_path = _cover_source(proj, cover)
        if src_path is None:
            # A cover whose picture is gone moves to a picture that exists.
            cover = await _repair_missing_cover(project_id)
            src_path = _cover_source(proj, cover) if cover else None
            if src_path is None:
                raise HTTPException(404, "cover file missing on disk")

    # Cards use 480 px; the dataset hero asks for more.
    target = max(240, min(1600, int(w or 480)))
    name_hash = hashlib.sha1(cover.encode("utf-8")).hexdigest()[:8]
    # Keyed by the cover's name and the size, so each size and each new cover
    # gets its own file.
    cached_path = proj / f"cover_thumb_v2_{name_hash}_{target}.jpg"
    try:
        src_mtime = src_path.stat().st_mtime
    except OSError:
        src_mtime = 0.0

    # Re-rendered when missing or older than the source.
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
                im = ImageOps.exif_transpose(im).convert("RGB")
                W, H = im.size
                longest = max(W, H)
                if longest > target:
                    scale = target / longest
                    im = im.resize((int(W * scale), int(H * scale)), PILImage.LANCZOS)
                else:
                    im = _upscale_cover_if_small(im, target)
            tmp = cached_path.with_suffix(_unique_tmp_suffix())
            im.save(tmp, format="JPEG", quality=78, optimize=True, progressive=True)
            store.replace(tmp, cached_path)

        # One render per variant at a time; waiters reuse the result.
        lock = _thumb_render_lock(project_id, f"cover_thumb_{target}")
        async with lock:
            try:
                cached_mtime = cached_path.stat().st_mtime if cached_path.exists() else 0.0
            except OSError:
                cached_mtime = 0.0
            if cached_mtime < src_mtime:
                try:
                    await loop.run_in_executor(None, _render)
                except Exception as e:
                    print(f"[cover-thumb] render failed for {project_id}: {e}")
                    # The original cover is served if rendering fails.
                    return await _serve_cached_image(project_id, src_path.parent.name, cover, src_path)

    return await _serve_cached_image(project_id, "cover_thumb", cached_path.name, cached_path)


@app.post("/api/v2/references/embed_crops")
async def v2_embed_crops(
    image: UploadFile = File(...),
    boxes: str = Form(...),
    return_crop: bool = Form(False),
):
    """Embed each box crop with DINOv2. `boxes` is a JSON list of [x0, y0, x1, y1] in pixels."""
    import base64
    import v2_dinov2

    if not v2_dinov2.is_loaded():
        raise HTTPException(503, "the image encoder is still loading; try again shortly")

    try:
        bbs = json.loads(boxes)
        if not isinstance(bbs, list):
            raise ValueError("boxes must be a JSON array")
    except Exception as e:
        raise HTTPException(400, f"invalid boxes payload: {e}")

    raw = await _read_upload(image)
    if not raw:
        raise HTTPException(400, "empty image upload")
    try:
        image_pil = _open_upright(raw)
    except Exception as e:
        raise HTTPException(400, f"could not decode image: {e}")
    W, H = image_pil.size

    loop = asyncio.get_running_loop()

    def _run():
        kept_indices: list[int] = []
        kept_xyxy: list[tuple[int, int, int, int]] = []
        kept_crops: list[PILImage.Image] = []
        kept_squares: list[PILImage.Image] = []
        for i, bb in enumerate(bbs):
            try:
                if not (isinstance(bb, list) and len(bb) == 4):
                    print(f"[v2-embed] box {i} skipped, not a 4-tuple: {bb!r}")
                    continue
                x0 = max(0, int(round(float(bb[0]))))
                y0 = max(0, int(round(float(bb[1]))))
                x1 = min(W, int(round(float(bb[2]))))
                y1 = min(H, int(round(float(bb[3]))))
                if x1 - x0 < 4 or y1 - y0 < 4:
                    print(f"[v2-embed] box {i} skipped, too small after clipping: {bb!r} -> ({x0},{y0},{x1},{y1}) on {W}x{H}")
                    continue
                crop = image_pil.crop((x0, y0, x1, y1))
                square = v2_dinov2.center_square_crop(crop)
                kept_indices.append(i)
                kept_xyxy.append((x0, y0, x1, y1))
                kept_crops.append(crop)
                kept_squares.append(square)
            except Exception as e:
                print(f"[v2-embed] box {i} crop failed: {e}")

        vecs = v2_dinov2.encode_images_batch(kept_squares) if kept_squares else None

        results: list[dict] = []
        for k, (i, xyxy, crop) in enumerate(zip(kept_indices, kept_xyxy, kept_crops)):
            entry: dict = {
                "index": i,
                "box": list(xyxy),
                "embedding": [round(float(x), 6) for x in vecs[k].tolist()] if vecs is not None else [],
                # The centroid build re-encodes any other EMBED_VERSION.
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
        # The references editor waits on this call.
        async with state["gpu_lock"].interactive():
            crops = await loop.run_in_executor(None, _run)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"embed_crops failed: {exc}")

    return {"width": W, "height": H, "crops": crops}


# Reference centroids: per-label embeddings of the reference boxes, which
# detections are compared against to pick their label.

# project_id -> (manifest_mtime, by_label_dino).
_V2_REFS_CACHE: dict[str, tuple[float, dict]] = {}
_V2_REFS_CACHE_LOCK = __import__("threading").Lock()


def _v2_load_or_backfill_reference_embeddings(project_id: str) -> dict:
    """Per-label DINOv2 embeddings of every reference box, {label_lower:
    [np.ndarray]}, re-encoding missing or stale ones and saving them to the
    manifest."""
    import numpy as _np
    import v2_dinov2 as _v2d
    disk_mtime = _manifest_disk_mtime(project_id)
    if disk_mtime > 0:
        with _V2_REFS_CACHE_LOCK:
            cached = _V2_REFS_CACHE.get(project_id)
            if cached is not None and cached[0] == disk_mtime:
                return cached[1]
    # copy=False: the manifest is mutated only right before save_manifest.
    manifest = load_manifest(project_id, copy=False) or {}
    refs = manifest.get("references") or []
    by_label_dino: dict[str, list[_np.ndarray]] = {}
    dirty = False
    proj = project_dir(project_id)
    refs_dir = proj / "references"

    # Stale or missing embeddings, queued per reference so each image is decoded once.
    pending_by_ref: dict[int, list[tuple[int, str, list[float], list | None]]] = {}
    for ri, ref in enumerate(refs):
        for di, d in enumerate(ref.get("detections") or []):
            label = (d.get("label") or "").strip()
            if not label:
                continue
            key = label.lower()

            d_emb = d.get("embedding")
            d_emb_v = d.get("embed_version")
            d_fresh = (
                isinstance(d_emb, list) and len(d_emb) > 0
                and d_emb_v == _v2d.EMBED_VERSION
            )
            if d_fresh:
                by_label_dino.setdefault(key, []).append(_np.asarray(d_emb, dtype=_np.float32))
                continue

            box = d.get("box") or []
            if not (isinstance(box, list) and len(box) == 4):
                continue
            mask_polys = None
            m = d.get("mask")
            if isinstance(m, dict):
                mask_polys = m.get("polygons")
            pending_by_ref.setdefault(ri, []).append(
                (di, key, [float(x) for x in box], mask_polys),
            )

    if pending_by_ref:
        if not _v2d.is_loaded():
            print("[v2-centroids] DINOv2 not loaded; backfill deferred")
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
                    image_pil = _open_upright(p)
                except Exception as e:
                    print(f"[v2-centroids] couldn't open {p}: {e}")
                    continue
                W, H = image_pil.size
                squares: list[PILImage.Image] = []
                kept: list[tuple[int, str]] = []
                for di, key, box, mask_polys in items:
                    x0 = max(0, int(round(box[0])))
                    y0 = max(0, int(round(box[1])))
                    x1 = min(W, int(round(box[2])))
                    y1 = min(H, int(round(box[3])))
                    if x1 - x0 < 4 or y1 - y0 < 4:
                        continue
                    crop = _v2d.inpaint_bbox_crop(image_pil, (x0, y0, x1, y1), mask_polys)
                    squares.append(_v2d.center_square_crop(crop))
                    kept.append((di, key))
                if not squares:
                    continue
                d_vecs = _v2d.encode_images_batch(squares)
                for k, (di, key) in enumerate(kept):
                    v = d_vecs[k]
                    refs[ri]["detections"][di]["embedding"] = [round(float(x), 6) for x in v.tolist()]
                    refs[ri]["detections"][di]["embed_version"] = _v2d.EMBED_VERSION
                    by_label_dino.setdefault(key, []).append(_np.asarray(v, dtype=_np.float32))
                    dirty = True
                print(
                    f"[v2-centroids] backfilled {len(kept)} embedding(s) for ref {ref.get('id')} "
                    f"(dinov{_v2d.EMBED_VERSION})"
                )

    if dirty:
        manifest["references"] = refs
        save_manifest(project_id, manifest)

    blank_label_count = 0
    for ref in refs:
        for d in ref.get("detections") or []:
            if not (d.get("label") or "").strip():
                blank_label_count += 1
    counts_dino = {k: len(v) for k, v in by_label_dino.items()}
    print(
        f"[v2-centroids] project={project_id} dino={counts_dino}"
        + (f" blank_label_dets={blank_label_count}" if blank_label_count else "")
    )
    # The mtime after any save above.
    new_mtime = _manifest_disk_mtime(project_id)
    if new_mtime > 0:
        with _V2_REFS_CACHE_LOCK:
            _V2_REFS_CACHE[project_id] = (new_mtime, by_label_dino)
    return by_label_dino


# Patch-level matching of detection crops against reference crops.

_PATCH_STORE_VERSION = 8
_PATCH_TOP_K_QUERY = int(os.environ.get("V2_PATCH_TOP_K", "10"))


def _v2_patch_store_path(project_id: str) -> Path:
    return project_dir(project_id) / f"v2_patches_v{_PATCH_STORE_VERSION}.npz"


def _v2_patch_match_enabled() -> bool:
    # When off, the specific resolver uses pooled-cosine scoring.
    return os.environ.get("V2_PATCH_MATCH", "on").lower() not in ("0", "off", "false")


def _v2_load_patch_store(project_id: str) -> dict[str, "np.ndarray"]:
    """The project's NPZ patch store as a flat dict, or {} when missing or unreadable."""
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
    """Atomic NPZ rewrite. The temp name ends in .npz, or np.savez_compressed would append one."""
    import numpy as _np
    p = _v2_patch_store_path(project_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.parent / f"{p.stem}.{os.getpid()}.{_uuid.uuid4().hex[:8]}.tmp.npz"
    _np.savez_compressed(tmp, **store)
    tmp.replace(p)


def _v2_load_or_backfill_patch_tokens(
    project_id: str,
) -> dict[str, list[tuple["np.ndarray", "np.ndarray"]]]:
    """Per-label [(tokens, fg_mask)] DINOv2 patch grids of the reference boxes,
    encoding and storing missing ones. NPZ keys are `<ref_id>__<det_idx>__`
    plus tokens or fg. Returns {} when patch matching is off or DINOv2 is not
    loaded."""
    import numpy as _np
    if not _v2_patch_match_enabled():
        return {}
    import v2_dinov2 as _v2d
    if not _v2d.is_loaded():
        return {}

    manifest = load_manifest(project_id) or {}
    refs = manifest.get("references") or []
    refs_dir = project_dir(project_id) / "references"
    store = _v2_load_patch_store(project_id)

    by_label_dino: dict[str, list[tuple[_np.ndarray, _np.ndarray]]] = {}
    # ri -> [(di, label, box, mask_polys)]
    pending_by_ref: dict[int, list[tuple[int, str, list[float], list | None]]] = {}

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
            if d_tokens is not None and d_fg is not None and d_tokens.shape[0] > 0:
                by_label_dino.setdefault(label, []).append((d_tokens, d_fg.astype(bool)))
                continue

            box = d.get("box") or []
            if not (isinstance(box, list) and len(box) == 4):
                continue
            mask_polys = None
            m = d.get("mask")
            if isinstance(m, dict):
                mask_polys = m.get("polygons")
            pending_by_ref.setdefault(ri, []).append(
                (di, label, [float(x) for x in box], mask_polys)
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
                image_pil = _open_upright(p)
            except Exception as e:
                print(f"[v2-patches] open failed {p}: {e}")
                continue
            W, H = image_pil.size
            for di, label, box, mask_polys in items:
                x0 = max(0, int(round(box[0])))
                y0 = max(0, int(round(box[1])))
                x1 = min(W, int(round(box[2])))
                y1 = min(H, int(round(box[3])))
                if x1 - x0 < 4 or y1 - y0 < 4:
                    continue
                clean = _v2d.inpaint_bbox_crop(image_pil, (x0, y0, x1, y1), mask_polys)
                square = _v2d.center_square_crop(clean)
                tokens, fg = _v2d.encode_image_patches(square)
                if tokens.shape[0] > 0:
                    store[f"{ref_id}__{di}__tokens"] = tokens.astype(_np.float16)
                    store[f"{ref_id}__{di}__fg"] = fg
                    by_label_dino.setdefault(label, []).append((tokens, fg))
                    added += 1
        if added > 0:
            try:
                _v2_save_patch_store(project_id, store)
                print(f"[v2-patches] backfilled {added} array(s) for project={project_id}")
            except Exception as e:
                print(f"[v2-patches] save failed (in-memory result still valid): {e}")

    counts_dino = {k: len(v) for k, v in by_label_dino.items()}
    print(f"[v2-patches] project={project_id} dino={counts_dino}")
    return by_label_dino


def _v2_score_labels_patchwise(
    q_tokens: "np.ndarray",
    q_fg: "np.ndarray",
    refs_by_label_patches: dict,
    *,
    top_k_query: int = _PATCH_TOP_K_QUERY,
) -> dict[str, float]:
    """{label_lower: score}, the mean of the top_k_query best cosine matches of the
    query foreground patches against the label's reference foreground patches.
    q_tokens is (P_q, D) and L2-normalised; q_fg is a (P_q,) bool mask."""
    import numpy as _np

    q = q_tokens[q_fg] if bool(q_fg.any()) else q_tokens
    if q.shape[0] == 0:
        return {}

    out: dict[str, float] = {}
    for label, refs in refs_by_label_patches.items():
        ref_chunks: list[_np.ndarray] = []
        for r_tokens, r_fg in refs:
            r_fg_bool = r_fg.astype(bool) if r_fg.dtype != bool else r_fg
            sel = r_tokens[r_fg_bool] if bool(r_fg_bool.any()) else r_tokens
            if sel.shape[0] > 0:
                # Stored tokens are float16.
                ref_chunks.append(sel.astype(_np.float32, copy=False))
        if not ref_chunks:
            continue
        R = _np.concatenate(ref_chunks, axis=0)  # (sum_n_fg_ref, D)

        # Best match per query patch, (P_q,).
        sims_per_q = (q.astype(_np.float32, copy=False) @ R.T).max(axis=1)

        k = min(top_k_query, int(sims_per_q.shape[0]))
        if k <= 0:
            continue
        top = _np.partition(sims_per_q, -k)[-k:]
        out[label] = float(top.mean())

    return out


def _v2_compute_class_thresholds(
    by_label_arr: dict,
    *,
    floor: float = 0.20,
    cap: float = 0.40,
    sigma_mult: float = 3.0,
    fallback: float = 0.40,
) -> dict[str, float]:
    """Per-label accept threshold max(floor, min(cap, mean - sigma_mult * std)) of the
    leave-one-out self-scores, or `fallback` below 3 references. cap equals
    fallback, so a threshold can only fall below the default."""
    import numpy as _np

    out: dict[str, float] = {}
    for key, arr in by_label_arr.items():
        if arr is None or arr.shape[0] < 3:
            out[key] = fallback
            continue

        n = int(arr.shape[0])
        total = arr.sum(axis=0)  # (D,)
        loo_centroids = (total - arr) / (n - 1)  # (N, D)
        norms = _np.linalg.norm(loo_centroids, axis=1, keepdims=True)
        norms = _np.maximum(norms, 1e-8)
        loo_centroids = loo_centroids / norms
        self_scores = (loo_centroids * arr).sum(axis=1)

        mean = float(self_scores.mean())
        std = float(self_scores.std())
        threshold = mean - sigma_mult * std
        threshold = max(floor, min(cap, threshold))
        out[key] = float(threshold)

    return out


def _v2_compute_reference_quality(refs: list[dict], by_label_arr: dict) -> dict:
    """Leave-one-out quality of each labelled reference detection, from DINOv2
    embeddings: {ref_id: {detection_idx: {self_score, other_score, other_label,
    quality, warning}}}. self_score is against the other references of its label."""
    import numpy as _np

    label_sum: dict[str, _np.ndarray] = {}
    label_count: dict[str, int] = {}
    for lab, arr in by_label_arr.items():
        if arr is None or arr.shape[0] == 0:
            continue
        label_sum[lab] = arr.sum(axis=0)
        label_count[lab] = int(arr.shape[0])

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

            self_score: float | None = None
            if label in label_count and label_count[label] > 1:
                loo_sum = label_sum[label] - q
                loo_count = label_count[label] - 1
                loo_centroid = loo_sum / loo_count
                n = float(_np.linalg.norm(loo_centroid))
                if n > 1e-8:
                    self_score = float((loo_centroid / n) @ q)

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


@app.get("/api/v2/projects/{project_id}/reference_quality")
async def v2_reference_quality(project_id: str):
    """Per-reference quality scores, after re-encoding stale reference embeddings."""
    if not project_dir(project_id).exists():
        raise HTTPException(404)
    try:
        async with state["gpu_lock"]:
            by_label = await asyncio.to_thread(
                _v2_load_or_backfill_reference_embeddings, project_id)
    except Exception as e:
        print(f"[v2-ref-quality] load failed: {e}")
        return {"references": {}}
    arr = _v2_stack_refs(by_label)
    manifest = load_manifest(project_id) or {}
    refs = manifest.get("references") or []
    quality = _v2_compute_reference_quality(refs, arr)
    return {"references": quality}


# kNN scoring averages a label's K closest references, or all of them when it has fewer.
KNN_TOP_K = 3


def _v2_stack_refs(by_label: dict) -> dict[str, "np.ndarray"]:
    """Stack each label's reference embeddings into an (N, D) float32 array."""
    import numpy as _np
    out: dict[str, _np.ndarray] = {}
    for key, vecs in by_label.items():
        if not vecs:
            continue
        out[key] = _np.stack(
            [_np.asarray(v, dtype=_np.float32) for v in vecs], axis=0,
        )
    return out


# Floor on cosine distance, capping a reference's weight at 1000.
_KNN_DIST_EPS = 1e-3


def _v2_score_labels_knn_weighted(
    q: "np.ndarray",
    refs_by_label_arr: dict,
    k: int = KNN_TOP_K,
) -> dict[str, float]:
    """{label_key: mean of the k highest cosine similarities, each weighted by
    1 / max(1 - sim, eps)}. q and the references are L2-normalised."""
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


def _v2_resolve_label_specific(
    embedding: list[float],
    gd_label: str | None,
    refs_by_label: dict,
    label_display: dict[str, str],
    *,
    reject_embed_threshold: float = 0.4,
    # From _v2_compute_class_thresholds; replaces reject_embed_threshold per label.
    class_thresholds: dict[str, float] | None = None,
    # Patch grids. A label's patch score replaces its pooled cosine.
    query_patch_tokens: "np.ndarray | None" = None,
    query_patch_fg: "np.ndarray | None" = None,
    refs_by_label_patches: dict | None = None,
    # Top-two margin below which a kept detection is flagged ambiguous;
    # defaults to V2_AMBIGUOUS_MARGIN.
    ambiguous_margin: float | None = None,
) -> dict:
    """Resolver for specific datasets: the reference embeddings choose the
    label; `gd_label`, the segmenter's, stands when there are none."""
    import numpy as _np

    sims_dino: dict[str, float] = {}
    sims_patch: dict[str, float] = {}

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

    if embedding and refs_by_label:
        q = _np.asarray(embedding, dtype=_np.float32)
        sims_dino = _v2_score_labels_knn_weighted(q, refs_by_label)

    # Labels without reference patches keep their pooled score.
    if sims_patch:
        merged_dino: dict[str, float] = {}
        for k in set(sims_dino.keys()) | set(sims_patch.keys()):
            merged_dino[k] = sims_patch.get(k, sims_dino.get(k, 0.0))
        sims_dino = merged_dino
    sims = dict(sims_dino)

    sorted_sims_desc = sorted(sims.values(), reverse=True)
    embed_margin = (
        sorted_sims_desc[0] - sorted_sims_desc[1]
        if len(sorted_sims_desc) >= 2 else
        0.0
    )
    best_key = max(sims, key=sims.get) if sims else None
    best_sim = sims[best_key] if best_key else None

    # The embedding's top label, never the segmenter's.
    pred_key = best_key
    pred_label = label_display.get(best_key, best_key) if best_key else None
    pred_source: str | None = "embed" if best_key else None

    # The ambiguous flag only marks a detection for review; the label stands.
    _ambiguous_margin = (
        float(ambiguous_margin)
        if ambiguous_margin is not None
        else float(os.environ.get("V2_AMBIGUOUS_MARGIN", "0.005"))
    )

    def _make(reject_reason: str | None, *, rejected: bool) -> dict:
        final_sim = sims.get(pred_key) if pred_key else None
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
            "sims_dino": {label_display.get(k, k): round(float(v), 4) for k, v in sims_dino.items()},
            "sims_patch": {label_display.get(k, k): round(float(v), 4) for k, v in sims_patch.items()},
            "patch_match_used": bool(sims_patch),
        }

    if not sims:
        if gd_label:
            pred_key = gd_label.strip().lower()
            pred_label = gd_label
            pred_source = "gd"
        return _make(None, rejected=False)

    final_sim = sims.get(pred_key) if pred_key else None
    threshold_used = reject_embed_threshold
    if class_thresholds is not None and pred_key is not None:
        threshold_used = class_thresholds.get(pred_key, reject_embed_threshold)
    if final_sim is not None and final_sim < threshold_used:
        return _make("embed", rejected=True)

    try:
        sims_str = ", ".join(
            f"{label_display.get(k, k)}={v:.3f}"
            for k, v in sorted(sims.items(), key=lambda kv: -kv[1])
        )
        print(
            f"[v2-resolve-specific] gd={gd_label!r} "
            f"-> pred={pred_label!r} src={pred_source} "
            f"margin={embed_margin:.3f} thr={threshold_used:.3f} sims=[{sims_str}]"
        )
    except Exception:
        pass

    return _make(None, rejected=False)


def _dataset_type(project_id: str) -> dict:
    """Type "specific" when the dataset has reference images, else "general"."""
    try:
        has_refs = bool((load_manifest(project_id, copy=False) or {}).get("references"))
    except Exception:
        has_refs = False
    return {
        "type": "specific" if has_refs else "general",
        "reason": "reference images present" if has_refs else "no reference images",
        "source": "auto",
    }


@app.get("/api/v2/projects/{project_id}/dataset-type")
async def v2_dataset_type(project_id: str):
    if not project_dir(project_id).exists():
        raise HTTPException(404)
    return await asyncio.to_thread(_dataset_type, project_id)


# Licence details of recent Openverse results by image URL, kept with the
# imports made from them.
_OPENVERSE_META: OrderedDict[str, dict] = OrderedDict()
_OPENVERSE_META_MAX = 5000
_OPENVERSE_META_KEYS = ("title", "creator", "license", "license_version", "license_url",
                        "attribution", "foreign_landing_url")


def _remember_openverse(results: list[dict]) -> None:
    for r in results:
        url = r.get("url") if isinstance(r, dict) else None
        if isinstance(url, str) and url:
            _OPENVERSE_META[url] = {k: r[k] for k in _OPENVERSE_META_KEYS if r.get(k)}
            _OPENVERSE_META.move_to_end(url)
    while len(_OPENVERSE_META) > _OPENVERSE_META_MAX:
        _OPENVERSE_META.popitem(last=False)


@app.get("/api/openverse/search")
async def openverse_search(q: str, count: int = 5, commercial: bool = False):
    """Up to `count` (max 250) Creative Commons images from Openverse; `commercial`
    keeps licences that allow commercial use."""
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
    _remember_openverse(results)
    # Drops results whose thumbnail is missing, an HTML page or a tiny placeholder.
    try:
        validated = await loop.run_in_executor(
            None, lambda: openverse.validate_thumbnails(results),
        )
        results = validated
    except Exception as e:
        print(f"[openverse] thumbnail validation failed (returning unfiltered): {e}")
    return {"query": query, "results": results, "commercial": commercial}




def collect_tags(manifest: dict) -> list[str]:
    """manifest['tags'], then any other labels on editedBoxes, reference detections
    and import detections and edited boxes, lowercased."""
    seen: set[str] = set()
    ordered: list[str] = []

    for t in manifest.get("tags", []) or []:
        tl = (t or "").strip().lower()
        if tl and tl not in seen:
            seen.add(tl)
            ordered.append(tl)

    extra_labels: set[str] = set()

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

    for ref in manifest.get("references", []) or []:
        if not isinstance(ref, dict):
            continue
        for d in ref.get("detections", []) or []:
            if not isinstance(d, dict):
                continue
            label = (d.get("label") or "").strip().lower()
            if label and label != "new":
                extra_labels.add(label)

    for imp in manifest.get("imports", []) or []:
        if not isinstance(imp, dict):
            continue
        for d in imp.get("detections", []) or []:
            if not isinstance(d, dict):
                continue
            label = (d.get("label") or "").strip().lower()
            if label and label != "new":
                extra_labels.add(label)
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


# /api/projects responses, fresh for _PROJECTS_RESPONSE_TTL_S seconds, until
# a listed project's manifest changes on disk, and until the engine saves,
# creates or deletes any dataset: two saves in one file-time tick leave the
# mtime as it was, and a new dataset is in no cached response.

_PROJECTS_RESPONSE_CACHE: dict[tuple, dict] = {}
_PROJECTS_RESPONSE_LOCK = __import__("threading").Lock()
_PROJECTS_RESPONSE_TTL_S = 2.0
_PROJECTS_GEN = [0]


def _projects_changed() -> None:
    with _PROJECTS_RESPONSE_LOCK:
        _PROJECTS_GEN[0] += 1


def _projects_response_cache_get(key: tuple) -> object | None:
    """The cached response, or None when missing, expired or a listed manifest changed."""
    import time as _t
    with _PROJECTS_RESPONSE_LOCK:
        entry = _PROJECTS_RESPONSE_CACHE.get(key)
        if entry is None:
            return None
        age = _t.time() - entry["ts"]
        if age > _PROJECTS_RESPONSE_TTL_S or entry["gen"] != _PROJECTS_GEN[0]:
            return None
        for pid, mtime in entry["mtimes"].items():
            if _manifest_disk_mtime(pid) > mtime:
                return None
        return entry["body"]


def _projects_response_cache_put(key: tuple, body: object, mtimes: dict[str, float], gen: int) -> None:
    """Caches a response built from the state at generation `gen`."""
    import time as _t
    with _PROJECTS_RESPONSE_LOCK:
        if len(_PROJECTS_RESPONSE_CACHE) > 64:
            _PROJECTS_RESPONSE_CACHE.clear()
        _PROJECTS_RESPONSE_CACHE[key] = {
            "ts": _t.time(),
            "gen": gen,
            "body": body,
            "mtimes": dict(mtimes),
        }


# (project_id, "<folder or manifest key>/<filename>") -> blurhash, filled on first use.
_BLURHASH_CACHE: dict[tuple[str, str], str] = {}


# LRU cache of served image bytes.

import collections as _collections


_IMAGE_CACHE: "_collections.OrderedDict[tuple[str, str, str], tuple[bytes, str, float]]" = _collections.OrderedDict()
_IMAGE_CACHE_LOCK = __import__("threading").Lock()
_IMAGE_CACHE_BYTES = 0
_IMAGE_CACHE_BUDGET_BYTES = 0  # set at startup by _size_image_cache


def _set_image_cache_budget(budget_bytes: int) -> None:
    global _IMAGE_CACHE_BUDGET_BYTES
    _IMAGE_CACHE_BUDGET_BYTES = max(0, int(budget_bytes))


def _image_cache_get(project_id: str, subdir: str, filename: str, disk_mtime: float) -> tuple[bytes, str] | None:
    """(bytes, content_type), or None when missing or the file's mtime differs."""
    key = (project_id, subdir, filename)
    with _IMAGE_CACHE_LOCK:
        entry = _IMAGE_CACHE.get(key)
        if entry is None:
            return None
        data, ctype, cached_mtime = entry
        if cached_mtime != disk_mtime:
            global _IMAGE_CACHE_BYTES
            _IMAGE_CACHE_BYTES -= len(data)
            del _IMAGE_CACHE[key]
            return None
        _IMAGE_CACHE.move_to_end(key)
        return data, ctype


def _image_cache_put(project_id: str, subdir: str, filename: str, data: bytes, ctype: str, disk_mtime: float) -> None:
    """Insert, evicting least recently used entries to fit the budget. Entries over
    half the budget are not cached."""
    if _IMAGE_CACHE_BUDGET_BYTES <= 0:
        return
    if len(data) > _IMAGE_CACHE_BUDGET_BYTES // 2:
        return
    key = (project_id, subdir, filename)
    global _IMAGE_CACHE_BYTES
    with _IMAGE_CACHE_LOCK:
        old = _IMAGE_CACHE.pop(key, None)
        if old is not None:
            _IMAGE_CACHE_BYTES -= len(old[0])
        _IMAGE_CACHE[key] = (data, ctype, disk_mtime)
        _IMAGE_CACHE_BYTES += len(data)
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
    """Serve an image file through the LRU cache, with ETag and Cache-Control headers."""
    from fastapi.responses import Response, FileResponse
    if not full_path.exists():
        raise HTTPException(404)
    try:
        st = full_path.stat()
        disk_mtime = st.st_mtime
    except OSError:
        disk_mtime = 0.0
    etag = f'W/"{int(disk_mtime)}-{st.st_size}"' if disk_mtime else None
    extra_headers = {
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
    try:
        data = await asyncio.to_thread(full_path.read_bytes)
    except Exception as e:
        print(f"[image-cache] read failed for {full_path}: {e}")
        return FileResponse(str(full_path), headers=extra_headers)
    _image_cache_put(project_id, subdir, filename, data, ctype, disk_mtime)
    return Response(content=data, media_type=ctype, headers=extra_headers)


# Must match the palette in ui/app/v2/labelColours.ts, in order.
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
    """The UI's signed 32-bit string hash, `_hash` in BoxEditor and labelColours."""
    h = 0
    for ch in s:
        h = (h * 31 + ord(ch)) & 0xFFFFFFFF
        if h >= 0x80000000:
            h -= 0x100000000
    return h


def _build_label_colour_map(
    all_labels: list[str],
) -> dict[str, tuple[int, int, int]]:
    """Label -> palette colour, avoiding collisions. Must match the UI's
    buildProjectLabelColourMap."""
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
    bg_dim: float = 0.28,        # 0 is black, 1 is undimmed
    bg_blur_radius: float = 5.0,
    bg_desat: float = 0.65,      # 1 is greyscale
    tint_strength: float = 0.34, # 1 is solid label colour
) -> "PILImage.Image":
    """Downsized RGB preview: kept masked detections bright and tinted by label
    over a dimmed, blurred background, or the plain downsized image if none."""
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

    union_mask = PILImage.new("L", base.size, 0)
    drw = ImageDraw.Draw(union_mask)
    for d in kept:
        for polygon in (d["mask"].get("polygons") or []):
            if len(polygon) < 3:
                continue
            pts = [(int(p[0] * scale), int(p[1] * scale)) for p in polygon]
            drw.polygon(pts, fill=255)

    bg_layer = base
    if bg_desat > 0:
        grey = base.convert("L").convert("RGB")
        bg_layer = PILImage.blend(base, grey, bg_desat)
    black = PILImage.new("RGB", base.size, (0, 0, 0))
    dimmed = PILImage.blend(bg_layer, black, 1.0 - bg_dim)
    if bg_blur_radius > 0:
        dimmed = dimmed.filter(ImageFilter.GaussianBlur(radius=bg_blur_radius))
    feather = union_mask.filter(ImageFilter.GaussianBlur(radius=1.0))
    composite = PILImage.composite(base, dimmed, feather)

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
    """Cached labelled preview, by import_id. The _v4 suffix versions the render style."""
    d = project_dir(project_id) / "thumbs"
    d.mkdir(exist_ok=True)
    return d / f"{import_id}__lp_v4.jpg"


def _invalidate_labelled_preview(project_id: str, import_id: str) -> None:
    """Delete one import's cached labelled preview. Called on label edits and deletes."""
    try:
        _labelled_preview_path(project_id, import_id).unlink(missing_ok=True)
    except Exception as e:
        print(f"[labelled-preview] invalidate failed for {project_id}/{import_id}: {e}")


# (project_id, kind) -> lock serialising first renders of one thumbnail.
_THUMB_RENDER_LOCKS: dict[tuple[str, str], asyncio.Lock] = {}


def _thumb_render_lock(project_id: str, kind: str) -> asyncio.Lock:
    key = (project_id, kind)
    lock = _THUMB_RENDER_LOCKS.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _THUMB_RENDER_LOCKS[key] = lock
    return lock


def _unique_tmp_suffix() -> str:
    """Temp-file suffix unique per call, not only per process."""
    return f".{os.getpid()}.{_uuid.uuid4().hex[:8]}.tmp.jpg"


def _preview_source(imp: "dict | None", detections: list) -> list:
    """Boxes for a labelled preview: the edited boxes once edited, even when
    empty, else `detections`."""
    edited = (imp or {}).get("editedBoxes")
    if isinstance(edited, list) and (edited or (imp or {}).get("editedBoxesSet")):
        return edited
    return detections


def _bake_labelled_preview_sync(
    project_id: str,
    import_id: str,
    src_path: "Path",
    detections: list[dict],
    project_labels: list[str],
) -> None:
    """Render and store the labelled preview JPEG. Runs in an executor; never raises."""
    cached_path = _labelled_preview_path(project_id, import_id)
    # Boxes edited by hand, even during a labelling run, win over `detections`.
    with _MANIFEST_CACHE_LOCK:
        imp = _MANIFEST_IMPORT_INDEX.get(project_id, {}).get(import_id)
    detections = _preview_source(imp, detections)
    try:
        with PILImage.open(src_path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            preview = _render_labelled_preview(im, detections, project_labels)
        cached_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cached_path.with_suffix(_unique_tmp_suffix())
        preview.save(tmp, format="JPEG", quality=72, optimize=True, progressive=True)
        store.replace(tmp, cached_path)
    except Exception as e:
        print(f"[labelled-preview] bake failed for {project_id}/{import_id}: {e}")


def _encode_blurhash_from_path(path: "Path") -> str | None:
    """Blurhash of the image at `path`, or None on any failure."""
    if not path.exists():
        return None
    try:
        import blurhash as _bh
        from PIL import Image as _PIL
        with _PIL.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
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


# Manifest key -> folder of the files it lists.
_BLURHASH_FOLDERS = {"references": "references", "imports": "images"}
_BLURHASH_TASKS: set = set()
# Datasets with a backfill running.
_BLURHASH_PROJECTS: set[str] = set()


def _encode_blurhashes(project_id: str, items: list[tuple[str, str]]) -> list[tuple[str, str, str]]:
    """Blurhashes of `items`, [(manifest key, filename)], as (key, filename, hash),
    also put in the cache. Runs in an executor; never raises."""
    encoded: list[tuple[str, str, str]] = []
    for key, fn in items:
        try:
            path = project_dir(project_id) / _BLURHASH_FOLDERS.get(key, key) / fn
            h = _encode_blurhash_from_path(path)
            if h:
                _BLURHASH_CACHE[(project_id, f"{key}/{fn}")] = h
                encoded.append((key, fn, h))
        except Exception as e:
            print(f"[blurhash] encode failed for {project_id}/{key}/{fn}: {e}")
    return encoded


async def _blurhash_backfill(project_id: str, items: list[tuple[str, str]]) -> None:
    """Encodes the missing blurhashes of `items` and stores them in the manifest,
    under its write lock. Never raises."""
    try:
        loop = asyncio.get_running_loop()
        encoded = await loop.run_in_executor(None, _encode_blurhashes, project_id, items)
        if not encoded:
            return
        lock = await _manifest_write_lock(project_id)
        async with lock:
            manifest = load_manifest(project_id)
            if not manifest:
                return
            changed = 0
            for key, fn, h in encoded:
                for entry in manifest.get(key) or []:
                    if isinstance(entry, dict) and entry.get("filename") == fn and not entry.get("blurhash"):
                        entry["blurhash"] = h
                        changed += 1
            if changed:
                save_manifest(project_id, manifest)
                print(f"[blurhash] backfilled {changed} hash(es) for {project_id}")
    except Exception as e:
        print(f"[blurhash] backfill failed for {project_id}: {e}")


def _start_blurhash_backfill(loop, project_id: str, items: list[tuple[str, str]]) -> None:
    """One backfill per dataset at a time; the next read queues what is still missing."""
    if not items or project_id in _BLURHASH_PROJECTS:
        return
    _BLURHASH_PROJECTS.add(project_id)
    task = loop.create_task(_blurhash_backfill(project_id, items))
    _BLURHASH_TASKS.add(task)

    def _done(t):
        _BLURHASH_TASKS.discard(t)
        _BLURHASH_PROJECTS.discard(project_id)

    task.add_done_callback(_done)


def _cached_blurhash(project_id: str, filename: str, subdir: str) -> str | None:
    """Blurhash of <project_id>/<subdir>/<filename>, encoded on the first miss."""
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


# The name the favouritedBy lists hold for this machine's user.
LOCAL_USER = "local"


@app.get("/api/projects")
async def list_projects(
    offset: int = 0,
    limit: int | None = None,
    sort: str | None = None,
    q: str | None = None,
):
    """Dataset cards as a list, or as {total, items, offset, limit} when `limit`
    is given."""
    q_norm = (q or "").strip().lower()
    _cache_key = (int(offset), -1 if limit is None else int(limit), sort or "", q_norm)
    _cached = _projects_response_cache_get(_cache_key)
    if _cached is not None:
        return _cached
    # Read before the scan: a save during it leaves the result uncached.
    _gen = _PROJECTS_GEN[0]

    running_projects: set[str] = set()
    try:
        for j in state["jobs"].jobs.values():
            if j.status in ("queued", "running") and j.project:
                running_projects.add(j.project)
    except Exception as e:
        print(f"[list_projects] failed to read job state: {e}")
    items = []
    # dataset_id -> the {id, name} of the Project that lists it.
    try:
        _ds_container = containers.dataset_container_index()
    except Exception as e:
        print(f"[list_projects] container index failed: {e}")
        _ds_container = {}

    cache_mtimes: dict[str, float] = {}
    for _pid in _iter_project_ids():
        manifest_mtime = _manifest_disk_mtime(_pid)
        cache_mtimes[_pid] = manifest_mtime
        # The card sidecar when it is at least as new as the manifest, else the manifest.
        card = _read_workspace_card_sidecar(_pid)
        try:
            card_mtime = _workspace_card_sidecar_path(_pid).stat().st_mtime if card else 0.0
        except OSError:
            card_mtime = 0.0
        if card is None or card_mtime < manifest_mtime:
            _kick_workspace_card_refresh(_pid)
            # copy=False: read only.
            card = _build_workspace_card_payload(_pid, load_manifest(_pid, copy=False))
        items.append({
            **{k: card[k] for k in _CARD_KEYS if k in card},
            "favouritedByMe": LOCAL_USER in (card.get("favouritedBy") or []),
            "running": (card.get("id") or _pid) in running_projects,
        })
    for it in items:
        info = _ds_container.get(str(it.get("id")))
        it["container"] = {"id": info["id"], "name": info["name"]} if info else None

    # Filtered before sorting and paging, so `total` counts the matches.
    if q_norm:
        def _matches(it: dict) -> bool:
            haystack = [it.get("name"), *(it.get("tags") or []),
                        *(it.get("label_aliases") or {}).values()]
            return any(q_norm in str(h).lower() for h in haystack if h)
        items = [it for it in items if _matches(it)]

    # Layered stable sorts; favourites come first.
    if (sort or "").strip().lower() == "newest":
        items.sort(key=lambda x: x.get("createdAt") or "", reverse=True)
    else:
        items.sort(key=lambda x: x.get("updatedAt") or "", reverse=True)
    items.sort(key=lambda x: 0 if x.get("favouritedByMe") else 1)

    total = len(items)
    paginate = limit is not None and limit > 0
    if paginate:
        off = max(0, int(offset))
        lim = max(0, int(limit))
        page = items[off:off + lim]
    else:
        page = items

    for it in page:
        cover = it.get("thumbnail")
        subdir = it.pop("_cover_subdir", None)
        if cover and subdir:
            try:
                it["cover_blurhash"] = _cached_blurhash(it["id"], cover, subdir)
            except Exception as e:
                print(f"[list_projects] blurhash skipped for {it['id']}: {e}")

    if paginate:
        body = {"total": total, "items": page, "offset": off, "limit": lim}
    else:
        body = page
    _projects_response_cache_put(_cache_key, body, cache_mtimes, _gen)
    return body


# Projects: containers of datasets.

# The Project fields the UI reads; a project.json written by another version can hold more.
_CONTAINER_KEYS = ("id", "name", "cover", "coverLuma", "cover_updated", "dataset_ids", "created", "updated")


def _container_payload(c: dict) -> dict:
    return {k: c[k] for k in _CONTAINER_KEYS if k in c}


def _container_card(c: dict) -> dict:
    """Compact shape for the workspace grid: cover, last-updated, count."""
    return {
        "id": c.get("id"),
        "name": c.get("name"),
        "cover": c.get("cover"),
        "n_datasets": len(c.get("dataset_ids") or []),
        "updated": c.get("updated"),
        "cover_updated": c.get("cover_updated"),
        "created": c.get("created"),
    }


def _container_detail(c: dict) -> dict:
    """Project page payload: the container and its datasets."""
    datasets = []
    for did in (c.get("dataset_ids") or []):
        m = load_manifest(did, copy=False) or {}
        datasets.append({
            "id": did,
            "name": m.get("name") or did,
            "cover": m.get("cover"),
            "n_images": len(m.get("imports") or m.get("results") or []),
            "hasModel": bool(m.get("hasModel")),
            "updated": m.get("updatedAt"),
            "derived": bool(m.get("derived")),
        })
    return {**_container_payload(c), "datasets": datasets}


class ContainerCreateIn(BaseModel):
    name: str


@app.post("/api/containers")
async def container_create(payload: ContainerCreateIn):
    name = (payload.name or "").strip() or "Untitled project"
    c = containers.create_container(name)
    add_event("container_create", container=c["id"], name=name)
    return _container_payload(c)


@app.get("/api/containers")
async def container_list():
    cards = [_container_card(c) for c in containers.list_containers()]
    cards.sort(key=lambda c: (c.get("updated") or ""), reverse=True)
    return {"containers": cards}


@app.get("/api/containers/{container_id}")
async def container_get(container_id: str):
    c = containers.load_container(container_id)
    if not c:
        raise HTTPException(404, "project not found")
    # coverLuma for a cover stored without one; saved without bumping `updated`.
    if c.get("cover") and c.get("coverLuma") is None:
        try:
            from PIL import ImageStat as _ImageStat
            with PILImage.open(_container_cover_path(container_id)) as _im:
                _im = _im.convert("RGB")
                _w, _h = _im.size
                _band = _im.crop((0, int(_h * 0.5), _w, _h)).convert("L")
                c["coverLuma"] = round(_ImageStat.Stat(_band).mean[0])
        except Exception as e:
            print(f"[container cover-luma backfill] {container_id}: {e}")
            c["coverLuma"] = 0  # not retried
        try:
            containers.save_container(c, bump_updated=False)
        except Exception:
            pass
    return _container_detail(c)


class ContainerPatchIn(BaseModel):
    name: str | None = None
    cover: str | None = None


@app.patch("/api/containers/{container_id}")
async def container_patch(container_id: str, payload: ContainerPatchIn):
    c = containers.load_container(container_id)
    if not c:
        raise HTTPException(404, "project not found")
    if payload.name is not None:
        c["name"] = ((payload.name or "").strip() or "Untitled project")[:120]
    if payload.cover is not None:
        c["cover"] = payload.cover or None
        c["cover_updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    containers.save_container(c)
    return _container_payload(c)


@app.delete("/api/containers/{container_id}")
async def container_delete(container_id: str):
    c = containers.load_container(container_id)
    if not c:
        raise HTTPException(404, "project not found")
    ids = list(c.get("dataset_ids") or [])
    if any(_dataset_job_active(did) for did in ids):
        raise HTTPException(409, "a job is running on one of this project's datasets")
    # The dataset folders move out first; if one cannot, nothing is detached.
    try:
        containers.delete_container(container_id)
    except OSError as e:
        raise HTTPException(409, f"{e}; close it and try again")
    # Datasets are detached, not deleted.
    for did in ids:
        lock = await _manifest_write_lock(did)
        async with lock:
            m = load_manifest(did)
            if m and (m.get("container_id") or "") == container_id:
                m["container_id"] = ""
                save_manifest(did, m)
    add_event("container_delete", container=container_id)
    return {"ok": True}


@app.post("/api/containers/{container_id}/datasets/{dataset_id}")
async def container_add_dataset(container_id: str, dataset_id: str):
    c = containers.load_container(container_id)
    if not c:
        raise HTTPException(404, "project not found")
    m = load_manifest(dataset_id)
    if not m:
        raise HTTPException(404, "dataset not found")
    lock = await _manifest_write_lock(dataset_id)
    async with lock:
        m = load_manifest(dataset_id)
        m["container_id"] = container_id
        save_manifest(dataset_id, m)
        # The folder moves into the Project's folder, unless a job holds its paths.
        if not _dataset_job_active(dataset_id):
            store.move_dataset(dataset_id, container_id)
    if dataset_id not in (c.get("dataset_ids") or []):
        c.setdefault("dataset_ids", []).append(dataset_id)
        containers.save_container(c)
    add_event("dataset_add", container=container_id, dataset=dataset_id)
    return {"ok": True}


@app.delete("/api/containers/{container_id}/datasets/{dataset_id}")
async def container_remove_dataset(container_id: str, dataset_id: str):
    c = containers.load_container(container_id)
    if not c:
        raise HTTPException(404, "project not found")
    m = load_manifest(dataset_id)
    if m:
        lock = await _manifest_write_lock(dataset_id)
        async with lock:
            m = load_manifest(dataset_id)
            if (m.get("container_id") or "") == container_id:
                m["container_id"] = ""
                save_manifest(dataset_id, m)
            if not _dataset_job_active(dataset_id):
                store.move_dataset(dataset_id, None)
    c["dataset_ids"] = [d for d in (c.get("dataset_ids") or []) if d != dataset_id]
    containers.save_container(c)
    add_event("dataset_remove", container=container_id, dataset=dataset_id)
    return {"ok": True}


def _container_cover_path(container_id: str) -> Path:
    return workspace.dir() / "_storage" / "containers" / container_id / "cover.jpg"


@app.post("/api/containers/{container_id}/cover")
async def container_upload_cover(container_id: str, file: UploadFile = File(...)):
    """Set the Project cover, stored as JPEG."""
    c = containers.load_container(container_id)
    if not c:
        raise HTTPException(404, "project not found")
    data = await _read_upload(file, "cover")
    try:
        img = PILImage.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img).convert("RGB")
    except PILImage.DecompressionBombError:
        raise HTTPException(413, "image dimensions exceed the maximum allowed")
    except Exception:
        raise HTTPException(400, "not an image")
    img.thumbnail((1600, 1600), PILImage.LANCZOS)
    img = _upscale_cover_if_small(img, 1280)
    # Mean luminance, 0-255, of the bottom half, where the title is drawn.
    try:
        from PIL import ImageStat as _ImageStat
        _w, _h = img.size
        _band = img.crop((0, int(_h * 0.5), _w, _h)).convert("L")
        c["coverLuma"] = round(_ImageStat.Stat(_band).mean[0])
    except Exception:
        c["coverLuma"] = 0
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    store._atomic_write(_container_cover_path(container_id), buf.getvalue())
    c["cover"] = "cover.jpg"
    # The file name is fixed, so this tells the UI the cover changed.
    c["cover_updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    containers.save_container(c)
    return {"cover": "cover.jpg", "coverLuma": c.get("coverLuma", 0), "cover_updated": c["cover_updated"]}


@app.get("/api/containers/{container_id}/cover")
async def container_serve_cover(container_id: str):
    c = containers.load_container(container_id)
    if not c:
        raise HTTPException(404, "project not found")
    path = _container_cover_path(container_id)
    if not c.get("cover") or not path.is_file():
        raise HTTPException(404, "no cover")
    return FileResponse(path, headers={"Cache-Control": "public, max-age=900"})


@app.get("/api/containers/{container_id}/activity")
async def container_activity(container_id: str, limit: int = 100):
    """Recent activity across the Project and its datasets."""
    c = containers.load_container(container_id)
    if not c:
        raise HTTPException(404, "project not found")
    limit = max(1, min(int(limit or 100), 300))
    return {"activity": audit.list_activity(container_id, c.get("dataset_ids") or [], limit=limit)}


# Split reads: /overview has project metadata and per-tile summaries,
# /annotations the detections and edited boxes, without embeddings.

def _kick_blurhash_backfill(project_id: str, m: dict) -> None:
    """Copy cached blurhashes onto the manifest dict and queue encodes for the rest."""
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
    if not needs:
        return
    try:
        _start_blurhash_backfill(asyncio.get_running_loop(), project_id, needs)
    except RuntimeError:
        # A worker thread, as /overview builds in: the event loop runs it.
        loop = state.get("loop")
        if loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(_start_blurhash_backfill, loop, project_id, needs)


def _sort_imports_desc(entries: list[dict]) -> list[dict]:
    """Newest createdAt first, entries without one last, as the UI's
    compareImportedMediaDesc."""
    def key(e: dict) -> tuple[int, float, str]:
        # (has_no_ts, -ms, id)
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
    """Per-tile summaries of references or imports: counts and labels, no geometry."""
    out: list[dict] = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        dets = e.get("detections") or []
        edited = e.get("editedBoxes") if isinstance(e.get("editedBoxes"), list) else None
        edited_set = bool(e.get("editedBoxesSet"))
        # Edited boxes count once edited, even when empty.
        if edited is not None and edited_set:
            counted = edited
            label_pool = [str(b.get("label") or "") for b in edited if isinstance(b, dict)]
        elif edited:
            counted = edited
            label_pool = [str(b.get("label") or "") for b in edited if isinstance(b, dict)]
        else:
            counted = [d for d in dets if isinstance(d, dict) and not d.get("rejected")]
            label_pool = [
                str(
                    d.get("pred_label")
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
        # Origin of a URL import, such as kind "openverse" and its url; else None.
        src = e.get("source") if isinstance(e.get("source"), dict) else None
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
            "label_stats": label_stats_pool,
            "has_edits": bool(edited),
            "background": bool(e.get("background")),
            "source": src,
            # Epoch ms; the labelled preview's cache-buster.
            "labelledAt": e.get("labelledAt"),
            "n_augmentations": int(e.get("n_augmentations") or 0),
            # A derived crop's label in the parent dataset, else None.
            "derivedLabel": ((e.get("derivedFrom") or {}).get("label") or None)
            if isinstance(e.get("derivedFrom"), dict) else None,
        })
    return out


_EMBEDDING_FIELDS = (
    "embedding",
    "embed_version",
    # Fields of stored detections that the engine does not write.
    "siglip_embedding",
    "embedding_siglip",
    "siglip_version",
)

# Left out of the bulk /annotations; /annotations/{import_id} keeps them.
_HEAVY_DETECTION_FIELDS = (
    "mask",
    "polygons",
    "crop_jpg_b64",
    "embed_sims",
    "embed_sims_dino",
    # A field of stored detections that the engine does not write.
    "embed_sims_siglip",
)


def _strip_embedding(d: dict) -> dict:
    """Shallow copy of `d` without the _EMBEDDING_FIELDS."""
    slim = dict(d)
    for k in _EMBEDDING_FIELDS:
        slim.pop(k, None)
    return slim


def _strip_embedding_and_heavy(d: dict) -> dict:
    """_strip_embedding, also without the _HEAVY_DETECTION_FIELDS."""
    slim = dict(d)
    for k in _EMBEDDING_FIELDS:
        slim.pop(k, None)
    for k in _HEAVY_DETECTION_FIELDS:
        slim.pop(k, None)
    return slim


def _strip_edited_box(b: dict) -> dict:
    """An edited box without its mask polygons."""
    slim = dict(b)
    slim.pop("mask", None)
    slim.pop("polygons", None)
    return slim


def _detection_annotations(entries: list[dict], *, include_edits: bool, include_heavy: bool = False) -> dict[str, dict]:
    """{entry id: {detections, editedBoxes, timings}} without embeddings; masks
    and score dicts only with include_heavy."""
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
            # Without editedBoxesSet, an empty editedBoxes means unedited.
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
    """The /overview payload, run in a worker thread. imports_limit and
    imports_offset page the import tiles; imports_total counts them all."""
    _kick_blurhash_backfill(project_id, m)
    all_imports = _sort_imports_desc(m.get("imports") or [])
    if imports_limit is not None:
        sliced_imports = all_imports[imports_offset : imports_offset + imports_limit]
    else:
        sliced_imports = all_imports
    # Labels of the last labelling run; the current tags when that is not
    # recorded but an image is labelled.
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
        "tags": m.get("tags") or [],
        "labelsLastRun": labels_last_run,
        "settingsLastRun": m.get("settingsLastRun"),
        "label_aliases": m.get("label_aliases") or {},
        "labelColours": m.get("labelColours") or {},
        # import id -> "good", "bad" or "unsure".
        "verdicts": m.get("verdicts") or {},
        "cover": m.get("cover"),
        "cover_uploaded": bool(m.get("cover_uploaded")),
        "cover_blurhash": m.get("cover_blurhash"),
        "v2": bool(m.get("v2")),
        "createdAt": m.get("createdAt"),
        "updatedAt": m.get("updatedAt"),
        "dataset_type": _dataset_type(project_id),
        "derived": m.get("derived") or None,
        "hasModel": m.get("hasModel"),
        "references": _tile_overview(m.get("references") or []),
        "imports": _tile_overview(sliced_imports),
        "imports_total": len(all_imports),
        "imports_offset": imports_offset,
        # Over all imports; must match filterCounts in the UI's DatasetGallery.
        "filter_counts": _compute_filter_counts(
            all_imports, m.get("verdicts") or {},
        ),
    }


def _compute_filter_counts(
    all_imports: list[dict], verdicts: dict,
) -> dict[str, int]:
    """Gallery filter counts over all imports."""
    counts = {"all": 0, "unlabelled": 0, "unrated": 0, "good": 0, "bad": 0, "unsure": 0}
    for e in all_imports:
        if not isinstance(e, dict):
            continue
        counts["all"] += 1
        background = bool(e.get("background"))  # done, and nothing to rate
        has_dets = _import_has_labels(e)
        if not has_dets and not background:
            counts["unlabelled"] += 1
        eid = e.get("id")
        v = verdicts.get(eid) if eid else None
        if eid and not v and has_dets and not background:
            counts["unrated"] += 1
        if v == "good":
            counts["good"] += 1
        elif v == "bad":
            counts["bad"] += 1
        elif v == "unsure":
            counts["unsure"] += 1
    return counts


@app.get("/api/v2/projects/{project_id}/initial", response_class=FastJSONResponse)
async def get_project_initial(
    project_id: str,
    response: Response,
    n: int = _INITIAL_PAYLOAD_LIMIT,
):
    """First-paint payload: project meta, the first `n` import tiles and lite
    stats, from the initial sidecar when it is current."""
    if not project_dir(project_id).exists():
        raise HTTPException(404)
    # No browser caching: a reload after a delete must not show the old payload.
    response.headers["Cache-Control"] = "no-store, must-revalidate"
    n = max(1, min(int(n), _INITIAL_PAYLOAD_LIMIT))
    side = await asyncio.to_thread(_read_initial_sidecar, project_id)
    if side is not None:
        # The sidecar can vanish between the read and this stat.
        try:
            sidecar_mtime = _initial_sidecar_path(project_id).stat().st_mtime
        except OSError:
            sidecar_mtime = None
        manifest_mtime = _manifest_disk_mtime(project_id)
        if sidecar_mtime is not None and sidecar_mtime >= manifest_mtime:
            _kick_sidecar_refresh(project_id)
            imps = (side.get("imports") or [])[:n]
            return {**side, "imports": imps}
    # Missing or stale sidecar: built inline.
    _kick_sidecar_refresh(project_id)
    m = await asyncio.to_thread(load_manifest, project_id, False)
    if not m:
        raise HTTPException(404)
    overview_payload = await asyncio.to_thread(
        _build_overview_payload, project_id, m,
        _FAST_OVERVIEW_LIMIT, 0,
    )
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
        "dataset_type": overview_payload.get("dataset_type"),
        "hasModel": overview_payload.get("hasModel"),
        "references": overview_payload.get("references") or [],
        "imports": (overview_payload.get("imports") or [])[:n],
        "imports_total": overview_payload.get("imports_total") or 0,
        "stats": stats_payload,
    }


@app.get("/api/v2/projects/{project_id}/overview", response_class=FastJSONResponse)
async def get_project_overview(
    project_id: str,
    imports_limit: int | None = None,
    imports_offset: int = 0,
):
    """Project metadata and import tiles, without detections, embeddings or
    masks. imports_limit and imports_offset page the tiles."""
    if not project_dir(project_id).exists():
        raise HTTPException(404)
    # A first page within _FAST_OVERVIEW_LIMIT comes from the overview sidecar.
    if (
        imports_offset == 0
        and imports_limit is not None
        and imports_limit <= _FAST_OVERVIEW_LIMIT
    ):
        side = await asyncio.to_thread(_read_overview_sidecar, project_id)
        if side is not None:
            # The sidecar can vanish between the read and this stat.
            try:
                sidecar_mtime = _overview_sidecar_path(project_id).stat().st_mtime
            except OSError:
                sidecar_mtime = None
            manifest_mtime = _manifest_disk_mtime(project_id)
            # A sidecar without cover_uploaded is stale.
            if (sidecar_mtime is not None and sidecar_mtime >= manifest_mtime
                    and "cover_uploaded" in side):
                _kick_sidecar_refresh(project_id)
                imps = (side.get("imports") or [])[:imports_limit]
                return {**side, "imports": imps, "imports_offset": 0}
            _kick_sidecar_refresh(project_id)
        else:
            _kick_sidecar_refresh(project_id)
    disk_mtime = _manifest_disk_mtime(project_id)
    # Warms the lite dataset-stats sidecar, one rebuild per project at a time.
    asyncio.create_task(_payload_revalidate(
        project_id, "dataset-stats-lite",
        lambda: _persist_dataset_stats(project_id, True),
    ))
    cache_name = f"overview:{imports_offset}:{imports_limit}"
    cached, fresh = _payload_cache_get_swr(project_id, cache_name, disk_mtime)
    if cached is not None and fresh:
        return cached
    if cached is not None:
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


@app.get("/api/v2/projects/{project_id}/annotations/{import_id}", response_class=FastJSONResponse)
async def get_project_annotation_for_import(project_id: str, import_id: str):
    """One import's detections with masks and score dicts, but no embeddings."""
    if not project_dir(project_id).exists():
        raise HTTPException(404)
    # Loaded to refresh the import index.
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
    """Bulk annotations for scope "all", "refs" or "imports"."""
    out: dict = {}
    if scope in ("all", "refs"):
        # With their masks: at most REF_MAX_POOL references, and an edit
        # writes back what the editor holds.
        out["references"] = _detection_annotations(
            m.get("references") or [], include_edits=False, include_heavy=True,
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


@app.get("/api/v2/projects/{project_id}/annotations", response_class=FastJSONResponse)
async def get_project_annotations(project_id: str, scope: str = "all"):
    """Detections and edited boxes without embeddings; `scope` "refs" or "imports" returns one half."""
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
    m = await asyncio.to_thread(load_manifest, project_id, False)
    if not m:
        return m
    payload = await asyncio.to_thread(_build_annotations_payload_scoped, m, scope_norm)
    _payload_cache_put(project_id, cache_key, disk_mtime, payload)
    return payload


class UpdateProjectIn(BaseModel):
    tags: list[str] | None = None
    verdicts: dict[str, str] | None = None
    editedBoxes: dict[str, list[dict]] | None = None
    cover: str | None = None
    hasModel: bool | None = None
    # {canonical_lower: display_name}, for display only; None leaves them unchanged.
    label_aliases: dict[str, str] | None = None
    # {canonical_lower: "#rrggbb"}; None leaves them unchanged.
    labelColours: dict[str, str] | None = None


@app.put("/api/projects/{project_id}")
async def update_project(project_id: str, payload: UpdateProjectIn):
    if not project_dir(project_id).exists():
        raise HTTPException(404)

    data = payload.model_dump(exclude_none=True)
    if "cover" in payload.model_fields_set and payload.cover is None:
        data["cover"] = None  # null resets to the default cover

    # Only #rrggbb values are kept; "" or "#" clears an override.
    new_label_colours = data.pop("labelColours", None)
    colours: dict[str, str] | None = None
    if isinstance(new_label_colours, dict):
        colours = {}
        for k, v in new_label_colours.items():
            if not isinstance(k, str) or not isinstance(v, str):
                continue
            vs = v.strip()
            if vs in ("", "#"):
                colours[k.strip().lower()] = ""
            elif len(vs) == 7 and vs.startswith("#") and all(c in "0123456789abcdefABCDEF" for c in vs[1:]):
                colours[k.strip().lower()] = "#" + vs[1:].lower()

    # cover_thumb joins the cover into a path, so it must name one of this dataset's pictures.
    cover = data.get("cover")
    if cover:
        proj = project_dir(project_id)
        if (cover != Path(cover).name or cover in (".", "..")
                or not any((proj / sub / cover).is_file() for sub in ("images", "references"))):
            raise HTTPException(400, "cover must be one of this dataset's pictures")

    # Read, merge and save under the write lock, so a concurrent upload or edit is kept.
    lock = await _manifest_write_lock(project_id)
    async with lock:
        manifest = load_manifest(project_id)
        if colours is not None:
            # Merged, so a PUT can carry only the changed labels.
            merged = dict(manifest.get("labelColours") or {})
            for k, v in colours.items():
                if v:
                    merged[k] = v
                else:
                    merged.pop(k, None)
            data["labelColours"] = merged
        manifest.update(data)
        # A picked cover, or a reset, replaces an uploaded one.
        if "cover" in data:
            manifest["cover_uploaded"] = False
        manifest["updatedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        save_manifest(project_id, manifest)
    # The next read must not serve a payload from before this save.
    _invalidate_project_payloads(project_id)
    return {"ok": True}


def _rmtree_retrying(path: Path, attempts: int = 6) -> None:
    """rmtree with retries: a background writer can make it fail on Windows
    with "directory not empty" or "in use"."""
    for i in range(attempts):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except OSError:
            if i == attempts - 1:
                raise
            time.sleep(0.2 * (i + 1))


@app.delete("/api/projects/{project_id}")
async def delete_project(project_id: str):
    p = project_dir(project_id)
    if not p.exists():
        raise HTTPException(404)
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
    # Tombstoned before teardown, so a racing save_manifest cannot recreate the folder.
    _mark_project_deleted(project_id)
    removed = False
    try:
        for j in list(state["jobs"].jobs.values()):
            if j.project != project_id:
                continue
            if j.status in ("queued", "running"):
                state["jobs"].cancel(j.id)
                # Terminal even if the runner never checks cancel_event.
                j.status = "cancelled"
                if j.finished_at is None:
                    j.finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        # Off the event loop: a blocked loop makes the desktop shell restart the engine.
        await asyncio.to_thread(_rmtree_retrying, p)
        removed = True
        invalidate_manifest_cache(project_id)
        store.delete_dataset(project_id)
    except Exception:
        # The tombstone stays only once the folder is gone.
        if not removed:
            _unmark_project_deleted(project_id)
        raise
    return {"ok": True}


class DuplicateProjectIn(BaseModel):
    # Defaults to "Copy of <source name>".
    name: str | None = None


@app.post("/api/projects/{project_id}/duplicate")
async def duplicate_project(project_id: str, payload: DuplicateProjectIn | None = None):
    """Copy a dataset into a new dataset outside any Project."""
    src = project_dir(project_id)
    if not src.exists() or not manifest_path(project_id).exists():
        raise HTTPException(404, "dataset not found")
    src_manifest = load_manifest(project_id)
    if not src_manifest:
        raise HTTPException(404, "dataset not found")

    new_id = _uuid.uuid4().hex
    for _ in range(5):
        if not store.dataset_exists(new_id) and not _is_project_deleted(new_id):
            break
        new_id = _uuid.uuid4().hex

    name = (payload.name.strip() if (payload and payload.name and payload.name.strip()) else "")
    if not name:
        name = f"Copy of {src_manifest.get('name') or 'dataset'}"

    # Registered but not created: copytree requires dst not to exist.
    dst = store.reserve_dataset_dir(new_id, name)

    loop = asyncio.get_running_loop()

    def _copy() -> None:
        shutil.copytree(src, dst)
        # Every top-level JSON except dataset.json is a cache of the source.
        try:
            for f in dst.glob("*.json"):
                if f.name != "dataset.json":
                    f.unlink()
        except Exception as e:
            print(f"[duplicate] sidecar cleanup failed for {new_id}: {e}")

    try:
        await loop.run_in_executor(None, _copy)
    except Exception as e:
        try:
            if dst.exists():
                shutil.rmtree(dst)
        except Exception:
            pass
        print(f"[duplicate] copy failed {project_id} -> {new_id}: {e}")
        raise HTTPException(500, "duplicate failed")

    m = load_manifest(new_id) or dict(src_manifest)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    m["id"] = new_id
    m["name"] = name
    m["createdAt"] = now
    m["updatedAt"] = now
    m["favouritedBy"] = []
    m.pop("container_id", None)
    m.pop("derived", None)
    save_manifest(new_id, m)
    add_event("project_duplicate", project=new_id, source=project_id)
    return {"id": new_id, "name": name}


class RenameIn(BaseModel):
    name: str


@app.post("/api/projects/{project_id}/rename")
async def rename_project(project_id: str, payload: RenameIn):
    """Set the display name in the manifest; the folder is not renamed."""
    new_name = (payload.name or "").strip()
    if not new_name:
        raise HTTPException(400, "name required")
    if not project_dir(project_id).exists():
        raise HTTPException(404)
    lock = await _manifest_write_lock(project_id)
    async with lock:
        manifest = load_manifest(project_id)
        manifest["name"] = new_name
        manifest["updatedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        save_manifest(project_id, manifest)
    return {"id": project_id, "name": new_name}


@app.post("/api/projects/{project_id}/favourite")
async def favourite_project(project_id: str):
    """Star or unstar the dataset; starred datasets list first."""
    if not project_dir(project_id).exists():
        raise HTTPException(404, "dataset not found")
    lock = await _manifest_write_lock(project_id)
    async with lock:
        manifest = load_manifest(project_id)
        if not manifest:
            raise HTTPException(404, "dataset not found")
        fav = [u for u in (manifest.get("favouritedBy") or []) if isinstance(u, str)]
        fav = [u for u in fav if u != LOCAL_USER] if LOCAL_USER in fav else fav + [LOCAL_USER]
        manifest["favouritedBy"] = fav
        save_manifest(project_id, manifest)
    return {"favouritedByMe": LOCAL_USER in fav}


# Every box field that can hold a label name.
_LABEL_KEYS = ("label", "predLabel", "pred_label", "gdLabel", "gd_label", "gd_variant")


def _rename_label_fields(manifest: dict, old: str, new: str) -> int:
    """Rename a label throughout one manifest, merging into an existing label.
    Returns the occurrences renamed; the tag list counts once."""
    renamed = 0
    ol = old.strip().lower()

    # Deduped case-insensitively, so a rename onto an existing label merges.
    tags = manifest.get("tags", []) or []
    out_tags: list[str] = []
    seen_lower: set[str] = set()
    hit_tag = False
    for t in tags:
        s = (t or "").strip()
        target = new if s.lower() == ol else s
        if s.lower() == ol:
            hit_tag = True
        if target.lower() not in seen_lower:
            seen_lower.add(target.lower())
            out_tags.append(target)
    if hit_tag:
        renamed += 1
    manifest["tags"] = out_tags

    def _boxes(lst) -> None:
        nonlocal renamed
        for b in lst or []:
            if not isinstance(b, dict):
                continue
            for k in _LABEL_KEYS:
                v = b.get(k)
                if isinstance(v, str) and v.strip().lower() == ol:
                    b[k] = new
                    renamed += 1

    for r in manifest.get("results", []) or []:  # V1 layout
        _boxes(r.get("detections"))
    for boxes in (manifest.get("editedBoxes", {}) or {}).values():  # V1 layout
        if isinstance(boxes, list):
            _boxes(boxes)
    for imp in manifest.get("imports", []) or []:  # V2 layout
        if not isinstance(imp, dict):
            continue
        _boxes(imp.get("detections"))
        _boxes(imp.get("editedBoxes"))
    # References too, or the next labelling run brings the old label back.
    for ref in manifest.get("references", []) or []:
        if not isinstance(ref, dict):
            continue
        v = ref.get("label")
        if isinstance(v, str) and v.strip().lower() == ol:
            ref["label"] = new
            renamed += 1
        _boxes(ref.get("detections"))

    # Colour and alias entries move to the new label; an existing entry there wins.
    for map_key in ("labelColours", "label_aliases"):
        m = manifest.get(map_key)
        if not isinstance(m, dict):
            continue
        for k in list(m.keys()):
            if isinstance(k, str) and k.strip().lower() == ol:
                val = m.pop(k)
                m.setdefault(new.strip().lower(), val)

    return renamed


class RenameLabelRequest(BaseModel):
    old_label: str
    new_label: str


@app.post("/api/projects/{project_id}/labels/rename")
async def rename_label(project_id: str, body: RenameLabelRequest):
    """Rename a label throughout the dataset and its augmentations."""
    proj = project_dir(project_id)
    if not proj.exists():
        raise HTTPException(404)

    old = (body.old_label or "").strip()
    new = (body.new_label or "").strip()
    if not old or not new:
        raise HTTPException(400, "old_label and new_label are required")
    if old == new:
        # A change of case is a rename.
        return {"ok": True, "renamed": 0}

    if _label_job_active(project_id):
        raise HTTPException(409, "the dataset is being labelled; rename the label when that finishes")
    lock = await _manifest_write_lock(project_id)
    async with lock:
        manifest = load_manifest(project_id)
        if not manifest:
            raise HTTPException(404)

        renamed = _rename_label_fields(manifest, old, new)
        renamed += _relabel_augmentations(
            project_id, lambda lab: new if lab.strip().lower() == old.strip().lower() else lab)

        manifest["updatedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        save_manifest(project_id, manifest)

    return {"ok": True, "renamed": renamed, "tags": manifest.get("tags") or []}


# Label usage and renames across a Project's datasets.


@app.get("/api/containers/{container_id}/labels")
async def container_labels(container_id: str):
    c = containers.load_container(container_id)
    if not c:
        raise HTTPException(404, "project not found")
    ds_ids = list(c.get("dataset_ids") or [])

    def _collect() -> list[dict]:
        agg: dict[str, dict] = {}
        for pid in ds_ids:
            m = load_manifest(pid, copy=False)
            if not m:
                continue
            seen_here: set[str] = set()

            def _touch(s: str, boxes: int) -> None:
                a = agg.setdefault(s, {"label": s, "datasets": 0, "boxes": 0})
                a["boxes"] += boxes
                if s not in seen_here:
                    a["datasets"] += 1
                    seen_here.add(s)

            for t in m.get("tags") or []:
                s = (t or "").strip().lower()
                if s:
                    _touch(s, 0)
            for _img, _sz, b in _project_box_iter(m):
                s = _box_label(b).strip().lower()
                if s:
                    _touch(s, 1)
        return sorted(agg.values(), key=lambda x: (-x["boxes"], x["label"]))

    labels = await asyncio.to_thread(_collect)
    return {"datasets": len(ds_ids), "labels": labels}


@app.post("/api/containers/{container_id}/labels/rename")
async def container_rename_label(container_id: str, body: RenameLabelRequest):
    c = containers.load_container(container_id)
    if not c:
        raise HTTPException(404, "project not found")
    old = (body.old_label or "").strip()
    new = (body.new_label or "").strip()
    if not old or not new:
        raise HTTPException(400, "old_label and new_label are required")
    if old == new:
        # A change of case is a rename.
        return {"ok": True, "renamed": 0, "datasetsTouched": 0}

    ds_ids = list(c.get("dataset_ids") or [])
    if any(_label_job_active(pid) for pid in ds_ids):
        raise HTTPException(409, "a dataset in this project is being labelled; rename when that finishes")

    def _one(pid: str) -> int:
        m = load_manifest(pid)
        if not m:
            return 0
        n = _rename_label_fields(m, old, new)
        n += _relabel_augmentations(pid, lambda lab: new if lab.strip().lower() == old.strip().lower() else lab)
        if n:
            m["updatedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            save_manifest(pid, m)
        return n

    total = 0
    touched = 0
    for pid in ds_ids:
        lock = await _manifest_write_lock(pid)
        async with lock:
            n = await asyncio.to_thread(_one, pid)
        if n:
            total += n
            touched += 1
    return {"ok": True, "renamed": total, "datasetsTouched": touched}


@app.post("/api/v2/projects/{project_id}/clear_all_annotations")
async def clear_all_annotations(project_id: str):
    """Reset every import to unlabelled and pending; tags, colours and aliases are kept."""
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
            entry.pop("background", None)
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
        # V1 maps keyed by filename.
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


class ScheduleJobIn(BaseModel):
    project: str
    kind: str  # label_charlie, purge_label, train_yolox or label_model
    params: dict = {}


@app.post("/api/jobs")
async def schedule_job(payload: ScheduleJobIn):
    """Queue a job on a dataset; returns its id."""
    if payload.kind not in ("label_charlie", "purge_label", "train_yolox", "label_model"):
        raise HTTPException(400, f"unknown kind: {payload.kind}")
    proj = project_dir(payload.project)
    if not proj.exists():
        raise HTTPException(404, "dataset not found")
    if payload.kind in _LABEL_WRITERS and _label_job_active(payload.project):
        raise HTTPException(409, "this dataset is already being labelled or having a label deleted; "
                                 "wait for that to finish")
    # copy=False: read only.
    project_manifest = load_manifest(payload.project, copy=False) or {}
    if not project_manifest:
        raise HTTPException(404, "dataset not found")

    n_images = 0
    if payload.kind == "label_charlie":
        manifest = project_manifest
        force_relabel = bool((payload.params or {}).get("force_relabel"))
        # As _run_label_charlie_job selects: all but background images on a
        # relabel, else the unlabelled ones.
        n_images = sum(
            1
            for e in (manifest.get("imports") or [])
            if isinstance(e, dict) and not e.get("background")
            and (force_relabel or not _import_has_labels(e))
        )
        if n_images == 0:
            raise HTTPException(
                400,
                "no images to process: add some to the dataset first"
                if force_relabel
                else "no unlabelled images to process",
            )
    elif payload.kind == "train_yolox":
        manifest = project_manifest

        def _is_labelled(e: dict) -> bool:
            if e.get("editedBoxesSet"):
                edited = e.get("editedBoxes")
                return isinstance(edited, list) and len(edited) > 0
            return bool(e.get("detections"))

        n_images = sum(
            1
            for e in (manifest.get("imports") or [])
            if isinstance(e, dict) and _is_labelled(e)
        )
        if n_images < 2:
            raise HTTPException(
                400, "training needs at least 2 labelled images: label some first"
            )
        if _train_job_active(payload.project):
            raise HTTPException(409, "a training run is already queued or running for this dataset")
        if _augment_job_active(payload.project):
            raise HTTPException(409, "augmentations are being generated for this dataset; train when that finishes")
        import train_yolox as _ty

        try:
            if any(_run_is_live(rd) for rd in _iter_run_dirs(store.dataset_dir(payload.project))):
                raise HTTPException(409, "a training run for this dataset is live in another process")
        except KeyError:
            raise HTTPException(404, "dataset not found")

        prm = payload.params or {}
        resume = str(prm.get("resumeRun") or "").strip()
        if resume:
            rd = _ty.find_run_dir(payload.project, resume)
            if rd is None or not (rd / "ckpt_last.pt").is_file():
                raise HTTPException(400, "nothing to resume: that run has no checkpoint")
            if _run_is_live(rd):
                raise HTTPException(409, "that run is still training in another process")
        init_run = str(prm.get("initRun") or "").strip()
        if init_run:
            problem = _ty.init_run_problem(_ty.find_run_dir(payload.project, init_run),
                                           _train_cfg_from_params(prm, "cpu").size)
            if problem:
                raise HTTPException(400, problem)
        eval_project = str(prm.get("evalProject") or "").strip()
        if eval_project:
            if eval_project == payload.project:
                raise HTTPException(400, "the held-out dataset must be a different dataset")
            if not store.dataset_exists(eval_project):
                raise HTTPException(404, "held-out dataset not found")
    elif payload.kind == "label_model":
        import train_yolox as _ty

        prm = payload.params or {}
        run_dir = _ty.find_run_dir(payload.project, str(prm.get("run") or "").strip())
        if (run_dir is None or not (run_dir / "report.json").is_file()
                or not any(run_dir.glob("*_int8.onnx"))):
            raise HTTPException(400, "that run has no exported model")
        rep = _read_run_json(run_dir / "report.json")
        if not isinstance(rep, dict) or rep.get("exportRejected"):
            raise HTTPException(400, "that run's export was rejected by ST Edge AI; train again")
        n_images = len(_label_model_targets(project_manifest, bool(prm.get("overwrite"))))
        if n_images == 0:
            raise HTTPException(
                400, "no unlabelled images to label (turn on overwrite to re-label "
                     "auto-labelled images; hand-edited images are never touched)")

    job = state["jobs"].schedule(
        kind=payload.kind,
        project=payload.project,
        params=payload.params,
        n_images=n_images,
    )
    return {"jobId": job.id, **job.to_public()}


@app.delete("/api/v2/projects/{project_id}/jobs/{job_id}")
async def v2_cancel_project_job(project_id: str, job_id: str):
    """Cancel a job of this project."""
    job = state["jobs"].jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    if job.project != project_id:
        raise HTTPException(404, "job not found")
    if not state["jobs"].cancel(job_id):
        raise HTTPException(409, "job already finished")
    return {"ok": True}


@app.get("/api/projects/{project_id}/jobs/active")
async def project_active_job(project_id: str):
    """The project's running or queued labelling job, or null."""
    proj = project_dir(project_id)
    if not proj.exists():
        raise HTTPException(404, "dataset not found")
    candidates = [
        j for j in state["jobs"].jobs.values()
        if j.project == project_id
        and j.kind in ("label_charlie", "purge_label")
        and j.status in ("running", "queued")
    ]
    if not candidates:
        return None
    # Prefer running over queued, then most recent.
    candidates.sort(key=lambda j: j.queued_at, reverse=True)
    candidates.sort(key=lambda j: 0 if j.status == "running" else 1)
    return candidates[0].to_public()


@app.get("/api/projects/{project_id}/jobs/{job_id}")
async def get_project_job(project_id: str, job_id: str):
    """Status of a job of this project."""
    job = state["jobs"].jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    if job.project != project_id:
        raise HTTPException(403, "job does not belong to this project")
    return job.to_public()


# Dataset export.

def _safe_slug(s: str) -> str:
    """Reduce a string to [a-zA-Z0-9_-], for headers, zip entries and filenames."""
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", (s or "export").strip()) or "export"


def _image_index(manifest: dict) -> dict[str, dict]:
    """{filename: {width, height, boxes}} over V1 results and V2 imports. A V2
    import's boxes are its edited boxes once edited, else its unrejected detections."""
    out: dict[str, dict] = {}

    # V1: sizes from results, boxes from the top-level editedBoxes.
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

    for entry in (manifest.get("imports") or []):
        if not isinstance(entry, dict):
            continue
        fname = entry.get("filename")
        if not fname:
            continue
        slot = out.setdefault(fname, {"width": 0, "height": 0, "boxes": []})
        w = int(entry.get("width") or 0)
        h = int(entry.get("height") or 0)
        if w and not slot["width"]:
            slot["width"] = w
        if h and not slot["height"]:
            slot["height"] = h
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
    """Yield (image_filename, (W, H), box) for every box, V1 and V2."""
    idx = _image_index(manifest)
    for img_name, info in idx.items():
        sz = (info["width"], info["height"])
        for b in info["boxes"]:
            yield img_name, sz, b


def _image_names_list(manifest: dict) -> list[str]:
    """Every original image filename in the project, V1 + V2 combined."""
    return list(_image_index(manifest).keys())


# Shorter box side after letterboxing to the input shape, in px. Must match
# the UI's BOX_FAIL_PX and BOX_WARN_PX.
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
    """False when the box's size status is excluded: red is "fail", orange "warn"."""
    status = _box_status(_scaled_min_side(x0, y0, x1, y1, img_w, img_h, in_w, in_h))
    if status == "fail" and exclude_red:
        return False
    if status == "warn" and exclude_orange:
        return False
    return True


def _iter_aug_copies(project_id: str, manifest: dict):
    """Yield (export_name, parent_filename, canvas_w, canvas_h, import_id,
    copy_name, path, detections) for each augmentation copy on disk, boxes or
    none. export_name is aug_<parent_stem>_<k>.jpg."""
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
        if not aug_source.is_current(data, entry):
            continue
        canvas_w = int(data.get("width") or 0)
        canvas_h = int(data.get("height") or 0)
        parent_stem = Path(parent).stem
        for copy_name, dets in (data.get("copies") or {}).items():
            if not isinstance(dets, list) or Path(copy_name).name != copy_name:
                continue
            jpeg_path = aug_root / import_id / copy_name
            if not jpeg_path.is_file():
                continue
            export_name = f"aug_{parent_stem}_{Path(copy_name).stem}.jpg"
            yield export_name, parent, canvas_w, canvas_h, import_id, copy_name, jpeg_path, dets


def _iter_augmentations(project_id: str, manifest: dict):
    """Yield (export_name, parent_filename, canvas_w, canvas_h, box, import_id,
    copy_name) for each box of each augmentation copy on disk; box has x0, y0,
    x1, y1, label and mask.polygons."""
    for export_name, parent, canvas_w, canvas_h, import_id, copy_name, _path, dets in _iter_aug_copies(
        project_id, manifest
    ):
        for det in dets:
            if not isinstance(det, dict):
                continue
            box = det.get("box") or []
            if not (isinstance(box, (list, tuple)) and len(box) == 4):
                continue
            if canvas_w > 0 and canvas_h > 0:
                box = _clip_box_to(box, canvas_w, canvas_h)
                if box is None:
                    continue
            adapted = {
                "x0": float(box[0]),
                "y0": float(box[1]),
                "x1": float(box[2]),
                "y1": float(box[3]),
                "label": det.get("label") or "",
            }
            polys = [p for p in (det.get("polys") or []) if isinstance(p, list)]
            if polys and canvas_w > 0 and canvas_h > 0:
                polys = [q for q in (_clip_poly_to(p, canvas_w, canvas_h) for p in polys) if q]
            if polys:
                adapted["mask"] = {"polygons": polys}
            yield export_name, parent, canvas_w, canvas_h, adapted, import_id, copy_name


def _aug_files_to_bundle(project_id: str, manifest: dict) -> list[tuple[str, Path]]:
    """[(export_name, path)] for each augmentation copy on disk."""
    out: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for export_name, _parent, _w, _h, _iid, _cn, path, _dets in _iter_aug_copies(project_id, manifest):
        if export_name not in seen:
            seen.add(export_name)
            out.append((export_name, path))
    return out


def _build_readme(manifest: dict) -> str:
    """Plain-text README for an export zip."""
    name = manifest.get("name") or "Untitled dataset"
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return (
        "PixelKit dataset export\n"
        "=======================\n"
        "\n"
        f"Dataset: {name}\n"
        f"Exported at: {now_iso} (UTC)\n"
        "\n"
        "Built with PixelKit: https://github.com/ohmlab-ltd/pixelkit\n"
    )


def _licence_name(code: str | None, version: str | None) -> str:
    """A licence name such as "CC BY-SA 4.0" from Openverse's code and version,
    or "" when the code is unknown."""
    code = (code or "").strip().lower()
    if not code:
        return ""
    name = {"cc0": "CC0", "pdm": "Public Domain Mark"}.get(code, "CC " + code.upper())
    return f"{name} {(version or '').strip()}".strip()


def _build_attributions(manifest: dict, aug_parent_by_export: dict[str, str]) -> str:
    """ATTRIBUTIONS.txt for the exported images that came from Openverse, or ""."""
    sources = {e["filename"]: e["source"] for e in manifest.get("imports") or []
               if isinstance(e, dict) and e.get("filename") and isinstance(e.get("source"), dict)
               and e["source"].get("kind") == "openverse"}
    if not sources:
        return ""
    copies: dict[str, list[str]] = {}
    for export_name, parent in aug_parent_by_export.items():
        if parent in sources:
            copies.setdefault(parent, []).append(export_name)
    out = ["Images from Openverse (https://openverse.org) in this export, with their",
           "creators and licences. Files named aug_<name>_<n>.jpg are copies PixelKit",
           "changed: rotated, cropped, recoloured, blurred or placed on other backgrounds.",
           ""]
    for fn in sorted(sources):
        src = sources[fn]
        out.append(fn)
        if src.get("cropped"):
            out.append("  A crop of the original picture.")
        if copies.get(fn):
            out.append("  Changed copies: " + ", ".join(sorted(copies[fn])))
        licence = _licence_name(src.get("license"), src.get("license_version"))
        if src.get("attribution"):
            out.append("  " + " ".join(str(src["attribution"]).split()))
        elif licence:
            out.append(f'  "{src.get("title") or "Untitled"}" by {src.get("creator") or "an unknown creator"}, {licence}')
        else:
            out.append("  Licence not recorded: check it on the source page.")
        if src.get("license_url"):
            out.append("  Licence: " + str(src["license_url"]))
        out.append("  Source: " + str(src.get("foreign_landing_url") or src.get("url") or "unknown"))
        out.append("")
    return "\n".join(out)


def _box_xyxy(b: dict) -> tuple[float, float, float, float] | None:
    """(x0, y0, x1, y1) from x0/y0/x1/y1 keys, a box_xyxy list or a box list;
    None when missing or empty."""
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
    """The box's first non-empty label field, or ""."""
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
    # Tags first, so class indices are stable across exports.
    for t in (manifest.get("tags") or []):
        tl = (t or "").strip().lower()
        if tl and tl not in seen:
            seen.add(tl)
            out.append(tl)
    for _img, _sz, b in _project_box_iter(manifest):
        lbl = _box_label(b).lower()
        if lbl and lbl not in seen:
            seen.add(lbl)
            out.append(lbl)
    return out


def _load_image_bytes(project_id: str, img_name: str) -> bytes | None:
    """Original image bytes from images/, or None."""
    try:
        return (project_dir(project_id) / "images" / img_name).read_bytes()
    except Exception as e:
        print(f"[export] reading {img_name} failed: {e}")
        return None


def _split_for(parent_name: str, train_split: float) -> str:
    """Stable "train" or "val" from a sha1 bucket of the file name; train_split is
    the train fraction. Augmentations pass their parent's name, so share its split."""
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
    """COCO dicts by split, {"train": ..., "val": ...}; val is left out when empty."""
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

    # Images are registered before their annotations.
    img_idx = _image_index(manifest)
    for img_name, info in img_idx.items():
        w, h = info["width"], info["height"]
        if not w or not h:
            continue
        _register_image(_split_for(img_name, train_split), img_name, w, h)

    if with_augmentations:
        for export_name, parent, cw, ch, *_rest in _iter_aug_copies(project_id, manifest):
            if export_name in id_by_file or not cw or not ch:
                continue
            _register_image(_split_for(parent, train_split), export_name, int(cw), int(ch))

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
        "contributor": "PixelKit",
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
    """YOLO .txt contents keyed by (split, txt_name), one line per box, normalised 0..1:

        <class_id> x1 y1 x2 y2 x3 y3 ...   polygon, with include_segmentations
        <class_id> cx cy w h               box
    """
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
                if len(flat) >= 6:  # at least three points
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

    for img_name, sz, b in _project_box_iter(manifest):
        W, H = sz
        split = _split_for(img_name, train_split)
        _emit(split, Path(img_name).stem + ".txt", W, H, b)

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
    """Pascal VOC annotation XML for one image, boxes only."""
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
            f"    <name>{_xml_escape(label)}</name>\n"
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
        f"  <filename>{_xml_escape(img_name)}</filename>\n"
        "  <size>\n"
        f"    <width>{W}</width>\n"
        f"    <height>{H}</height>\n"
        "    <depth>3</depth>\n"
        "  </size>\n"
        "  <segmented>0</segmented>\n"
        + ("\n".join(obj_blocks) + "\n" if obj_blocks else "")
        + "</annotation>\n"
    )


@app.get("/api/projects/{project_id}/export")
# A plain def, so FastAPI runs it in its threadpool and a long zip does not stall other requests.
def export_project(
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
    """Export the dataset as a zip in `format`: yolo, coco, voc, cvat, labelstudio or masks."""
    if not project_dir(project_id).exists():
        raise HTTPException(404, "dataset not found")
    manifest = load_manifest(project_id)
    if not manifest:
        raise HTTPException(404, "dataset not found")

    if format not in ("yolo", "coco", "voc", "cvat", "labelstudio", "masks"):
        raise HTTPException(400, f"unsupported format: {format}")
    if not include_boxes and not include_segmentations:
        raise HTTPException(400, "at least one of include_boxes / include_segmentations must be true")
    if format == "voc" and not include_boxes:
        raise HTTPException(400, "Pascal VOC only supports bounding boxes; include_boxes must be true")
    if format == "masks" and not include_segmentations:
        raise HTTPException(400, "PNG masks are built from segmentations; include_segmentations must be true")
    train_split = max(0.0, min(1.0, float(train_split)))

    base = _safe_slug(manifest.get("name") or project_id)
    image_names = _image_names_list(manifest)
    aug_files = _aug_files_to_bundle(project_id, manifest) if include_images else []
    aug_parent_by_export = _aug_parent_lookup(project_id, manifest) if include_images else {}
    with_augs = bool(aug_files)
    readme = _build_readme(manifest)
    attributions = _build_attributions(manifest, aug_parent_by_export) if include_images else ""

    def _write_images(zf, *, train_subdir: str, val_subdir: str) -> None:
        for img_name in image_names:
            data = _load_image_bytes(project_id, img_name)
            if data is None:
                continue
            split = _split_for(img_name, train_split)
            sub = train_subdir if split == "train" else val_subdir
            zf.writestr(f"{sub}/{img_name}" if sub else img_name, data)
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
            if attributions:
                zf.writestr("ATTRIBUTIONS.txt", attributions)
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
            if attributions:
                zf.writestr("ATTRIBUTIONS.txt", attributions)
            zf.writestr("classes.txt", "\n".join(cats) + "\n")
            # Ultralytics data.yaml; the loader finds labels/ by replacing images/ in each image path.
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

    if format in ("cvat", "labelstudio", "masks"):
        rows = _export_rows(
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
        cats = _categories(manifest)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            zf.writestr("README.txt", readme)
            if attributions:
                zf.writestr("ATTRIBUTIONS.txt", attributions)
            if format == "cvat":
                xml = _build_cvat_xml(
                    rows,
                    cats,
                    manifest.get("name") or project_id,
                    include_boxes=include_boxes,
                    include_segmentations=include_segmentations,
                )
                zf.writestr("annotations.xml", xml)
                if include_images:
                    # Flat: CVAT matches images by bare file name.
                    _write_images(zf, train_subdir="images", val_subdir="images")
            elif format == "labelstudio":
                tasks = _build_labelstudio(
                    rows,
                    include_boxes=include_boxes,
                    include_segmentations=include_segmentations,
                )
                zf.writestr("tasks.json", json.dumps(tasks, indent=2))
                from xml.sax.saxutils import quoteattr

                label_tags = "".join(f"    <Label value={quoteattr(c)}/>\n" for c in cats)
                controls = ""
                if include_boxes:
                    controls += (
                        '  <RectangleLabels name="label" toName="image">\n'
                        + label_tags + "  </RectangleLabels>\n"
                    )
                if include_segmentations:
                    controls += (
                        '  <PolygonLabels name="polygon" toName="image">\n'
                        + label_tags + "  </PolygonLabels>\n"
                    )
                zf.writestr(
                    "labeling-config.xml",
                    '<View>\n  <Image name="image" value="$image"/>\n' + controls + "</View>\n",
                )
                if include_images:
                    _write_images(zf, train_subdir="images", val_subdir="images")
            else:  # masks
                import cv2

                class_idx = {c: i + 1 for i, c in enumerate(cats)}
                zf.writestr(
                    "labels.txt",
                    "0 background\n"
                    + "".join(f"{i} {name}\n" for name, i in class_idx.items()),
                )
                for row in rows:
                    mask = np.zeros((row["height"], row["width"]), dtype=np.uint8)
                    for a in row["anns"]:
                        cls = class_idx.get(a["label"])
                        if not cls:
                            continue
                        for pts in a["polys"]:
                            cv2.fillPoly(
                                mask,
                                [np.array(pts, dtype=np.int32).reshape(-1, 1, 2)],
                                int(cls),
                            )
                    ok, png = cv2.imencode(".png", mask)
                    if ok:
                        stem = Path(row["file"]).stem
                        zf.writestr(f"masks/{row['split']}/{stem}.png", png.tobytes())
                if include_images:
                    _write_images(zf, train_subdir="images/train", val_subdir="images/val")
        return Response(
            buf.getvalue(),
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{base}-{format}.zip"'},
        )

    # format == "voc"
    img_idx = _image_index(manifest)
    train_stems: list[str] = []
    val_stems: list[str] = []
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.writestr("README.txt", readme)
        if attributions:
            zf.writestr("ATTRIBUTIONS.txt", attributions)
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
            for export_name, parent, cw, ch, *_rest in _iter_aug_copies(project_id, manifest):
                if cw and ch:
                    grouped.setdefault(export_name, (int(cw), int(ch), parent, []))
            for export_name, _parent, _cw, _ch, det, _iid, _cn in _iter_augmentations(project_id, manifest):
                if export_name in grouped:
                    grouped[export_name][3].append(det)
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
            _write_images(zf, train_subdir="JPEGImages", val_subdir="JPEGImages")
    return Response(
        buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{base}-voc.zip"'},
    )


def _export_rows(
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
) -> list[dict]:
    """Rows for the CVAT, Label Studio and mask exporters:
    [{file, split, width, height, anns: [{label, box, polys}]}]."""
    in_w, in_h = _parse_input_shape(input_shape)
    rows: dict[str, dict] = {}

    def _reg(file_name: str, w, h, split: str) -> None:
        rows[file_name] = {
            "file": file_name,
            "split": split,
            "width": int(w),
            "height": int(h),
            "anns": [],
        }

    img_idx = _image_index(manifest)
    for img_name, info in img_idx.items():
        w, h = info["width"], info["height"]
        if not w or not h:
            continue
        _reg(img_name, w, h, _split_for(img_name, train_split))
    if with_augmentations:
        for export_name, parent, cw, ch, *_rest in _iter_aug_copies(project_id, manifest):
            if export_name in rows or not cw or not ch:
                continue
            _reg(export_name, cw, ch, _split_for(parent, train_split))

    def _append(file_name: str, W: int, H: int, b: dict) -> None:
        row = rows.get(file_name)
        if row is None:
            return
        rect = _box_xyxy(b)
        if rect is None:
            return
        x0, y0, x1, y1 = rect
        if not _keep_by_size(x0, y0, x1, y1, W, H, in_w, in_h, exclude_red, exclude_orange):
            return
        label = _box_label(b).lower()
        polys: list[list[tuple[float, float]]] = []
        for poly in (b.get("mask") or {}).get("polygons") or []:
            pts = [
                (float(pt[0]), float(pt[1]))
                for pt in poly
                if isinstance(pt, (list, tuple)) and len(pt) >= 2
            ]
            if len(pts) >= 3:
                polys.append(pts)
        if include_segmentations and not include_boxes and not polys:
            return
        row["anns"].append({"label": label, "box": (x0, y0, x1, y1), "polys": polys})

    for img_name, sz, b in _project_box_iter(manifest):
        W, H = sz
        if not W or not H:
            continue
        _append(img_name, W, H, b)
    if with_augmentations:
        for export_name, _parent, cw, ch, det, _iid, _cn in _iter_augmentations(project_id, manifest):
            _append(export_name, int(cw), int(ch), det)
    return list(rows.values())


def _build_cvat_xml(
    rows: list[dict],
    cats: list[str],
    task_name: str,
    *,
    include_boxes: bool,
    include_segmentations: bool,
) -> str:
    """CVAT "images 1.1" XML: a polygon where a mask exists and segmentations are on, else a box."""
    import xml.etree.ElementTree as ET

    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")
    ET.SubElement(task, "name").text = task_name
    labels_el = ET.SubElement(task, "labels")
    for c in cats:
        le = ET.SubElement(labels_el, "label")
        ET.SubElement(le, "name").text = c
    for i, row in enumerate(rows):
        img = ET.SubElement(
            root,
            "image",
            id=str(i),
            name=row["file"],
            width=str(row["width"]),
            height=str(row["height"]),
        )
        for a in row["anns"]:
            x0, y0, x1, y1 = a["box"]
            if a["polys"] and include_segmentations:
                for pts in a["polys"]:
                    ET.SubElement(
                        img,
                        "polygon",
                        label=a["label"],
                        occluded="0",
                        source="manual",
                        points=";".join(f"{x:.2f},{y:.2f}" for x, y in pts),
                    )
            elif include_boxes:
                ET.SubElement(
                    img,
                    "box",
                    label=a["label"],
                    occluded="0",
                    source="manual",
                    xtl=f"{x0:.2f}",
                    ytl=f"{y0:.2f}",
                    xbr=f"{x1:.2f}",
                    ybr=f"{y1:.2f}",
                )
    return ET.tostring(root, encoding="unicode", xml_declaration=True)


def _build_labelstudio(
    rows: list[dict],
    *,
    include_boxes: bool,
    include_segmentations: bool,
) -> list[dict]:
    """Label Studio tasks in percent coordinates: boxes from "label", polygons
    from "polygon", both on "image"."""
    tasks: list[dict] = []
    for row in rows:
        W, H = row["width"], row["height"]
        results: list[dict] = []
        for a in row["anns"]:
            x0, y0, x1, y1 = a["box"]
            if a["polys"] and include_segmentations:
                for pts in a["polys"]:
                    results.append({
                        "type": "polygonlabels",
                        "from_name": "polygon",
                        "to_name": "image",
                        "original_width": W,
                        "original_height": H,
                        "value": {
                            "points": [
                                [round(x / W * 100, 2), round(y / H * 100, 2)]
                                for x, y in pts
                            ],
                            "polygonlabels": [a["label"]],
                        },
                    })
            elif include_boxes:
                results.append({
                    "type": "rectanglelabels",
                    "from_name": "label",
                    "to_name": "image",
                    "original_width": W,
                    "original_height": H,
                    "value": {
                        "x": round(x0 / W * 100, 2),
                        "y": round(y0 / H * 100, 2),
                        "width": round((x1 - x0) / W * 100, 2),
                        "height": round((y1 - y0) / H * 100, 2),
                        "rotation": 0,
                        "rectanglelabels": [a["label"]],
                    },
                })
        tasks.append({
            "data": {"image": f"images/{row['file']}"},
            "annotations": [{"result": results}],
        })
    return tasks


def _aug_parent_lookup(project_id: str, manifest: dict) -> dict[str, str]:
    """{export_name: parent_filename} for each augmentation copy on disk."""
    out: dict[str, str] = {}
    for export_name, parent, *_rest in _iter_aug_copies(project_id, manifest):
        out.setdefault(export_name, parent)
    return out


# Pipeline Charlie import routes.


async def _charlie_load_image_pil(
    image: UploadFile | None,
    project_id: str,
    import_id: str,
    *,
    endpoint_tag: str,
) -> "PILImage.Image":
    """The stored import when project_id and import_id are given, else the upload, as upright RGB."""
    if project_id and import_id:
        proj = project_dir(project_id)
        if not proj.exists():
            raise HTTPException(404, "dataset not found")
        # copy=False: read only.
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

    if image is None:
        raise HTTPException(
            400,
            "either (project_id + import_id) or an image file must be provided",
        )
    raw = await _read_upload(image)
    if not raw:
        print(
            f"[{endpoint_tag}] 400 empty upload: filename={image.filename!r} "
            f"content_type={image.content_type!r}"
        )
        raise HTTPException(400, "empty image upload")
    try:
        with PILImage.open(io.BytesIO(raw)) as im:
            return ImageOps.exif_transpose(im).convert("RGB")
    except Exception as e:
        print(
            f"[{endpoint_tag}] 400 decode failed: "
            f"filename={image.filename!r} size_bytes={len(raw)} "
            f"head_hex={raw[:16].hex()} err={type(e).__name__}: {e}"
        )
        raise HTTPException(400, f"could not decode image: {e}")


async def _detect_point_unified(image_pil, point_xy, *, candidate_labels, cache_key=None):
    """Click-to-detect with SAM 3 at the point over the candidate labels; 422
    when it finds nothing."""
    charlie = state.get("charlie")
    if charlie is None:
        raise HTTPException(503, "SAM 3 is not loaded")
    loop = asyncio.get_running_loop()
    W, H = image_pil.size
    px = min(max(float(point_xy[0]), 0.0), float(W - 1))
    py = min(max(float(point_xy[1]), 0.0), float(H - 1))
    labels = [str(t).strip() for t in (candidate_labels or []) if str(t).strip()]
    if not labels:
        raise HTTPException(422, "no mask found at that point")

    def _run_sam3():
        return charlie.segment_point(image_pil, [px, py], labels, cache_key=cache_key)

    try:
        async with state["gpu_lock"].interactive():
            detection, sam3_timings = await loop.run_in_executor(None, _run_sam3)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"click to detect failed: {exc}")
    if detection is None:
        raise HTTPException(422, "no mask found at that point")
    score = float(detection["gd_score"]) if detection.get("gd_score") is not None else None
    return {
        "pipeline": "charlie", "width": W, "height": H,
        "timings": {**sam3_timings, "route": "sam3"},
        "route": "sam3",
        "box_xyxy": [float(c) for c in detection["box"]],
        "mask": {"polygons": (detection["mask"]["polygons"]
                              if isinstance(detection.get("mask"), dict) else [])},
        "mask_score": score,
        "label": detection.get("gd_label") or None, "score": score,
        "rejected": False, "reject_reason": None,
    }


@app.post("/api/charlie/imports/detect_point")
async def charlie_imports_detect_point(
    image: UploadFile | None = File(None),
    point: str = Form(...),
    project_id: str = Form(""),
    import_id: str = Form(""),
    labels: str = Form(""),
):
    """Click-to-detect at `point`, with candidate labels from `labels` or the dataset's tags."""
    try:
        pt = json.loads(point)
        if not (isinstance(pt, list) and len(pt) == 2):
            raise ValueError("point must be [x, y]")
        px, py = float(pt[0]), float(pt[1])
    except Exception as e:
        print(f"[charlie/detect_point] 400 invalid point: raw={point!r} err={e}")
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

    image_pil = await _charlie_load_image_pil(
        image, project_id, import_id, endpoint_tag="charlie/detect_point",
    )
    W, H = image_pil.size

    loop = asyncio.get_running_loop()

    charlie = state.get("charlie")
    if charlie is None:
        raise HTTPException(503, "SAM 3 is not loaded")

    def _run_sam3():
        return charlie.segment_point(
            image_pil, [px, py], candidate_labels,
            cache_key=(f"{project_id}:{import_id}" if import_id else None),
        )

    try:
        async with state["gpu_lock"].interactive():
            detection, sam3_timings = await loop.run_in_executor(None, _run_sam3)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"click to detect failed: {exc}")

    if detection is None:
        raise HTTPException(422, "no mask found at that point")

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


@app.post("/api/charlie/imports/classify_box")
async def charlie_imports_classify_box(
    image: UploadFile | None = File(None),
    box: str = Form(...),
    labels: str = Form(...),
    project_id: str = Form(""),
    import_id: str = Form(""),
):
    """Label a drawn box with the reference resolver on a specific dataset, else with SAM3."""
    import v2_dinov2 as _v2d
    if not _v2d.is_loaded():
        raise HTTPException(503, "the image encoder is still loading; try again shortly")

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
            "reason": "no dataset given, so there are no references to compare with",
        }

    image_pil = await _charlie_load_image_pil(
        image, project_id, import_id, endpoint_tag="charlie/classify_box",
    )

    loop = asyncio.get_running_loop()
    timings: dict[str, float] = {}

    # A SAM3 mask for the crop; without one the crop is the plain box.
    def _segment():
        t0 = time.perf_counter()
        charlie = state.get("charlie")
        detection = None
        if charlie is not None:
            try:
                detection, _t = charlie.segment_box(
                    image_pil, coords_f, candidate_list or None,
                    cache_key=(f"{project_id}:{import_id}" if import_id else None),
                )
            except Exception as e:
                print(f"[charlie/classify_box] sam3 mask failed: {e}")
        ms = (time.perf_counter() - t0) * 1000.0
        return ((detection or {}).get("mask") or None), ms

    try:
        # The box editor waits on this call.
        async with state["gpu_lock"].interactive():
            mask_payload, mask_ms = await loop.run_in_executor(None, _segment)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"classify_box segment failed: {exc}")
    timings["sam3_mask_ms"] = round(mask_ms, 1)

    mask_polys = (
        mask_payload.get("polygons")
        if isinstance(mask_payload, dict) else None
    )

    dataset_type = _dataset_type(project_id)["type"]
    try:
        if dataset_type == "specific":
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
        raise HTTPException(500, f"labelling the box failed: {exc}")

    # Only the specific path sets `ambiguous`.
    if verdict and verdict.get("ambiguous"):
        _charlie_tiebreak(verdict, verdict["sims"])

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
    """Mask and label for a drawn box: the SAM3 hit over the candidate labels that best matches it."""
    charlie = state.get("charlie")
    if charlie is None:
        raise HTTPException(503, "SAM 3 is not loaded")

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

    def _run():
        return charlie.segment_box(image_pil, coords_f, candidate_list)

    try:
        async with state["gpu_lock"].interactive():
            detection, sam3_timings = await loop.run_in_executor(None, _run)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"labelling the box failed: {exc}")

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


@app.post("/api/charlie/imports/segment_box")
async def charlie_imports_segment_box(
    image: UploadFile | None = File(None),
    box: str = Form(...),
    labels: str = Form(""),
    project_id: str = Form(""),
    import_id: str = Form(""),
):
    """Mask for a drawn, moved or resized box: the SAM3 hit over the given
    labels, else the dataset's, that best matches it."""
    charlie = state.get("charlie")
    if charlie is None:
        raise HTTPException(503, "SAM 3 is not loaded")
    try:
        coords = json.loads(box)
        if not (isinstance(coords, list) and len(coords) == 4):
            raise ValueError("box must be [x0, y0, x1, y1]")
        coords_f = [float(c) for c in coords]
    except Exception as e:
        raise HTTPException(400, f"invalid box payload: {e}")
    candidate_labels: list[str] = []
    try:
        parsed = json.loads(labels) if labels else []
        if isinstance(parsed, list):
            candidate_labels = [str(t).strip() for t in parsed if str(t).strip()]
    except Exception:
        candidate_labels = []
    if not candidate_labels and project_id:
        candidate_labels = [str(t) for t in (load_manifest(project_id, copy=False) or {}).get("tags") or []]

    image_pil = await _charlie_load_image_pil(
        image, project_id, import_id, endpoint_tag="charlie/segment_box",
    )
    loop = asyncio.get_running_loop()

    def _run():
        return charlie.segment_box(
            image_pil, coords_f, candidate_labels or None,
            cache_key=(f"{project_id}:{import_id}" if import_id else None),
        )

    try:
        async with state["gpu_lock"].interactive():
            detection, sam3_timings = await loop.run_in_executor(None, _run)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"segmenting the box failed: {exc}")

    mask = (detection or {}).get("mask")
    if not mask:
        raise HTTPException(422, "no mask found in box")
    return {
        "pipeline": "charlie",
        "timings": sam3_timings,
        "box_xyxy": coords_f,
        "mask": mask,
    }


# Jobs, model manager and settings.


@app.get("/api/jobs/active")
async def jobs_active():
    """Queued and running jobs. The queue is in memory, so a restart clears it."""
    jm = state["jobs"]
    out = [
        j.to_public()
        for j in jm.list(limit=100)
        if j.status in ("queued", "running")
    ]
    return {"jobs": out}


@app.get("/api/jobs/{job_id}")
async def job_get(job_id: str):
    """One job in any status; the CLI reads it once the job leaves /api/jobs/active."""
    for j in state["jobs"].list(limit=500):
        if j.id == job_id:
            return j.to_public()
    raise HTTPException(404, "job not found")


@app.get("/api/models/status")
async def models_status():
    return await asyncio.to_thread(models_mgr.status)


@app.post("/api/models/{name}/load")
async def models_load(name: str):
    if name not in models_mgr.REGISTRY:
        raise HTTPException(404, f"unknown model: {name}")
    if not models_mgr.is_downloaded(name):
        raise HTTPException(409, f"{name} weights not downloaded yet")
    if os.environ.get("PK_DISABLE_MODELS", "").lower() in ("1", "true", "yes", "on"):
        raise HTTPException(409, "models are disabled in this engine (PK_DISABLE_MODELS)")
    if state.get("device") == "cpu" and _configured_device() != "cpu":
        raise HTTPException(
            409, "no GPU on this machine (switch Compute to CPU in Settings to force CPU)"
        )
    task = {"sam3": _bg_load_charlie, "dinov2": _bg_load_dino}[name]
    asyncio.create_task(task())
    return {"ok": True, "loading": name}


class HfTokenIn(BaseModel):
    token: str


@app.post("/api/settings/hf-token")
async def hf_token_set(payload: HfTokenIn):
    tok = (payload.token or "").strip()
    if not tok:
        raise HTTPException(400, "empty token")
    result = await asyncio.to_thread(models_mgr.validate_token, tok)
    if not result.get("valid"):
        raise HTTPException(400, result.get("detail") or "token rejected by Hugging Face")
    models_mgr.set_hf_token(tok)
    result["configured"] = True
    if result.get("sam3Access") and not state.get("sam3_off", True):
        if models_mgr.is_downloaded("sam3"):
            if state.get("charlie") is None:
                asyncio.create_task(_bg_load_charlie())
        else:
            async def _dl_then_load() -> None:
                await asyncio.to_thread(models_mgr.start_download, "sam3")
                while True:
                    await asyncio.sleep(5)
                    if models_mgr.is_downloaded("sam3"):
                        await _bg_load_charlie()
                        return
                    with models_mgr._DL_LOCK:
                        rec = dict(models_mgr._DOWNLOADS.get("sam3") or {})
                    if rec.get("status") == "error":
                        return
            asyncio.create_task(_dl_then_load())
    return result


@app.delete("/api/settings/hf-token")
async def hf_token_clear():
    models_mgr.clear_hf_token()
    return {"ok": True}


@app.get("/api/settings")
async def get_settings():
    import workspace as _ws
    cfg = _ws.load_config()
    pref = (cfg.get("device") or "").strip().lower()
    env = (os.environ.get("PK_DEVICE") or "").strip().lower()
    has_mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    valid_pref = pref in ("cuda", "mps", "cpu") or bool(re.fullmatch(r"cuda:\d+", pref))
    gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            try:
                props = torch.cuda.get_device_properties(i)
                gpus.append(
                    {
                        "index": i,
                        "name": props.name,
                        "vramGb": round(props.total_memory / 1024**3),
                    }
                )
            except Exception:
                continue
    return {
        "workspace": str(_ws.dir()),
        "device": state.get("device"),
        "devicePreference": pref if valid_pref else "auto",
        "deviceEnvOverride": env or None,
        "gpuAvailable": bool(torch.cuda.is_available() or has_mps),
        "gpus": gpus,
        "hfTokenConfigured": bool(models_mgr.hf_token()),
        "sam3Repo": models_mgr.SAM3_REPO,
    }


class WorkspaceIn(BaseModel):
    path: str


@app.post("/api/settings/workspace")
async def set_workspace_path(payload: WorkspaceIn):
    """Save a new workspace location, used from the next start."""
    p = (payload.path or "").strip()
    if not p:
        raise HTTPException(400, "empty path")
    import workspace as _ws
    new = _ws.set_workspace(p)
    return {"ok": True, "workspace": str(new), "restartRequired": True}


class DeviceIn(BaseModel):
    device: str


@app.post("/api/settings/device")
async def set_device_preference(payload: DeviceIn):
    """Save the compute device, used from the next start; PK_DEVICE overrides it."""
    want = (payload.device or "").strip().lower()
    if want not in ("auto", "cuda", "mps", "cpu") and not re.fullmatch(r"cuda:\d+", want):
        raise HTTPException(400, "device must be one of: auto, cuda, mps, cpu, cuda:<n>")
    import workspace as _ws

    def _change(cfg: dict) -> None:
        if want == "auto":
            cfg.pop("device", None)
        else:
            cfg["device"] = want

    _ws.update_config(_change)
    return {"ok": True, "devicePreference": want, "restartRequired": True}


class LegacyImportIn(BaseModel):
    path: str


@app.post("/api/import-legacy")
async def import_legacy_endpoint(payload: LegacyImportIn):
    """Import a PixelKit server backup, a folder or zip, with import_legacy.py."""
    raw_path = (payload.path or "").strip()
    if not raw_path:  # Path("") is ".", the engine's own folder
        raise HTTPException(400, "empty path")
    src = Path(raw_path).expanduser()
    if not src.exists():
        raise HTTPException(400, f"path not found: {src}")

    import importlib.util

    mod_path = Path(__file__).resolve().parent.parent / "import_legacy.py"
    if not mod_path.exists():
        raise HTTPException(500, f"import_legacy.py not bundled at {mod_path}")
    spec = importlib.util.spec_from_file_location("pixelkit_import_legacy", mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    try:
        await asyncio.to_thread(mod.run, src)
    except HTTPException:
        raise
    except SystemExit as e:  # raised when src has nothing to import
        raise HTTPException(400, f"legacy import failed: {e}")
    except Exception as e:  # surfaced verbatim in the desktop dialog
        raise HTTPException(500, f"legacy import failed: {e}")
    # The id->path index scans once per process.
    store.rescan()
    return {"ok": True}


# Trained models: YOLOX runs under <dataset>/models/<model dir>/<run id>/.

_RUN_ID_RE = re.compile(r"^[0-9]{8}T[0-9]{6}Z$")
_FAILURE_FILE_RE = re.compile(r"^[0-9]{2,3}_[A-Za-z0-9_-]{1,64}\.jpg$")


# A run whose history.json is younger than this many seconds is training in some process,
# this engine or another. The trainer rewrites the file every 30 s.
_RUN_LIVE_SECONDS = 180


def _history_is_fresh(hist: dict | None) -> bool:
    ts = (hist or {}).get("updatedAt")
    if not isinstance(ts, str):
        return False
    try:
        t = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return False
    return (datetime.now(timezone.utc) - t).total_seconds() < _RUN_LIVE_SECONDS


def _run_status(run_dir: Path, rep: dict | None, hist: dict | None) -> tuple[str, bool]:
    """(status, resumable): done or rejected from report.json, else training while
    history.json is fresh, else interrupted when a checkpoint exists, else failed."""
    if rep is not None:
        return ("rejected" if rep.get("exportRejected") else "done"), False
    resumable = (run_dir / "ckpt_last.pt").is_file()
    hs = (hist or {}).get("status")
    if hs in ("training", "exporting") and _history_is_fresh(hist):
        return "training", False
    if resumable:
        return "interrupted", True
    return "failed", False


def _run_is_live(run_dir: Path) -> bool:
    """True while some process is training this run."""
    if (run_dir / "report.json").is_file():
        return False
    hist = _read_run_json(run_dir / "history.json")
    if not isinstance(hist, dict):
        return False
    return hist.get("status") in ("training", "exporting") and _history_is_fresh(hist)


def _read_run_json(path: Path):
    """A run's JSON file, or None. Retries: on Windows a read can meet the trainer's replace."""
    for i in range(4):
        try:
            return _json_loads(path.read_bytes())
        except FileNotFoundError:
            if not path.parent.is_dir():
                return None
        except PermissionError:
            pass
        except Exception:
            return None
        if i < 3:
            time.sleep(0.03)
    return None


def _run_entry(run_dir: Path) -> dict | None:
    rep = _read_run_json(run_dir / "report.json")
    hist = _read_run_json(run_dir / "history.json")
    if rep is None and hist is None:
        return None  # not listed
    status, resumable = _run_status(run_dir, rep, hist)
    src = rep if rep is not None else (hist or {})
    thr = (rep or {}).get("threshold") or None
    history = (rep or {}).get("history") if rep is not None else (hist or {}).get("history")
    history = history if isinstance(history, list) else []
    try:
        epochs = int(src.get("epochs") or 0)
        qat = int(src.get("qat_epochs") if rep is not None else src.get("qatEpochs") or 0)
    except (TypeError, ValueError):
        epochs, qat = 0, 0
    return {
        "run": run_dir.name,
        "model": src.get("model") or run_dir.parent.name,
        "size": src.get("size") or ("tiny" if run_dir.parent.name == "yolox_tiny" else "nano"),
        "status": status,
        "resumable": resumable,
        "error": (hist or {}).get("error") if rep is None else None,
        "imgsz": src.get("imgsz"),
        "epochs": epochs or None,
        "qatEpochs": qat,
        "epochsDone": len(history),
        "totalEpochs": (epochs + qat) or None,
        "images": (rep or {}).get("images"),
        "split": (rep or {}).get("split"),
        "batch": (rep or {}).get("batch"),
        "ap50Fp32": (rep or {}).get("ap50_fp32"),
        "ap50Int8": (rep or {}).get("ap50_int8"),
        "minutes": (rep or {}).get("minutes"),
        "hasExport": (run_dir / "export.zip").is_file(),
        "exportRejected": bool((rep or {}).get("exportRejected")),
        "initRun": (rep or {}).get("initRun"),
        "resumed": bool((rep or {}).get("resumed")),
        "threshold": ({**{k: thr.get(k) for k in ("conf", "f1", "precision", "recall", "calibrated")},
                       "fallback": bool(thr.get("fallback") or (not thr.get("calibrated")
                                                                and not thr.get("f1")
                                                                and not thr.get("precision")))}
                      if isinstance(thr, dict) else None),
        "failuresCount": len((rep or {}).get("failures") or []),
        "hasPredictions": (run_dir / "predictions.json").is_file(),
        "warningsCount": len((rep or {}).get("warnings") or []),
        # ST Edge AI compiler summary, when the compiler ran.
        "neuron6": {
            k: ((rep or {}).get("neuron6") or {}).get(k)
            for k in ("verified", "available", "error", "reason",
                      "epochsSoftware", "epochsHardware", "epochsTotal",
                      "weightsBytes", "activationsBytes",
                      "poolRemainingBytes", "poolUsedPct")
        },
        # Live curves; a finished run's come from the detail endpoint.
        "history": history if rep is None else None,
    }


def _iter_run_dirs(d: Path):
    import train_yolox as _ty

    for model_dir in _ty.MODEL_DIRS:
        root = d / "models" / model_dir
        if not root.is_dir():
            continue
        for run_dir in root.iterdir():
            if run_dir.is_dir() and _RUN_ID_RE.fullmatch(run_dir.name):
                yield run_dir


@app.get("/api/v2/projects/{project_id}/train/preflight")
async def train_preflight(
    project_id: str,
    imgsz: int = 256,
    split: float = 0.8,
    size: str = "nano",
    augmentations: int = 1,
    evalProject: str = "",
):
    """Image and class counts, blockers and warnings for training, from the trainer's checks."""
    import train_yolox as _ty

    if not store.dataset_exists(project_id):
        raise HTTPException(404, "dataset not found")
    cfg = _train_cfg_from_params({
        "imgsz": imgsz, "split": split, "size": size,
        "augmentations": bool(augmentations), "evalProject": evalProject,
    }, "cpu")
    if cfg.eval_project and not store.dataset_exists(cfg.eval_project):
        return {"ok": False, "blockers": ["held-out dataset not found"], "warnings": [],
                "classes": [], "images": None}
    try:
        return await asyncio.to_thread(_ty.preflight, project_id, cfg)
    except ValueError as e:
        return {"ok": False, "blockers": [str(e)], "warnings": [], "classes": [], "images": None}


@app.get("/api/v2/projects/{project_id}/models")
async def trained_models_list(project_id: str):
    try:
        d = store.dataset_dir(project_id)
    except KeyError:
        raise HTTPException(404, "dataset not found")
    out = []
    for run_dir in sorted(_iter_run_dirs(d), key=lambda p: p.name, reverse=True):
        entry = _run_entry(run_dir)
        if entry is not None:
            out.append(entry)
    return {"models": out}


@app.get("/api/v2/projects/{project_id}/models/{run_id}")
async def trained_model_detail(project_id: str, run_id: str):
    """One run with its report.json, or with its history.json when it has no report."""
    import train_yolox as _ty

    if not _RUN_ID_RE.fullmatch(run_id):
        raise HTTPException(400, "bad run id")
    if not store.dataset_exists(project_id):
        raise HTTPException(404, "dataset not found")
    run_dir = _ty.find_run_dir(project_id, run_id)
    if run_dir is None:
        raise HTTPException(404, "run not found")
    entry = _run_entry(run_dir)
    if entry is None:
        raise HTTPException(404, "run not found")
    report = _read_run_json(run_dir / "report.json")
    history = _read_run_json(run_dir / "history.json") if report is None else None
    return {**entry, "report": report, "live": history}


@app.get("/api/v2/projects/{project_id}/models/{run_id}/export")
async def trained_model_export(project_id: str, run_id: str):
    import train_yolox as _ty

    if not _RUN_ID_RE.fullmatch(run_id):
        raise HTTPException(400, "bad run id")
    if not store.dataset_exists(project_id):
        raise HTTPException(404, "dataset not found")
    run_dir = _ty.find_run_dir(project_id, run_id)
    z = (run_dir / "export.zip") if run_dir is not None else None
    if z is None or not z.is_file():
        raise HTTPException(404, "export not found")
    name = _safe_slug((load_manifest(project_id, copy=False) or {}).get("name") or project_id)
    return FileResponse(
        z,
        media_type="application/zip",
        filename=f"{name}-neuron6-{run_id}.zip",
    )


@app.get("/api/v2/projects/{project_id}/models/{run_id}/predictions")
async def trained_model_predictions(project_id: str, run_id: str):
    """The run's predictions.json: each val image's ground truth and predicted boxes."""
    import train_yolox as _ty

    if not _RUN_ID_RE.fullmatch(run_id):
        raise HTTPException(400, "bad run id")
    if not store.dataset_exists(project_id):
        raise HTTPException(404, "dataset not found")
    run_dir = _ty.find_run_dir(project_id, run_id)
    path = (run_dir / "predictions.json") if run_dir is not None else None
    if path is None or not path.is_file():
        raise HTTPException(404, "no predictions for this run")
    return FileResponse(path, media_type="application/json")


@app.get("/api/v2/projects/{project_id}/models/{run_id}/failures/{name}")
async def trained_model_failure_image(project_id: str, run_id: str, name: str):
    import train_yolox as _ty

    if not _RUN_ID_RE.fullmatch(run_id) or not _FAILURE_FILE_RE.fullmatch(name):
        raise HTTPException(400, "bad id")
    if not store.dataset_exists(project_id):
        raise HTTPException(404, "dataset not found")
    run_dir = _ty.find_run_dir(project_id, run_id)
    if run_dir is None:
        raise HTTPException(404, "run not found")
    root = (run_dir / "failures").resolve()
    target = (root / name).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        raise HTTPException(403, "forbidden")
    if not target.is_file():
        raise HTTPException(404, "image not found")
    return FileResponse(target, media_type="image/jpeg",
                        headers={"Cache-Control": "private, max-age=86400"})


@app.post("/api/v2/projects/{project_id}/models/{run_id}/resume")
async def trained_model_resume(project_id: str, run_id: str, request: Request):
    """Resume an interrupted run from its checkpoint, which carries the run's settings."""
    import train_yolox as _ty

    if not _RUN_ID_RE.fullmatch(run_id):
        raise HTTPException(400, "bad run id")
    if not store.dataset_exists(project_id):
        raise HTTPException(404, "dataset not found")
    body = {}
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    # No await from here to schedule(), so two concurrent resumes cannot both pass.
    run_dir = _ty.find_run_dir(project_id, run_id)
    if run_dir is None or not (run_dir / "ckpt_last.pt").is_file():
        raise HTTPException(400, "nothing to resume: that run has no checkpoint")
    if (run_dir / "report.json").is_file():
        raise HTTPException(409, "that run already finished")
    if _train_job_active(project_id):
        raise HTTPException(409, "a training run is already queued or running for this dataset")
    if _augment_job_active(project_id):
        raise HTTPException(409, "augmentations are being generated for this dataset; resume when that finishes")
    if _run_is_live(run_dir):
        raise HTTPException(409, "that run is still training in another process")
    params = {"resumeRun": run_id}
    dev = str((body or {}).get("device") or "").strip()
    if dev:
        params["device"] = dev
    job = state["jobs"].schedule(kind="train_yolox", project=project_id, params=params, n_images=0)
    return {"jobId": job.id, **job.to_public()}


@app.delete("/api/v2/projects/{project_id}/models/{run_id}")
async def trained_model_delete(project_id: str, run_id: str):
    """Delete a run. Refused while a training or model-labelling job for the dataset is live."""
    import shutil

    import train_yolox as _ty

    if not _RUN_ID_RE.fullmatch(run_id):
        raise HTTPException(400, "bad run id")
    if not store.dataset_exists(project_id):
        raise HTTPException(404, "dataset not found")
    run_dir = _ty.find_run_dir(project_id, run_id)
    if run_dir is None:
        raise HTTPException(404, "run not found")
    if _train_job_active(project_id) or _run_is_live(run_dir):
        raise HTTPException(409, "a training run is live for this dataset; wait for it to finish")
    if any(j.project == project_id and j.kind == "label_model" and j.status in ("queued", "running")
           for j in state["jobs"].jobs.values()):
        raise HTTPException(409, "a labelling job is using this dataset's runs; wait for it to finish")
    try:
        await asyncio.to_thread(shutil.rmtree, run_dir)
    except OSError as e:
        raise HTTPException(500, f"could not delete the run: {e}")
    await _sync_has_model(project_id)
    return {"ok": True, "run": run_id}


# Dataset snapshots: dataset.json and annotations/*.json, without images, as zips in snapshots/.

_SNAP_ID_RE = re.compile(r"^[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}$")


def _snapshot_write(d: Path) -> dict:
    """Zip the annotation state into snapshots/ and return its meta. Called on a worker thread."""
    dj = d / "dataset.json"
    if not dj.exists():
        raise HTTPException(404, "dataset.json missing")
    h = hashlib.blake2b(digest_size=4)
    payload: list[tuple[str, bytes]] = []
    data = dj.read_bytes()
    h.update(data)
    payload.append(("dataset.json", data))
    ann_dir = d / "annotations"
    if ann_dir.exists():
        for p in sorted(ann_dir.glob("*.json")):
            b = p.read_bytes()
            h.update(p.name.encode())
            h.update(b)
            payload.append((f"annotations/{p.name}", b))
    snap_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + h.hexdigest()
    sdir = d / "snapshots"
    sdir.mkdir(exist_ok=True)
    path = sdir / f"{snap_id}.zip"
    tmp = sdir / f"{snap_id}.zip.tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for name, b in payload:
            zf.writestr(name, b)
    store.replace(tmp, path)
    return {
        "id": snap_id,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "sizeBytes": path.stat().st_size,
        "annotations": len(payload) - 1,
    }


def _snapshot_meta(p: Path) -> dict:
    stem = p.name[:-4]
    try:
        created = (
            datetime.strptime(stem.split("-")[0], "%Y%m%dT%H%M%SZ")
            .replace(tzinfo=timezone.utc)
            .isoformat()
        )
    except ValueError:
        created = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat()
    return {"id": stem, "createdAt": created, "sizeBytes": p.stat().st_size}


@app.get("/api/v2/projects/{project_id}/snapshots")
async def snapshots_list(project_id: str):
    try:
        d = store.dataset_dir(project_id)
    except KeyError:
        raise HTTPException(404, "dataset not found")
    sdir = d / "snapshots"
    out = []
    if sdir.exists():
        for p in sorted(sdir.glob("*.zip"), reverse=True):
            if _SNAP_ID_RE.fullmatch(p.name[:-4]):
                out.append(_snapshot_meta(p))
    return {"snapshots": out}


def _project_job_active(project_id: str) -> bool:
    mgr = state.get("jobs")
    if mgr is None:
        return False
    try:
        return any(
            j.project == project_id and j.status in ("queued", "running")
            for j in mgr.list()
        )
    except Exception:
        return False


@app.post("/api/v2/projects/{project_id}/snapshots")
async def snapshots_create(project_id: str):
    try:
        d = store.dataset_dir(project_id)
    except KeyError:
        raise HTTPException(404, "dataset not found")
    # The write lock keeps a flush from landing between dataset.json and the annotation files.
    lock = await _manifest_write_lock(project_id)
    async with lock:
        return await asyncio.to_thread(_snapshot_write, d)


@app.post("/api/v2/projects/{project_id}/snapshots/{snapshot_id}/restore")
async def snapshots_restore(project_id: str, snapshot_id: str):
    if not _SNAP_ID_RE.fullmatch(snapshot_id):
        raise HTTPException(400, "bad snapshot id")
    try:
        d = store.dataset_dir(project_id)
    except KeyError:
        raise HTTPException(404, "dataset not found")
    path = d / "snapshots" / f"{snapshot_id}.zip"
    if not path.exists():
        raise HTTPException(404, "snapshot not found")
    # A running job holds a loaded manifest that it flushes later.
    if _project_job_active(project_id):
        raise HTTPException(409, "a job is running on this dataset; restore when it finishes")

    def _do() -> dict:
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
            if "dataset.json" not in names:
                raise HTTPException(400, "corrupt snapshot: no dataset.json")
            new_manifest = zf.read("dataset.json")
            anns = {
                Path(n).name: zf.read(n)  # basename only, against zip-slip
                for n in names
                if n.startswith("annotations/") and n.endswith(".json") and n.count("/") == 1
            }
        # Records of images deleted since the snapshot are dropped.
        m_new = store._loads(new_manifest)
        live = {"imports": d / "images", "references": d / "references"}
        gone: set[str] = set()
        for key, folder in live.items():
            kept = []
            for e in m_new.get(key) or []:
                fn = (e or {}).get("filename") if isinstance(e, dict) else None
                if fn and (folder / fn).is_file():
                    kept.append(e)
                elif isinstance(e, dict) and e.get("id"):
                    gone.add(str(e["id"]))
            if key in m_new:
                m_new[key] = kept
        _replace_missing_cover(m_new)
        if gone:
            anns = {fn: b for fn, b in anns.items() if Path(fn).stem not in gone}
        new_manifest = store._dumps(m_new)
        safety = _snapshot_write(d)
        ann_dir = d / "annotations"
        ann_dir.mkdir(exist_ok=True)
        for p in ann_dir.glob("*.json"):
            p.unlink()
        for fn, b in anns.items():
            tmp = store.tmp_path(ann_dir / fn)
            tmp.write_bytes(b)
            store.replace(tmp, ann_dir / fn)
        tmp = store.tmp_path(d / "dataset.json")
        tmp.write_bytes(new_manifest)
        # dataset.json last: its mtime is the manifest cache's validity stamp.
        store.replace(tmp, d / "dataset.json")
        # The files changed behind store.save()'s digest cache and the manifest cache.
        store.forget(project_id)
        invalidate_manifest_cache(project_id)
        _invalidate_project_payloads(project_id)
        for e in m_new.get("imports") or []:  # previews show the restored labels
            if isinstance(e, dict) and e.get("id"):
                _invalidate_labelled_preview(project_id, str(e["id"]))
        return {"ok": True, "restored": snapshot_id, "safetySnapshot": safety["id"],
                "dropped": len(gone)}

    lock = await _manifest_write_lock(project_id)
    async with lock:
        return await asyncio.to_thread(_do)



# Reference editor tools, over pipeline_charlie and _detect_point_unified.

async def _reference_image_pil(
    image: UploadFile | None, project_id: str, filename: str, endpoint: str
) -> "PILImage.Image":
    if image is not None:
        raw = await _read_upload(image)
        if raw:
            img = PILImage.open(io.BytesIO(raw))
            return ImageOps.exif_transpose(img).convert("RGB")
    if project_id and filename:
        refs_root = (project_dir(project_id) / "references").resolve()
        p = (refs_root / filename).resolve()
        try:
            p.relative_to(refs_root)
        except ValueError:
            raise HTTPException(403, "forbidden")
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
        raise HTTPException(503, "SAM 3 is not loaded")
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
        raise HTTPException(503, "SAM 3 is not loaded")
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
    result = await _detect_point_unified(image_pil, pt, candidate_labels=candidate)
    fl = (force_label or "").strip()
    if fl and isinstance(result, dict):
        result["label"] = fl
    return result



# The exported UI, ui/out or PIXELKIT_UI_DIR. Registered last, so every API route matches first.

def _ui_dir() -> Path | None:
    env = (os.environ.get("PIXELKIT_UI_DIR") or "").strip()
    cand = Path(env).expanduser() if env else Path(__file__).resolve().parents[2] / "ui" / "out"
    return cand if (cand / "index.html").is_file() else None


_UI_DIR = _ui_dir()
if _UI_DIR:
    print(f"[server] serving UI from {_UI_DIR}")


@app.get("/{full_path:path}", include_in_schema=False)
async def serve_ui(full_path: str):
    if _UI_DIR is None or full_path.startswith("api/") or full_path == "api":
        raise HTTPException(404, "not found")
    p = (_UI_DIR / full_path).resolve() if full_path else _UI_DIR / "index.html"
    try:
        p.relative_to(_UI_DIR)
    except ValueError:
        raise HTTPException(404, "not found")
    if p.is_file():
        return FileResponse(p)
    html = _UI_DIR / f"{full_path}.html"
    if html.is_file():
        return FileResponse(html)
    if full_path.split("/", 1)[0] == "app":
        return FileResponse(_UI_DIR / "app.html")
    nf = _UI_DIR / "404.html"
    if nf.is_file():
        return FileResponse(nf, status_code=404)
    raise HTTPException(404, "not found")


if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8001, reload=False)
