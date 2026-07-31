"""Lightweight performance instrumentation.

Two things live here:

1.  A per-request middleware that captures response size + serialise
    time for any v2 / v3 endpoint, plus the headline meta endpoints
    (initial, overview, annotations, dataset-stats) tagged for the
    rollup logs.

2.  A POST /api/perf/log handler that accepts batched events from
    the FE and appends them to a daily-rotating ndjson on disk.

Designed to be cheap enough to leave on in production:
- middleware only stamps a perf_counter on request start, reads a
  few headers on exit, writes one ndjson line via an async background
  task (never blocks the response).
- The log writer batches via a single asyncio.Queue + a background
  writer task so concurrent requests never contend for the file.
- Gated on PIXELKIT_PERF_LOG=1 (FE side ships NEXT_PUBLIC_PERF_LOG).

The output lives under logs/perf-YYYYMMDD.ndjson. Rotate manually
or hook up logrotate; this module won't try to be a log shipper.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware


# Toggle. Cheaper to short-circuit at the middleware boundary than
# inside it, but the perf-log writer is fire-and-forget anyway.
_ENABLED = os.environ.get("PIXELKIT_PERF_LOG", "0") == "1"

# Single asyncio.Queue + writer task, set on app startup. We do this
# lazily inside the middleware so importing this module has no side
# effects (matters because server.py imports this at top-level).
_log_queue: asyncio.Queue[dict] | None = None
_writer_task: asyncio.Task | None = None

# Where to write. PIXELKIT_PERF_DIR overrides; defaults to logs/ at
# the backend repo root so it's gitignored next to the other ndjson
# log files.
_LOG_DIR = Path(os.environ.get("PIXELKIT_PERF_DIR") or
                Path(__file__).resolve().parents[1] / "logs")

# Endpoints we care about logging. Other auth / static / health
# endpoints are skipped to keep the log signal-to-noise high.
_LOGGED_PREFIXES = (
    "/api/v2/projects",
    "/api/v3/projects",
    "/api/projects",
    "/api/jobs",
    "/api/charlie",
)


def perf_enabled() -> bool:
    return _ENABLED


def _today_log_path() -> Path:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    return _LOG_DIR / f"perf-{datetime.utcnow().strftime('%Y%m%d')}.ndjson"


async def _writer_loop(queue: asyncio.Queue[dict]) -> None:
    """Consume events from the queue and append them to the per-day
    log file. Batches by reading whatever's already queued so a
    burst doesn't open/close the file 100 times."""
    while True:
        first = await queue.get()
        batch: list[dict] = [first]
        # Drain whatever's already queued (non-blocking).
        while True:
            try:
                batch.append(queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        # Write the batch in one open() call.
        try:
            path = _today_log_path()
            lines = "\n".join(json.dumps(e, separators=(",", ":")) for e in batch) + "\n"
            # Run the blocking write off the event loop so a slow
            # disk can't pause request handling.
            await asyncio.get_running_loop().run_in_executor(
                None, lambda: _append(path, lines),
            )
        except Exception as exc:
            # Never raise from the writer — the alternative is
            # losing request handling. Just print for debug.
            print(f"[perf-log] writer error: {exc}")


def _append(path: Path, payload: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(payload)


def _ensure_writer() -> None:
    """Spin up the queue + writer task on first use. Idempotent;
    safe to call from the middleware on every request."""
    global _log_queue, _writer_task
    if _log_queue is not None:
        return
    _log_queue = asyncio.Queue(maxsize=2048)
    _writer_task = asyncio.create_task(_writer_loop(_log_queue))


def log_event(event: dict) -> None:
    """Fire-and-forget. Queue is bounded; if it fills (very
    unlikely under normal load) we drop the event silently rather
    than block the caller."""
    if not _ENABLED:
        return
    if _log_queue is None:
        _ensure_writer()
        assert _log_queue is not None
    try:
        _log_queue.put_nowait(event)
    except asyncio.QueueFull:
        # Drop on overflow. Better to lose a perf log than to back
        # up the response path.
        pass


class PerfMiddleware(BaseHTTPMiddleware):
    """Stamps request start + read response Content-Length + path on
    exit. Skips any path we don't care about so we don't drown the
    log in static-asset misses.

    Cost: one perf_counter() + dict allocation per request. Worth it.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        if not _ENABLED:
            return await call_next(request)
        path = request.url.path
        if not path.startswith(_LOGGED_PREFIXES):
            return await call_next(request)
        _ensure_writer()
        t0 = time.perf_counter()
        response = await call_next(request)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        # Content-Length might be missing for streamed responses.
        cl = response.headers.get("Content-Length")
        log_event({
            "kind": "request",
            "ts": time.time(),
            "method": request.method,
            "path": path,
            "status": response.status_code,
            "elapsed_ms": round(dt_ms, 2),
            "bytes": int(cl) if cl and cl.isdigit() else None,
        })
        return response


# ─── /api/perf/log endpoint ─────────────────────────────────────────


class PerfEvent(BaseModel):
    kind: str
    ts: float | None = None
    # Open shape — every other field is application-specific. The FE
    # sends marks like { kind: "fetch", path: "...", elapsed_ms: 8 }
    # or { kind: "long-task", duration: 120 }. We don't validate
    # beyond ensuring kind is present.
    data: dict[str, Any] | None = None


class PerfBatch(BaseModel):
    # Single batch endpoint to keep request overhead low. FE buffers
    # in-memory and POSTs every ~30 s or before unload.
    events: list[PerfEvent]
    # Optional client identifier. Doesn't have to be unique; just
    # gives us a way to filter logs to a specific session.
    session: str | None = None


async def perf_log_handler(batch: PerfBatch, request: Request) -> JSONResponse:
    if not _ENABLED:
        return JSONResponse({"ok": True, "logged": 0, "disabled": True})
    _ensure_writer()
    user_agent = request.headers.get("user-agent", "")[:200]
    referer = request.headers.get("referer", "")[:200]
    for e in batch.events:
        log_event({
            "kind": e.kind,
            "ts": e.ts or time.time(),
            "session": batch.session,
            "ua": user_agent,
            "referer": referer,
            **(e.data or {}),
        })
    return JSONResponse({"ok": True, "logged": len(batch.events)})
