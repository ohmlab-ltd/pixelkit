"""In-memory job queue with N concurrent async workers.

Replaces the fire-and-forget `asyncio.create_task(...)` model — every long-
running ML pass (auto-label, backfill segmentation) goes through here so:
  - GPU access is arbitrated by the priority gate (not by the queue), so
    multiple runners can interleave image-by-image instead of one job
    blocking the next entirely
  - The frontend gets a uniform SSE feed per job
  - The Terminal page can show queue length, current job, cost, history
  - Cancelling a queued job is trivial

History is persisted to audit.db via JobManager.on_finish (server.py wires
that to add_event). On startup, JobManager.hydrate_from_audit() rehydrates
finished jobs into self.jobs so the Terminal sees the same history across
restarts. Active (queued/running) jobs that didn't reach a terminal state
before the previous backend died are loaded as `interrupted`.
"""
from __future__ import annotations

import asyncio
import json
import secrets
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

# Cost model. Flat hourly cloud spend in USD — replaces the older
# power-draw × kWh estimate. The terminal still receives the value
# through the legacy `costPence` JSON field (its name is wire-shape
# compatible) but the units are now USD cents: 100 cents = $1.
COST_USD_PER_HOUR = 0.207
# Legacy fields kept on the /stats response for backwards-compat
# with any old client; the terminal stopped reading them in the
# accompanying frontend commit.
POWER_W = 350.0
COST_PENCE_PER_KWH = 28.0

JobStatus = str  # "queued" | "running" | "done" | "failed" | "cancelled"
JobKind = str    # "label" | "segment"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Job:
    id: str
    kind: JobKind
    project: str
    params: dict
    user: str
    status: JobStatus = "queued"
    queued_at: str = field(default_factory=_utcnow)
    started_at: str | None = None
    finished_at: str | None = None
    progress: dict = field(default_factory=dict)  # {index, total, image, phase}
    error: str | None = None
    elapsed_s: float = 0.0
    n_images: int = 0
    # Filled in once the job completes.
    cost_pence: float = 0.0

    # Internal — wall-clock start used for live elapsed/cost while running.
    _start_monotonic: float | None = None

    def to_public(self) -> dict:
        elapsed = self.live_elapsed_s()
        return {
            "id": self.id,
            "kind": self.kind,
            "project": self.project,
            "params": self.params,
            "user": self.user,
            "status": self.status,
            "queuedAt": self.queued_at,
            "startedAt": self.started_at,
            "finishedAt": self.finished_at,
            "progress": self.progress,
            "error": self.error,
            "elapsedS": round(elapsed, 1),
            "n_images": self.n_images,
            "costPence": round(elapsed_to_cost(elapsed), 5),
        }

    def live_elapsed_s(self) -> float:
        if self.status == "running" and self._start_monotonic is not None:
            return max(0.0, time.monotonic() - self._start_monotonic)
        return self.elapsed_s


def elapsed_to_cost(elapsed_s: float) -> float:
    """Cloud spend for `elapsed_s` seconds at COST_USD_PER_HOUR.

    Returns USD cents so the value lines up with the terminal's
    `costPence` field (renamed in copy to "$" without changing the
    field name): cents = hours * dollars-per-hour * 100."""
    hours = max(0.0, elapsed_s) / 3600.0
    return hours * COST_USD_PER_HOUR * 100.0


# Type alias for the runner the manager invokes for each job kind.
# Receives (job, emit, cancel_event). emit(event, data) pushes an SSE event.
JobRunner = Callable[["Job", Callable[[str, Any], Awaitable[None]], asyncio.Event], Awaitable[None]]


class JobManager:
    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        # Per-job SSE listeners. Multiple clients can subscribe to one job.
        self.listeners: dict[str, list[asyncio.Queue]] = {}
        self.cancel_events: dict[str, asyncio.Event] = {}
        # N concurrent worker tasks. Each pulls jobs off the same
        # asyncio.Queue — multi-consumer-safe — so several runners can
        # be in flight at once. The GPU gate (`state["gpu_lock"]`)
        # arbitrates actual CUDA access, letting augment + label
        # interleave image-by-image instead of one blocking the other
        # for the whole run.
        self.worker_tasks: list[asyncio.Task] = []
        self.runners: dict[str, JobRunner] = {}
        # Bounded history so memory doesn't grow forever in long sessions.
        self.history_cap = 500
        # Optional hook fired once per job after it reaches a terminal state
        # (done / failed / cancelled). Used by the server to write the
        # finished job to the audit log.
        self.on_finish: Callable[[Job], None] | None = None

    def register_runner(self, kind: str, runner: JobRunner) -> None:
        self.runners[kind] = runner

    def _fire_on_finish(self, job: "Job") -> None:
        if self.on_finish is None:
            return
        try:
            self.on_finish(job)
        except Exception as e:  # noqa: BLE001
            import traceback
            tb = traceback.format_exc()
            print(f"[jobs] on_finish hook failed for {job.id} ({job.kind}): {type(e).__name__}: {e}\n{tb}")

    def start_worker(self, n: int = 3) -> None:
        """Ensure N live worker tasks. Idempotent — restarts only the
        slots whose task died. Multiple workers let augment +
        label_charlie + image-processing run concurrently; the GPU
        gate keeps actual inference serialised, but the inter-image
        scaffolding (manifest writes, IO, image decode) happens in
        parallel and the gate ping-pongs between them per image."""
        # Drop dead/None entries, then top up to n.
        self.worker_tasks = [t for t in self.worker_tasks if not t.done()]
        while len(self.worker_tasks) < n:
            self.worker_tasks.append(asyncio.create_task(self._worker()))

    @property
    def worker_task(self) -> asyncio.Task | None:
        """Back-compat shim — callers (e.g. the server-side watchdog)
        used to inspect a single `worker_task`. Return the first live
        slot so a `t.done()` check still tells them whether at least
        one worker is alive."""
        for t in self.worker_tasks:
            if not t.done():
                return t
        return self.worker_tasks[0] if self.worker_tasks else None

    async def run_inline(
        self,
        kind: str,
        project: str,
        params: dict,
        user: str,
        runner: Callable[["Job", Callable[[str, Any], Awaitable[None]], asyncio.Event], Awaitable[Any]],
        n_images: int = 1,
    ) -> Any:
        """Track an ad-hoc backend task in the same job ledger as queued runs,
        but execute it immediately on the caller's task instead of waiting for
        the worker. Returns whatever the runner returns. Used for short
        interactive ML calls (per-box SAM2, per-box GD classify) so they show
        up in the Terminal alongside batch jobs."""
        job = Job(
            id=secrets.token_hex(6),
            kind=kind,
            project=project,
            params=params,
            user=user or "anonymous",
            n_images=n_images,
        )
        cancel_event = asyncio.Event()
        self.jobs[job.id] = job
        self.cancel_events[job.id] = cancel_event

        job.status = "running"
        job.started_at = _utcnow()
        job._start_monotonic = time.monotonic()
        await self._emit(job, "running", job.to_public())

        async def emit(event: str, data: Any) -> None:
            if event == "progress" and isinstance(data, dict):
                job.progress = {
                    "index": data.get("index"),
                    "total": data.get("total"),
                    "image": data.get("image"),
                    "phase": "running",
                }
            elif event == "status" and isinstance(data, dict):
                job.progress = {**job.progress, "phase": data.get("phase", "running"), "total": data.get("total")}
            await self._emit(job, event, data)

        try:
            result = await runner(job, emit, cancel_event)
            job.status = "cancelled" if cancel_event.is_set() else "done"
            return result
        except Exception as e:  # noqa: BLE001
            job.status = "failed"
            job.error = str(e)
            raise
        finally:
            job.finished_at = _utcnow()
            if job._start_monotonic is not None:
                job.elapsed_s = time.monotonic() - job._start_monotonic
            job.cost_pence = elapsed_to_cost(job.elapsed_s)
            await self._emit(job, job.status, job.to_public())
            self._fire_on_finish(job)
            self._gc_history()

    def schedule(self, kind: str, project: str, params: dict, user: str, n_images: int = 0) -> Job:
        job = Job(
            id=secrets.token_hex(6),
            kind=kind,
            project=project,
            params=params,
            user=user or "anonymous",
            n_images=n_images,
        )
        self.jobs[job.id] = job
        self.cancel_events[job.id] = asyncio.Event()
        self.queue.put_nowait(job.id)
        self._gc_history()
        return job

    def cancel(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if not job:
            return False
        if job.status in ("done", "failed", "cancelled"):
            return False
        ev = self.cancel_events.get(job_id)
        if ev:
            ev.set()
        if job.status == "queued":
            job.status = "cancelled"
            job.finished_at = _utcnow()
        return True

    def list(self, status: str | None = None, limit: int = 200) -> list[Job]:
        items = list(self.jobs.values())
        if status:
            items = [j for j in items if j.status == status]
        items.sort(key=lambda j: j.queued_at, reverse=True)
        return items[:limit]

    def hydrate_from_audit(self, events: list[dict]) -> int:
        """Re-populate `self.jobs` from audit.db rows so Terminal history
        survives a backend restart. `events` is a list of `kind="job"` rows
        as returned by audit.list_events. Skips ids already present (e.g.
        from a fresh job that's just landed). Returns count loaded."""
        loaded = 0
        for e in events:
            jid = e.get("id")
            if not jid or jid in self.jobs:
                continue
            status = e.get("status") or "done"
            # Anything not at a terminal state when audit was written is
            # treated as interrupted — the previous process died mid-job.
            if status not in ("done", "failed", "cancelled", "interrupted"):
                status = "interrupted"
            # `job_kind` is the audit blob field (renamed from `kind` to
            # avoid clashing with the row-level kind column when writing).
            # Old rows that used the unrenamed field (or no field at all)
            # fall back through `kind` to "label" for forward compat.
            job_kind = e.get("job_kind") or e.get("kind") or "label"
            if job_kind == "job":  # row-level "job" leaked through, treat as unknown
                job_kind = "label"
            self.jobs[jid] = Job(
                id=jid,
                kind=job_kind,
                project=e.get("project") or "",
                params={},
                user=e.get("user") or "anonymous",
                status=status,
                queued_at=e.get("ts") or _utcnow(),
                started_at=e.get("ts"),
                finished_at=e.get("ts"),
                error=e.get("error"),
                elapsed_s=float(e.get("elapsed_s") or 0.0),
                n_images=int(e.get("n_images") or 0),
                cost_pence=float(e.get("cost_pence") or 0.0),
            )
            loaded += 1
        return loaded

    def stats(self) -> dict:
        today = datetime.now(timezone.utc).date().isoformat()
        running = [j for j in self.jobs.values() if j.status == "running"]
        queued = [j for j in self.jobs.values() if j.status == "queued"]
        today_jobs = [j for j in self.jobs.values() if (j.queued_at or "").startswith(today)]
        total_cost_today = sum(elapsed_to_cost(j.live_elapsed_s()) for j in today_jobs)
        return {
            "running": len(running),
            "queued": len(queued),
            "todayCount": len(today_jobs),
            "todayCostPence": round(total_cost_today, 4),
            "powerW": POWER_W,
            "costPencePerKwh": COST_PENCE_PER_KWH,
        }

    async def subscribe(self, job_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self.listeners.setdefault(job_id, []).append(q)
        # If the job already finished, replay a single done event so late
        # subscribers don't hang.
        job = self.jobs.get(job_id)
        if job and job.status in ("done", "failed", "cancelled"):
            q.put_nowait({"event": job.status, "data": json.dumps(job.to_public())})
        return q

    def unsubscribe(self, job_id: str, q: asyncio.Queue) -> None:
        listeners = self.listeners.get(job_id) or []
        if q in listeners:
            listeners.remove(q)

    async def _emit(self, job: Job, event: str, data: Any) -> None:
        payload = {"event": event, "data": json.dumps(data)}
        for q in list(self.listeners.get(job.id, [])):
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                pass

    async def _worker(self) -> None:
        # The whole loop body sits inside a top-level try/except so no
        # exception (including ones from `_emit` / `_fire_on_finish` /
        # JSON encoding issues in `to_public`) can kill the worker. If
        # one job blows up the next still gets picked up.
        while True:
            try:
                await self._process_one()
            except asyncio.CancelledError:
                # Cancellation is the intended way to stop the worker
                # (e.g. on shutdown) — let it propagate.
                raise
            except Exception as e:  # noqa: BLE001
                print(f"[jobs] worker recovered from unexpected error: {e}")

    async def _process_one(self) -> None:
        job_id = await self.queue.get()
        job = self.jobs.get(job_id)
        if not job:
            return
        if job.status == "cancelled":
            return

        runner = self.runners.get(job.kind)
        if runner is None:
            job.status = "failed"
            job.error = f"no runner for kind {job.kind!r}"
            job.finished_at = _utcnow()
            try:
                await self._emit(job, "failed", job.to_public())
            except Exception as e:
                print(f"[jobs] emit failed for {job.id}: {e}")
            return

        cancel_event = self.cancel_events.get(job.id) or asyncio.Event()
        self.cancel_events[job.id] = cancel_event

        job.status = "running"
        job.started_at = _utcnow()
        job._start_monotonic = time.monotonic()
        try:
            await self._emit(job, "running", job.to_public())
        except Exception as e:
            print(f"[jobs] emit failed for {job.id}: {e}")

        async def emit(event: str, data: Any) -> None:
            if event == "progress" and isinstance(data, dict):
                job.progress = {
                    "index": data.get("index"),
                    "total": data.get("total"),
                    "image": data.get("image"),
                    "phase": "running",
                }
            elif event == "status" and isinstance(data, dict):
                job.progress = {**job.progress, "phase": data.get("phase", "running"), "total": data.get("total")}
            try:
                await self._emit(job, event, data)
            except Exception as e:
                print(f"[jobs] emit failed for {job.id}: {e}")

        try:
            await runner(job, emit, cancel_event)
            if cancel_event.is_set():
                job.status = "cancelled"
            else:
                job.status = "done"
        except Exception as e:  # noqa: BLE001
            job.status = "failed"
            job.error = str(e)
            print(f"[jobs] runner {job.kind!r} for {job.id} failed: {e}")
        finally:
            job.finished_at = _utcnow()
            if job._start_monotonic is not None:
                job.elapsed_s = time.monotonic() - job._start_monotonic
            job.cost_pence = elapsed_to_cost(job.elapsed_s)
            try:
                await self._emit(job, job.status, job.to_public())
            except Exception as e:
                print(f"[jobs] terminal emit failed for {job.id}: {e}")
            try:
                self._fire_on_finish(job)
            except Exception as e:
                print(f"[jobs] on_finish failed for {job.id}: {e}")

    def _gc_history(self) -> None:
        if len(self.jobs) <= self.history_cap:
            return
        # Drop oldest finished jobs first.
        finished = sorted(
            (j for j in self.jobs.values() if j.status in ("done", "failed", "cancelled")),
            key=lambda j: j.finished_at or "",
        )
        excess = len(self.jobs) - self.history_cap
        for j in finished[:excess]:
            self.jobs.pop(j.id, None)
            self.listeners.pop(j.id, None)
            self.cancel_events.pop(j.id, None)
