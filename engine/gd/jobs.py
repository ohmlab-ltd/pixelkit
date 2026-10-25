"""In-memory job queue with N concurrent async workers."""
from __future__ import annotations

import asyncio
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

JobStatus = str  # "queued" | "running" | "done" | "failed" | "cancelled" | "interrupted"
JobKind = str


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Job:
    id: str
    kind: JobKind
    project: str
    params: dict
    status: JobStatus = "queued"
    queued_at: str = field(default_factory=_utcnow)
    started_at: str | None = None
    finished_at: str | None = None
    progress: dict = field(default_factory=dict)  # {index, total, image, phase}
    error: str | None = None
    elapsed_s: float = 0.0
    n_images: int = 0
    # The runner's return value when it is a dict; the CLI reads it from GET /api/jobs/{id}.
    result: dict | None = None

    _start_monotonic: float | None = None

    def to_public(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "project": self.project,
            "params": self.params,
            "status": self.status,
            "queuedAt": self.queued_at,
            "startedAt": self.started_at,
            "finishedAt": self.finished_at,
            "progress": self.progress,
            "error": self.error,
            "elapsedS": round(self.live_elapsed_s(), 1),
            "n_images": self.n_images,
            "result": self.result,
        }

    def live_elapsed_s(self) -> float:
        if self.status == "running" and self._start_monotonic is not None:
            return max(0.0, time.monotonic() - self._start_monotonic)
        return self.elapsed_s


# Called as runner(job, emit, cancel_event); emit("progress" or "status", data) sets job.progress.
JobRunner = Callable[["Job", Callable[[str, Any], Awaitable[None]], asyncio.Event], Awaitable[None]]


class JobManager:
    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.cancel_events: dict[str, asyncio.Event] = {}
        # CUDA access is arbitrated by the GPU gate, state["gpu_lock"], not by the workers.
        self.worker_tasks: list[asyncio.Task] = []
        self.runners: dict[str, JobRunner] = {}
        self.history_cap = 500
        # Called once per finished job; the server writes it to the audit log.
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
        """Ensure `n` live worker tasks, replacing any that died."""
        self.worker_tasks = [t for t in self.worker_tasks if not t.done()]
        while len(self.worker_tasks) < n:
            self.worker_tasks.append(asyncio.create_task(self._worker()))

    def schedule(self, kind: str, project: str, params: dict, n_images: int = 0) -> Job:
        job = Job(
            id=secrets.token_hex(6),
            kind=kind,
            project=project,
            params=params,
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
        """Load jobs from audit.list_events `kind="job"` rows; returns the count loaded."""
        loaded = 0
        for e in events:
            jid = e.get("id")
            if not jid or jid in self.jobs:
                continue
            status = e.get("status") or "done"
            # Not finished when the previous process exited.
            if status not in ("done", "failed", "cancelled", "interrupted"):
                status = "interrupted"
            # The job's kind is in job_kind; `kind` is the audit row's own column.
            job_kind = e.get("job_kind") or e.get("kind") or "label"
            if job_kind == "job":
                job_kind = "label"
            self.jobs[jid] = Job(
                id=jid,
                kind=job_kind,
                project=e.get("project") or "",
                params={},
                status=status,
                queued_at=e.get("ts") or _utcnow(),
                started_at=e.get("ts"),
                finished_at=e.get("ts"),
                error=e.get("error"),
                elapsed_s=float(e.get("elapsed_s") or 0.0),
                n_images=int(e.get("n_images") or 0),
            )
            loaded += 1
        return loaded

    async def _worker(self) -> None:
        while True:
            try:
                await self._process_one()
            except asyncio.CancelledError:
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
            return

        cancel_event = self.cancel_events.get(job.id) or asyncio.Event()
        self.cancel_events[job.id] = cancel_event

        job.status = "running"
        job.started_at = _utcnow()
        job._start_monotonic = time.monotonic()

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
            out = await runner(job, emit, cancel_event)
            job.result = out if isinstance(out, dict) else None
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
            self._fire_on_finish(job)

    def _gc_history(self) -> None:
        if len(self.jobs) <= self.history_cap:
            return
        finished = sorted(
            (j for j in self.jobs.values() if j.status in ("done", "failed", "cancelled")),
            key=lambda j: j.finished_at or "",
        )
        excess = len(self.jobs) - self.history_cap
        for j in finished[:excess]:
            self.jobs.pop(j.id, None)
            self.cancel_events.pop(j.id, None)
