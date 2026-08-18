"""Local SQLite event log - job history + project activity.

The portable build keeps this for two things only: rehydrating finished-job
history across restarts (JobManager.hydrate_from_audit) and the per-project
activity feed. The SaaS billing/demo/telemetry rollups are gone.

The DB lives in the workspace (events.db). SQLite is stdlib - no extra
dependency.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

import workspace

DB_PATH = workspace.dir() / "events.db"

_conn: sqlite3.Connection | None = None
_lock = threading.Lock()


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute(
            """CREATE TABLE IF NOT EXISTS events (
                id   INTEGER PRIMARY KEY AUTOINCREMENT,
                ts   TEXT NOT NULL,
                kind TEXT NOT NULL,
                data TEXT NOT NULL
            )"""
        )
        conn.execute("CREATE INDEX IF NOT EXISTS events_kind_ts ON events (kind, ts DESC)")
        _conn = conn
    return _conn


def add_event(kind: str, **data) -> None:
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    blob = json.dumps(data, default=str)
    with _lock:
        _get_conn().execute(
            "INSERT INTO events (ts, kind, data) VALUES (?, ?, ?)",
            (ts, kind, blob),
        )


def list_events(kind: str | None = None, limit: int = 200) -> list[dict]:
    cur = _get_conn().cursor()
    if kind:
        cur.execute(
            "SELECT ts, kind, data FROM events WHERE kind = ? ORDER BY ts DESC LIMIT ?",
            (kind, limit),
        )
    else:
        cur.execute(
            "SELECT ts, kind, data FROM events ORDER BY ts DESC LIMIT ?",
            (limit,),
        )
    out: list[dict] = []
    for ts, k, blob in cur.fetchall():
        try:
            payload = json.loads(blob)
        except Exception:
            payload = {}
        out.append({"ts": ts, "kind": k, **payload})
    return out


def list_activity(container_id: str, dataset_ids: list[str], limit: int = 100) -> list[dict]:
    """Project activity feed: recent events that belong to a container -- either
    a container-level event (data.container == container_id) or an event on one
    of its datasets (data.project / data.dataset in dataset_ids). Normalised so
    the FE renders a uniform timeline. The ACTOR is the originator (data.actor,
    falling back to data.user/username) -- never the container owner."""
    ids = {d for d in (dataset_ids or []) if d}
    cur = _get_conn().cursor()
    # Scan a recent window and filter in Python (same approach as the usage
    # rollups). Cheap enough at this scale; index by container later if needed.
    cur.execute("SELECT ts, kind, data FROM events ORDER BY ts DESC LIMIT 3000")
    out: list[dict] = []
    for ts, kind, blob in cur.fetchall():
        try:
            d = json.loads(blob)
        except Exception:
            continue
        c = d.get("container")
        p = d.get("project") or d.get("dataset")
        if not ((container_id and c == container_id) or (p and p in ids)):
            continue
        out.append({
            "ts": ts,
            "kind": kind,
            "job_kind": d.get("job_kind"),
            "actor": d.get("actor") or d.get("user") or d.get("username"),
            "container": c,
            "dataset": p,
            "status": d.get("status"),
            "n_images": d.get("n_images"),
            "member": d.get("member"),
            "role": d.get("role"),
            "name": d.get("name"),
        })
        if len(out) >= limit:
            break
    return out

