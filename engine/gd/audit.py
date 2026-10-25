"""SQLite event log in the workspace: job history and project activity."""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone

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


# The event kinds the engine writes; an events.db can hold others.
ACTIVITY_KINDS = ("job", "container_create", "container_delete", "dataset_add", "dataset_remove",
                  "project_duplicate")


def list_activity(container_id: str, dataset_ids: list[str], limit: int = 100) -> list[dict]:
    """Recent events of a container or of its `dataset_ids`, newest first."""
    ids = {d for d in (dataset_ids or []) if d}
    cur = _get_conn().cursor()
    cur.execute("SELECT ts, kind, data FROM events ORDER BY ts DESC LIMIT 3000")
    out: list[dict] = []
    for ts, kind, blob in cur.fetchall():
        if kind not in ACTIVITY_KINDS:
            continue
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
            "container": c,
            "dataset": p,
            "status": d.get("status"),
            "n_images": d.get("n_images"),
            "name": d.get("name"),
        })
        if len(out) >= limit:
            break
    return out

