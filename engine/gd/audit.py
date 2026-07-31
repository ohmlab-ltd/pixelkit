"""Local SQLite audit log for the Terminal page.

Stores three event kinds across restarts so the operator can see what's
happened on this backend without hitting the auth Postgres:

    nsfw_block  — user, project, file, score, classification
    signup      — username, name, image, email, provider
    job         — id, kind, project, user, status, elapsed_s, cost_pence

The DB lives at backend/audit.db (sibling of projects/) and is gitignored.
SQLite is in stdlib so there's no extra dependency.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path


DB_PATH = Path(__file__).resolve().parent.parent / "audit.db"

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


def count_events(kind: str) -> int:
    cur = _get_conn().cursor()
    cur.execute("SELECT COUNT(*) FROM events WHERE kind = ?", (kind,))
    return int(cur.fetchone()[0])


def rename_user_in_events(old: str, new: str) -> int:
    """Rewrite every audit row that mentions `old` to use `new` instead.

    Without this the per-user usage rollups (sum_labelled_images_for_user
    and the Terminal feed) keep counting against the previous handle,
    which lets a user reset their monthly quota for free by renaming.
    Touches `data.user` and `data.username` in-place; both are used
    across job / signup / demo events."""
    if not old or not new or old == new:
        return 0
    cur = _get_conn().cursor()
    # Cheap LIKE pre-filter so we only parse rows that could plausibly
    # match. JSON encoding always wraps usernames in quotes, so a
    # quoted-substring match is precise enough.
    pattern = f'%"{old}"%'
    cur.execute("SELECT id, data FROM events WHERE data LIKE ?", (pattern,))
    rows = cur.fetchall()
    updated = 0
    with _lock:
        for row_id, blob in rows:
            try:
                d = json.loads(blob)
            except Exception:
                continue
            changed = False
            if d.get("user") == old:
                d["user"] = new
                changed = True
            if d.get("username") == old:
                d["username"] = new
                changed = True
            if changed:
                _get_conn().execute(
                    "UPDATE events SET data = ? WHERE id = ?",
                    (json.dumps(d, default=str), row_id),
                )
                updated += 1
    return updated


def sum_labelled_images_for_user(user: str, *, since_iso: str) -> int:
    """Total auto-labelled images recorded for `user` since `since_iso`.

    Rolled up from the `job` audit rows: only completed `label` /
    `label_lite` / `label_charlie` jobs with `status == "done"` count,
    so cancelled or failed runs don't burn through the quota. The
    cutoff is supplied by the Next.js side (which knows whether to
    anchor on the user's Stripe billing period or their signup-day-of-
    month) so this layer stays plan-agnostic.

    The `label_charlie` kind is V2's labelling pipeline; older
    workflows ran under `label` / `label_lite`. Missing the charlie
    kind here is what made the profile page show a permanent zero for
    auto-labelled-images on V2 accounts."""
    return _sum_job_images_for_user(
        user,
        since_iso=since_iso,
        kinds=("label", "label_lite", "label_charlie"),
    )


def sum_uploaded_images_for_user(user: str, *, since_iso: str) -> int:
    """Total images uploaded into a project for `user` since `since_iso`.

    Mirrors sum_labelled_images_for_user but reads the `upload` job
    kind that the import flow writes once a batch finishes. The
    counter is needed by the profile usage panel and the credit
    cost calculation (uploaded / 800 credits per image)."""
    return _sum_job_images_for_user(user, since_iso=since_iso, kinds=("upload",))


def _sum_job_images_for_user(
    user: str,
    *,
    since_iso: str,
    kinds: tuple[str, ...],
) -> int:
    """Shared rollup that filters job-audit rows by user, kind, and
    success status, then sums `n_images`. Extracted so the labelled +
    uploaded helpers stay readable and consistent."""
    if not user or not since_iso or not kinds:
        return 0
    cur = _get_conn().cursor()
    # ISO 8601 strings sort lexicographically, so a >= comparison on
    # the ts column is a correct chronological filter.
    cur.execute("SELECT data FROM events WHERE kind = 'job' AND ts >= ?", (since_iso,))
    kinds_set = set(kinds)
    total = 0
    for (blob,) in cur.fetchall():
        try:
            d = json.loads(blob)
        except Exception:
            continue
        if d.get("user") != user:
            continue
        if d.get("job_kind") not in kinds_set:
            continue
        if d.get("status") != "done":
            continue
        try:
            total += int(d.get("n_images") or 0)
        except (TypeError, ValueError):
            continue
    return total


def count_demo_for_ip(ip: str) -> int:
    """Count successful demo runs from a given IP TODAY (UTC). Only
    `status == "ok"` rows burn against the per-IP daily quota — NSFW
    rejections don't punish the user. The count resets at UTC midnight,
    so a visitor gets a fresh allowance each day.

    `ts` is a fixed-width ISO-8601 UTC string (see add_event), so a
    lexicographic `>=` against today's midnight is a correct date filter.
    """
    day_start = (
        datetime.now(timezone.utc)
        .replace(hour=0, minute=0, second=0, microsecond=0)
        .isoformat(timespec="seconds")
    )
    cur = _get_conn().cursor()
    cur.execute(
        "SELECT data FROM events WHERE kind = 'demo' AND ts >= ?",
        (day_start,),
    )
    n = 0
    for (blob,) in cur.fetchall():
        try:
            d = json.loads(blob)
        except Exception:
            continue
        # Reruns (run_token re-use) are logged for Terminal visibility but
        # must NOT count toward the per-IP quota — they're free re-runs on
        # new images. Exclude them here so logging a rerun never burns quota.
        if d.get("ip") == ip and d.get("status") == "ok" and not d.get("rerun"):
            n += 1
    return n


def sum_training_blocks_for_user(user: str, *, since_iso: str) -> int:
    """Total billed training blocks (15-min credits) for `user` since
    `since_iso`, summed from `ml_billing` rows. Lets training time draw
    from the SAME monthly credit pool as labelling/uploads: the credit
    gate adds this to its usage total. `ts` is fixed-width ISO-8601 UTC
    so a lexicographic `>=` is a correct date filter (same trick as
    sum_labelled_images_for_user)."""
    if not user or not since_iso:
        return 0
    cur = _get_conn().cursor()
    cur.execute("SELECT data FROM events WHERE kind = 'ml_billing' AND ts >= ?", (since_iso,))
    total = 0
    for (blob,) in cur.fetchall():
        try:
            d = json.loads(blob)
        except Exception:
            continue
        if d.get("user") != user and d.get("username") != user:
            continue
        try:
            total += int(d.get("blocks") or 0)
        except (TypeError, ValueError):
            continue
    return total


def sum_training_blocks_for_job(job_id: str) -> int:
    """Total billed blocks recorded for a single job. Used by
    MLJobStore to reconcile `charged_blocks` after a crash (the ledger is
    the durable source of truth). A cheap LIKE pre-filter keeps us from
    parsing every billing row."""
    if not job_id:
        return 0
    cur = _get_conn().cursor()
    cur.execute(
        "SELECT data FROM events WHERE kind = 'ml_billing' AND data LIKE ?",
        (f'%"job_id": "{job_id}"%',),
    )
    total = 0
    for (blob,) in cur.fetchall():
        try:
            d = json.loads(blob)
        except Exception:
            continue
        if d.get("job_id") != job_id:
            continue
        try:
            total += int(d.get("blocks") or 0)
        except (TypeError, ValueError):
            continue
    return total
