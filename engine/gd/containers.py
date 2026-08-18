"""Projects (containers of datasets) - portable build.

A "Project" (user-facing term) is a CONTAINER that holds many datasets. It is
distinct from the legacy code-level "project", which is actually a single
DATASET. The SaaS build gave containers members/roles/privacy; the portable
build has one local user, so every access check passes. The role/member
function shapes are kept so server.py routes and stored JSON stay compatible.

Storage: <workspace>/projects/<project-slug>/project.json (folder resolution
lives in store.py - folder names are cosmetic slugs, identity is the JSON id).
"""
from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import store

ROLE_OWNER = "owner"
ROLE_EDITOR = "editor"
ROLE_VIEWER = "viewer"
ROLES = (ROLE_OWNER, ROLE_EDITOR, ROLE_VIEWER)

# Max longest-edge (px) uploaded images are resized to. A per-Project setting:
# the FE caps each upload to this on the way out. Datasets inherit the value
# from their Project.
MAX_INPUT_DEFAULT = 1500
MAX_INPUT_MIN = 512
MAX_INPUT_MAX = 4096


def clamp_max_input(value) -> int:
    try:
        v = int(value)
    except (TypeError, ValueError):
        return MAX_INPUT_DEFAULT
    return max(MAX_INPUT_MIN, min(MAX_INPUT_MAX, v))


_lock = threading.Lock()


def set_persistence_hooks(*, on_save=None, on_delete=None) -> None:
    """Cloud-mirror hooks from the SaaS build. Kept as a no-op so any stale
    caller is harmless; local disk is the only storage now."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _norm(u: str | None) -> str:
    return (u or "").strip().lower()


def container_dir(container_id: str) -> Path:
    return store.container_dir(container_id)


# ── persistence ───────────────────────────────────────────────────────────────

def load_container(container_id: str) -> dict | None:
    if not container_id:
        return None
    try:
        p = store.container_path(container_id)
    except KeyError:
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_container(container: dict, *, bump_updated: bool = True) -> None:
    cid = container.get("id")
    if not cid:
        raise ValueError("container has no id")
    if bump_updated:
        container["updated"] = _now_iso()
    d = store.create_container_dir(cid, container.get("name") or "untitled")
    p = d / "project.json"
    tmp = p.with_suffix(".json.tmp")
    with _lock:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(container, f, indent=2)
        os.replace(tmp, p)


def create_container(
    name: str,
    owner: str,
    *,
    private: bool = True,
    cover: str | None = None,
) -> dict:
    cid = uuid.uuid4().hex
    now = _now_iso()
    o = _norm(owner) or "local"
    c = {
        "id": cid,
        "name": (name or "").strip()[:120] or "Untitled project",
        "owner": o,
        "members": [{"username": o, "role": ROLE_OWNER}],
        "private": bool(private),
        "cover": cover,
        "max_input_size": MAX_INPUT_DEFAULT,
        "dataset_ids": [],
        "created": now,
        "updated": now,
    }
    save_container(c)
    return c


def delete_container(container_id: str) -> None:
    store.delete_container_dir(container_id)


def list_containers_for_user(username: str) -> list[dict]:
    """Single-user build: every container, regardless of the name passed."""
    out: list[dict] = []
    for cid in store.iter_container_ids():
        c = load_container(cid)
        if c:
            out.append(c)
    return out


def list_public_containers() -> list[dict]:
    """No public/community feed in the portable build."""
    return []


# ── membership / roles (single-user: local is owner of everything) ───────────

def member_role(container: dict | None, username: str) -> str | None:
    return ROLE_OWNER if container else None


def role_at_least(role: str | None, minimum: str) -> bool:
    return role is not None


def set_member(container: dict, username: str, role: str) -> dict:
    return container


def remove_member(container: dict, username: str) -> dict:
    return container


# ── access decisions ──────────────────────────────────────────────────────────

def can_read(container: dict | None, username: str | None) -> bool:
    return container is not None


def can_write(container: dict | None, username: str | None) -> bool:
    return container is not None


def can_manage(container: dict | None, username: str | None) -> bool:
    return container is not None


# ── dataset <-> container wiring ──────────────────────────────────────────────

def dataset_container_id(manifest: dict | None) -> str:
    return _norm((manifest or {}).get("container_id")) if manifest else ""


def dataset_container_index() -> dict[str, dict]:
    """Reverse map dataset_id -> {id, name, private} from every container's
    dataset_ids, for workspace cards."""
    index: dict[str, dict] = {}
    for cid in store.iter_container_ids():
        c = load_container(cid)
        if not c:
            continue
        info = {
            "id": c.get("id") or cid,
            "name": c.get("name") or "",
            "private": bool(c.get("private")),
        }
        for ds_id in c.get("dataset_ids") or []:
            if ds_id:
                index[str(ds_id)] = info
    return index


def dataset_is_private(manifest: dict | None) -> bool:
    return False


def dataset_access(manifest: dict | None, username: str | None) -> dict:
    """Single local user: full access to any dataset that exists."""
    if not manifest:
        return {"private": True, "readable": False, "writable": False, "manageable": False}
    return {"private": False, "readable": True, "writable": True, "manageable": True}
