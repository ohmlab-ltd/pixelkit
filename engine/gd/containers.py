"""Project containers (teams).

A "Project" (user-facing term) is a CONTAINER that holds many datasets and has
members with roles. It is distinct from the legacy code-level "project", which
is actually a single DATASET (projects/<id>/manifest.json). To avoid a giant
rename we keep dataset ids/manifests as-is and add this separate container
entity stored at containers/<id>/project.json.

Access model:
  - roles: owner > editor > viewer
      viewer  -> read (view datasets, images, exports)
      editor  -> read + write (upload, label, edit boxes, train)
      owner   -> manage (rename, cover, privacy, add/remove members, delete)
  - privacy inherits: a dataset placed in a container takes the container's
    privacy; a private container's datasets are private; changing container
    privacy cascades to every dataset in it.
  - a private container is readable only by its members; a public container is
    readable by anyone (guests included).

Self-contained (stdlib only) so auth.py can import it without a circular
dependency (containers.py never imports server.py / auth.py).
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONTAINERS_DIR = ROOT / "containers"
CONTAINERS_DIR.mkdir(parents=True, exist_ok=True)

ROLE_OWNER = "owner"
ROLE_EDITOR = "editor"
ROLE_VIEWER = "viewer"
ROLES = (ROLE_OWNER, ROLE_EDITOR, ROLE_VIEWER)
_RANK = {ROLE_VIEWER: 0, ROLE_EDITOR: 1, ROLE_OWNER: 2}

# Max longest-edge (px) uploaded images are resized to. A per-Project setting:
# the FE caps each upload to this on the way out. Default 1500 matches the
# historical client default exactly, so projects that never touch the setting
# behave identically. Adjustable up to 4K for projects that want crisp originals
# (at higher storage cost). Datasets inherit the value from their Project.
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

# Optional durability hooks injected by the host app (server.py) so every
# container write/delete is mirrored to cloud storage (R2). Keeps this module
# stdlib-only (auth.py imports it) — the host wires R2 in via set_persistence_hooks
# on startup. When unset (e.g. local dev, or R2 unconfigured) containers are
# local-disk-only, exactly as before.
_save_hook = None    # Callable[[dict], None]
_delete_hook = None  # Callable[[str], None]


def set_persistence_hooks(*, on_save=None, on_delete=None) -> None:
    """Register cloud-mirror callbacks. on_save(container_dict) runs after every
    successful local write; on_delete(container_id) after every delete."""
    global _save_hook, _delete_hook
    _save_hook = on_save
    _delete_hook = on_delete


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _norm(u: str | None) -> str:
    return (u or "").strip().lower()


def container_dir(container_id: str) -> Path:
    return CONTAINERS_DIR / container_id


def _path(container_id: str) -> Path:
    return CONTAINERS_DIR / container_id / "project.json"


# ── persistence ───────────────────────────────────────────────────────────────

def load_container(container_id: str) -> dict | None:
    if not container_id:
        return None
    p = _path(container_id)
    if not p.exists():
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
    p = _path(cid)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    with _lock:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(container, f)
        os.replace(tmp, p)
    # Mirror to cloud storage so the container survives a disk loss the same way
    # dataset image bytes do. Best-effort: the local write already succeeded.
    if _save_hook is not None:
        try:
            _save_hook(container)
        except Exception:
            pass


def create_container(
    name: str,
    owner: str,
    *,
    private: bool = True,
    cover: str | None = None,
) -> dict:
    cid = uuid.uuid4().hex
    now = _now_iso()
    o = _norm(owner)
    c = {
        "id": cid,
        "name": (name or "").strip()[:120] or "Untitled project",
        "owner": o,
        "members": [{"username": o, "role": ROLE_OWNER}],
        "private": bool(private),
        "cover": cover,           # R2 key or filename of the uploaded cover
        "max_input_size": MAX_INPUT_DEFAULT,  # px longest edge for uploads
        "dataset_ids": [],
        "created": now,
        "updated": now,
    }
    save_container(c)
    return c


def delete_container(container_id: str) -> None:
    d = container_dir(container_id)
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
    # Remove the cloud mirror too so a deleted container doesn't get restored
    # from R2 on the next boot.
    if _delete_hook is not None:
        try:
            _delete_hook(container_id)
        except Exception:
            pass


def list_containers_for_user(username: str) -> list[dict]:
    """Every container the user is a member of (any role)."""
    u = _norm(username)
    out: list[dict] = []
    if not u or not CONTAINERS_DIR.exists():
        return out
    for d in CONTAINERS_DIR.iterdir():
        if not d.is_dir():
            continue
        c = load_container(d.name)
        if c and member_role(c, u) is not None:
            out.append(c)
    return out


def list_public_containers() -> list[dict]:
    """Every public (non-private) container, for the Community feed/carousel."""
    out: list[dict] = []
    if not CONTAINERS_DIR.exists():
        return out
    for d in CONTAINERS_DIR.iterdir():
        if not d.is_dir():
            continue
        c = load_container(d.name)
        if c and not bool(c.get("private")):
            out.append(c)
    return out


# ── membership / roles ────────────────────────────────────────────────────────

def member_role(container: dict | None, username: str) -> str | None:
    """The user's role in this container, or None if not a member."""
    u = _norm(username)
    if not u or not container:
        return None
    for m in container.get("members") or []:
        if _norm(m.get("username")) == u:
            r = m.get("role")
            return r if r in ROLES else ROLE_VIEWER
    return None


def role_at_least(role: str | None, minimum: str) -> bool:
    if role is None:
        return False
    return _RANK.get(role, -1) >= _RANK.get(minimum, 99)


def set_member(container: dict, username: str, role: str) -> dict:
    """Add or update a member's role. Owner row is never downgraded here -- use
    a transfer-ownership flow for that."""
    u = _norm(username)
    if not u:
        return container
    role = role if role in ROLES else ROLE_VIEWER
    members = [m for m in (container.get("members") or []) if _norm(m.get("username")) != u]
    members.append({"username": u, "role": role})
    container["members"] = members
    return container


def remove_member(container: dict, username: str) -> dict:
    """Remove a member. The owner cannot be removed (transfer ownership first)."""
    u = _norm(username)
    owner = _norm(container.get("owner"))
    if u == owner:
        return container
    container["members"] = [
        m for m in (container.get("members") or []) if _norm(m.get("username")) != u
    ]
    return container


# ── access decisions ──────────────────────────────────────────────────────────

def can_read(container: dict | None, username: str | None) -> bool:
    """Public container -> anyone; private -> members only."""
    if not container:
        return False
    if not bool(container.get("private")):
        return True
    return member_role(container, username or "") is not None


def can_write(container: dict | None, username: str | None) -> bool:
    """Editor or owner (upload, label, edit, train)."""
    return role_at_least(member_role(container or {}, username or ""), ROLE_EDITOR)


def can_manage(container: dict | None, username: str | None) -> bool:
    """Owner only (rename, cover, privacy, members, delete)."""
    if not container:
        return False
    if _norm(container.get("owner")) == _norm(username) and username:
        return True
    return member_role(container, username or "") == ROLE_OWNER


# ── dataset <-> container wiring ───────────────────────────────────────────────
# The dataset manifest carries `container_id`. These resolve a dataset's
# EFFECTIVE access by consulting its container when present. auth.py uses these
# so the existing project guards become membership-aware.

def dataset_container_id(manifest: dict | None) -> str:
    return _norm((manifest or {}).get("container_id")) if manifest else ""


def dataset_container_index() -> dict[str, dict]:
    """Reverse map dataset_id -> {id, name, private} built from every container's
    `dataset_ids`. Lets the workspace list attach a clickable Project chip to each
    dataset card without reading manifests. Cheap: one pass over containers/."""
    index: dict[str, dict] = {}
    if not CONTAINERS_DIR.exists():
        return index
    for d in CONTAINERS_DIR.iterdir():
        if not d.is_dir():
            continue
        c = load_container(d.name)
        if not c:
            continue
        info = {"id": c.get("id") or d.name, "name": c.get("name") or "", "private": bool(c.get("private"))}
        for ds_id in c.get("dataset_ids") or []:
            if ds_id:
                index[str(ds_id)] = info
    return index


def dataset_is_private(manifest: dict | None) -> bool:
    """Effective privacy: inherited from the container when the dataset is in
    one, else the dataset's own `private` flag."""
    cid = dataset_container_id(manifest)
    if cid:
        c = load_container(cid)
        if c is not None:
            return bool(c.get("private"))
    return bool((manifest or {}).get("private"))


def dataset_access(manifest: dict | None, username: str | None) -> dict:
    """Resolve {private, readable, writable, manageable} for a dataset given the
    caller. Container datasets use membership; standalone datasets use the
    manifest owner. `username` may be None (anonymous)."""
    if not manifest:
        return {"private": True, "readable": False, "writable": False, "manageable": False}
    u = _norm(username)
    cid = dataset_container_id(manifest)
    if cid:
        c = load_container(cid)
        if c is not None:
            return {
                "private": bool(c.get("private")),
                "readable": can_read(c, u),
                "writable": can_write(c, u),
                "manageable": can_manage(c, u),
            }
        # Dangling container reference: fail closed (treat as private, no access
        # except the dataset owner) rather than leak.
    private = bool(manifest.get("private"))
    owner = _norm(manifest.get("owner"))
    is_owner = bool(owner and u and owner == u)
    return {
        "private": private,
        "readable": (not private) or is_owner,
        "writable": is_owner,
        "manageable": is_owner,
    }
