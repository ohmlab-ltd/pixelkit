"""Projects, which contain datasets; in code a "project" is a dataset.

Stored as <workspace>/projects/<project-slug>/project.json.
"""
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone

import store

_lock = threading.Lock()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# Persistence

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
    tmp = store.tmp_path(p)
    with _lock:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(container, f, indent=2)
        store.replace(tmp, p)


def create_container(name: str) -> dict:
    cid = uuid.uuid4().hex
    now = _now_iso()
    c = {
        "id": cid,
        "name": (name or "").strip()[:120] or "Untitled project",
        "cover": None,
        "dataset_ids": [],
        "created": now,
        "updated": now,
    }
    save_container(c)
    return c


def delete_container(container_id: str) -> None:
    store.delete_container_dir(container_id)


def list_containers() -> list[dict]:
    out: list[dict] = []
    for cid in store.iter_container_ids():
        c = load_container(cid)
        if c:
            out.append(c)
    return out


# Datasets in projects

def dataset_container_index() -> dict[str, dict]:
    """dataset_id -> {id, name} of the container that lists it."""
    index: dict[str, dict] = {}
    for cid in store.iter_container_ids():
        c = load_container(cid)
        if not c:
            continue
        info = {"id": c.get("id") or cid, "name": c.get("name") or ""}
        for ds_id in c.get("dataset_ids") or []:
            if ds_id:
                index[str(ds_id)] = info
    return index
