"""Filesystem dataset store.

Layout under <workspace>/projects/:

    <project-slug>/                 # a project
        project.json
        <dataset-slug>/             # a dataset
            dataset.json            # metadata, labels, per-image index
            images/                 # original images
            annotations/            # one JSON per image
            references/ augmentations/ thumbs/
    <dataset-slug>/                 # a dataset in no project, same layout

load() joins dataset.json and annotations/ into one manifest; save() splits it
and writes only changed annotation files. Folder names are slugs; the "id"
field is the identity.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import time
import unicodedata
import uuid
from pathlib import Path

try:
    import orjson

    def _dumps(obj) -> bytes:
        return orjson.dumps(obj, option=orjson.OPT_SORT_KEYS | orjson.OPT_INDENT_2)

    def _loads(data: bytes):
        return orjson.loads(data)
except Exception:  # pragma: no cover
    def _dumps(obj) -> bytes:
        return json.dumps(obj, sort_keys=True, indent=2).encode("utf-8")

    def _loads(data: bytes):
        return json.loads(data)

import workspace

# Import-record keys stored in annotations/<id>.json; the rest stay in dataset.json.
ANNOTATION_KEYS = ("detections", "editedBoxes", "timings")

_LOCK = threading.RLock()
_SCANNED = False
_DATASETS: dict[str, Path] = {}
_CONTAINERS: dict[str, Path] = {}
# pid -> {import_id -> digest of last-written annotation payload}
_ANN_DIGESTS: dict[str, dict[str, bytes]] = {}
# Bumped as each save() starts and ends; a load() keeps the digests it read
# only when no save overlapped it.
_SAVE_GEN: dict[str, int] = {}


# Slugs

def slugify(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name or "")).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()
    return s[:60] or "untitled"


def _unique_child(parent: Path, slug: str) -> Path:
    cand = parent / slug
    n = 2
    while cand.exists():
        cand = parent / f"{slug}-{n}"
        n += 1
    return cand


def _safe_id(raw: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(raw))[:80]


def tmp_path(path: Path) -> Path:
    """A temp file beside `path`, unique to this write."""
    return path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")


def replace(src: Path, dst: Path, attempts: int = 20) -> None:
    """os.replace, retried on Windows sharing violations."""
    for i in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == attempts - 1:
                Path(src).unlink(missing_ok=True)
                raise
            time.sleep(0.01 * (i + 1))


def read_bytes(path: Path, attempts: int = 20) -> bytes:
    """Path.read_bytes, retried on Windows sharing violations: a concurrent
    replace() of the same file briefly refuses readers."""
    for i in range(attempts):
        try:
            return path.read_bytes()
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(0.01 * (i + 1))
    raise AssertionError("unreachable")


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = tmp_path(path)
    tmp.write_bytes(data)
    replace(tmp, path)


# Scanning

def _read_id(path: Path) -> str | None:
    try:
        return str(_loads(read_bytes(path)).get("id") or "") or None
    except Exception:
        return None


def _scan_locked() -> None:
    global _SCANNED
    _DATASETS.clear()
    _CONTAINERS.clear()
    root = workspace.projects_dir()
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        cj = entry / "project.json"
        dj = entry / "dataset.json"
        if cj.exists():
            cid = _read_id(cj)
            if cid:
                _CONTAINERS[cid] = entry
            for sub in sorted(entry.iterdir()):
                sdj = sub / "dataset.json"
                if sub.is_dir() and sdj.exists():
                    pid = _read_id(sdj)
                    if pid:
                        _DATASETS[pid] = sub
        elif dj.exists():
            pid = _read_id(dj)
            if pid:
                _DATASETS[pid] = entry
    _SCANNED = True


def _ensure_scanned() -> None:
    with _LOCK:
        if not _SCANNED:
            _scan_locked()


def rescan() -> None:
    with _LOCK:
        _scan_locked()


def forget(pid: str) -> None:
    """Drop a dataset's annotation digests. Call it after replacing its
    annotation files on disk, or save() skips rewriting them."""
    with _LOCK:
        _ANN_DIGESTS.pop(pid, None)
        _SAVE_GEN[pid] = _SAVE_GEN.get(pid, 0) + 1


# Datasets

def dataset_exists(pid: str) -> bool:
    _ensure_scanned()
    with _LOCK:
        return pid in _DATASETS


def dataset_dir(pid: str) -> Path:
    _ensure_scanned()
    with _LOCK:
        try:
            return _DATASETS[pid]
        except KeyError:
            raise KeyError(f"unknown dataset id: {pid!r}") from None


def iter_dataset_ids() -> list[str]:
    _ensure_scanned()
    with _LOCK:
        return list(_DATASETS.keys())


def create_dataset_dir(pid: str, name: str, container_id: str | None = None) -> Path:
    _ensure_scanned()
    with _LOCK:
        if pid in _DATASETS:
            return _DATASETS[pid]
        parent = _CONTAINERS.get(container_id or "") or workspace.projects_dir()
        d = _unique_child(parent, slugify(name))
        (d / "images").mkdir(parents=True, exist_ok=True)
        (d / "annotations").mkdir(exist_ok=True)
        _DATASETS[pid] = d
        return d


def reserve_dataset_dir(pid: str, name: str, container_id: str | None = None) -> Path:
    """Register a dataset folder path without creating it, for shutil.copytree."""
    _ensure_scanned()
    with _LOCK:
        if pid in _DATASETS:
            return _DATASETS[pid]
        parent = _CONTAINERS.get(container_id or "") or workspace.projects_dir()
        d = _unique_child(parent, slugify(name))
        _DATASETS[pid] = d
        return d


def delete_dataset(pid: str) -> None:
    _ensure_scanned()
    with _LOCK:
        d = _DATASETS.pop(pid, None)
        _ANN_DIGESTS.pop(pid, None)
    if d and d.exists():
        shutil.rmtree(d, ignore_errors=True)


def _relocate(path: Path, parent: Path) -> Path:
    """Rename a dataset folder under `parent`, or leave it and return its path.
    Not shutil.move: its copy fallback can delete part of the source and fail."""
    target = _unique_child(parent, path.name)
    try:
        os.rename(path, target)
    except OSError as e:
        print(f"[store] {path.name} stays where it is (could not move it: {e})")
        return path
    return target


def move_dataset(pid: str, container_id: str | None) -> Path:
    """Move a dataset folder into or out of a project folder.
    Membership is in the JSON, so a failed move leaves the folder where it is."""
    _ensure_scanned()
    with _LOCK:
        d = dataset_dir(pid)
        parent = _CONTAINERS.get(container_id or "") or workspace.projects_dir()
        if d.parent == parent:
            return d
        _DATASETS[pid] = _relocate(d, parent)
        return _DATASETS[pid]


# Manifests

def _annotations_dir(d: Path) -> Path:
    return d / "annotations"


def manifest_stamp(pid: str) -> float:
    """dataset.json mtime, which every save() changes."""
    try:
        return (dataset_dir(pid) / "dataset.json").stat().st_mtime
    except (KeyError, OSError):
        return 0.0


def load(pid: str) -> dict | None:
    try:
        d = dataset_dir(pid)
    except KeyError:
        return None
    with _LOCK:
        gen = _SAVE_GEN.get(pid, 0)
    dj = d / "dataset.json"
    try:
        raw = read_bytes(dj)
    except FileNotFoundError:
        return None
    try:
        manifest = _loads(raw)
    except Exception:
        return None
    ann_dir = _annotations_dir(d)
    digests: dict[str, bytes] = {}
    imports = manifest.get("imports")
    if isinstance(imports, list):
        for imp in imports:
            if not isinstance(imp, dict):
                continue
            iid = str(imp.get("id") or "")
            if not iid:
                continue
            path = ann_dir / f"{_safe_id(iid)}.json"
            # Any other read error propagates: an image loaded without its
            # annotations could be saved back without them.
            try:
                raw = read_bytes(path)
            except FileNotFoundError:
                continue
            try:
                ann = _loads(raw)
            except Exception:
                continue
            for key in ANNOTATION_KEYS:
                if key in ann:
                    imp[key] = ann[key]
            digests[iid] = hashlib.blake2b(_dumps(ann), digest_size=16).digest()
    with _LOCK:
        if _SAVE_GEN.get(pid, 0) == gen:
            _ANN_DIGESTS[pid] = digests
    # Every dataset uses the V2 view; the flag reaches disk at the next save.
    manifest["v2"] = True
    return manifest


def save(pid: str, manifest: dict) -> None:
    d = dataset_dir(pid)  # KeyError for an unknown id
    ann_dir = _annotations_dir(d)
    with _LOCK:
        _SAVE_GEN[pid] = _SAVE_GEN.get(pid, 0) + 1
        digests = _ANN_DIGESTS.setdefault(pid, {})
    try:
        _save_files(d, ann_dir, manifest, digests)
    finally:
        with _LOCK:
            _SAVE_GEN[pid] = _SAVE_GEN.get(pid, 0) + 1


def _save_files(d: Path, ann_dir: Path, manifest: dict, digests: dict[str, bytes]) -> None:
    slim = dict(manifest)
    imports = manifest.get("imports")
    live_files: set[str] = set()
    if isinstance(imports, list):
        slim_imports = []
        for imp in imports:
            if not isinstance(imp, dict):
                slim_imports.append(imp)
                continue
            iid = str(imp.get("id") or "")
            ann = {k: imp[k] for k in ANNOTATION_KEYS if k in imp}
            slim_imports.append({k: v for k, v in imp.items() if k not in ANNOTATION_KEYS})
            if not iid:
                continue
            live_files.add(f"{_safe_id(iid)}.json")
            if ann:
                payload = _dumps(ann)
                digest = hashlib.blake2b(payload, digest_size=16).digest()
                if digests.get(iid) != digest:
                    _atomic_write(ann_dir / f"{_safe_id(iid)}.json", payload)
                    digests[iid] = digest
            elif iid in digests or (ann_dir / f"{_safe_id(iid)}.json").exists():
                (ann_dir / f"{_safe_id(iid)}.json").unlink(missing_ok=True)
                digests.pop(iid, None)
        slim["imports"] = slim_imports
        if ann_dir.is_dir():
            for f in ann_dir.iterdir():
                if f.suffix == ".json" and f.name not in live_files:
                    f.unlink(missing_ok=True)

    _atomic_write(d / "dataset.json", _dumps(slim))


# Containers

def container_exists(cid: str) -> bool:
    _ensure_scanned()
    with _LOCK:
        return cid in _CONTAINERS


def container_dir(cid: str) -> Path:
    _ensure_scanned()
    with _LOCK:
        try:
            return _CONTAINERS[cid]
        except KeyError:
            raise KeyError(f"unknown container id: {cid!r}") from None


def container_path(cid: str) -> Path:
    return container_dir(cid) / "project.json"


def iter_container_ids() -> list[str]:
    _ensure_scanned()
    with _LOCK:
        return list(_CONTAINERS.keys())


def create_container_dir(cid: str, name: str) -> Path:
    _ensure_scanned()
    with _LOCK:
        if cid in _CONTAINERS:
            return _CONTAINERS[cid]
        d = _unique_child(workspace.projects_dir(), slugify(name))
        d.mkdir(parents=True, exist_ok=True)
        _CONTAINERS[cid] = d
        return d


def delete_container_dir(cid: str) -> None:
    """Delete a container folder after moving its datasets out to the top level."""
    _ensure_scanned()
    with _LOCK:
        d = _CONTAINERS.get(cid)
        if not d or not d.exists():
            _CONTAINERS.pop(cid, None)
            return
        stuck = []
        for pid, path in list(_DATASETS.items()):
            if path.parent == d:
                _DATASETS[pid] = _relocate(path, workspace.projects_dir())
                if _DATASETS[pid] == path:
                    stuck.append(path.name)
        if stuck:
            raise OSError(f"could not move {', '.join(stuck)} out of the project folder "
                          "(a file inside is in use)")
        _CONTAINERS.pop(cid, None)
    shutil.rmtree(d, ignore_errors=True)


def new_id() -> str:
    return uuid.uuid4().hex[:12]
