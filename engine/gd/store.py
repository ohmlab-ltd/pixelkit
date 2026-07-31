"""Filesystem dataset store — the portable build's database.

On-disk layout (all under <workspace>/projects/):

    <project-slug>/                 # a Project (container of datasets)
        project.json                # container metadata
        <dataset-slug>/             # one folder per dataset
            dataset.json            # metadata + labels + per-image index
            images/                 # original images, nothing else
            annotations/            # one JSON per image: boxes/masks/labels
            outputs/ references/ augmentations/ thumbs/ ...
    <dataset-slug>/                 # dataset not (yet) in a project
        dataset.json
        ...

The SaaS build kept one giant manifest.json per dataset holding every box of
every image. Here that splits into:
  - dataset.json  — everything EXCEPT per-image geometry. Each record in
    manifest["imports"] keeps its metadata (filename, dims, blurhash,
    labelled flags, ...) but loses the heavy keys.
  - annotations/<import_id>.json — the geometry: {detections, editedBoxes,
    timings} exactly as they sat in the import record.

load() recomposes the exact in-memory manifest shape the rest of the engine
expects, so 20k lines of callers don't change. save() decomposes, writing
only annotation files whose content actually changed (digest cache) —
labelling jobs save after every image, so a full rewrite of N files per save
would be O(N^2) over a job.

Folder names are cosmetic slugs; identity lives in the JSON ("id" field).
An in-process index (id -> Path) is built by scanning at first use.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import unicodedata
import uuid
from pathlib import Path

try:
    import orjson

    def _dumps(obj) -> bytes:
        return orjson.dumps(obj, option=orjson.OPT_SORT_KEYS | orjson.OPT_INDENT_2)

    def _loads(data: bytes):
        return orjson.loads(data)
except Exception:  # pragma: no cover - orjson is a hard dep, but stay safe
    def _dumps(obj) -> bytes:
        return json.dumps(obj, sort_keys=True, indent=2).encode("utf-8")

    def _loads(data: bytes):
        return json.loads(data)

import workspace

# Per-import keys that move into annotations/<id>.json. Everything else in an
# import record is index metadata and stays in dataset.json.
ANNOTATION_KEYS = ("detections", "editedBoxes", "timings")

_LOCK = threading.RLock()
_SCANNED = False
_DATASETS: dict[str, Path] = {}
_CONTAINERS: dict[str, Path] = {}
# pid -> {import_id -> digest of last-written annotation payload}
_ANN_DIGESTS: dict[str, dict[str, bytes]] = {}


# ---------------------------------------------------------------- slugs

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


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, path)


# ---------------------------------------------------------------- scanning

def _read_id(path: Path) -> str | None:
    try:
        return str(_loads(path.read_bytes()).get("id") or "") or None
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


# ---------------------------------------------------------------- datasets

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
    """Register a dataset folder path WITHOUT creating it — for flows like
    duplicate that need shutil.copytree (dst must not pre-exist)."""
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


def move_dataset(pid: str, container_id: str | None) -> Path:
    """Physically move a dataset folder into (or out of) a project folder.
    Folder location is cosmetic — the logical link stays in the JSONs."""
    _ensure_scanned()
    with _LOCK:
        d = dataset_dir(pid)
        parent = _CONTAINERS.get(container_id or "") or workspace.projects_dir()
        if d.parent == parent:
            return d
        target = _unique_child(parent, d.name)
        shutil.move(str(d), str(target))
        _DATASETS[pid] = target
        return target


def rename_dataset_dir(pid: str, new_name: str) -> Path:
    """Best-effort folder rename to track a dataset rename."""
    _ensure_scanned()
    with _LOCK:
        d = dataset_dir(pid)
        target = _unique_child(d.parent, slugify(new_name))
        try:
            d.rename(target)
        except OSError:
            return d
        _DATASETS[pid] = target
        return target


# ---------------------------------------------------------------- manifests

def _annotations_dir(d: Path) -> Path:
    return d / "annotations"


def manifest_stamp(pid: str) -> float:
    """Change stamp for cache invalidation: dataset.json mtime (save()
    always rewrites dataset.json, so any change bumps it)."""
    try:
        return (dataset_dir(pid) / "dataset.json").stat().st_mtime
    except (KeyError, OSError):
        return 0.0


def load(pid: str) -> dict | None:
    try:
        d = dataset_dir(pid)
    except KeyError:
        return None
    dj = d / "dataset.json"
    try:
        manifest = _loads(dj.read_bytes())
    except FileNotFoundError:
        return None
    except Exception:
        return None
    ann_dir = _annotations_dir(d)
    digests: dict[str, bytes] = {}
    imports = manifest.get("imports")
    if isinstance(imports, list):
        for imp in imports:
            iid = str(imp.get("id") or "")
            if not iid:
                continue
            path = ann_dir / f"{_safe_id(iid)}.json"
            try:
                raw = path.read_bytes()
            except FileNotFoundError:
                continue
            except OSError:
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
        _ANN_DIGESTS[pid] = digests
    return manifest


def save(pid: str, manifest: dict) -> None:
    d = dataset_dir(pid)  # KeyError for unknown ids — callers create first
    ann_dir = _annotations_dir(d)
    with _LOCK:
        digests = _ANN_DIGESTS.setdefault(pid, {})

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
            elif iid in digests:
                # geometry got dropped from the record — remove the file
                (ann_dir / f"{_safe_id(iid)}.json").unlink(missing_ok=True)
                digests.pop(iid, None)
        slim["imports"] = slim_imports
        # imports removed from the dataset take their annotation files along
        if ann_dir.is_dir():
            for f in ann_dir.iterdir():
                if f.suffix == ".json" and f.name not in live_files:
                    f.unlink(missing_ok=True)

    _atomic_write(d / "dataset.json", _dumps(slim))


# ---------------------------------------------------------------- containers

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
    """Delete a container folder. Datasets inside move up to unfiled first —
    deleting a project must not silently destroy its datasets (the caller
    deletes datasets explicitly when that's intended)."""
    _ensure_scanned()
    with _LOCK:
        d = _CONTAINERS.pop(cid, None)
        if not d or not d.exists():
            return
        for pid, path in list(_DATASETS.items()):
            if path.parent == d:
                target = _unique_child(workspace.projects_dir(), path.name)
                shutil.move(str(path), str(target))
                _DATASETS[pid] = target
    shutil.rmtree(d, ignore_errors=True)


def new_id() -> str:
    return uuid.uuid4().hex[:12]
