"""Import a PixelKit server backup into a workspace.

Input, a folder or a backup zip that contains it:

    <src>/containers/<id>/project.json
    <src>/projects/<uuid>/manifest.json
    <src>/projects/<uuid>/imports/            original images
    <src>/projects/<uuid>/references|augmentations|image_embeddings/

Usage:
    python import_legacy.py <src-dir-or-backup.zip> [--workspace <dir>]

Dataset and project ids are kept. Images missing from <src> must first be
copied into <src>/projects/<id>/images/.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "gd"))


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text("utf-8"))
    except Exception as e:
        print(f"  ! unreadable {path}: {e}")
        return None


def run(src: Path, workspace_override: str | None = None) -> None:
    if workspace_override:
        import os

        os.environ["PIXELKIT_WORKSPACE"] = workspace_override

    import store
    import workspace

    print(f"workspace: {workspace.dir()}")

    tmp: tempfile.TemporaryDirectory | None = None
    if src.is_file() and src.suffix == ".zip":
        tmp = tempfile.TemporaryDirectory(prefix="pixelkit-import-")
        print(f"extracting {src.name} ...")
        with zipfile.ZipFile(src) as zf:
            # Refused when it would not fit with 1 GB to spare; zipfile stops
            # each member at its declared size.
            need = sum(i.file_size for i in zf.infolist())
            free = shutil.disk_usage(tmp.name).free
            if need > free - (1 << 30):
                tmp.cleanup()
                raise SystemExit(f"{src.name} unpacks to {need >> 20} MB and {free >> 20} MB is free")
            zf.extractall(tmp.name)
        src = Path(tmp.name)

    containers_dir = src / "containers"
    projects_dir = src / "projects"
    project_dirs: list[Path] | None = None
    if not projects_dir.is_dir() and (src / "manifest.json").is_file():
        # src is one project folder: import it alone, not its siblings.
        projects_dir = src.parent
        project_dirs = [src]
    if not projects_dir.is_dir() and not containers_dir.is_dir():
        raise SystemExit(f"no projects/ or containers/ under {src}")

    n_containers = 0
    if containers_dir.is_dir():
        for d in sorted(containers_dir.iterdir()):
            cj = d / "project.json"
            c = _load_json(cj) if cj.is_file() else None
            if not c or not c.get("id"):
                continue
            cid = str(c["id"])
            if store.container_exists(cid):
                print(f"  = container {c.get('name')!r} already present, skipping")
                continue
            cdir = store.create_container_dir(cid, c.get("name") or cid)
            (cdir / "project.json").write_text(json.dumps(c, indent=2), "utf-8")
            n_containers += 1
            print(f"  + container {c.get('name')!r} -> {cdir.name}/")

    n_datasets = 0
    if projects_dir.is_dir():
        for d in project_dirs if project_dirs is not None else sorted(projects_dir.iterdir()):
            mj = d / "manifest.json"
            if not d.is_dir() or not mj.is_file():
                continue
            m = _load_json(mj)
            if not m:
                continue
            pid = str(m.get("id") or d.name)
            # The index scan skips a dataset.json without an id.
            m["id"] = pid
            if store.dataset_exists(pid):
                print(f"  = dataset {m.get('name')!r} already present, skipping")
                continue
            container_id = str(m.get("container_id") or "") or None
            ddir = store.create_dataset_dir(pid, m.get("name") or pid, container_id)
            # Originals are in imports/ or images/.
            for src_name in ("imports", "images"):
                sdir = d / src_name
                if sdir.is_dir():
                    for f in sdir.iterdir():
                        if f.is_file():
                            shutil.copy2(f, ddir / "images" / f.name)
            for extra in ("references", "augmentations", "augment_backgrounds", "image_embeddings"):
                sdir = d / extra
                if sdir.is_dir():
                    shutil.copytree(sdir, ddir / extra, dirs_exist_ok=True)
            store.save(pid, m)
            n_imgs = sum(1 for _ in (ddir / "images").iterdir())
            n_datasets += 1
            print(f"  + dataset {m.get('name')!r} ({n_imgs} images) -> {ddir}")

    if tmp:
        tmp.cleanup()
    print(f"done: {n_containers} project(s), {n_datasets} dataset(s) imported.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", help="backup folder, with projects/ or containers/, or backup zip")
    ap.add_argument("--workspace", help="target workspace (default: the configured one)")
    args = ap.parse_args()
    run(Path(args.src).expanduser(), args.workspace)
