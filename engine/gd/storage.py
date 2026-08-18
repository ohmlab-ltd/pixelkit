"""Local filesystem storage for the portable build.

Drop-in replacement for the SaaS build's R2Storage (Cloudflare R2 via boto3).
The engine still talks in R2-style keys - ``projects/{pid}/images/{fn}``,
``projects/{pid}/outputs/{fn}``, ... - and this class resolves them onto the
workspace's per-dataset folders via the store index, so call sites don't
change. Bytes are read/written directly; "presigned URLs" no longer exist
(the API serves files itself with FileResponse).
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Iterable

from PIL import Image as PILImage

import store
import workspace


class LocalStorage:
    # ---- key helpers (same shapes the SaaS build used) ----
    @staticmethod
    def image_key(project: str, filename: str) -> str:
        return f"projects/{project}/images/{filename}"

    @staticmethod
    def output_key(project: str, filename: str) -> str:
        return f"projects/{project}/outputs/{filename}"

    @staticmethod
    def project_prefix(project: str) -> str:
        return f"projects/{project}/"

    @staticmethod
    def avatar_key(user_id: str) -> str:  # legacy shape; unused locally
        return f"avatars/{user_id}.jpg"

    # ---- key -> path ----
    @staticmethod
    def resolve(key: str) -> Path:
        parts = [p for p in str(key).split("/") if p not in ("", ".", "..")]
        if len(parts) >= 2 and parts[0] == "projects":
            try:
                base = store.dataset_dir(parts[1])
            except KeyError:
                raise FileNotFoundError(f"unknown dataset in key: {key!r}") from None
            return base.joinpath(*parts[2:]) if len(parts) > 2 else base
        # anything else (legacy avatars/containers keys) lands in a misc dir
        return workspace.dir().joinpath("_storage", *parts)

    # ---- ops ----
    def put_bytes(self, key: str, data: bytes, content_type: str | None = None) -> None:
        path = self.resolve(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + f".tmp{os.getpid()}")
        tmp.write_bytes(data)
        os.replace(tmp, path)

    def put_fileobj(self, key: str, fileobj, content_type: str | None = None) -> None:
        path = self.resolve(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + f".tmp{os.getpid()}")
        with open(tmp, "wb") as f:
            shutil.copyfileobj(fileobj, f)
        os.replace(tmp, path)

    def get_bytes(self, key: str) -> bytes:
        return self.resolve(key).read_bytes()

    def get_pil(self, key: str) -> PILImage.Image:
        img = PILImage.open(self.resolve(key))
        img.load()
        return img

    def exists(self, key: str) -> bool:
        try:
            return self.resolve(key).exists()
        except FileNotFoundError:
            return False

    def delete(self, key: str) -> None:
        try:
            self.resolve(key).unlink(missing_ok=True)
        except FileNotFoundError:
            pass

    def delete_prefix(self, prefix: str) -> int:
        try:
            root = self.resolve(prefix)
        except FileNotFoundError:
            return 0
        if not root.exists():
            return 0
        n = sum(1 for p in root.rglob("*") if p.is_file())
        shutil.rmtree(root, ignore_errors=True)
        return n

    def copy(self, src_key: str, dst_key: str) -> None:
        src = self.resolve(src_key)
        dst = self.resolve(dst_key)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)

    def move_prefix(self, src_prefix: str, dst_prefix: str) -> int:
        src = self.resolve(src_prefix)
        if not src.exists():
            return 0
        dst = self.resolve(dst_prefix)
        dst.parent.mkdir(parents=True, exist_ok=True)
        n = sum(1 for p in src.rglob("*") if p.is_file())
        shutil.move(str(src), str(dst))
        return n

    def list_keys(self, prefix: str) -> Iterable[str]:
        try:
            root = self.resolve(prefix)
        except FileNotFoundError:
            return
        if not root.exists():
            return
        base = str(prefix).rstrip("/")
        for p in sorted(root.rglob("*")):
            if p.is_file():
                yield f"{base}/{p.relative_to(root).as_posix()}"

    def presigned_get_url(self, key: str, expires: int | None = None) -> str:
        raise NotImplementedError("local storage serves files directly")


# Back-compat alias so `from storage import R2Storage` keeps importing.
R2Storage = LocalStorage

_INSTANCE: LocalStorage | None = None


def from_env() -> LocalStorage:
    """Always available locally (the SaaS version returned None without R2
    credentials and the API 503'd; that failure mode is gone)."""
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = LocalStorage()
    return _INSTANCE
