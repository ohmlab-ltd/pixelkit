"""Cloudflare R2 storage for project images and annotated outputs.

R2 is S3-compatible, so this is just boto3 with a custom endpoint. Bytes flow
browser ↔ R2 directly via short-lived presigned URLs (zero egress on R2), so
the FastAPI backend never serves image bytes — only redirects.

Configure via env vars:
    R2_ACCOUNT_ID         — your Cloudflare account id
    R2_ACCESS_KEY_ID      — R2 API token "Access Key ID"
    R2_SECRET_ACCESS_KEY  — R2 API token "Secret Access Key"
    R2_BUCKET             — bucket name
    R2_PRESIGN_TTL        — optional, seconds (default 3600)

CORS on the bucket must allow GET from the frontend origin so canvas/img loads
work cross-origin. Configure once in the Cloudflare dashboard:
    [{"AllowedOrigins": ["https://your.app", "http://localhost:3000"],
      "AllowedMethods": ["GET"], "AllowedHeaders": ["*"]}]

Key layout (matches the old on-disk layout one-to-one):
    projects/<name>/images/<file>
    projects/<name>/outputs/<file>
"""
from __future__ import annotations

import io
import os
from typing import Iterable

import boto3
from botocore.client import Config
from PIL import Image as PILImage


# 24h presigned-URL TTL. Has to comfortably exceed the browser's SWR
# window for the 302 redirect (max-age=900 + swr=14400 ≈ 4h 15min) plus
# the server-side URL cache TTL (5h) — otherwise a browser using a stale
# cached 302 will follow it to an expired URL and get a 403 from R2.
# Override with R2_PRESIGN_TTL to shrink in dev (NOT in prod — keep 24h+).
_DEFAULT_TTL = int(os.getenv("R2_PRESIGN_TTL", "86400"))


class R2Storage:
    def __init__(
        self,
        account_id: str,
        access_key_id: str,
        secret_access_key: str,
        bucket: str,
        presign_ttl: int = _DEFAULT_TTL,
    ):
        self.bucket = bucket
        self.presign_ttl = presign_ttl
        self.client = boto3.client(
            "s3",
            endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            # R2 requires SigV4 with us-east-1 as a placeholder region.
            region_name="auto",
            config=Config(signature_version="s3v4"),
        )

    # ---- key helpers ----

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
    def avatar_key(user_id: str) -> str:
        # Always write JPEG so the browser cache keys cleanly on user_id alone.
        return f"avatars/{user_id}.jpg"

    # ---- core ops ----

    def put_bytes(self, key: str, data: bytes, content_type: str | None = None) -> None:
        kwargs = {"Bucket": self.bucket, "Key": key, "Body": data}
        if content_type:
            kwargs["ContentType"] = content_type
        self.client.put_object(**kwargs)

    def put_fileobj(self, key: str, fileobj, content_type: str | None = None) -> None:
        extra = {"ContentType": content_type} if content_type else {}
        self.client.upload_fileobj(fileobj, self.bucket, key, ExtraArgs=extra)

    def get_bytes(self, key: str) -> bytes:
        obj = self.client.get_object(Bucket=self.bucket, Key=key)
        return obj["Body"].read()

    def get_pil(self, key: str) -> PILImage.Image:
        data = self.get_bytes(key)
        return PILImage.open(io.BytesIO(data))

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except self.client.exceptions.ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=key)

    def delete_prefix(self, prefix: str) -> int:
        """Delete every key under `prefix`. Used when removing or renaming a project."""
        n = 0
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            contents = page.get("Contents") or []
            if not contents:
                continue
            self.client.delete_objects(
                Bucket=self.bucket,
                Delete={"Objects": [{"Key": o["Key"]} for o in contents]},
            )
            n += len(contents)
        return n

    def copy(self, src_key: str, dst_key: str) -> None:
        self.client.copy_object(
            Bucket=self.bucket,
            Key=dst_key,
            CopySource={"Bucket": self.bucket, "Key": src_key},
        )

    def move_prefix(self, src_prefix: str, dst_prefix: str) -> int:
        """Rename all keys under src_prefix → dst_prefix. R2 has no rename op,
        so this is copy-then-delete; safe but not atomic."""
        n = 0
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=src_prefix):
            for obj in page.get("Contents") or []:
                src = obj["Key"]
                dst = dst_prefix + src[len(src_prefix):]
                self.copy(src, dst)
                self.delete(src)
                n += 1
        return n

    def list_keys(self, prefix: str) -> Iterable[str]:
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents") or []:
                yield obj["Key"]

    # ---- presigned URLs ----

    def presigned_get_url(self, key: str, expires: int | None = None) -> str:
        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expires or self.presign_ttl,
        )


def from_env() -> R2Storage | None:
    """Build an R2Storage from env vars, or return None if unconfigured.
    Lets the server boot in dev without R2 set up — endpoints that need it
    will 503 until env is populated."""
    needed = ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET")
    if any(not os.getenv(k) for k in needed):
        return None
    return R2Storage(
        account_id=os.environ["R2_ACCOUNT_ID"],
        access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        bucket=os.environ["R2_BUCKET"],
    )
