"""Backend JWT auth — bearer-token verification for the FastAPI app.

The frontend's NextAuth `jwt` callback signs a short-lived HS256 token
carrying the authenticated user's `username` as `sub`. The browser
attaches it as `Authorization: Bearer <token>` on every backend call;
this module verifies it and turns it into a `current_user` dependency
for FastAPI routes.

Why HS256 + stdlib (no PyJWT):
    * Two binaries (frontend + backend) sharing one secret is the
      simplest auth pattern that still resists tampering. RS/ES would
      need a key-rotation story we don't have yet.
    * The verify path is < 30 lines using `hmac` + `hashlib`, so we
      can leave it untouched even if PyJWT bumps majors.
    * Format is JWT-compatible (header.payload.sig with HS256), so we
      can swap to PyJWT later without changing the wire protocol.

Set `BACKEND_AUTH_SECRET` on BOTH the frontend (.env.local) and the
backend (.env). Without it the backend refuses authenticated routes
with 503 — fail-closed by design so a misconfigured deploy doesn't
silently accept anonymous requests.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time

from fastapi import Depends, Header, HTTPException, Request

# Membership-aware access. containers.py is stdlib-only and never imports auth/
# server, so this import is cycle-free. A dataset's effective access is resolved
# through its Project container when it has one (container_id), else the legacy
# standalone-owner rule -- so with no containers yet behaviour is unchanged.
import containers

BACKEND_AUTH_SECRET = os.environ.get("BACKEND_AUTH_SECRET", "").strip()
TOKEN_LEEWAY_SECONDS = 60  # tolerate 60 s of clock skew for `exp`


def resolve_terminal_token() -> str | None:
    """Terminal/admin token from env, or None when unset. Fail-closed:
    no hardcoded fallback — callers must 503 when this is None so a
    deploy that forgot to set TERMINAL_TOKEN can't be reached with a
    token baked into the source (which would also leak via git history).
    Mirrors the BACKEND_AUTH_SECRET fail-closed posture below."""
    return os.environ.get("TERMINAL_TOKEN", "").strip() or None

# Name of the cookie that carries the same HS256 bearer for requests a browser
# can't put an Authorization header on -- specifically `<img src>` loads of
# private project images. The frontend sets it (host-only or Domain-scoped to a
# shared parent so it reaches the API origin); the backend reads it as a
# fallback when the Authorization header is absent. Header still wins when both
# are present (explicit API calls).
AUTH_COOKIE_NAME = os.environ.get("BACKEND_AUTH_COOKIE", "pk_auth").strip() or "pk_auth"


def _token_from_request(request: "Request", authorization: str) -> str | None:
    """Pull the bearer token from the Authorization header, falling back to the
    auth cookie. Returns the raw token string or None. Used by every auth
    dependency so cookie-carrying requests (image <img> loads) authenticate
    exactly like header-carrying API (fetch) calls."""
    if authorization and authorization.lower().startswith("bearer "):
        tok = authorization.split(None, 1)[1].strip()
        if tok:
            return tok
    try:
        tok = request.cookies.get(AUTH_COOKIE_NAME)
    except Exception:
        tok = None
    return tok.strip() if tok else None


def _b64u_encode(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")


def _b64u_decode(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def sign_jwt(payload: dict, *, secret: str | None = None, ttl_seconds: int = 3600) -> str:
    """Mint an HS256 JWT. Used in tests and by the optional debug
    endpoint; the frontend mints its own via `jose` so this isn't on
    the hot path."""
    if secret is None:
        secret = BACKEND_AUTH_SECRET
    if not secret:
        raise RuntimeError("BACKEND_AUTH_SECRET not configured")
    now = int(time.time())
    body = {**payload, "iat": now, "exp": now + ttl_seconds}
    header_b64 = _b64u_encode(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
    payload_b64 = _b64u_encode(json.dumps(body, separators=(",", ":")).encode())
    signing_input = f"{header_b64}.{payload_b64}".encode()
    sig = hmac.new(secret.encode(), signing_input, hashlib.sha256).digest()
    return f"{header_b64}.{payload_b64}.{_b64u_encode(sig)}"


def verify_jwt(token: str, *, secret: str | None = None) -> dict | None:
    """Verify an HS256 JWT and return its payload, or None on any
    failure (bad shape, bad signature, expired). We don't raise here —
    the caller turns None into 401 with a generic message so the
    response can't be used as a signal-rich oracle."""
    if secret is None:
        secret = BACKEND_AUTH_SECRET
    if not secret or not token:
        return None
    try:
        header_b64, payload_b64, sig_b64 = token.split(".")
    except ValueError:
        return None
    signing_input = f"{header_b64}.{payload_b64}".encode()
    expected = hmac.new(secret.encode(), signing_input, hashlib.sha256).digest()
    try:
        actual = _b64u_decode(sig_b64)
    except Exception:
        return None
    if not hmac.compare_digest(expected, actual):
        return None
    try:
        payload = json.loads(_b64u_decode(payload_b64))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    exp = payload.get("exp")
    if isinstance(exp, (int, float)) and time.time() > exp + TOKEN_LEEWAY_SECONDS:
        return None
    return payload


# ─── FastAPI dependencies ─────────────────────────────────────────────────────

def _extract_username(payload: dict) -> str | None:
    """Pull the username out of the token payload. NextAuth's default
    JWT puts the user's database id in `sub`; we add `username` as a
    custom claim on the frontend side."""
    for key in ("username", "sub"):
        v = payload.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def current_user(
    request: Request,
    authorization: str = Header(default=""),
) -> str:
    """FastAPI dependency: returns the authenticated username, else
    401. Use as `Depends(current_user)` on any route that needs to
    know who is calling. Also stashes the full JWT payload on
    `request.state.token_payload` so downstream deps (the credit
    gate, future plan-aware logic) can read claims without re-
    verifying the token."""
    if not BACKEND_AUTH_SECRET:
        # Fail-closed: a deploy without a configured secret is a
        # misconfiguration, not a free-pass to anonymous access.
        raise HTTPException(503, "auth not configured")
    token = _token_from_request(request, authorization)
    if not token:
        raise HTTPException(401, "missing bearer token")
    payload = verify_jwt(token)
    if not payload:
        raise HTTPException(401, "invalid or expired token")
    username = _extract_username(payload)
    if not username:
        raise HTTPException(401, "token has no user")
    # Stash on the request so route handlers + downstream deps can
    # access without an extra dependency injection. token_payload
    # holds the verified JWT claims (plan, sub_status, beta_exp,
    # cycle_start), used by the credit-enforcement gate.
    request.state.username = username
    request.state.token_payload = payload
    return username


def require_user_match(path_param_name: str = "username"):
    """Dependency factory: the `{path_param_name}` URL segment must
    match the bearer-token user. Use as
    `dependencies=[Depends(require_user_match("username"))]` on user-
    scoped routes (rename, delete data, etc.)."""

    def _dep(request: Request, user: str = Depends(current_user)) -> str:
        target = request.path_params.get(path_param_name) or ""
        if user.lower() != str(target).strip().lower():
            raise HTTPException(403, "not your account")
        return user

    return _dep


def require_project_owner(load_manifest_fn):
    """Dependency factory: the {project_id} URL segment must point to
    a project whose manifest `owner` matches the bearer-token user.

    Takes the host module's `load_manifest` function as an argument
    rather than importing it directly to avoid a circular import
    (server.py ↔ auth.py).
    """

    def _dep(request: Request, user: str = Depends(current_user)) -> str:
        project_id = request.path_params.get("project_id") or ""
        if not project_id:
            raise HTTPException(400, "project_id required")
        # copy=False — read-only access check. Without this every
        # authenticated mutation endpoint (every upload, every box
        # drag, every label-job schedule) paid a 300-500ms deepcopy
        # of the full manifest before the handler even ran.
        manifest = load_manifest_fn(project_id, copy=False)
        if not manifest:
            raise HTTPException(404, "project not found")
        # WRITE access: standalone dataset -> the owner; dataset inside a
        # Project container -> any editor-or-owner member. Legacy unowned
        # standalone datasets resolve to writable=False, so they stay blocked
        # exactly as before (no free-for-all).
        if not containers.dataset_access(manifest, user)["writable"]:
            raise HTTPException(403, "not your project")
        return user

    return _dep


def require_dataset_creator(load_manifest_fn):
    """Dependency factory: DESTROY access for a dataset. The bearer must be the
    dataset's own creator (manifest ``owner``) — NOT merely a Project editor, and
    NOT even the Project owner. Permanently deleting a dataset is irreversible
    and personal, so only the person who made it may do it, whatever their role
    in the containing Project.

    Legacy standalone datasets with no recorded owner fall back to the write
    rule so they aren't left permanently undeletable; any dataset that DOES
    record an owner is strictly creator-only.
    """

    def _dep(request: Request, user: str = Depends(current_user)) -> str:
        project_id = request.path_params.get("project_id") or ""
        if not project_id:
            raise HTTPException(400, "project_id required")
        manifest = load_manifest_fn(project_id, copy=False)
        if not manifest:
            raise HTTPException(404, "project not found")
        owner = (manifest.get("owner") or "").strip().lower()
        if owner:
            if owner == (user or "").strip().lower():
                return user
            raise HTTPException(403, "only the dataset's creator can delete it")
        # No owner on record (legacy): fall back to the standard write rule.
        if containers.dataset_access(manifest, user)["writable"]:
            return user
        raise HTTPException(403, "only the dataset's creator can delete it")

    return _dep


def request_username(request: "Request", authorization: str = "") -> str | None:
    """Resolve the authenticated username from the Authorization header or the
    pk_auth cookie, non-raising (None if absent/invalid). For routes that want
    to know the caller without forcing auth -- e.g. container reads where a
    public container is anonymous-readable but a private one needs a member."""
    if not BACKEND_AUTH_SECRET:
        return None
    token = _token_from_request(request, authorization)
    if not token:
        return None
    payload = verify_jwt(token)
    if not payload:
        return None
    return _extract_username(payload)


def can_read_project_request(request: "Request", authorization: str, manifest: dict | None) -> bool:
    """Non-raising read-access check for endpoints whose project id isn't a
    path param (so they can't use the require_project_read_access dependency) --
    e.g. the /api/jobs/{job_id}/events SSE, which resolves the project off the
    job. Public projects/containers: anyone. Private: owner or container member
    (token via header OR pk_auth cookie). Mirrors the read guard's policy."""
    if not manifest:
        return False
    username = None
    token = _token_from_request(request, authorization)
    if token and BACKEND_AUTH_SECRET:
        payload = verify_jwt(token)
        if payload:
            username = _extract_username(payload)
    return bool(containers.dataset_access(manifest, username)["readable"])


def require_project_read_access(load_manifest_fn):
    """Dependency factory for *read* endpoints. Public projects are
    readable by anyone (no auth required). Private projects are
    readable only by their owner; anyone else gets a 404 (we
    intentionally do not 403 -- a 403 would confirm the project's
    existence to an attacker who guessed or scraped its UUID).

    Use on every GET that returns project metadata or per-image data:
    /initial, /overview, /annotations, /annotations/{import_id} and
    anything similar. Mutations should still use
    `require_project_owner` (strict).
    """

    def _dep(request: Request, authorization: str = Header(default="")) -> str | None:
        project_id = request.path_params.get("project_id") or ""
        if not project_id:
            raise HTTPException(400, "project_id required")
        # copy=False — read-only privacy + access check.
        manifest = load_manifest_fn(project_id, copy=False)
        if not manifest:
            raise HTTPException(404, "project not found")
        # Resolve the caller (token via header or pk_auth cookie); may be None
        # for an anonymous reader of a public dataset/container.
        username = None
        token = _token_from_request(request, authorization)
        if token and BACKEND_AUTH_SECRET:
            payload = verify_jwt(token)
            if payload:
                username = _extract_username(payload)
        # READ access via the container (public -> anyone; private -> member),
        # or the standalone-owner rule. Any denial returns 404 (not 403) so the
        # response can't confirm a private UUID exists.
        if containers.dataset_access(manifest, username)["readable"]:
            return username
        raise HTTPException(404, "project not found")

    return _dep


def require_project_manage(load_manifest_fn):
    """Dependency factory: MANAGE access (rename, cover, privacy, members,
    delete). Standalone dataset -> the owner; dataset in a Project container ->
    the container OWNER only (editors can do work but not manage)."""

    def _dep(request: Request, user: str = Depends(current_user)) -> str:
        project_id = request.path_params.get("project_id") or ""
        if not project_id:
            raise HTTPException(400, "project_id required")
        manifest = load_manifest_fn(project_id, copy=False)
        if not manifest:
            raise HTTPException(404, "project not found")
        if not containers.dataset_access(manifest, user)["manageable"]:
            raise HTTPException(403, "requires owner")
        return user

    return _dep
