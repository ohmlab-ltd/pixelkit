"""Auth stubs for the portable build - single local user, no accounts.

The SaaS build verified HS256 bearer tokens minted by NextAuth and enforced
container roles per route. The portable engine binds to 127.0.0.1 and serves
exactly one user, so every dependency below resolves to that user and every
access check passes. The function names, signatures and Depends-factory
shapes are kept identical to the SaaS module so the ~130 route definitions
in server.py don't change.
"""
from __future__ import annotations

from fastapi import Header, Request

LOCAL_USER = "local"


def resolve_terminal_token() -> str | None:
    # Admin "terminal" surface is gone in the portable build.
    return None


def current_user(request: Request, authorization: str = Header(default="")) -> str:
    request.state.username = LOCAL_USER
    request.state.token_payload = {"sub": LOCAL_USER, "username": LOCAL_USER}
    return LOCAL_USER


def request_username(request: Request, authorization: str = "") -> str | None:
    return LOCAL_USER


def can_read_project_request(request: Request, authorization: str, manifest: dict | None) -> bool:
    return True


def require_user_match(path_param_name: str = "username"):
    def _dep(request: Request) -> str:
        return current_user(request)

    return _dep


def _always_local(load_manifest_fn):
    def _dep(request: Request) -> str:
        return current_user(request)

    return _dep


def require_project_owner(load_manifest_fn):
    return _always_local(load_manifest_fn)


def require_dataset_creator(load_manifest_fn):
    return _always_local(load_manifest_fn)


def require_project_manage(load_manifest_fn):
    return _always_local(load_manifest_fn)


def require_project_read_access(load_manifest_fn):
    def _dep(request: Request, authorization: str = Header(default="")) -> str | None:
        return LOCAL_USER

    return _dep
