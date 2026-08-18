"""Profanity gate - stubbed out in the portable build.

The SaaS build rejected profane project/label names before they landed
in shared, public-facing storage. The portable app is a single local
user writing into their own workspace folder, so moderating their text
is pointless; every call site keeps its `assert_clean(...)` shape and
this module simply accepts everything. (Kept as a module so the ~11
call sites in server.py stay untouched.)
"""
from __future__ import annotations


def contains_profanity(text: str) -> str | None:
    return None


def assert_clean(text: str, field: str = "input") -> None:
    return None


def reload_list() -> int:
    return 0
