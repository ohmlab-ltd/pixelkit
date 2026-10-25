"""The desktop build ships every engine file the engine reads at run time."""
import fnmatch
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _engine_gd_filter() -> list[str]:
    build = json.loads((ROOT / "desktop" / "package.json").read_text("utf-8"))["build"]
    entry = next(e for e in build["extraResources"] if e.get("from") == "../engine/gd")
    return entry["filter"]


def _shipped(rel: str, patterns: list[str]) -> bool:
    # electron-builder's "**/" also matches the folder itself.
    return any(fnmatch.fnmatch(rel, p) or fnmatch.fnmatch(rel, p.removeprefix("**/")) for p in patterns)


def test_the_yolox_licence_that_exports_copy_is_shipped():
    assert (ROOT / "engine" / "gd" / "vendor" / "yolox" / "LICENSE").is_file()
    assert _shipped("vendor/yolox/LICENSE", _engine_gd_filter())


def test_every_engine_module_is_shipped():
    patterns = _engine_gd_filter()
    for p in (ROOT / "engine" / "gd").rglob("*.py"):
        assert _shipped(p.relative_to(ROOT / "engine" / "gd").as_posix(), patterns), p
