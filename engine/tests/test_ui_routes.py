"""Every engine path the UI calls is one the engine serves. URLs built from a
constant, such as `${segBase}/segment_box`, are resolved from the string
literals of that constant in the same file; a path after the engine's origin,
as in `${API_BASE}/api/...`, is read as written."""
import re
from pathlib import Path

UI = Path(__file__).resolve().parents[2] / "ui"

_LITERAL = re.compile(r"""(?:[`"']|\$\{\w+\})(/api/[^`"'\s?#]*)""")
_BUILT = re.compile(r"\$\{(\w+)\}(/[A-Za-z0-9_/${}.\-]*)")
_CONST = re.compile(r"const\s+(\w+)\s*=([^;\n]+)")
_API_STRING = re.compile(r"""["'`](/api/[^"'`$]*)["'`]""")


def _ui_paths():
    files = sorted(list((UI / "app").rglob("*.ts")) + list((UI / "app").rglob("*.tsx"))
                   + list((UI / "lib").rglob("*.ts")))
    for f in files:
        if ".test." in f.name:
            continue
        text = f.read_text("utf-8")
        consts = {m.group(1): _API_STRING.findall(m.group(2)) for m in _CONST.finditer(text)}
        built = [(m.group(1), m.group(2)) for m in _BUILT.finditer(text)]
        bases = {v for name, _ in built for v in consts.get(name, [])}
        for m in _LITERAL.finditer(text):
            if m.group(1) not in bases:
                yield f, m.group(1)
        for name, rest in built:
            for base in consts.get(name, []):
                yield f, base + rest


def _normalise(path: str) -> str:
    path = re.sub(r"\$\{[^}]*$", "", path)  # a template that goes on to the next line
    path = re.sub(r"(?<!/)\$\{[^}]*\}", "", path)  # a query string or suffix
    path = re.sub(r"\$\{[^}]*\}", "X", path)
    return path.rstrip("/")


def test_every_path_the_ui_calls_is_served(app_module):
    served = []
    for r in app_module.app.routes:
        p = getattr(r, "path", "")
        if p.startswith("/api"):
            rx = re.escape(p)
            rx = re.sub(r"\\\{[^}]*:path\\\}", ".+", rx)
            rx = re.sub(r"\\\{[^}]*\\\}", "[^/]+", rx)
            served.append(re.compile(f"^{rx}$"))
    missing = sorted({f"{_normalise(u)} ({f.relative_to(UI)})" for f, u in _ui_paths()
                      if not any(rx.match(_normalise(u)) for rx in served)})
    assert not missing, "\n".join(missing)
