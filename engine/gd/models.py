"""Model manager — download state, HF token, progress.

Owns everything about getting weights onto disk:
  - the registry of models the portable build uses (SAM3 required+gated,
    DINOv2 required, the small VLM optional),
  - the user's Hugging Face token (needed for the gated facebook/sam3):
    validated against the hub, stored in the app-config dir (never in the
    workspace, which may be shared/synced), exported as HF_TOKEN so
    transformers/hub pick it up,
  - background downloads into <workspace>/weights (HF_HOME) with byte-level
    progress and a disk-space preflight.

Loading models onto the GPU stays in server.py (it owns `state` and the
load lock); this module only answers "is it on disk / how far along is the
download / may we access the repo".
"""
from __future__ import annotations

import os
import shutil
import threading
from typing import Any

import workspace

# Read repo ids from the same env vars the loaders use, without importing
# the (torch-heavy) loader modules.
SAM3_REPO = os.environ.get("SAM3_MODEL_ID", "facebook/sam3")
DINO_REPO = os.environ.get("V2_DINO_MODEL", "facebook/dinov2-large")
VLM_REPO = os.environ.get("VLM_MODEL", "Qwen/Qwen3-VL-2B-Instruct")

# Skip alternative-format mirrors some repos carry (GGUF/ONNX etc.) —
# transformers only needs configs + safetensors + tokenizer/processor files.
_ALLOW_PATTERNS = [
    "*.json", "*.txt", "*.model", "*.safetensors",
    "tokenizer*", "preprocessor*", "processor*", "chat_template*", "*.py",
]

REGISTRY: dict[str, dict[str, Any]] = {
    "sam3": {
        "repo": SAM3_REPO,
        "label": "SAM 3 (auto-labelling)",
        "gated": True,
        "required": True,
        "approx_gb": 3.6,
    },
    "dinov2": {
        "repo": DINO_REPO,
        "label": "DINOv2 (reference matching)",
        "gated": False,
        "required": True,
        "approx_gb": 1.3,
    },
    "vlm": {
        "repo": VLM_REPO,
        "label": "Qwen3-VL 2B (label tiebreak, optional)",
        "gated": False,
        "required": False,
        "approx_gb": 4.5,
    },
}

# name -> {status: idle|downloading|done|error, downloaded_bytes, total_bytes, error}
_DOWNLOADS: dict[str, dict[str, Any]] = {}
_DL_LOCK = threading.Lock()


# ---------------------------------------------------------------- token

def _config_token() -> str:
    return str(workspace.load_config().get("hf_token") or "").strip()


def hf_token() -> str | None:
    """Config wins; a manually-exported HF_TOKEN env works too."""
    return _config_token() or (os.environ.get("HF_TOKEN") or "").strip() or None


def apply_token_env() -> None:
    """Export the stored token so transformers/hub calls inherit it.
    Call before any model load. No-op when nothing is configured."""
    tok = _config_token()
    if tok:
        os.environ["HF_TOKEN"] = tok


def set_hf_token(token: str) -> None:
    cfg = workspace.load_config()
    cfg["hf_token"] = token.strip()
    workspace.save_config(cfg)
    try:  # token at rest: owner-only file perms (POSIX; no-op on Windows)
        os.chmod(workspace.config_dir() / "config.json", 0o600)
    except OSError:
        pass
    apply_token_env()


def clear_hf_token() -> None:
    cfg = workspace.load_config()
    cfg.pop("hf_token", None)
    workspace.save_config(cfg)
    os.environ.pop("HF_TOKEN", None)


def validate_token(token: str | None = None) -> dict[str, Any]:
    """Network check: who is this token, and does it unlock SAM3?
    Distinguishes the failure modes the setup screen must explain:
    invalid token / valid but license not accepted / all good."""
    import huggingface_hub as hh
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

    tok = (token or hf_token() or "").strip() or None
    out: dict[str, Any] = {
        "configured": bool(tok),
        "valid": None,
        "username": None,
        "sam3Access": None,
        "detail": None,
    }
    if not tok:
        out["detail"] = "no token configured"
        return out
    try:
        who = hh.whoami(token=tok)
        out["valid"] = True
        out["username"] = who.get("name")
    except Exception as e:
        out["valid"] = False
        out["detail"] = f"token rejected by Hugging Face: {e}"
        return out
    try:
        hh.auth_check(SAM3_REPO, token=tok)
        out["sam3Access"] = True
    except GatedRepoError:
        out["sam3Access"] = False
        out["detail"] = (
            f"token is valid but has no access to {SAM3_REPO} — "
            "accept the license on the model page first"
        )
    except RepositoryNotFoundError:
        out["sam3Access"] = False
        out["detail"] = f"{SAM3_REPO} not found (private or renamed?)"
    except HfHubHTTPError as e:
        out["detail"] = f"access check failed: {e}"
    except Exception as e:
        out["detail"] = f"access check failed: {e}"
    return out


# ---------------------------------------------------------------- status

def is_downloaded(name: str) -> bool:
    """True when a complete snapshot sits in the local cache (no network)."""
    from huggingface_hub import snapshot_download

    spec = REGISTRY[name]
    try:
        snapshot_download(
            spec["repo"], local_files_only=True, allow_patterns=_ALLOW_PATTERNS,
        )
        return True
    except Exception:
        return False


def _loaded(name: str) -> bool:
    """Loaded-on-device state, without importing torch-heavy modules that
    aren't already in the process."""
    import sys

    mod_name = {"sam3": "pipeline_charlie", "dinov2": "v2_dinov2", "vlm": "vlm_validate"}[name]
    mod = sys.modules.get(mod_name)
    if mod is None:
        return False
    try:
        return bool(mod.is_loaded())
    except Exception:
        return False


def free_disk_gb() -> float:
    return shutil.disk_usage(workspace.weights_dir()).free / 1e9


def status() -> dict[str, Any]:
    models = {}
    with _DL_LOCK:
        downloads = {k: dict(v) for k, v in _DOWNLOADS.items()}
    for name, spec in REGISTRY.items():
        models[name] = {
            "repo": spec["repo"],
            "label": spec["label"],
            "gated": spec["gated"],
            "required": spec["required"],
            "approxGb": spec["approx_gb"],
            "downloaded": is_downloaded(name),
            "loaded": _loaded(name),
            "download": downloads.get(name),
        }
    return {
        "models": models,
        "weightsDir": str(workspace.weights_dir()),
        "freeDiskGb": round(free_disk_gb(), 1),
        "hfTokenConfigured": bool(hf_token()),
    }


# ---------------------------------------------------------------- download

def _make_tqdm(progress: dict[str, Any]):
    """A tqdm stand-in the hub drives per file; we aggregate bytes across
    files into one progress dict the status endpoint reports."""
    from tqdm.std import tqdm as _tqdm

    class _Progress(_tqdm):
        def update(self, n=1):
            if self.unit == "B" and n:
                with _DL_LOCK:
                    progress["downloaded_bytes"] += int(n)
            return super().update(n)

    return _Progress


def start_download(name: str) -> dict[str, Any]:
    """Kick a background download; returns the progress record. Safe to
    call repeatedly — an in-flight download is returned, a finished one
    restarts only if the snapshot is incomplete."""
    if name not in REGISTRY:
        raise KeyError(name)
    spec = REGISTRY[name]

    with _DL_LOCK:
        rec = _DOWNLOADS.get(name)
        if rec and rec["status"] == "downloading":
            return dict(rec)
        rec = {
            "status": "downloading",
            "downloaded_bytes": 0,
            "total_bytes": int(spec["approx_gb"] * 1e9),
            "error": None,
        }
        _DOWNLOADS[name] = rec

    if free_disk_gb() < spec["approx_gb"] + 2.0:
        with _DL_LOCK:
            rec["status"] = "error"
            rec["error"] = (
                f"not enough disk space: need ~{spec['approx_gb']} GB "
                f"(+2 GB headroom), {free_disk_gb():.1f} GB free"
            )
        return dict(rec)

    def _run() -> None:
        from huggingface_hub import snapshot_download

        try:
            apply_token_env()
            snapshot_download(
                spec["repo"],
                allow_patterns=_ALLOW_PATTERNS,
                tqdm_class=_make_tqdm(rec),
                token=hf_token(),
            )
            with _DL_LOCK:
                rec["status"] = "done"
        except Exception as e:
            with _DL_LOCK:
                rec["status"] = "error"
                rec["error"] = str(e)

    threading.Thread(target=_run, name=f"pk-download-{name}", daemon=True).start()
    return dict(rec)
