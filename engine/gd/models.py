"""Model weights on disk: registry, Hugging Face token and downloads."""
from __future__ import annotations

import json
import os
import shutil
import threading
from pathlib import Path
from typing import Any

import workspace

# The same env vars the loaders read.
SAM3_REPO = os.environ.get("SAM3_MODEL_ID", "facebook/sam3")
DINO_REPO = os.environ.get("V2_DINO_MODEL", "facebook/dinov2-large")

# Files transformers needs; GGUF and ONNX mirrors are skipped.
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
}

# name -> {status: downloading|done|error, downloaded_bytes, total_bytes, error}
_DOWNLOADS: dict[str, dict[str, Any]] = {}
_DL_LOCK = threading.Lock()


# Token

def _config_token() -> str:
    return str(workspace.load_config().get("hf_token") or "").strip()


def hf_token() -> str | None:
    """The configured token, else the HF_TOKEN environment variable, else None."""
    return _config_token() or (os.environ.get("HF_TOKEN") or "").strip() or None


def apply_token_env() -> None:
    """Export the stored token as HF_TOKEN; call before any model load."""
    tok = _config_token()
    if tok:
        os.environ["HF_TOKEN"] = tok


def set_hf_token(token: str) -> None:
    workspace.update_config(lambda c: c.__setitem__("hf_token", token.strip()))
    apply_token_env()


def clear_hf_token() -> None:
    workspace.update_config(lambda c: c.pop("hf_token", None))
    os.environ.pop("HF_TOKEN", None)


def validate_token(token: str | None = None) -> dict[str, Any]:
    """Ask Hugging Face whether the token is valid and can access SAM3_REPO."""
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
            f"token is valid but has no access to {SAM3_REPO} - "
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


# Status

def is_downloaded(name: str) -> bool:
    """True when the local cache holds a complete snapshot; uses no network."""
    from huggingface_hub import snapshot_download

    spec = REGISTRY[name]
    try:
        snap = Path(snapshot_download(
            spec["repo"], local_files_only=True, allow_patterns=_ALLOW_PATTERNS,
        ))
    except Exception:
        return False
    # The folder exists before the weights land. Complete means every
    # shard the index lists, or a single weights file.
    index = snap / "model.safetensors.index.json"
    if index.is_file():
        try:
            shards = set(json.loads(index.read_text("utf-8"))["weight_map"].values())
        except (OSError, ValueError, KeyError):
            return False
        return bool(shards) and all((snap / s).is_file() for s in shards)
    return any(p.is_file() for p in snap.glob("*.safetensors"))


def _loaded(name: str) -> bool:
    """Whether the model is loaded, without importing its module."""
    import sys

    mod_name = {"sam3": "pipeline_charlie", "dinov2": "v2_dinov2"}[name]
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


# Download

# huggingface_hub's bar for bytes received from the network. The bytes written
# to disk, which match the files' sizes, are counted from its other bar.
_TRANSFER_BAR_DESC = "Downloading bytes"


def _make_tqdm(progress: dict[str, Any]):
    """A tqdm class that adds the bytes written for every file to `progress`."""
    from tqdm.std import tqdm as _tqdm

    class _Progress(_tqdm):
        def __init__(self, *args, **kwargs):
            # Read here: a disabled bar sets no attributes.
            self._counts_bytes = kwargs.get("unit") == "B" and kwargs.get("desc") != _TRANSFER_BAR_DESC
            super().__init__(*args, **kwargs)

        def update(self, n=1):
            if self._counts_bytes and n:
                with _DL_LOCK:
                    progress["downloaded_bytes"] += int(n)
            return super().update(n)

    return _Progress


def start_download(name: str) -> dict[str, Any]:
    """Start a background download and return its progress record; an
    in-flight download is returned as it is."""
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
