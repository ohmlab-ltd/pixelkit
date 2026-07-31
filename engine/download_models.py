"""Pre-download all model weights so the server has them ready on first run.

Run once after cloning:
    conda run -n groundingdino python download_models.py

Worker box only needs the VLM:
    python download_models.py --vlm-only
"""
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = ROOT / "weights"

SAM2_CHECKPOINT = WEIGHTS_DIR / "sam2.1_hiera_large.pt"
SAM2_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"

# VLM model id mirrors the runtime default in gd/vlm_validate.py so
# the pre-download lands the same weights the server will load. Set
# the same VLM_MODEL env var to override both.
VLM_MODEL_ID = os.environ.get("VLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
VLM_CACHE_DIR = Path(
    os.environ.get("VLM_CACHE_DIR")
    or os.environ.get("HF_HOME")
    or (ROOT / "models_cache")
)


def download_sam2():
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    if SAM2_CHECKPOINT.exists():
        print(f"[ok] SAM2.1 large already present ({SAM2_CHECKPOINT})")
        return
    print(f"Downloading SAM2.1 Hiera Large (~900 MB)...")
    tmp = SAM2_CHECKPOINT.with_suffix(".pt.tmp")
    try:
        last = [0]
        def _progress(count, block, total):
            pct = min(100, count * block * 100 // max(1, total))
            if pct != last[0]:
                last[0] = pct
                print(f"\r  {pct}% ({count * block // 1_048_576} MB / {total // 1_048_576} MB)", end="", flush=True)
        urllib.request.urlretrieve(SAM2_URL, tmp, reporthook=_progress)
        print()
        tmp.rename(SAM2_CHECKPOINT)
        print(f"[ok] saved to {SAM2_CHECKPOINT}")
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        print(f"[error] SAM2 download failed: {exc}", file=sys.stderr)
        sys.exit(1)


def pull_vlm():
    """Pre-pull VLM weights from HuggingFace into the local cache so
    the worker's first boot doesn't have to download ~16 GB during
    its lifespan startup. Idempotent — snapshot_download is a no-op
    when the cache already has the requested revision."""
    print(f"Pre-downloading VLM weights: {VLM_MODEL_ID}")
    print(f"Cache dir: {VLM_CACHE_DIR / 'hub'}")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "[error] huggingface_hub not installed. "
            "On the worker box: pip install huggingface_hub",
            file=sys.stderr,
        )
        sys.exit(1)
    cache_dir = VLM_CACHE_DIR / "hub"
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        path = snapshot_download(
            repo_id=VLM_MODEL_ID,
            cache_dir=str(cache_dir),
            # Skip the gguf / onnx / consolidated.safetensors mirrors —
            # we only need the standard HF safetensors shards. Cuts the
            # download from ~30 GB to ~16 GB for many VLM repos that
            # ship multiple formats.
            allow_patterns=[
                "*.json", "*.txt", "*.safetensors",
                "tokenizer*", "preprocessor*", "chat_template*",
                "*.py",
            ],
        )
        print(f"[ok] VLM weights cached at {path}")
    except Exception as exc:
        print(f"[error] VLM download failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    args = sys.argv[1:]
    vlm_only = "--vlm-only" in args
    if not vlm_only:
        download_sam2()
    pull_vlm()
    print("\nAll models ready.")
