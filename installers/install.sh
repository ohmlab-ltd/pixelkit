#!/usr/bin/env bash
# PixelKit dev/CLI install (macOS + Linux). The end-user product is the
# desktop app (Phase 8); this sets up the engine + UI from a checkout.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python3}
$PY -c 'import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)' \
  || { echo "Python 3.11+ required (set PYTHON=...)"; exit 1; }

echo "→ engine venv"
$PY -m venv engine/.venv
V=engine/.venv/bin
$V/pip -q install --upgrade pip

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "→ NVIDIA GPU detected — CUDA torch"
  $V/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
else
  echo "→ no NVIDIA GPU — default torch (Metal on macOS, CPU on Linux)"
fi
$V/pip install -e engine

if command -v npm >/dev/null 2>&1; then
  echo "→ building UI"
  (cd ui && npm ci && npm run build)
else
  echo "! npm not found — engine API works, UI won't be served (install Node, run: cd ui && npm ci && npm run build)"
fi

echo
$V/pixelkit doctor || true
echo "Run: engine/.venv/bin/pixelkit   → http://127.0.0.1:8001"
