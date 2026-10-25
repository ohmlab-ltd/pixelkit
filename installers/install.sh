#!/usr/bin/env bash
# Sets up the engine and UI from a checkout, on macOS or Linux.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python3}
$PY -c 'import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)' \
  || { echo "Python 3.11+ required (set PYTHON=...)"; exit 1; }

echo "Creating engine/.venv"
$PY -m venv engine/.venv
V=engine/.venv/bin
$V/pip -q install --upgrade pip

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "NVIDIA GPU found: installing CUDA PyTorch"
  $V/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
else
  echo "No NVIDIA GPU: using the default PyTorch (Metal on macOS, CPU on Linux)"
fi
$V/pip install -e engine

if command -v npm >/dev/null 2>&1; then
  echo "Building the UI"
  (cd ui && npm ci && npm run build)
else
  echo "npm not found. The engine API works, but the UI will not be served. Install Node, then run: cd ui && npm ci && npm run build"
fi

echo
$V/pixelkit doctor || true
echo "Run engine/.venv/bin/pixelkit, then open http://127.0.0.1:8001"
