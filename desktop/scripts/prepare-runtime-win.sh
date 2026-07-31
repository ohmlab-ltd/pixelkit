#!/usr/bin/env bash
# Build desktop/runtime-win/ — relocatable Windows CPython + engine deps,
# cross-fetched from macOS/Linux via pip --platform (wheels only).
# CUDA torch (cu126) so NVIDIA machines get full labelling; CPU-only
# machines still run every dataset/annotation API.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=../engine/.venv/bin/python
URL=$(curl -s https://api.github.com/repos/astral-sh/python-build-standalone/releases/latest \
  | python3 -c "import json,sys;print([a['browser_download_url'] for a in json.load(sys.stdin)['assets'] if 'cpython-3.12' in a['browser_download_url'] and 'x86_64-pc-windows-msvc-install_only.tar.gz' in a['browser_download_url'] and not a['browser_download_url'].endswith('.sha256')][0])")
curl -sL "$URL" -o /tmp/pbs-win.tar.gz
rm -rf runtime-win && mkdir -p runtime-win
tar -xzf /tmp/pbs-win.tar.gz -C runtime-win && mv runtime-win/python runtime-win/py
$PY -m pip install --platform win_amd64 --python-version 3.12 --implementation cp \
  --only-binary=:all: --target runtime-win/py/Lib/site-packages \
  torch torchvision --index-url https://download.pytorch.org/whl/cu126
$PY -m pip install --platform win_amd64 --python-version 3.12 --implementation cp \
  --only-binary=:all: --target runtime-win/py/Lib/site-packages \
  -r ../engine/requirements-win.txt
du -sh runtime-win
