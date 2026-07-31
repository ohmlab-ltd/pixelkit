#!/usr/bin/env bash
# Build desktop/runtime/ — relocatable CPython + the engine's pinned deps.
set -euo pipefail
cd "$(dirname "$0")/.."
URL=$(curl -s https://api.github.com/repos/astral-sh/python-build-standalone/releases/latest \
  | python3 -c "import json,sys;print([a['browser_download_url'] for a in json.load(sys.stdin)['assets'] if 'cpython-3.12' in a['browser_download_url'] and 'aarch64-apple-darwin-install_only.tar.gz' in a['browser_download_url'] and not a['browser_download_url'].endswith('.sha256')][0])")
curl -sL "$URL" -o /tmp/pbs-python.tar.gz
rm -rf runtime && mkdir -p runtime
tar -xzf /tmp/pbs-python.tar.gz -C runtime && mv runtime/python runtime/py
runtime/py/bin/python3 -m pip install --no-warn-script-location -r ../engine/requirements.lock
runtime/py/bin/python3 -c "import torch, transformers, fastapi, cv2; print('runtime OK')"
