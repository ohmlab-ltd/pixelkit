#!/usr/bin/env bash
# Builds desktop/runtime/: relocatable CPython with the engine's pinned deps.
set -euo pipefail
cd "$(dirname "$0")/.."
# The python-build-standalone release desktop/bootstrap.js installs on Windows.
URL=https://github.com/astral-sh/python-build-standalone/releases/download/20260807/cpython-3.12.13%2B20260807-aarch64-apple-darwin-install_only.tar.gz
# As in the release's SHA256SUMS.
SHA256=4201588fc5051c2ba988abbe1f033d318965ee378fadf7fb7ef79882ba7be84b
curl -sfL "$URL" -o /tmp/pbs-python.tar.gz
echo "$SHA256  /tmp/pbs-python.tar.gz" | shasum -a 256 -c -
rm -rf runtime && mkdir -p runtime
tar -xzf /tmp/pbs-python.tar.gz -C runtime && mv runtime/python runtime/py
# The install_only archive has CPython's licence only. These are the licences
# of the libraries the build links, from the same release's full archive.
mkdir -p runtime/py/licenses && cp licenses/python/*.txt runtime/py/licenses/
runtime/py/bin/python3 -m pip install --require-hashes --no-warn-script-location -r ../engine/requirements-mac.txt
runtime/py/bin/python3 -c "import torch, transformers, fastapi, cv2, onnx, onnxruntime, pi_heif; print('runtime OK')"
