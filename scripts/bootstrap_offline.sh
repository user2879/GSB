#!/usr/bin/env bash
# Reproduce the local virtual environment without network access.
# Wheels live in ./wheelhouse and were repacked from the local package cache.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3.11 -m venv .venv || true
if ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
  .venv/bin/python /tmp/get-pip.py --no-index \
    --find-links /home/zsl/.feedback/runtime-home/42807608d2188fa8058c036e0b334f6603b24a8e77c29d84b3750c2706d217b2/.local/lib/python3.11/site-packages/virtualenv/seed/wheels/embed \
    --find-links /home/zsl/.feedback/runtime-home/b7696e2fb2bad00f9f3b4c78520a607ad7163a0367d3ded913269af7da5e83dd/.local/lib/python3.11/site-packages/virtualenv/seed/wheels/embed
fi
.venv/bin/python -m pip install --no-index --find-links wheelhouse numpy Pillow pytest
.venv/bin/python -c "import numpy, PIL, pytest; print(numpy.__version__, PIL.__version__, pytest.__version__)"
