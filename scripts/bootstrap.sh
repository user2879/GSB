#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python3 --version

if [ ! -d ".venv" ]; then
  if ! python3 -m venv .venv; then
    python3 -m venv --without-pip .venv
  fi
fi

if ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
  curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/shard-diag-get-pip.py
  .venv/bin/python /tmp/shard-diag-get-pip.py
fi

.venv/bin/python -m pip install 'numpy==1.26.4'
.venv/bin/python -m py_compile shard_diag/*.py
.venv/bin/python -m shard_diag.cli doctor
