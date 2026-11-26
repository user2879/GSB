#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="/home/zsl/.feedback/python-base/bin/python3.11"
fi

if [ ! -x ".venv/bin/python" ]; then
  if "$PYTHON_BIN" -c 'import ensurepip' >/dev/null 2>&1; then
    "$PYTHON_BIN" -m venv .venv
  else
    virtualenv -p "$PYTHON_BIN" .venv
  fi
fi

# pip may use its local cache/wheel mirror; no runtime network access is used by quota_lab.
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python - <<'PY'
import sys
import numpy as np
assert sys.version_info[:2] == (3, 11), sys.version
assert np.__version__ == "1.26.4", np.__version__
print(f"environment-ok python={sys.version.split()[0]} numpy={np.__version__}")
PY
