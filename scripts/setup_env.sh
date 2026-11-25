#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d .venv ]; then
  if command -v virtualenv >/dev/null 2>&1; then
    virtualenv .venv
  else
    python3 -m venv .venv
  fi
fi

# shellcheck disable=SC1091
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .

python - <<'PY'
import numpy
import pytest
print("numpy", numpy.__version__)
print("pytest", pytest.__version__)
try:
    import tkinter
    print("tkinter", tkinter.TkVersion)
except Exception as exc:
    print("tkinter unavailable:", exc)
    print("Ubuntu/WSL fix: sudo apt-get update && sudo apt-get install -y python3-tk")
PY
