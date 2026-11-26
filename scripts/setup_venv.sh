#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.11}"

if [[ ! -d .venv ]]; then
  if ! "$PYTHON_BIN" -m venv .venv; then
    echo "Standard venv bootstrap failed; falling back to --without-pip." >&2
    "$PYTHON_BIN" -m venv --without-pip .venv
  fi
fi

source .venv/bin/activate

if ! python -m pip --version >/dev/null 2>&1; then
  python -m pip install \
    --target .venv/lib/python3.11/site-packages \
    --upgrade 'pip==24.0'
fi

python -m pip install 'numpy==1.26.4' 'pytest==8.2.2'
python -c "import numpy, pytest; print('numpy', numpy.__version__); print('pytest', pytest.__version__)"
