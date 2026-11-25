#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -x .venv/bin/python || ! -x .venv/bin/pip ]]; then
  if ! python3.11 -m venv --system-site-packages .venv; then
    # Some WSL Python distributions omit ensurepip.  The local virtualenv
    # seeder is offline and already distributed with this environment.
    python3.11 - <<'PY'
import shutil
shutil.rmtree('.venv', ignore_errors=True)
PY
    virtualenv --system-site-packages .venv
  fi
fi

.venv/bin/python - <<'PY'
import numpy
import pytest
print(f"numpy={numpy.__version__} pytest={pytest.__version__}")
PY

.venv/bin/python -m pip install --no-build-isolation --no-deps -e .
printf 'environment ready: %s/.venv\n' "$ROOT"
