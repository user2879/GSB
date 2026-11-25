#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ ! -x ".venv/bin/python" ]; then
  if command -v virtualenv >/dev/null 2>&1; then
    virtualenv .venv
  else
    python3 -m venv .venv
  fi
fi

# Debian/Ubuntu images may lack ensurepip for python3.11.
if ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
  if command -v virtualenv >/dev/null 2>&1; then
    virtualenv .venv
  else
    echo "Run: sudo apt install python3.11-venv python3-tk" >&2
    exit 2
  fi
fi

. .venv/bin/activate
python -m pip install -e '.[test]'

cloudshard demo --workdir artifacts/demo
cloudshard verify \
  --spec artifacts/demo/declaration.generated.json \
  --ledger artifacts/demo/ledger.sqlite
cloudshard replay \
  --spec artifacts/demo/declaration.generated.json \
  --out artifacts/demo/replay.json \
  --batch-check
cloudshard acceptance --workdir artifacts/acceptance
pytest -q

echo "Demo and acceptance artifacts are under artifacts/demo and artifacts/acceptance."
