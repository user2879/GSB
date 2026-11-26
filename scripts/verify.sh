#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -d .venv ]]; then
  PY=.venv/bin/python
else
  PY=python3
fi
"$PY" -m unittest discover -s tests
"$PY" -m rope_lab.cli verify --root .
"$PY" scripts/tk_smoke.py
