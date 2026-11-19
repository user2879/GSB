#!/usr/bin/env bash
# Linux/WSL acceptance run mirroring scripts/acceptance.ps1.
set -euo pipefail
cd "$(dirname "$0")/.."
export QT_QPA_PLATFORM=offscreen
export PYTHONPATH=src
PY=.venv/bin/python
$PY scripts/check_gl_context.py
$PY -m pcrecon generate
$PY -m pcrecon run --undo
$PY -m pcrecon crash-demo
$PY -m pcrecon hash-verify
$PY -m unittest discover -s tests
