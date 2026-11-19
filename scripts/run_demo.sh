#!/usr/bin/env bash
set -euo pipefail
PY=.venv/bin/python
[ -x "$PY" ] || PY=python3
"$PY" -m dwb_diag.cli demo --root data/shell-demo
