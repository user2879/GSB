#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
$PY -m app.main generate
$PY -m app.main run "$@"
