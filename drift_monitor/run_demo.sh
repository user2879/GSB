#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PY="$ROOT/.venv/bin/python"
cd "$HERE"
PYTHONPATH=src "$PY" -m pytest tests -q
PYTHONPATH=src "$PY" -m drift_monitor.interactive --config configs/default_monitor.yaml \
  --tier small --output output/demo --command run
