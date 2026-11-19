#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "Missing .venv. See README.md for environment setup." >&2
  exit 1
fi
exec .venv/bin/python -m tactical_path demo --config config/demo.yaml --workspace workspace/demo

