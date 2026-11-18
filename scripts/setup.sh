#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3.11 -m venv .venv
if [[ "${1:-}" == "--offline" ]]; then
  .venv/bin/python -m pip install --no-index --find-links wheelhouse -r requirements.txt
else
  .venv/bin/python -m pip install --disable-pip-version-check -r requirements.txt
fi

printf 'Environment ready: %s\n' "$ROOT/.venv/bin/python"
