#!/usr/bin/env bash
set -euo pipefail

test -r . && test -w .

if [[ -d ".venv" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

python --version
python -m compileall -q main.py gsb tests
python -m unittest discover -s tests -v
python -m gsb.cli verify
