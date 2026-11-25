#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -d .venv ]]; then
  if ! python3 -m venv .venv 2>/tmp/fixed_render_venv.err; then
    echo "python3 -m venv failed; trying userland virtualenv..." >&2
    cat /tmp/fixed_render_venv.err >&2
    virtualenv -p python3 .venv
  fi
fi

# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
pip install numpy pillow
python -m fixed_render.cli init-samples --root .

echo "Setup complete. Run: .venv/bin/python -m fixed_render.cli verify --root ."
