#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -x .venv/bin/python ]]; then
  if python3 -m venv .venv >/tmp/shardmesh-venv.log 2>&1; then
    :
  else
    echo "python3 -m venv unavailable; falling back to virtualenv." >&2
    virtualenv -p python3 .venv
  fi
fi

# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

shardmesh init --dir artifacts/demo
python -m unittest discover -s tests
shardmesh run --dir artifacts/demo
shardmesh verify --dir artifacts/verification --dynamic-ports
shardmesh chaos --dir artifacts/chaos

echo "Demo complete. See artifacts/demo/current-view.json and artifacts/verification/verification-report.json."
