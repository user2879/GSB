#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
if [[ ! -x "$PY" ]]; then
  echo "missing .venv; run: python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt" >&2
  exit 2
fi

"$PY" -m pip install -r requirements.txt
"$PY" -m compileall -q edge_shard_diag tests scripts/generate_demo.py
"$PY" -m unittest discover -s tests -v
"$PY" -m edge_shard_diag acceptance
"$PY" -m edge_shard_diag receipts --quiet

echo "ACCEPTANCE_OK report=artifacts/validation-report.json"
