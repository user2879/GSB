#!/usr/bin/env bash
# Reproducible local demo: no Docker, no external network, no accounts.
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
if [[ ! -x "$PY" ]]; then
  echo "virtual environment missing; create it with:" >&2
  echo "  python3 -m venv .venv && .venv/bin/pip install numpy" >&2
  exit 1
fi

# Headless Tk display: prefer an existing Xvfb :99, then :0, else skip UI smoke.
if [[ -z "${DISPLAY:-}" ]]; then
  if [[ -e /tmp/.X11-unix/X99 ]]; then export DISPLAY=:99; fi
fi

echo "== dependency check =="
"$PY" -c "import numpy, sqlite3, tkinter; print('numpy/sqlite3/tkinter ok')"

echo "== unit tests =="
"$PY" -m unittest discover -s tests -v

echo "== end-to-end demo (real node subprocesses + forced termination) =="
"$PY" -m shardmeta.demo \
  --samples samples \
  --ledger samples/shardmeta-ledger.sqlite3 \
  --report samples/demo-report.json

echo "== report written to samples/demo-report.json =="
