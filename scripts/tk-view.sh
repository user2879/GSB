#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
LEDGER="${1:-artifacts/demo/ledger.sqlite}"
SNAPSHOT="${2:-artifacts/demo/snapshot.json}"

if [ -d "artifacts/tk-root/usr/lib/x86_64-linux-gnu" ]; then
  export LD_LIBRARY_PATH="$ROOT/artifacts/tk-root/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export TCL_LIBRARY="$ROOT/artifacts/tk-root/usr/share/tcltk/tcl8.6"
  export TK_LIBRARY="$ROOT/artifacts/tk-root/usr/share/tcltk/tk8.6"
fi

if [ -x ".venv/bin/python" ]; then
  if ! LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" TCL_LIBRARY="${TCL_LIBRARY:-}" TK_LIBRARY="${TK_LIBRARY:-}" \
      .venv/bin/python -c 'import tkinter' >/dev/null 2>&1; then
    FEEDBACK_SITE="/home/zsl/.feedback/python-base/lib/python3.11/site-packages"
    if [ -d "$FEEDBACK_SITE/tkinter" ]; then
      export PYTHONPATH="$FEEDBACK_SITE${PYTHONPATH:+:$PYTHONPATH}"
    fi
  fi
  exec .venv/bin/python -m cloudshard.tk_view --ledger "$LEDGER" --snapshot "$SNAPSHOT" "${@:3}"
fi
exec python -m cloudshard tk --ledger "$LEDGER" --snapshot "$SNAPSHOT"
