#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

TKLOCAL="$HOME/.local/tk-local/usr/lib/x86_64-linux-gnu"
if [[ -d "$TKLOCAL" ]]; then
  export LD_LIBRARY_PATH="$TKLOCAL${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export TCL_LIBRARY="$HOME/.local/tk-local/usr/share/tcltk/tcl8.6"
  export TK_LIBRARY="$HOME/.local/tk-local/usr/share/tcltk/tk8.6"
fi

exec .venv/bin/python -m vertical_workbench.cli verify --root "$ROOT"
