#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TKROOT="${TKROOT:-$ROOT/.runtime/tkroot}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"

if [[ -d "$TKROOT/usr/lib/x86_64-linux-gnu" ]]; then
  export LD_LIBRARY_PATH="$TKROOT/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export TCL_LIBRARY="$TKROOT/usr/share/tcltk/tcl8.6"
  export TK_LIBRARY="$TKROOT/usr/share/tcltk/tk8.6"
fi

exec "$PYTHON_BIN" "$@"
