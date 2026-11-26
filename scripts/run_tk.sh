#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
if [[ -d ".sysroot/root/usr/lib/x86_64-linux-gnu" ]]; then
  export LD_LIBRARY_PATH="$PWD/.sysroot/root/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export TCL_LIBRARY="$PWD/.sysroot/root/usr/share/tcltk/tcl8.6"
  export TK_LIBRARY="$PWD/.sysroot/root/usr/share/tcltk/tk8.6"
fi
source .venv/bin/activate
exec python -m feedback_pe.cli ui
