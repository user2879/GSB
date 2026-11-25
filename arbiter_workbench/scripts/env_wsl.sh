#!/usr/bin/env bash
# WSL environment bootstrap for Tkinter when system python3-tk cannot be
# installed via apt (no sudo). Pre-staged Tcl/Tk libraries are wired in.
set -euo pipefail

TK_LIBS=/home/zsl/tklibs/usr/lib/x86_64-linux-gnu
TKROOT_LIBS=/home/zsl/.local/tkroot/usr/lib/x86_64-linux-gnu
export LD_LIBRARY_PATH="${TK_LIBS}:${TKROOT_LIBS}:${LD_LIBRARY_PATH:-}"
export TCL_LIBRARY=/home/zsl/tklibs/usr/share/tcltk/tcl8.6
export TK_LIBRARY=/home/zsl/tklibs/usr/share/tcltk/tk8.6

exec "$@"
