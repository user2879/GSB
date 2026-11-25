#!/usr/bin/env bash

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export LD_LIBRARY_PATH="$ROOT/vendor/tk/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TCL_LIBRARY="$ROOT/vendor/tk/usr/share/tcltk/tcl8.6"
export TK_LIBRARY="$ROOT/vendor/tk/usr/share/tcltk/tk8.6"
echo "Tk vendor environment enabled: $ROOT/vendor/tk"
