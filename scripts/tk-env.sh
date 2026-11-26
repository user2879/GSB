#!/usr/bin/env bash
# Enable the user-local Tk runtime used on this WSL Ubuntu image when
# /usr/lib/x86_64-linux-gnu/libtk8.6.so is not installed system-wide.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -z "${TK_ROOT:-}" ]]; then
  for candidate in \
    "$ROOT/.tk-runtime" \
    "$HOME/.local/tkroot" \
    "/home/zsl/.local/tkroot"; do
    if [[ -d "$candidate/usr/lib" ]]; then
      TK_ROOT="$candidate"
      break
    fi
  done
fi

if [[ -n "${TK_ROOT:-}" && -d "$TK_ROOT" ]]; then
  PY_MM="$(python -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)"
  export LD_LIBRARY_PATH="$TK_ROOT/usr/lib/x86_64-linux-gnu:$TK_ROOT/usr/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  if [[ -n "$PY_MM" && -d "$TK_ROOT/usr/lib/$PY_MM" ]]; then
    export PYTHONPATH="$TK_ROOT/usr/lib/$PY_MM:$TK_ROOT/usr/lib/$PY_MM/lib-dynload${PYTHONPATH:+:$PYTHONPATH}"
  fi
  if [[ -d "$TK_ROOT/usr/share/tcltk/tcl8.6" ]]; then
    export TCL_LIBRARY="$TK_ROOT/usr/share/tcltk/tcl8.6"
  fi
  if [[ -d "$TK_ROOT/usr/share/tcltk/tk8.6" ]]; then
    export TK_LIBRARY="$TK_ROOT/usr/share/tcltk/tk8.6"
  fi
fi
