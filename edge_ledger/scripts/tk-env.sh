#!/usr/bin/env bash
# Source this file after defining DIR to enable the local, non-pip Tk build.

if [ -d "$DIR/.sysroot/usr/lib/x86_64-linux-gnu" ]; then
  export LD_LIBRARY_PATH="$DIR/.sysroot/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export TCL_LIBRARY="$DIR/.sysroot/usr/share/tcltk/tcl8.6"
  export TK_LIBRARY="$DIR/.sysroot/usr/share/tcltk/tk8.6"
fi

if ! "$DIR/.venv/bin/python" -c 'import tkinter' >/dev/null 2>&1; then
  BASE_SITE="$(python3 -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null || true)"
  if [ -n "$BASE_SITE" ] && [ -d "$BASE_SITE/tkinter" ]; then
    mkdir -p "$DIR/.tk-python"
    ln -sfn "$BASE_SITE/tkinter" "$DIR/.tk-python/tkinter"
    for so in "$BASE_SITE"/_tkinter*.so; do
      [ -e "$so" ] || continue
      ln -sf "$so" "$DIR/.tk-python/$(basename "$so")"
    done
    export PYTHONPATH="$DIR/.tk-python${PYTHONPATH:+:$PYTHONPATH}"
  fi
fi
