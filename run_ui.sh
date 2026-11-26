#!/usr/bin/env bash
# Tkinter 诊断界面启动脚本。
# 本机缺少系统级 libtk 时，使用仓库旁可用的 vendored Tk 库。
set -euo pipefail
cd "$(dirname "$0")"
VENDORED_TK=/home/zsl/feedback/uidfc5909d3-default-290/vendor/tk/usr/lib/x86_64-linux-gnu
VENDORED_TCL=/home/zsl/tklibs/usr/lib/x86_64-linux-gnu
VENDORED_TCL_LIB=/home/zsl/tklibs/usr/share/tcltk/tcl8.6
VENDORED_TK_LIB=/home/zsl/tklibs/usr/share/tcltk/tk8.6
if ! python3 -c "import tkinter; tkinter.Tk().destroy()" 2>/dev/null; then
  export LD_LIBRARY_PATH="${VENDORED_TK}:${VENDORED_TCL}:${LD_LIBRARY_PATH:-}"
  export TCL_LIBRARY="${VENDORED_TCL_LIB}"
  export TK_LIBRARY="${VENDORED_TK_LIB}"
fi
exec python3 -m src.ui "$@"
