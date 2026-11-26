#!/usr/bin/env bash
# 启用项目内置的本地 Tk 运行时（系统未安装 python3-tk 时使用）。
# 用法： source scripts/tk-env.sh
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$(cd "$DIR/.." && pwd)"

if [ -d "$ROOT/.sysroot/usr/lib/x86_64-linux-gnu" ]; then
  export LD_LIBRARY_PATH="$ROOT/.sysroot/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export TCL_LIBRARY="$ROOT/.sysroot/usr/share/tcltk/tcl8.6"
  export TK_LIBRARY="$ROOT/.sysroot/usr/share/tcltk/tk8.6"
fi
if [ -d "$ROOT/.tk-python" ]; then
  export PYTHONPATH="$ROOT/.tk-python${PYTHONPATH:+:$PYTHONPATH}"
fi
