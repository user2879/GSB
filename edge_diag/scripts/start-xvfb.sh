#!/usr/bin/env bash
# 无头环境下在用户命名空间内启动本地 Xvfb（显示号 :77）。
# 已有可用 DISPLAY 时无需运行本脚本。
set -euo pipefail
TKROOT="${TKROOT:-/home/zsl/.local/tkroot}"
XKBLIB="${XKBLIB:-/home/zsl/.feedback/codex-input/06d801aa08ea8673/311/delivery-B/project/.runtime/xkbfiles/usr/lib/x86_64-linux-gnu}"
WORK="$(mktemp -d)"
mkdir -p "$WORK/upper" "$WORK/work"
cp "$TKROOT/usr/bin/xkbcomp" "$WORK/upper/xkbcomp"

unshare -r -m bash -c '
  set -e
  mount -t tmpfs tmpfs /tmp
  mkdir -p /tmp/.X11-unix && chmod 1777 /tmp/.X11-unix
  mount -t overlay overlay -o lowerdir=/usr/bin,upperdir='"$WORK"'/upper,workdir='"$WORK"'/work /usr/bin
  export LD_LIBRARY_PATH='"$XKBLIB"':'"$TKROOT"'/usr/lib/x86_64-linux-gnu
  exec '"$TKROOT"'/usr/bin/Xvfb :77 -screen 0 1280x800x24 -ac
' &
XPID=$!
sleep 2
echo "Xvfb :77 已启动 (pid $XPID)"
echo "连接方式: DISPLAY=unix/:77 （经抽象套接字）"
