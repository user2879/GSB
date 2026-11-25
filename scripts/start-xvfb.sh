#!/bin/bash
# 在用户命名空间内启动本地 Xvfb（显示号 :99），抽象 socket 供宿主侧 Tk 连接
set -e
source "$(dirname "$0")/env.sh"
WORK="$PROJECT_ROOT/.runtime/xvfb"
mkdir -p "$WORK"
if pgrep -f "Xvfb :99" >/dev/null; then
  echo "Xvfb :99 already running (pid $(pgrep -f 'Xvfb :99' | head -1))"
  exit 0
fi
cat > "$WORK/start-xvfb-ns.sh" <<INNER
#!/bin/bash
set -e
WORK=$WORK
TKROOT=$TKROOT
DEPS=$DEPS
XKBLIB=$PROJECT_ROOT/.runtime/xkbfiles/usr/lib/x86_64-linux-gnu
mkdir -p \$WORK/newroot
mount --rbind / \$WORK/newroot
mount --make-rslave \$WORK/newroot 2>/dev/null || true
mount -t tmpfs tmpfs \$WORK/newroot/usr/bin
for b in env sh bash cat ls; do cp /usr/bin/\$b \$WORK/newroot/usr/bin/\$b; done
cp \$TKROOT/usr/bin/xkbcomp \$WORK/newroot/usr/bin/xkbcomp
chmod 755 \$WORK/newroot/usr/bin/*
mount -t tmpfs tmpfs \$WORK/newroot/tmp
chmod 1777 \$WORK/newroot/tmp
mkdir -p \$WORK/newroot/tmp/.X11-unix \$WORK/newroot/tmp/putold
chmod 1777 \$WORK/newroot/tmp/.X11-unix
mount -t tmpfs tmpfs \$WORK/newroot/tmp/putold
cd \$WORK/newroot
pivot_root . tmp/putold
cd /
umount -l /tmp/putold 2>/dev/null || true
export LD_LIBRARY_PATH=\$XKBLIB:\$TKROOT/usr/lib/x86_64-linux-gnu:\$TKROOT/usr/lib:\$DEPS/usr/lib/x86_64-linux-gnu
export XKB_CONFIG_ROOT=/usr/share/X11/xkb
exec \$TKROOT/usr/bin/Xvfb :99 -screen 0 1280x800x24 -ac -nolisten tcp
INNER
unshare -Urmpf bash "$WORK/start-xvfb-ns.sh" >"$WORK/xvfb.log" 2>&1 &
for _ in $(seq 1 30); do
  if grep -q "abstract connective" "$WORK/xvfb.log" 2>/dev/null || pgrep -f "Xvfb :99" >/dev/null; then
    sleep 0.5; pgrep -f "Xvfb :99" >/dev/null && break
  fi
  sleep 0.3
done
pgrep -f "Xvfb :99" >/dev/null && echo "Xvfb :99 started (pid $(pgrep -f 'Xvfb :99' | head -1))" || { echo "Xvfb failed"; cat "$WORK/xvfb.log"; exit 1; }
