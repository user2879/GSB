#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/vendor/tk}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

cd "$WORK"
apt-get download libtcl8.6 libtk8.6
for deb in ./*.deb; do
  dpkg-deb -x "$deb" root
done
mkdir -p "$OUT"
cp -a root/. "$OUT/"

echo "Tk/Tcl libraries extracted to $OUT"
echo "The mdr Tk bootstrap loads them automatically; for a raw shell use:"
echo "  export LD_LIBRARY_PATH=$OUT/usr/lib/x86_64-linux-gnu:\$LD_LIBRARY_PATH"
echo "  export TCL_LIBRARY=$OUT/usr/share/tcltk/tcl8.6"
echo "  export TK_LIBRARY=$OUT/usr/share/tcltk/tk8.6"
