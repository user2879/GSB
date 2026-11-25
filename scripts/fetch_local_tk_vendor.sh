#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

mkdir -p "$ROOT/vendor/tk"
cd "$WORK"
apt-get download libtcl8.6 libtk8.6 tcl8.6 tk8.6 libxft2 libxss1
for package in ./*.deb; do
  dpkg-deb -x "$package" "$ROOT/vendor/tk"
done

echo "Tk vendor libraries installed under $ROOT/vendor/tk"
echo "Run: source scripts/tk_vendor_env.sh"
