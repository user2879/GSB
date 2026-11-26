#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT_DIR/.sysdeps/debs" "$ROOT_DIR/.sysdeps/root"
cd "$ROOT_DIR/.sysdeps/debs"
apt-get download libtcl8.6 libtk8.6 libxft2 libxss1
for package in ./*.deb; do
  dpkg-deb -x "$package" "$ROOT_DIR/.sysdeps/root"
done
echo "Local Tk libraries extracted under .sysdeps/root; doctor preloads them automatically."
