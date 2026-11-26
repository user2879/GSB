#!/usr/bin/env bash
# Rootless WSL fallback when `sudo apt install python3-tk` is unavailable.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p .local-debs .local-tk
cd .local-debs
apt-get download python3-tk libtk8.6 libtcl8.6 tk8.6 tcl8.6 libxft2 libxss1
for package in ./*.deb; do
  dpkg-deb -x "$package" ../.local-tk
done
echo "Tk userland extracted to .local-tk; rope_lab.tk_bootstrap enables it automatically."
