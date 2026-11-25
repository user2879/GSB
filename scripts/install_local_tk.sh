#!/usr/bin/env bash
set -euo pipefail

# No-root fallback for Ubuntu/WSL when python3-tk cannot be installed system-wide.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

cd "$WORK"
apt-get download libtk8.6 libtcl8.6 tk8.6 tcl8.6 libxft2 libxss1
mkdir -p "$ROOT/.local/tk"
for package in ./*.deb; do
  dpkg-deb -x "$package" "$ROOT/.local/tk"
done

echo "Local Tk installed under $ROOT/.local/tk"
echo "Verify with: $ROOT/.venv/bin/python -m evidence_diag --root $ROOT gui --smoke-test"
