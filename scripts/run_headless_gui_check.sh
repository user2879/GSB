#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -n "${DISPLAY:-}" ]]; then
  ./scripts/tk_python.sh scripts/gui_smoke.py
elif command -v xvfb-run >/dev/null 2>&1; then
  xvfb-run -a ./scripts/tk_python.sh scripts/gui_smoke.py
elif [[ -x .runtime/tkroot/usr/bin/xvfb-run ]]; then
  PATH="$ROOT/.runtime/tkroot/usr/bin:$PATH" .runtime/tkroot/usr/bin/xvfb-run -a ./scripts/tk_python.sh scripts/gui_smoke.py
else
  echo "Xvfb/xvfb-run not found. Install xvfb or set DISPLAY to a usable X server." >&2
  exit 2
fi
