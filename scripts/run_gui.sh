#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Preferred system path: sudo apt install python3-tk.
# The project-local .tk-runtime fallback is discovered automatically.
if [[ -d "$HOME/.local/fixed-render-tk/usr/lib/x86_64-linux-gnu" ]]; then
  export FIXED_RENDER_TK_ROOT="${FIXED_RENDER_TK_ROOT:-$HOME/.local/fixed-render-tk}"
fi

exec .venv/bin/python -m fixed_render.cli gui \
  --report "${1:-artifacts/diagnostic_report.json}" \
  --smoke-ms "${TK_SMOKE_MS:-0}"
