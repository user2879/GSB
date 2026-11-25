#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Virtual environment not found. Create it first:" >&2
  echo "  python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt" >&2
  exit 1
fi

"$PYTHON" "$ROOT/scripts/generate_sample.py" --root "$ROOT"
"$PYTHON" -m evidence_diag --root "$ROOT" run --mode full
"$PYTHON" -m evidence_diag --root "$ROOT" run --mode incremental
"$PYTHON" -m evidence_diag --root "$ROOT" diagnose
"$PYTHON" -m evidence_diag --root "$ROOT" demo
"$PYTHON" -m evidence_diag --root "$ROOT" check-env > "$ROOT/reports/environment.json"
"$PYTHON" -m evidence_diag --root "$ROOT" gui --smoke-test > "$ROOT/reports/gui_smoke.json"
"$PYTHON" -m unittest discover -v 2>&1 | tee "$ROOT/reports/tests.log"
