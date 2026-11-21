#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -z "${PYTHON:-}" && -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi
export PYTHONPATH="$ROOT/src"

"$PYTHON" -m ner_workbench generate --out samples --seed 20261121 --samples 32 >/tmp/ner-generate.json
"$PYTHON" -m pytest -q

"$PYTHON" -m ner_workbench run --samples samples --out runs/validation-workers1 --workers 1
"$PYTHON" -m ner_workbench run --samples samples --out runs/validation-workers4 --workers 4
"$PYTHON" -m ner_workbench compare \
  runs/validation-workers1/report.json runs/validation-workers4/report.json \
  --out runs/validation-comparison.json

"$PYTHON" -m ner_workbench run --samples samples --out runs/validation-resume --workers 1 --stop-after 12
"$PYTHON" -m ner_workbench run --samples samples --out runs/validation-resume --workers 1 --resume
"$PYTHON" -m ner_workbench compare \
  runs/validation-workers1/report.json runs/validation-resume/report.json \
  --out runs/validation-resume-comparison.json

printf '\nValidation complete. Reports: runs/validation-workers1/report.json and runs/validation-workers4/report.json\n'
