#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

"${PYTHON_BIN}" -m feature_diag validate --config sample/experiment.yaml
"${PYTHON_BIN}" -m feature_diag run \
  --config sample/experiment.yaml \
  --out artifacts/reports/diagnostic-report \
  --cache-dir artifacts/runtime-final-cache \
  --snapshot-dir artifacts/runtime-final-snapshot \
  --max-workers 4
"${PYTHON_BIN}" -m feature_diag run \
  --config sample/experiment.yaml \
  --out artifacts/reports/diagnostic-report-resumed \
  --cache-dir artifacts/runtime-final-cache \
  --snapshot-dir artifacts/runtime-final-snapshot \
  --max-workers 4
