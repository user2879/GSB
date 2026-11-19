#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

"${PYTHON_BIN}" -m compileall -q src tests
"${PYTHON_BIN}" -m feature_diag validate --config sample/experiment.yaml
"${PYTHON_BIN}" -m feature_diag validate --config sample/experiment_invalid.yaml
"${PYTHON_BIN}" -m pytest -q
bash scripts/run_demo.sh
