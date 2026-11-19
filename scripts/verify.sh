#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv-local/bin/python}"

"${PYTHON_BIN}" scripts/generate_samples.py
"${PYTHON_BIN}" -m unittest discover -s tests -v
"${PYTHON_BIN}" scripts/run_demo.py
