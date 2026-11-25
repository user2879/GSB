#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON:-.venv/bin/python}"
"${PYTHON_BIN}" -m tile_anomaly run --config sample/config/detection.yaml --reset --json
