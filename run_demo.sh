#!/usr/bin/env bash
# Reproducible end-to-end demo. Works on Linux/macOS; on Windows use the
# individual `python -m aedflow ...` commands shown in docs_technical.md.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=.
PY="${PYTHON:-$(command -v python || command -v python3)}"
"$PY" -m aedflow generate
"$PY" -m aedflow run
"$PY" -m aedflow drill
"$PY" -m aedflow stale-drill
"$PY" -m unittest tests.test_pipeline
