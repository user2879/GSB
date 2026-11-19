#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=src
.venv/bin/python scripts/generate_samples.py
.venv/bin/python -m rexapp full samples output
.venv/bin/python -m rexapp interactive samples output
.venv/bin/python scripts/acceptance_check.py
.venv/bin/python -m unittest discover -s tests
