#!/usr/bin/env bash
# Reproduce the local environment (Python 3.11 venv + NumPy, stdlib sqlite3).
set -euo pipefail
cd "$(dirname "$0")/.."
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
.venv/bin/python -c "import sys, numpy, sqlite3; print(sys.version); print('numpy', numpy.__version__); print('sqlite', sqlite3.sqlite_version)"
