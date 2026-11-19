#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=.
.venv/bin/python -m unittest discover -s tests
.venv/bin/python scripts/run_demo.py
.venv/bin/python scripts/crash_recovery_demo.py
.venv/bin/python scripts/error_chain_demo.py >/dev/null
.venv/bin/python scripts/reproducibility_check.py >/dev/null
echo "ALL ACCEPTANCE STEPS PASSED"
