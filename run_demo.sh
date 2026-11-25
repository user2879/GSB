#!/usr/bin/env bash
# Deterministic local end-to-end demonstration (no containers, loopback only).
set -euo pipefail
cd "$(dirname "$0")"

python scripts_make_samples.py
python scripts_make_receipts.py

echo "== unit + integration tests =="
python -m unittest discover -s tests -v

echo "== acceptance scenarios =="
python scripts_acceptance_demo.py

echo "== live gateway over subprocess backends =="
python -m grayroute.cli run --invalidate data/invalidate.json
