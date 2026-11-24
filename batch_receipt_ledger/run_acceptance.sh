#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 -m pip install -r requirements.txt
python3 -m pytest tests -v
python3 -m batch_ledger.cli demo \
  --sample-dir samples/acceptance \
  --out artifacts/acceptance
