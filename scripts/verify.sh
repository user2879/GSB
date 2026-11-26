#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p artifacts/verification

source .venv/bin/activate

pytest -q 2>&1 | tee artifacts/verification/pytest-latest.txt
python -m waterlease.demo 2>&1 | tee artifacts/verification/demo-latest.txt
python -m waterlease.cli \
  --spec-dir artifacts/spec-v2 \
  --ledger ledger/demo.sqlite3 \
  replay 2>&1 | tee artifacts/verification/replay-latest.txt

echo "Verification complete. See artifacts/demo-report.json and artifacts/verification/."
