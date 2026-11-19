#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python3 -m pip install -r requirements.txt
python3 tools/generate_samples.py
python3 -m vecflow preflight --log-dir logs
python3 -m vecflow validate --log-dir logs
python3 -m vecflow run --reset-state --log-dir logs
python3 -m vecflow run --ledger artifacts/crash_ledger.sqlite --snapshot-dir artifacts/crash_snapshots --log-dir logs --reset-state --force-crash-after 2 || test $? -eq 137
python3 -m vecflow recover --ledger artifacts/crash_ledger.sqlite --log-dir logs
python3 -m vecflow replay --ledger artifacts/crash_ledger.sqlite --snapshot-dir artifacts/crash_snapshots --log-dir logs
python3 -m vecflow gui --log-dir logs
