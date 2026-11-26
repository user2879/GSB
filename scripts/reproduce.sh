#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ ! -x .venv/bin/python ]; then
  python3.11 -m venv --without-pip .venv || python3 -m venv --without-pip .venv
fi
if ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
  curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/tcp-observer-get-pip.py
  .venv/bin/python /tmp/tcp-observer-get-pip.py --quiet
fi
.venv/bin/python -m pip install --quiet 'numpy==1.26.4'
.venv/bin/python -m pip install --quiet -e .

PYTHONPATH="$ROOT/src" .venv/bin/python -m unittest discover -s tests -v

PYTHONPATH="$ROOT/src" .venv/bin/tcp-observer all --out artifacts/verify-a
PYTHONPATH="$ROOT/src" .venv/bin/tcp-observer all --out artifacts/verify-b

for file in \
  summary.json \
  conclusions.json \
  ledger_core.jsonl \
  ledger.sqlite \
  normalized_receipts.json \
  edit_recompute.json \
  scheduler_demo.json \
  snapshot_replacement.json \
  recovery/recovery_report.json \
  recovery/recovered_ledger_core.jsonl; do
  cmp -s "artifacts/verify-a/$file" "artifacts/verify-b/$file"
  echo "BYTE-IDENTICAL $file"
done

echo "REPRODUCE_OK summary=$(sha256sum artifacts/verify-a/summary.json | cut -d' ' -f1)"
echo "REPRODUCE_OK ledger=$(sha256sum artifacts/verify-a/ledger_core.jsonl | cut -d' ' -f1)"
