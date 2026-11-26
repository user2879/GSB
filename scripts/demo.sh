#!/usr/bin/env bash
# Reproducible end-to-end demonstration of the GSB subscription pipeline.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONPATH=src

RUN=artifacts/runs/demo-final
CRASH=artifacts/runs/crash-demo

echo "== 1. generate deterministic sample =="
python -m gsb.cli gen --out samples/demo --events 48

echo "== 2. full run with real loopback subprocesses =="
python -m gsb.cli run \
  --spec samples/demo/spec.json \
  --events samples/demo/events.ndjson \
  --run-dir "$RUN" --demo-stale-renew

echo "== 3. verify planned batches against the committed ledger =="
python -m gsb.cli verify \
  --spec samples/demo/spec.json \
  --events samples/demo/events.ndjson \
  --run-dir "$RUN"

echo "== 4. forced termination (hard exit 137) then recovery replay =="
python -m gsb.cli run \
  --spec samples/demo/spec.json \
  --events samples/demo/events.ndjson \
  --run-dir "$CRASH" --crash-after 3 || true
python -m gsb.cli recover \
  --spec samples/demo/spec.json \
  --events samples/demo/events.ndjson \
  --run-dir "$CRASH"
