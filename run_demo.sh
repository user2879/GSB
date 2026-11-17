#!/usr/bin/env bash
# Deterministic end-to-end demonstration (no network, workspace IO only).
set -euo pipefail

cd "$(dirname "$0")"
export PYTHONPATH="src:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
PY=python3.10

# Fresh directories every run so recovery state never masks the demonstration.
RUN_ID="run-$(date +%Y%m%d-%H%M%S)"
INCR="outputs/demo_incremental-${RUN_ID}"
FULL="outputs/demo_full-${RUN_ID}"
mkdir -p "$INCR" "$FULL"

echo "== [1/4] incremental batch run =="
$PY -m tsbench.cli run \
  --config inputs/experiment.yaml \
  --updates inputs/updates.json \
  --output "$INCR"

echo "== [2/4] one-shot full rebuild of the final state =="
$PY -m tsbench.cli full-final \
  --config inputs/experiment.yaml \
  --updates inputs/updates.json \
  --output "$FULL"

echo "== [3/4] field-by-field consistency (covers 2 forecast steps, all series) =="
$PY -m tsbench.cli verify \
  --full-report "$FULL/reports/report-batch-09999.json" \
  --incremental-report "$INCR/reports/report-batch-00003.json"

echo "== [4/4] recovery point =="
$PY -m tsbench.cli recovery \
  --config inputs/experiment.yaml \
  --output "$INCR"
