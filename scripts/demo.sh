#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LEDGER=".runtime/demo.sqlite"
mkdir -p .runtime samples
.venv/bin/python -c "from pathlib import Path; [p.unlink() for p in Path('.runtime').glob('demo.sqlite*')]"

.venv/bin/python -m shard_diag.cli doctor
.venv/bin/python -m shard_diag.cli validate --spec examples/cluster.json --events examples/events.json
.venv/bin/python -m shard_diag.cli ports --spec examples/cluster.json
.venv/bin/python -m shard_diag.cli samples --spec examples/cluster.json --out samples
.venv/bin/python -m shard_diag.cli run \
  --spec examples/cluster.json \
  --events examples/events.json \
  --ledger "$LEDGER" \
  --output .runtime/demo-run.json
.venv/bin/python -m shard_diag.cli verify \
  --spec examples/cluster.json \
  --events examples/events.json \
  --ledger "$LEDGER"
.venv/bin/python -m shard_diag.cli recompute \
  --spec examples/cluster.json \
  --old-events examples/events.json \
  --new-events examples/events_priority_edit.json \
  --ledger "$LEDGER" \
  --output .runtime/demo-incremental.json
.venv/bin/python -m unittest discover -s tests

echo "DEMO_OK ledger=$LEDGER report=.runtime/demo-run.json"
