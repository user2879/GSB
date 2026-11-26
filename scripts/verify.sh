#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p artifacts

REPORT="artifacts/verification-results.txt"
{
  echo "== time =="
  date -Is
  echo "== kernel / wsl =="
  uname -a
  if [ -r /etc/os-release ]; then . /etc/os-release; echo "distro=${PRETTY_NAME}"; fi
  echo "== workdir =="
  pwd
  test -w . && echo "workdir_read_write=yes"
  echo "== python =="
  python3 --version
  echo "== dependency install/check =="
  python3 -m pip install numpy==1.26.4
  python3 - <<'PY'
import numpy
import sqlite3
print("numpy", numpy.__version__)
print("sqlite3", sqlite3.sqlite_version)
PY
  echo "== environment and ports =="
  python3 -m shard_diag.cli check-env --spec config/spec.json
  echo "== pytest =="
  python3 -m pytest -q
  echo "== deterministic samples =="
  python3 -m shard_diag.cli materialize-samples \
    --spec config/spec.json \
    --out samples/expected_summaries.json
  echo "== end-to-end loopback dispatch / sqlite replay =="
  python3 -m shard_diag.cli run \
    --spec config/spec.json \
    --db artifacts/demo-ledger.sqlite3 \
    --reset \
    --report artifacts/run-report.json
  echo "== invalidation sample =="
  python3 -m shard_diag.cli invalidation-plan \
    --spec config/spec.json \
    --change '{"type":"capacity_version","node_id":"node-a","new_version":4}'
  python3 -m shard_diag.cli invalidation-plan \
    --spec config/spec.json \
    --change '{"type":"resource_units","shard_id":"s1-extract-a","new_value":9}'
  echo "== verification complete =="
} 2>&1 | tee "$REPORT"

chmod +x scripts/verify.sh
