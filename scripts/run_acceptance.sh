#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p artifacts/reports artifacts/ledger artifacts/logs artifacts/crash-demo/logs

echo "[1/8] unit and integration tests"
python3 -m unittest discover -s tests -v 2>&1 | tee artifacts/reports/test-results.txt

echo "[2/8] specification validation"
python3 -m sharddiag.cli validate --spec config/spec.sample.json

echo "[3/8] loopback port check"
python3 -m sharddiag.cli ports --spec config/spec.sample.json

echo "[4/8] real TCP scheduler run"
python3 -m sharddiag.cli run \
  --spec config/spec.sample.json \
  --db artifacts/ledger/sample.sqlite3 \
  --report artifacts/reports/run-report.json \
  --basis-out artifacts/reports/derivations.json \
  --snapshot artifacts/ledger/snapshot.json \
  --log-dir artifacts/logs \
  --fresh

echo "[5/8] invalid subset recomputation"
python3 -m sharddiag.cli invalidate \
  --spec config/spec.sample.json \
  --out artifacts/reports/invalidation.json

echo "[6/8] capacity evolution and lease generation"
python3 -m sharddiag.cli evolve-demo \
  --spec config/spec.sample.json \
  --db artifacts/ledger/evolution.sqlite3 \
  --out artifacts/reports/evolution.json \
  --log-dir artifacts/logs/evolution \
  --fresh

echo "[7/8] cancellation/freeze scenario (CLI exit 10 is expected)"
python3 -m sharddiag.cli run \
  --spec config/spec.cancel.json \
  --db artifacts/ledger/cancel.sqlite3 \
  --report artifacts/reports/cancel-report.json \
  --snapshot artifacts/ledger/cancel-snapshot.json \
  --log-dir artifacts/logs/cancel \
  --fresh || test $? -eq 10

echo "[8/8] SIGKILL replay demonstration"
python3 -m sharddiag.cli run \
  --spec config/spec.sample.json \
  --db artifacts/crash-demo/crash.sqlite3 \
  --log-dir artifacts/crash-demo/logs \
  --fresh \
  --kill-after-terminal 1 || test $? -eq 137
sleep 1
python3 -m sharddiag.cli run \
  --spec config/spec.sample.json \
  --db artifacts/crash-demo/crash.sqlite3 \
  --report artifacts/crash-demo/recovered-report.json \
  --basis-out artifacts/crash-demo/recovered-derivations.json \
  --snapshot artifacts/crash-demo/recovered-snapshot.json \
  --log-dir artifacts/crash-demo/logs

python3 scripts/collect_acceptance.py > artifacts/reports/acceptance-summary.stdout.json
cp artifacts/reports/acceptance-summary.stdout.json artifacts/reports/acceptance-summary.json
echo "Acceptance summary: artifacts/reports/acceptance-summary.json"
