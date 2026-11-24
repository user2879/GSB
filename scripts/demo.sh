#!/usr/bin/env bash
# Repeatable local demo. Deterministic samples only; no network services.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p artifacts run
$PY -m batchdiag.cli gen-samples --samples-dir samples

echo "== 1. strict spec validation (E_SPEC carries field + line)"
$PY -m batchdiag.cli validate --requests samples/requests.json --workers samples/workers.json --rules samples/merge_rules.json

echo "== 2. normalize r1/r2/r3 onto one reviewable ledger shape"
for fmt in r1 r2 r3; do
  $PY -m batchdiag.cli normalize --requests samples/requests.json --workers samples/workers.json --rules samples/merge_rules.json --receipts samples/receipts_$fmt.json > run/normalize_$fmt.json
done

echo "== 3. deterministic plan and full rebuild"
$PY -m batchdiag.cli plan --requests samples/requests.json --workers samples/workers.json --rules samples/merge_rules.json > run/plan.json
$PY -m batchdiag.cli run --requests samples/requests.json --workers samples/workers.json --rules samples/merge_rules.json --ledger artifacts/ledger.sqlite3

echo "== 4. invalidation boundary + incremental vs full (byte identical)"
$PY - <<'PYEOF'
from batchdiag.samples import generate_changed_workers
generate_changed_workers("samples", "samples/workers_v2.json",
                         bump_worker="w-bravo", new_version=2,
                         new_seed=20261125+200)
PYEOF
$PY -m batchdiag.cli incremental --requests samples/requests.json --workers samples/workers.json --new-workers samples/workers_v2.json --rules samples/merge_rules.json --ledger artifacts/ledger_inc.sqlite3 > run/incremental.json

echo "== 5. shuffled request stream rebuilds byte identically"
$PY -m batchdiag.cli verify --requests samples/requests.json --workers samples/workers.json --rules samples/merge_rules.json --ledger-a artifacts/a.sqlite3 --ledger-b artifacts/b.sqlite3

echo "== 6. real loopback workers, quota-constrained thread pool"
$PY -m batchdiag.cli scheduler --requests samples/requests.json --workers samples/workers.json --rules samples/merge_rules.json > run/scheduler.json

echo "== 7. real socket timeout reports E_TIMEOUT"
$PY -m batchdiag.cli scheduler --requests samples/requests.json --workers samples/workers.json --rules samples/merge_rules.json --timeout-demo --stall-ms 3000 > run/scheduler_timeout.json || true

echo "== 8. tampered snapshot is replaced wholesale, not merged"
$PY -m batchdiag.cli snapshot --requests samples/requests.json --workers samples/workers.json --rules samples/merge_rules.json --ledger run/snapshot.sqlite3 --tamper > run/snapshot.json

echo "== 9. forced kill: only committed batches trusted, unknown rerun"
set +e
$PY -m batchdiag.cli crash --requests samples/requests.json --workers samples/workers.json --rules samples/merge_rules.json --ledger run/crash.sqlite3 --after 2
echo "crash exit code: $? (137 = SIGKILL, expected)"
set -e
$PY -m batchdiag.cli recover --requests samples/requests.json --workers samples/workers.json --rules samples/merge_rules.json --ledger run/crash.sqlite3 > run/recover.json
$PY -m batchdiag.cli replay --requests samples/requests.json --workers samples/workers.json --rules samples/merge_rules.json --ledger run/crash.sqlite3

echo "== 10. optional Tk GUI degrades to CLI on headless hosts"
$PY -m batchdiag.cli gui --requests samples/requests.json --workers samples/workers.json --rules samples/merge_rules.json --ledger artifacts/ledger.sqlite3 || true

echo "ALL DEMO STEPS OK"
