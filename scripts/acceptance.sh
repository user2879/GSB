#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="$ROOT/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  PY="$(command -v python3)"
fi

OUT="${1:-$ROOT/artifacts/acceptance-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT"
# The fixed OUT path is an acceptance workspace; reset only its top-level run DB.
rm -f "$OUT/ledger.db" "$OUT/ledger.db-wal" "$OUT/ledger.db-shm" "$OUT/snapshot.json"

echo "== python/numpy =="
"$PY" --version
"$PY" -c 'import numpy; print("numpy", numpy.__version__)'

echo "== unit and integration tests =="
"$PY" -m unittest discover -s tests -v 2>&1 | tee "$OUT/unittest.txt"

echo "== spec and port preflight =="
"$PY" -m gsb validate samples/local_spec.json | tee "$OUT/validate.json"
"$PY" -m gsb ports samples/local_spec.json | tee "$OUT/ports.json"

echo "== fixed-point attribution =="
"$PY" -m gsb fixedpoint-report | tee "$OUT/fixedpoint.json"

echo "== real subprocess/TCP run =="
"$PY" -m gsb run samples/local_spec.json \
  --db "$OUT/ledger.db" \
  --snapshot "$OUT/snapshot.json" \
  --log-dir "$OUT/node-logs" | tee "$OUT/run.json"

echo "== independent replay verification =="
"$PY" -m gsb verify --spec samples/local_spec.json --db "$OUT/ledger.db" | tee "$OUT/verify.json"

echo "== invalidation and incremental/full consistency =="
"$PY" -m gsb evolve \
  --old samples/local_spec.json \
  --new samples/evolved_capacity_spec.json \
  --workdir "$OUT/evolution" | tee "$OUT/evolve.json"

echo "ACCEPTANCE_ARTIFACTS=$OUT"
