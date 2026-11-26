#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="$ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "missing $PYTHON; create the virtualenv and install numpy/pytest first" >&2
  exit 2
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="$ROOT/artifacts/validation/$STAMP"
mkdir -p "$OUT"

echo "== versions =="
"$PYTHON" - <<'PY' | tee "$OUT/versions.txt"
import sys
import numpy
import pytest
print(sys.version)
print("numpy", numpy.__version__)
print("pytest", pytest.__version__)
PY

echo "== pytest =="
"$PYTHON" -m pytest 2>&1 | tee "$OUT/pytest.txt"

echo "== validate specs =="
"$PYTHON" -m cacheflow validate 2>&1 | tee "$OUT/validate.json"

echo "== end-to-end acceptance with SIGKILL =="
"$PYTHON" -m cacheflow accept --work-dir "$OUT/acceptance" 2>&1 | tee "$OUT/acceptance-cli.json"

W="$OUT/acceptance"
echo "== independent read-only replay =="
"$PYTHON" -m cacheflow replay \
  --nodes "$W/nodes.json" \
  --artifacts "$W/artifacts.json" \
  --events "$W/events.json" \
  --ledger "$W/cacheflow.sqlite3" \
  --out "$OUT/post-kill-replay.canonical.json" 2>&1 | tee "$OUT/replay.json"
cmp "$W/post-kill-replay.canonical.json" "$OUT/post-kill-replay.canonical.json"

echo "== quota evolution incremental/full comparison =="
"$PYTHON" -m cacheflow evolve \
  --evolution examples/evolution.json \
  --cutoff-seq 20 \
  --out-dir "$OUT/evolution-quota" 2>&1 | tee "$OUT/evolution-quota.json"

echo "== node-removal evolution subset =="
"$PYTHON" -m cacheflow evolve \
  --evolution examples/evolution-node-removal.json \
  --cutoff-seq 24 \
  --out-dir "$OUT/evolution-node" 2>&1 | tee "$OUT/evolution-node.json"

echo "$OUT" > "$ROOT/artifacts/validation/LATEST"
echo "VALIDATION_OUTPUT=$OUT"
