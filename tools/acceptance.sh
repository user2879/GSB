#!/usr/bin/env bash
# End-to-end acceptance run against the local deterministic sample.
set -euo pipefail
cd "$(dirname "$0")/.."

python3 tools/make_sample.py
python3 - <<'PY'
import os
for path in ("out/ledger.db", "out/snapshot.json", "out/crash.db"):
    if os.path.exists(path):
        os.remove(path)
PY

echo "=== 1. run (rate limit + auth + real loopback forwarding) ==="
python3 -m workbench.cli run

echo "=== 2. quick view (incremental cache, one read) ==="
python3 -m workbench.cli view | head -20

echo "=== 3. authoritative recompute + dual-path check ==="
python3 -m workbench.cli recompute

echo "=== 4. snapshot save / open + replay verification ==="
python3 -m workbench.cli snapshot-save
python3 -m workbench.cli snapshot-open

echo "=== 5. forced-kill recovery ==="
python3 -m tests.crash_worker out/crash.db || true
python3 -m workbench.cli --ledger out/crash.db recover

echo "=== 6. unit tests ==="
python3 -m unittest discover -s tests
