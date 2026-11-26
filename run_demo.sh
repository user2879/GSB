#!/usr/bin/env bash
# Deterministic end-to-end demonstration, no network or containers.
set -euo pipefail
cd "$(dirname "$0")"
python3 - <<'PY'
import shutil, os
shutil.rmtree("snapshots", ignore_errors=True)
os.makedirs("snapshots", exist_ok=True)
PY
python3 -m vdu generate
python3 -m vdu run --tier coarse
echo "--- fine tier ---"
python3 -m vdu run --tier fine
