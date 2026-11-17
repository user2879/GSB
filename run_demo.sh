#!/usr/bin/env bash
# Deterministic local end-to-end demonstration (no containers).
set -euo pipefail

cd "$(dirname "$0")"
PY=.venv/bin/python

echo "== 1. regenerate deterministic samples =="
"$PY" tools/generate_samples.py --out samples

echo "== 2. reset local work directory =="
"$PY" -c "import shutil; shutil.rmtree('artifacts/run', ignore_errors=True)"

echo "== 3. first pipeline run (full compute) =="
"$PY" tools/run_pipeline.py --work-dir artifacts/run > artifacts/first_run.json

echo "== 4. second run (incremental reuse) =="
"$PY" tools/run_pipeline.py --work-dir artifacts/run > artifacts/second_run.json

echo "== 5. business error code fixtures =="
"$PY" tools/validate_error_codes.py

echo "== 6. unit and integration tests =="
"$PY" -m unittest discover -s tests

echo "demo complete; see artifacts/run/snapshot_store/snapshots/"
