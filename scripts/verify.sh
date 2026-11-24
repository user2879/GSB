#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python -m pip install -r requirements.txt
python -m pytest tests/test_workbench.py -v

python -m workbench.cli init-sample --dir examples
python -m workbench.cli validate \
  --config examples/sample-config.json \
  --events examples/events.jsonl

set +e
python -m workbench.cli validate --config examples/invalid-input.json
invalid_exit=$?
set -e
if [[ "$invalid_exit" -ne 2 ]]; then
  echo "expected invalid-input validation exit code 2, got $invalid_exit" >&2
  exit 1
fi

python -m workbench.cli run \
  --config examples/sample-config.json \
  --events examples/events.jsonl \
  --out artifacts/demo \
  --parallel \
  --protocol-self-check

python -m workbench.cli run \
  --config examples/sample-config.json \
  --events examples/events.jsonl \
  --out artifacts/resume-demo \
  --max-events 16

python -m workbench.cli run \
  --config examples/sample-config.json \
  --events examples/events.jsonl \
  --out artifacts/resume-demo \
  --protocol-self-check

python - <<'PY'
from workbench.snapshot import SnapshotStore

full = SnapshotStore("artifacts/demo").read_current_report()
resumed = SnapshotStore("artifacts/resume-demo").read_current_report()
assert full == resumed, "resumed report differs from full report"
assert full["code_counts"]["E_CAPACITY_EXCEEDED"] == 1
assert full["code_counts"]["E_PROTOCOL_VIOLATION"] == 2
print("verification OK: resumed report equals full report")
PY
