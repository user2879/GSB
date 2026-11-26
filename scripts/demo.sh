#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# shellcheck source=scripts/tk-env.sh
source "$ROOT/scripts/tk-env.sh" || true

if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  virtualenv -p python3 "$ROOT/.venv"
fi

# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
python -m pip install -q --upgrade pip
python -m pip install -q 'numpy>=1.24' pytest
python -m pip install -q -e .

lease-arbitrator env-check
lease-arbitrator validate --spec samples/spec.json --events samples/events.ndjson
lease-arbitrator demo
lease-arbitrator verify \
  --spec samples/spec.json \
  --events samples/events.ndjson \
  --ledger ledger/demo.sqlite3
lease-arbitrator water-report \
  --spec samples/spec.json \
  --events samples/events.ndjson \
  --at-ms 0 \
  --out reports/water-adjudication.json
lease-arbitrator invalidate \
  --spec samples/spec.json \
  --events samples/events.ndjson \
  --edit samples/edit-delete-lease.json \
  --out reports/invalidation-delete.json
pytest -q
