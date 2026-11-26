#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=scripts/tk-env.sh
source "$ROOT/scripts/tk-env.sh" || true
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
pytest -q
lease-arbitrator verify --spec samples/spec.json --events samples/events.ndjson --ledger ledger/demo.sqlite3
