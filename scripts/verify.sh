#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
SPEC="${1:-artifacts/demo/declaration.generated.json}"
LEDGER="${2:-artifacts/demo/ledger.sqlite}"

if [ -x ".venv/bin/cloudshard" ]; then
  exec .venv/bin/cloudshard verify --spec "$SPEC" --ledger "$LEDGER"
fi
exec python -m cloudshard verify --spec "$SPEC" --ledger "$LEDGER"
