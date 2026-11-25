#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate

./scripts/check_ports.sh
python -m lease_middleware validate --sample-dir samples
python -m lease_middleware demo --root .
python -m lease_middleware verify \
  --sample-dir samples \
  --ledger artifacts/demo/order-forward/ledger.sqlite

echo "Demo artifacts: artifacts/demo/comparison.json"
