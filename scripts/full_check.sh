#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate

./scripts/check_ports.sh
python -m compileall -q lease_middleware tests
pytest -q
python -m lease_middleware validate --sample-dir samples
python -m lease_middleware demo --root .
python -m lease_middleware verify --sample-dir samples \
  --ledger artifacts/demo/order-forward/ledger.sqlite
python -m lease_middleware crash-demo --root .
for kind in spec lease snapshot verify port timeout reply; do
  python -m lease_middleware inject --kind "$kind" --root . >/dev/null
done
echo "FULL_CHECK_OK"
