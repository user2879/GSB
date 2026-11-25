#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate

for kind in spec lease snapshot verify port timeout reply; do
  echo "== inject $kind =="
  python -m lease_middleware inject --kind "$kind" --root .
done
