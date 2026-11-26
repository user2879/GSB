#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -d .venv ]]; then
  virtualenv -p python3.11 .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -r requirements.txt
pip install -e . >/dev/null

python scripts/check_loopback.py
python scripts/generate_samples.py
pytest -q
arg-review all --samples samples --state-dir artifacts

echo "acceptance artifacts:"
echo "  artifacts/diagnostic_report.json"
echo "  artifacts/edit_equivalence_report.json"
echo "  artifacts/ledger.json"
echo "  artifacts/snapshots/current"
