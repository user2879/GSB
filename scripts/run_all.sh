#!/usr/bin/env bash
# Deterministic local verification: preflight, unit tests, acceptance scenarios.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate

echo "== preflight =="
python -m arbplatform.cli envcheck | tee results/envcheck.stdout.json

echo "== unit tests =="
python -m unittest discover -s tests -p 'test_*.py' -v

echo "== acceptance: multi-env arbitration =="
python -m arbplatform.cli demo

echo "== acceptance: scale-adaptive path selection =="
python -m arbplatform.cli large

echo "== acceptance: snapshot replacement + forced-termination recovery =="
python -m arbplatform.cli recovery
