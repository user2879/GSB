#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

python3 -m condrender.cli generate-samples --root .
python3 -m condrender.cli validate --root .
python3 -m unittest discover -s tests -v
python3 -m condrender.cli verify --root . --artifacts-dir artifacts/verification

echo "Verification report: artifacts/verification/VERIFICATION_REPORT.md"
