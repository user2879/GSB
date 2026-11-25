#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

python3 --version
python3 -m pip install -r requirements.txt
python3 -m reconflow demo --root artifacts/demo
python3 -m unittest discover -s tests -v

echo
echo "Verification report: $ROOT/artifacts/demo/verification_report.json"
echo "Diagnostic text:      $ROOT/artifacts/demo/diagnostic_view.txt"
echo "Tk diagnostic UI:     PYTHONPATH=src python3 -m reconflow ui --root artifacts/demo"
