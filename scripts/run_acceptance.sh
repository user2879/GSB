#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "[1/5] Python version"
python3 --version

echo "[2/5] Compile"
python3 -m compileall -q vertical_ocr tests

echo "[3/5] Unit tests"
python3 -m unittest discover -s tests

echo "[4/5] Deterministic sample + CLI run"
python3 -m vertical_ocr.cli sample --root demo_workspace
python3 -m vertical_ocr.cli validate --root demo_workspace
python3 -m vertical_ocr.cli run --root demo_workspace

echo "[5/5] Full acceptance (conversion, cache, budget, injection, SIGKILL recovery)"
python3 -m vertical_ocr.cli acceptance --root artifacts/acceptance-run

echo "Conversion round-trip examples"
python3 -m vertical_ocr.cli convert \
  --from v3 --to v1 \
  --input demo_workspace/exchange/results.v3.txt \
  --output demo_workspace/exchange/acceptance.v1.txt \
  --ledger-dir demo_workspace/ledger
python3 -m vertical_ocr.cli convert \
  --from v1 --to v3 \
  --input demo_workspace/exchange/acceptance.v1.txt \
  --output demo_workspace/exchange/acceptance.restored.v3.txt \
  --ledger-dir demo_workspace/ledger

echo
echo "ACCEPTANCE SUMMARY"
cat artifacts/acceptance-run/acceptance_summary.md
echo
echo "Machine report: artifacts/acceptance-run/acceptance_report.json"
