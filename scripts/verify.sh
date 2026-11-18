#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"

"$PYTHON" -m pytest | tee artifacts/pytest.txt

"$PYTHON" -m remix_workbench run \
  --samples-root samples \
  --config configs/remix.json \
  --workdir artifacts/release_run \
  --report artifacts/report.json

"$PYTHON" -m remix_workbench run \
  --samples-root samples \
  --config configs/remix.json \
  --workdir artifacts/release_run \
  --report artifacts/report_rerun.json \
  --skip-serial-compare

"$PYTHON" -m remix_workbench run-sequence \
  --samples-root samples \
  --config configs/remix.json \
  --edits configs/edit_sequence.json \
  --workdir artifacts/sequence_final \
  --report artifacts/report_sequence.json

"$PYTHON" -m remix_workbench run \
  --samples-root samples \
  --config configs/remix.json \
  --injections configs/fault_injections.json \
  --workdir artifacts/injected_final \
  --report artifacts/report_injected.json

"$PYTHON" scripts/write_verification.py
