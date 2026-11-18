#!/usr/bin/env bash
# Deterministic local entry point. Cleans artifacts before every run so the
# acceptance flow starts from a blank snapshot/report state.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ ! -x ".venv/bin/python" ]]; then
  python3 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt
fi

rm -rf artifacts/snapshots artifacts/reports artifacts/reference_snapshots
mkdir -p artifacts/snapshots artifacts/reports

.venv/bin/python -m rag_app.cli generate --sample-dir sample
.venv/bin/python -m rag_app.cli full \
  --snapshot-dir artifacts/snapshots \
  --report artifacts/reports/full_report.json
.venv/bin/python -m rag_app.cli replay \
  --snapshot-dir artifacts/snapshots \
  --report artifacts/reports/replay_report.json
# Leave a clean, config-aligned snapshot for resume verification.
.venv/bin/python -m rag_app.cli full \
  --snapshot-dir artifacts/snapshots \
  --report artifacts/reports/full_report.json
.venv/bin/python -m rag_app.cli resume \
  --snapshot-dir artifacts/snapshots
.venv/bin/python -m pytest tests -q
