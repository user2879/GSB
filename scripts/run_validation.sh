#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
cd "$ROOT"

"$PYTHON" -m pointcloud_studio generate --out "$ROOT/artifacts/sample"
"$PYTHON" -m pointcloud_studio validate \
  --sample "$ROOT/artifacts/sample" \
  --workspace "$ROOT/artifacts/run_workspace" \
  --report "$ROOT/artifacts/validation_report.json"
