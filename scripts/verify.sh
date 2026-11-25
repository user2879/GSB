#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -x .venv/bin/python ]]; then
  scripts/bootstrap.sh
fi

.venv/bin/python -m pytest -q
.venv/bin/python -m shard_diag demo --root "$ROOT"

