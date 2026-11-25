#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

find ledger -maxdepth 1 -type f -name '*.sqlite*' -delete
rm -f ledger/demo_report.json
find . -type d -name __pycache__ -prune -exec rm -rf {} +
rm -rf .pytest_cache
printf 'removed ledger snapshots, demo report, and Python caches\n'

