#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
[ -f samples/sample.json ] || python3 main.py generate --input samples/sample.json
python3 main.py gui --input samples/sample.json --state-dir state/workbench
