#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p workspace
python3 -m rts_replay --workspace workspace demo
python3 -m unittest discover -s tests
