#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
pytest -q
python3 -m remote_seg demo --out runtime/final-demo
