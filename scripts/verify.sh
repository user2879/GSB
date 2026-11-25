#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m mvrv generate --root .
python3 -m mvrv validate --root .
python3 -m unittest discover -s tests -v
python3 -m mvrv acceptance --root .
