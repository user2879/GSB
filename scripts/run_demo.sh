#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")/.."
python -m unittest discover -s tests -v
python -m ts_pipeline demo --root demo_workspace --seed 20261122 --force
