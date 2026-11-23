#!/usr/bin/env bash
# Deterministic local demonstration; no network or GPU required.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

python scripts/generate_samples.py --sample-dir samples/demo
python -m hotblock_diag.cli --sample-dir samples/demo --workspace runs/demo_full --fresh
python -m hotblock_diag.cli --sample-dir samples/demo --workspace runs/demo_full
python -m hotblock_diag.cli --sample-dir samples/demo --workspace runs/demo_inject \
    --injections samples/demo/injections_block.json --fresh
python -m unittest discover -s tests
