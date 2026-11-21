#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 -m unittest discover -s tests -v 2>&1 | tee validation_tests.txt
python3 -m aqeval demo --workspace validation_workspace 2>&1 | tee validation_output.txt
python3 -m aqeval generate --workspace sample_workspace 2>&1 | tee -a validation_output.txt
python3 -m aqeval run --workspace sample_workspace 2>&1 | tee -a validation_output.txt
