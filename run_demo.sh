#!/usr/bin/env bash
set -euo pipefail
python3 -m gpu_diag.cli init-samples
python3 -m gpu_diag.cli run
python3 -m gpu_diag.cli recompute
python3 -m gpu_diag.cli verify-ledger
python3 -m gpu_diag.cli ui
