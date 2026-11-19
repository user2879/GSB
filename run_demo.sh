#!/usr/bin/env bash
# Deterministic terrain occlusion-culling demo (POSIX entry point).
set -euo pipefail
python3 -m pip install --user -r requirements.txt
mkdir -p artifacts/demo
python3 -m terrain_culling.demo --out artifacts/demo --frames 16
python3 -m pytest tests -q
