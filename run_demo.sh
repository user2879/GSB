#!/usr/bin/env bash
# WSL/Linux/macOS 一键演示
set -euo pipefail
WORKSPACE="${1:-./demo_workspace}"
SEED="${2:-20261122}"
python3 --version
python3 -m tier_mover.cli init "$WORKSPACE" --seed "$SEED"
python3 -m tier_mover.cli plan "$WORKSPACE"
python3 -m tier_mover.cli run "$WORKSPACE"
python3 -m tier_mover.cli status "$WORKSPACE"
python3 -m tier_mover.cli replay "$WORKSPACE"
python3 -m tier_mover.cli demo "${WORKSPACE}-demo" --seed "$SEED"
