#!/usr/bin/env sh
set -eu
python3 -m terrain_diag demo --output artifacts/demo --force "$@"
