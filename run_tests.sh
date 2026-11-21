#!/usr/bin/env bash
set -euo pipefail
python3 -m unittest discover -s tier_mover/tests -p 'test_*.py' -v
