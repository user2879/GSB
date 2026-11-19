#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"
PYTHONPATH="${ROOT}" "${ROOT}/.venv/bin/python" scripts/run_acceptance.py

