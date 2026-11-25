#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/tk-env.sh
. "$DIR/scripts/tk-env.sh"
cd "$DIR"
exec "$DIR/.venv/bin/python" -m edge_ledger.acceptance --out artifacts/final-acceptance --tk-smoke "$@"
