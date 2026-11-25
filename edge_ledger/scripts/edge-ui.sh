#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/tk-env.sh
. "$DIR/scripts/tk-env.sh"
exec "$DIR/.venv/bin/python" -m edge_ledger.ui "$@"
