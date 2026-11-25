#!/usr/bin/env bash
# Deterministic local acceptance run. No Docker, no external network.
set -euo pipefail
cd "$(dirname "$0")"
# shellcheck disable=SC1091
source .venv/bin/activate

# Headless Tk shim (local user-space Tcl/Tk libraries when system Tk missing).
if [ -d /tmp/tkdeps2/usr/lib/x86_64-linux-gnu ]; then
  export LD_LIBRARY_PATH="/tmp/tkdeps2/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
  export TCL_LIBRARY="/tmp/tkdeps2/usr/share/tcltk/tcl8.6"
  export TK_LIBRARY="/tmp/tkdeps2/usr/lib/tcltk/x86_64-linux-gnu/tk8.6"
fi

echo "== environment =="
python -c "import sys,sqlite3,numpy,tkinter; print('python',sys.version.split()[0],'numpy',numpy.__version__,'tk',tkinter.TkVersion,'sqlite',sqlite3.sqlite_version)"

echo "== unit tests (real loopback subprocesses included) =="
python -m unittest discover -s tests -v

echo "== deterministic samples =="
python -m rlcb.cli sample --out samples/acceptance >/dev/null

echo "== full run with real 127.0.0.1 service subprocesses =="
python -c "import shutil; shutil.rmtree('artifacts/acceptance', ignore_errors=True)"
python -m rlcb.cli run \
  --services samples/acceptance/services.json \
  --requests samples/acceptance/requests.json \
  --ledger artifacts/acceptance/ledger.db \
  --out artifacts/acceptance --mode full

echo "== scratch recompute consistency (per-request) =="
python -m rlcb.cli verify \
  --services samples/acceptance/services.json \
  --requests samples/acceptance/requests.json \
  --ledger artifacts/acceptance/ledger.db

echo "== forced-kill recovery demonstration =="
python scripts/demo_force_kill.py
echo "ALL ACCEPTANCE STEPS PASSED"
