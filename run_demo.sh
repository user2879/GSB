#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# The delivery environment had no passwordless sudo. These optional paths make
# the locally extracted Tk deb packages usable without changing WSL globally.
if [[ -d ".sysroot/root/usr/lib/x86_64-linux-gnu" ]]; then
  export LD_LIBRARY_PATH="$PWD/.sysroot/root/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export TCL_LIBRARY="$PWD/.sysroot/root/usr/share/tcltk/tcl8.6"
  export TK_LIBRARY="$PWD/.sysroot/root/usr/share/tcltk/tk8.6"
fi

if [[ ! -x ".venv/bin/python" ]]; then
  python3 -m venv .venv
fi
# Debian/Ubuntu images may lack ensurepip for the local Python build. The
# fallback keeps --system-site-packages so the locally available numpy is used.
if ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
  python3 -m venv --without-pip --system-site-packages .venv
fi
source .venv/bin/activate
python -c 'import numpy' 2>/dev/null || python -m pip install -r requirements.txt

python scripts/generate_fixtures.py
python -m feedback_pe.demo | tee out/demo_stdout.json
python -m unittest discover -s tests -v 2>&1 | tee out/unittest_verification.log

echo "Demo complete. See out/demo_summary.json and docs/VALIDATION_RESULTS.md."
