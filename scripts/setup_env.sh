#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python 3.11 is required. Set PYTHON_BIN to an installed interpreter." >&2
  exit 2
fi

if [[ "${SKIP_APT:-0}" != "1" ]]; then
  if command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
    sudo apt-get update
    sudo apt-get install -y "$PYTHON_BIN" python3.11-venv python3-tk xvfb
  else
    echo "No passwordless sudo: skipping apt installation; the existing distro package set must provide Python 3.11, venv and Tk." >&2
  fi
fi

"$PYTHON_BIN" -m venv --clear .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python scripts/generate_samples.py --root "$ROOT"
python -m edge_ledger.cli --root "$ROOT" build

python - <<'PY'
try:
    import tkinter
except Exception as exc:
    raise SystemExit(f"Tkinter import failed: {exc}")
print(f"Tk {tkinter.TkVersion} is importable")
PY

echo "Environment ready. Activate with: source .venv/bin/activate"

