#!/usr/bin/env bash
# Linux/WSL equivalent. On Debian/Ubuntu WSL, install Tk first:
#   sudo apt-get install python3-tk
set -euo pipefail
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -c "import sqlite3, numpy; print('core environment ready (tkinter optional)')"
