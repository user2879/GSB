#!/usr/bin/env bash
set -euo pipefail

python3 --version
python3 -m pip install -r requirements.txt
if ! python3 -c 'import tkinter' >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y python3-tk
fi
python3 -m gsb.cli doctor
