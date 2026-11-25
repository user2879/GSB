#!/usr/bin/env bash
set -euo pipefail

sudo apt-get update
sudo apt-get install -y \
  python3 python3-dev python3-pip python3-venv \
  python3-tk python3-numpy python3-yaml

python3 -m pip install -r requirements.txt
python3 - <<'PY'
import sys
import numpy
import yaml
print("python", sys.version)
print("numpy", numpy.__version__)
print("pyyaml", yaml.__version__)
try:
    import tkinter
    print("tk", tkinter.TkVersion)
except Exception as exc:
    print("tkinter not available:", exc)
    raise
PY
