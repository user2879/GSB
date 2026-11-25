#!/usr/bin/env bash
# Reproduce the virtualenv without network sample data. Requires an interpreter
# with venv support; pip pulls numpy 1.26.4 and pysqlite3-binary only.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-/home/zsl/.feedback/python-base/bin/python3.11}"
"$PYTHON" -m venv --without-pip .venv
SP="$(.venv/bin/python -c 'import site; print(site.getsitepackages()[0])')"
BASE_PIP="$(/home/zsl/.feedback/python-base/bin/python3.11 -c 'import os, pip; print(os.path.dirname(pip.__file__))')"
PYTHONPATH="$BASE_PIP" .venv/bin/python -m pip install --target "$SP" \
  --upgrade --ignore-installed -r requirements.txt
