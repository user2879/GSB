#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  if ! "${PYTHON_BIN}" -m venv "${VENV_DIR}"; then
    "${PYTHON_BIN}" -m venv --without-pip "${VENV_DIR}"
  fi
fi

if ! "${VENV_DIR}/bin/python" -m pip --version >/dev/null 2>&1; then
  curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/voicecal-get-pip.py
  "${VENV_DIR}/bin/python" /tmp/voicecal-get-pip.py
fi

"${VENV_DIR}/bin/python" -m pip install --disable-pip-version-check -r requirements.txt
"${VENV_DIR}/bin/python" -c 'import sys, numpy; print(f"Python {sys.version.split()[0]}, NumPy {numpy.__version__}")'
