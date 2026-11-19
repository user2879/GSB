#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3.11}"
VENV="${ROOT}/.venv"

if [[ ! -x "${VENV}/bin/python" ]]; then
    if ! "${PYTHON}" -m venv "${VENV}"; then
        [[ -x "${VENV}/bin/python" ]] || exit 1
    fi
fi

if ! "${VENV}/bin/python" -c 'import numpy' >/dev/null 2>&1; then
    WHEEL="$(find "${ROOT}/wheelhouse" -maxdepth 1 -name 'numpy-1.26.4-cp311-cp311-manylinux*_x86_64.whl' | head -n 1)"
    if [[ -z "${WHEEL}" ]]; then
        echo "Bundled Linux NumPy wheel was not found" >&2
        exit 1
    fi
    SITE="$("${VENV}/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
    "${VENV}/bin/python" -m zipfile -e "${WHEEL}" "${SITE}"
fi

"${VENV}/bin/python" -c 'import numpy; print("NumPy", numpy.__version__)'
"${VENV}/bin/python" "${ROOT}/scripts/generate_samples.py"
