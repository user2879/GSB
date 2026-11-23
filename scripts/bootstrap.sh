#!/usr/bin/env bash
# 离线创建虚拟环境并从本地 wheel 安装依赖；无需网络。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"

if [ ! -d ".venv" ]; then
  if "$PYTHON_BIN" -m virtualenv --version >/dev/null 2>&1; then
    "$PYTHON_BIN" -m virtualenv -q .venv
  else
    "$PYTHON_BIN" -m venv .venv
  fi
fi

.venv/bin/pip install --no-index --find-links vendor/wheels \
  -r requirements.txt

.venv/bin/python -c "import numpy, PIL; print('numpy', numpy.__version__, 'Pillow', PIL.__version__)"
