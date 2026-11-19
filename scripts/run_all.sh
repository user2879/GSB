#!/usr/bin/env bash
# 端到端验收流水线（Linux/macOS/WSL）。
# 运行前清理 samples/reports/artifacts，保证两次运行初始状态一致。
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -x ".venv/bin/python" ]; then
  python3.11 -m venv --without-pip .venv
  python3.11 -m ensurepip 2>/dev/null || true
  .venv/bin/python -m pip install -r requirements.txt
fi

.venv/bin/python -m refrac --workspace . clean
.venv/bin/python -m refrac --workspace . suite
.venv/bin/python -m unittest discover -s tests -v
