#!/usr/bin/env bash
# 本地确定性运行：生成样本 -> 全量链路 -> 注入隔离 -> 快照恢复判定
set -euo pipefail
cd "$(dirname "$0")/.."

python3.11 scripts/generate_samples.py
python3.11 -m voicecmd.cli run --report diagnostic_report.json
python3.11 -m voicecmd.cli run --injection configs/injection_full.json --report injection_report.json
python3.11 -m voicecmd.cli resume-check
python3.11 -m pytest tests/ -q
