#!/usr/bin/env bash
# 本地确定性演示：初始化 -> 全量运行 -> 命令流增量 -> 真实 SIGKILL 恢复
set -euo pipefail
cd "$(dirname "$0")"
python3 -m ancient_ocr.cli init --root .
python3 -m ancient_ocr.cli run --root .
python3 -m ancient_ocr.cli commands --root . commands/demo_commands.json
python3 -m ancient_ocr.cli crash-demo --root . || true
python3 -m ancient_ocr.cli crash-demo --root .
python3 -m unittest discover -s tests
