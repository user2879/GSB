#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 1. 单元与集成测试 =="
python -m unittest discover -s tests -q

echo "== 2. 重新生成固定种子样本 =="
python scripts/generate_samples.py --out samples/base --seed 20261123

echo "== 3. 全量 / 交互 / 恢复 三模式 =="
python -m segpos.runner full        --sample-dir samples/base --snapshot-dir run/snap-full --report run/report-full.json
python -m segpos.runner interactive --sample-dir samples/base --snapshot-dir run/snap-inc  --report run/report-inc.json
python -m segpos.runner restore     --sample-dir samples/base --snapshot-dir run/snap-inc  --report run/report-restore.json

echo "== 4. 增量与全量逐字段等价 =="
python scripts/verify_equivalence.py samples/base

echo "== 5. 首响预算挂起与排空语义 =="
PYTHONPATH=src:. python scripts/demo_budget.py samples/base

echo "ALL_ACCEPTANCE_CHECKS_PASSED"
