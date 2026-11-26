#!/usr/bin/env bash
# 可重复演示：规格校验 -> 离线确定性调度 -> 真实回环下发 -> 重建比对 -> 重放校验
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONPATH=src
mkdir -p artifacts

echo "== 1. 规格强校验 =="
python -m dispatch_tokens validate samples/demo_spec.json

echo "== 2. 离线确定性引擎（任意批次切分结论一致）=="
python -m dispatch_tokens run samples/demo_spec.json \
  --ledger artifacts/demo.offline.db \
  --report artifacts/demo.offline.report.json --offline

echo "== 3. 真实子进程 + TCP JSON 行回环下发 =="
ss -ltn | grep -E '4910[123]' && { echo "port conflict"; exit 3; } || true
python -m dispatch_tokens run samples/demo_spec.json \
  --ledger artifacts/demo.live.db \
  --report artifacts/demo.live.report.json

echo "== 4. 增量结果 vs 空白全量重建（E_DIFF 校验）=="
python -m dispatch_tokens verify samples/demo_spec.json \
  --report artifacts/demo.live.report.json

echo "== 5. 账本恢复重放（E_REPLAY 校验）=="
python -m dispatch_tokens replay samples/demo_spec.json \
  --ledger artifacts/demo.live.db

echo "== 6. 强制终止恢复演示 =="
python scripts/forced_kill_demo.py
echo "ALL DEMOS OK"
