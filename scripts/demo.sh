#!/usr/bin/env bash
# 跨故障域租约仲裁与配额预算诊断工作台 · 可重复演示（Linux/macOS）
# 用法：在仓库根目录执行  bash scripts/demo.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python3}
echo "== 1. 环境自检 =="
$PY -c "import sys, sqlite3, numpy; print('python', sys.version.split()[0]); print('sqlite3', sqlite3.sqlite_version); print('numpy', numpy.__version__)"
if ! $PY -c "import tkinter" 2>/dev/null; then
  echo "tkinter 不可用：工作台自动降级为无头模式"
fi

echo "== 2. 生成确定性样本（小/中两档）=="
$PY -m faultisolation.cli gen-samples --out samples

run_scale() {
  local name=$1 spec=$2 sess=$3 ledger=$4
  echo "== 3.$name 强校验与路径选择 =="
  $PY -m faultisolation.cli --spec "$spec" --session "$sess" validate
  $PY -m faultisolation.cli --spec "$spec" --session "$sess" plan
  echo "== 4.$name 真实回环实时仲裁 =="
  $PY -m faultisolation.cli --spec "$spec" --session "$sess" live --ledger "$ledger" --replace
  echo "== 5.$name 独立校验（逐字节/槽位/账本）=="
  $PY -m faultisolation.cli --spec "$spec" --session "$sess" verify --ledger "$ledger"
  echo "== 6.$name 整数定点诊断 =="
  $PY -m faultisolation.cli --spec "$spec" --session "$sess" diagnose
}

run_scale small  samples/small/spec-small.json   samples/small/session.json   artifacts/small-live.sqlite
run_scale medium samples/medium/spec-medium.json samples/medium/session.json artifacts/medium-live.sqlite

echo "== 7. 强制终止与账本恢复演示 =="
$PY -c "from faultisolation.recovery_demo import run_recovery_demo; import json; print(json.dumps(run_recovery_demo('samples/small/spec-small.json','samples/small/session.json','artifacts/recovery-ledger.sqlite'), ensure_ascii=False, indent=2, default=str))"

echo "== 8. 测试套件 =="
$PY -m pytest tests -q
echo "演示完成。"
