#!/usr/bin/env bash
# 一键演示：生成样本 -> 完整下发/双路回收 -> 强制终止恢复 -> 独立重放。
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"

# 激活虚拟环境（若存在）
if [ -f "$DIR/../.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$DIR/../.venv/bin/activate"
fi

SPEC="samples/demo_spec.json"
DB="state/ledger.sqlite3"

echo "==> [1/4] 生成确定性声明与事件样本"
python -m edge_cache.cli gen-sample --spec "$SPEC"

echo "==> [2/4] 启动节点子进程，运行下发、续约、双路回收与预算约束"
python -c "import pathlib,glob;[p.unlink() for p in pathlib.Path('state').glob('ledger.sqlite3*')]"
python -m edge_cache.cli run --spec "$SPEC" --db "$DB"

echo "==> [3/4] 强制终止 node-b 后依据账本恢复并续跑"
python -c "import pathlib,glob;[p.unlink() for p in pathlib.Path('state').glob('ledger.sqlite3*')]"
python -m edge_cache.cli crash-demo --spec "$SPEC" --db "$DB" \
  --crash-seq 3 --crash-node node-b

echo "==> [4/4] 独立只读重放校验"
python -m edge_cache.cli verify --spec "$SPEC" --db "$DB"

echo "==> 演示完成。打开界面: python -m edge_cache.cli ui --spec $SPEC --db $DB"
