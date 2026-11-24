#!/usr/bin/env bash
# 本地确定性样本验收：三格式归一、快速/权威逐字节一致、强制终止恢复、演进替换。
set -euo pipefail

cd "$(dirname "$0")/.."

# 无头 Linux 校验环境的 Tk 本地库（Windows 官方 Python 自带 Tk，无需此行）。
if [ -d /tmp/tkdeps2/usr/lib/x86_64-linux-gnu ]; then
  export LD_LIBRARY_PATH="/tmp/tkdeps2/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
fi

V1=samples/v1
V2=samples/v2
DB=artifacts/demo.db
CRASH=artifacts/demo-crash.db
python - "$DB" "$CRASH" <<'PY'
import glob, os, sys
for path in glob.glob(sys.argv[1] + "*") + glob.glob(sys.argv[2] + "*"):
    if os.path.exists(path):
        os.remove(path)
PY

echo "== 1. 正常运行（真实回环子进程 + r1/r2/r3 三格式） =="
python -m receipt_ledger.cli --workers "$V1/workers.json" \
  --requests "$V1/requests.json" --rules "$V1/rules.json" \
  --db "$DB" run

echo "== 2. 权威重算 vs 快速缓存逐字节一致 =="
python -m receipt_ledger.cli --workers "$V1/workers.json" \
  --requests "$V1/requests.json" --rules "$V1/rules.json" \
  --db "$DB" recompute

echo "== 3. 强制终止（提交 3 批后 os._exit 137） =="
set +e
python -m receipt_ledger.cli --workers "$V1/workers.json" \
  --requests "$V1/requests.json" --rules "$V1/rules.json" \
  --db "$CRASH" run --crash-after-commits 3
echo "crash exit code: $?"
set -e

echo "== 4. 仅依据完整提交批次恢复，未提交批次标注 unknown =="
python -m receipt_ledger.cli --workers "$V1/workers.json" \
  --requests "$V1/requests.json" --rules "$V1/rules.json" \
  --db "$CRASH" resume

echo "== 5. 声明 v2 演进：陈旧标记 + 整体替换 + 只读备份 =="
python -m receipt_ledger.cli --workers "$V2/workers.json" \
  --requests "$V2/requests.json" --rules "$V2/rules.json" \
  --db "$DB" --spec-version 2 run

echo "== 6. 链式摘要与重放校验 =="
python -m receipt_ledger.cli --workers "$V2/workers.json" \
  --requests "$V2/requests.json" --rules "$V2/rules.json" \
  --db "$DB" --spec-version 2 verify

echo "全部验收步骤完成。"
