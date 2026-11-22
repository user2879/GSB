#!/usr/bin/env bash
# 强制终止与恢复判定：真实 kill -9 -> 依赖已提交快照续跑 -> 逐字段比对
set -euo pipefail
cd "$(dirname "$0")/.."

DATA=${DATA:-data}
STATE=${STATE:-state_kill}
REPORTS=${REPORTS:-reports_kill}
mkdir -p "$REPORTS"

# 参考：未中断完整运行
for k in 0 1 2; do
  python3 -m tlc run --data "$DATA" --batch "$k" --state "${STATE}_ref" --report "$REPORTS/ref${k}.json" > /dev/null
done

# 被 kill 的运行：阶段延迟便于在运行中强制终止
for k in 0 1; do
  python3 -m tlc run --data "$DATA" --batch "$k" --state "$STATE" --report "$REPORTS/k${k}.json" > /dev/null
done
TLC_STAGE_DELAY_MS=1500 python3 -m tlc run --data "$DATA" --batch 2 --state "$STATE" --report "$REPORTS/victim.json" &
PID=$!
sleep 2
kill -9 "$PID" 2>/dev/null || true
wait "$PID" 2>/dev/null || true
echo "已强制终止 PID=$PID"

# 恢复续跑（依赖已提交快照与暂存区断点）
python3 -m tlc run --data "$DATA" --batch 2 --state "$STATE" --report "$REPORTS/resumed2.json"

python3 - << PYEOF
import json
ref = json.load(open('$REPORTS/ref2.json'))
res = json.load(open('$REPORTS/resumed2.json'))
assert ref == res, '恢复后结果与未中断运行不一致'
print('OK: 强制终止恢复后与未中断运行逐字段一致')
PYEOF
