#!/usr/bin/env bash
# 可重复演示：冷启动采集 -> 失效触发传播 -> 快照保存/恢复重放 + 三类异常码。
set -euo pipefail
cd "$(dirname "$0")/.."
WORK=artifacts/demo_run
rm -rf "$WORK"
mkdir -p "$WORK"

echo "== 场景1 冷启动采集（真实回环，两批次） =="
python3 -m degprop --topology samples/topology.json --events samples/events.ndjson \
  --ledger "$WORK/session.db" --log-file "$WORK/run.log" --log-level WARNING

echo "== 场景2 依赖失效触发降级传播（结论见上，批次2 拒绝/超时集中） =="
echo "== 场景3 快照恢复重放校验 =="
python3 -m degprop --mode resume --topology samples/topology.json \
  --events samples/events.ndjson --ledger "$WORK/session.db" --log-level WARNING

echo "== 异常注入 E_CYCLE =="
python3 -m degprop --inject cycle --ledger "$WORK/cycle.db" --log-level WARNING || true
echo "== 异常注入 E_CAPACITY =="
python3 -m degprop --inject capacity --ledger "$WORK/cap.db" --log-level WARNING || true
echo "== 异常注入 E_SNAPSHOT =="
python3 -m degprop --inject snapshot --ledger "$WORK/snap.db" \
  --log-file "$WORK/snap.log" --log-level WARNING || true
echo "== 异常注入 E_REPLAY（篡改账本结局后重放） =="
python3 -m degprop --inject replay --ledger "$WORK/replay.db" --log-level WARNING || true
echo "DEMO DONE"
