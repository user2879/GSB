#!/usr/bin/env bash
# RegBench 一键可重复演示（Linux/macOS；Windows 命令见 README）。
# 不依赖网络/容器/GPU，仅需 Python>=3.10 与 NumPy。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
PY="${PYTHON:-python3}"

ART="$ROOT/artifacts/demo"
SNAP_FULL="$ART/snap_full"
SNAP_TERM="$ART/snap_sigterm"
SNAP_KILL="$ART/snap_sigkill"
mkdir -p "$ART"

echo "== [0] 环境 =="
"$PY" --version
"$PY" -c "import numpy; print('numpy', numpy.__version__)"

echo "== [1] 配置校验（非法样本应收集多条 E_CONFIG_INVALID） =="
"$PY" -m regbench.cli check-config \
  --config samples/invalid_config.yaml \
  --report "$ART/invalid_report.json" || true

echo "== [2] 一次性全量训练（含单方案配额 E_TASK_QUOTA_EXCEEDED） =="
"$PY" -m regbench.cli run \
  --config samples/experiment.yaml \
  --snapshot-dir "$SNAP_FULL" \
  --report "$ART/full_report.json" --reset

echo "== [3] 小预算演示 E_BUDGET_EXHAUSTED =="
"$PY" -m regbench.cli run \
  --config samples/tight_budget_experiment.yaml \
  --snapshot-dir "$ART/snap_tight" \
  --report "$ART/tight_budget_report.json" --reset

echo "== [4] 交互命令流回放与精确失效/缓存复用 =="
"$PY" -m regbench.cli replay \
  --config samples/experiment.yaml \
  --commands samples/commands.txt \
  --snapshot-dir "$ART/snap_replay" \
  --report "$ART/replay_report.json" --reset

echo "== [5] 单任务异常隔离（梯度阶段 failpoint） =="
"$PY" -m regbench.cli run \
  --config samples/experiment.yaml \
  --snapshot-dir "$ART/snap_failpoint" \
  --report "$ART/failpoint_report.json" --reset \
  --failpoint "train:2:lasso|pen=0.1" || true

echo "== [6] 串行/并发/中断恢复三路逐字段一致性 =="
"$PY" -m regbench.cli consistency-check \
  --config samples/experiment.yaml \
  --snapshot-dir "$ART/snap_consistency" \
  --report "$ART/consistency_report.json" \
  --interrupt-epoch 3

echo "== [7] 优雅退出信号（SIGTERM）后恢复，结果与全量一致 =="
REGBENCH_CONFIG=samples/long_experiment.yaml \
REGBENCH_SNAPSHOT="$SNAP_TERM" \
REGBENCH_KILL=sigterm REGBENCH_EPOCH=7 REGBENCH_TASK=lasso \
  "$PY" scripts/interruptible_run.py
REGBENCH_CONFIG=samples/long_experiment.yaml \
REGBENCH_SNAPSHOT="$SNAP_TERM" \
REGBENCH_KILL=none REGBENCH_REPORT="$ART/sigterm_resume_report.json" \
  "$PY" scripts/interruptible_run.py

echo "== [8] 强制终止（SIGKILL）后仅依据完整快照恢复 =="
set +e
REGBENCH_CONFIG=samples/long_experiment.yaml \
REGBENCH_SNAPSHOT="$SNAP_KILL" \
REGBENCH_KILL=sigkill REGBENCH_EPOCH=9 REGBENCH_TASK=ridge \
  "$PY" scripts/interruptible_run.py
KILL_EXIT=$?
set -e
echo "sigkill 子进程退出码=$KILL_EXIT（预期 137）"
REGBENCH_CONFIG=samples/long_experiment.yaml \
REGBENCH_SNAPSHOT="$SNAP_KILL" \
REGBENCH_KILL=none REGBENCH_REPORT="$ART/sigkill_resume_report.json" \
  "$PY" scripts/interruptible_run.py

echo "== [9] 恢复结果与长任务全量结果逐字段比对 =="
"$PY" -m regbench.cli run \
  --config samples/long_experiment.yaml \
  --snapshot-dir "$ART/snap_long_full" \
  --report "$ART/long_full_report.json" --reset
"$PY" scripts/verify_resume_equality.py \
  "$ART/long_full_report.json" \
  "$ART/sigterm_resume_report.json" \
  "$ART/sigkill_resume_report.json"

echo "演示完成，报告目录: $ART"
ls -1 "$ART"
