#!/usr/bin/env bash
# 本地确定性端到端验证：生成样本 -> 全量运行 -> 增量/并发/格式切换 ->
# 异常注入 -> 信号中断恢复。全部离线，仅依赖 Python3.10+ 与 NumPy。
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3.10}
ROOT=${DEMO_ROOT:-./demo_output}
CFG=config/alignment.json
SPEC=config/format_spec.json

reset() { [ -d "$1" ] && "$PY" -c "import shutil,sys; shutil.rmtree(sys.argv[1])" "$1" || true; }

echo "== 1. 生成确定性样本 =="
reset "$ROOT"
$PY -m xmodal.cli generate --input "$ROOT/input" --count 8

echo "== 2. 全量运行（v2）=="
$PY -m xmodal.cli run --input "$ROOT/input" --workdir "$ROOT/work_v2" \
  --config "$CFG" --spec "$SPEC"

echo "== 3. 二次运行验证全量复用 =="
$PY -m xmodal.cli run --input "$ROOT/input" --workdir "$ROOT/work_v2" \
  --config "$CFG" --spec "$SPEC"

echo "== 4. 串行(workers=1)与并发(workers=8)一致性 =="
$PY -m xmodal.cli run --input "$ROOT/input" --workdir "$ROOT/work_w1" \
  --config "$CFG" --spec "$SPEC" --workers 1
$PY -m xmodal.cli run --input "$ROOT/input" --workdir "$ROOT/work_w8" \
  --config "$CFG" --spec "$SPEC" --workers 8

echo "== 5. 切换 v1 二进制格式（仅转换阶段失效）=="
$PY -c "import shutil; shutil.copytree('$ROOT/work_v2','$ROOT/work_v1')"
$PY -m xmodal.cli run --input "$ROOT/input" --workdir "$ROOT/work_v1" \
  --config "$CFG" --spec "$SPEC" --format-version 1

echo "== 6. 异常注入（六类故障 + 业务码映射）=="
$PY -m xmodal.cli run --input "$ROOT/input" --workdir "$ROOT/work_inj" \
  --config "$CFG" --spec "$SPEC" --injections config/injections.json \
  --format-version 1

echo "== 7. SIGTERM 优雅退出（退出码 0）后恢复 =="
$PY scripts/signal_demo.py "$ROOT"

echo "== 8. 运行单元测试套件 =="
$PY -m unittest discover -s xmodal/tests -p "test_*.py"

echo "演示完成，报告位于 $ROOT/*/report.json"
