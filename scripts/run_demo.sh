#!/usr/bin/env bash
# 一键本地演示：固定种子样本 -> 增量/全量运行 -> 一致性对比 -> 注入诊断
set -euo pipefail
cd "$(dirname "$0")/.."

WORK="output/demo"
python3 - <<PY
import shutil
shutil.rmtree("${WORK}", ignore_errors=True)
PY

echo "== 1. 固定种子生成样本 =="
python3 -m rs_calib.cli generate --out sample_data --seed 20261122

echo "== 2. 增量运行 =="
python3 -m rs_calib.cli run --sample-dir sample_data --output-dir "${WORK}/inc" \
    --mode incremental

echo "== 3. 空白全量重建 =="
python3 -m rs_calib.cli run --sample-dir sample_data --output-dir "${WORK}/full" \
    --mode full

echo "== 4. 逐字段一致性对比 =="
python3 -m rs_calib.cli compare \
    "${WORK}/inc/diagnostic_report_incremental.json" \
    "${WORK}/full/diagnostic_report_full.json"

echo "== 5. 异常注入诊断 =="
python3 -m rs_calib.cli run --sample-dir sample_data \
    --output-dir "${WORK}/inc-inj" --mode incremental \
    --injection sample_data/injection_config.json

echo "演示完成，报告位于 ${WORK}"
