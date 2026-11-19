#!/usr/bin/env bash
# 多视点光度立体 x 可微渲染交替优化 —— POSIX 本地完整示例（与 .bat 等价）
set -euo pipefail

PY="${PYTHON:-python3}"
WS="${WS:-workspace}"

echo "[1/7] 生成 256x256 确定性观测集（6 光源 x 2 相机 = 12 视点）"
"$PY" -m photometric_altopt.cli generate --workspace "$WS" --size 256

echo "[2/7] 解析梯度有限差分核对"
"$PY" -m photometric_altopt.cli gradcheck --workspace "$WS" --samples 32

echo "[3/7] 交替优化基线运行（含真值核对）"
"$PY" -m photometric_altopt.cli run --workspace "$WS" --run-name baseline --max-iterations 8 --check-truth

echo "[4/7] 强制终止模拟（第 3 轮后崩溃），再恢复并续写"
"$PY" -m photometric_altopt.cli run --workspace "$WS" --run-name crash_demo --max-iterations 8 --crash-after 2
"$PY" -m photometric_altopt.cli resume --workspace "$WS" --run-name crash_demo --max-iterations 8

echo "[5/7] 参数层异常注入：非有限反照率 / 零向量光源 / 退化视点集合"
set +e
"$PY" -m photometric_altopt.cli run --workspace "$WS" --run-name f_nonfinite --inject-nonfinite
"$PY" -m photometric_altopt.cli run --workspace "$WS" --run-name f_zero --inject-zero-light
"$PY" -m photometric_altopt.cli run --workspace "$WS" --run-name f_degenerate --inject-degenerate
set -e

echo "[6/7] 管线层异常注入：渲染阶段跳过视点 3 与 7"
"$PY" -m photometric_altopt.cli run --workspace "$WS" --run-name f_skipview --skip-view 3,7 --max-iterations 3

echo "[7/7] 打印工作区内容哈希清单"
"$PY" -m photometric_altopt.cli manifest "$WS/trajectories/baseline"
echo DONE
