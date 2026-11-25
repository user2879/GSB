# 可重复运行手册

## 环境准备（Windows 10+ / 本地 Python 3.11）
```
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install numpy==1.26.4 pytest==8.2.2
pytest --version
```
Linux/macOS 下等价命令为 `python3 -m venv .venv && source .venv/bin/activate`。
本工程不使用容器、不访问外部服务，所有样本均由固定整数种子
`scripts/generate_samples.py` 中的 `SEED=20261125` 生成。

## 1. 生成确定性样本
```
python scripts/generate_samples.py
```
产物：
- `samples/weights.jsonl`：四层权重包（层名、形状、扁平权重、内容版本、分组标记）。
- `samples/calibration.json`：12 条校准输入/参考输出与 small/medium/large 档位。
- `samples/retrieval_head_replacement.json`：单层权重替换样本。
- `receipts/set_a`、`receipts/set_b`：q1/q2/q3 回执与一份畸形回执。

## 2. 全量运行
```
python -m qcalib.cli run \
  --weights samples/weights.jsonl \
  --calibration samples/calibration.json \
  --config configs/pipeline.yaml \
  --state .work/state \
  --output .work/ledger.json \
  --report .work/report.json
```

## 3. 命令序列增量运行
`configs/commands.json` 覆盖：回执目录切换、位宽/分组调整、舍入与饱和策略切换、
单层权重替换、数据规模档位切换、蒸馏参数变更、预算变更。
```
python -m qcalib.cli run ... --commands configs/commands.json --output .work/full.json
```
逐命令增量运行后与全量比对（忽略瞬态复用视图）：
```
python -m qcalib.cli diff .work/incremental.json .work/full.json \
  --ignore cache reuse_ledger canonical_invalidation_ledger
```
返回码 0 表示逐字段一致。

## 4. 受约束预算演示
`configs/pipeline_constrained.yaml` 触发 `E_QUOTA_EXCEEDED` 与
`E_BUDGET_PENDING`，待办层进入待办队列而不是抛异常终止。

## 5. 强制中断与恢复
```
python -m qcalib.cli run ... --fault before_commit:ledger_aggregation
python -m qcalib.cli recover ... --output .work/recovered.json
```
故障注入返回码 2；已提交阶段保留在 `state/stages/`，最终账本未整体替换前
不会被恢复判定接受。去掉故障后在同一状态目录重跑，账本与空白全量一致。

## 6. 测试
```
pytest -q
```
