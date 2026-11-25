# 逐层激活定点化诊断管线

针对多层激活进入 Softmax / LayerNorm 两个数值敏感算子的定点化误差诊断，
覆盖定点整数算子、v1 定宽行 / v2 JSON 双版本互转、逐级误差归因、
行级指纹缓存与增量复用、预算调度、阶段化快照与故障注入隔离。

## 环境

- Python 3.11（Windows 使用官方安装包并勾选加入 PATH）
- 依赖：`pip install numpy==1.26.4`，其余仅标准库

## 快速开始

```bash
python scripts/generate_samples.py
python -m unittest discover -s tests -v
python scripts/run_diagnostic.py
```

产物：

- `samples/`：固定种子张量包、算子配置、v1/v2 权重、编辑命令流、注入配置
- `reports/diagnostic_report.json`：可复核诊断报告
- `reports/roundtrip/`：v2→v1→v2 往返产物与 sidecar 旁路
- `reports/snapshot/`：阶段化快照与 manifest

## 报告关键结论

报告顶层给出四个可验收布尔/计数结论：

- `incremental_equals_full`：任意命令序列后增量运行与空白全量重建逐字段一致
- `parallel_equals_serial`：线程池并发与串行结果一致
- `injections.all_detected`：四类故障注入均被隔离并按业务码记录
- `budget_pending_count`：本运行 `E_OP_BUDGET_PENDING` 挂起计数

## 代码地图

- 定点引擎与算子：`fxpdiag/fixedpoint.py`、`fxpdiag/operators.py`
- 双版本与旁路：`fxpdiag/weights.py`、`fxpdiag/conversion.py`、`fxpdiag/convert_service.py`
- 误差与归因：`fxpdiag/error_track.py`
- 指纹账本与增量：`fxpdiag/fingerprints.py`、`fxpdiag/ledger.py`、`fxpdiag/pipeline.py`
- 调度/快照/注入：`fxpdiag/scheduler.py`、`fxpdiag/snapshots.py`、`fxpdiag/loader.py`
- 设计细节：`docs/DESIGN.md`
