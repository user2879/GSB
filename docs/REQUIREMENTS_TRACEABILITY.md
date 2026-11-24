# 需求追踪与技术说明

## 实现映射

| 要求 | 实现位置 | 验证证据 |
| --- | --- | --- |
| 配置/样本校验、错误码和失败清单 | `vibpipe/config.py`, `vibpipe/samples.py`, `vibpipe/errors.py` | 5 类配置注入、3 类样本注入、`tests/test_pipeline.py` |
| 五阶段确定性定点计算 | `vibpipe/fixedpoint.py`, `vibpipe/stages.py`, `vibpipe/imputation.py` | `serial_concurrent_equivalence`、定点数学单测 |
| 窗口统计、万分比漂移和边界 | `vibpipe/stats.py` | 示例结果 `[稳定, 漂移, 稳定]`；精确相等/半步单测 |
| 双档位误差传播与翻转分类 | `vibpipe/precision.py` | `precision_attribution`、`boundary_and_flip_rules` |
| 内容指纹缓存键和复用账本 | `vibpipe/cache.py`, `vibpipe/engine.py` | `warm_cache_snapshot_recovery`、`incremental_edit_sequence` |
| 编辑序列等价和预算待办 | `vibpipe/verify.py` | 7 步编辑对拍、`point_budget_pending_and_completion` |
| 有界线程池和稳定顺序 | `vibpipe/concurrency.py` | 4 线程屏障测试、串行/并发产品逐字段相等 |
| 原子快照、恢复和故障继续 | `vibpipe/snapshot.py`, `vibpipe/faults.py` | `injection_and_batch_faults`、CLI 热启动 `resume` |

## 定点口径

- 核心计算只使用 Python 任意精度整数和 `Decimal` 解析结果，不把样本先转成二进制浮点再累加。
- 配置 `decimal_width=12` 时，规范缩放因子为 `10^12`；各阶段仍可声明自己的正整数 `scale`。
- 除法统一使用 half-up：正数 `(2n+d)//(2d)`，负数对称处理。
- 档位 `Q3/Q6/Q9/Q12/Q15/Q18` 表示十进量化宽度；高档归因自动选择下一档。
- 自然对数使用整数域归一化和 atanh 奇数项级数，迭代次数由 `params.iterations` 固定。

## 漂移判据

对参考均值 `r`、观测均值 `o` 和阈值 `t`（万分比整数）：

```text
delta = abs(o - r)
denominator = max(abs(r), drift_denominator_floor_q)
drift iff delta * 10000 >= t * denominator
```

使用整数交叉相乘，避免比例舍入影响结论。完全相等按漂移处理；若到阈值曲面的距离不超过当前档位半个量化步长，则写入 `boundary_features` 和判定依据。

## 精确失效规则

阶段缓存键为：

```text
SHA256(code_version + upstream_product_fingerprint
       + stage_parameter_fingerprint + decimal_width + gear)
```

因此：

- 样本变化改变源内容指纹，并沿真实 DAG 向下传播。
- 阶段参数或引用变化只改变该阶段键；未变化上游直接复用。
- 档位/位宽进入每个阶段键，切换后按内容键重新计算或复用。
- 若某阶段重算后产物指纹未变，下游可凭相同上游内容指纹精确复用；这是阶段边界上的内容寻址结论。

## 实际结果

最近一次本地执行：

- `python -m unittest discover -s tests`：9 个测试全部通过。
- `python -m vibpipe.verify`：8 组验收全部通过。
- 示例漂移：`clip:xy_energy` 425bp/阈值430 稳定；`clip:yz_energy` 1259bp/阈值1000 漂移；`clip:xz_energy` 385bp/阈值400 稳定。
- 预算 `--budget 39`：状态 `budget_pending`，8 个批次各有一条 `E_POINT_BUDGET_PENDING`，持久化 216 条稳定排序待办任务。
- 热启动：恢复结论 `resume`，全局版本递增，`source` 与五个阶段均为 `REUSED`。
- 运行时注入：`INJ-BATCH-COMPUTE-009A` 和 `INJ-SNAPSHOT-DISK-009B` 命中，游标为批次 5，截断阶段仍保留其余 56 条样本产物。

机器可读结果见 `artifacts/verification_report.json`。Windows 操作步骤见 `docs/WINDOWS_RUNBOOK.md`。
