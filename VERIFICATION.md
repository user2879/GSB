# 验证记录

执行日期：2026-11-20（Asia/Shanghai）

## 测试命令

```bash
.venv/bin/pytest -q
```

结果：

```text
...............                                                          [100%]
16 passed
```

覆盖项：

- `E_CONFIG_INVALID` 的未知列、类型不符、目标列缺失、非正阶数、非正预算、非法精度六类错误逐方案隔离。
- 交叉组合按阶数、列名排序；哈希桶截断与跳过策略及降级明细。
- 数据生成、缺失填充、类别编码和设计矩阵可复现。
- PCA 最大绝对分量符号归一；随机投影矩阵与逐元素结果复现。
- 四阶段逐样本误差、首次越界环节、精度敏感样本。
- 维度边界翻转的 `E_BOUNDARY_SENSITIVE` 与 `projection_accumulation` 来源。
- PCA 参数变化只失效 PCA 路径及下游；其他路径命中。
- 基数预算变化使交叉产物及全部下游失效。
- 样本规模档变化复用未变档，并将移除档标记为 retired。
- crossing 阶段提交后中断，再次运行从完整快照恢复；未完成 `.tmp` 产物被忽略。
- 全局版本变化触发 `E_SNAPSHOT_STALE`，恢复阶段为空。
- 并发度 1 与 3 的方案指标、边界、敏感清单、失败清单和缓存账本逐字段一致。
- 主样例产生真实精度敏感和边界敏感结果；二次运行完整恢复且分析指纹不变。

## 最终样例

配置：`sample/experiment.yaml`

```bash
.venv/bin/python -m feature_diag run \
  --config sample/experiment.yaml \
  --out artifacts/reports/diagnostic-report \
  --cache-dir artifacts/runtime-final-cache \
  --snapshot-dir artifacts/runtime-final-snapshot \
  --max-workers 4
```

关键结果：

| 指标 | 值 |
|---|---:|
| 有效方案 | 2 |
| 阶数 × 路径任务 | 27 |
| 精度敏感样本记录 | 42 |
| 边界候选 | 357 |
| `E_BOUNDARY_SENSITIVE` | 42 |
| 首次运行缓存产物 | 144 |
| 分析指纹 | `caffa36269dca162553672022f9ca5ffb6704d7d2c6e4b3952007d0a8019b9bf` |

## 快照恢复

```bash
.venv/bin/python -m feature_diag run \
  --config sample/experiment.yaml \
  --out artifacts/reports/diagnostic-report-resumed \
  --cache-dir artifacts/runtime-final-cache \
  --snapshot-dir artifacts/runtime-final-snapshot \
  --max-workers 4
```

关键结果：

| 指标 | 值 |
|---|---|
| 恢复阶段 | `crossing, reductions, metrics, diagnostics` |
| 本次提交阶段 | 无 |
| 恢复点 | `diagnostics` |
| 快照复用产物 | 144 |
| 分析指纹 | `caffa36269dca162553672022f9ca5ffb6704d7d2c6e4b3952007d0a8019b9bf` |

首次与恢复报告的分析指纹完全一致；数值、边界、敏感清单和报告顺序不随缓存/快照恢复改变。

## 并发与陈旧快照

机器可读汇总见 `artifacts/reports/verification-summary.json`。

- 串行有效并发 1、并行有效并发 2（受场景配置上限约束）；两者分析指纹均为 `caffa36269dca162553672022f9ca5ffb6704d7d2c6e4b3952007d0a8019b9bf`。
- 方案指标、边界诊断、交叉诊断、降维诊断、精度敏感清单、失败清单和缓存账本逐字段相等。
- 将全局版本改为 `1:999` 后复用 `1:1` 快照，报告记录 `E_SNAPSHOT_STALE`，恢复阶段为空，旧产物未被使用。

## 产物

- `artifacts/reports/diagnostic-report/report.json`
- `artifacts/reports/diagnostic-report/report.md`
- `artifacts/reports/diagnostic-report-resumed/report.json`
- `artifacts/reports/diagnostic-report-resumed/report.md`
- `artifacts/reports/verification-summary.json`
