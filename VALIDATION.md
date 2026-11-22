# 实际验证结果

验证日期：2026-11-23（Asia/Shanghai）

## 环境

- Python：3.11.0rc1
- 已安装 NumPy：1.26.4（项目运行期不调用 NumPy API，兼容 NumPy 2.x）
- pytest：8.3.3
- 执行位置：本地工作区，未使用 Docker、GPU、外部账号、未授权站点或网络数据
- 样本种子：`20261123`

代码仅使用 `pathlib`、`os.replace`、`tempfile`、`ThreadPoolExecutor` 等跨平台标准库；Windows 上使用同一命令即可运行。

## 样本规模

```text
rows=24
groups=3 (g_numeric, g_sensor, g_ratio)
columns=8
bit_width_candidates=[4,6,8]
active_bit_width=8 (gear=2)
fractional_bits=3
plan_batches=5
```

样本包含 6 个缺失单元、精确零值、分位尾部异常值、低端/高端饱和候选，以及更新、追加、删除、行重分组和列重划分动作。

## 自动化测试

```text
$ python -m pytest -q
..........                                                               [100%]
11 passed in 18.29s
```

覆盖项：

1. 完整业务产物、独立组参数、四种误差来源、零值绝对误差分支、确定性重跑。
2. 跨运行单组修正：未受影响组命中缓存，最终数据与全量重建逐字节一致。
3. 同一计划从 `b02_row_regroup` 部分快照继续：含实际复用清单在内与全量重建逐字节一致。
4. `E_CONFIG_INVALID`、`E_DATA_INVALID`、`E_GROUP_UNKNOWN`；失败行不进入有效表/标定产物。
5. 预算拆分、队首待办、`E_CALIB_BUDGET_PENDING` 事件与账本快照。
6. 配置哈希陈旧与单阶段版本陈旧恢复；`E_SNAPSHOT_STALE` 后结果仍与基准一致。
7. gear/位宽/小数位宽切换会改变真实定点中间值和反量化值。
8. 线程池上限约束及单线程/多线程结果逐字节一致。
9. 列重划分后组列集合和敏感性统计重新分布，旧组统计不复用。
10. 对称方案零中心定点参数；负值、零点和正负 q 区间。
11. `python -m groupcalib.pipeline` 成功/失败退出码与诊断文件。

## 最终实跑

命令：

```bat
python -m groupcalib.pipeline --config inputs/config.json --table inputs/features.jsonl --plan inputs/plan.json --out out
python -m groupcalib.pipeline --config inputs/config.json --table inputs/features.jsonl --plan inputs/plan.json --out out_reference --full-rebuild
python -m groupcalib.pipeline --config inputs/config.json --table inputs/features.jsonl --plan inputs/plan.json --out out_partial --through-batch b03_small_corrections
python -m groupcalib.pipeline --config inputs/config.json --table inputs/features.jsonl --plan inputs/plan.json --out out_partial
```

最终摘要：

```json
{
  "business_event_counts": {"E_CALIB_BUDGET_PENDING": 11},
  "business_code_counts": {},
  "completed_groups": ["g_numeric", "g_ratio", "g_sensor"],
  "exit_code": 0,
  "failure_count": 0,
  "recomputed_stage_count": 132,
  "reused_stage_count": 48,
  "status": "ok"
}
```

预算与分层：

```text
element_budget_per_batch=48
budget_cycles=16
consumed_budget=702
threshold_exceedances=8
quantized_elements=201
clipped_elements=57
outlier_elements=126
zero_value_branch_elements=3
dominant_sources={dequant_rounding, quantile_discretization, saturation_truncation, scale_quantization}
thread_pool_size=3
peak_concurrency=1（执行器上限为 3；短任务在 GIL 下实际重叠可低于上限）
```

规范产物 SHA-256：

| 产物 | SHA-256 |
| --- | --- |
| `calibration.jsonl` | `bfeeac904c3849cc5231a171f87126a3bce626081a3b9498687540f542722237` |
| `quantized.jsonl` | `27fcb52417b08e1af01039f977425d89124f6b2ac814e82ee8d0c1e93fb807a2` |
| `attribution.jsonl` | `f3fe48fc58d787f28d8e82e5d635044c2f01ee60202321fec53c1859102a6dc8` |
| `sensitivity.jsonl` | `e4a810d8f3c37e5215e1347e872effbacc09441c333c224b360777c5a9782425` |
| `groups.jsonl` | `b0a1e3c0f124ab4ab7e7400c1dda8fd9b9b8319659f28eaf70f7f732b31c5802` |
| `threshold_exceedances.jsonl` | `5881479f124a3e6291047b2b8f603d579818d16be5789fa653043ddbdfa45ef4` |
| `effective_table.jsonl` | `f9c46cf9ca5c839ed806bc0de8e6a2af4826553204e7ad2ea2dcd2d6f676d881` |
| `reuse_recompute.jsonl` | `d98d2ba580744b9863413f51e2203d353b69a5563fdd89a56c474076a009a9c8` |
| `canonical_reuse_plan.jsonl` | `8b9deafb585eac576754fce6cd9c187e71701a3d1c15072263cbaf50ab6f9a13` |
| `batch_ledger.jsonl` | `d826b01dde476b9657a92adec5ecc434b39e4d8b19de8c949e96ccf44804490a` |
| `manifest.json` | `61b2fe2dccbedba006fc51293e9536756fe6ea2e95ad9ac742e5ed02f2875e86` |

逐文件比较结果：

```text
full run == full-rebuild reference: all listed canonical files byte-identical
full run == partial-through-b03 then complete: all listed canonical files byte-identical
```

阶段版本：

```json
{
  "error_attribution": "06194ca6557aafb64d139ed3c985493b1a485e329017ed4fc46df4e15c3a0db9",
  "fixed_rescale": "ae360cd9bdb2b9e42c3963469cc5dc5142ee68106d8c90894d2e88a1d14569dd",
  "quantile_calibration": "3f8141591c3008d42c9e0da3897deb3698c8e2fccbb2623b6cc163dbc463c275",
  "sensitivity": "f875c4ea63b473b5cf93f2296121d9f1a00ac4680f07f037126983bb40c433b4"
}
```

## 故障覆盖结果

- 配置错误样本：退出码 `2`，失败清单只含 `E_CONFIG_INVALID`，不提交标定产物。
- 数据错误样本：退出码 `3`，同时出现 `E_DATA_INVALID` 与 `E_GROUP_UNKNOWN`；未知组行和非法值行不出现在 `effective_table.jsonl`。
- 预算：账本记录 16 个预算周期、11 次待办事件；所有待办最终清空，已完成组结果与全量运行一致。
- 陈旧恢复：故意改坏配置哈希或 sensitivity 阶段版本后，输出 `E_SNAPSHOT_STALE`，重建产物与基准逐字节一致。
- 并发：单线程与 4 线程配置的规范数据产物逐字节一致，实际峰值均不超过配置上限。
