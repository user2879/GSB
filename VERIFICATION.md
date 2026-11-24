# 实际验证记录

验证日期：2026-11-24  
工作区：`/home/zsl/feedback/uidfc5909d3-default-226`  
Python：`3.11.0rc1`  
NumPy：`1.26.4`

## 执行命令

```bash
python tools/build_samples.py
python -m opexlab replay --reset
python -m opexlab convert --to v2 --output runtime/conversions/weights_v2_from_v1.jsonl
python -m opexlab convert --to v1 --output runtime/conversions/weights_v1_from_v2.txt
python -m opexlab inject
python -m opexlab verify
python -m unittest discover -s tests -v
```

## 完整验收

结果：`14/14` 通过，报告位于 `runtime/reports/verification.json`。

1. `v1_v2_roundtrip_restorable_fields` — 11 条 v1 与 11 条 v2 权重可还原字段一致。
2. `lossy_conversion_sidecar` — 不可还原的 scale/bias 残差与命名方案进入旁路。
3. `softmax_deterministic_and_normalized` — 重复运行完全一致，输出整数和等于 `2^12`。
4. `v1_v2_same_logical_parameters_same_fixed_result` — 11 行 LayerNorm 统计量与输出一致。
5. `quant_level_changes_softmax_integer_vector` — 档位 0/2 产生不同整数向量。
6. `repeated_run_business_fields_equal` — 相同输入重复运行业务字段一致。
7. `single_activation_row_invalidation_exact` — 单行编辑只重算该行三个下游阶段。
8. `single_layer_parameter_invalidation_exact` — 单层参数只重算转换与该层输出，统计量复用。
9. `incremental_final_equals_blank_full_business_fields` — 最终增量与空白全量一致。
10. `incremental_final_cache_keys_equal_blank_full` — 阶段缓存键与内容哈希一致。
11. `incremental_final_reuse_lists_equal_blank_full` — 行级失效/复用清单一致。
12. `blank_command_replay_cache_plan_stable` — 从空白重放同一命令流，清单稳定。
13. `serial_parallel_schedule_and_results_equal` — 串行与 4 线程结果、预算、待办、失败一致。
14. `fault_injections_hit_and_map_to_business_codes` — 五类故障全部命中。

## 关键数值

Softmax 首行（`fractional_bits=12`）：

```text
quant_level=0 -> [332, 115, 1070, 1187, 505, 887]
quant_level=2 -> [332, 116, 1072, 1184, 504, 888]
```

两次输出整数和均为 `4096`，但向量不同，证明量化档位真实进入定点计算。

## 异常注入

报告位于 `runtime/reports/injection_report.json`，`all_matched=true`。

| 编号 | 故障 | 预期/实际业务码 | 命中数 |
|---|---|---|---:|
| `INJ-001` | 截断权重行，第 6 行宽 34/36 | `E_FIELD_TRUNCATED` | 1 |
| `INJ-002` | `layer_alpha/2` 激活为 NaN | `E_NONFINITE` | 1 |
| `INJ-003` | `layer_gamma/0` 方案越界 | `E_SCHEME_UNKNOWN` | 1 |
| `INJ-004` | `layer_alpha/1` 位宽 7 | `E_WIDTH_OUT_OF_RANGE` | 1 |
| `INJ-005` | 批次预算 5 < 首行元素 6 | `E_BUDGET_TOO_SMALL` | 4 |

## 命令回放与快照

从空白回放 8 条命令后：

- 最终交换版本：`v1`
- 全局逻辑版本：`8`
- 最终快照：`r0008-s0009`
- 完成层：`layer_alpha`、`layer_beta`、`layer_gamma`
- 已消耗预算：`55`
- 待办行：`0`
- 失败：`0`
- 最终缓存：四个阶段各复用 11 行、重算 0 行

## 双向转换

- v1 → v2：11 行，失败 0，旁路文件已生成；
- v2 → v1：11 行，失败 0，旁路文件已生成；
- 可还原的千分比、偏置码、量化位宽逐字段一致；
- 不可还原字段均带残差与丢弃信息。

## 单元测试

```text
Ran 9 tests in 1.6s
OK
```

覆盖加载校验、双向转换、损失旁路、Softmax 定点确定性、LayerNorm 下限保护、v1/v2 路径等价、物理缓存复用、调度预算/配额/并发上限、原子快照、注入和完整验收套件。
