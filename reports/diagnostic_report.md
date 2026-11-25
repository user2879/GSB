# 光伏红外组件账本诊断报告

- 总体结果：通过
- Python：3.11.0rc1
- 平台：Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.35
- 检查数：28；通过：28；失败：0
- 样本组件数：72；真值一致：72/72

## 检查结果

| ID | 结果 | 说明 |
| --- | --- | --- |
| `EXCHANGE_ROUNDTRIP` | PASS | v2 -> v1+sidecar -> v2 logical representation is identical |
| `EXCHANGE_E_FIELD_TRUNCATED` | PASS | short and illegal v1 fields emit E_FIELD_TRUNCATED |
| `INCREMENTAL_BASELINE_DETERMINISM` | PASS | blank baseline domain ledger is deterministic |
| `COMMAND_01_EXCHANGE_V2` | PASS | incremental and blank full rebuild match after EXCHANGE_V2 |
| `COMMAND_02_GRID_5X7` | PASS | incremental and blank full rebuild match after GRID_5X7 |
| `COMMAND_03_PERCENTILES` | PASS | incremental and blank full rebuild match after PERCENTILES |
| `COMMAND_04_THRESHOLDS` | PASS | incremental and blank full rebuild match after THRESHOLDS |
| `COMMAND_05_QUANT_Q4_L32` | PASS | incremental and blank full rebuild match after QUANT_Q4_L32 |
| `COMMAND_06_QUANT_Q2_L16` | PASS | incremental and blank full rebuild match after QUANT_Q2_L16 |
| `COMMAND_07_REPLACE_IMAGE_A` | PASS | incremental and blank full rebuild match after REPLACE_IMAGE_A |
| `COMMAND_08_DELETE_IMAGE_C` | PASS | incremental and blank full rebuild match after DELETE_IMAGE_C |
| `COMMAND_09_EXCHANGE_V1` | PASS | incremental and blank full rebuild match after EXCHANGE_V1 |
| `INCREMENTAL_SEQUENCE_ALL_MATCH` | PASS | every command step matches a blank full rebuild field-for-field |
| `BUDGET_STABLE_PREFIX` | PASS | different exhaustion points consume the same stable sorted prefix |
| `BUDGET_PENDING_SNAPSHOT` | PASS | budget shortfall creates ordered pending queue and committed manifest |
| `SNAPSHOT_RECOVERY_FULL_EQUAL` | PASS | resume from pending snapshot yields full ledger and identical deduction sequence |
| `SNAPSHOT_VERSION_CONFIG_DETECTION` | PASS | startup recognizes schema version and configuration hash changes; stale temp transactions are ignored |
| `CONCURRENCY_LIMIT` | PASS | thread pool never exceeds configured maximum and results stay sorted |
| `CONCURRENCY_UPPER_BOUND_VALIDATION` | PASS | concurrency above hard upper limit is rejected |
| `INJ01_TRUNCATED_PNG_ISOLATED` | PASS | truncated PNG emits E_IMAGE_FORMAT while neighbor image completes |
| `INJ02A_COLORMAP_CONFIG` | PASS | out-of-table pseudo-color mapping config emits E_CONFIG_INVALID |
| `INJ02B_COLORMAP_PIXEL_CODE` | PASS | unknown pseudo-color pixel code emits E_IMAGE_FORMAT |
| `INJ03_SILENT_CELL_ISOLATED` | PASS | silent cell is degraded and does not pollute other cells |
| `INJ04_BBOX_OUTSIDE_GRID` | PASS | INJ04_bbox_outside_grid_truth.json emits E_TRUTH_INVALID |
| `INJ05_BBOX_OUTSIDE_IMAGE` | PASS | INJ05_bbox_outside_image_truth.json emits E_TRUTH_INVALID |
| `INJ06_INVALID_CONFIG` | PASS | independent invalid config reports all parameter violations |
| `INJ07_FIELD_TRUNCATED` | PASS | injected v1 truncation and illegal characters emit E_FIELD_TRUNCATED |
| `BOUNDARY_EQUAL_THRESHOLD` | PASS | equality to a threshold selects the higher grade and marks the boundary |

## 增量步骤

| 步骤 | 域一致 | 事件顺序一致 | 旁路一致 | 缓存断言 | 组件数 |
| --- | --- | --- | --- | --- | --- |
| EXCHANGE_V2 | True | True | True | True | 72 |
| GRID_5X7 | True | True | True | True | 105 |
| PERCENTILES | True | True | True | True | 105 |
| THRESHOLDS | True | True | True | True | 105 |
| QUANT_Q4_L32 | True | True | True | True | 105 |
| QUANT_Q2_L16 | True | True | True | True | 105 |
| REPLACE_IMAGE_A | True | True | True | True | 105 |
| DELETE_IMAGE_C | True | True | True | True | 70 |
| EXCHANGE_V1 | True | True | True | True | 70 |

## 注入隔离

- `INJ01_TRUNCATED_PNG_ISOLATED`: 通过 — truncated PNG emits E_IMAGE_FORMAT while neighbor image completes
- `INJ02A_COLORMAP_CONFIG`: 通过 — out-of-table pseudo-color mapping config emits E_CONFIG_INVALID
- `INJ02B_COLORMAP_PIXEL_CODE`: 通过 — unknown pseudo-color pixel code emits E_IMAGE_FORMAT
- `INJ03_SILENT_CELL_ISOLATED`: 通过 — silent cell is degraded and does not pollute other cells
- `INJ04_BBOX_OUTSIDE_GRID`: 通过 — INJ04_bbox_outside_grid_truth.json emits E_TRUTH_INVALID
- `INJ05_BBOX_OUTSIDE_IMAGE`: 通过 — INJ05_bbox_outside_image_truth.json emits E_TRUTH_INVALID
- `INJ06_INVALID_CONFIG`: 通过 — independent invalid config reports all parameter violations
- `INJ07_FIELD_TRUNCATED`: 通过 — injected v1 truncation and illegal characters emit E_FIELD_TRUNCATED
