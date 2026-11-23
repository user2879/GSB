# 诊断报告：07_injection

- 轮次总数/完成：12 / 8
- 定点位宽/档位：12 / 256
- 复用命中：0，重算轮次：[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
- 预算待办：[]
- 快照恢复：False
- 注入命中：['INJ-NONFINITE-01', 'INJ-DIM-01', 'INJ-ZEROGAIN-01', 'INJ-REPLACE-01']
- 失败码统计：{'E_UPSTREAM_BLOCKED': 2, 'E_NONFINITE_SAMPLE': 1, 'E_CALIB_BUDGET_PENDING': 8, 'E_INJECTION_DIM_MISMATCH': 1, 'E_ZERO_GAIN': 2, 'E_ROUND_OUT_OF_RANGE': 1}

## 情感与说话人轨迹

| 轮次 | 情感 | 说话人 | 轨迹段 | 断开原因 |
| --- | --- | --- | --- | --- |
| 0 | calm | S1 | SEG-001 | SHORT_SEGMENT |
| 1 | calm | S2 | SEG-002 | SHORT_SEGMENT |
| 2 | calm | S1 | SEG-003 | - |
| 4 | happy | S1 | SEG-003 | - |
| 6 | calm | S1 | SEG-003 | - |
| 8 | calm | S1 | SEG-003 | - |
| 10 | calm | S1 | SEG-003 | SPEAKER_SWITCH |
| 11 | calm | S2 | SEG-004 | SHORT_SEGMENT |

## 批次预算

- 批次 1：消耗 4/4，复用 0，待办 [4, 5, 6, 7, 8, 9, 10, 11]
- 批次 2：消耗 4/4，复用 0，待办 [8, 9, 10, 11]
- 批次 3：消耗 4/4，复用 0，待办 []

## 失效/复用判定（前 20 条）

- R0 evidence: recompute (recompute:no_cache_entry)
- R0 calibration: recompute (recompute:evidence_invalidated)
- R0 attribution: recompute (recompute:calibration_invalidated)
- R1 evidence: recompute (recompute:no_cache_entry)
- R1 calibration: recompute (recompute:evidence_invalidated)
- R1 attribution: recompute (recompute:upstream_calibration_invalidated)
- R2 evidence: recompute (recompute:no_cache_entry)
- R2 calibration: recompute (recompute:evidence_invalidated)
- R2 attribution: recompute (recompute:upstream_calibration_invalidated)
- R3 evidence: recompute (recompute:no_cache_entry)
- R3 calibration: recompute (recompute:evidence_invalidated)
- R3 attribution: recompute (recompute:upstream_calibration_invalidated)
- R4 evidence: recompute (recompute:no_cache_entry)
- R4 calibration: recompute (recompute:evidence_invalidated)
- R4 attribution: recompute (recompute:upstream_calibration_invalidated)
- R5 evidence: recompute (recompute:no_cache_entry)
- R5 calibration: recompute (recompute:evidence_invalidated)
- R5 attribution: recompute (recompute:upstream_calibration_invalidated)
- R6 evidence: recompute (recompute:no_cache_entry)
- R6 calibration: recompute (recompute:evidence_invalidated)

## 失败清单

- E_UPSTREAM_BLOCKED @ R3 calibration: upstream evidence unavailable
- E_NONFINITE_SAMPLE @ R3 evidence: non-finite frame feature detected [INJ-NONFINITE-01]
- E_CALIB_BUDGET_PENDING @ R4 evidence: segment budget exhausted, retained for next batch
- E_UPSTREAM_BLOCKED @ R5 calibration: upstream evidence unavailable
- E_CALIB_BUDGET_PENDING @ R5 evidence: segment budget exhausted, retained for next batch
- E_INJECTION_DIM_MISMATCH @ R5 evidence: evidence dimensionality injection triggered [INJ-DIM-01]
- E_CALIB_BUDGET_PENDING @ R6 evidence: segment budget exhausted, retained for next batch
- E_ZERO_GAIN @ R7 calibration: zero gain injection triggered [INJ-ZEROGAIN-01]
- E_ZERO_GAIN @ R7 calibration: zero gain injection triggered [INJ-ZEROGAIN-01]
- E_CALIB_BUDGET_PENDING @ R7 evidence: segment budget exhausted, retained for next batch
- E_CALIB_BUDGET_PENDING @ R8 evidence: segment budget exhausted, retained for next batch
- E_ROUND_OUT_OF_RANGE @ R9 calibration: replacement round out of range [INJ-REPLACE-01]
- E_CALIB_BUDGET_PENDING @ R9 evidence: segment budget exhausted, retained for next batch
- E_CALIB_BUDGET_PENDING @ R10 evidence: segment budget exhausted, retained for next batch
- E_CALIB_BUDGET_PENDING @ R11 evidence: segment budget exhausted, retained for next batch
