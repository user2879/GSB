# 诊断报告：01_baseline

- 轮次总数/完成：12 / 12
- 定点位宽/档位：12 / 256
- 复用命中：0，重算轮次：[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
- 预算待办：[]
- 快照恢复：False
- 注入命中：[]
- 失败码统计：{'E_CALIB_BUDGET_PENDING': 8}

## 情感与说话人轨迹

| 轮次 | 情感 | 说话人 | 轨迹段 | 断开原因 |
| --- | --- | --- | --- | --- |
| 0 | calm | S1 | SEG-001 | SHORT_SEGMENT |
| 1 | calm | S2 | SEG-002 | SHORT_SEGMENT |
| 2 | calm | S1 | SEG-003 | SHORT_SEGMENT |
| 3 | happy | S2 | SEG-004 | SHORT_SEGMENT |
| 4 | happy | S1 | SEG-005 | SHORT_SEGMENT |
| 5 | happy | S2 | SEG-006 | SHORT_SEGMENT |
| 6 | calm | S1 | SEG-007 | - |
| 7 | calm | S1 | SEG-007 | - |
| 8 | calm | S1 | SEG-007 | SPEAKER_SWITCH |
| 9 | happy | S2 | SEG-008 | SHORT_SEGMENT |
| 10 | calm | S1 | SEG-009 | SHORT_SEGMENT |
| 11 | calm | S2 | SEG-010 | SHORT_SEGMENT |

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

- E_CALIB_BUDGET_PENDING @ R4 evidence: segment budget exhausted, retained for next batch
- E_CALIB_BUDGET_PENDING @ R5 evidence: segment budget exhausted, retained for next batch
- E_CALIB_BUDGET_PENDING @ R6 evidence: segment budget exhausted, retained for next batch
- E_CALIB_BUDGET_PENDING @ R7 evidence: segment budget exhausted, retained for next batch
- E_CALIB_BUDGET_PENDING @ R8 evidence: segment budget exhausted, retained for next batch
- E_CALIB_BUDGET_PENDING @ R9 evidence: segment budget exhausted, retained for next batch
- E_CALIB_BUDGET_PENDING @ R10 evidence: segment budget exhausted, retained for next batch
- E_CALIB_BUDGET_PENDING @ R11 evidence: segment budget exhausted, retained for next batch
