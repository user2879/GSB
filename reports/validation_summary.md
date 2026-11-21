# 候选框交换与跨帧匹配验证报告

- 数据集种子：`20261122`
- 序列数：`3`
- 帧数：`21`
- 候选框数：`99`
- 检查总数：`18`
- 通过：`18`
- 失败：`0`
- 总状态：`PASS`

## 关键结果

- 单线程 / 多线程状态一致：`True`
- 增量 / 空白全量最终状态一致：`True`
- 增量缓存键账本投影一致：`True`
- v2→v1 坐标偏差直方图桶数：`4`
- v2→v1 最大坐标偏差：`0.4000000000000057`
- v2→v1 最大得分偏差：`0.0002999999999999947`
- 类别映射完整还原：`True`
- 注入项命中：`7/7`
- 强制终止后恢复点：`9`
- 恢复前存在最终退出报告：`False`

## 编辑序列

| 操作 | 帧复用 | 帧重算 | 轨迹复用 | 轨迹重算 |
| --- | ---: | ---: | ---: | ---: |
| EDIT-001 switch_version | 0 | 21 | 0 | 22 |
| EDIT-002 replace_frame_candidates | 20 | 1 | 17 | 5 |
| EDIT-003 replace_image | 20 | 1 | 17 | 5 |
| EDIT-004 set_iou_threshold | 21 | 0 | 0 | 32 |
| EDIT-005 set_pool_capacity | 0 | 21 | 0 | 32 |
| EDIT-006 replace_frame_candidates | 20 | 1 | 27 | 5 |

## 检查明细

- `PASS` `concurrency_state_equivalence`
- `PASS` `failure_order_stable`
- `PASS` `pool_reentry_observed`
- `PASS` `unchanged_run_uses_cache`
- `PASS` `incremental_state_equivalence`
- `PASS` `incremental_ledger_projection`
- `PASS` `edit_sequence_invalidation_semantics`
- `PASS` `incremental_does_not_reuse_affected_tracks`
- `PASS` `conversion_v1_to_v2_exact`
- `PASS` `conversion_v2_to_v1_residual_captured`
- `PASS` `conversion_class_mapping_restored`
- `PASS` `score_residual_histogram_nonzero`
- `PASS` `version_mismatch_isolation`
- `PASS` `injection_all_items_hit_with_business_codes`
- `PASS` `injection_failed_frames_excluded_from_tracks`
- `PASS` `injection_continues_other_frames`
- `PASS` `forced_termination_and_snapshot_recovery`
- `PASS` `recovered_state_matches_full_run`

## 说明

- 进程退出码只表达运行状态：`0` 成功、`2` 配置/IO 错误、`3` 未预期异常、`75` 模拟强制终止。
- `E_xxx` 业务码只写入 JSON 报告或 stderr，不参与进程退出码。
- 全量重建与增量运行比较的是最终帧、轨迹、偏差和失败清单；`reused` 操作标记按定义不同，因此以最终 cache-key 账本投影比较可审计依赖关系。
- 强制终止测试通过独立 Python 进程执行，终止前不会生成最终验证报告。
