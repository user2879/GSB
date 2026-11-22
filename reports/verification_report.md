# 表格线性模型正则化实验台验收报告

生成时间：2026-11-23（本地验收运行）  
Python：3.11.0rc1；NumPy：1.26.4；PyYAML：6.0.2  
平台：Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.35

## 验收结论

全部验收项通过。

| 检查项 | 结果 |
|---|---|
| `three_schemes_trained` | PASS |
| `baseline_ok` | PASS |
| `full_cache_hit` | PASS |
| `rerun_semantic_equal` | PASS |
| `penalty_only_retrains_target` | PASS |
| `version_switch_reuses_training` | PASS |
| `single_column_invalidation` | PASS |
| `incremental_equals_full_semantics` | PASS |
| `incremental_equals_full_stage_versions` | PASS |
| `incremental_equals_full_export` | PASS |
| `incremental_equals_full_sidecars` | PASS |
| `exported_v1_reloads_sparse_indices` | PASS |
| `v1_v2_roundtrip_within_tolerance` | PASS |
| `sidecar_restores_named_mix` | PASS |
| `budget_codes_and_order` | PASS |
| `serial_parallel_semantics_equal` | PASS |
| `failure_isolated` | PASS |
| `sigint_exit_code_130` | PASS |
| `sigint_snapshot_committed` | PASS |
| `sigint_resume_equals_full` | PASS |
| `sigkill_has_live_checkpoints` | PASS |
| `sigkill_resume_equals_full` | PASS |
| `stale_resume_code` | PASS |
| `unit_tests` | PASS |

## 样本训练

| model | scheme | status | epochs | RMSE | accuracy@0.5 |
|---|---|---|---:|---:|---:|
| elastic_balanced | elastic_net | max_epochs | 55 | 0.390679 | 0.820896 |
| lasso_compact | l1 | max_epochs | 55 | 0.419514 | 0.791045 |
| ridge_core | l2 | max_epochs | 55 | 0.389773 | 0.805970 |

- 语义摘要：`result_1c716a004f8bb9aef07344ed408bc173bb81fa58eae0529ac1e64aea5b51f138`
- 三个模型均逐轮记录 loss 与系数摘要；重复运行缓存命中且语义摘要不变。
- 最终编辑序列：惩罚系数调整 → 切换 v1 → 替换 `utilization` 单列 → 调整 `ridge_core` 白名单。
- 最终增量与空白全量语义摘要一致：`True`。
- 最终增量与全量导出的 `models.v1.txt`、旁路 sidecar 字节一致。

## 双版本互转

- v2 样例 → v1 定宽行 → 重新加载，系数/截距最大绝对偏差 `0.000500000000`，容差 `0.001`。
- sidecar 包含截断残差、原精度标签与丢弃的命名信息；重新加载后恢复弹性网络 `mix=0.55`。
- 实际导出的 v1 稀疏特征序号可回读：`lasso_compact=[0, 1, 3, 4, 5, 6]`，`ridge_core=[0, 1, 2, 3, 7, 8]`。
- 持续保留的验收导出：`reports/acceptance_models.v1.txt`、`reports/acceptance_sidecars/`。

## 预算、并发与恢复

- 预算：失败顺序 `['E_MODEL_BUDGET_EXHAUSTED', 'E_TRAIN_BUDGET_EXHAUSTED']`；待办 `['ridge_core']`；持久化消耗轮次 `8`。
- 串行/并发语义摘要一致：`True`；转换阶段故障只影响 `lasso_compact`，验证汇总仅含 2 个成功模型。
- SIGINT：退出码 `130`，快照状态 `interrupted`，续跑与全量一致：`True`。
- SIGKILL：持久化 live checkpoint 数量 `2`，续跑与全量一致：`True`。
- 严格恢复在配置变化时返回 `E_SNAPSHOT_STALE`。

## 自动化测试

```
test_validation_metrics_are_finite (test_trainer.TrainerTests.test_validation_metrics_are_finite) ... ok
----------------------------------------------------------------------
Ran 24 tests in 16.619s
OK
```

完整机器可读结果见 `reports/acceptance_results.json`。
