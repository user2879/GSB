# 实测验收报告

- 运行环境：Python 3.11.0rc1，NumPy 1.26.4，
  SQLite 3.37.2；无外部服务、无网络。
- 执行时间基准：本地单调时钟纳秒转毫秒。

## 单元/集成测试
- 用例总数：35
- 结果：全部通过
- 失败：0，错误：0

## 两条估算路径逐字节一致性
- 样本：samples/cluster.json（24 分钟 x 4 服务）
- 需求序列逐字节一致：True
- 分配结论逐分钟逐字节一致：True
- 路径选择：minutes=24，边界=12，
  默认路径=增量（严格大于边界才切换）。

## 裁剪、拒绝量与延迟
- 任一分钟分配总和 <= 全局预算：已由 tests/test_allocator.py 与强预算样本验证。
- 拒绝量恒非负，按 requests - allocated*capacity 计算。
- 定点统计：均值 floor 求和*scale/n，P95 最近秩，最大值精确；
  量化误差上界 = 1 个定点单位（1/1000 毫秒）。
- E_LATENCY：注入时钟样本后 24/24 超预算样本均登记并保留，严格模式抛出 E_LATENCY。

## 强制终止恢复
- 模拟仅提交前 7 分钟后重开：unknown_minutes=7..23，恢复后补齐至 24 分钟，
  重放校验通过，已提交分钟不被覆盖（tests/test_ledger_recovery.py）。

## 异常注入退出码
{
  "E_SCHEMA": 2,
  "E_LENGTH": 3,
  "E_SNAPSHOT": 8
}
（映射：E_SCHEMA=2, E_LENGTH=3, E_SNAPSHOT=8，均写入结构化 JSONL 日志。）

## Tkinter 说明
- 本 Linux 沙箱缺少 libtk8.6，GUI 采用延迟导入；视图模型在无头环境完整测试。
- Windows 标准 python.org 安装包自带 Tcl/Tk，`run_workbench.bat` 或
  `py -m workbench.cli gui` 可直接打开五个标签页。

## 详细测试输出
```text
test_budget_never_exceeded (test_allocator.AllocatorTest.test_budget_never_exceeded) ... ok
test_overflow_trim_reason (test_allocator.AllocatorTest.test_overflow_trim_reason) ... ok
test_quota_and_max_caps_enforced (test_allocator.AllocatorTest.test_quota_and_max_caps_enforced) ... ok
test_quota_trim_reason (test_allocator.AllocatorTest.test_quota_trim_reason) ... ok
test_rejection_non_negative_and_based_on_capacity (test_allocator.AllocatorTest.test_rejection_non_negative_and_based_on_capacity) ... ok
test_stable_ordering_budget_pressure (test_allocator.AllocatorTest.test_stable_ordering_budget_pressure) ... ok
test_committed_payloads_are_canonical_and_digestible (test_end_to_end.EndToEndTest.test_committed_payloads_are_canonical_and_digestible) ... ok
test_full_cli_lifecycle_and_error_codes (test_end_to_end.EndToEndTest.test_full_cli_lifecycle_and_error_codes) ... ok
test_incremental_window_fixed_point_outputs (test_end_to_end.EndToEndTest.test_incremental_window_fixed_point_outputs) ... ok
test_view_model_render_order_is_stable (test_end_to_end.EndToEndTest.test_view_model_render_order_is_stable) ... ok
test_capacity_integer_rule (test_estimator.EstimatorRuleTest.test_capacity_integer_rule) ... ok
test_ceiling_demand_clamps_to_max (test_estimator.EstimatorRuleTest.test_ceiling_demand_clamps_to_max) ... ok
test_repeatability_byte_identical (test_estimator.EstimatorRuleTest.test_repeatability_byte_identical) ... ok
test_zero_latency_capacity_defined (test_estimator.EstimatorRuleTest.test_zero_latency_capacity_defined) ... ok
test_equality_checker_locates_first_diff (test_estimator.TwoPathTest.test_equality_checker_locates_first_diff) ... ok
test_full_and_incremental_byte_identical (test_estimator.TwoPathTest.test_full_and_incremental_byte_identical) ... ok
test_ring_window_state_reused (test_estimator.TwoPathTest.test_ring_window_state_reused) ... ok
test_selection_boundary_is_explicit (test_estimator.TwoPathTest.test_selection_boundary_is_explicit) ... ok
test_length_fault (test_faults.FaultInjectionTest.test_length_fault) ... ok
test_schema_fault (test_faults.FaultInjectionTest.test_schema_fault) ... ok
test_snapshot_fault (test_faults.FaultInjectionTest.test_snapshot_fault) ... ok
test_replay_mismatch_raises_e_replay (test_ledger_recovery.LedgerRecoveryTest.test_replay_mismatch_raises_e_replay) ... ok
test_simulated_crash_leaves_unknown_minutes (test_ledger_recovery.LedgerRecoveryTest.test_simulated_crash_leaves_unknown_minutes) ... ok
test_transactional_minute_commit_and_recovery_unknown (test_ledger_recovery.LedgerRecoveryTest.test_transactional_minute_commit_and_recovery_unknown) ... ok
test_version_evolution_marks_old_version_stale (test_ledger_recovery.LedgerRecoveryTest.test_version_evolution_marks_old_version_stale) ... ok
test_duplicate_service_id_rejected (test_loader.LoaderTest.test_duplicate_service_id_rejected) ... ok
test_min_greater_than_max_rejected (test_loader.LoaderTest.test_min_greater_than_max_rejected) ... ok
test_negative_budget_rejected_first (test_loader.LoaderTest.test_negative_budget_rejected_first) ... ok
test_negative_latency_and_bool_integer_rejected (test_loader.LoaderTest.test_negative_latency_and_bool_integer_rejected) ... ok
test_series_length_mismatch_rejected_as_schema_at_load (test_loader.LoaderTest.test_series_length_mismatch_rejected_as_schema_at_load) ... ok
test_valid_deterministic_load (test_loader.LoaderTest.test_valid_deterministic_load) ... ok
test_budget_violation_keeps_sample_and_logs (test_preview_latency.LatencyTest.test_budget_violation_keeps_sample_and_logs) ... ok
test_fixed_point_stats_order_independent (test_preview_latency.LatencyTest.test_fixed_point_stats_order_independent) ... ok
test_preview_matches_authority_and_never_rewrites_ledger (test_preview_latency.PreviewTest.test_preview_matches_authority_and_never_rewrites_ledger) ... ok
test_preview_query_is_single_cache_read (test_preview_latency.PreviewTest.test_preview_query_is_single_cache_read) ... ok

----------------------------------------------------------------------
Ran 35 tests in 6.494s

OK
```
