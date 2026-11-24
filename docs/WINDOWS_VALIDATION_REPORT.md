# 实际验证报告

- 生成时间（UTC）：2026-11-24T17:51:19.951379+00:00
- 平台：`Windows-11-10.0.26300-SP0`
- Python：`3.12.16`
- 总结果：通过

## 样本生成

- 段落数：24
- 初始块数：53
- 命令数：14

## 增量与全量等价

- 逐字段一致：`True`
- 预算待办已命中：`True`
- 每条命令均比较块完整向量/定点向量、摘要、分桶成员、召回、账本与快照视图。

## 故障注入

- 命中：9/9；全部命中：`True`
- 业务码：`E_CORPUS_INVALID`、`E_CONFIG_INVALID`、`E_RECEIPT_INVALID`、`E_VECTOR_NONFINITE`、`E_SNAPSHOT_RECOVERY`、`E_BOUNDARY_FLIP`、`E_POINT_BUDGET_PENDING`。

## 自动化测试

- 用例数：11
- 失败：0；错误：0；跳过：0
- 结果：通过

```text
test_01_valid_sample_loads (test_pipeline.PipelineTest.test_01_valid_sample_loads) ... ok
test_02_invalid_input_codes (test_pipeline.PipelineTest.test_02_invalid_input_codes) ... ok
test_03_chunk_projection_deterministic (test_pipeline.PipelineTest.test_03_chunk_projection_deterministic) ... ok
test_04_quant_level_changes_vectors_and_buckets (test_pipeline.PipelineTest.test_04_quant_level_changes_vectors_and_buckets) ... ok
test_05_receipt_roundtrip_and_authority (test_pipeline.PipelineTest.test_05_receipt_roundtrip_and_authority) ... ok
test_06_full_command_stream_equivalence (test_pipeline.PipelineTest.test_06_full_command_stream_equivalence) ... ok
test_07_budget_pending_and_snapshot_resume (test_pipeline.PipelineTest.test_07_budget_pending_and_snapshot_resume) ... ok
test_08_concurrency_order_and_cap (test_pipeline.PipelineTest.test_08_concurrency_order_and_cap) ... ok
test_09_fault_injections_all_hit (test_pipeline.PipelineTest.test_09_fault_injections_all_hit) ... ok
test_10_boundary_diagnostics_include_reference (test_pipeline.PipelineTest.test_10_boundary_diagnostics_include_reference) ... ok
test_11_random_valid_sequences_match_full_build (test_pipeline.PipelineTest.test_11_random_valid_sequences_match_full_build) ... ok

----------------------------------------------------------------------
Ran 11 tests in 21.516s

OK

```

## 解释器说明

- Windows 默认 `py`/`python` 为 Python 3.14.7。
- `py -0p` 同时列出 uv 管理的 `Astral/CPython3.12.16`，实际路径为 `C:\Users\27857\AppData\Roaming\uv\python\cpython-3.12.16-windows-x86_64-none\python.exe`。
- 该受 uv 管理的基础解释器启用了 PEP 668 保护，因此验证时用它创建独立 venv：`C:\temp\dense-index-win-239\.venv`。
- venv 内实际依赖为 NumPy 2.5.5、PyYAML 6.0.4；完整 `run_validation.py` 在真实 Windows 11 上退出码为 0。
