# 运行报告 — 量化卷积算子图编辑与调度应用

## 环境
- 工作目录: /home/zsl/feedback/uidfc5909d3-default-4/qcg
- Python: 3.11.0rc1 (Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.35)
- NumPy: 1.26.4  PyYAML: 6.0.2  pytest: 8.3.3
- 网络/GPU/容器: 均不使用，纯本地 CPU 确定性计算
- 路径: 全部使用相对路径（Windows 兼容，仅用正斜杠与 os.path）

## 样本可复现性
固定种子 20261121，重复执行 scripts/generate_samples.py 产物逐字节一致
（SHA-256 校验和存于 data/quant_pack/sample_checksums.json）:
- `data/graphs_v1/sample_graph.v1.txt`: `d978303b0b919bf3...`
- `data/graphs_v2/sample_graph.v2.json`: `a01856fa73f977ce...`
- `data/quant_pack/sample_pack.le.bin`: `2f0f88ebf6cc2966...`
- `data/quant_pack/sample_shapes.json`: `9c601225aceb9ea1...`

## 测试结果
- pytest 汇总: **46 passed in 1.42s**
- 端到端验证 scripts/verify.py:
  - [PASS] version_roundtrip
  - [PASS] quant_pack_le
  - [PASS] incremental_invalidation
  - [PASS] serial_equals_parallel
  - [PASS] budget_pending_queue
  - [PASS] lru_eviction

## 业务要求对照
1. 双版本解析与统一 IR、E_GRAPH_INVALID 节点/字段定位:
   tests/test_parsers.py（定宽 24 字符、枚举、正有限缩放、无环/边存在）
2. 双向互转与降级旁路、E_LOSSY_DOWNGRADE 序号:
   tests/test_conversion.py（舍入/写回/残差/精度标签精确还原）
3. 整数定点卷积与 E_ACC_OVERFLOW、溢出三策略、确定性重复:
   tests/test_kernel.py
4. 反量化五种舍入、中点边界标记、裁剪、位宽/舍入切换真实改变结果:
   tests/test_dequant.py；边界标记跨版本保持 test_end_to_end.py
5. 内容指纹缓存键与精确下游失效、复用/重算清单:
   tests/test_engine.py::test_automatic_invalidation_without_previous_graph
   及 test_incremental_matches_full_rebuild
6. 全局/单节点/单批三重预算、待算队列、LRU 淘汰与重算审计:
   test_node_quota_exhausted / test_global_budget_pending_queue /
   test_batch_recompute_budget_splits_rounds /
   test_multi_round_budget_resumption / test_eviction_recompute_audit_ledger
7. 并发==串行、增量==全量（张量/反量化/缓存键/账本/队列/失败顺序）:
   test_serial_and_concurrent_identical /
   test_diamond_graph_parallel_wave /
   test_deterministic_budget_deduction_order /
   test_arbitrary_edit_sequence_matches_rebuild
8. 四阶段原子快照、E_CONFIG_MISMATCH 续跑判定、四类注入隔离:
   test_snapshot_staging_and_manifest / test_snapshot_resume_mismatch /
   test_resume_succeeds_with_identical_identity /
   test_injection_isolation / test_pack_truncation_injection /
   test_sidecar_tamper_injection / test_bitwidth_overflow_injection

## 错误码实现
E_GRAPH_INVALID, E_LOSSY_DOWNGRADE, E_ACC_OVERFLOW,
E_NODE_BUDGET_EXHAUSTED, E_GLOBAL_BUDGET_PENDING, E_RECOMPUTE_PENDING,
E_CONFIG_MISMATCH, E_QUANT_PACK_INVALID, E_INJECTION_FAULT
（定义于 qcg/errors.py，故障记录含 node_id/field/stage/injection_id/seq）

## 复现步骤
```bash
cd qcg
python3 scripts/generate_samples.py
python3 -m pytest -q
python3 scripts/verify.py
```
