# 需求与实现对照

| 需求 | 实现位置 | 验证/产物 |
| --- | --- | --- |
| 语料逐行校验、唯一段落 ID、非空文档/序号、受控词表 | `dense_index/corpus.py` | `tests/test_pipeline.py:test_02`，`outputs/failure_manifest.json` |
| YAML 配置与枚举/范围校验 | `dense_index/config.py` | `FI-002`、`FI-003` |
| 命令流结构校验 | `dense_index/corpus.py:load_command_stream` | `dense_index/commands.py` |
| v1/v2/v3 回执字段校验 | `dense_index/receipts.py` | `test_02`、`FI-004` |
| 滑窗分块、末块补位与标注 | `dense_index/chunking.py` | `test_03`，块字段 `padded/padding_count` |
| 子词 n-gram、FNV 哈希桶、计数向量 | `dense_index/features.py` | `test_03` |
| 配置指纹派生固定投影矩阵 | `dense_index/features.py:make_projection_matrix` | 两次构建 `canonical_dumps` 相同 |
| 定点整数、步长、残差、裁剪坐标 | `dense_index/quant.py:quantize_vector` | 块记录 `fixed_vector/quant_step/residuals` |
| 符号位分桶与零/半步决胜记录 | `dense_index/quant.py:bucket_key_from_vector` | `sign_tie_breaks`，`FI-008` |
| 量化档位真实改变整数、桶键、摘要 | `QUANT_LEVEL_FRACTION_SHIFT`，`dense_index/engine.py` | `test_04` |
| 三版本归一、来源版本和缺失字段 | `dense_index/receipts.py:normalize_receipt` | `outputs/unified_ledger.jsonl` |
| 升降级旁路元数据与无损往返 | `convert_with_sidecar`、`restore_from_sidecar` | `test_05` |
| v3 摘要和重复引用确定权威回执 | `build_unified_ledger` | `test_05`，`authority_reason=v3_content_digest_match` |
| 分层缓存键和精确失效/复用清单 | `dense_index/engine.py:_build_invalidation` | `outputs/invalidation_report.json` |
| 替换块失效引用桶但复用同桶兄弟块 | `sibling_reuse` | `cmd-003-replace-block` |
| 投影维度稳定前缀切换不重新投影未变块 | `projection_pool_dim` 与 `projection_cache_key` | `cmd-007-projection-dim` 重投影数为 0 |
| 定点余弦、阈值边界和字典序 | `dense_index/quant.py:diagnose_pair`，`dense_index/engine.py:run_recall` | `recall_report.json` |
| 高一位宽参考分数与差值 | `reference_similarity` | 每个 `boundary_blocks` 条目 |
| 边界翻转码和原因分类 | `E_BOUNDARY_FLIP` | `FI-007`、`FI-008`、`test_10` |
| 14 条命令增量/全量逐字段一致 | `dense_index/replay.py:run_stream_equivalence` | `equivalence_report.json` |
| 额外随机有效命令序列对拍 | `compare_command_sequence` | `test_11` |
| 投影点预算与稳定待办 | `IndexEngine.run_batch` | `FI-009`，`cmd-014-budget-pending` |
| 线程池、并发上限、稳定任务键和顺序输出 | `dense_index/concurrency.py`，块字段 `task_keys` | `test_08` |
| 阶段临时目录、整体提交、版本/配置哈希 | `dense_index/snapshot.py` | `outputs/incremental/snapshots` |
| 启动恢复与截断快照回退 | `SnapshotStore.load_current/recover` | `test_07`、`FI-006` |
| 非有限向量隔离 | `assert_finite_vector` 与注入配置 | `FI-005` |

## 缓存键层次

```text
content_cache_key   = H(content_fingerprint + chunk_fingerprint)
feature_cache_key   = H(content_fingerprint + feature_fingerprint)
projection_cache_key= H(feature_cache_key + projection_fingerprint)
fixed_cache_key     = H(projection_cache_key + projection_dim + quant_fingerprint)
bucket_cache_key    = H(fixed_cache_key + bucket_fingerprint + bucket_key)
membership_key      = H(bucket_id + ordered bucket_cache_keys + bucket_fingerprint)
ledger_cache_key    = H(entry + upstream membership key + param_fingerprint + quant_level)
```

投影维度切换使用固定的稳定坐标前缀：引擎首次计算到 `projection_pool_dim=32`，配置在已验证的 `8/12/16/24` 维度间切换时只截取前缀，因此桶位所用的前 `bucket_bits` 个符号坐标不变即可直接复用投影，不重新执行矩阵乘法。

## 一致性比较范围

每条命令稳定后，会从空白重放同一命令前缀并比较：

- 块边界、词项、补位和内容摘要；
- 特征计数、投影向量和定点向量；
- 量化步长、残差、决胜记录和向量摘要；
- 分桶键、分桶成员及顺序；
- 统一账本及权威标记；
- 查询召回、边界诊断和排序；
- 快照内规范化视图。

增量执行报告中的重算/复用列表还会检查为全局稳定顺序的规范子序列。
