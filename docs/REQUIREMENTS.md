# 需求追踪矩阵

| # | 业务要求 | 实现位置 | 验证测试 / 命令 |
| --- | --- | --- | --- |
| 1 | 加载层逐项校验、错误入清单不中断 | `loader.py`、`schemas.py`、`errors.py` | `tests/test_loader.py`、`tests/test_schema.py`、`acceptance_check.py` 第 6 节 |
| 2 | 指纹派生固定种子的确定性加权采样，记录采样序号/模板出现次序/来源 | `generator.py`、`utils.derive_seed` | `tests/test_determinism.py` |
| 3 | 四维度 float64 打分、加权总分、ID 字典序平局决断、≥6 位有效数字 | `scorer.py`、`utils.round_sig/format_sig` | `tests/test_determinism.py`、产物 `*_text` 字段 |
| 4 | 子词 Jaccard 近重复分组、代表样本、边界固定规则标注 | `dedup.py`、`tokenizer.py` | `tests/test_dedup.py`（含 Jaccard 恰等于阈值的 merge/split 用例） |
| 5 | 三阶段内容/参数指纹缓存键、模板与槽位引用索引、精确失效 | `pipeline.py`、`generator.build_reference_index`、各模块 `cache_key` | `tests/test_incremental.py::test_invalidation_and_reuse_counts`、`acceptance_check.py` 第 3 节 |
| 6 | 任意批次序列增量结果与全量重建逐字段一致，差异入诊断报告 | `runner.compare_artifacts`、影子全量构造 | `tests/test_incremental.py::test_every_batch_matches_full_rebuild_field_by_field`、`test_built_in_shadow_diagnostic_reports_match` |
| 7 | 临时目录 + 整体替换的阶段快照、版本/哈希清单、恢复点与 `E_CONFIG_MISMATCH` | `snapshot.py`、`runner.resume_check` | `tests/test_snapshot.py`、`tests/test_incremental.py::test_config_mismatch_rejects_resume` |
| 8 | 线程池并发受限、稳定排序键、并发/调度无关顺序、JSON 诊断报告 | `scorer.score_all`、`dedup._compute_edges`、`pipeline.build_report` | `tests/test_concurrency.py`、`tests/test_determinism.py::test_concurrent_and_serial_scoring_identical` |

## 错误码对照

| 错误码 | 触发条件 |
| --- | --- |
| `E_SEED_INVALID` | 种子 JSONL 行非法 JSON、schema 不符、必填缺失、槽位未在模板/答案中出现、种子 ID 或槽位来源 ID 重复 |
| `E_CONFIG_INVALID` | 构造配置 schema/业务校验失败、任务类型无种子、权重引用未知模板或来源、目标数非非负整数、权重和为零、阈值越界等 |
| `E_VOCAB_INVALID` | 词表非数组、空词表、token 重复、切分规则不支持 |
| `E_PARAMS_INVALID` | 打分参数维度不符、权重非非负有限数、权重和为零、长度区间非法 |
| `E_BATCH_INVALID` | 批次计划结构非法、未知操作、批次号重复、批次后无任何正权重覆盖存活模板/来源 |
| `E_CONFIG_MISMATCH` | 续跑时基础输入指纹与快照清单不一致，拒绝续跑 |
| `E_SNAPSHOT_CORRUPT` | 清单无法解析或清单自身指纹校验失败 |
| `OK` | 运行完成（失败清单仍可能包含被隔离的单条加载错误） |
