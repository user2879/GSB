# 实际验证记录

验证日期：2026-11-22  
工作目录：`/home/zsl/feedback/uidfc5909d3-default-57`

## 1. 环境

```text
Python  3.11.0rc1
NumPy   1.26.4
PyYAML  6.0.2
OS      Linux（代码仅使用 Windows 10/11 同样支持的 pathlib / os.replace / threading）
```

依赖安装命令：

```bash
pip install numpy pyyaml
# 或
pip install -r requirements.txt
```

未使用 Docker、容器、网络数据集或外部附件。

## 2. 自动化验收

命令：

```bash
python -m unittest discover -s tests -v
```

实际输出结论：

```text
Ran 14 tests in 14.963s
OK
```

覆盖矩阵：

| 测试 | 验收点 | 实际结果 |
|---|---|---|
| `test_dataset_generation_is_deterministic_and_contains_requested_shapes` | 固定种子、缺失、高基数、目标结构、冗余 | 通过；两次内容指纹一致 |
| `test_preprocess_is_elementwise_stable_and_method_switch_changes_artifact` | 预处理逐元素一致；切换方式改变指纹/统计量 | 通过；`value_copy` 被识别为冗余列 |
| `test_three_methods_are_deterministic` | 三方法两次运行标签与轨迹一致 | 通过 |
| `test_empty_cluster_and_all_noise_are_retained_for_ranking` | 空簇/全噪声不丢弃 | 通过；两者均保留在排名 |
| `test_gaussian_singular_covariance_is_regularized` | 协方差奇点不抛异常 | 通过；正则系数被记录 |
| `test_serial_and_parallel_are_fieldwise_equivalent_with_budget_pressure` | 低预算下串行/并行逐字段一致 | 通过；完成顺序不同但 digest/排名/标签/预算/失败一致 |
| `test_cache_budget_does_not_change_computational_result` | 缓存 1 块 vs 8 块结果一致 | 通过；1 块配置 eviction/recompute 更多但 digest 相同 |
| `test_lru_eviction_recomputes_identically` | LRU 淘汰后重算一致 | 通过 |
| `test_incremental_reuse_only_rebuilds_affected_candidate` | 仅修改一个 GMM 候选，其余复用 | 通过；preprocess=0，cluster=1，evaluation=1 次计算 |
| `test_partial_sample_replacement_invalidates_preprocess_and_all_candidates` | 替换单样本导致全链失效 | 通过 |
| `test_snapshot_resume_and_version_mismatch_are_fail_closed` | 快照续跑和版本失效 | 通过；第二次运行 cluster/evaluation 计算次数均为 0 |
| `test_all_injection_types_hit_and_map_to_business_codes` | 四类注入命中及业务码 | 通过；4/4 命中 |
| `test_validation_business_codes_do_not_abort_other_columns` | 加载/配置错误隔离 | 通过；其余列继续加载 |
| `test_missing_target_structure_emits_stable_code` | 目标结构缺失业务码 | 通过 |

## 3. 演示运行结果

命令：

```bash
python app.py run --config configs/demo.yaml
```

关键结果：

```text
status                  completed_with_failures
determinism_digest      det-8ee26bf9bc55b0c3b69d6db9
generated rows          360（INJ-004 将请求上限压到 360）
feature matrix          360 × 8
dropped redundant       value_copy ≡ order_value
preprocess content fp   pre-436a8e6862ca77623c9446c4
preprocess artifact fp  preart-3d35307810157769d3d1e741
budget reserved         126
iterations executed     72
global remaining        24
injection hits          4 / 4
```

排名前 7 个可计算候选：

| Rank | Method | Silhouette | Calinski-Harabasz | Density LL | Flags |
|---:|---|---:|---:|---:|---|
| 1 | GaussianMixture | 0.364800 | 234.283 | -7.9632 | — |
| 2 | KMeans | 0.308923 | 157.090 | -6.7311 | — |
| 3 | GaussianMixture | 0.261438 | 170.862 | -7.4811 | — |
| 4 | KMeans | 0.228951 | 118.935 | -6.4742 | — |
| 5 | GaussianMixture | 0.223783 | 135.621 | -7.0528 | — |
| 6 | DBSCAN | 0.208905 | 48.205 | -4.2228 | — |
| 7 | DBSCAN | — | — | — | `ALL_NOISE`, `METRIC_UNDEFINED` |

第 7 名证明全噪声候选没有被静默丢弃。另有一个 KMeans 候选被注入为 `n_clusters=0`，状态保留为 `invalid`，不影响其他候选。

## 4. 串行/并行 A/B 一致性

命令：

```bash
python app.py verify --config configs/demo.yaml
```

实际输出：

```text
digest_equal=True ranking_equal=True labels_equal=True budget_equal=True failures_equal=True
serial_digest=det-8ee26bf9bc55b0c3b69d6db9
parallel_digest=det-8ee26bf9bc55b0c3b69d6db9
```

自动化测试同时观察到串行与并行任务的 `completion_order` 不同，但：

- 候选排名一致；
- 每样本硬归属一致；
- 指标表顺序一致；
- 预算扣减序列一致；
- 失败清单顺序一致。

低预算测试使用 `global_iteration_budget=20`、`per_candidate_iteration_quota=4`，出现 `budget_aborted` 候选；上述字段仍完全一致。

## 5. 预算调度实测

演示配置在 stable-key 顺序下预提交预算：

```text
sequence  candidate                                  grant  after  reason
1         DBSCAN:6fc44acd7d692d29                   18     132    within_request
2         DBSCAN:d22a62a3bd68de4a                   18     114    within_request
3         GaussianMixture:0ac8a06906a581e8          18      96    candidate_quota
4         GaussianMixture:7d4c1d585ee7d9c4          18      78    candidate_quota
5         GaussianMixture:cc92a79dd0dc08b6          18      60    candidate_quota
6         KMeans:6742f3d767cb5380                   18      42    candidate_quota
7         KMeans:cc8f7d0114d4e2c1                   18      24    candidate_quota
```

三个 GMM 候选在单候选配额处截断，仍输出硬归属、轨迹与指标，并记录 `SCHED_CANDIDATE_QUOTA_HIT`。

## 6. 快照与恢复

同一输出目录连续运行两次：

```bash
python app.py run --config configs/demo.yaml
python app.py run --config configs/demo.yaml
```

第二次运行：

```text
preprocess.source       snapshot
snapshot.valid          True
snapshot.resume         True
snapshot_resume_calls   15
cluster_compute_calls   0
evaluation_compute_calls 0
reuse status            reused=15, rebuilt=1（仅诊断汇总）
determinism_digest      与首次运行相同
```

`completed_candidates` 在 manifest 中保留 7 个有效候选。将 `snapshot.version` 改为 2 后，旧活动目录被移动到 `outputs/snapshots/invalidated/`，失败清单出现：

```text
SNAPSHOT_VERSION_INVALID
```

没有静默沿用旧状态。

## 7. 增量复用实测

测试场景：

1. 先以原配置运行并填充 artifact/snapshot。
2. 仅把第一个 GMM 候选从 `n_components=2, covariance_type=full` 改为 `n_components=2, covariance_type=diag`。
3. 在已有输出目录重跑，再在全新目录全量运行同一新配置。

实际结果：

```text
preprocess_compute_calls     0
cluster_compute_calls        1
evaluation_compute_calls     1
reuse status                 reused=13, rebuilt=3
incremental digest == fresh digest
incremental ranking == fresh ranking
```

其中 3 个 rebuilt 是：受影响候选 cluster、对应 evaluation、诊断汇总。其余候选均从 artifact 直接复用；快照因配置哈希变化记录 `SNAPSHOT_VERSION_INVALID`，但不影响计算结果的 digest。

替换单个样本 `S00001.order_value = 987.5` 后，数据集、预处理内容指纹发生变化，全部 7 个可执行候选与评估均重建。

## 8. 缓存预算实测

自动化测试对比 `cache_resident_blocks=1` 与 `=8`：

```text
determinism_digest_equal     True
small_cache_evictions        > large_cache_evictions
small_cache_recomputes       > 0
```

原因：分块大小固定取自 `scheduler.block_size=64`，驻留容量变化只影响淘汰与重算次数，不改变距离/指标累加顺序。

## 9. 注入命中与业务码

演示注入清单实际命中 4/4：

| ID | 类型 | 阶段 | 业务码 | 结果 |
|---|---|---|---|---|
| INJ-001 | `non_finite_feature` | preprocess | `INJ_NON_FINITE_FEATURE` | `S00017.random_score` 注入后以训练中位数替换 |
| INJ-002 | `hyperparameter_out_of_range` | cluster | `INJ_HYPERPARAMETER_OUT_OF_RANGE` | 一个 KMeans 候选被拒绝，其余继续 |
| INJ-003 | `threshold_boundary_equal` | evaluation | `INJ_THRESHOLD_BOUNDARY_EQUAL` | 阈值与分数均为 `0.22378254643281548`，inclusive `>=` 接受 |
| INJ-004 | `data_size_over_limit` | load | `INJ_DATA_SIZE_OVER_LIMIT` | 361 行请求截断为 360 行后继续 |

INJ-003 的实际证据：

```json
{
  "score": 0.22378254643281548,
  "threshold": 0.22378254643281548,
  "delta": 0.0,
  "operator": ">=",
  "decision": "accept",
  "boundary_hit": true
}
```

## 10. 配置业务码实测

专项测试一次性构造未知列、类型不符、零簇数、非正并发、非正全局预算，并确认不存在提前中断：

```text
CFG_UNKNOWN_COLUMN
CFG_TYPE_MISMATCH
CFG_CLUSTER_ZERO
CFG_CONCURRENCY_NONPOSITIVE
CFG_GLOBAL_BUDGET_NONPOSITIVE
```

移除数据集 `structure` 后另测得：

```text
CFG_TARGET_STRUCTURE_MISSING
```

未知列与类型不符时，预处理仍保留并使用其余合法列。

## 11. 交互控制台冒烟

启动：

```bash
python app.py serve --config configs/demo.yaml --port 8877
```

实际 HTTP 检查：

```text
GET  /api/health       200
GET  /                200（离线 HTML）
GET  /api/config       200，2,639 bytes
GET  /api/report       200，87,367 bytes
GET  /static/app.js    200，33,106 bytes
POST /api/verify       200，串行/并行五项一致性均为 true
```

测试请求中 `injections.hit=4`、`ranking=7`，服务端返回报告并可被页面直接渲染。页面不依赖外部 CDN、字体或图表库。

## 12. 结论

- 三方法在同一输入与配置下重复运行结果一致。
- 串行、并发、低预算和不同缓存容量下，最终计算字段保持一致。
- 增量复用只重建受影响候选及其下游，未失效项计算调用数为 0。
- 阶段快照能够续跑；版本/配置/注入身份不一致时会失效并归档。
- 四类真实故障均已命中，并映射到稳定业务码。
- 全部 14 项自动化验收通过。
