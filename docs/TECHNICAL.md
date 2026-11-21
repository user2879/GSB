# 技术设计与语义说明

## 1. 范围与运行约束

工作台是纯本地 Python 应用，运行时依赖只有：

- Python 3.10+
- NumPy
- PyYAML

不使用 Docker、数据库服务、远程数据集、远程字体或 CDN。交互页面的 HTML/CSS/JS 全部由 `clustering_workbench/static/` 提供。Windows 上使用 `pathlib`、`os.replace` 和线程锁完成缓存与快照操作，不依赖 `fcntl` 等 POSIX 专用能力。

## 2. 组件与调用关系

```text
app.py
  ├─ run     → WorkbenchPipeline.run()
  ├─ verify  → 两个隔离输出目录，串行/并行各跑一次
  └─ serve   → WorkbenchServer → /api/run /api/verify /api/report

WorkbenchPipeline
  ├─ config.py         YAML + 默认值 + overrides + 稳定业务码
  ├─ data.py           确定性数据生成 + 目标结构校验
  ├─ preprocess.py     编码/缺失填充/标准化或分位数 + 指纹
  ├─ methods.py        KMeans / DBSCAN / GaussianMixture
  ├─ metrics.py        轮廓/CH/密度 LL + 边界判定 + 排名
  ├─ cache.py          线程安全 LRU 块缓存
  ├─ snapshots.py      四阶段版本化快照 + 原子指针发布
  ├─ artifacts.py      跨运行内容指纹寻址缓存
  └─ injections.py     故障命中、隔离与业务码映射
```

## 3. 一次运行的时序

```text
load config + overrides
  ↓
validate concurrency / global budget / snapshot identity
  ↓
generate deterministic table
  ↓
resolve preprocess identity → snapshot / artifact / rebuild
  ↓
normalize candidates → inject / validate → stable-key sort
  ↓
pre-commit budget grants in stable-key order
  ↓
resolve each candidate cluster identity → snapshot / artifact / thread-pool execution
  ↓
resolve each evaluation identity → snapshot / artifact / rebuild
  ↓
inclusive boundary injection → fixed-key ranking
  ↓
diagnostic summary + reuse ledger
  ↓
commit metrics stage → commit diagnostics stage → atomic manifest
  ↓
latest_report.json + latest_core.json
```

## 4. 配置、数据与加载语义

### 4.1 数据描述

`samples/demo_dataset.yaml` 的 `dataset.columns` 支持：

- `numeric`, `distribution = normal | uniform | lognormal | beta | constant`
- `categorical`, 固定 `categories` 或按 `cardinality + prefix` 生成
- `derived`, 从已生成列按乘数/偏移派生
- 按列 `missing_rate`
- 目标列 `latent_target: true` 与 `group_labels`
- 数据集级 `latent_groups`，用于生成真实潜在结构

每列 RNG 由 `SHA256("column:<seed>:<name>")` 派生，因此列生成不依赖其他列的抽样次数。整表内容指纹包含固定种子、生成行数、样本 ID、全部列值和目标列。

### 4.2 加载隔离

以下问题不会终止其余列加载：

| 场景 | 业务码 | 行为 |
|---|---|---|
| `include_columns` 或 `expected_types` 引用未知列 | `CFG_UNKNOWN_COLUMN` | 跳过该列，继续其余列 |
| `expected_types` 与数据描述类型不同 | `CFG_TYPE_MISMATCH` | 跳过该列，继续其余列 |
| 目标结构或目标列缺失 | `CFG_TARGET_STRUCTURE_MISSING` | 记录失败，特征加载继续 |
| `n_clusters`/`n_components <= 0` | `CFG_CLUSTER_ZERO` | 仅拒绝该候选 |
| `max_workers <= 0` | `CFG_CONCURRENCY_NONPOSITIVE` | 记录失败，运行时退化为 1 |
| `global_iteration_budget <= 0` | `CFG_GLOBAL_BUDGET_NONPOSITIVE` | 记录失败，候选无预算 |

## 5. 预处理与指纹

### 5.1 处理顺序

1. 按配置顺序筛选列；未知列/类型不符列跳过。
2. 类别值按字典序映射为整数编码；缺失保持 `NaN`。
3. 训练段为固定行序的前 `round(n × train_fraction)` 行。
4. 每列使用训练段中位数填充缺失值。
5. 在训练段上检测完全相同的冗余列，只保留首次出现列。
6. 对保留列执行以下之一：
   - `standard`: `(x - mean) / max(std, epsilon)`
   - `quantile`: 在固定分位网格上取得参考值，再用分段线性 CDF 映射到 `[0,1]`
7. 保留填充值、均值、标准差、方差、分位参考值、类别映射、训练 mask 和删除列信息。

所有行累加均按升序行索引执行。缓存块大小与缓存容量解耦，保证不同驻留预算下浮点累加顺序完全相同。

### 5.2 指纹链

```text
dataset_content_fp = H(seed, rows, sample_ids, columns, target)
dataset_schema_fp  = H(descriptor, content_fp, types, nonfinite_locations)

preprocess_param_fp = H(version, method, train_fraction, drop_redundant,
                        epsilon, quantile_points, selected_columns)
preprocess_content_fp = H(dataset_schema_fp, preprocess_param_fp)
preprocess_artifact_fp = H(content_fp, matrix_hash, stats, dropped_columns, train_mask)

cluster_identity_fp = H(preprocess_artifact_fp, method, params,
                        admitted_iteration_grant, algorithm_version)
cluster_artifact_fp = H(cluster_identity_fp, label_hash, status,
                        iterations, trajectory, diagnostics, flags)

evaluation_identity_fp = H(cluster_artifact_fp, metric_names,
                           density_reg_covar, primary_metric)
evaluation_artifact_fp = H(identity, label_hash, metrics, flags, boundary_check)

diagnostics_identity_fp = H(all evaluation_artifact_fp, injection_hash,
                            failure_codes, flag_counts)
```

候选复用账本额外明确记录：

- 候选引用的预处理内容指纹与产物指纹；
- 完整超参数集合及其指纹；
- cluster/evaluation 的 identity 与 artifact 指纹；
- `source = rebuilt | artifact | snapshot`；
- 是否发生 identity 变化、是否出现“指纹相同却重建”的审计异常。

## 6. 三种确定性算法

### 6.1 KMeans

- 初始质心：先选择按特征字典序最小的点，再用 farthest-first 选取其余质心；距离并列时选择最小行索引。
- 分配并列：`argmin` 选择最小簇编号。
- 每轮记录簇内平方和 `inertia` 轨迹。
- 收敛条件：标签未变化且最大质心位移 `<= tol`。
- 预算成本：一个训练轮次计 1 次迭代。
- 空簇：保留旧质心、保留空簇标记，不删除候选。

### 6.2 DBSCAN

- 按行索引升序计算邻域，邻域数组按点索引升序排列。
- 核心点条件：`neighbor_count >= min_samples`。
- 从最小未访问点开始 BFS/队列扩展；边界点归入当前簇。
- 预算成本：一次完整密度扫描计 1 次迭代；诊断中保留实际点扫描数。
- 全噪声：输出全 `-1` 标签、`ALL_NOISE` 标记，但候选仍参与排名表。

### 6.3 GaussianMixture

- 初始均值与 KMeans 使用同一 farthest-first 规则；初始责任为一热编码。
- 支持 `full`, `diag`, `spherical`, `tied` 四种协方差。
- 每轮 E-step 使用 log-sum-exp；M-step 按责任度更新权重、均值和协方差。
- 所有协方差在求逆/行列式前加入 `reg_covar`；数值异常时追加固定 `1e-8` 对角抖动。
- 收敛条件：相邻轮总对数似然差 `<= tol`。
- 预算成本：一个 EM 轮次计 1 次迭代。
- 空组件：保留旧均值、标记 `EMPTY_CLUSTER`；协方差正则事件记入诊断。

## 7. 指标与排名

### 7.1 指标

- 轮廓系数：排除 DBSCAN 噪声点；簇数 `<2` 或有效点数不足时为 `None`。
- Calinski-Harabasz：排除噪声点；计算类间/类内离散度，退化结构为 `None`。
- 密度对数似然：
  - GMM 使用最终混合模型逐样本对数似然均值；
  - 其他方法使用候选硬归属对应的条件高斯密度，正则化后求均值；
  - 噪声点使用全局正则高斯密度。

### 7.2 排名

主指标降序排序；`None` 永远排在有效分数之后。同分时按：

```text
(method_name, canonical_hyperparameter_json, occurrence_index)
```

升序决定。空簇、全噪声或指标未定义候选不会被丢弃，只是保留诊断标记并可能排在末尾。

## 8. 预算调度

### 8.1 稳定键

```text
stable_key = "<method>:<sha256(canonical_normalized_params)[:16]>"
```

重复的完全相同候选追加确定性 occurrence 后缀。预算分配、线程提交和报告中候选集合均按稳定键排序。

### 8.2 预提交模型

调度器在启动线程池前按稳定键顺序执行预算准入：

```text
remaining = global_budget
for candidate in sorted_by_stable_key:
    grant = min(per_candidate_quota, remaining)
    if grant == 0:
        abort(candidate, SCHED_GLOBAL_BUDGET_EXHAUSTED)
    else:
        remaining -= grant
        candidate.iteration_limit = grant
```

因此“预算扣减序列”与线程实际调度顺序无关。报告同时区分：

- `budget_reserved`: 预提交的迭代额度总和；
- `iterations_executed_total`: 算法实际执行的迭代总和；
- 候选级 `grant / iterations_executed / limit_reason`。

未收敛候选在额度耗尽时保留已得标签，并记录：

- `SCHED_CANDIDATE_QUOTA_HIT`
- `SCHED_GLOBAL_BUDGET_TRUNCATED`

## 9. 块缓存

`ResidentBlockCache` 是线程安全 LRU：

- 无论命中或未命中，每次访问都递增全局访问序号。
- 超预算时淘汰 `min(last_access, key_lexicographic)`。
- 相同访问序号下按块键字典序决断。
- 被淘汰块再次访问时重新调用确定性 compute 回调；结果必须一致。

评估器用同一个 `scheduler.block_size` 访问特征块和指标分数块。缓存容量只改变命中/淘汰/重算记录，不改变指标数值。

## 10. 增量失效矩阵

| 变化 | 失效范围 |
|---|---|
| 某方法超参数 | 该候选 cluster → 其 evaluation；诊断汇总重算 |
| 预处理方式 | preprocess → 全部 cluster → 全部 evaluation → 诊断 |
| 替换部分样本 | dataset fp → preprocess → 全部 cluster/evaluation → 诊断 |
| 预算额度/单候选配额 | 受影响候选的 cluster identity 及下游 |
| 注入清单 | 快照身份失效；注入报告与诊断重算 |
| 仅并发度 | 配置哈希变化使快照失效；计算产物保持相同 digest |

同一运行内，未失效候选必须从 artifact/snapshot 直接加载。账本中的 compute call 计数可证明未发生重算。

## 11. 阶段快照

阶段顺序固定为：

```text
preprocess → clusters → metrics → diagnostics
```

每次阶段提交：

1. 将 `payload.json` 和二进制矩阵写入 `snapshots/active/.tmp/...`。
2. 内容寻址得到版本目录 `stages/<stage>/v-<hash>`。
3. 原子重命名临时目录为不可变版本目录。
4. 原子替换 `stage-pointers/<stage>.json`，发布新版本。
5. 原子重写 `manifest.json`。

manifest 记录：

- `global_version`
- `config_hash`
- `injection_hash`
- `completed_candidates`
- `budget_consumed`
- `budget_reserved`
- `iterations_executed`
- `stage_versions`

启动时若任一身份字段不一致，活动快照整体归档到 `snapshots/invalidated/`，新快照从空状态开始，并写入 `SNAPSHOT_VERSION_INVALID`。不会静默合并旧清单。

## 12. 异常注入

| 注入类型 | 目标阶段 | 业务码 | 隔离动作 |
|---|---|---|---|
| `non_finite_feature` | preprocess | `INJ_NON_FINITE_FEATURE` | 定位样本，以训练统计量替换并继续 |
| `hyperparameter_out_of_range` | cluster | `INJ_HYPERPARAMETER_OUT_OF_RANGE` | 仅拒绝该候选 |
| `threshold_boundary_equal` | evaluation | `INJ_THRESHOLD_BOUNDARY_EQUAL` | 将阈值设为候选分数，验证 `>=` 包含边界 |
| `data_size_over_limit` | load | `INJ_DATA_SIZE_OVER_LIMIT` | 截断到上限后继续生成与运行 |

每条注入报告包含 `id`、`type`、`stage`、`business_code`、`hit`、`isolated` 和 `evidence`。报告按注入编号排序。

## 13. 失败清单与一致性

失败对象排序键：

```text
(stage, item, code, message)
```

该排序与线程完成顺序、缓存淘汰顺序和预算耗尽时机无关。串行/并行 A/B 校验比较：

- `determinism_digest`
- ranking
- 每候选 labels
- budget log
- 完整 failures 列表

`determinism_digest` 有意排除仅描述“从何处续跑”的快照身份告警；这些告警仍完整保留在失败清单和快照页。

## 14. HTTP API

| Method | Path | 用途 |
|---|---|---|
| GET | `/api/health` | 健康检查 |
| GET | `/api/config` | 原始配置、解析配置与样本尺度选项 |
| GET | `/api/report` | 最近一次完整报告 |
| GET | `/api/codes` | 业务码目录 |
| POST | `/api/run` | 用界面参数覆盖后运行 |
| POST | `/api/verify` | 两个临时工作区执行串行/并行校验 |

`/api/run` 支持覆盖预处理方式、样本倍率、线程数、全局预算、单候选配额、缓存块数、快照版本和注入开关。所有覆盖随后进入配置哈希。

## 15. 已知边界与解释

- 迭代预算采用“预提交”语义，而不是运行中以完成顺序动态扣减；这是保证并发与串行预算序列一致的核心设计。
- DBSCAN 不是迭代训练算法，因此一次完整密度 pass 计为 1 个预算单位，同时单独记录点扫描数。
- 轮廓系数使用精确两两距离，适合本工作台的中小规模合成表；大规模数据应另加采样策略。
- 快照以不可变版本目录 + 原子指针发布实现“整体替换”；旧版本保留用于审计，不会就地修改。
- 报告含时间戳和耗时，不参与 `determinism_digest`；计算核心、指纹与标签参与 digest。
