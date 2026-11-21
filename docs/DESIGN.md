# DriftBench 技术设计

## 1. 数据契约

### 1.1 曝光日志 `exposures.jsonl`

每行一个对象，必需字段：

| 字段 | 约束 |
|---|---|
| `request_id` | 非空字符串，全文件唯一 |
| `experiment_group` | 必须是配置声明的 control 或 experiment 标识 |
| `user_id`, `item_id` | 非空字符串 |
| `rank` | 正整数 |
| `model_score` | 有限数值 |
| `clicked`, `converted` | 整数 0 或 1 |
| `event_ms` | 非负整数 |

非法行产生 `E_LOG_INVALID`，带 `line`、`request_id`、`field_path`，其余行继续加载。

### 1.2 口径清单 `specs.jsonl`

字段为 `spec_version`、`metric_name`、`truncation_rank`、`normalization`、`dedup_count`、`sample_filter`。版本必须唯一；指标支持 `ctr`/`cvr`；归一化支持 `none`、`rank_inverse`、`position_decay`；`sample_filter` 是条件对象数组或字段到条件对象的映射，字段必须是曝光 schema 中存在的字段。错误产生 `E_SPEC_INVALID` 并继续。

### 1.3 配置 `config.json`

必需字段：

- `control_group`, `experiment_group`
- `bucket_hash_seed`, `bucket_count`
- `active_spec_version`
- `fixed_decimal_places`, `quantization_step`
- `concurrency`, `batch_size`
- `snapshot_commit_interval_batches`
- `bucket_property_key`, `fallback_policy`
- `drift_tolerance`, `min_affected_ratio`, `attribution_order`
- `snapshot_version`

`fallback_policy=fixed_group` 时还需 `missing_attribute_group`。任何配置错误都产生 `E_CONFIG_INVALID`（含字段路径）并中止该次加载；`drift_tolerance` 必须大于 0，归因顺序必须是四类来源的一个完整排列。

### 1.4 用户属性与编辑计划

用户属性支持 `{"users": {user_id: {...}}}` 或直接 `user_id → object` 映射。编辑计划的 `batches[].operations[]` 支持：

- `add_exposures`：`events` 数组；
- `fix_exposure`：`request_id` 与非空 `patch`；
- `switch_spec_version`：`spec_version`；
- `change_bucket_seed`：`seed`。

## 2. 分桶与指标

分桶哈希输入为 `seed \x1f user_id \x1f canonical(attribute)`。缺少用户或缺少属性时，按 `user_id_only` 用 `None` 属性重新哈希，或以 `fixed_group` 直接归入配置组。桶奇偶映射为 control/treatment。

指标流水线固定：

1. 样本过滤；
2. `rank <= truncation_rank`；
3. `dedup_count=true` 时按 `(request_id, item_id)` 排序后每个用户保留首条；
4. 归一化权重：`none=1`，`rank_inverse=floor(1e6/rank)`，`position_decay=floor(1e6/log2(rank+1))`；
5. 按指标计算分子/分母权重。

`ctr`：分母为全部参与者权重，分子为点击权重。`cvr`：分母为点击权重，分子为点击且转化权重。比率以固定点整数输出，量化档位作用于固定点单位。

## 3. 漂移与归因

漂移对是活动版本和自然版本序中的直接前驱。每个分组输出：

- 两版本指标、固定点差值；
- 四类来源的 `changed`、受影响事件集合/计数/指纹；
- 所有来源并集的影响占比；
- `significant`、`boundary`、`boundary_samples`。

判定固定为：

```text
boundary    = abs(delta_fixed) == tolerance_fixed
significant = not boundary
              and abs(delta_fixed) > tolerance_fixed
              and affected_ratio_fixed >= min_affected_ratio_fixed
```

归因从旧口径出发，按配置顺序逐项替换一个来源为目标值。每一步都保存指标值、分子/分母权重与参与者数量。由于贡献定义为相邻量化指标的差，贡献和与总差值精确一致。

为判断顺序敏感性，对全部已变化来源做排列。如果同一来源在不同顺序下的贡献极差超过容差，则标记该来源，同时仍采用配置顺序进行决断，并输出 `E_ATTRIB_AMBIGUOUS` 警告。未变化来源显式贡献 0。

## 4. 缓存与失效

| 阶段 | 缓存键主要上游 | 典型失效 |
|---|---|---|
| bucketing | 用户集合、属性值、seed、桶数、回退策略 | seed、属性或新用户变化 |
| metrics | 分组事件内容、口径指纹、定点/量化配置 | 分组事件或口径变化 |
| drift | 两版本 metrics 键、容差、显著阈值 | 任一指标变化或选中版本对变化 |
| attribution | drift 键、归因顺序、容差 | 漂移或归因配置变化 |

缓存键都包含阶段标签、上游内容指纹与相关配置指纹。阶段产物中保存事件引用列表或指纹，指标报告直接给出参与请求集合指纹。

复用/重算日志逐键记录。`ContentCache.get_or_compute` 是唯一入口：命中时不调用计算函数，未命中时首次调用并记录。因此 `actual_compute_calls == len(recomputed)` 可作为一致性不变量。

## 5. 并发确定性

`ParallelRunner` 使用固定上限的 `ThreadPoolExecutor`。每次 `map_tasks` 先按稳定键排序，再提交；收集时再次按稳定键排序。后续累加、去重、来源顺序与失败列表都在单一线程按稳定键完成。并发度仅影响执行时序，不影响事件顺序、浮点/固定点计算顺序或输出。

任务稳定键：

- 分桶：`bucketing|user_id`；
- 指标：`metrics|spec_version|group|request_id|item_id`；
- 漂移：`drift|from>to|group|source`，差值任务使用 `__delta__`；
- 归因：来源按配置顺序串行应用，排列检查与中间指标使用口径指纹缓存。

实际最大活动线程数由锁计数记录，不得超过 `config.concurrency`。

## 6. 快照

```text
snapshot/
  MANIFEST.json
  stages/
    bucketing/versions/<product_version>/entries.json
    bucketing/current -> versions/<product_version>
    metrics/...
    drift/...
    attribution/...
```

提交单个阶段时，先在临时目录写完整 `entries.json`，再原子替换版本目录；随后原子替换 `current` 符号链接；所有阶段完成后原子替换清单。清单记录 `global_version`、`config_fingerprint`、`spec_versions`、活动版本和阶段产物版本。

启动检查清单中的全局版本、配置指纹和口径集合。不匹配时报告 `E_SNAPSHOT_STALE`，不加载旧产物，继续以全量方式重建并在结束时整体提交。

## 7. 异常注入

`python3 -m driftbench injections` 对每个注入项从初始状态重放到指定批次之前，然后在临时样本目录中真正落盘并重新调用加载层：

| 编号 | 注入 | 预期诊断 |
|---|---|---|
| I01 | `NaN` 模型分数 | `E_LOG_INVALID`, `model_score` |
| I02 | 未知实验分组 | `E_LOG_INVALID`, `experiment_group` |
| I03 | 活动口径不存在 | `E_CONFIG_INVALID`, `active_spec_version` |
| I04 | 非法归因顺序 | `E_CONFIG_INVALID`, `attribution_order` |
| I05 | 容差为零 | `E_CONFIG_INVALID`, `drift_tolerance` |

报告逐项记录 `injection_id`、批次、命中布尔、期望/实际错误码与匹配失败对象。

## 8. 验证策略

- 加载：非法行继续、配置中止、错误码/字段路径；
- 固定点与样本语义：截断、去重、过滤、排序、请求集合指纹；
- 漂移：恰好等于容差时边界且不显著；
- 归因：四来源、残差为零、目标复算；
- 增量：初始与 7 个编辑状态逐字段比较，并核对实际复用/重算清单；
- 并发：并发度 1 与 4 指纹一致、活动线程上限；
- 快照：磁盘复用、配置变更陈旧重建；
- 注入：五项全部命中。

测试命令：

```bash
python3 -m unittest discover -s tests -v
python3 -m driftbench all --sample-dir samples --output-dir artifacts --snapshot-dir artifacts/snapshots
```
