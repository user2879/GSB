# 架构与一致性设计

## 1. 模块边界

| 模块 | 职责 |
| --- | --- |
| `config_loader.py` | YAML/JSON 解析、正整数校验、特征 DAG 拓扑排序、计划结构校验 |
| `data_loader.py` | JSONL 逐行解析、缺失标记、有限数、行标识唯一、同实体严格递增 |
| `plan.py` | 追加、修正、窗口调整、批次回滚与最终状态物化 |
| `compute.py` | 拓扑序窗口计算、块键、持久块复用、汇总与跳过明细 |
| `cache.py` | 线程安全 LRU，访问序号与块键字典序淘汰 |
| `leases.py` | 分片映射、签发、冲突等待、续期、到期回收、工作单元接管 |
| `engine.py` | 稳定配额调度、线程池、精确失效、快照恢复与报告 |
| `snapshot.py` | 不可变代际、阶段文件、清单哈希、原子 `CURRENT` 发布 |
| `injections.py` | 外部清单驱动的真实运行时故障隔离 |

## 2. 确定性窗口计算

对实体 `e` 的特征 `f` 和目标行 `r`，候选源行集合为：

```text
W(e, r, f) = { s | s.entity = e,
                   r.timestamp - f.window_ms <= s.timestamp <= r.timestamp }
```

候选行按实体 ID、时间戳、行 ID 升序读取。缺失值不进入以下统计：

- `count`: 非缺失值数量；
- `sum`: `math.fsum` 非缺失值，空集为 `0.0`；
- `mean`: 非缺失和/非缺失数量，空集为 `null`；
- `latest`: 时间戳最大的非缺失源值，空集为 `null`。

每个结果记录 `source_row_ids`、`upstream_features`、`missing_sources`、块指纹和租约版本。缺失跳过明细由最终块集合去重生成，因此缓存淘汰造成的物理重算不会重复污染跳过清单。

## 3. 内容寻址块键

块键为 SHA-256 规范化 JSON，载荷包括：

```text
entity_id
feature name
target row_id / timestamp_ms
feature parameter fingerprint (window, aggregation, source, depends_on)
stable entity lease version
missing marker
for each source row in window:
  row_id, timestamp_ms
  raw value token       # 原始字段特征
  or upstream block key # 上游特征依赖
```

原始浮点值用 `float.hex()` 规范化，避免文本格式化差异。租约版本是稳定的“实体租约纪元”（实体 ID + 分片 + 固定纪元标记），不随每次签发/续期变化；样本内容变化由源行内容指纹表达，从而允许追加后旧窗口继续复用。

物理缓存是全局 LRU。`get` 更新访问序号；超出块数预算时淘汰 `(last_access_sequence, block_key)` 最小项。同一计算任务内，持久快照块只在逻辑块首次访问时尝试复用；之后若被 LRU 淘汰再次引用，会真实重算，且块键和值与首次计算相同。

## 4. 精确失效

计划编辑成功后，在调度前计算失效块集合：

- **追加**：只新增该实体的新目标时间戳 × 全部特征块；旧窗口键不变，可复用。
- **单行数值修正**：对每个原始字段特征，删除目标时间戳落在 `[t, t + window_ms]` 的块；沿拓扑序向下游传播，每个受影响上游行 `s` 再删除下游目标时间戳落在 `[s, s + downstream_window]` 的块。
- **窗口调整**：先求被调特征的依赖闭包；全部实体重新进入调度，闭包内特征全部目标块失效。独立特征继续复用。
- **行 ID/实体/时间戳变更**：保守地清空新旧实体全部派生值。
- **批次回滚**：丢弃回滚点之后的派生状态，按计划物化器恢复到目标批次，全部实体重新调度。

报告中的 `invalidation_manifest` 记录每批的变更字段、失效块旧指纹、新目标块和原因。`cache.block_executions` 记录实际 `reuse`/`recompute` 事件及复用来源，`cache.reused_blocks` 与 `cache.recomputed_blocks` 是实际块键集合。

## 5. 配额与并发调度

1. 待办实体按实体 ID 升序。
2. 每个实体成本为 `rows × feature_count`。
3. 仅当累计行数和累计点数都不超限时才预订整实体；不拆分实体。
4. 预订顺序在提交线程池之前固定，因此配额扣减序列与线程调度无关。
5. 未容纳实体保留在 `pending`，并记录 `E_BATCH_QUOTA_EXHAUSTED`。
6. 线程池最大并发严格取 `thread_pool_concurrency`；报告记录实测峰值 `max_observed_concurrency`。
7. 结果、失败、注入命中和回滚后的实体顺序都按稳定键排序。

每个实体任务携带 `task_sort_key = "<entity_id>|window_features"`，并先取得该实体租约后才进入计算。

## 6. 租约状态机

```text
ABSENT --acquire--> HELD(owner, token, version, issued_ns, expires_ns)
HELD --renew(now < expires)--> HELD(expires = now + ttl)
HELD --now >= expires, contender--> RECLAIMED --> GRANTED(contender)
HELD --renew(now >= expires)--> RECLAIMED + E_LEASE_EXPIRED
HELD --release--> ABSENT
```

- 冲突申请记录 `E_LEASE_CONFLICT`，包含请求者、持有者、当前单调时钟读数和到期读数，然后等待条件变量，不覆盖有效租约。
- 回收事件包含 `previous_owner`、`reclaimed_at_ns`、`expires_ns`、`basis=monotonic_deadline_reached` 和时钟来源。
- 工作线程有后台续期线程；续期间隔严格小于 TTL。计算提交前再次校验 token 与到期点。
- 测试中的接管等待由竞争线程阻塞完成，不使用固定睡眠断言。

## 7. 快照布局

```text
snapshot-root/
  CURRENT                                  # {"generation":N,"manifest_sha256":"..."}
  generations/gen-00000001/
    COMMITTED
    manifest.json
    state.json
    stages/
      load_validation.json
      window_features.json
      entity_summary.json
  .tmp-*                                   # 未发布，恢复时忽略
```

提交顺序：

1. 创建唯一临时目录；
2. 写 `state.json` 和阶段产物并 `fsync`；
3. 生成包含全局版本、配置哈希、计划哈希、基础输入哈希、完成批次/实体、消耗点数、逐实体租约版本、数据/特征指纹和阶段哈希的清单；
4. 写清单自哈希与 `COMMITTED`；
5. 临时目录重命名为不可变 `generations/gen-N`；
6. 原子替换 `CURRENT`。

恢复先读取 `CURRENT`，校验指针、清单自哈希、`COMMITTED`、`state.json` 哈希、三阶段哈希，再用计划物化器独立重建已完成批次的数据和特征，与快照逐指纹比对。任何不一致均输出 `E_SNAPSHOT_STALE`。

## 8. 失败码

| 码 | 场景 |
| --- | --- |
| `E_ROW_INVALID` | 坏 JSON、重复行标识、空实体、非法时间戳、倒序、非有限数值 |
| `E_FEATURE_DAG_INVALID` | 缺失原始字段/上游、环路、非法窗口或来源 |
| `E_CONFIG_INVALID` | 配额/TTL/并发/缓存/时钟等配置非法；含注入 TTL=0、缓存<1 块 |
| `E_PLAN_INVALID` | 未知计划操作、非法批次/回滚目标 |
| `E_LEASE_CONFLICT` | 有效租约冲突，等待回收 |
| `E_LEASE_EXPIRED` / `E_LEASE_LOST` | 到期后续期或提交前失去租约 |
| `E_BATCH_QUOTA_EXHAUSTED` | 单批行数/点数不足，实体保留待办 |
| `E_SNAPSHOT_STALE` | 版本、配置、计划、租约版本或内容哈希不一致 |
| `E_SNAPSHOT_MISSING` | 明确要求恢复但无已提交代 |
