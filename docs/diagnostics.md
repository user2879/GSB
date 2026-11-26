# 分片下发数据库侧诊断首版说明

本版本只依赖本机回环 TCP、Python 3.11+、NumPy 1.26.4 与标准库 `sqlite3`；不使用 Docker，不访问外部网络。

## 快速验证

```bash
python3 --version
python3 -m pip install numpy==1.26.4
./scripts/verify.sh
```

也可分步执行：

```bash
python3 -m shard_diag.cli check-env --spec config/spec.json
python3 -m pytest -q
python3 -m shard_diag.cli run --spec config/spec.json \
  --db artifacts/demo-ledger.sqlite3 --reset
```

实际验证输出写入 `artifacts/verification-results.txt`，端到端 JSON 报告写入 `artifacts/run-report.json`。

## 声明与强校验

样本声明位于 `config/spec.json`，结构包括：

- `nodes[]`：`node_id`、`host`、`port`、`concurrency_quota`、`lease_limit`、`seed`、`capacity_version`。
- `shards[]`：`shard_id`、`node_id`、`dependencies`、`sample_rows`、`resource_units`、`priority`。
- `revision_chain[]`：严格递增的主库修订序号。
- `delay_batches[]`：节点、主库修订、从库已回放修订和延迟样本毫秒序列。
- 全局 `global_resource_budget`、`scale_power`、`max_revision_lag`、`dispatch_delay_limit_ms`、`lease_ttl_ms`。

`shard_diag/spec.py` 按声明顺序报告第一个非法对象：异常码为 `E_SPEC`，并带 `object`（JSON 指针或对象标识）与 `field`。校验覆盖 ID 唯一、`host` 必须是字面量 `127.0.0.1`、端口唯一且为高位端口、正整数配额、依赖存在、依赖图无环、修订严格递增、回放进度不超前等。

稳定拓扑序由 `shard_diag/topology.py` 的 Kahn 算法生成：就绪集合使用 `(-priority, shard_id)` 最小堆，因此优先级和分片标识均有确定顺序。

## 确定性样本与聚合

`shard_diag/determinism.py` 用 SHA-256 混合节点种子与分片标识，再派生 NumPy PCG64 种子。样本是固定 `<u8/<i4/<i8` 字段的结构化数组，因此 `ndarray.tobytes()` 可逐位比较。

聚合使用 `np.lexsort` 按 `(group_a, group_b)` 显式排序，再用 `np.add.reduceat` 计算组数、行数、指标和与加权指标和。批次只做切片聚合，父进程按 `(group_a, group_b)` 合并后再次显式排序。任意下发顺序、任意不重叠批次切分和任意并发度均得到相同摘要。

摘要字段绑定 `node_id`、`shard_id`、`sample_rows`、`capacity_version`、`participating_rows`、`sample_digest` 和排序后的分组结果。

## 回环节点子进程与租约

`shard_diag/node_protocol.py` 为每个节点启动：

```bash
python3 -m shard_diag.node_worker --host 127.0.0.1 --port ... ...
```

父子进程之间是逐行 JSON。Worker 使用 `ThreadingMixIn`，支持租约授予、乐观续约、释放、容量 fence 和批次聚合。

`shard_diag/leases.py` 的租约包含 `lease_id`、`shard_id`、`node_id`、`granted_at_ms`、`expires_at_ms`、`renewal_count`、`lease_version` 和状态。时间一律来自 `time.monotonic_ns()` 并向下取整为毫秒；过期只能由真实经过的时间触发，测试通过真实 `sleep` 验证。

续约必须携带所见 `lease_version`。节点保存当前版本并做乐观比较；版本不符返回 `E_LEASE` 和 `current_version`，父进程不延长到期时刻。

## 并行调度与三重约束

`shard_diag/scheduler.py` 使用真实 `ThreadPoolExecutor`。调度线程在同一把锁内原子预留：

- 在途 `resource_units` 之和不超过 `global_resource_budget`；
- 节点活跃分片数不超过 `concurrency_quota`；
- 节点活跃租约预留数不超过 `lease_limit`。

无法满足硬约束的分片保持 `QUEUED`，记录 `wait_started_ms`、`wait_ended_ms` 和 `wait_ms`。依赖未全部成功时只停留在 `PENDING`，不会进入队列。测试用阻塞 Worker 观测峰值，证明全局预算、并发配额和租约上限均未越界。

精确整数账本始终是硬放行边界；误差界只用于标注“边界不可区分”，不会让资源整数账本身超支。

## 定点水位与误差归因

令 `p=scale_power ∈ {0,1,2,3}`，`S=10^p`。非负数量统一量化为：

```text
Q(x) = saturate_int64(x * S)
```

整数输入乘 `S` 无截断；所有归一化除法使用非负数向下取整（等价向零截断）。中间值超过 int64 上限时饱和到 `2^63-1` 并标记 `saturated=true`，同时把合成误差界扩张为 int64 最大值，表示不再做细粒度区分。

候选分片的租约归一化延迟水位为：

```text
W_q = floor(D_ms * S * R / L)
```

其中 `R` 是资源单位需求，`L` 是节点租约上限。同口径阈值为 `floor(D_limit_ms * S * R / L)`。

误差段（定点 q 单位）：

```text
E_quant  = 1                         # 单次 floor 的保守上界，真实误差 < 1 q
E_jitter = ceil(J_ms * S * R / L)
E_lease  = ceil(K * 1ms * S * R / L) # K=租约授予/续约/释放/撤销/过期重排次数
E_total  = E_quant + E_jitter + E_lease
```

两个候选 `a,b` 的水位差满足 `|W_a-W_b| <= E_a+E_b` 时判定为不可区分。此时不允许水位口径重排，改按 `(topo_rank, shard_id)` 显式决胜，并写 `error_affected=true`。若差距超过合成界，则较低水位胜出。

主导段阈值显式规定为：某段占比至少 `0.60` 才标记为该段，否则标记 `mixed`。

全局预算误差上界同时给出两种口径：

- 先聚合再做一次预算比率除法：截断界 `<1 q`，保守报告 `1`；
- 对 `N` 个分片的独立截断项求和：截断界 `<N q`，保守报告 `N`；
- 安全上界为 `N + ΣE_jitter + ΣE_lease`。

样本 `config/spec.json` 刻意令 node-a 的 `D=8` 与 node-b 的 `D=9` 在 `R=1,L=4,p=0` 下都得到 `W=floor(D/4)=2`。前者精确口径可下发，后者精确口径已超过 `D_limit=8`，但定点水位差为 0 且落在误差界内，因此由稳定拓扑序放行并标注误差影响。

## 容量版本换代

容量版本变化时：

1. `LeaseManager.revoke_for_capacity()` 将该节点所有 `ACTIVE` 旧租约标记为 `REVOKED`，调用节点 `fence_capacity` 清空 Worker 租约状态；
2. SQLite `occupancy` 中对应活跃占用整体改为 `capacity_revoked`，活跃资源和租约数归零；
3. `ConclusionStore.mark_stale()` 将旧 summary/water/decision 从 `current_view()` 移除并保留为 stale 审计记录；
4. `shard_diag/invalidation.py` 推导节点引用分片及反向依赖闭包，失效集合按稳定拓扑序输出。

## 失效传播与全量重建

`plan_change()` 支持：

- `revision_chain`：全局或指定节点的水位/修订闸门失效，再传播下游；
- `capacity_version`：节点引用分片直接失效，摘要绑定的容量版本、租约、水位和结论均重算；
- `resource_units`：变更分片及其全部下游失效；
- `scale_factor`：所有水位、误差归因和裁决失效。

未在 `invalid_shards` 中的对象不会被写入；测试逐字节比较其规范 JSON。`assert_matches_full_rebuild()` 从空白声明重建确定性视图并逐字段比较，首个差异以 `E_DIVERGE` 报告对象和字段路径。

## SQLite 账本与强制终止恢复

`shard_diag/ledger.py` 使用标准库 `sqlite3`。数据库包含：

- `meta`：整数账本格式版本 `PRAGMA user_version=1`、声明摘要和 session id；
- `outcomes`：每个分片结局的规范 JSON、整数 `format_version`、声明摘要和行内容摘要；
- `occupancy`：租约/资源占用及 `active/reclaimed/revoked/completed_released` 状态；
- `progress_cursor`：与结局同一事务推进的拓扑游标。

`commit_outcome()` 在同一 SQLite 事务写入 outcome、内容摘要、占用终态和进度游标。重新打开时，如果格式版本或 `spec_digest` 与当前声明不一致，会删除旧 DB/WAL/SHM 并整体重建，不做增量合并。

节点子进程被 `Popen.kill()` 强杀后：

- 在途 RPC 失败，分片状态为 `UNKNOWN`，不写 outcome；
- 调度预留的资源/节点名额在 `finally` 中回收，账本占用标记 `reclaimed`；
- 重启节点后，只从已提交 outcome 恢复成功分片；
- 未提交分片进入重跑集合；
- `verify_ledger_rebuild()` 先校验行内容摘要，再逐字段比较重算摘要、资源、水位等，首个不一致以 `E_REPLAY` 报告。

强杀时无法获得可信退出报告，本版本也不伪造该报告；可确认的是进程已由 `kill/wait` 回收、未提交记录不存在且占用被回收。

## 主要模块

- `shard_diag/spec.py`：JSON 加载与 `E_SPEC` 强校验。
- `shard_diag/determinism.py`：NumPy 确定性样本、排序聚合和批次合并。
- `shard_diag/node_worker.py` / `node_protocol.py`：回环 JSON Lines 子进程。
- `shard_diag/leases.py`：真实时钟租约和乐观版本。
- `shard_diag/scheduler.py`：线程池、队列、等待计时和三重配额。
- `shard_diag/water.py`：定点量化、误差界和稳定比较。
- `shard_diag/invalidation.py`：影响传播、陈旧代际和 `E_DIVERGE`。
- `shard_diag/ledger.py` / `diagnostics.py`：SQLite 原子账本、快照替换和 `E_REPLAY`。
- `tests/`：27 个本地单元/集成测试，包含真实子进程、真实 sleep 和强杀恢复。
