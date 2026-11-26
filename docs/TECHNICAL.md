# 本地分片租约水位诊断工作台

本实现不使用 Docker 或外部服务。每个节点是由父进程通过 `subprocess.Popen` 拉起的真实 Python 子进程，在 `127.0.0.1` 上监听 TCP，使用换行分隔 JSON（JSON Lines）通信。

## 快速运行

```bash
python3 -m pip install -r requirements.txt
# Debian/Ubuntu/WSL 如缺少 Tkinter：
# sudo apt-get install -y python3-tk

python3 -m workbench init examples/sample-spec.json
python3 -m workbench validate examples/sample-spec.json
python3 -m workbench run examples/sample-spec.json --workdir artifacts/run
python3 -m workbench export-samples examples/sample-spec.json --out-dir artifacts/samples
python3 -m workbench acceptance --output artifacts/acceptance
python3 -m workbench gui
python3 -m pytest
```

一键验收报告写入 `artifacts/acceptance/acceptance-report.json`；pytest 实测输出保存在 `artifacts/pytest-validation.txt`。

## 模块地图

- `workbench/spec.py`：JSON 规范加载、强类型校验、首个 `E_SPEC` 字段与 JSON Pointer 位置。
- `workbench/graph.py`：稳定拓扑排序、后代传播。
- `workbench/samples.py`：`(节点 seed, shard_id)` 确定性样本、NumPy 分组聚合、批次合并与摘要。
- `workbench/node_process.py`：节点子进程、Threading TCP Server、聚合回执、节点侧租约乐观版本表。
- `workbench/protocol.py`：真实 TCP 连接、握手、`settimeout`、`E_TIMEOUT/E_FIELD` 映射。
- `workbench/orchestrator.py`：子进程拉起、端口等待、`SIGKILL`/`wait()` 回收和重启。
- `workbench/ledger.py`：SQLite schema、原子事务、租约状态机、资源占用、水位/误差/游标提交。
- `workbench/water.py`：定点水位、截断/饱和规则、逐分片/节点/全局误差界和误差带决胜。
- `workbench/invalidation.py`：容量版本、配额、节点引用和依赖图的失效传播。
- `workbench/recovery.py`：账本重放校验和两份结论视图的逐字段比较，差异抛 `E_REPLAY`。
- `workbench/snapshot.py`：会话快照版本/摘要识别和临时文件 `os.replace` 整体替换。
- `workbench/engine.py`：依赖就绪、三重容量约束、并行调度、租约签发/续约/释放/过期/换代回收。
- `workbench/gui.py`：Tkinter 诊断台。
- `tests/`：36 个本地测试，覆盖规范、协议、聚合、调度、租约、水位、失效、快照、SIGKILL 恢复和 GUI 构造。

## 规范校验

支持字段见 `examples/sample-spec.json`。关键校验包括：

- 节点、分片 ID 唯一；`host` 必须是字面量 `127.0.0.1`。
- 端口为 `1..65535` 整数且全局唯一；布尔值不被当作整数。
- `concurrency_quota`、`lease_limit` 为非负整数；`sample_rows`、`resource_units` 为正整数。
- 上游必须存在，Kahn 拓扑排序发现环即拒绝。
- `capacity_chain[].version` 严格递增，`parent` 必须引用更早存在的版本。
- `events[].seq` 严格递增、不重复。

语义错误结构为：

```json
{"code":"E_SPEC","field":"nodes.port","location":"$.nodes[1].port","message":"..."}
```

## 真实回环协议与进程终止

每条连接执行：

1. 客户端发送 `{"type":"hello","protocol_version":1,"node_id":"..."}`。
2. 节点回 `hello_ack`；握手套接字超时由真实 `socket.settimeout` 控制。
3. 父进程发送 `aggregate_batch` 或 `lease_sync/lease_renew/lease_release`。
4. 节点回一行 JSON；聚合回执必须包含 `request_id/node_id/shard_id/capacity_version/elapsed_ms_monotonic/summary`。

错误映射：

- 端口占用或子进程在握手前退出：`E_PORT`，详情含端口、returncode 和 stderr 尾部。
- 连接/握手/读取超时：`E_TIMEOUT`，带 `stage=connect|read`。
- 回执缺字段：`E_FIELD`，带缺失字段路径。
- 租约版本不符：节点直接回 `E_LEASE`，父进程不更新到期时间。

强制终止使用 `Popen.kill()`（Linux/WSL 上为 `SIGKILL`）并调用 `Popen.wait()` 回收；测试断言 returncode 为 `-9`。线程池或套接字超时只用于等待 I/O，不会被描述为已经终止子进程。

## 确定性样本与聚合

样本种子规则：

```text
entropy = SHA256(str(node_seed) + "\\x1f" + shard_id) 的 8 个大端 uint32
rng     = numpy.random.default_rng(numpy.random.SeedSequence(entropy))
group   = rng.integers(0, 8, size=sample_rows, dtype=int64)
value   = rng.integers(0, 1000, size=sample_rows, dtype=int64)
```

聚合使用 `np.bincount`、`np.add.at`、`np.minimum.at`、`np.maximum.at` 得到每组的行数、整数和、平方和、最小值、最大值和定点均值。批次摘要是可合并的分组交换量；最终摘要绑定：

- `node_id`
- `shard_id`
- `sample_rows`
- `participating_rows`
- `capacity_version`
- `sample_digest`
- 全部分组聚合值

因此样本生成顺序、TCP 下发顺序、批次数量和线程并发度不会改变最终 `summary_digest`。测试覆盖 1/3/7/13 等批次切分、乱序批次合并和 32 线程并发生成。

## 租约与配额状态

租约字段为 `lease_id/shard_id/node_id/granted_at_ms/expires_at_ms/renew_count/lease_version`，另存 `capacity_version` 和状态/回收原因。

- 签发：SQLite 先原子写入 `in_flight` 分片、active 租约和 active 资源占用，再把租约版本同步给节点。
- 续约：父进程携带期望 `lease_version` 请求节点；节点用锁保护的版本表做 CAS，成功后版本加一、续约次数加一、到期时间延长。
- 释放：本地条件更新 active -> released，并尽力通知节点删除租约。
- 过期：默认使用 `time.monotonic_ns()`；`expires_at_ms <= now` 时条件更新 active -> expired。
- 容量换代：同一租约行只能通过 `WHERE state='active'` 的条件更新变成 `superseded`。过期回收和换代回收谁先拿到该行谁生效，另一个调用影响行数为 0，因此同一配额槽不会双重回收或被两个分片同时持有。

## 调度约束

每轮调度先回收过期租约，再计算依赖就绪集合和水位裁决。准入在同一把进程锁和 SQLite 活跃占用查询后完成：

```text
active_resource_units + shard.resource_units <= global_budget_units
node_active_shards + 1               <= node.concurrency_quota
node_active_leases + 1               <= node.lease_limit
```

- 只有全部上游 `completed` 且有 `summary_digest` 时，分片才就绪。
- `frozen=true` 的分片不进入就绪集合、不创建租约、不写 active 占用。
- 每轮使用 `admitted_this_pass` 和 `in_flight` 集合保证同一分片最多下发一次。
- 因预算或节点容量不能准入时记录 `wait_started_ms/wait_ended_ms/wait_ms`。
- 并行执行使用 `ThreadPoolExecutor`；SQLite 提交集中在调度线程，避免跨线程事务竞争。

## 定点水位与误差

会话参数 `session.scale_exponent` 只允许 0、1、2、3，缩放因子 `F = 10^scale_exponent`。

```text
scaled(x)       = saturate_int64(x * F)
water_fp(x, c)  = saturate_int64(floor(x * F / c))   # c > 0
```

- 所有非负数除法向负无穷/零方向截断；记录 `truncation_direction=floor_nonnegative` 和 `division_remainder`。
- 超过 signed int64 边界时饱和到 `2^63-1` 或 `-2^63`，并置 `saturated=true`。
- 分母为 0 的容量按饱和水位处理。
- 分片记录给出 `quantization_truncation_error_fp=1` 和换算成全局资源单位的原始误差界。
- 节点水位取配额、租约、资源三个水位的最大值；两节点比较的保守误差界为 `2 + source_error_fp`。
- 全局聚合误差上界为各分片误差界之和：`shard_count + version_error + dependency_error + lease_error`。

三类来源在 `error_attribution` 中分别列出：

- `version_error_fp`：节点容量版本到链头的父边距离，每条边计 1 个定点单位。
- `dependency_error_fp`：裁决时不确定/失效上游数；就绪且有效时为 0，演进传播时可追溯。
- `lease_error_fp`：在途租约到期边界，每个裁决/在途分片保守计 1。

当预算缺口或两个候选水位差不超过合成误差界时，候选被标为 `error_affected=true`，按稳定 `(topo_index, shard_id)` 决定胜者。硬容量约束仍优先：精确整数账下不可放行的候选只会排队，不会因误差带超卖。

## 失效传播

`plan_evolution(old_spec, new_spec, old_attribution)` 比较：

- 节点 `capacity_version/seed/fixed_point_exponent`：摘要失效。
- 节点 `concurrency_quota/lease_limit`：水位/调度归属失效。
- 节点 `host/port` 或节点删除：放置/引用失效。
- 分片样本行数、资源、依赖、优先级、固定节点引用：对应结论失效。

直接失效集合再沿反向依赖图做闭包传播。输出按新规范的稳定拓扑序和 `shard_id` 排序，并为每个直接变更或传播分片给出 `propagation_steps[].basis/invalid_upstreams`。

重算时只把失效分片标记为 `stale`；未失效行、租约、分片级水位和误差记录不更新。测试对这些整行做 canonical JSON 字节比较，并把重算后的整库结论视图与新规范空白重建逐字段比较。

## SQLite、快照与恢复

SQLite 表包括 `shards/leases/resource_occupancy/water_records/error_attribution/outcome_commits/progress_cursor/invalidations/meta`。

每个分片结局在同一个事务中完成：

- 写 completed shard 和 `summary_json/summary_digest`；
- 把租约改为 released/completed；
- 把资源占用改为 released；
- 写分片/节点/全局水位；
- 写误差归因；
- 写 outcome commit；
- 更新单行 progress cursor。

所有格式版本为整数 `1`，记录携带 SHA-256 `content_digest`。`outcome_commits.content_digest` 绑定完整提交载荷；`shards.content_digest` 绑定可跨空白重建复现的业务结论包，避免把 incremental cursor 历史误当业务结论差异。

快照打开规则：

- `snapshot_format_version` 不符、`spec_digest` 与当前规范不符或快照摘要损坏：整体从当前 SQLite 视图重建。
- 写临时文件后 `fsync + os.replace`，不做字段级增量合并。

SIGKILL 恢复：

1. 父进程对子进程执行真实 `kill()` 并 `wait()`，测试确认 `-9`。
2. SQLite 中 `in_flight/dispatched/unknown` 分片被置为 `unknown/E_REPLAY_RERUN_REQUIRED`。
3. 对应 active 租约条件变为 `reclaimed/forced_termination_recovery`，active 占用变为 `reclaimed`，配额立即回收。
4. 重启节点后，可调用 `restore_active_leases_to_nodes()` 从已提交租约重建节点乐观版本表。
5. 未完成分片重跑并提交；`compare_conclusion_views()` 按拓扑序比较 shard、summary、lease、occupancy、error、water，首个差异抛出含 `shard_id/field/left/right` 的 `E_REPLAY`。

被 SIGKILL 的节点不可能给出退出报告；本实现不等待或承诺此类报告。
