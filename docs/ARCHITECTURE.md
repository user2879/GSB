# 架构与需求映射

## 模块

- `src/cloudshard/spec.py`：JSON schema、整数类型、ID/端口唯一、引用、依赖 DAG、修订父链和事件序列强校验。
- `src/cloudshard/samples.py`：由 `sample_seed`、节点整数种子和稳定分片序号派生 uint32 种子，使用 NumPy 随机生成确定性整数样本。
- `src/cloudshard/aggregate.py`：NumPy 分组聚合、SHA-256 摘要和回执字段校验。
- `src/cloudshard/node_server.py`：节点子进程入口；只绑定 `127.0.0.1`，以 JSON Lines 处理握手、计算和租约乐观校验。
- `src/cloudshard/cluster.py`：用 `sys.executable -m cloudshard.node_server` 启动/停止真实子进程，读取 ready/startup_error。
- `src/cloudshard/client.py`：真实 TCP 套接字、连接/读写超时、握手、计算下发、租约 RPC 和错误码映射。
- `src/cloudshard/adjudicate.py`：确定性容量权益状态机；分配全局单位、节点租约槽和并发槽，裁决到期/释放/取消/配额回收。
- `src/cloudshard/scheduler.py`：`ThreadPoolExecutor` 调度、依赖就绪、稳定优先级、三重容量检查、等待计时、TCP 投射和结局提交。
- `src/cloudshard/invalidate.py`：显式分片变更、节点引用和依赖图反向传播；输出稳定拓扑序和每步依据。
- `src/cloudshard/ledger.py`：SQLite schema、整数 `format_version=1`、内容摘要、原子分片结局事务和崩溃窗口。
- `src/cloudshard/recovery.py`：从已提交分派记录识别未知在途分片并先回收。
- `src/cloudshard/snapshot.py`：临时文件、`fsync`、`os.replace` 原子快照和格式版本识别。
- `src/cloudshard/verifier.py`：只读账本 + 原始声明的独立重放校验。
- `src/cloudshard/acceptance.py`：本地一键验收和真实 `SIGKILL` 演练。
- `tests/`：核心不变量、配额阻塞测试和真实流水线 E2E。

## 一致性设计

### 唯一归属

每个有效租约对应一个主归属键 `lease:<lease_id>`，并分配具体权益：

- `global:unit:NNNN`
- `node:<node>:lease-slot:NNN`
- `node:<node>:concurrency-slot:NNN`

回收采用 compare-and-set：只有 `status=active` 的租约能进入归属序列一次。到期事件先赢时，后续配额回收或显式撤销只写 `duplicate_reclamation_skipped` 决策，不产生第二次归属变化。事件按 `seq` 顺序归约；任意分批大小只影响调用边界，不改变最终 canonical JSON bytes。

### 调度

调度器在主线程的临界状态中检查：

```text
used_global_units + shard.resource_units <= global_budget_units
active_leases(node) < node.lease_limit
active_shards(node) < node.concurrency_quota
```

预留成功后才提交给线程池。线程池最大线程数为各节点并发配额之和，因此真实工作线程不会超过总并发口径。容量不足的就绪分片记录 `wait_started_ms`，成功预留时补 `wait_ended_ms` 和持续时间。

### 失效与字节保持

修订版 2 的样例显式改变 `s2`，并改变被 `s5.node_refs` 引用的 `n3`。反向依赖传播得到稳定集合：

```text
s5, s2, s4, s6, s7
```

`s0/s1/s3` 不重新下发。`outcomes` 按 `(shard, revision, attempt)` 追加，`current_view.outcome_canonical` 保存规范字节；校验器逐字节比较未受影响分片的首版结论仍为当前结论。

### SQLite 原子边界

每个分片完成时，单个 `BEGIN IMMEDIATE ... COMMIT` 写入：

- `event_log`
- `outcomes`
- `current_view`
- `occupancy` 释放状态
- `attempts` 完成状态
- `leases` 释放状态
- `ownership`
- 全局唯一 `progress` 游标

所有业务行带 `format_version=1` 和 `content_digest`。快照文件通过同目录临时文件 + `fsync` + `os.replace` 原子替换，并带 `snapshot_format_version`、`ledger_format_version` 与 `snapshot_digest`。
