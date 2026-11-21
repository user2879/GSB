# 技术设计

## 1. 组件边界

`cluster_lease` 包按职责拆分：

| 模块 | 职责 |
| --- | --- |
| `loader.py` | JSON/NDJSON 加载、首错定位和受控注入 |
| `state_machine.py` | 无 IO 的确定性租约状态机 |
| `service.py` | 单集群 TCP 仲裁服务子进程 |
| `orchestrator.py` | `python -m cluster_lease.service` 子进程编排和协议客户端 |
| `ledger.py` | SQLite schema、当前视图、批事务、摘要和快照恢复 |
| `runtime.py` | run/recover/evolve 会话编排 |
| `snapshots.py` | 快照构造、完整性校验、整体恢复和重放比较 |
| `gui.py` | 只读 Tkinter 诊断界面 |
| `cli.py` | 命令行、业务码、结构化日志和报告落盘 |

模块之间不共享可变全局状态。每个集群由独立子进程持有内存状态，主席账本以 SQLite 为唯一持久化真值。

## 2. 输入校验

校验顺序固定为集群声明、持有者声明、事件流；同一层按文件顺序检查。因此非法输入总能返回第一个违规字段。

集群字段：

- `cluster_id`: 非空字符串，全局唯一。
- `host`: 必须为字面量 `127.0.0.1`，不接受 `localhost`。
- `port`: `1..65535` 整数，全局不重复。
- `lease_duration_ms`: 正整数。
- `renewal_interval_ms`: 正整数，且严格小于租约时长。
- `capacity`: 正整数。

持有者字段：

- `holder_id`: 非空字符串，全局唯一。
- `priority`: 整数，不允许布尔值伪装整数。
- `targets`: 非空、无重复、全部存在的集群标识列表。

事件字段：

- `event_seq`: 从 1 开始连续递增且唯一。
- `timestamp_ms`: 非负整数，单调不降。
- `cluster_id`、`holder_id`: 必须存在，且事件持有者必须声明目标集群。
- `event_type`: `apply/renew/release` 或对应中文标签。

任何正常校验失败返回 `E_SPEC`。命令行注入在解析前修改内存中的原始对象：缺失引用映射 `E_REF`，时长关系映射 `E_TIMING`。

## 3. TCP JSON Lines 协议

子进程启动命令固定为：

```text
<当前 Python> -m cluster_lease.service --config <runtime>/service-<cluster>.json
```

服务先对 `127.0.0.1` 声明端口执行 `bind/listen`。端口被占用时向 stderr 输出 `E_PORT` 并退出；编排器在启动前也会做一次真实回环端口探测，避免把外部监听器误判成仲裁服务。

握手：

```json
{"type":"hello","protocol":1,"cluster_id":"alpha"}
{"type":"hello","ok":true,"protocol":1,"cluster_id":"alpha","parameter_version":1}
```

握手连接设置真实 `socket.settimeout()`。超时返回 `E_HANDSHAKE`；任一必需回执字段缺失返回 `E_REPLY`。

租约请求：

```json
{"type":"lease","event_seq":1,"timestamp_ms":0,"cluster_id":"alpha","holder_id":"h-mid","event_type":"apply"}
```

裁决回执固定包含：

- `ok`, `type`, `event_seq`, `cluster_id`, `holder_id`, `event_type`
- `decision`, `reason`, `transitions`
- `active_holders`, `used_capacity`, `capacity`
- `watermark_fixed`, `watermark_scale`, `watermark_truncation`
- `parameter_version`

恢复请求 `type=restore`；只读真实时钟探测 `type=status`；优雅退出 `type=shutdown`。所有回执照规范 JSON 序列化。

## 4. 状态机

每个集群维护 `holder_id -> ActiveLease` 映射：

```text
ActiveLease =
  state, priority, started_ms, last_renewed_ms, expires_ms,
  renewal_count, last_event_seq, provenance, real_deadline_ns
```

处理顺序：

1. 在请求时间戳（逻辑时钟）和服务端 `time.monotonic_ns()`（真实时钟）下回收到期租约。
2. `apply`: 若持有者已持有则拒绝；容量可用则授予；容量满时比较最低优先级活跃租约，只有严格更高优先级可以接管。
3. `renew`: 只有仍在有效期内的当前持有者可以续约，并刷新到期时刻。
4. `release`: 当前持有者立即释放并归还容量；非持有者释放是幂等无迁移。

抢占时优先选择 `(priority, holder_id)` 字典序最小的活跃租约，确保相同输入产生相同迁移序列。所有活跃集合按 `holder_id` 排序输出。

每次迁移记录：

```json
{
  "cluster_id":"alpha",
  "event_seq":6,
  "holder_id":"h-mid",
  "before_state":"renewed",
  "after_state":"expired",
  "reason":"higher_priority_takeover",
  "active_count":1,
  "timestamp_ms":3500
}
```

## 5. 时钟语义

- `logical`: 仅以事件 `timestamp_ms` 和 `expires_ms` 判断到期。重放和演进强制使用该模式。
- `real`: 仅以服务端单调时钟判断到期。
- `hybrid`: 两种条件满足任一即到期，是普通 run 默认模式。

恢复时使用 `last_renewed_ms + ledger.lease_duration_ms` 重算逻辑到期，并以最后提交事件时间计算逻辑剩余量，再映射到服务端单调截止点。报告字段：

- `true_renewal_continuation`: 账本中已有成功续约记录。
- `recovery_recomputed_continuation`: 从授予时刻和账本租约时长重算延续。

## 6. SQLite 与唯一裁决

主要表：

| 表 | 用途 |
| --- | --- |
| `meta` | 账本格式版本、流游标、整体内容摘要 |
| `cluster_versions` | 每集群当前参数版本和优先级摘要 |
| `decisions` | 每个参数版本、每事件裁决 |
| `active_leases` | 当前有效租约；历史陈旧行保留只读 |
| `ownership_intervals` | 每个持有者每次持有权的开始/结束事件与时刻 |
| `watermarks` | 每个事件后的已用容量与定点水位 |
| `progress` | 每集群最后事件序号和时间戳 |
| `batch_runs` | `inflight/committed/unknown` 批状态 |
| `snapshots` | 快照目录 |

唯一性：

- `decisions` 主键 `(cluster_id,event_seq,parameter_version)`。
- 部分唯一索引 `(cluster_id,event_seq) WHERE is_stale=0`，保证当前视图一事件一裁决。
- 当前有效租约同样有部分唯一索引；参数演进后旧行被标记陈旧，不参与当前显示。

批事务流程：

1. 独立事务插入 `inflight` 批标记。
2. 服务端逐事件裁决。
3. 同一事务插入裁决、水位、最终有效租约，更新进度与批状态为 `committed`，重算整体摘要。

若进程在步骤 1 和 3 之间被强制终止，SQLite 中只留下 `inflight`。下一次恢复将其改为 `unknown`，报告要求重跑对应序号区间；进度游标不会越过未提交事件。

## 7. 摘要规则

规范 JSON 使用 UTF-8、键排序、紧凑分隔符，并禁止 NaN/Infinity。单事件裁决摘要输入为：

```text
cluster_id
event_seq
conclusion (= decision)
reason
used_capacity / capacity
watermark_fixed / watermark_scale
active_leases sorted by holder_id:
  holder_id, state, started_ms, expires_ms
```

整体账本摘要由以下内容按集群标识和事件序号稳定排序后计算：

- 整数格式版本
- 每集群参数版本、参数摘要、进度
- 当前裁决的事件序号、结论、原因与单事件摘要
- 当前有效租约集合
- 当前持有权起止区间
- 当前水位事件与水位摘要

快照内容摘要覆盖：快照整数格式版本、事件流摘要、参数版本、流游标、各集群参数、优先级、裁决、有效租约、水位和账本摘要。`created_at_utc` 与 `snapshot_id` 不参与内容摘要。

## 8. 参数演进

`evolve` 比较新旧集群声明和优先级映射。受影响条件包括容量、租约时长、续约间隔、端口/主机或优先级摘要变化。对每个受影响集群：

1. 参数版本加一。
2. 按新参数启动新服务进程。
3. 重放该集群截至已提交进度游标的事件。
4. 在一个 SQLite 事务中把旧裁决、水位、有效租约标记陈旧，写入新裁决/水位/有效租约和进度。
5. 输出被撤销有效租约、保留的只读旧记录和被当前视图丢弃的陈旧引用。

整个过程是整体替换，不把旧有效租约增量合并到新版本。

## 9. 快照恢复与重放

打开快照顺序：

1. 校验整数格式版本和快照内容摘要。摘要不符立即 `E_SNAPSHOT`，不恢复。
2. 比较当前参数版本、事件流摘要和账本摘要。若不一致，整体替换 SQLite 当前视图。
3. 再次确认恢复后的账本摘要等于快照账本摘要。
4. 用快照参数版本、优先级和端口启动临时服务进程。
5. 按快照游标重放事件，逐事件比较结论、原因、容量、水位和有效租约。
6. 首个差异返回 `E_REPLAY`，包含 `cluster_id`、`event_seq`、`field`、`expected`、`actual`。

非法快照摘要不会被信任；摘要合法但与当前会话不同时，恢复成功仍返回 `E_SNAPSHOT` 以明确报告差异依据。

## 10. GUI

`gui` 子命令使用 SQLite `mode=ro` URI 连接。界面稳定按集群标识排序，展示：

- 当前有效租约、持有者、优先级、起始/续约/到期时刻和逻辑剩余时长
- 已用容量与定点水位
- 水位曲线（NumPy 可用时使用，否则标准库回退）
- 状态迁移轨迹
- 参数版本和快照目录

刷新只执行查询。tkinter 缺失或无显示时，CLI 返回 `E_ENV`，不影响核心命令。
