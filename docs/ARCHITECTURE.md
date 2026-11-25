# 架构与不变量

## 数据流

```text
samples/*.json
    │  spec.py 强校验（E_SPEC + 首个字段位置）
    ▼
SpecBundle ───────────────► engine.ReplayEngine（无 IO 的权威重放）
    │                              ▲
    │ service.LiveScenarioRunner   │ verify.py 独立重放
    ▼                              │
scheduler.QuotaScheduler           │
    │  全局在途配额 + 节点在途配额   │
    ▼                              │
ThreadPoolExecutor → protocol.LeaseNodeClient
    │ 真实 TCP JSON-line            │
    ▼                              │
node_server.NodeTagState 子进程     │
    │ 当前 lease_id -> tag          │
    ▼                              │
SQLite entries/intents/header ─────┘
```

`ReplayEngine` 不读取时钟、线程标识、端口或启动顺序。所有到期时间来自事件中的 `at_ms` 和申请中的 `ttl_ms`，因此同一事件流在任意节点启动顺序下生成相同字节结论。

## 代数链

初始授予为 `generation_before = null`、`generation_after = 1`。节点当前标签从 1 开始；每次匹配标签的续约、释放或回收都令标签加一。

显式许可表位于 `lease_middleware/statemachine.py`：

- `null -> granted` by `grant`
- `granted -> renewed` by `renew`
- `renewed -> renewed` by `renew`
- `granted|renewed -> released` by `release`
- `granted|renewed -> expired` by `expire`

请求标签小于当前标签时为过期/回退，大于当前标签时为跨代跳号；两者均返回 `E_LEASE` 和节点当前标签，且不得修改状态、代数或 `expires_at_ms`。

## 唯一归属裁决

每个容量权益由 `equity_id` 标识。中心维护 `owners[equity_id]`，授予时若已有有效持有者则拒绝。

有效回收以 `(equity_id, generation_before)` 去重：

- TTL 扫描先命中：记录 `reason=ttl_expired`、`counted_reclamation=true`；之后同代的节点换代作废只写 `change=suppressed, effective=false, counted_reclamation=false`。
- 节点换代先命中：记录 `reason=generation_superseded`；之后同代 TTL 扫描只写抑制记录。

因此同一份容量权益的同一代不会被两种原因各计数一次，也不会同时归属两个节点。

## 配额调度

`QuotaScheduler` 在条件变量后维护 FIFO 意图队列，可跳过因节点配额暂不可准入的任务，选择队列中第一个同时满足：

```text
global_active < global_in_flight
node_active[node_id] < node_in_flight[node_id]
```

的请求。准入后才向真实 `ThreadPoolExecutor` 投射。每条等待记录包含单调时钟毫秒的开始、结束和 `wait_ms`。工作线程结束后在临界区释放配额并唤醒调度线程。

节点 `in_flight_limit <= lease_limit`；引擎还会校验有效租约数不超过 `lease_limit`。

## 版本传播

`version_evolved` 事件递增 `topology_version` 和/或 `quota_version`，并给出 `changed_node_ids`、`removed_node_ids` 或 `scope=global`。`propagation.py` 按稳定顺序输出：

1. 变更节点；
2. 节点上的有效租约当前代数；
3. 代数链和引用过这些代数的事件序号；
4. 容量权益关联租约；
5. 实际失效集合、抑制集合和未变租约集合。

只有实际失效子集进入回收状态迁移；其他租约的代数、归属和容量切片可逐字节保持不变。

## 账本与恢复

SQLite 表：

- `ledger_header(id=1, format_version INTEGER, content_digest TEXT)`
- `ledger_meta(key, value)`
- `intents(intent_id, event_seq, lease_id, op, payload_json, status, created_monotonic_ms)`
- `entries(id AUTOINCREMENT, event_seq, record_kind, lease_id, generation, payload_json, payload_digest)`

每条业务记录有规范化 JSON 和 SHA-256 摘要；全局摘要按 entry id 形成哈希链。授予、续约、释放、到期回收、旧代数作废及其归属记录在同一个 SQLite 事务中写入，并在同一事务更新摘要和意图状态。

强杀后：

- 已提交事务存在；
- 未提交事务由 SQLite 回滚；
- 已单独提交但未完成的 `intents` 被列为 `unknown_leases`；
- 恢复不猜测网络结果，只承认完整提交的 `entries`。

会话快照是单个 JSON envelope：整数 `format_version`、`content_digest` 和 `state`。打开时若版本或摘要不匹配，`recover_snapshot()` 从 SQLite 读取全部已提交记录并原子 `os.replace` 整体替换快照，不做字段级合并。

## 独立校验

`lease_middleware verify` 不读取运行时对象或快照，只读取：

- `samples/nodes.json`
- `samples/lease_requests.json`
- `samples/events.json`
- 指定 SQLite 账本

它重新构造全部状态、记录和归属，先验证账本哈希链，再按 `entries[0]`、`entries[1]` 的顺序比较；第一个差异以 `E_VERIFY` 和 JSON 字段路径报告。
