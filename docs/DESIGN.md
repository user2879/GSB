# 设计与不变量

## 1. 分层

```text
JSON 声明/事件
    |
    v
spec.py         强类型加载、字段/行号错误、统计摘要
    |
    +-------------------------------+
    |                               |
    v                               v
engine.py / state.py             runtime.py
纯确定性仲裁内核               真实子进程 + TCP + 线程池
    |                               |
    |                               +--> transport.py / node_server.py
    |                               +--> scheduler.py
    |                               +--> ledger.py
    v
verifier.py                      recovery.py
独立双路径对拍                  强杀后完整提交前缀重放
```

`engine.py` 不创建套接字、不写 SQLite；两条仲裁路径只依赖原始声明和事件。`runtime.py` 将同一内核映射到真实回环子进程，并用节点回执复核中央标签。`verifier.py` 和 `recovery.py` 不读取主流程中间缓存。

## 2. 声明与事件校验

- 节点标识、租约标识唯一；节点地址必须是字面量 `127.0.0.1`；全声明端口唯一。
- `global_concurrency`、`max_leases`、TTL、续约间隔、配额版本均为正整数。
- 每个租约的 `renew_interval_ms < ttl_ms`，且必须引用已声明节点。
- 事件只允许 `grant`、`renew`、`release`、`quota_invalidate`，字段集合按操作强校验。
- `seq` 严格递增且唯一；`at_ms` 非负且单调不减；标签非负。
- 首个非法字段以固定校验顺序报告为 `E_SPEC`，事件错误带其物理行号；声明错误使用字段键定位行。
- JSON 重复键、非有限数值、未知字段、空行均拒绝。

## 3. 节点 JSON 行协议

每个节点是一个独立 Python 子进程，在声明的 `127.0.0.1:port` 上监听。中心每次请求建立真实 TCP 连接，先发送：

```json
{"op":"hello","protocol":1,"request_id":"..."}
```

节点回执握手后，中心发送一个操作帧，节点返回一行规范 JSON。连接、握手和读取均设置真实套接字超时。监听绑定失败为 `E_PORT`，套接字超时为 `E_TIMEOUT`。

节点为每份租约维护 `label` 与状态。`grant`、`renew`、`release` 成功后标签严格加一；`renew`/`release` 的 `expected_label` 不匹配时返回 `E_STALE`，包括 `current_label` 和 `state`，且不修改节点代数或状态。节点状态受锁保护，若干连接可并行处理。

## 4. 状态机与配额槽位

`state.py` 中的 `ALLOWED_TRANSITIONS` 是完整许可表。每次迁移记录：

```text
event_seq, lease_id, at_ms, reason,
from_state, to_state, label_before, label_after
```

标签不能通过外部参数跳跃；`transition(..., new_label=current+2)` 测试路径也返回 `E_LEASE`，且异常发生在写入前。拒绝路径保留原状态、标签、槽位和到期时间。

容量使用两个独立维度：

- 全局槽：`1..global_concurrency`；
- 节点槽：每个节点的 `1..max_leases`。

槽位取最小空闲值，释放后允许后续租约复用，但任一时刻同一槽位只有唯一租约所有者。`SlotPool.assert_consistent()` 在每次仲裁后检查槽位所有权、活跃租约与容量上限。

## 5. 到期、配额换代与唯一归属

事件中的 `at_ms` 是从会话单调时钟起点计量的逻辑时刻。真实运行时，工作线程用 `time.monotonic_ns()` 等待到该时刻后才进行仲裁和节点调用；规范化到期时刻是 `at_ms + ttl_ms`，因此账本可在另一进程中按原始事件流重算。

每次事件前，按 `(expires_at_ms, lease_id)` 排序回收所有已到期活跃租约。`quota_invalidate` 先更新该 `(node_id, lease_type)` 的配额纪元，再按 `(quota_version, lease_id)` 排序回收所有更老代数的活跃租约。每条回收记录包含：

```text
触发 event_seq, lease_id, reason, label_before/after,
owner_before, global_slot, node_slot, expires_at_ms
```

回收后状态置为 `expired`、标签加一、槽位置空并写入归属变化。已释放或已到期对象不会被第二次回收，因此同一容量权益不会同时计入到期回收和配额作废回收。

## 6. 两条仲裁路径

逐事件路径按 `seq` 推进：到期扫描 -> 应用事件。

批量路径按 `at_ms` 划分帧，先按 `node_id + lease_type` 聚合，帧边界只做一次到期扫描，然后按事件 `seq` 统一裁决并回填。共享全局/节点槽的授予顺序仍由事件序号唯一确定，避免依赖字典或哈希遍历。批量统计输出帧数、分组数和回填顺序。

默认路径选择规则位于 `engine.BATCH_RULE`：

```text
batch iff event_count >= 10 AND node_count >= 2 AND remaining_capacity >= 2
```

输出包含规则、实际参数、每条件是否满足、余量、阈值边界和未触发时所需增量。

## 7. 逐字节投影

对拍只比较决定业务结论的规范投影：

- 每事件：`seq/op/lease_id/node_id/lease_type/status/label/state/global_slot/node_slot/expires_at_ms`；
- 所有回收：触发事件、原因、标签、槽位、原所有者和到期时刻；
- 配额纪元；
- 最终租约状态。

投影使用排序键、紧凑分隔符和 UTF-8 规范 JSON，再比较完整字节与 SHA-256。差异时先定位事件序号，再按固定字段顺序报告首个不同字段，错误码为 `E_DIVERGE`。

## 8. 并行调度

`QuotaScheduler` 在把任务提交给 `ThreadPoolExecutor` 之前，同时获取：

- 一个全局在途槽；
- 一个节点在途槽。

任一配额不足时在条件变量上排队，不创建执行任务、不发网络请求。获得槽位时用 `time.monotonic_ns()/1_000_000` 记录等待开始与结束毫秒。`snapshot()` 输出上限、实际峰值、每请求槽位和等待区间。

运行时按租约链串行化同一租约的动作，不同租约的动作可在同一线程池轮次并行下发。配额耗尽的授予在中央仲裁阶段即返回 `E_QUOTA`，节点不会收到该请求。

## 9. SQLite 账本与事务

账本使用 WAL 与 `synchronous=FULL`。核心表：

```text
session              单行会话与当前完整快照
session_replacements 整体替换审计
commits              每事件提交、格式版本、声明版本、内容摘要
transitions          每次状态迁移审计
ownership            每次归属变化审计
pending              已预留但尚未提交的请求
recoveries           强制终止后的未知请求与释放槽位
```

每个事件在一次 `BEGIN IMMEDIATE` 事务中写入 commit、transition、ownership，并更新完整快照；成功后删除 pending。每行都带整数格式版本与内容摘要。

打开账本时比较：

```text
format_version == 1
declaration_version == 当前组合声明版本
spec_digest == 当前声明摘要
sha256(payload_json) == snapshot.content_digest
```

任一不一致即删除当前活动 commits/transitions/ownership/pending/recoveries，写入全新初始快照并记录 `session_replacements`，不做增量合并。`--fresh` 也会执行同一个整体替换路径。

## 10. 强杀恢复与重放

运行事件时，若存在节点动作，先在独立事务中写入 pending，再发请求；节点回执与中央代数吻合后，才提交事件事务。强制终止可能留下 pending，但不会留下半提交事件。

恢复流程：

1. 读取当前声明并验证快照版本/摘要；
2. 将所有 pending 标为 `unknown`，记录预留的全局/节点槽位为已释放；
3. 仅从当前完整提交快照和 commits 重建；
4. 按原始事件流从头重算到最后一个已提交事件；
5. 逐事件比较结局、推进标签、配额槽、到期时刻、回收、迁移和归属；
6. 比较重算快照与账本快照；
7. 首个差异抛出 `E_REPLAY`，携带 `event_seq` 与 `field`。

重放不声称恢复未提交动作的节点状态，也不承诺强杀进程生成退出报告。真实重启时应启动新节点进程；账本恢复只信任完整提交记录。

## 11. 确定性边界

- 事件、租约、节点和迁移枚举全部显式排序或在固定列表中处理。
- TTL/到期值使用事件逻辑单调时刻，实际等待由 `time.monotonic_ns()` 驱动。
- 运行时等待、线程槽位与进程 PID 不进入业务投影，因此不会污染双路径字节比较。
- 容量槽允许释放后复用；唯一性要求是同一时刻唯一，而不是永久不复用。
