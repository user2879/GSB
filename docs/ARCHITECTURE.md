# 配置中心租约分发与变更预算裁决系统 — 技术说明

## 1. 组件与进程模型

- 平台 `configlease/platform.py`：编排器与裁决者，单进程，持有 SQLite 与结构化日志。
- 节点 `configlease/node.py`：每个注册节点是一个**独立 OS 子进程**
  （`asyncio.create_subprocess_exec` 启动 `python -m configlease.node`），
  绑定唯一 `127.0.0.1` 高位端口，通过 JSON 行（每行一个 JSON、`\n` 分隔）通信。
- 平台与节点之间：握手长连接用于存活探测；并行裁决使用**每请求临时连接**，
  节点进程内所有连接共享同一份 `NodeState`，从而在保持状态一致的同时避免
  单连接串行点与同键串行化锁形成嵌套死锁。
- 纯规则层 `configlease/engine.py`：不含任何 socket/子进程，平台在线裁决与
  离线重放共用同一套函数，保证重放可复现。

## 2. 数据与确定性样本

- 修订清单 `manifest.json`：`rev_seq` 单调、`parent` 指向上一修订、`changes`
  为整数键到整数值的映射；解析期做连续性、重复、整数类型校验，损坏即 `E_MANIFEST`。
- 确定性样本 `sample.py`：以 `rev_seq` 推导固定 NumPy `default_rng` 种子，
  生成 64 个浮点样本，聚合为 `count/sum/mean/min/max/std`（固定精度、JSON 友好）。
  同一修订在任意节点、任意应用交错下摘要逐字节一致。

## 3. 租约与裁决

租约字段：`lease_id, request_id, node_id, rev_seq, low_seq, high_seq, keys,
issued_ms, deadline_ms, ttl_ms`。过期判定使用 `time.monotonic_ns()` 的单调毫秒时钟。

裁决顺序（`_arbitrate_once`）：

1. `validate_request`：越界修订 `E_REV`、未覆盖键 `E_KEY`、已回滚修订
   `E_ROLLBACK`、租约过期 `E_EXPIRED`，全部在消耗预算之前判定，拒绝不推进游标。
2. 双重预算：先全局下发令牌，再每节点修订令牌；FIFO `Condition` 排队，
   耗尽即真实排队、不可绕过。
3. 同键串行化 `KeyConflictSerializer`：原子“全部键可用且为最小
   `(node_id, rev_seq)` 才进入”，杜绝多键部分持有的锁顺序死锁；单次裁决由
   `asyncio.wait_for` 施加真实超时，超时重试保留原 `request_id` 与 `rev_seq`。

预算语义：

- 全局下发预算 = 在途裁决闸门，租约到达终态（应用/拒绝）即释放令牌。
- 每节点修订预算 = 该节点**累计应用修订数**上限，成功应用永久占用，
  失败/规则拒绝退还。

## 4. 回滚

`rollback(target_seq)` 生成一个**全新单调递增**修订号；严格晚于 target 的旧修订
标记 `superseded`；窗口与被回滚区间相交且未过期的 active 租约整体撤销并标
`E_ROLLBACK`；每个节点通过 `replace_view` 用历史修订折叠出的完整视图**整体替换**
（非增量合并），并返回受影响租约集合与重建所依据的历史修订序列。

## 5. 持久化与强制终止恢复

SQLite（WAL、`synchronous=FULL`）六张表：`revisions, leases, node_cursors,
budgets, events, snapshots`。所有状态迁移在 `BEGIN IMMEDIATE` 事务内提交，
kill 时只会留下完整的旧记录或完整的新记录。

`recover()`：

- active 且未落定的租约改写为 `unknown`（`E_RECOVER`），要求重新申请；
  存在 unknown 租约时 `build_exit_report()` 直接拒绝，绝不编造退出报告。
- 依据 applied 租约重算每节点游标/视图，依据终态租约重算全局与节点预算消耗，
  与落库行不符即记录 `E_RECOVER` 并校正。
- 重启节点后用 `replace_view` 整体重建节点内存视图。

## 6. 快照与重放

快照保存修订版本、租约集合、节点游标、视图与预算，并附 SHA-256 内容摘要与
版本号。重开时 `assert_snapshot_compatible` 比较版本与摘要，`E_VERSION` 触发
租约/游标状态的**整体替换**而非增量合并；随后 `replay_lease_records` 按记录的
租约序列重跑同一套校验与视图折叠，逐字段比对，任何不可复现以 `E_REPLAY`
报告具体修订与字段。

## 7. 异常注入与日志

仅通过输入文件/参数注入，不改源码：

- `--inject-manifest`：把清单改写为截断 JSON → `E_MANIFEST`。
- `--inject-port`：预先 `bind+listen` 占用节点端口 → `E_PORT`。
- `--inject-version`：改写快照版本号/摘要 → `E_VERSION`。

所有事件以 JSON 行写入 `logs/<session>.jsonl`，并镜像到 `events` 表，
可用 `configlease logs --node/--rev/--code` 按节点、修订、错误码检索。

## 8. 错误码对照

| 码 | 触发条件 |
| --- | --- |
| E_MANIFEST | 清单文件缺失/JSON 损坏/结构非法 |
| E_PORT | 节点端口已被占用或 bind 失败 |
| E_START | 子进程无法拉起或启动行非法 |
| E_HANDSHAKE | 真实套接字连接/读 ready 帧超时 |
| E_KEY | 变更键超出租约覆盖键集合 |
| E_REV | 修订号超出租约 `[low, high]` 区间 |
| E_ROLLBACK | 对已回滚修订下发，或租约因回滚被撤销 |
| E_APPLY | 节点回传游标/视图/摘要与平台期望不符 |
| E_EXPIRED | 单调时钟判定租约已过期 |
| E_RECOVER | 崩溃恢复中的 unknown 租约/游标或预算差异 |
| E_VERSION | 快照版本或内容摘要不一致 |
| E_REPLAY | 重放结果与快照逐字段比对不可复现 |
| E_TIMEOUT | 单次裁决真实超时 |
