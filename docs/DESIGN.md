# LeaseCache 技术设计

## 1. 目标与边界

LeaseCache 把“制品缓存命中”建模为可被流水线运行临时持有的租约资源。一次租约授予同时表达四件事：

1. 某个制品键由哪个缓存节点提供服务；
2. 服务节点为此占用多少容量槽位；
3. 哪些节点拥有与该制品相同摘要/引用签名的可复用副本；
4. 账本中当前唯一的容量所有者是谁。

组件只使用本机 `127.0.0.1`、Python 子进程、真实 TCP、线程池和 SQLite；不使用容器、远程服务或人为超时。

## 2. 模块与数据流

```text
UTF-8 JSON spec
  │
  ├─ spec.py 强校验 ──> PipelineSpec / NodeSpec / ArtifactSpec / StageSpec
  │
  ├─ digest.py ──> node_cache/<node>/<key-hash>.artifact + manifest.json
  │
  ├─ node_manager.py ──spawn──> node_server.py (127.0.0.1:port)
  │        │                         │
  │        └──── real TCP JSON line ──┘
  │
  ├─ planner.py / arbiter.py ──> reuse closure + one service node + vetoes
  │
  ├─ scheduler.py ── ThreadPoolExecutor + global/stage/node semaphores
  │        │
  │        └─ begin PENDING → TCP lease → commit ACTIVE in one SQLite txn
  │
  ├─ expiry.py ── time.monotonic_ns deadline → LEASE_EXPIRED txn
  │
  ├─ ledger.py ── events + nodes + leases + claims + summaries + sources + progress
  │
  ├─ verifier.py ── independent digest/replay/arbitration comparison
  │
  └─ engine.py ── report + canonical business snapshot
```

## 3. 规格与首个违规定位

`spec.py` 按根、节点数组、流水线、阶段数组、制品数组的顺序校验。遇错立即抛出 `E_SPEC`，其中：

- `field`：如 `nodes[1].port`、`pipeline.stages[0].artifacts[0].references`；
- `expected`：首个违规字段的约束；
- `actual`：实际值。

校验包含节点与制品键唯一、阶段标识唯一、`host == "127.0.0.1"`、端口在 `[1024,65535]` 且全局唯一、容量/租约上限/阶段和全局并发/租期为正整数、种子为 `uint32`、引用存在且引用图无环。循环检测用颜色 DFS，并定位到形成回边的制品字段。

## 4. 确定性副本摘要

显式规则：

```text
key_word = first 8 bytes of SHA256(UTF-8(key)), big endian
sequence = NumPy SeedSequence([node.random_seed, key_word])
vector   = PCG64(sequence).integers(0, 2^32, size=16, dtype=uint32)
payload  = canonical-json({
             "format": "leasecache-artifact/v1",
             "key": key,
             "seed": random_seed,
             "vector": [16 uint32 values]
           })
digest   = SHA256(payload)
```

`canonical-json` 固定使用 `sort_keys=True`、紧凑分隔符和 UTF-8；不依赖 Python 字典遍历顺序。节点每次启动都从上述规则重写文件并重新读取校验，因此篡改的本地文件不会延续到重启后的节点。

## 5. 复用与唯一服务节点

规范摘要取节点标识序最小节点的当前摘要。某节点对制品可复用，当且仅当：

- 账本 `sources.valid=1`；
- 该节点当前摘要等于规范摘要；
- 引用列表等于规范引用列表；
- 拓扑闭包内所有引用制品均可复用。

`source_nodes` 是所有满足条件的节点，按标识排序；这使输出不依赖遍历顺序。

服务节点仲裁排序键固定为：

```text
(-摘要匹配, -剩余容量槽位, 节点标识)
```

先判定节点是否摘要匹配、活动租约是否达到 `lease_limit`、剩余槽位是否足够。不可用节点不能被后续排序键“绕过”限制；所有候选不可用时调度器记录容量排队并重试。非选中候选保留 `DIGEST_MISMATCH`、`E_LEASE_LIMIT`、`E_CAPACITY`、`FEWER_FREE_SLOTS` 或 `NODE_ID_ORDER` 否决理由。

## 6. 真实回环协议

每个请求是一行紧凑 UTF-8 JSON，每个回执也是一行 JSON。Lease 请求示例：

```json
{"artifact_keys":["src/app"],"lease_id":"node-a:lease:1:3","lease_ms":350,"op":"lease","protocol_version":1,"request_seq":3,"service_delay_ms":0}
```

回执必须含：

- `ack_version`；
- `node_id` / `request_seq` / `lease_id`；
- `artifact_keys` / `digests` / `refs`；
- `capacity.{used_slots,total_slots,free_slots,active_leases,lease_limit}`；
- `expires_mono_ms` / `elapsed_mono_ms` / `mono_ms`。

`NodeClient` 使用真实 `socket.create_connection` 与 `settimeout`。错误码：

- 端口预检/子进程 bind 失败：`E_PORT`；
- 握手连接/读取超时：`E_HANDSHAKE`；
- 回执缺字段/非法 JSON/身份不符：`E_ACK`。

以上错误均写入 JSONL 结构化日志；节点日志、引擎日志分别位于 `<workspace>/runtime/logs/`。

## 7. 调度与排队

`ParallelLeaseScheduler` 使用真实 `ThreadPoolExecutor(max_workers=global_concurrency)`。每层拓扑制品并行提交，依赖层必须完成后才进入下一层。每个网络请求持有三层限额：

- 全局 `global_concurrency`；
- 制品所属阶段的 `stage.concurrency`；
- 所选节点的 `lease_limit`，即每节点在途请求配额。

信号量未立即取得时记录：

```json
{
  "queue_seq": 4,
  "kind": "node",
  "request_seq": 3,
  "artifact_key": "src/app",
  "node_id": "node-c",
  "stage_id": "source",
  "enqueued_mono_ms": 12345,
  "wait_started_mono_ms": 12345,
  "wait_ended_mono_ms": 12402,
  "waited_mono_ms": 57
}
```

容量/租约上限耗尽时另记录 `kind="capacity"`。`request_seq` 在拓扑序上预分配，作为稳定标识；队列序号由进程内单调计数器分配。

## 8. 账本事务与唯一归属

核心表：

- `events`：追加式事件序列；
- `nodes`：每节点容量余量与活动租约计数；
- `leases`：`PENDING/ACTIVE/EXPIRED/RELEASED/INVALIDATED/UNKNOWN/CANCELLED`；
- `claims`：`artifact_key -> lease_id/node_id`，主键保证一个制品同时只有一个容量所有者；
- `summaries`：每节点/制品当前摘要与引用；
- `sources`：可复用来源有效性；
- `decisions`：复用、仲裁和否决理由；
- `progress`：编辑哈希、事件游标、阶段、已完成键。

每次 `LEASE_GRANTED / LEASE_RENEWED / LEASE_EXPIRED / QUOTA_RELEASED / LEASE_UNKNOWN / LEASE_INVALIDATED` 都在同一 `BEGIN IMMEDIATE` 事务内更新事件、容量、租约/claim、来源与进度，最后重算 `meta.content_digest`。

到期只依据 `time.monotonic_ns() // 1_000_000`。`_close_active_lease` 只接受 `ACTIVE` 行；因此到期线程与配额释放线程即使同时尝试，也只有一个事务能把同一租约迁出 `ACTIVE`。节点容量槽位与活动租约计数在该唯一事务中同时释放。

## 9. 增量推进

启动时先真实重启节点并重新生成全部摘要，`sync_digests` 将节点当前摘要/引用与账本 `summaries` 比较。规范摘要或引用发生变化时：

- 无效化相关 `sources`；
- 将相关活动租约转为 `INVALIDATED` 并释放容量；
- 从进度完成集合删除相关键；
- 稳定返回受影响键集合。

调度器基于当前 `sources` 计算复用闭包，只对不可复用键发起真实租约。未编辑键可零请求复用。全量重建使用新数据库，最终节点摘要、来源集合、复用结论、仲裁结果、节点容量和完成键与增量运行一致。

删除节点或制品时，`reconcile_spec` 在一个事务中清理当前 `nodes/summaries/sources/decisions/claims` 投影并更新进度；历史 lease/event 保留。由于 `LEASE_GRANTED` 事件记录授予当时的节点容量与租约上限，验证器即使面对已从当前规格删除的节点，也能独立重放其历史容量变化。

## 10. 中断恢复

授予流程先提交 `PENDING`，再执行真实网络授予，最后在单个事务中提交 `ACTIVE`。进程被强制终止时：

- 已完整提交的事务保留；
- 未提交事务由 SQLite 回滚，不产生可见租约；
- 下次启动读取 `clean_shutdown=0`，把遗留 `ACTIVE/PENDING` 标为 `UNKNOWN`；
- 相关来源失效，相关键必须重新申请；
- 强制终止路径不写标准退出报告。

节点子进程带父进程看门狗；父进程消失后主动退出，因此恢复启动不会长期占用旧端口。

## 11. 独立验证

`verifier.py` 只读原始规格与 SQLite（URI `mode=ro`），不启动节点、不调用调度器，并独立执行：

1. 重新运行 NumPy 摘要规则，逐制品键/节点比较 `summaries`；
2. 从事件逐条重放容量占用、租约状态和全局 claim，验证不超容量/租约上限且同一判定不重复释放；
3. 从已产出键和规范摘要重新推导有效来源集合；
4. 重新推导复用闭包、唯一服务节点和否决理由；
5. 校验进度游标、整数版本和全账本内容摘要。

首个差异以 `E_VERIFY` 返回，携带 `artifact_key`、`node_id`、`field`、`expected` 与 `actual`。

## 12. “逐字段一致”的口径

验收比较的是 `business_snapshot`：

- 节点容量/活动租约/剩余槽位与每节点摘要；
- 每制品规范摘要、引用、来源集合、可复用结论、最终服务节点；
- 当前活动租约（最终为空）；
- 编辑哈希、最终阶段与完成键集合。

墙钟/单调时钟时间戳、排队等待毫秒、事件总序号以及恢复审计事件本身属于运行历史，不应也不可能在“发生过强制中断的账本”和“全新账本”之间相同；它们被保留用于审计而不参与业务一致性比较。所有业务结论字段均参与逐字段递归比较，样本结果为 0 差异。

## 13. Windows 运行注意事项

- 子进程通过 `sys.executable -m leasecache.node_server` 启动，不依赖 POSIX fork。
- Windows 使用 `CREATE_NEW_PROCESS_GROUP`；停止时先发送 `shutdown`，超时后用 `taskkill /T /F`。
- 父进程存活检测使用 `OpenProcess/GetExitCodeProcess/CloseHandle` 完整 ctypes 签名。
- 端口预检在 Windows 上不设置 `SO_REUSEADDR`，避免误判已占用监听端口；POSIX 仅为绕过 `TIME_WAIT` 设置。
- GUI 使用随 Python 分发的 Tkinter，不引入第三方图形库。
