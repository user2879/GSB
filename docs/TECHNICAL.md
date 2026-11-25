# 跨节点制品缓存租约仲裁中间件

本交付是一个可在 Windows 开发机离线运行的 Python 3.11 组件。它从本地 JSON/JSONL 声明读取节点、流水线和事件，启动本机 `127.0.0.1` 节点子进程，通过 TCP JSON Lines 完成分片探测、写入、读取、续约和释放，并在 SQLite 账本中原子提交唯一裁决、容量变化、租约生命周期和恢复游标。

## 快速运行

```powershell
python --version
python -m pip install numpy==1.26.4

python -m lease_arbitrator init-sample --dir samples/demo --force
python -m lease_arbitrator run --sample samples/demo --state-dir .runtime/demo
python -m lease_arbitrator verify --sample samples/demo --state-dir .runtime/demo
python -m lease_arbitrator ui --state-dir .runtime/demo
```

无图形环境时可输出稳定的只读诊断快照：

```powershell
python -m lease_arbitrator ui --state-dir .runtime/demo --text
python -m lease_arbitrator doctor --require-tk
```

也可直接执行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/demo.ps1
powershell -ExecutionPolicy Bypass -File scripts/recovery-check.ps1
```

完整测试：

```powershell
python -m unittest discover -s tests
```

## 声明文件

样本生成器写出：

- `nodes.json`：节点 ID、字面量 `host: "127.0.0.1"`、唯一端口、正整数 `capacity_slots`、正整数 `lease_limit`、整数 `seed`，以及全局并发、分片字节数和套接字超时。
- `pipeline.json`：阶段并发配额、制品键、阶段归属、制品引用和水印定点比例。
- `events.jsonl`：严格递增 `seq` 的 `write`、`read`、`renew`、`release`、`wait`、`quota_change` 事件。
- `expected-core-digests.json`：样本生成时复算的黄金摘要，便于人工核对。

`E_SPEC` 错误结构包含首个非法条目的 `entry_id` 与 `field`。校验覆盖节点 ID/制品键唯一、端口字面量和全局唯一、正整数/整数类型（排除布尔值）、引用存在、DAG 无环、事件序号严格递增等。

## 确定性内容与摘要

内容来源版本：`CONTENT_SOURCE_VERSION = 1`。

分片字节由 NumPy 1.26.4 的 `np.random.PCG64 + SeedSequence` 生成，种子材料显式绑定来源版本、节点种子、制品键 SHA-256 词组、分片序号和字节数；不使用 Python 进程相关的哈希随机化。

摘要规则在 `lease_arbitrator/content.py` 中固定为 SHA-256 长度前缀组合，依次写入：

1. 域字符串 `cross-node-lease-arbitrator/shard-digest/v1`
2. `source_version`
3. `node_id`
4. `artifact_key`
5. `shard_index`
6. `shard_bytes`
7. 原始内容字节

每个字段使用 8 字节大端长度前缀，随后是 UTF-8 字节；原始内容前另加 `raw-bytes:` 标记和 8 字节内容长度。因此摘要同时绑定节点、键、分片序号、长度和真实字节。账本与协议回执均输出 `digest`、`shard_bytes`、`source_version`。

黄金向量：

```text
node_id=node-a seed=1101 key=core shard=0 bytes=96
digest=1c8ffa12a00d2b7777da358ed51bd665700a893c071fbfcd503f32cf54bada5a
```

## 子进程与 JSON Lines 协议

父进程使用 `sys.executable -m lease_arbitrator.node_worker` 启动节点；Windows 下设置 `CREATE_NO_WINDOW`，并通过 `PYTHONPATH` 指向仓库根目录。节点绑定声明的 `127.0.0.1` 端口，每个 TCP 连接按行交换 JSON。

写请求包含：

```json
{
  "action": "write",
  "request_seq": 1,
  "artifact_key": "core",
  "shard_index": 0,
  "lease_ms": 60000,
  "quota_version": 1
}
```

成功回执包含 `lease_id`、`lease_version`、`digest`、`quota_version`、`elapsed_ms`、`shard_bytes` 和 `source_version`。`elapsed_ms` 来自节点内 `time.monotonic_ns()` 差值。

错误分类：

- `E_PORT`：运行前探测到端口仍被占用，或节点日志/退出码显示绑定 `EADDRINUSE/10048`。
- `E_HANDSHAKE`：在显式握手截止时间内未完成真实 TCP 连接和 `ping` JSON 行交换；套接字设置真实 `timeout`。
- `E_FIELD`：回执缺少 `lease_id`、`digest`、`quota_version`、`elapsed_ms` 等必填字段。
- `E_LEASE`：续约版本与节点当前乐观版本不符，或租约已过期/未知。

节点同时监听 stdin EOF 和父 PID；父进程被强杀后管道关闭，节点会自行退出，避免回环端口孤儿。

## 仲裁与预占

写事件先并行 `probe` 候选节点并在本地复算摘要，然后使用稳定排序键：

```text
(digest_match 优先, -remaining_slots, node_id 升序)
```

选择唯一服务节点。相同摘要匹配状态和相同剩余槽位的候选会记录 `indistinguishable=true` / `indistinguishable_node_ids`，最终由节点 ID 字典序打破平局。未中选候选保留为对照，并写入 `DIGEST_MISMATCH`、`LOWER_REMAINING_SLOTS`、`NODE_ID_TIE_BREAK`、`NO_CAPACITY_SLOTS`、`LEASE_LIMIT_REACHED`、`PROBE_FAILED` 等理由。

预占 ID 形如 `RSV-e4-a1-node-a`。裁决后、真实写入前，协调器先将目标节点 `used_slots += 1`。若写入返回失败、回执字段/摘要不合法或事件带 `cancel_after_reserve=true`，则只能用该预占 ID 回滚这一笔槽位；其他预占 ID 不受影响。样本第 4 个事件刻意在容量 2 的 `node-a` 上制造最后槽位预占：预占水印为 `1000/1000`，回滚后恢复为 `500/1000`。

## 并发、租约与水印

`DeliveryScheduler` 背后是真实 `ThreadPoolExecutor`。`QuotaGate` 按固定顺序获取：

1. 全局并发信号量；
2. 阶段并发配额；
3. 节点并发配额（声明中的 `node_concurrency`，缺省等于 `lease_limit`）。

信号量不足时先尝试非阻塞获取；失败后记录 `queued=true`、`wait_start_monotonic_ms`、`wait_end_monotonic_ms` 和 `wait_ms`，再阻塞等待。

租约行保存授予时刻、墙钟到期、单调时钟到期、TTL、租约版本、配额版本和状态。首次授予版本为 1；续约必须携带客户端看到的版本，父进程再转发节点做乐观校验。样本中第 6 个事件从 v1 成功续约到 v2，第 7 个事件仍提交 v1，因此收到 `E_LEASE` 且不延长。

`wait` 事件执行真实 `time.sleep(duration_ms / 1000)`，随后按 `time.monotonic_ms()` 判定到期。到期回收在同一状态转换中释放一个槽位和一个活跃租约计数，并写 `reason="expiry"` 的 `reclaim` 记录。

水印使用会话缩放因子 `S`（样本为 1000）：

```text
watermark_fixed = floor(used_slots * S / capacity_slots)
```

对非负比例，截断方向是向零取整（等价于 floor）。比例量化误差严格小于 `1/S`，定点整数误差严格小于 1 个定点单位。每条水印记录还包含已用、剩余、活跃租约数和租约余量。

## 配额/拓扑替换与陈旧防护

`quota_change` 事件要求新 `quota_version` 单调增大，可携带新的全局/阶段/节点并发配额，也可用完整 `active_nodes` 列表表达节点删除或新增。

替换批次在同一 SQLite 事务中：

- 将所有旧配额版本下仍 `active` 的租约改为 `revoked_quota`，并释放槽位和租约计数；
- 写 `reason="quota_replacement"` 的回收记录；
- 将旧 `current` 仲裁和水印结论标记为 `stale` / `stale_reason="quota_or_topology_replacement"`；
- 写入新配额版本的水印；
- 更新事件游标、当前配额版本、活动节点和配额配置；
- 推进哈希链和账本头部摘要。

同一租约只能从 `active` 转换一次。若到期批次先提交，配额替换查询不到 active 行；若配额替换先提交，到期扫描也不会再次回收，因此同一份容量权益不会产生双重回收。

## SQLite 账本与崩溃恢复

账本格式版本为整数 `1`，数据库包含：

- `meta`：格式版本、声明摘要、当前配额、活动节点、水印比例、创世摘要和头部哈希；
- `ledger_entries`：每个未知标记或事件批次的前后哈希、完整规范化 payload；
- `event_batches` / `unknown_batches`；
- `records`：仲裁、预占、回滚、租约授予、续约、释放、回收、队列指标、水印和错误；
- `node_state`、`leases`、`conclusions`；
- `cursor_state`：已完成事件进度。

哈希链规则为：

```text
entry_digest = SHA256(prev_digest + canonical_json(payload))
```

每个事件批次的记录、租约/状态更新、结论、配额元数据和游标在同一个 SQLite 事务提交。运行前先写 `state/inflight/batch-...json`；提交成功后删除。若用隐藏参数 `--crash-batch N` 注入 `os._exit(97)`，进程不会执行退出清理或写完成报告，inflight 标记保留。

恢复流程：

```powershell
python -m lease_arbitrator recover --sample samples/demo --state-dir .runtime/recovery-demo
python -m lease_arbitrator verify  --sample samples/demo --state-dir .runtime/recovery-demo
# 此时返回 E_UNKNOWN(exit code 8)，提示必须重跑
python -m lease_arbitrator run     --sample samples/demo --state-dir .runtime/recovery-demo
python -m lease_arbitrator verify  --sample samples/demo --state-dir .runtime/recovery-demo
```

`recover` 只把 inflight 标记作为“存在未知批次”的证据并写入 `unknown_batches`，不会据未提交内容恢复容量。重跑时新节点先从已提交的 `leases` 表 rehydrate 乐观版本和剩余 TTL，然后以 attempt+1 重放该事件；成功批次原子地把未知批次标记为 resolved。

## 独立校验与 `E_VERIFY`

`python -m lease_arbitrator verify` 以 SQLite `mode=ro` 打开账本，不写账本，也不启动节点。它只读取：

- 原始 `nodes.json`、`pipeline.json`、`events.jsonl`；
- 账本哈希链、记录、租约表和节点状态表。

校验器按记录顺序重建容量、预占、租约开关和活动节点集合，独立复算内容摘要、水印和稳定仲裁，再与物化表/结论比较。发现差异时抛出 `E_VERIFY`，其 `entry_id` 为制品键（全局项为 `*`），`details.node_id` 为节点，`field` 为首个排序后的不一致字段。若存在 open unknown batch，则返回 `E_UNKNOWN` 并要求重跑。

## Tkinter 诊断界面

`lease_arbitrator/ui.py` 将只读视图模型与 Tk 控件分离：

- `build_view_model()`：以 read-only URI 查询并生成稳定排序数据；
- `render_text_snapshot()`：无 Tk 环境下验证同一视图模型；
- `run_tk()`：延迟导入 `tkinter`，展示水印、同分/不可区分标记、预占结果、排队等待、未知批次、陈旧结论数量和账本版本/摘要。

刷新只执行 `SELECT`，不更新账本。当前 Linux 沙箱缺少系统 `libtk8.6.so`，因此真实 Tk 窗口未能在此环境启动；官方 Windows Python 3.11 安装包通常包含 Tk/Tcl，目标机可用 `python -m lease_arbitrator ui ...` 打开窗口。

## 实际验证摘要

见 `docs/VERIFICATION.md`，其中记录了本次交付环境中的真实测试命令和结果。
