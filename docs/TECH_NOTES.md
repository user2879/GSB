# CacheFlow 技术说明

## 1. 范围与运行模型

CacheFlow 是本地回环多节点制品缓存流水线：

- 每个 `node_id` 对应一个独立 Python 子进程，绑定 `127.0.0.1:<port>`。
- 节点与流水线之间使用换行分隔 JSON（JSONL），握手、连接和读取均设置真实 `socket` 超时。
- 流水线先对事件做确定性规划，得到归属节点、槽位和代数链，再通过真实 TCP 调用节点。
- 全局并发与每节点在途上限由 `cacheflow/scheduler.py` 的条件变量和 `ThreadPoolExecutor` 共同执行；等待时间来自 `time.monotonic_ns()`。
- SQLite 账本保存规格摘要、事件、代数迁移、当前租约、槽位占用、归属裁决、回收、在途请求、提交哈希链和规范快照。

主要模块：

- `cacheflow/spec.py`：三份 JSON 声明加载与 `E_SPEC` 强校验。
- `cacheflow/content.py`：确定性分片字节和绑定摘要。
- `cacheflow/statemachine.py`：`granted`、`renewed`、`released`、`expired` 显式迁移表与 `E_GEN` 代数链。
- `cacheflow/ownership.py`：候选节点评估、摘要匹配、容量/节点 ID 排序与否决理由。
- `cacheflow/scheduler.py`：真实线程池、全局/节点在途闸门、单调时钟等待记录、独占控制屏障。
- `cacheflow/protocol.py` / `cacheflow/node.py` / `cacheflow/cluster.py`：JSONL 客户端、节点子进程和进程生命周期。
- `cacheflow/ledger.py`：SQLite 原子事务、整数 `format_version=1`、摘要链、未知在途请求恢复。
- `cacheflow/engine.py` / `cacheflow/report.py`：确定性重放引擎和逐字节规范结论。
- `cacheflow/evolve.py`：配额/节点集合演进的影响图、子集重算和全量重建一致性校验。
- `cacheflow/replay.py`：只读 SQLite 连接的独立重放校验。
- `cacheflow/acceptance.py`：动态端口样本、真实运行、`SIGKILL`、未知请求回收、重启和逐字节复核。

## 2. JSON 声明

示例位于 `examples/nodes.json`、`examples/artifacts.json`、`examples/events.json`。

强校验包括：

- `node_id`、`artifact_key`、租约授予唯一。
- `host` 必须是字符串字面量 `127.0.0.1`；`(host, port)` 全局唯一。
- `capacity_slots`、`max_leases`、`global_concurrency`、`quota_version` 必须是正整数；布尔值不按整数接受。
- 事件 `seq` 必须为正整数，且在数组中严格递增，因此不重复。
- 分片索引从 `0` 连续；第一个区间从 `0` 开始，后一片 `start` 等于前一片 `end`，最后 `end == total_bytes`，且不得越界。
- 事件引用的制品、节点和先于引用出现的租约授予必须存在。

首个错误抛出 `SpecError(code="E_SPEC", entry=..., field=...)`，CLI 输出 JSON 错误对象并拒绝调度。

## 3. 确定性内容与摘要

分片字节使用版本化、长度前缀的显式预像，再由 SHAKE-256 生成指定长度：

```text
length_prefixed("cacheflow.content.v1")
length_prefixed(node_seed)
length_prefixed(node_id)
length_prefixed(artifact_key)
length_prefixed(shard_index)
length_prefixed(length)
```

分片摘要为 SHA-256，预像依次包含：

```text
length_prefixed("cacheflow.shard-digest.v1")
length_prefixed(node_id)
length_prefixed(artifact_key)
length_prefixed(shard_index)
length_prefixed(len(content))
length_prefixed(content)
```

因此摘要显式绑定节点、制品键、分片序号和内容字节。流水线写入后核对节点回执，读取后再次核对摘要和字节。

## 4. 唯一归属裁决

每个候选节点先生成全部分片的派生摘要，并可与实际读写回执或外部观察摘要比较。排序规则固定为：

```text
digest_match 降序 -> free_slots 降序 -> node_id 升序
```

候选还必须满足：

- 空闲槽位不少于分片数；
- 当前活跃租约数小于 `max_leases`；
- 若声明固定 `node_id`，只能选择该节点。

所有候选都写入 `candidates` 对照列表；非胜者记录 `digest_mismatch:*`、`insufficient_capacity`、`lease_limit_reached`、`free_slots_desc_lost`、`node_id_order_lost` 等理由。

## 5. 代数状态机

显式许可表为：

| 当前状态 | 触发 | 新状态 |
| --- | --- | --- |
| `pending` | `grant` | `granted` |
| `granted` | `renew` | `renewed` |
| `renewed` | `renew` | `renewed` |
| `granted` / `renewed` | `release` | `released` |
| `granted` / `renewed` | `expire` / `invalidate` | `expired` |

授予后的当前代数为 `1`。接受一次续约、释放、到期回收或作废回收时，代数必须且只能从 `N` 进到 `N+1`。节点收到的请求标签等于当前代数才执行；否则返回：

```json
{"ok": false, "error": {"code": "E_STALE", "current_generation": 2}}
```

节点不前进代数。重放链发现跨代跳变、回退或非法状态迁移时返回 `E_GEN`，并保留原代数。迁移记录按 `(event_seq, ordinal, lease_id)` 稳定排序。

## 6. 调度与真实计时

`Scheduler(global_limit, node_limits)` 在条件变量中维护：

- `inflight_global <= global_limit`
- `inflight_node[node_id] <= node_limits[node_id]`
- 同一 `group`（通常是 `lease_id`）同时最多一个事件，避免同代数请求乱序。
- `exclusive=True` 的配额/控制屏障必须等在途请求排空，并且其后的事件不能提前越过屏障。

作业只在获得名额后投入最大线程数为 `global_limit` 的 `ThreadPoolExecutor`，不会先占满线程再用 `sleep` 模拟队列。每条等待记录包含 `seq`、`node_id`、`wait_started_ms`、`wait_ended_ms`、`wait_ms`，时钟为 `time.monotonic_ns() // 1_000_000`。

TTL 到期由节点内 `time.monotonic()` 截止时间判定；`expire` 请求可真实等待截止点，而不是由测试延迟伪装到期。

## 7. 容量与回收唯一性

每个制品分片在授予时映射到节点上的唯一整数 `slot_id`。SQLite 部分唯一索引保证处于 `provisional` 或 `active` 的记录满足：

- `(node_id, slot_id)` 唯一；
- `(artifact_key, shard_index)` 唯一。

到期回收写 `reclaimed_expiry` / `reason_kind=expiry`；配额换代写 `reclaimed_quota` / `reason_kind=quota_invalidation`；节点移除和种子变化分别写 `reclaimed_node`、`reclaimed_seed`。一次回收关闭该租约当时仍活跃的 hold，因此同一 hold 不会同时进入到期和作废两个回收集合。

规范结论 `conclusion.canonical.json` 使用：

```python
json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
```

归属、容量、迁移、决策和业务回收均进入该结论；本地单调调度耗时只进入 `run-report.json` / `wait_records` 表，不破坏跨运行的容量结论逐字节复核。

## 8. 演进与失效子集

`cacheflow evolve` 沿以下引用传播影响：

```text
quota_version / node_id -> lease_id -> generation -> hold/shard -> event_seq
```

输出稳定排序的：

- `affected_node_ids`
- `affected_lease_ids`，按 `(node_id, last_event_seq, lease_id)`
- `affected_shards`，按 `(node_id, opened_seq, artifact_key, shard_index)`
- `affected_event_seqs`
- 每一步 `{order, from_ref, to_ref, relation, event_seq, basis}`

增量路径从已有规范结论水合状态，只应用影响集合；随后从空白重放原始事件前缀并应用同一演进。`first_difference()` 找到首个 JSON 字段路径，不一致抛 `E_DIVERGE`。未受影响租约的规范字节在比较前后必须相同。

## 9. SQLite 账本与强杀恢复

账本启用整数 `PRAGMA user_version=1`，每张业务提交和快照行也带 `format_version=1`。`commits` 表保存前一摘要和当前规范载荷摘要，形成哈希链。

以下数据在同一个 SQLite 事务中提交：

- 事件与节点回执；
- 代数迁移；
- `lease_current`；
- hold/槽位状态；
- 归属裁决；
- 到期、释放、作废回收；
- 在途请求完成状态和提交摘要。

请求发出前可先在独立事务中写入 `inflight_requests(status='inflight')` 和 `provisional` hold。节点被 `SIGKILL` 后：

1. 已提交的代数、归属和 hold 可由重启节点通过只读 SQLite 连接恢复。
2. 未完成业务提交的请求仍是 `inflight`。
3. `Ledger.recover_unknown_inflight()` 将其标记为 `unknown_rerun_required`，并先把 provisional hold 改为 `reclaimed_unknown`。
4. 重跑成功后再写入正式授予、归属和 active hold。

`SIGKILL` 不产生节点退出报告；验收只记录管理侧观察到的 `returncode=-9` 和套接字 EOF。

## 10. 独立重放命令

```bash
.venv/bin/python -m cacheflow replay \
  --nodes <work>/nodes.json \
  --artifacts <work>/artifacts.json \
  --events <work>/events.json \
  --ledger <work>/cacheflow.sqlite3
```

该命令使用 SQLite `mode=ro` 只读打开账本，核对原始声明摘要，从已提交 `events` 重建状态，与规范化表结论和 `snapshots.latest` 逐字段/逐字节比较。首个差异以 `E_VERIFY` 和 JSON 路径输出。

## 11. 常用命令

```bash
python3 -m venv --without-pip .venv
curl -fsSL https://bootstrap.pypa.io/get-pip.py | .venv/bin/python
.venv/bin/python -m pip install numpy pytest

.venv/bin/python -m cacheflow validate
.venv/bin/python -m cacheflow run --work-dir workspace/sample
.venv/bin/python -m cacheflow replay --ledger workspace/sample/cacheflow.sqlite3
.venv/bin/python -m cacheflow evolve --cutoff-seq 20
.venv/bin/python -m cacheflow accept --work-dir workspace/acceptance
.venv/bin/python -m pytest -q
./scripts/verify.sh
```
