# 本地推理分片下发诊断技术说明

本实现只依赖 Python 3.10+ 标准库、NumPy 与可选 Tkinter；不使用 Docker、外部服务、账号、GPU 或非回环地址。

## 目录与入口

- `shard_diag/spec.py`：节点、分片、配额版本链、事件流 JSON 强校验。
- `shard_diag/samples.py`：固定种子样本、NumPy 整数定点前向、损失摘要。
- `shard_diag/protocol.py` / `shard_diag/node_worker.py`：真实 TCP JSON-line 客户端与节点子进程。
- `shard_diag/graph.py`：稳定拓扑、缺失依赖、下游失效传播。
- `shard_diag/quota.py`：配额版本解析、定点水位、三级误差界与决胜。
- `shard_diag/ledger.py`：SQLite 原子事务、租约、占用、水位、快照和重放。
- `shard_diag/engine.py`：就绪冻结、三重约束调度、并行下发、提交和全量重建比对。
- `shard_diag/runtime.py`：节点子进程启动、端口探测、SIGKILL 与重启。
- `shard_diag/workspace.py`：增删分片、替换样本、追加配额版本和传播报告。
- `shard_diag/acceptance.py`：一键端到端验收并写 `artifacts/acceptance-result.json`。
- `declarations/`、`samples/`、`ledger/`：声明、确定性样本和 SQLite/快照/节点日志。
- `tests/test_diagnostics.py`：11 个覆盖异常链路和验收语义的测试。

常用命令：

```bash
python3 -m pip install -r requirements.txt
python3 -m shard_diag.cli validate
python3 -m shard_diag.cli regenerate-samples
python3 -m shard_diag.cli plan
python3 -m shard_diag.cli run --batch-size 3
python3 -m shard_diag.cli replay
python3 -m shard_diag.cli propagate --changed-shard shard-a
python3 -m shard_diag.cli replace-sample shard-a 0 '[4,-3,2,1]' 7
python3 -m shard_diag.cli set-scale 100
python3 -m shard_diag.cli acceptance
python3 -m shard_diag.cli gui
```

## 1. 声明强校验与 `E_SPEC`

加载顺序固定为 `nodes.json`、`shards.json`、`quota_versions.json`、`events.json`。校验按数组顺序返回第一个非法条目：

- `node_id`、`shard_id` 非空且唯一。
- 节点 `host` 必须等于字符串 `127.0.0.1`；端口范围为 1024–65535，且所有节点全局唯一。
- `concurrency`、`lease_limit` 必须是非负整数；布尔值不按整数接受。
- 定点缩放因子必须属于 `{1,10,100,1000,10000,1024,4096}`。
- 配额 `version` 为正整数且唯一；根必须是 `version=1,parent_version=null`，父版本必须存在且号更小，因此父链单调且无环。
- 事件 `seq` 必须唯一、为正并按流单调排列，`type` 必须属于支持的事件集合。

错误对象形如 `{"ok": false, "code": "E_SPEC", "entry": "nodes[1]", "field": "port", ...}`。

分片的悬空 `upstream` 不在加载期删除，而在就绪判定中冻结并记录 `missing_upstream`，用于验证依赖缺失语义。

## 2. 真实回环 TCP 与错误码

节点由以下命令启动为独立子进程：

```bash
python3 -m shard_diag.node_worker --root . --node-id node-a
```

协议是一行一个 JSON 对象：

1. 客户端发送 `hello`，节点回 `hello_ack`。
2. 客户端发送 `start`，节点回 `start_ack`。
3. 客户端按任意批次发送 `batch`，节点逐批回 `batch_ack`。
4. 客户端发送 `finish`，节点回 `receipt`。
5. 续约使用独立连接发送 `renew`，节点做版本号 CAS。

连接超时使用 `socket.create_connection(..., timeout=...)`，读取使用 `sock.settimeout(...)` 后调用 buffered reader 的 `readline()`。没有用线程 `join(timeout)` 冒充套接字超时。

- 绑定端口收到 `EADDRINUSE` 或父进程预绑定探测失败：`E_PORT`。
- 连接或 `readline()` 触发真实 `socket.timeout`：`E_TIMEOUT`。
- 租约版本不符：`E_LEASE`。
- 账本重放发现不一致：`E_REPLAY`，带首个 `entry=shard_id` 与 `field`。

## 3. 确定性样本与损失摘要

样本种子按显式字符串派生：

```text
seed_material = "sample|seed={node.seed}|shard={shard_id}|quota={quota_version}"
seed = uint32(sha256(seed_material)[:8])
rng = numpy.random.RandomState(seed)
```

基础样本由稳定的 `RandomState.randint` 与整数教师公式生成；`samples/overrides.json` 中的单行替换按 `shard_id -> row_index -> {x,y}` 覆盖。`samples/<shard>.json` 可由固定种子和覆盖项反复重建。

前向与累加规则：

```text
pred_q[r]   = sum_j x[r,j] * weights_q[j] + bias_q
target_q[r] = y[r] * fixed_scale
residual_q  = pred_q - target_q
SSE         = exact_integer_sum(residual_q^2)
loss_sum_q  = floor(SSE / fixed_scale)       # 非负数，向下取整
mean_q      = floor(loss_sum_q / row_count)
```

NumPy 负责随机样本、int64 前向和分块；平方和转入 Python 任意精度整数精确累加，最后按有符号 int64 饱和。这样任意 TCP 批次切分都不会因中间 int64 溢出而改变结果。

损失摘要输入包含：

```text
algorithm, node_id, shard_id, participating_rows, quota_version,
param_digest, sample_digest, fixed_scale, sum_loss_q, mean_loss_q,
overflow flags and truncation rule
```

最终：

```text
loss_digest = "ld1:" + sha256(canonical_json(basis))
```

规范化 JSON 使用 UTF-8、排序键、无空白、禁止 NaN。下发顺序、批次大小和整体并发只影响时间，不进入 `loss_digest`。主进程收到回执后会独立复算并比对摘要。

## 4. 稳定拓扑与就绪冻结

`stable_topology()` 使用 Kahn 算法；每一步在零入度集合中按 `shard_id` 字典序用堆选择，输出稳定拓扑序。缺失上游被视为外部引用，同时记录在 `Topology.missing`。

分片状态包括：

- `committed`：账本中有已提交且带 `loss_digest` 的结局。
- `ready`：所有已知上游均已提交且摘要可用；悬空依赖除外时不会就绪。
- `waiting_upstream`：上游尚未形成结局。
- `frozen_missing_dependency`：存在清单中不存在的上游，字段 `missing_upstream` 给出排序后的标识。
- `frozen_upstream_failure`：上游为 `unknown`、`failed` 或回收状态。
- `frozen_cycle`：已知分片依赖图存在环。
- `leased`：存在活动租约。

只有 `ready` 且无活动租约的分片会进入授权流程。冻结分片不会调用 `grant_lease()`，因此不会预占全局或节点资源。

## 5. 定点水位与误差裁决

设会话缩放因子为 `S`，原始资源整数为 `U`：

```text
Q(U) = saturate_int64(U * S)
```

资源单位、每节点租约上限、并发配额和已用量都输出原始整数与定点 q 值。所有水位量非负，除法采用 floor；另提供 `trunc_div_toward_zero()` 用于有符号场景，方向为绝对值相除后按符号向零截断。乘法和转换结果统一做 signed int64 饱和。

误差界单位为 q：

- `quota_baseline_error_q`：跨缩放因子转换父版本基线时，每个标量至多 1 q；当前原始整数直接量化且未换尺度时为 0。
- `shard_accumulation_error_q`：总分片需求拆成 `B` 批时，保守上界 `B-1` q；精确余数为 `(U*S) mod B`。
- `node_comparison_error_q`：归一化节点比较使用 `floor(remaining_q*S/capacity_q)`，反算到资源 q 的保守上界为 `ceil(capacity_q/S)`。
- `synthetic_error_bound_q`：上述三项之和；候选裁决再叠加该候选自身的 `B-1` 分批界。

若 `abs(remaining_q - demand_q) <= candidate_error_bound_q`，决策记录 `"error_affected": true`。受影响候选不再依赖微小数值差，而按 `(stable_topology_rank, shard_id)` 显式选择，并在每个 decision 中写 `tie_rule`、`tie_break_key`、`error_tie_winner`。

## 6. 失效传播与空白重建一致性

`propagation_report()` 从以下直接变更集合出发：

- 显式修改的分片；
- 节点声明变化引用到的全部分片；
- 配额版本中节点配额或单批需求作用到的分片；
- 全局预算或会话尺度的真实变化；
- 激活新的 `quota_version` 时，因回执和摘要绑定配额版本而使全部分片带 `activated_quota_version_binding`。

随后沿子边做 BFS，得到所有直接和间接下游。输出按稳定拓扑序与 `shard_id` 排序，每一步的 `basis` 标明 `direct_change`、`upstream:<id>`、节点引用或配额原因。

增量重算时：

1. `mark_invalidated()` 只把失效集合的结局状态改为 `invalidated`。
2. 未失效行、租约原始 blob 和 `shard:<id>` 水位 blob 不执行 UPDATE。
3. 重算后用 SHA-256 比较未失效记录的 `outcome`、`lease_blob`、`water_blob`，实现逐字节保持校验。
4. 另建空白 SQLite 账本，从当前声明全量真实 TCP 调度；`compare_deterministic_outcomes()` 逐字段比较。

实例级 `lease_id`、墙钟/单调时间戳、历史计数和快照版本不参与确定性字段比较；这些字段的类型和存在性仍在结构中保留。

## 7. 并行调度与租约互斥

调度器使用 `ThreadPoolExecutor` 和 `FIRST_COMPLETED`：

- 授权在主线程和 SQLite 立即事务中完成，避免两个线程同时写额度。
- 每个循环按优先级、拓扑序和分片 ID 形成候选顺序。
- 授权时原子检查全局预算、节点资源上限、每节点 `lease_limit`、每节点 `concurrency`。
- 一个分片在数据库上有部分唯一索引 `WHERE status='active'`，保证同一 `shard_id` 不会有两个活动租约。

租约包含 `granted_monotonic_ns`、`expires_monotonic_ns`、`grant_number` 和 `lease_version`。到期判断只使用 `time.monotonic_ns()`。续约先向节点发送 `expected_version`，节点 CAS 成功后返回新版本，账本再做条件更新；版本不符抛 `E_LEASE`，且不更新到期时间。

回收在同一事务中按固定归属规则分区：

1. 先把 `expires_monotonic_ns <= now` 的活动租约标为 `reaped_expired`。
2. 再把仍为 `active` 且 `quota_version <> current` 的租约标为 `reaped_generation`。

因为第一步已经改变状态，同一租约和同一笔 `resource_holdings` 不可能同时进入两个回收集合。活动占用汇总只统计 `resource_holdings.status='active'`，回收时同一行只更新一次。

## 8. SQLite、快照、强杀恢复与 `E_REPLAY`

默认账本为 `ledger/diagnostic.sqlite`，验收账本为 `ledger/acceptance.sqlite`。SQLite 使用 `synchronous=FULL`。

每个分片完成时，单个 `BEGIN IMMEDIATE ... commit` 事务写入：

- `shard_outcomes`：完整回执、损失摘要和规范化 outcome blob；
- `leases`：租约归属与状态；
- `resource_holdings`：资源单位占用/释放；
- `water_records`：分片水位依据及当前全局/节点水位；
- `progress_cursor`：拓扑位置、最后分片、完成数和快照版本；
- `commit_events`：不可变事件载荷。

所有关键表都有整数 `format_version=1` 与 `content_digest`（blob 的 SHA-256）。

`ledger/session_snapshot.json` 外层包含 `format_version`、整数 `snapshot_version`、`content_digest` 和完整 `state`。加载时若 JSON 损坏、格式版本不符、状态摘要不符或快照版本不一致，直接从 SQLite 账本重建并整体替换文件，不做增量字段合并。

节点被 `SIGKILL` 后：

- 已提交事务中的结局仍在 SQLite；
- 未提交的活动租约和占用由 `recover_interrupted()` 一次性改为 `unknown_reclaimed` / `reclaimed_unknown`；
- `shard_state` 标为 `unknown`，调度器将其作为需要重跑的根候选或等待其上游；
- 不依赖被强杀进程生成退出报告。

`replay_and_verify()` 按事件 seq 重放：`shard_committed` 写入预期结局，`invalidation` 移除已失效预期，后续提交再覆盖。最终按稳定分片序比对行字段；第一个不一致会抛出带 `entry=<shard_id>`、`field=<field>` 的 `E_REPLAY`。

## 可独立复核路径

最快路径：

```bash
python3 -m pytest -q
python3 -m shard_diag.cli acceptance
```

然后查看：

- `artifacts/pytest-output.txt`
- `artifacts/acceptance-result.json`
- `artifacts/cli-run.json`
- `artifacts/replay-result.json`
- `artifacts/environment.txt`
- `docs/VERIFICATION.md`
