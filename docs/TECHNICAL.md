# Quota Lab 技术说明

本地多节点分片配额水位、定点误差传播、租约归属和崩溃恢复实验环境。仅依赖 Python 3.11 标准库与 `numpy==1.26.4`，不使用 Docker，不访问外部站点。

## 1. 快速复现

```bash
bash scripts/setup_venv.sh
bash scripts/acceptance.sh 2>&1 | tee artifacts/acceptance-output.txt
```

一键脚本会完成：

1. 创建/修复 `.venv` 并校验 Python、NumPy 版本。
2. 生成 `configs/` 声明、容量版本文件和 `samples/event_stream.jsonl`。
3. 校验 `E_SPEC` 首个错误字段。
4. 验证回环端口绑定/释放、`E_PORT` 和真实套接字 `E_TIMEOUT`。
5. 运行 15 个 unittest/集成测试。
6. 启动两个真实节点子进程并完成 6 个分片。
7. 执行空白重建一致性检查。
8. 容量版本 v1 → v2 演进、失效传播、字节保留和子集重算。
9. `SIGKILL` 强杀、SQLite 恢复、配额回收和重放校验。

关键结果写入：

- `artifacts/run-result.json`：正常调度、裁决、等待和误差报告。
- `artifacts/network-result.json`：端口/超时注入结果。
- `artifacts/unittest-results.txt`：自动化测试输出。
- `artifacts/rebuild-check.json`：空白重建逐字段一致性。
- `artifacts/evolve-result.json`：失效集合、传播依据、字节保留和重算结果。
- `artifacts/recovery-result.json`：强杀后的未知分片、配额回收和重跑结果。
- `artifacts/acceptance-summary.json`：验收摘要。

## 2. 模块边界

- `quota_lab/spec.py`：声明/清单/容量文档加载、强类型校验、稳定拓扑排序和摘要。
- `quota_lab/samples.py`：固定种子 NumPy 样本生成、JSONL 事件流加载与强校验。
- `quota_lab/fixedpoint.py`：缩放、整数 floor 除法、残差、误差向量和贡献占比。
- `quota_lab/aggregation.py`：NumPy 分组聚合和分片水位计算。
- `quota_lab/node_server.py`：节点子进程、真实 TCP JSONL 服务、聚合、租约 CAS 和容量推进。
- `quota_lab/processes.py`：`subprocess.Popen` 编排、握手超时、端口占用诊断和 Cluster 生命周期。
- `quota_lab/protocol.py`：真实 TCP 连接、发送/接收超时和结构化错误映射。
- `quota_lab/leases.py`：租约字段、单调时钟过期、乐观续约和到期/版本作废竞争控制。
- `quota_lab/scheduler.py`：`ThreadPoolExecutor` 投射、三重配额、等待计时、边界裁决和账本提交。
- `quota_lab/invalidation.py`：直接/间接失效推导与稳定拓扑传播步骤。
- `quota_lab/ledger.py`：SQLite schema、原子结局事务、内容摘要、恢复和版本不符整体替换。
- `quota_lab/verify.py`：空白重建逐字段对比和跨版本字节保留校验。
- `quota_lab/cli.py`：命令行入口。

## 3. 声明与校验

示例入口：

```bash
.venv/bin/python -m quota_lab.cli init-demo
.venv/bin/python -m quota_lab.cli validate \
  --declaration configs/declaration.json \
  --shards configs/shards.json \
  --capacity configs/capacity.json \
  --events samples/event_stream.jsonl
```

校验规则包括：

- `node_id`、`shard_id` 唯一。
- `host` 必须是字符串字面量 `127.0.0.1`。
- 端口必须是整数 `1..65535`，且节点间全局唯一。
- `concurrency_quota`、`lease_limit` 必须是非负整数；布尔值不按整数接受。
- `seed` 必须位于 `0..2^32-1`。
- `depends_on` 引用必须存在、不能自依赖，DAG 不能有环。
- `resource_units` 必须是正整数。
- 事件流的 `session_id`、分片归属、连续 `seq`、分组范围和行数必须与声明一致。

所有声明错误返回：

```json
{"ok": false, "error": {"code": "E_SPEC", "first_error_field": "declaration.nodes[0].port"}}
```

## 4. 真实进程与 JSON Lines

每个节点由父进程用当前 `.venv` Python 启动：

```text
python -m quota_lab.node_server --spec <runtime-spec.json> --node-id <id>
```

节点在 `127.0.0.1:<port>` 上真实 `bind/listen/accept`。每个请求是一行 JSON，每个回执也是一行 JSON。父进程使用 `socket.create_connection(..., timeout=...)` 和 `sock.settimeout(...)` 同时约束连接与读取。

聚合回执包含：

- `node_id`
- `shard_id`
- `capacity_version`
- `summary`
- `elapsed_ms`：节点内 `time.monotonic_ns()` 差值整除 `1_000_000`
- `monotonic_ms`

端口已被占用时，子进程绑定失败并输出 `E_PORT`，父进程在握手阶段报告 `E_PORT` 和字段 `nodes.<id>.port`。读取超时映射为 `E_TIMEOUT`。

## 5. 定点误差传播

会话缩放因子为 `p ∈ {0,1,2,3}`，`Q = 10^p`。

### 5.1 量化和截断

- 资源、预算、配额和已用量先乘以 `Q` 成为整数 `*_q`。
- 所有除法使用 Python 整数 `//`，规则显式声明为 `floor_toward_negative_infinity`；本样例中的资源和样本值均非负，因此等价于向下截断。
- 一次 `n // d` 的残差为 `r = n - (n // d) * d`，精确误差是有理数 `r/d`，保守整数界为 `1 q-unit`。
- 所有 q 值按 signed int64 边界饱和：超过 `2^63-1` 记为 `INT64_MAX`，低于 `-2^63` 记为 `INT64_MIN`，并在计算结果中保留 `saturated` 标记。

分片级误差来自：

1. NumPy 聚合均值定点化：`mean_q = floor(sum * Q / row_count)`。
2. 水位传播：`water_q = floor(demand_q * pressure_num / pressure_den)`。

节点级误差来自节点水位均值除法；全局误差来自全局平均除法，以及分片残差沿节点和全局求和后的保守界。

### 5.2 裁决

对候选分片 `i`，输出整数误差向量：

```text
E_i = E_shard_components + E_node_components + E_global_components
```

预算差值：

```text
G_q = budget_q - active_demand_q - candidate_demand_q
```

- `G_q > E_i`：误差界外可确信容纳。
- `|G_q| <= E_i`：预算边界不可区分，标注 `budget_indistinguishable=true`。
- 硬配额仍以精确整数需求为准：只有 `G_q >= 0` 才会放行，因此真实在途需求峰值不会超过预算。

两个候选水位差：

```text
D_q = |water_a_q - water_b_q|
B_pair = node_judgement_bound_a + node_judgement_bound_b
```

若 `D_q <= B_pair`，二者在误差界内不可区分，按 `(topological_rank, shard_id)` 字典序放行，并完整输出两个 `error_vector`、差值、边界和决胜键。

三级贡献使用整数基点：

```text
share_bp = contribution_q * 10000 // total_q
```

当某级 floor 占比大于等于 `dominance_threshold_bp`（样例为 5000，即 50%）时标记为主导级；否则为 `mixed_below_threshold`。

## 6. 租约和三重调度

租约字段包括 `lease_id`、`shard_id`、`node_id`、`granted_monotonic_ms`、`expires_monotonic_ms`、`renew_count`、`lease_version` 和容量版本。

调度约束：

```text
active_resource_demand_q <= global_budget_q
active_shards(node) <= concurrency_quota(node)
active_leases(node) <= lease_limit(node)
```

所有节点调用只发生在 `ThreadPoolExecutor` worker 中；准入检查和配额计数在同一把调度锁下完成。不能准入的稳定就绪分片记录 `wait_started_ms`，放行时记录 `wait_ended_ms` 和整数 `wait_ms`。

Kahn 拓扑排序在每批零度分片中按“整数 `priority` 降序、`shard_id` 升序”选择，因此最终 `topological_rank` 已包含显式优先级；边界决胜再使用 `(topological_rank, shard_id)`。约束结果同时输出精确预算判定、`budget_q + aggregate_error_bound_q` 的误差扩展上限，以及 `peak_q + error_bound_q <= budget_q` 的保守误差裕量判定。

续约时父进程把 `lease_id` 和 `expected_version` 发到节点子进程做 CAS。版本不符返回 `E_LEASE`，节点不更新版本，父进程也不延长 `expires_monotonic_ms`。`renew_count >= lease_max_renewals` 后直接以 `E_LEASE` 拒绝。

到期只能由 `time.monotonic_ns()` 判断。到期回收和容量版本作废都通过 `active -> expired|version_void` 的一次性状态转换竞争；SQLite/内存锁保证只有一个转换成功并释放一次租约额度。

## 7. 容量演进和仅失效子集

`quota_lab/invalidation.py` 比较新旧声明：

- 节点 `capacity_version`、`concurrency_quota`、`lease_limit` 变化：该节点全部分片直接失效。
- 分片 `resource_units`、`sample_rows`、依赖或节点引用变化：该分片直接失效。
- 全局缩放/核算参数变化：全部分片失效。
- 随后沿反向依赖图 BFS 推导所有直接和间接下游。

失效集合按 `(topological_rank, shard_id)` 稳定排序，并输出 direct reason 和每条依赖传播边。

`evolve` 命令把未失效分片的 `summary_json`、`lease_ownership_json`、`watermark_json` 原文复制到新摘要的 SQLite 账本，再只调度失效集合。之后：

- `byte_preservation_after_run` 对比旧库和新库中的原始 UTF-8 字节及 SHA-256。
- `rebuild_consistency` 从当前声明和事件流重新运行 NumPy/定点核算，逐字段对比摘要、水位和确定性租约归属。

真实单调时间戳和节点耗时属于运行时观测值；一致性命令比较其类型/顺序，不要求跨进程的毫秒起点相同。租约“归属”视图只包含确定性的 ID、节点、容量版本、最终版本号和续约次数，因此可逐字段复现。

## 8. SQLite、快照替换和恢复

账本包含：

- `ledger_header.format_version INTEGER`
- `meta` 中的 `ledger_format_version`、`spec_digest`、`content_digest`
- `shard_records`
- `leases`
- `ledger_events`
- 单行 `progress_cursor`

每个分片结局在一个 `BEGIN IMMEDIATE ... COMMIT` 中写入聚合摘要、租约、资源占用、水位、误差归因、receipt、outcome digest 和进度游标，然后更新账本内容摘要。

重新打开 SQLite 时，如果整数格式版本或 `spec_digest` 与当前声明不一致，会删除活动 DB 及 `-wal`/`-shm` 后整体重建，不做字段级增量合并。JSON 会话快照也执行同样的 version/digest 识别和整体替换。

强杀演练中，`--kill-after-commits` 在首个结局事务提交成功后由主进程执行 `SIGKILL`。因此：

- 已提交分片可恢复。
- 已写 `running` 起始事务但未提交结局的分片，重启后标记为 `unknown_uncommitted_outcome`。
- 恢复事务先把这些记录的 `active_q` 置零、租约标记 `quota_reclaimed` 并写 `quota_reclaimed` 事件，然后才允许调度。
- 未提交分片必须重跑。
- 被杀进程没有机会写 `artifacts/crash-result.json`；恢复结果明确包含 `no force-termination exit report can be reconstructed`。

Linux 子进程通过 `PR_SET_PDEATHSIG=SIGTERM` 跟随父进程退出，因此父进程被 `SIGKILL` 后节点端口也会释放；这一点已在演练后通过端口检查确认。

## 9. 主要命令

```bash
# 完整验收
bash scripts/acceptance.sh

# 仅测试
.venv/bin/python -m unittest discover -s tests -v

# 正常运行
.venv/bin/python -m quota_lab.cli run \
  --declaration configs/declaration.json \
  --shards configs/shards.json \
  --capacity configs/capacity.json \
  --events samples/event_stream.jsonl \
  --ledger artifacts/run.sqlite3 \
  --result artifacts/run-result.json \
  --reset-ledger

# 重建校验
.venv/bin/python -m quota_lab.cli rebuild-check \
  --declaration configs/declaration.json \
  --shards configs/shards.json \
  --events samples/event_stream.jsonl \
  --ledger artifacts/run.sqlite3

# 容量版本演进
.venv/bin/python -m quota_lab.cli evolve \
  --old-declaration configs/declaration.json \
  --old-shards configs/shards.json \
  --new-declaration configs/declaration.v2.json \
  --new-shards configs/shards.json \
  --new-capacity configs/capacity.v2.json \
  --events samples/event_stream.jsonl \
  --old-ledger artifacts/run.sqlite3 \
  --new-ledger artifacts/evolved.sqlite3 \
  --result artifacts/evolve-result.json \
  --reset-new-ledger
```
