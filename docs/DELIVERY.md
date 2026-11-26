# 分片下发诊断首版交付说明

本目录是一个无容器、无外部服务依赖的 WSL Ubuntu 本地实现。节点是真实独立子进程，在 `127.0.0.1` 上监听 TCP，父调度器通过 JSON Lines 下发样本批次并收回执；账本写入本地 SQLite。

## 一键验证

```bash
cd /home/zsl/feedback/uidfc5909d3-default-397
bash scripts/run_acceptance.sh
```

也可以分步执行：

```bash
python3 -m unittest discover -s tests -v
python3 -m sharddiag.cli validate --spec config/spec.sample.json
python3 -m sharddiag.cli ports    --spec config/spec.sample.json
python3 -m sharddiag.cli run \
  --spec config/spec.sample.json \
  --db artifacts/ledger/sample.sqlite3 \
  --report artifacts/reports/run-report.json \
  --basis-out artifacts/reports/derivations.json \
  --snapshot artifacts/ledger/snapshot.json \
  --log-dir artifacts/logs --fresh
python3 -m sharddiag.cli invalidate --spec config/spec.sample.json \
  --out artifacts/reports/invalidation.json
python3 -m sharddiag.cli evolve-demo --spec config/spec.sample.json \
  --db artifacts/ledger/evolution.sqlite3 \
  --out artifacts/reports/evolution.json \
  --log-dir artifacts/logs/evolution --fresh
python3 scripts/collect_acceptance.py
```

快照校验与整体恢复也可单独执行：

```bash
python3 -m sharddiag.cli snapshot-verify --snapshot artifacts/ledger/snapshot.json
python3 -m sharddiag.cli snapshot-restore \
  --snapshot artifacts/ledger/snapshot.json \
  --db /tmp/restored-ledger.sqlite3
```

可选 Tk 查看器：

```bash
python3 -m sharddiag.cli gui --report artifacts/reports/run-report.json
```

## 主要代码

- `sharddiag/spec.py`：JSON 声明强校验、稳定 Kahn 拓扑排序、Spec 摘要。
- `sharddiag/deterministic.py`：种子派生、NumPy 整数分组聚合、批次合并、推导依据。
- `sharddiag/protocol.py` / `sharddiag/node.py`：JSON Lines TCP 客户端与节点子进程。
- `sharddiag/cluster.py`：`subprocess.Popen` 编排、真实握手、端口占用识别、父死信号。
- `sharddiag/leases.py`：租约状态机、单调时钟过期、版本号、容量代次互斥回收。
- `sharddiag/fixedpoint.py`：`10**0..3` 定点水位、截断余数、INT64 饱和、三段误差归因。
- `sharddiag/engine.py`：依赖就绪、线程池调度、预算/租约/并发三重闸门、等待计时、账本提交。
- `sharddiag/invalidation.py`：节点引用 + 依赖闭包失效传播和逐字节校验。
- `sharddiag/ledger.py`：SQLite 原子批次、链式摘要、崩溃恢复、快照整体替换。
- `sharddiag/scenarios.py`：容量演进、旧版本租约整体撤销和跨代续约 `E_LEASE` 证据。
- `sharddiag/gui.py`：可选 Tkinter 只读查看器。

## 声明与错误码

示例声明：`config/spec.sample.json`；取消/冻结场景：`config/spec.cancel.json`。

校验覆盖：ID 唯一、`host` 必须是字面量 `127.0.0.1`、端口 `1..65535` 且全局唯一、非负/正整数、未知字段拒绝、依赖存在/禁自引/DAG 无环、容量版本严格递增且显式引用直接父版本。错误形如：

```json
{
  "ok": false,
  "code": "E_SPEC",
  "field": "nodes.port",
  "location": "$.nodes[0].port",
  "message": "must be <= 65535"
}
```

运行期稳定错误码包括 `E_PORT`、`E_TIMEOUT`、`E_PROTO`、`E_LEASE`、`E_LEDGER`。套接字连接、握手和读取均设置真实 `socket.settimeout()`；测试用不响应的真实监听端验证 `E_TIMEOUT`，没有用 sleep 伪装套接字超时。

## 确定性聚合规则

种子派生规则为：

```text
words = SHA256("sharddiag-samples-v1\n<node_seed>\n<shard_id>") 的 8 个 big-endian uint32
rng   = np.random.Generator(PCG64(SeedSequence([node_seed & 0xffffffff, *words])))
group_id = rng.integers(0, 8, ..., dtype=int64)
value_q  = rng.integers(-10000, 10001, ..., dtype=int64)
weight   = rng.integers(1, 10, ..., dtype=int64)
```

节点用 `np.bincount` 与 `np.add.at` 做整数 count/sum/weighted sum/min/max；均值使用 Python `divmod` 的 floor 除法。每个 TCP 批次返回可交换的局部分组，父进程合并后重新计算均值。最终摘要绑定 `node_id`、`shard_id`、`sample_rows`、`capacity_version`、`participating_rows`、样本哈希和全部分组整数结果；批次大小只出现在推导依据中，不进入摘要，因此 `1/3/7/16/...` 批次切分和局部回执乱序合并均得到同一 `summary_digest`。

每个分片的完整推导依据可用：

```bash
python3 -m sharddiag.cli explain \
  --spec config/spec.sample.json \
  --shard s01_a_ingest \
  --out /tmp/s01-basis.json
```

首版已生成全量依据：`artifacts/reports/derivations.json`。

## 调度、租约与冻结

调度器只在精确整数检查通过后签发租约并提交 `lease_granted` 批次，然后把工作投入 `ThreadPoolExecutor`。任一时刻：

- 活跃租约资源单位总和不得超过 `global_budget_units`；
- 节点活跃租约数不得超过 `lease_limit`；
- 节点活跃分片数不得超过 `concurrency_quota`。

因预算或配额不能放行的就绪分片进入 `queued`，记录单调时钟毫秒的 `wait_started_monotonic_ms`、`wait_ended_monotonic_ms` 和 `wait_ms`。示例中的 `s00_a_side` 会因 node-a 单并发配额排队。

分片只有在全部上游 `succeeded` 且摘要可用后才就绪。上游失败、取消或冻结时，下游写为 `frozen` 并在 `missing_upstream_ids` 中列出未成功上游；冻结分片不签发租约、不预占资源。取消事件在事件序号折叠阶段成为终态，因此不会下发，也不会在租约过期后重试。同一 Engine 运行轮次中每个分片最多下发一次。

租约字段包括 `lease_id`、`shard_id`、`node_id`、授予/到期单调毫秒、`renew_count`、`lease_version` 和容量代次。续约先把 `expected_version` 发给节点子进程做乐观校验；版本不符返回 `E_LEASE` 和 `basis_generation`，且不延长到期时间。续约次数达到 `max_renewals` 后再次续约同样返回 `E_LEASE`。

## 定点水位与误差归因

会话缩放因子 `K in 0..3`，`S = 10**K`。非负比率统一向下截断：

```text
water_q = floor(S * numerator / denominator)
residual = (S * numerator) mod denominator
```

有符号均值按 toward-negative-infinity 的 floor 除法。乘法或中间值超过 SQLite INT64 范围时饱和到 `2**63-1` 并标记 `saturated=true`。

报告内容：

- `water_report.per_shard_quantization`：逐分片 `water_q`、截断余数和精确误差分数；
- `water_report.per_node_water`：资源、并发、租约水位及两两比较的 `2_q` 量化界；
- `water_report.global_budget_aggregation`：逐分片 floor 聚合的保守上界 `n_q`；
- `water_report.error_attribution`：量化截断、租约归属、依赖等待三段边界、主导段和资源单位阈值。

精确预算判断仍使用原始整数和 `scaled_resource_q = resource_units * S`，误差界只用于标记“数值水位接近、判定受误差影响”。当预算边际或两个候选水位差落入合成误差界时，放行者按同优先级近误差集合中的稳定拓扑秩、再按 `shard_id` 字典序决定，并在 `decisions[*]` 中记录 `decision_affected_by_error=true`。

## 失效传播与容量演进

`sharddiag/invalidation.py` 先找引用变更节点的分片作为直接失效集，再沿 `shard.depends_on` 反向求全部传递后代；输出按 `(topo_rank, shard_id)` 排序，并给出每一步 `direct_node_reference` 或 `dependency_propagation` 依据。

示例从容量版本 1 演进到 2，node-a 口径变化，失效集为：

```text
s00_a_side, s01_a_ingest, s03_a_transform, s05_c_join, s07_sink
```

未失效集为：

```text
s02_b_ingest, s04_b_transform, s06_c_aggregate
```

命令会先用旧版本计算全部记录哈希，再只重算失效子集，最后逐项验证未失效记录哈希不变。容量演进场景还会真实调用三个节点的 `capacity_advance`，整体撤销所有指向旧版本的 active 租约；过期回收和代次回收共用一次 active 状态迁移，因此同一条租约不会被两边重复计数。

`invalidate` 输出同时包含 `verification`（聚合记录）和 `water_verification`（定点水位记录）。租约的不可变授予批次保存在 SQLite 批次链中；状态可以追加/更新为撤销，但 `lease_id/shard_id/node_id` 归属身份摘要在演进前后保持一致。

## SQLite 批次、快照与强杀恢复

SQLite 中每个批次包含整数 `format_version=1`、`payload_digest` 和链式 `chain_digest`。租约和分片状态与批次负载在同一个 SQLite 事务中提交；连接使用 `synchronous=FULL` 和 WAL。部分事务在强杀时由 SQLite 回滚。

恢复规则：

1. 重放并校验连续批次、负载摘要和链头；
2. 发现上次进程遗留的 active 租约或运行态分片时写一条 `recovery_reconcile` 原子批次；
3. 遗留租约改为 `reclaimed_interrupted`，资源和名额先回收；
4. 未形成终态批次的分片标记 `unknown`，新运行重新调度；
5. 已完整提交的 `succeeded` 分片和摘要保留，不重算。

快照是完整 JSON envelope，带整数格式版本和 `content_digest`。重新打开发现版本或摘要不一致时拒绝增量合并；可信快照通过 `replace_sqlite_with_snapshot()` 删除旧 DB/WAL/SHM 后整体重建，不可信或被篡改的快照文件由 `replace_snapshot_if_invalid()` 用新的完整状态原子替换。

强杀演示：

```bash
python3 -m sharddiag.cli run ... --fresh --kill-after-terminal 1
# Shell reports 137; no normal CLI report/snapshot is generated by the killed process.
python3 -m sharddiag.cli run ...   # same --db, without --fresh
```

本次实际恢复批次为：

```json
{
  "incarnation": 1,
  "unknown_shard_ids": ["s01_a_ingest"],
  "reclaimed_lease_ids": ["lease-s01_a_ingest-gen1"]
}
```

恢复后的 canonical ledger digest 与干净运行逐位相同：

```text
6ad0b96443c9d8edbec54ff207c216be79d5e6a8ada93d8b962ee6c3167b2201
```

## 实际验证结果

验收机器信息和汇总见 `artifacts/reports/acceptance-summary.json`，完整测试输出见 `artifacts/reports/test-results.txt`。最近一次结果：

- Python `3.11.0rc1`
- NumPy `1.26.4`
- sqlite3 `3.37.2`
- Tk `ok 8.6.12`
- `Ran 24 tests in 9.350s`，结果 `OK`
- 3 个声明端口运行前均空闲，真实节点运行后由测试和 CLI 清理
- 主样本 8/8 分片成功，`s01_a_ingest` 完成两次续约，最终 `lease_version=3`
- 取消场景中 `s04_b_transform=cancelled`，`s06_c_aggregate`、`s07_sink=frozen`
- 失效校验 `ok=true`，无意外变更项
- 演进场景初始占用 13 单位，代次回收后 active 占用为 0，旧代次续约返回 `E_LEASE`
- SIGKILL 后重启成功，账本摘要与干净运行一致

## 环境备注

标准安装脚本是 `scripts/install_ubuntu_deps.sh`。本次 WSL 会话的 `sudo` 需要密码，因此没有声称通过 apt 新安装软件包；当前 Python/NumPy/sqlite3 已可用，Tk 在运行时提供的 `/tmp/tkdeps2` shim 下可导入并可创建 `Tk()`。在普通 root 可用的 Ubuntu WSL 中，可直接运行安装脚本或按其中命令安装系统 Tk 包。
