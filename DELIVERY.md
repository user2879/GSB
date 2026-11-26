# WaterLease 交付说明

本地多节点分片计算的租约水位、依赖失效、SQLite 账本与崩溃恢复演示。所有网络通信均为 `127.0.0.1` 回环 TCP + JSON Lines，样本由固定整数种子在本机生成。

## 快速运行

```bash
# 若 .venv 尚未创建
bash scripts/setup_venv.sh

source .venv/bin/activate
pytest -q
python -m waterlease.demo
```

也可一键验证：

```bash
bash scripts/verify.sh
```

常用 CLI：

```bash
.venv/bin/python -m waterlease.cli --spec-dir spec validate
.venv/bin/python -m waterlease.cli --spec-dir spec --ledger ledger/demo.sqlite3 run --recover
.venv/bin/python -m waterlease.cli --spec-dir artifacts/spec-v2 --ledger ledger/demo.sqlite3 replay
```

演示会生成：

- `spec/`：首版节点、分片、容量链和事件声明。
- `artifacts/spec-v2/`：运行时生成的配额/容量/分片演进声明。
- `artifacts/spec-crash/`：强制终止演示使用的慢节点声明。
- `ledger/demo.sqlite3`：增量演进账本。
- `ledger/full-rebuild-v2.sqlite3`：空白全量重建账本。
- `ledger/crash-recovery.sqlite3`：SIGKILL 恢复账本。
- `ledger/snapshot_backups/`：只读旧快照备份与差异列表。
- `artifacts/demo-report.json`：完整结构化验收报告。
- `artifacts/verification/`：最近一次 pytest、demo、replay 输出。

## 代码结构

- `waterlease/spec.py`：JSON 加载、强类型化与 `E_SPEC` 首错定位。
- `waterlease/topology.py`：稳定 Kahn 拓扑序、环路径提取。
- `waterlease/sampling.py`：整数种子、NumPy 分组聚合、批次 monoid 合并。
- `waterlease/protocol.py` / `node_worker.py`：JSON Lines 握手、RPC、节点租约乐观校验。
- `waterlease/process_manager.py`：独立子进程启动、端口/超时错误映射与回收。
- `waterlease/leases.py`：签发、续约、释放、`time.monotonic_ns()` 真实过期。
- `waterlease/water.py`：10 的幂定点水位、误差界和三段归因。
- `waterlease/invalidation.py`：节点/容量/依赖失效传播与依据。
- `waterlease/ledger.py`：SQLite 原子事务、快照替换、备份、恢复和重放。
- `waterlease/scheduler.py`：依赖就绪、三重资源闸门、并行调度和提交。
- `waterlease/demo.py`：完整验收流程。
- `tests/`：21 个聚焦测试，包含真实子进程、真实 TCP、真实超时和 SIGKILL 恢复。

## 关键语义

### 确定性样本与聚合

样本种子由 `SHA256("waterlease-v1:{node_seed}:{shard_id}")` 派生为固定 31-bit 整数，再交给 `numpy.random.default_rng`。聚合只使用 `int64` 计数、求和、最小值、最大值和 floor 均值。批次结果是可交换 monoid，因此批次大小、完成顺序和并发度不改变最终摘要。

摘要字段包括 `node_id`、`shard_id`、`capacity_version`、`sample_rows`、`participated_rows`、定点口径、分组结果、上游摘要和 `aggregate_digest`。

### 租约

租约包含：`lease_id`、`shard_id`、`node_id`、授予/到期单调时钟毫秒、`renew_count`、整数 `version` 和状态。续约把当前版本发送给节点子进程做 CAS 校验；节点版本不符返回 `E_LEASE`，本地不延长到期时间。到期判定直接使用 `time.monotonic_ns()` 与真实 sleep，不使用人工快进时钟。

### 调度约束

调度器在派发前同时检查：

1. `active_global_units + shard.resource_units <= session.global_resource_budget`；
2. 节点在途分片数 `< concurrency_quota`；
3. 节点活跃租约数 `< max_leases`。

不满足时分片留在稳定拓扑队列，并记录 `wait_started_monotonic_ms`、`wait_ended_monotonic_ms` 和 `wait_duration_ms`。上游必须已有当前提交摘要；`state="frozen"` 的分片只记录 `frozen_skip`，不签发租约、不写预留、不预占预算。

### 定点水位与误差

会话 `scale_exp` 仅允许 0、1、2、3，`scale = 10 ** scale_exp`。整数量化为 `q = raw * scale`；非负比率量化为：

```text
q = floor(numerator * scale / denominator)
remainder = numerator * scale - q * denominator
误差 = remainder / (denominator * scale)
```

截断方向是非负数 floor（等价向零截断）。q 值超过 signed int64 上限时饱和到 `2^63 - 1`，不回绕。

输出包括：

- 每分片资源 q、预算份额 q 和精确截断误差分数；
- 每节点直接水位比较误差界 `< 1/scale`，以及按节点分片数求和的保守界 `< shard_count/scale`；
- 全局预算聚合误差上界 `< shard_count/scale`，固定 q 单位上界为 `shard_count`；
- `capacity_version_chain`、`lease_generation`、`schedule_queue` 三段归因单位、basis point 占比和主导段。

主导阈值为严格大于 `5000 bp`（50%）；未超过时 `dominant_segment=null`。

### 失效与重建

`plan_invalidation(v1, v2)` 比较分片、节点、容量版本和会话水位参数。节点变更引用该节点全部分片，容量变更引用对应版本分片，分片规格变更标记自身；随后沿反向依赖边按稳定拓扑序传播。每一步均带 `basis`、字段路径、来源对象和边。

未进入失效集合的分片不会重新调度，当前 SQLite 行不更新，因此导出的 payload 字节保持一致。空白库按最终 v2 声明全量运行后，使用 `compare_deterministic_exports` 递归比较业务结论；首个差异以 `E_DIVERGE` 和 JSON 指针路径报告。

真实单调时间、RPC 耗时、随机 reservation id、运行期峰值等字段不参与跨进程确定性相等比较；它们仍逐笔写入账本。比较投影保留摘要、租约身份/版本、资源定点口径、水位、误差归因和依赖摘要，并把时间字段规范为非负不变量。

### SQLite 与崩溃恢复

每个分片派发前先提交 active reservation。结局提交在同一 SQLite 事务写入：

- outcome/current outcome；
- lease record；
- resource record；
- water record；
- error attribution；
- shard status；
- progress cursor；
- append-only commit。

每行携带整数 `format_version=1` 与 `content_digest`。重新打开时若 `format_version` 或 `spec_digest` 不一致，会把旧快照导出到 `ledger/snapshot_backups/` 下权限 `0444` 的 JSON 备份，同时写 `.diff.json` 和 `snapshot_replacements`，然后整体替换 singleton 快照。

SIGKILL 后，已完整提交的 outcome 可重放；存在但没有结局事务的 active reservation 会被标记为 `reclaimed`，对应分片状态标为 `unknown`，并先汇总回收资源单位。系统不会伪造 kill 时刻退出报告。

## 错误码

- `E_SPEC`：JSON/声明不合法，错误含字段路径与首条违规。
- `E_PORT`：回环端口绑定失败或被占用。
- `E_TIMEOUT`：真实 TCP 建连、握手或读取超时。
- `E_LEASE`：租约版本冲突、状态错误或真实过期。
- `E_DIVERGE`：增量账本与空白全量重建的确定性字段不一致。
- `E_CAPACITY`：资源/配额不变量或不可调度声明。
