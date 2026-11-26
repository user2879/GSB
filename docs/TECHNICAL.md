# 技术说明与验收口径

## 1. 运行拓扑

```text
JSON spec
   │ load_spec / parse_spec（E_SPEC）
   ▼
Scheduler ── ThreadPoolExecutor ── TCP JSON Lines ── Node subprocess × N
   │                                  └─ NumPy 分组聚合 / 定点 mean_q
   ├─ LeaseManager：版本、monotonic 过期、release-once
   ├─ Fixed-point water / boundary adjudication
   └─ Ledger（SQLite 单事务：outcome + lease + occupancy + water + cursor + digest）

evolve：旧 DB → 仅失效子集重算 → 与空白全量重建逐字段比较（E_DIVERGE）
verify：只读 SQLite + 原始 JSON spec → 重新生成样本并重推结局（E_VERIFY）
```

节点命令为 `python -m gsb.node --spec <spec> --node-id <id>`。父进程使用 `sys.executable -m gsb.node` 启动独立进程，并为每个节点建立新的进程组，便于正常停止时整组回收。

## 2. 规格与稳定顺序

`gsb/spec.py` 对根对象、节点、分片、依赖、预算和调度参数执行首个错误即返回的强校验。错误对象包含：

```json
{"ok": false, "code": "E_SPEC", "field": "nodes[1].port", "position": "nodes[1].port"}
```

稳定拓扑使用 Kahn 算法：就绪集合按 `(-priority, shard_id)` 排序，即优先级数值越大越先进入拓扑；同优先级按分片标识升序。所有边界决胜均使用 `(topo_index, shard_id)`，因此不依赖 dict/set 迭代顺序。

支持的缩放因子为：

- 二进制幂：`2^0 ... 2^20`
- 十进制幂：`10^0 ... 10^9`

样本使用十进制会话缩放因子 `10`，节点自身的 `fixed_scale` 可以不同，聚合均值按节点口径做整数除法。

## 3. JSON Lines 协议

每条消息是一行 UTF-8 JSON，以 `\n` 结束。主要请求：

- `hello`：握手并确认节点标识。
- `lease_grant` / `lease_renew`：租约签发与乐观版本续约。
- `aggregate`：携带 `shard_id`、`capacity_version`、`fixed_scale`、确定性 `samples` 和租约引用。

成功回执包含 `node_id`、`shard_id`、`capacity_version`、`summary`、`summary_digest` 与 `node_elapsed_ms`。`node_elapsed_ms` 来自 `time.perf_counter_ns()` 差值并转为整数毫秒；调度等待和租约过期使用 `time.monotonic_ns()`。

`gsb/protocol.py` 为建连和读取设置真实 socket timeout。对端接受连接但不回行时，测试在 80ms 内收到 `E_TIMEOUT`；端口被监听 socket 占用时，预检查和子进程绑定路径均归类为 `E_PORT`。

## 4. 租约与三重调度约束

租约字段与需求一一对应：`lease_id`、`shard_id`、`node_id`、`capacity_version`、`granted_monotonic_ms`、`expires_monotonic_ms`、`lease_version`、状态和一次性释放标记。

续约流程：

1. 父进程检查本地当前版本。
2. 通过真实 TCP 向节点发送 `expected_version`。
3. 节点用自己的租约表做乐观比较；不符则返回 `E_LEASE`，不增加版本也不延长 `expires_monotonic_ms`。
4. 节点匹配后版本加一，父进程才延长到期时刻并提交新版本。

过期由真实睡眠时间和单调时钟判定；测试使用 40ms TTL 后轮询 `expire_due()`，不使用模拟时间快进。

调度准入在单一线程的调度循环中完成，因此检查与占用递增之间没有并发插入。任一准入必须同时满足：

```text
active_units + shard.resource_units <= budget_units
active_leases_for_node < lease_limit
active_shards_for_node < concurrency_quota
```

通过后才创建租约、写 lease grant 记录、经 TCP 通知节点并提交到 `ThreadPoolExecutor`。无法准入时记录 `wait_started_ms`，准入时写 `wait_ended_ms` 和原因（`global_budget`、`node_lease_limit`、`node_concurrency_quota`、`upstream_dependency_not_committed`）。

硬预算安全始终使用原始整数单位；定点水位用于统一口径、误差归因和“不可区分”标注，不会把误差误当成可以突破硬预算的依据。

## 5. 定点水位与误差界

设会话缩放因子为 `S`。

### 5.1 整数资源/上限/已用量

资源单位、租约上限、并发上限和已用量首先统一编码为：

```text
Q(x) = sat_int64(x * S)
```

因为 `x` 是整数，这一段没有截断误差。所有乘法/除法均由 Python 任意精度整数完成，最后执行 int64 饱和：

```text
q >  2^63-1 =>  2^63-1, saturated=true
q < -(2^63-1) => -(2^63-1), saturated=true
```

非负水位一律向下/floor 截断；有符号余量使用 Python `//`，即向负无穷 floor，保持保守。

### 5.2 归一化边界水位

为了让缩放因子变化真实影响边界判断，预算比例水位定义为：

```text
W(x; B, S) = floor(S * x / B)
```

其中 `B` 是全局预算。单个分片的截断误差满足：

```text
0 <= S*x/B - W(x;B,S) < 1 q-unit
```

`single_shard_error_attribution()` 还会用 `Fraction` 返回该样本的实际余数，而不只是最坏上界。

### 5.3 三段合成误差

对 `n` 个独立量化项，采用保守且可加的三段拆分：

| 段落 | q-unit 上界 | 含义 |
| --- | ---: | --- |
| 量化截断 | `1` | 首个量化项的一次 floor |
| 逐节点累加 | `max(0, n-1)` | 其余项在节点水位累加中带入的截断 |
| 阈值比较 | `1` | 最后与预算/阈值比较的一次舍入不确定性 |
| 合计 | `n+1` | 合成误差界 |

若某节点已有 `k` 个在途项并加入 1 个候选，则 `n=k+1`，节点判定界为 `k+2` q-unit。全局有 `N` 个分片项时，全局预算聚合界为 `N+1` q-unit；换算回资源单位：

```text
resource_bound = (N+1) * B / S
```

这是保守上界；核心整数账本中的资源求和本身是精确的。

### 5.4 本地样本中的边界翻转

样本场景：预算 `B=40`，已用 29，候选需求 12；原始整数结论是“超出 1 单位”。代码实际输出：

| S | remaining water | demand water | Δq | 合成界 | 是否标注受误差影响 | 量化结论 |
| ---: | ---: | ---: | ---: | ---: | --- | --- |
| 10 | 2 | 3 | -1 | 4 | 是 | reject/ambiguous |
| 100 | 27 | 30 | -3 | 4 | 是 | reject/ambiguous |
| 1000 | 275 | 300 | -25 | 4 | 否 | reject/clear |

因此缩放因子变粗时，边界附近多个候选会同时落入“不可区分”区间；显式决胜仍按 `(topo_index, shard_id)` 选择并设置 `error_affected=true`。候选水位差同理：`B=100, S=10` 时需求 16 和 17 的水位都为 `1`，差值 0 落入误差界，稳定序中的 `a` 胜出且被标注。

## 6. 陈旧结论防护

`gsb/snapshot.py` 的 `SessionSnapshot` 带整数 `snapshot_format=1` 和内容摘要。`ViewStore.replace(new_snapshot)` 整体替换当前对象，并把所有旧 generation 标记为 `stale=true`；展示接口只返回当前快照，不逐条修补旧水位、轨迹或结论。

容量版本撤销与到期回收共用 `LeaseManager.release_once(lease_id, reason)`。只有第一个调用得到非空释放凭证，后续撤销/到期/恢复路径得到 `None`，因此同一份容量权益不会重复入账。测试覆盖“先撤销后到期/释放”的竞争等价情形。

## 7. 失效传播与增量一致性

`derive_invalidation(old, new)` 检查：

- 新增/删除分片；
- `dependencies`、`rows`、`resource_units`、`priority`、`node_id`、`group_count`；
- 全局 `scale_factor`；
- 节点 `seed`、`capacity_version`、`fixed_scale`、`group_count`；
- 显式或稳定哈希放置推导出的节点引用变化。

直接失效后，沿反向依赖图用 `all_descendants()` 找全部间接失效分片，并按新稳定拓扑序输出步骤，例如：

```text
s3 direct:changed:node.capacity_version@n3
s5 propagated:upstream_invalid:s3
s6 direct:changed:node.capacity_version@n3
```

增量路径只把 invalid 集合交给 Scheduler；其他 ledger outcome 被视为已满足依赖且不会重写。测试在增量运行前后读取未失效分片的规范化账本记录字节，断言完全相同。随后与空白 DB 全量真实 TCP 重建比较：

- `summary` / `summary_digest`
- `water_q`
- `node_id`
- `capacity_version`
- lease attribution（节点与容量版本）
- `propagation_basis`

首个差异立即以 `E_DIVERGE` 报告分片和字段。测试还故意篡改一个 `water_q` 验证该错误路径。

## 8. SQLite 原子提交与崩溃恢复

账本 PRAGMA：

```text
ledger_format_version = 1（ledger_meta 整数 + PRAGMA user_version）
synchronous = FULL
journal_mode = DELETE
```

租约签发先在独立事务写 `lease_records(status=granted)` 和 `occupancy_events(delta=+units)`。分片完成时，`commit_outcome()` 在同一个 `BEGIN IMMEDIATE` 事务中写：

- `outcomes`：聚合 JSON/摘要、租约、节点、容量版本、资源、`water_q`、缩放因子、等待/执行毫秒；
- `lease_records` 的完成状态和版本；
- `occupancy_events(delta=-units, reason=completed)`；
- 下一个进度游标；
- 全账本内容摘要。

因此 `SIGKILL` 不可能留下“outcome 已写一半但游标/水位未写”的完整记录。

崩溃恢复事务把所有 `granted/renewed` 租约改为 `reclaimed`，追加负向占用事件，并把没有 outcome 的 spec 分片列为 `unknown_shards`。重启后的调度只重跑 unknown 分片；已提交 outcome 跳过，依赖仍视为满足。

集成测试 `tests/test_recovery.py` 会：

1. 启动真实 `python -m gsb run ...` 子进程；
2. 等到一个快分片已提交、慢分片存在 active lease；
3. 对父进程组发送 `SIGKILL`，并按 PID 文件清理节点进程组；
4. 执行真实 CLI `recover`，断言回收慢分片的 3 个单位且将其列为 unknown；
5. 再次运行并执行只读 `verify`，断言两个分片重放一致。

独立校验使用 `file:...?mode=ro` SQLite URI，不写账本；它只读取账本和原始 spec，重新生成固定种子样本、重新聚合和计算水位，再定位第一个 `E_VERIFY` 字段差异。

## 9. 实际验证

最近一次执行：

```text
Python 3.11.0rc1
numpy 2.4.6（.venv）
Ran 25 tests in ~6.6s
OK
```

覆盖：

- `E_SPEC` 重复 ID/端口、host 字面量、缺失依赖、环、正整数和缩放因子；
- `E_PORT` 与 `E_TIMEOUT`；
- 真实 TCP 节点握手、聚合和乐观续约；
- 单调时钟真实过期与 release-once；
- 全局预算、节点租约上限、节点并发配额排队；
- 定点误差归因和缩放因子边界翻转；
- 快照格式版本、整体替换和旧视图陈旧；
- 容量版本失效传播、未失效记录字节不变、增量/全量一致和 `E_DIVERGE` 探针；
- 真实 `SIGKILL`、SQLite 回收和只读重放验证。

完整命令输出保存在 `artifacts/validation-report.txt`，机器可读产物在 `artifacts/acceptance-latest/`。

## 10. WSL/Ubuntu 环境备注

本次基础环境的 `python3 -m venv` 因缺少 `ensurepip` 失败，已用可用的 `virtualenv .venv` 创建环境并成功安装 NumPy。全新 Ubuntu/WSL 可执行：

```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip python3-tk
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

当前最小运行镜像能导入纯 Python 包，但 Tkinter 原生库缺失：

```text
ImportError: libtk8.6.so: cannot open shared object file
```

核心调度、TCP、NumPy、SQLite 和 CLI 不依赖 Tkinter；安装 `python3-tk` 后可使用 `python -m gsb gui --snapshot ...`。
