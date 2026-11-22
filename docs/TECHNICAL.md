# DriftWatch 技术说明

## 1. 处理流水线

1. `config.py` 解析并整体校验快照链、代理清单和判据，生成不可变领域对象与内容摘要。
2. `ledger.py` 计算声明摘要、快照链版本、代理集合版本和缩放口径版本，与当前活动代比对。相同则续用；不同则把旧代置为非活动，按传播规则标记受影响旧结论为 `stale`，创建空的新活动代并要求完整重算。
3. `processes.py` 预检所有 `127.0.0.1` 端口，以 `sys.executable -m driftwatch.agent_main` 启动子进程。Windows 使用 `CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP`；代理监控父进程，父进程被强杀后自行退出。
4. 每个快照先建立 `RUNNING` 批次，再构造“代理 × 主机 × 指标”任务。
5. `scheduler.py` 在全局并发与每代理并发双重约束下调度，每个任务独立建立套接字连接、握手并读取一行 JSON 响应。
6. `merge.py` 按稳定顺序选权威值，保留全部来源；`fixedpoint.py` 产出漂移、误差界和决胜结果。
7. `ledger.py` 在同一 SQLite 事务内写尝试、原始观测、归并、漂移、游标、批次摘要和会话视图摘要。事务失败或进程死亡时不会留下半批结论。
8. 恢复时只采纳 `COMMITTED`；对遗留 `RUNNING` 批次执行 `ABORTED` 并登记 `recovery_unknown`。随后可重放已提交批次并重采未知/未提交快照。

## 2. 快照与代理校验

快照检查包括：根对象、快照数组非空、`seq` 唯一非负整数、`parent_seq` 为 `null` 或存在整数、父图无环、三指标齐全、期望值为正整数、主机不重复、容差为 `0..10000` 整数。所有问题汇总后以一个 `E_SNAPSHOT` 抛出。

代理检查包括：数组非空、`agent_id` 唯一、`host` 必须等于 `"127.0.0.1"`、端口为 `1024..65535` 且全局唯一、种子为整数、聚合种类/窗口/版本合法、主机与指标属于声明快照、profile 的基值/离散度合法、每代理并发为正整数。所有问题汇总后以一个 `E_AGENT` 抛出。

版本摘要均使用排序键、紧凑分隔符的规范 JSON 的 SHA-256：

- `snapshot_chain_version = H(snapshot_chain)`
- `agent_set_version = H(sorted(agent_specs))`
- `scale_version = H(scale_exp, rounding, boundary_rule)`
- `declaration_digest = H(snapshot_chain, sorted(agent_specs), criteria)`

## 3. JSON 行协议

每个连接先发送 `hello`：

```json
{"op":"hello","protocol":1,"agent_id":"agent-alpha","client":"driftwatch"}
```

代理回复 `ok`、`agent_id`、`protocol`、`pid`。之后发送：

```json
{"op":"observe","request_id":"snapshot-2/edge-a/cpu_quota","snapshot_seq":2,"host_id":"edge-a","metric_id":"cpu_quota","observation_seq":201"}
```

响应中的 `record` 必须包含并严格校验七字段。协议读取使用累计截止时间，每次 `recv` 前按剩余时间设置 `socket.settimeout`；超时不会转为空值、零值或伪观测。

端口预检使用真实 `bind`。Windows 监听套接字使用 `SO_EXCLUSIVEADDRUSE`；POSIX 使用 `SO_REUSEADDR` 排除 `TIME_WAIT` 对重复演示的干扰。代理进程退出早于合法握手映射为 `E_PORT`，进程存活但握手期限耗尽映射为 `E_TIMEOUT`，响应结构错误映射为 `E_RECORD`。

## 4. 确定性代理

代理使用纯整数 SplitMix64 和 BLAKE2b 密钥派生，密钥包含：

```text
seed, host_id, metric_id, snapshot_seq,
aggregation.version, aggregation.kind, aggregation.window
```

同一快照、主机和指标在重试、重新进程和重放时产生相同样本。观测序号由协调方按快照与指标显式下发，重试不改变请求语义。代理在本地窗口内生成整数样本，并按声明的 `median`、`mean_floor`、`trimmed_mean`、`minimum` 或 `maximum` 聚合。聚合版本进入密钥，因此口径演进会产生新的观测摘要。

## 5. 权威归并与稀疏规则

对同一 `(snapshot_seq, host_id, metric_id)`：

1. 按 `(-observation_seq, agent_id)` 排序。
2. 同一代理只保留排序最前的一条，来源计数按代理去重。
3. 有效代理数 `< min_sources`：`status=SPARSE`、`error_code=E_SPARSE`、权威值和漂移判定为空。
4. 否则首条为权威，其余记录保留 `value - authority_value`。

这意味着若两个代理序号相同，代理标识字典序较小者胜出；该规则与线程完成顺序无关。

## 6. 定点漂移与误差公式

记：

- `S = 10^scale_exp`
- `E = expected`，`O = observed`
- `B = tolerance_bps`
- `D = max(abs(source - authority))`
- `sat(x)` 表示截断到 `[-2^63+1, 2^63-1]`

计算：

```text
expected_fixed = sat(E * S)
observed_fixed = sat(O * S)
drift_signed_fixed = sat((O - E) * S)
excess_fixed = max(0, drift_signed_fixed)
allowed_fixed = floor(E * S * B / 10000)
merge_error = sat(D * S)
observation_quantization_error = S
tolerance_comparison_error = 1
composite = sat(S + 1 + merge_error)
drift_ratio_bps = trunc_toward_zero(max(0, (O-E)*S) * 10000 / (E*S))
```

规则：

- 所有非负额度除法使用 `floor`。
- 有符号报告比例使用向零截断。
- 漂移差使用未饱和的有理大整数先计算，再饱和，避免两个已饱和固定值相减错误地抵消。
- 误差界单位为 fixed 最小单位。
- `excess > allowed + composite`：`REJECT`，不受边界误差影响。
- `excess + composite <= allowed`：`ACCEPT`，不受边界误差影响。
- 其余区间：`error_affected=true`。`tolerance_floor` 下若 `excess <= allowed` 则 `ACCEPT`，否则 `REJECT`；`conservative_reject` 一律 `REJECT`。

观测量化误差取 `S` 的理由是代理摘要为整数向下聚合，真实窗口聚合值可能位于报告整数之上不足一个原始单位；在 fixed 最小单位中上界为 `S`。容差额度向下取整最多损失不到 1 个 fixed 单位。归并偏离误差使用全部有效来源相对权威值的最大绝对原始偏离乘以 `S`。

## 7. 传播与整体替换

影响传播比较旧新声明：

- 代理增删：其旧或新的主机/指标覆盖。
- 同代理聚合版本变化：新覆盖范围。
- 覆盖变化、种子/端口/profile/配额变化：相应覆盖范围。
- 新增快照：该快照全部主机指标。
- 既有快照父引用、主机集合、期望或容差变化：对应项。
- 缩放版本、决胜规则或 `min_sources` 变化：所有当前主机指标。

结果以 `(snapshot_seq, host_id, metric_id, sorted(reasons))` 稳定排序并持久化。旧活动代的所有结论不再进入当前统计；受影响行额外标记 `stale=1`。新活动代从空视图开始，逐快照采集后完整形成新视图。

声明差异另存为稳定有序的 `generation_diffs`。若声明版本相同但持久化会话视图摘要/结论摘要与其内容不一致，也创建新活动代，并记录 `session_snapshot_mismatch` 差异。

## 8. 并发调度

调度器使用条件变量、待执行队列和实际运行计数，而不是简单地把所有任务塞进固定线程池。其选择规则是：

- 活跃总数 `< global_concurrency`。
- 该代理活跃数 `< min(criteria.per_agent_concurrency, agent.per_agent_concurrency)`。
- 重试到达 `retry_backoff_ms` 后才可调度。

每次尝试记录：

```text
wait_start_ms, wait_end_ms, run_start_ms, run_end_ms,
attempt_no, is_retry, status, error_code
```

所有时间来自 `time.monotonic_ns() // 1_000_000`。首次失败和重试是两行独立尝试记录；同一个任务的 `snapshot_seq/host/metric/observation_seq` 不变。

## 9. SQLite 账本

关键表：

- `ledger_info`：整数 `format_version`、已提交内容摘要。
- `generations`：活动代、四个声明版本/摘要、完整声明 JSON。
- `affected_items`、`generation_diffs`：受影响项及稳定差异依据。
- `session_snapshots`：版本、完整归并视图/漂移视图及摘要。
- `batches`：`RUNNING`/`COMMITTED`/`ABORTED`、计划项、摘要和游标。
- `observations`、`attempts`：来源观测和全部尝试。
- `merged_rows`、`drift_rows`：权威结论、误差界、`stale`。
- `cursors`：代内最后提交快照。
- `recovery_unknown`：未提交采集中的主机指标和是否已重采。

批次流程：

1. 独立事务写入 `RUNNING` 与计划项。
2. 无数据库写入地完成采集和计算。
3. 单事务插入全部结果、更新 `COMMITTED`、游标、会话视图摘要、未决重采状态和账本内容摘要。

因此强杀只可能留下第 1 步的 `RUNNING` 标记，不可能留下部分观测、部分权威值或部分漂移结论。SQLite 自身保证第 3 步全有或全无。

## 10. 恢复与重放

重启时：

1. 校验格式版本和已提交批次内容摘要。
2. 依据当前声明决定续用或整体换代码。
3. 将遗留 `RUNNING` 批次置为 `ABORTED`，按 `planned_json` 登记 `recovery_unknown`。
4. 对活动代已提交批次重新采集、重新归并、重新判定，逐快照、主机、指标和字段比较。首个差异抛 `E_REPLAY`。
5. 重采所有未提交/未知项并原子提交。

`recover_ledger(db)`/`driftwatch recover` 使用账本内保存的完整声明执行上述流程；`replay_ledger(db)`/`driftwatch replay` 只重放已提交批次，不重采。

重放排除 `monotonic_ms`；其余字段包括来源值、来源偏离、来源数量、权威代理/序号/值、聚合版本、漂移状态/判定、定点量、误差界、误差影响、饱和标志和原因均参与比较。

强杀不产生退出报告。`examples/crash_recovery_demo.py` 只报告账本中可证明的批次状态、恢复项和重放结果。
