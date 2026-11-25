# 技术说明

## 1. 数据契约

样本由 `qos_diag/samplegen.py` 使用 `numpy.random.default_rng(287)` 确定性生成：

- `samples/base/pipelines.json`：3 条管道，每条固定 `ingress -> relay -> egress`。
- `samples/base/shapers.json`：9 个整形器，端口从 `43011` 起全局唯一，主机必须是字符串字面量 `127.0.0.1`。
- `samples/base/events.json`：36 个全局到达毫秒单调不减的事件，按序号轮转到 3 条管道。
- `samples/base/config.json`：定点 scale、窗口/滑动步长、并发配额、真实超时和批次大小。
- `samples/edited_param`、`edited_burst`、`edited_arrival`：三类可重复编辑样本。

校验顺序固定为 pipelines、shapers、events；每个数组按文档顺序、对象字段按声明顺序检查。第一个错误抛出 `E_SPEC`，JSON 含 `object_id` 与 `field`。重复 ID/端口在遇到第二个对象时报错。

## 2. 三级整数定点桶

配置 `scale=s`，其中 `s ∈ {0,1,2,3}`，令 `Q=10^s`。速率、包长、容量和余量均转成 q-unit 整数：

```text
rate_q     = refill_rate_per_ms * Q
required_q = packet_length * Q
cap_q      = burst_capacity_tokens * Q
```

某一级上一状态为 `(balance_q, state_clock_ms)`，当前事件到达 `t`：

```text
elapsed_ms       = t - state_clock_ms
scheduled_q      = elapsed_ms * rate_per_ms * Q
available_q      = min(cap_q, balance_q + scheduled_q)
allow            = available_q >= required_q
after_q          = available_q - required_q  if allow else available_q
state_clock_ms   = t
```

- 非负量化/除法采用 floor，即向零截断。
- 乘法/加法超过有符号 64 位上限 `2^63-1` 时饱和到 `MAX_FIXED_Q`，并在记录中置 `overflow_saturated=true`；桶容量夹取优先表达业务突发上限。
- 入口 shape 时中继/出口标记 `evaluated=false`、`verdict="shape"`、`decision="skipped_upstream"`，其桶状态时钟不推进；下一次真正到达该级的事件会按完整间隔补桶。
- 每条记录输出判定前余量、补充分量、夹取后判定前余量、判定后余量、所需令牌、状态时钟和规则依据。

由于输入整数乘 `Q` 后仍为整数，核心余量没有实际量化误差；统计比率和均值的 floor 余数保存在账本/诊断中。

## 3. 逐事件路径与滑窗批量路径

### 逐事件路径

`ChainEngine.evaluate_event` 按事件顺序穿过三级链，状态保存在：

```text
(pipeline_id, stage) -> StageState(balance_q, state_clock_ms)
```

任意分批只是多次调用同一个 fold。测试覆盖 batch size `1,2,3,5,7,12,100`，canonical JSON bytes 与整流结果完全相同。

### 滑窗路径

主窗口桶为：

```text
primary_window_start = floor(arrival_ms / slide_ms) * slide_ms
```

窗口长度 `window_ms` 可以大于 `slide_ms`，从而形成重叠窗口。每个事件还计算所有包含它的网格窗口起点，供到达时间编辑时推导重叠失效。

窗口路径按管道、级别和稳定事件序，将连续的、同到达毫秒且同包长的事件聚成 run；一组只计算一次补充和可用余量，再用：

```text
allowed_count = min(group_size, available_q // required_q)
```

批量消耗 `allowed_count * required_q`，并按原始事件顺序物化每个事件的判定前后余量。该结果与逐事件路径 canonical bytes 相同；`diagnose()` 会强制断言 parity。

### 默认路径选择

令：

- `P=pipeline_count`
- `E=event_count`
- `W=active_window_count`
- `G=sum(max(events_per_pipeline-1,0))`
- `avg_gap_ms = sum(last_arrival-first_arrival per pipeline) / G`

显式工作量估计：

```text
C_event  = 3E + P
C_window = E + 3W + P
advantage = C_event - C_window
deadband  = max(config.selection_deadband_work, P, floor(E/20))
```

仅当 `advantage > deadband` 时选择 `window`；否则稳定决胜选择 `event`。输出包含 `P/E/G/avg_gap_ms_q/W/events_per_window_q`、两个成本、deadband、切换边界 `advantage >= deadband+1` 以及未切换时距边界的工作量。

## 4. 失效传播与增量重算

`derive_invalidation(old,new)` 生成稳定闭包：

- 管道某级 shaper 映射变化：该级及 `relay/egress` 下游级全部事件失效。
- 某 shaper 的速率/突发/host/port 变化：按 shaper 反查 `(pipeline, stage)`，同样使该级和下游级全事件失效。
- 事件 `arrival_ms` 或 `packet_length` 变化：该管道中从该事件起的时间/状态后缀在三级全部失效。
- 事件 ID/顺序变化或 scale/window/slide 变化：全量失效。
- 到达时间变化还加入旧/新时间点所属的全部重叠窗口。
- 对每个失效 `(pipeline, stage)`，从首个失效事件开始的所有后续活跃主窗口都进入 `affected_stage_windows`，表达桶状态的时间传播。

输出按 `pipeline_id`、`arrival_ms/event_seq`、stage rank 排序，包含：

- `changes`
- `propagation_steps`
- `restart_local_index`
- `affected_pipelines`
- `affected_event_seqs`
- `affected_events`
- `affected_windows`
- `affected_stage_windows`
- `affected_cells`

`recompute_incremental` 从首个失效事件前一条未失效记录恢复每级 `StageState`，只重算 cutoff 之后的失效级别；未受影响 stage record 直接深拷贝复用。随后：

1. `verify_unaffected_bytes` 逐 cell 比较 canonical bytes；
2. 再与新 spec 的全量 `diagnose()` 比较 canonical bytes。

演练样本中，p-02 relay 速率变化只失效 12 个事件 × relay/egress 共 24 cells，84 个无关 cells 逐字节不变；到达编辑和出口突发编辑也全部通过相同检查。

## 5. 定点统计与误差界

逐级统计包括 judged/allowed/shaped/skipped、shape ratio、最小/最大/均值/最终余量。比率：

```text
shape_ratio_q = floor(Q * shaped / judged)
```

截断余数单独保存，因此比率的真实误差小于 `1/Q`。

误差上界输出在 `diagnosis.error_bounds`：

- 单事件量化：理论 `<1 q-unit`；整数样本实际为 `0`。
- 窗口聚合：每个被量化/判定项至多贡献 1 q-unit，窗口边界给出 `3*event_count+3` 的保守上界，并按级列出 `evaluated_count+1`。
- 三级链：同一窗口三级边界之和为 `three_stage_bound_q`；全局保守上界为 `evaluated_stage_records + 3*window_count`。

差异分类函数 `classify_difference(delta, tolerance, uncertainty)`：

```text
lower = |delta| - uncertainty
upper = |delta| + uncertainty
lower > tolerance  -> exceeds_tolerance
upper <= tolerance -> within_tolerance
otherwise          -> indistinguishable
```

诊断中包含三类示例，分别对应确超容差、确未超容差和定点误差不可区分。

## 6. TCP 子进程与并发调度

每个 shaper 声明由以下命令启动为独立进程：

```bash
python -m qos_diag.shaper_server --spec-dir samples/base --shaper-id sh-01-ingress --ready-file ...
```

服务端绑定声明中的 `127.0.0.1:port`，读取自己的速率/容量声明；请求只携带事件、scale 和上一状态。这样请求是不可变输入，并行调用不会共享进程内可变状态。服务端返回包含同一确定性 record 和 `record_sha256` 的回执；调度器逐字节比较回执 record 与本地规范 record。

`ConcurrencyGovernor` 用条件变量维护：

- `global_inflight <= global_max_inflight`
- `pipeline_inflight[pid] <= pipeline_max_inflight`

未获得配额的工作线程在条件变量上排队；`acquire()` 用 `time.monotonic_ns()` 记录 `wait_start_ms/wait_end_ms/wait_ms`，并统计最大在途数和最大等待。

真实单次调用超时仅由以下 socket 设置产生：

```python
socket.create_connection((host, port), timeout=timeout_seconds)
sock.settimeout(timeout_seconds)
```

演练启动延迟 100ms 的 shaper 并设置 30ms 超时，实测约 32ms 得到 `socket timeout`。没有使用线程池 `wait_for`、`asyncio.wait_for` 或手工计时器伪造超时。

## 7. SQLite 账本与恢复

账本路径：`artifacts/ledger/qos.db`。

关键表：

- `meta`：整数 `format_version=1`、`parameter_revision`、`spec_digest`、`content_digest`。
- `revisions`：每个参数版本的父版本、变更、失效闭包和路径选择。
- `batches`：`planned/committed/unknown` 批次区间。
- `event_results`：每个 revision/event 的完整 canonical event JSON 与 SHA-256。
- `judgments`：逐事件逐级整数字段、等待时间、回执摘要/JSON 和 canonical record。
- `revision_artifacts`：路径选择、统计、误差界、窗口聚合。
- `cursor_state`：当前 revision 与最大已封存连续 `event_seq`。

`commit_event_batch` 在一个 SQLite 事务中写入 revision、batch committed、event rows、judgment rows、artifacts、cursor 和新的 content digest；`PRAGMA synchronous=FULL`、`journal_mode=DELETE`。新库先写临时文件再 `os.replace` 原子替换。

打开账本时检查：

- `PRAGMA user_version` 与 `meta.format_version`；
- schema 表完整性；
- 从所有逻辑表 canonical JSON 重算的 `content_digest`。

若版本或摘要不一致，旧文件被隔离为 `*.invalid-<monotonic-ms>`，随后用临时文件 + `os.replace` 整体替换为空账本，不做增量合并。

崩溃演练中，批次 `r1-b001` 先写入 `planned` 意图；shaper `sh-02-relay` 被强杀后该批次没有提交 judgment/cursor。重新打开时 `mark_unfinished_unknown()` 将其标记为 `unknown`，恢复流程从 `sealed_seq=5` 之后重放 `6..35`。恢复前后对已封存前缀 `0..5` 的 canonical bytes/SHA-256 做精确比较。

参数编辑产生 `revision=2`。同一事务先把父 revision 的 event/judgment 行复制到新版本，再覆盖失效事件和级别，同时写入 revision/change/propagation/cursor/digest。因此未受影响内容仍可逐项追溯。

## 8. 独立校验

```bash
python -m qos_diag verify --spec-dir samples/edited_param --ledger artifacts/ledger/qos.db
```

校验器以 SQLite read-only URI 打开账本，不写任何文件。它只读取：

- 原始 pipelines/shapers/events/config；
- 账本中的 revision、event、judgment、artifact 与传播记录。

随后独立执行：

1. spec 强校验和 digest；
2. 三级判定与双路径 parity；
3. 路径选择、窗口、统计和误差界；
4. 每个 revision 的 invalidation closure 与增量重放；
5. 每个 event/stage 字段、canonical record、receipt digest、等待区间和 cursor。

首个不一致以 `E_VERIFY` 抛出，并带 `event_seq/pipeline_id/stage/field/expected/actual`。测试中篡改 event 0 ingress 的余量并重算 digest 后，校验器准确停在 `event_seq=0, stage=ingress, field=balance_after_decision_q`。
