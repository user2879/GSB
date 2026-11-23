# 技术设计：确定性 WAL 崩溃恢复诊断

## 1. 目标与核心不变量

系统把“并行重放”拆成两个阶段：

1. **并行准备阶段**：按事务和页切分任务，在工作线程中生成确定性页字节、摘要和版本候选；同时执行全局槽位、每事务配额与页字节预算约束。
2. **规范提交阶段**：单线程定序器严格按 LSN 递增消费候选，并在 SQLite 事务中提交页版本、任务结局、检查点、预算、游标与账本事件。

这使线程调度只影响吞吐和时间线，不影响日志语义。必须始终成立：

- 每个页版本号严格大于该页上一版本，不能回退。
- 同一 LSN 的页标识、版本、摘要、可见性在所有隔离度下相同。
- 已提交批次的 SQLite 状态是恢复的唯一权威；页文件先写但晚于 SQLite 提交的孤儿不可见。
- 任何未完整提交的批次都不能推进账本游标。
- 可见事务集合与崩溃点只能来自完整、已持久化的提交前缀。
- 快照内容使用自摘要校验；恢复快照是整体替换，不是逐行合并。
- 刷新/渲染路径只读，不产生账本事件。

## 2. 模块与调用关系

```text
CLI / Tkinter GUI
       |
       +-- validation.load_sample
       |      +-- 页声明校验
       |      +-- NDJSON/内嵌 WAL 校验
       |      +-- 事务分组、崩溃点、目标版本、确定性摘要
       |
       +-- Storage
       |      +-- SQLite schema 与状态查询
       |      +-- 版本化页文件
       |      +-- 原子批次提交
       |      +-- 哈希链账本
       |
       +-- ReplayEngine
       |      +-- build_replay_tasks
       |      +-- ThreadPoolExecutor 并行准备
       |      +-- LSN 单线程定序提交
       |      +-- 强制终止演练入口
       |
       +-- verify_replay
       |      +-- 独立顺序基线
       |      +-- E_REPLAY 首差异与账本定位
       |
       +-- snapshots
              +-- 自摘要快照
              +-- 打开时磁盘日志/页比较
              +-- 全状态替换与稳定页差异
```

## 3. 输入校验顺序

`load_sample()` 是无副作用的纯读取与校验函数。校验顺序固定，确保“首个坏位置”稳定：

1. JSON 是否存在、是否可解析、根是否为对象。
2. `pages` 是否非空数组。
3. 每页：标识规范化、大小、初始版本、标识唯一性。
4. 预算字段与页字节预算下限。
5. `log_file` 或内嵌 `wal`。
6. 每条 WAL 按 LSN 顺序：
   - LSN 正整数类型、首条之后的连续性/唯一性；
   - 事务标识、页标识、页存在性；
   - 摘要格式；
   - commit/persist 布尔类型；
   - 目标版本递增与确定性摘要一致。
7. 事务分组连续性与提交标记位置。
8. 计算崩溃点和可见前缀。

任一错误立即抛出 `BusinessError`，不继续加载，不写 SQLite。`E_ORDER` 针对 LSN 重复/跳号/倒退；`E_PAGE` 针对页声明；其余日志字段问题为 `E_LOG`。异常上下文包括：

```json
{
  "first_invalid_lsn": 2,
  "lsn": 2,
  "field": "content_digest",
  "page_id": "p2"
}
```

## 4. 确定性页内容

给定页标识、页大小、目标版本和 LSN，生成固定字节流：

```python
prefix = f"WALPAGE\x1f{page_id}\x1fsize={size}\x1fversion={version}\x1flsn={lsn}\x1f".encode()
output = b""
counter = 0
while len(output) < size:
    output += sha256(prefix + counter.to_bytes(8, "big")).digest()
    counter += 1
return output[:size]
```

基底版本使用 `lsn=0`。每个 WAL 条目的目标版本等于该页初始版本加此前该页 WAL 条目数。输入中声明的 `content_digest` 必须与生成结果一致；工作线程重新生成时再次校验。

页文件路径为：

```text
pages/<sha256(page_id)[:20]>/v<version:08d>.bin
```

该规则与操作系统、线程、时钟、随机数和 JSON 键顺序无关。

## 5. 崩溃点、提交与持久化

事务组为相邻同 `txid` 的 WAL 序列。规则：

- 只有组最后一条可以 `commit=true`。
- 只有最后一组可以未提交。
- 非末尾组必须已提交。
- 一个提交组只要有任一 `persist=true`，其末端是持久候选。

扫描已完整提交前缀，取最后一个带持久化标记的组末端为 `crash_point_lsn`。`lsn <= crash_point_lsn` 的条目进入持久可见前缀；其余条目仍可写入诊断页版本，但 `visible=0`。

样本事务：

| 事务 | LSN | 结果 |
| --- | --- | --- |
| tx-001 | 1-3 | 提交并持久，可见 |
| tx-002 | 4-6 | 提交并持久，可见 |
| tx-003 | 7-10 | 提交并持久，可见；崩溃点 10 |
| tx-004 | 11-12 | 未提交，页版本不可见 |

`visible` 同时用于恢复后页视图、检查点摘要、GUI 不可见列表和校验中的事务集合比较。

## 6. 调度模型

### 6.1 任务切分

1. 以事务连续组为边界。
2. 事务内按页分组。
3. 每页按 `min(page_byte_budget // page_size, checkpoint_interval)` 条切块。
4. 每个任务以 `(run, txid, page, chunk)` 生成稳定 task id。

任务并行准备页字节；同一页同一事务的多个块即使准备阶段重叠，真正页写入仍由 LSN 定序器串行完成。

### 6.2 双预算

调度器维护：

- `len(active) < global_slots`
- `active_by_tx[txid] < per_tx_quota`

只有同时满足才提交到线程池。被任一条件阻塞的任务写入 `wait_started_ms`；获得槽位时写 `wait_ended_ms` 和 `started_ms`。全部时间使用 `time.monotonic()*1000` 取整，避免墙钟跳变。

线程任务完成只表示“候选已准备”，状态先变 `staged`。定序器消费其 LSN 后才把任务更新为 `partial` 或 `committed`。

### 6.3 定序器

```text
next_lsn = ledger_cursor + 1
while next_lsn <= final_lsn:
    submit_until_budget_full()
    collect_finished_futures_into_ready_effects()
    while ready_effects[next_lsn] exists:
        append effect
        next_lsn += 1
        if batch full or final:
            commit_batch(...)
    if stalled:
        wait(for any future)
```

`ready_effects` 可以乱序填充，但只允许严格递增的 LSN 被消费。这样“并发完成顺序”与“页版本推进顺序”彻底解耦。

## 7. SQLite 数据模型

主库 `<workspace>/state.sqlite3` 使用 WAL 模式和 `synchronous=FULL`。

| 表 | 作用 |
| --- | --- |
| `schema_meta` | 应用 schema 格式版本 |
| `app_meta` | 游标、崩溃点、日志/页/预算摘要、页表版本、运行计数器 |
| `pages` | 页声明、基础摘要、声明顺序、页表版本 |
| `budget_params` | 四类预算、整数参数版本、摘要 |
| `wal_entries` | LSN、事务、页、目标版本、生成摘要、可见性、事务组 |
| `page_versions` | 每页每版本、LSN、摘要、文件路径、可见性、批次 |
| `checkpoints` | 运行、间隔、游标、页数、可见状态 JSON 与摘要 |
| `replay_runs` | 每次运行的游标范围、预算快照、状态、终止说明 |
| `replay_batches` | LSN 区间、页数、字节、游标、检查点、账本前后哈希 |
| `replay_tasks` | 任务区间、预算、单调时间线、结局、消耗、批次 |
| `ledger_events` | 哈希链事件（LOAD、REPLAY_BATCH、RECOVERY_INCOMPLETE、CHECKPOINT_SAVED、RUN_FAILED） |
| `snapshot_meta` | 快照版本、自摘要、日志/页/预算摘要、游标、路径 |

关键索引/约束：

- `pages.page_id` 主键。
- `wal_entries.lsn` 主键。
- `page_versions(page_id, version)` 唯一。
- `kind='wal'` 的 `page_versions.lsn` 部分唯一索引。
- 恢复提交前再次查询该页最大版本，拒绝非递增版本。

## 8. 原子批次边界

`Storage.commit_batch()` 做四件事：

1. 在 SQLite 事务之外，将所有待提交页字节以版本化临时文件 + `os.replace` 原子写入目标路径。
2. `BEGIN IMMEDIATE` 后再次检查账本游标与 LSN 连续性。
3. 在同一事务内插入/更新：
   - 本批 `page_versions` 及摘要；
   - 任务 `partial/committed`、消耗字节、批次；
   - `checkpoints`；
   - `replay_batches`；
   - `app_meta.ledger_cursor`；
   - `ledger_events`；
   - 若最终批，运行状态完成。
4. 计算账本哈希并更新批次记录，然后 `COMMIT`。

哈希链：

```text
event_hash = SHA256(canonical_json({
  event_type, cursor, batch_id, task_id,
  prev_hash, payload_json, created_ms
}))
```

首事件 `prev_hash = 64 个 0`。批次保存 `prev_hash` 和 `ledger_hash`，可与事件行互相验证。

跨介质原子性说明：SQLite 与普通文件无法共享事务。采用先写文件、再提交数据库的顺序，最坏情况只产生无数据库引用的孤儿版本文件；恢复和快照清理不会把它当已提交页。反向顺序会产生“数据库引用不存在文件”的危险状态，因此不使用。

## 9. 崩溃恢复

恢复入口：

1. 找当前账本游标和最近检查点；本实现的自动检查点每个批次都产生，因此二者对齐。
2. 若上次运行仍为 `running` 或存在 `queued/running/staged/partial` 任务，更新为 `unknown`，记录 `RECOVERY_INCOMPLETE`，运行改为 `interrupted_unknown`。
3. 从账本最近运行的 `budget_json` 读取恢复预算，并与当前参数表及活动预算摘要比较；不一致 `E_REPLAY`，不静默改变语义。
4. 只选择 `lsn > ledger_cursor` 的日志建立任务。
5. 新任务使用新 run id；旧 unknown 行保留为诊断时间线，不覆盖历史。

强制终止演练通过 `os._exit(91)` 绕过 `finally`、线程收尾和报告生成，专门证明上述恢复依赖完整 SQLite 批次，而不是进程内状态。

## 10. 恢复校验算法

`verify_replay()` 读取工作区输入副本，重新执行 `load_sample()`，形成独立顺序基线。不读取运行中的调度结果作为期望值。

检查项顺序：

1. 账本哈希链与批次哈希。
2. 活动日志摘要。
3. 预算参数与活动预算摘要。
4. 页声明与基础摘要。
5. WAL 元数据（目标版本、生成摘要、commit/persist/visible）。
6. 每个 WAL 页版本：页、版本、事务、摘要、可见性和磁盘文件摘要。
7. 基底页文件摘要。
8. 可见事务集合。
9. 崩溃点。
10. 每个检查点的可见页集合、版本和摘要。

发现差异时构造 `E_REPLAY` 上下文：

```json
{
  "first_mismatch_lsn": 5,
  "page_id": "p3",
  "field": "content_digest",
  "expected": "...",
  "actual": "...",
  "ledger_record": {
    "batch_id": 2,
    "start_lsn": 5,
    "end_lsn": 8,
    "checkpoint_id": 2,
    "ledger_hash": "..."
  }
}
```

若差异在 LSN 0（页表/基底），账本记录退化为 LOAD 事件。

## 11. 快照与整体替换

快照顶层字段：

```json
{
  "snapshot_format_version": 1,
  "snapshot_id": "snap-...",
  "content_digest": "<canonical payload SHA-256>",
  "log_digest": "...",
  "pages_digest": "...",
  "budget_digest": "...",
  "page_table_version": 1,
  "budget_params_version": 1,
  "ledger_cursor": 12,
  "checkpoint_set": [],
  "wal_text": "...",
  "inputs_sample": "...",
  "tables": {},
  "app_meta": {},
  "page_files": {}
}
```

加载顺序：

1. 读 JSON。
2. 将除 `content_digest` 外的完整对象规范化，计算 SHA-256；不符立即 `E_SNAPSHOT`，零写入。
3. 检查整数格式版本、WAL 摘要、页/预算摘要、检查点集合、页文件覆盖、账本链。
4. 比较当前磁盘日志和当前页版本/页文件，生成稳定差异。
5. 预校验并写出快照页文件与输入副本。
6. 单事务删除并插入全部页表、WAL、页版本、检查点、运行、批次、任务、账本和快照元数据。
7. 清理数据库未引用的页文件。
8. 重新验证账本链。

`added`/`deleted`/`changed` 均按页标识稳定排序。`changed` 触发条件包括版本不同、提交摘要不同或当前磁盘文件摘要不同。

## 12. 异常处理与结构化日志

`BusinessError` 携带业务码、用户消息与普通映射上下文。CLI 输出 JSON；GUI 显示业务码和上下文。应用将错误追加到：

```text
<workspace>/logs/app.ndjson
<workspace>/logs/injections.ndjson    # 仅 --inject
```

记录包含 ISO 时间、级别、业务码、消息和上下文。`load_fresh()` 保留诊断日志，只有 `--action reset` 才删除日志与快照。

## 13. GUI 线程模型与只读刷新

- Tk 主线程只构建控件和渲染查询结果。
- 重放、校验、快照、强制终止演练在后台线程执行。
- 工作线程通过 `queue.Queue` 把业务结果/异常交回主线程；`root.after(120, ...)` 轮询并更新 UI。
- 刷新调用 `Storage.status()`、`tasks()`、`batches()` 等只读方法；不调用 `recover_orphan_tasks()`，因此不会因为打开/刷新而改写 unknown 状态。
- Tkinter 延迟导入，缺少 `_tkinter`/`libtk` 时返回 `E_GUI`，核心引擎和 CLI 仍可用。

## 14. Windows 兼容性

- 所有持久路径用 `pathlib.Path` 拼装，页文件 token 不含用户输入。
- 数据库只读 URI 使用 `Path.as_uri() + "?mode=ro"`。
- 页文件替换使用同目录临时文件和 `os.replace`。
- 快照、NDJSON、规范化 JSON 均显式 UTF-8 和 `\n`。
- 未调用 POSIX 专用 API；单调时钟和 SQLite 均为标准库跨平台实现。
- Python 3.10+ 语法，运行时无第三方包。

## 15. 设计取舍

- **并行准备、单线程序列化提交**：优先保证可证明的日志语义；这与“并行重放”的业务目标兼容，因为确定性内容生成、摘要计算和调度确实是并行的。
- **每批一次检查点**：批次大小等于检查点间隔，使恢复游标和最近检查点天然对齐，简化崩溃边界。
- **页文件版本化而非原地覆盖**：诊断和快照可以保留不可见历史版本；崩溃不会破坏上一已提交版本。
- **账本预算优先**：恢复不能因当前预算表被改动而静默改变任务划分。
- **全量快照替换**：符合题目要求，也避免跨日志/页表/账本的部分合并产生不可判定状态。
