# 列式页 WAL 崩溃恢复诊断组件设计

## 1. 交付结构

- `src/colpage_recovery/loader.py`：本地 JSON 加载、强校验与异常注入变异。
- `src/colpage_recovery/models.py`：页、WAL、恢复预算数据模型。
- `src/colpage_recovery/codec.py`：确定性列编码、NumPy 补丁生成、SHA-256 摘要和页文件解码。
- `src/colpage_recovery/scheduler.py`：真实 `ThreadPoolExecutor` 投射、全局槽位和单事务配额调度。
- `src/colpage_recovery/engine.py`：无 SQLite 依赖的确定性回放内核、检查点计划、崩溃点和可见性判定。
- `src/colpage_recovery/ledger.py`：SQLite schema、原子批次、账本修订、检查点、视图快照和陈旧代际。
- `src/colpage_recovery/service.py`：初始化、提交回放、强杀恢复、检查点重叠校验和 `E_REPLAY/E_SNAPSHOT` 定位。
- `src/colpage_recovery/viewmodel.py`：稳定排序的只读界面模型。
- `src/colpage_recovery/gui.py`：Windows Tkinter 诊断界面；无 Tk 原生库时可导入并返回 `E_GUI` 修复提示。
- `src/colpage_recovery/cli.py`：`validate/run/recover/demo/diagnose/gui/view` 命令。
- `scripts/demo.py`：无需安装即可运行的跨平台演示入口。
- `scripts/run_tests.py`：自动设置 `src` 路径的测试入口。
- `scripts/Run-Demo.ps1`：Windows PowerShell 一键检查依赖并运行演示。
- `samples/`：确定性页声明、WAL 和预算样本。

## 2. JSON 输入

`samples/pages.json` 声明：

- 唯一 `page_id`
- 正整数 `row_count`
- 唯一列名和受支持 dtype：`uint8/int32/int64/float32/float64`
- 与行数等长的 `null_bitmap` 数组，或 packbits 顺序的 hex 字符串

`samples/wal.json` 的 `entries[]` 声明：

- 唯一正整数 `lsn`
- 非空 `tx_id`
- 已声明的 `page_id`
- `[row_start,row_end)` 或 `row_start/row_end`
- 整数 `new_value_seed`
- 布尔或 `0/1` 的 `commit_flag`、`persist_flag`

恢复预算键：

- `global_parallel_slots`
- `per_transaction_replay_quota`
- `single_page_replay_byte_budget`
- `checkpoint_interval_entries`
- 可选 `ledger_batch_entries`，用于显式账本批次粒度；缺省为 `2`
- 可选 `work_delay_ms`，仅让调度时间线更容易观察，不影响内容
- 可选 `marker_protocol`：默认 `steal-no-force`，允许“未提交但脏页已持久化”组合；`wal-strict` 拒绝 `commit_flag=false,persist_flag=true`

常规结构、唯一性、类型、分组或标记协议错误返回 `E_WAL`，并带首个异常 `lsn` 和 `field`。行区间注入返回 `E_RANGE`，提交标记冲突返回 `E_COMMIT`。这两类继承 WAL 校验错误，但保留更精确的验收业务码。

## 3. 确定性页字节规则

页文件魔数为 `CLMPAGE1`，随后是小端页 ID、版本、LSN、行数、列定义、C-contiguous 列字节和 NumPy packbits 空值位图。

每个日志条目的目标页版本是该页截至当前 LSN 的条目序号。工作线程按以下输入生成行区间补丁：

```text
page_id | target_version | lsn | seed | row_index | column_index
```

混合函数使用固定常量的 uint64 乘法/异或回绕；浮点值由高位均匀整数映射到固定范围后转换为 `float32/float64`。因此：

- 工作线程不修改共享页状态。
- 补丁结果与线程名、启动顺序、等待时间无关。
- 提交线程仍严格按 LSN 顺序应用补丁并递增页版本。
- 任意批次大小、全局槽位和事务配额组合得到逐位相同的 `.colpage` 字节和 SHA-256 摘要。

摘分两层：

- `content_digest = sha256("colpage-content-v1" || page_bytes)`
- `version_digest = sha256(page_id|version|lsn|content_digest)`

## 4. 并行调度和预算

`run_quota_schedule` 使用 `ThreadPoolExecutor(max_workers=global_parallel_slots)`。调度线程只有在以下条件同时满足时才提交任务：

- `running_global < global_parallel_slots`
- `running_by_tx[tx_id] < per_transaction_replay_quota`

每个任务记录单调时钟毫秒：

- `wait_started_ms`
- `wait_ended_ms`
- `worker_started_ms`
- `worker_ended_ms`
- `wait_ms`、`run_ms`
- 启动后的全局/事务占用数和真实线程名

单页字节成本是受影响行 × 列宽之和。账本累计每页字节消耗；超过 `single_page_replay_byte_budget` 时以 `E_BUDGET` 拒绝。

## 5. 检查点和崩溃点

检查点按 WAL 顺序每 `checkpoint_interval_entries` 条生成一次，因此样本在 LSN 3、6 生成检查点。检查点包含：

- 已应用 `at_lsn`
- 每个页的 `page_id/version/lsn`
- `version_digest`
- `content_digest`
- 可回读的不可变页文件路径

崩溃点规则为：

```text
crash_after_lsn = max({lsn | commit_flag=true AND persist_flag=true})
```

样本中为 LSN 4。`commit_flag=false` 的事务集合是不可见事务；其每个物理页版本写入 `page_version` 但 `visible=0`，并进入崩溃/不可见版本列表。当前页视图只选择每页最新的 `visible=1` 版本，因此不可见物理版本不会作为当前可见页参与展示。

默认 `STEAL/NO-FORCE` 允许未提交脏页的 `persist_flag=true`；这正是必须用账本、检查点摘要和提交集合交叉校验的场景。

## 6. SQLite 原子账本和强杀

数据库默认位于 `state/<run>/recovery.db`，页内容位于 `state/<run>/pages/g<generation>/...`。

每个轮次：

1. 独立事务插入 `replay_batch(status='running')`，确保崩溃后能看到遗留尝试。
2. 工作线程生成确定性补丁。
3. 提交线程按 LSN 应用并写不可变版本文件，先 fsync 再准备 SQLite 提交。
4. 同一个 SQLite 事务写入：
   - `page_version`
   - `digest_registry`
   - `task_outcome`
   - `budget_usage`
   - `ledger_revision`
   - `progress_cursor`
   - 到达间隔时的 `checkpoint/checkpoint_page/snapshot`
5. 事务提交后批次才是 `committed`。

演示在第 2 轮完成所有 SQL 写入后、`commit()` 前调用 `os._exit(137)`，模拟强制终止。此时：

- running 批次行已经提交，因此仍可见。
- 完成事务未提交，SQLite 自动回滚页版本、任务结局、预算、游标和检查点插入。
- 已写入但未被账本引用的页文件是确定性孤儿文件，恢复时会重写到相同路径。

重启后 `mark_running_unknown` 把遗留 running 批次整体标记为 `unknown`，然后以新的 `attempt_no` 重跑同一逻辑轮次。

## 7. 检查点续做和 `E_REPLAY/E_SNAPSHOT`

恢复流程：

1. 将 running 批次标记为 unknown。
2. 校验所有已入账页文件与 `page_version/digest_registry` 摘要；不一致报 `E_SNAPSHOT`，定位首个 `page_id/lsn/content_digest`。
3. 取最近已提交检查点。样本强杀后检查点为 LSN 3，而已提交账本游标为 LSN 4。
4. 从检查点页文件重建页状态，按账本轮次先重放重叠区间 LSN 4。
5. 与不中断一次回放的同区间比较页版本号、内容摘要和版本摘要；回放内核自相不一致时报 `E_REPLAY`。
6. 重叠区间再与已提交账本比对；账本/快照被篡改时报 `E_SNAPSHOT`。
7. 从最后已提交游标之后重跑 unknown 轮次及剩余轮次，并原子提交。
8. 从检查点对完整剩余区间再做一次逐批校验。
9. 比较已提交事务集合和不可见版本集合；差异报 `E_REPLAY`，并定位首个页标识与字段。

## 8. 预算/页集合演进和陈旧防护

`spec_generation.spec_hash` 覆盖页集合、列、空值位图和完整预算。每次初始化时：

- 当前代际哈希未变：复用当前 G。
- 当前哈希变化：`G = max(G)+1`，即使哈希等于某个旧代际也不复用旧版本号。
- 所有旧代际的 `spec_generation/checkpoint/replay_batch/snapshot` 标记 `stale=1`，旧 `view_snapshot` 标记 `stale`。
- 新代际写入全新的当前 `view_snapshot` 和 `page_view` 集合；界面不把新旧页行做增量合并。

界面模型只读取 `current_generation` 的 WAL、页、检查点、轮次、修订和快照，同时显示陈旧代际/检查点/视图计数。

## 9. Tkinter 界面

Windows 官方 Python 3.11+ 通常自带 Tk/Tcl。运行：

```powershell
python -m colpage_recovery.cli gui --state-dir state/demo --report artifacts/demo_report.json
```

页签覆盖：

- 页列表、当前可见版本和摘要
- WAL 行区间与标记
- 调度时间线、线程、等待/运行毫秒和占用
- 单页预算占用
- 检查点及检查点页摘要
- 崩溃点 JSON 和不可见版本
- 账本修订与快照列表

所有 Treeview 按 `page_id/lsn/round/revision/snapshot_id` 等稳定键排序。
