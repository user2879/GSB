# 技术设计与实现说明

## 1. 分层与模块

| 模块 | 职责 |
|---|---|
| `model.py` | 不可变领域对象、确定性索引、规格/事务集/链/读点/预算/结论摘要 |
| `loader.py` | JSON/NDJSON 解析、严格 schema、引用与环校验 |
| `visibility.py` | 单读点可见性推导、理由链、逐查询与物化路径、一致性比较 |
| `planner.py` | 规模指标、阈值、工作量估算、交界带稳定决胜 |
| `scheduler.py` | 双配额派发、真实线程池、单调时钟轨迹 |
| `invalidation.py` | 事务结局/追加版本传播、陈旧标记、整体替换 |
| `ledger.py` | SQLite pending/committed 轮次、内容摘要、崩溃恢复 |
| `snapshot.py` | 版本化快照、内容摘要、整体替换、重放校验 |
| `engine.py` | 选路、并行执行、双路径复核、账本轮次编排 |
| `cli.py` / `gui.py` | 可脚本化验收与 Tk 可视化 |

数据流：

```text
JSON/NDJSON -> strict loader -> SpecIndex
                              |-> planner -> selected algorithm
                              |-> ThreadPool (global + per-read-point quotas)
                              |-> selected results + alternate results
                              |-> E_DIVERGE check
                              |-> SQLite pending -> atomic round commit
                              |-> snapshot identity + replay check
                              `-> Tk views / JSON/NDJSON artifacts
```

## 2. 规范化与字节确定性

`canonical_json()` 固定：

- `sort_keys=True`
- `separators=(",", ":")`
- `ensure_ascii=False`
- 禁止 NaN/Infinity
- UTF-8 编码

事务按 `id`、版本按 `(row_key, version)`、读点和查询按 `id` 排序。规格摘要由事务集版本、
版本链摘要、读点/查询集合版本和预算版本组成。理由链摘要覆盖规则版本、查询/读点/行键、
快照序号、谓词、可见版本、逐步理由、回退步数与终止码。

因此同一逻辑输入只要内容相同，数组排列不同也会得到相同规格摘要、结论摘要和理由摘要。
`tests/test_visibility.py` 对 JSON、乱序 JSON 与 NDJSON 三种输入直接比较结果规范字节。

## 3. 版本链与可见性

`SpecIndex` 按行键组织版本，链头取最大版本号，随后沿 `prev_version` 回溯。为防止异常
图造成遗漏，未进入主链路径的碎片再按版本号降序稳定补齐；加载阶段已拒绝缺失引用与环。
因此遍历顺序与输入数组顺序无关。

判定状态机：

```text
candidate version
  |
  +-- writer == reader and writer.pending -> SELF_UNCOMMITTED (visible)
  +-- writer.pending (other)               -> exclude WRITER_PENDING_NOT_SELF
  +-- writer.rolledback                    -> exclude WRITER_ROLLED_BACK
  +-- writer.committed and commit > snap   -> exclude COMMIT_AFTER_SNAPSHOT
  +-- writer.committed and commit <= snap  -> COMMITTED_AT_OR_BEFORE_SNAPSHOT (visible)
  `-- no candidate                         -> NO_VISIBLE_VERSION
```

谓词在可见版本确定后应用，谓词失败不会继续回退到更旧版本。这是“快照读到的版本再匹配
谓词”的语义，避免谓词改变可见性快照边界。

## 4. 两条算法路径

### 4.1 逐查询路径

每个查询调用一次 `resolve_visibility`，不共享可见版本缓存。适用于查询少、读点/行组合
分散或物化启动成本不划算的情况。

### 4.2 按读点物化路径

先对唯一 `(read_point_id, row_key)` 组合并行物化一次 `ResolvedVisibility`，再让共享该
组合的查询分别应用自己的谓词并生成理由链。适用于同一快照上同一行被大量查询复用的情况。

两路径最终都调用 `build_query_result()`，所以理由对象和摘要算法完全共用。`compare_results`
按查询 ID 排序，依次比较 `visible_version`、`exclusion_reasons`、
`reason_chain_digest`；首个差异抛 `E_DIVERGE`，并包含 `query_id`、`row_key`、`field`、
`left`、`right`。

## 5. 规划器

基础指标：

```text
Q = query_count
R = row_key_count
L = total_version_count / R
U = unique(read_point_id, row_key)

per_query_work    = Q * (L + 1.0) + 0.5 * Q
materialized_work = U * (L + 2.0) + 0.35 * U + 0.20 * Q
                  = U * (L + 2.35) + 0.20 * Q
relative_delta    = abs(materialized_work - per_query_work) / max(both, 1)
```

阈值：`Q >= 32`、`R >= 12`、`L >= 5.0`。至少 2 个跨界时规模先验为 `materialized`，否则为
`per_query`。若 `relative_delta <= 0.10`，进入交界带，按规模先验稳定决胜并设置
`threshold_influenced=true`；带外选较低估算值。

计划输出保留：

- `metrics`：关键规模参数
- `estimates`：两条路径工作量、差值、相对差和公式
- `boundary_values`：实际值、阈值、是否触发、距阈值距离、未触发边界
- `default_path`、`selected_path`、`threshold_influenced`、`reason_codes`

这允许复核“为什么选这条路径”，而不是只输出一个路径名。

## 6. 双配额调度

`run_scheduled_tasks()` 使用 `ThreadPoolExecutor(max_workers=global_concurrency)`。中央
派发循环维护：

- `active_global`
- `active_by_read_point[read_point_id]`
- 稳定排序后的 pending 队列

只有全局与读点配额同时满足时，任务才被提交。扫描 pending 时允许跳过被读点配额阻塞的
队头任务，调度其他可运行读点，避免不必要的队头阻塞；结果仍按任务 ID 排序返回。

时间字段全部来自 `time.monotonic_ns()`：

- 排队时间
- 首次因配额被阻塞的等待开始
- 获准提交时的等待结束
- 工作线程内实际开始/结束

同时用线程内计数器记录实际并发峰值，而不是仅记录派发器的账面并发。任务异常会等待所有
已运行任务结束后以 `E_SCHEDULER` 统一报告。

调度预算对象还保存 `allocation.tasks_by_read_point` 与
`allocation.blocked_tasks_by_read_point`，使账本可以复核“限额如何实际分配”，而不只保存
两个上限。

生产实现没有 `sleep`。测试中的 `sleep` 仅用于把并发窗口拉长，以验证硬上限。

## 7. SQLite 账本

### 7.1 表

| 表 | 用途 |
|---|---|
| `ledger_info` | 整数格式版本、schema 版本 |
| `pending_rounds` | 计算前登记；`building` 表示尚未完整提交 |
| `rounds` | 原子提交的轮次头、计划、调度、预算、进度、内容摘要 |
| `conclusions` | 每查询结论、可见版本、理由 JSON、理由摘要 |

### 7.2 两阶段持久化

1. `begin_pending()` 在独立自动提交事务中写入 `building` 和查询 ID 列表。
2. 计算完成后，`commit_round()` 显式 `BEGIN IMMEDIATE`，在**同一事务**内插入轮次头、
   全部结论并更新 pending 为 `committed`，最后提交。

强制终止发生在 2 之前时，只有 1 的结果存在。恢复时只读 `rounds`，因此不会读到半成品；
`pending_rounds` 中仍为 `building` 的查询以 `UNKNOWN_RERUN_REQUIRED` 返回。发生在 2 内
的终止由 SQLite 原子回滚，状态仍停留在 1。

轮次内容摘要覆盖轮次头、全部结论、计划、调度轨迹、预算和进度。恢复时会从数据库行重建
相同对象并重算摘要，不一致返回 `E_LEDGER`。

## 8. 会话快照

快照 identity 包含：

```text
transaction_set_version
chain_digest
budget_version
read_point_set_version
conclusion_digest
ledger_cursor (integer)
```

快照还保存完整规格、计划、结论和顶层 `content_digest`。保存采用同目录临时文件、
`fsync`、`os.replace`，避免半写文件。

加载时先校验格式版本与内容摘要，再校验规格。恢复时逐项比较当前 identity：

- 不一致：`E_SNAPSHOT`，`details.action=REPLACE_ALL`
- 显式整体替换：使用快照完整状态，不做字段级合并
- 替换后：从快照规格重放全部查询并比较结论摘要

## 9. 失效传播

事务结局变化传播链：

```text
transaction -> changed outcome
            -> all version records written by transaction
            -> queries on those row keys
            -> read points referenced by those queries
```

追加版本传播链：

```text
row_key -> appended version -> queries on row key -> read points
```

证据是结构化 `chain` 数组，受影响查询集合和证据均按稳定键排序。`mark_stale_results()`
生成 `STALE/FRESH` 视图；`replace_affected_results()` 校验所有未受影响查询的规范 JSON
字节完全不变，然后整体替换受影响集合。

内部模型允许 `pending` 事务以验证“未决 -> 提交/回滚”；本地 JSON 加载器仍严格限制为
`committed/rolledback`。

## 10. Tk 界面

诊断在后台守护线程执行，完成后通过 `after()` 回主线程刷新，避免阻塞 Tk 事件循环。界面
状态与领域对象分离：

- 概览页展示计划、规模、边界与一致性
- 版本链页展示行/版本父子树和完整值摘要
- 读点页展示快照序号线与查询理由链
- 调度页展示预算占用和每条任务的排队/等待/运行区间
- 快照/账本页展示全部版本标识、摘要、游标和恢复结论

字体优先选择 Microsoft YaHei / Noto Sans CJK 等 CJK 字体；Windows 官方 Python 无需额外
配置。截图是可选辅助能力，不影响 GUI 启动或诊断功能。
