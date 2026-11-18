# SQLite 慢查询诊断与改写工作台

面向本地 Windows 桌面/命令行的离线领域应用。系统使用 Python 标准库与本地文件 SQLite，不依赖网络、Docker 或外部数据集。核心入口包括 Tkinter 工作台、Python API 和可复现的 CLI 演示。

## 快速开始（Windows）

```bat
python --version
python -c "import tkinter, sqlite3; print(tkinter.TkVersion, sqlite3.sqlite_version)"

python -m slowquery_workbench check --json
python -m slowquery_workbench gui

python -m unittest discover -s tests -v
python -m slowquery_workbench demo --work-dir .workbench-demo --json
```

也可双击 `start_workbench.bat`。如果 `python` 不在 PATH 中，使用 Windows Python Launcher：`py -3 -m slowquery_workbench gui`。

## 功能与结构

- `slowquery_workbench/session.py`：SQLite 会话、版本高水位守卫、领域表/索引/统计、执行账本、结论失效事件、快照替换。
- `slowquery_workbench/query.py`：受限 SELECT 解析、规范化与稳定 `query_id`。
- `slowquery_workbench/diagnostics.py`：四类候选生成、预算内多次回放、`E_EQUIV` 作废、恢复后重放校验。
- `slowquery_workbench/workbench.py`：脚本与 GUI 共用的高层 API。
- `slowquery_workbench/gui.py`：Tkinter 桌面工作台。
- `slowquery_workbench/cli.py`：`check`、`gui`、`demo` 与三类真实异常注入。
- `samples/`：确定性样本配方、建表 SQL、查询清单和损坏输入样本。
- `tests/`：21 项标准库 `unittest` 验收测试。

## 数据与查询口径

`samples/data.sql` 使用固定算术公式和递归 CTE 生成 2,000 个客户与 20,000 个订单，不使用随机数；`samples/schema.sql` 建表，`samples/dataset.json` 提供索引与阈值清单，查询清单位于 `samples/queries.json`。

支持的查询面：一个或两个表的 `INNER JOIN`，投影，`=`、`<>`、`<`、`<=`、`>`、`>=`、`IN`、`BETWEEN` 与顶层 `AND`，以及可选的 `ORDER BY`/`LIMIT`。子查询、聚合、`OR`、外连接、DML、DDL（查询入口）和三个及以上表均返回 `E_PARSE`。

规范化会统一关键字/标识符大小写和空白，但保留字符串字面量大小写。`query_id = "q_" + sha256(canonical_sql)`，用于慢查询、候选和结论归属。

## 观测与慢查询判定

- 时钟基准写入会话元数据：`time.perf_counter_ns() / 1_000_000`，即本地单调计数毫秒。起止绝对计数只用于计算差值，不跨进程比较。
- 每条执行独立写入 `__sw_executions`：单调起止、实测耗时、扫描行数、结果行数、执行种类、候选 ID、data/index/stats 版本和状态。
- `scanned_rows` 的真实执行口径：对同一 `FROM + JOIN + WHERE` 语义实际运行配套 `COUNT(*)`，记录进入连接/过滤管线的行数；`ORDER BY`、投影和 `LIMIT` 不作为该计数的一部分。它不是常量估算。
- 慢查询按 `耗时 >= 耗时阈值 OR 扫描行 >= 扫描行阈值` 判定。阈值保存在会话中，但 `slow_queries()` 每次都从原始账本重新计算；修改阈值不会覆盖任何历史观测。

## 候选与回放

每条慢查询最多生成四类候选：

1. `predicate_merge`：合并同列上下界、移除重复谓词。
2. `index_hint`：对受限列加入 `INDEXED BY`，并引用索引名、索引集合版本与统计版本。
3. `projection_convergence`：把 `*` 收敛到连接、过滤、排序实际引用的列。
4. `join_order`：交换两表连接顺序。

原查询与每个候选均在单次单调超时和总预算内实际执行。候选默认采样 3 次并取中位数；每次结果行数都与原查询比较。不一致时候选状态为 `invalid_equiv`，写入 `E_EQUIV` 日志，并且永远不参与最优候选选择。`current_best()` 只返回未陈旧结论。

## 版本与失效传播

每个会话持久化 `data_version`、`index_version`、`stats_version`，并在 `__sw_version_guard` 保存高水位。任何数据追加、索引增删、统计刷新都会原子地推进对应版本。打开会话时，版本缺失、非整数或低于高水位均以 `E_SESSION` 拒绝；外部改坏索引/表结构快照则以 `E_VERSION` 拒绝。

每次版本变化都产生 `__sw_change_events`，记录版本来源、旧/新值、受影响表和稳定排序后的查询 ID。传播只更新引用受影响表的结论；未受影响结论的 `is_stale`、`stale_reason` 和 `updated_at` 保持不变。陈旧结论仍可审计，但不能出现在当前最优候选中。

## 快照与恢复校验

`save_snapshot()` 使用 SQLite backup API 整体复制会话文件，包含结构、数据版本、索引集合、统计快照、执行账本、全部候选与结论。

`restore_snapshot()` 先逐项比较 `schema_hash`、`data_version`、`index_set_hash`、`index_version`、`stats_version`。无论是否一致都会整体替换当前会话状态；若存在差异，替换完成后抛出 `E_VERSION` 并在 `details.differences` 中逐项列出 `current` 与 `snapshot` 值。调用方可捕获该错误后直接继续使用已恢复状态。

`replay_conclusions()` 会先验证查询标识、记录版本和候选是否存在，缺少任一条件即抛 `E_REPLAY`，`details.unreproducible` 中包含具体 `query_id` 与原因（例如 `candidate deleted: c_...`）。随后真实重放原查询和所有有效候选，比较结果行数、耗时统计量、严格最优候选和噪声容差内等价性，并持久化校验报告。

## 异常码

| 业务码 | 含义 |
|---|---|
| `E_PARSE` | 非支持 SELECT 面、子查询、聚合、DML 等 |
| `E_SESSION` | 会话不存在、元数据缺失、版本回退、非法会话参数 |
| `E_DATA` | SQLite 数据/约束/运行错误、损坏样本输入 |
| `E_VERSION` | 索引集合、快照或期望版本不一致 |
| `E_TIMEOUT` | 真实单调计时超过声明超时或总预算 |
| `E_EQUIV` | 候选行数与原始结果不一致，候选作废 |
| `E_REPLAY` | 版本不可用、查询/候选缺失等不可复现链路 |

所有 CLI 失败与关键业务事件均写入 JSONL；默认路径为工作目录的 `workbench.jsonl`，也可用 `--log` 指定。记录包含 `ts`、`level`、`event`、`code`、`message`、`details`。

## 真实异常注入

```bat
python -m slowquery_workbench demo --work-dir .fault-data --inject data --json
python -m slowquery_workbench demo --work-dir .fault-index --inject index-version --json
python -m slowquery_workbench demo --work-dir .fault-timeout --inject timeout --json
```

- `data`：把种子输入切换到 `samples/corrupt_data.sql`，由 SQLite 的真实 NOT NULL 约束失败返回 `E_DATA`。
- `index-version`：打开会话时注入错误的期望索引版本，返回 `E_VERSION` 并列出差异字段。
- `timeout`：以 `1 ms` 超时执行重查询；SQLite progress handler 与 `perf_counter_ns()` 截止时间共同中止，实测耗时超过阈值才返回 `E_TIMEOUT`。没有 `sleep` 或伪造错误。

## 脚本示例

```python
from slowquery_workbench import Workbench, WorkbenchError

wb = Workbench.create(r"C:\work\demo.sqlite", name="demo", overwrite=True)
wb.initialize_samples()

sql = wb.sample_queries()[0]["sql"]
observation = wb.execute_select(sql, timeout_ms=500)
result = wb.diagnose(sql, timeout_ms=500, budget_ms=4000,
                     threshold_ms=0, threshold_rows=0)
print(result["conclusion"]["optimal_candidate_id"])

snapshot = wb.save_snapshot(r"C:\work\demo-snapshot.sqlite")
wb.append_rows("orders", ["id", "customer_id", "status", "amount", "created_at"],
               [(999999, 1, "paid", 123.45, "2024-12-31")])
try:
    wb.restore_snapshot(snapshot)
except WorkbenchError as exc:
    assert exc.code == "E_VERSION"  # 状态已经整体替换完成

report = wb.replay_conclusions(repeats=2, timeout_ms=500, budget_ms=8000)
print(report["matched"])
wb.close()
```

## 实际验证记录

当前容器：Python `3.11.0rc1`，SQLite `3.37.2`。该 Linux 容器未安装 Tkinter，因此 GUI 仅完成源码编译校验；Windows 官方 Python 默认包含 Tkinter，桌面路径由 `python -m slowquery_workbench gui` 启动。

```text
$ python3 -m unittest discover -s tests -v
test_data_injection_is_genuine_e_data ... ok
test_index_version_injection_is_e_version ... ok
test_normal_demo_end_to_end ... ok
test_timeout_injection_uses_monotonic_timing ... ok
test_data_and_statistics_changes_propagate ... ok
test_four_candidate_classes_replay_and_choose_fastest ... ok
test_index_change_marks_only_affected_conclusions_stale ... ok
test_inequivalent_candidate_is_invalidated_and_not_selected ... ok
test_predicate_merge_removes_weaker_lower_bound ... ok
test_rejects_subquery_aggregate_and_dml ... ok
test_slow_threshold_uses_elapsed_or_scan_rows ... ok
test_supported_ranges_in_and_join ... ok
test_whitespace_and_case_produce_same_canonical_and_id ... ok
test_close_reopen_and_monotonic_versions ... ok
test_create_table_append_and_clock_declaration ... ok
test_observation_ledger_and_threshold_recalculation ... ok
test_rollback_and_missing_versions_reject_open_with_e_session ... ok
test_equal_version_snapshot_restores_without_conflict ... ok
test_missing_candidate_raises_e_replay_with_query_id_and_reason ... ok
test_restore_reports_version_diff_and_replaces_whole_state ... ok
test_version_change_makes_replay_unreproducible ... ok

Ran 21 tests in 14.149s
OK
```

正常演示的实测结果：四类候选均为 `predicate_merge`、`index_hint`、`projection_convergence`、`join_order`，全部 `valid`；索引变更后原结论变为陈旧；快照恢复因 `data_version` 差异返回 `E_VERSION` 且状态已整体替换；恢复后重放 `matched=true`。

异常注入实测：损坏输入返回 `E_DATA`（`NOT NULL constraint failed: orders.customer_id`）；索引版本不一致返回 `E_VERSION`（期望 2、实际 1）；真实超时返回 `E_TIMEOUT`（某次记录为 `elapsed_ms=1.108285`、`timeout_ms=1`）。耗时数字随机器负载变化，但判定始终来自实际单调计时。
