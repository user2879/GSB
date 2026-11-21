# MVCC 可见性诊断工作台

一个面向本地 Windows 环境的确定性 MVCC 快照可见性诊断工作台。它从本地 JSON 读取事务、版本链、读点与查询，在全局并发上限和每读点并发配额下，用真实线程池并行推导可见版本；同时把结论、理由链摘要、预算分配、调度轨迹和游标写入 SQLite 账本，支持强制终止恢复、声明演进后的整体替换和重放校验。

项目不依赖容器、外部数据库或网络。除标准库外只使用 NumPy 做谓词数值判定。

## 1. 环境与运行

目标环境：

- Windows 10/11 本地 Python 3.10+
- 标准库：`sqlite3`、`tkinter`、`threading`
- NumPy 1.24+
- 不需要 Docker 或其他容器

安装与验证：

```powershell
python --version
python -c "import sqlite3, tkinter, threading; print('stdlib ok')"
python -m pip install -r requirements.txt
python -m tkinter
```

如果当前机器没有图形显示或 `tkinter` 不可用，使用命令行演示：

```powershell
python scripts\demo.py
```

Windows 下也可以直接双击/执行：

```text
scripts\demo.bat
scripts\run_tests.bat
```

本次实现所在容器为 Linux + Python 3.11.0rc1 + NumPy 1.26.4，未安装 `tkinter`。因此本次实测使用 CLI 无显示模式；`tkinter` 界面代码已实现，并在无 Tk 时以 `E_UI` 明确降级。Windows Python 官方安装包通常自带 Tk，建议在目标机执行一次 `python -m tkinter` 确认。

## 2. 目录

```text
mvcc_workbench/
  loader.py          JSON 强校验与规范化
  visibility.py      可见版本选择、理由链、谓词、成本估算
  budget.py          稳定优先级预算分配
  scheduler.py       全局/每读点双重配额线程池调度
  ledger.py          SQLite 账本、摘要、恢复、快照
  workbench.py       冷启动、演进替换、重放、报告编排
  renderer.py        稳定文本渲染
  ui.py              Tkinter 诊断界面
  cli.py             命令行入口
samples/
  generate_samples.py
  base.json
  evolved.json
  invalid_chain_cycle.json
  invalid_txn_commit_regression.json
  invalid_snapshot_digest.json
scripts/
  demo.py            三次会话 + 崩溃恢复 + 异常注入演示
  demo.bat
  run_tests.bat
tests/               17 个 unittest 测试
```

## 3. 快速命令

冷启动判定：

```powershell
python -m mvcc_workbench.cli run --input samples\base.json --db artifacts\mvcc.sqlite --log artifacts\events.jsonl --json
```

提交结局与版本链演进后整体替换：

```powershell
python -m mvcc_workbench.cli run --input samples\evolved.json --db artifacts\mvcc.sqlite --log artifacts\events.jsonl --json
```

再次打开进行重放校验：

```powershell
python -m mvcc_workbench.cli run --input samples\evolved.json --db artifacts\mvcc.sqlite --log artifacts\events.jsonl --json
```

强制终止后的恢复检查：

```powershell
python -m mvcc_workbench.cli recover --input samples\base.json --db artifacts\mvcc.sqlite --json
```

打开 Tkinter 界面：

```powershell
python -m mvcc_workbench.cli ui --input samples\base.json --db artifacts\mvcc.sqlite
python -m mvcc_workbench.cli ui --input samples\base.json --db artifacts\mvcc.sqlite --run-before-open
```

异常注入样本：

```powershell
python -m mvcc_workbench.cli inject --kind chain_cycle --input samples\invalid_chain_cycle.json --json
python -m mvcc_workbench.cli inject --kind txn_commit_regress --input samples\invalid_txn_commit_regression.json --json
python -m mvcc_workbench.cli inject --kind snapshot_digest --input samples\invalid_snapshot_digest.json --db artifacts\inject.sqlite --json
```

## 4. JSON 样本格式

有效样本是一个 JSON 对象。`manifest` 保存快照比较所需的整数集合版本；`budget` 保存成本与并发预算；`visibility_policy` 显式声明自写可见规则。

```json
{
  "schema_version": 1,
  "manifest": {
    "schema_version": 1,
    "dataset_id": "local-mvcc-seed-20261122",
    "txn_set_version": 1,
    "chain_version": 1,
    "budget_version": 1,
    "read_point_set_version": 1
  },
  "transactions": [
    {"txn_id": "T3", "start_seq": 4, "outcome": "committed", "commit_seq": 8},
    {"txn_id": "T5", "start_seq": 7, "outcome": "rolledback"}
  ],
  "records": [
    {
      "row_key": "acct:1",
      "version_no": 1,
      "txn_id": "T3",
      "value_summary": {"balance": 135, "currency": "CNY"},
      "value_digest": "sha256:...",
      "prev_version": null
    }
  ],
  "read_points": [
    {"read_point_id": "RP01", "snapshot_seq": 9, "reader_txn_id": "T3"}
  ],
  "queries": [
    {
      "query_id": "Q01",
      "read_point_id": "RP01",
      "row_key": "acct:1",
      "predicate": {
        "mode": "all",
        "conditions": [{"column": "balance", "op": ">=", "value": 130}]
      },
      "priority": 10
    }
  ],
  "budget": {
    "global_cost_budget": 35,
    "round_cost_budget": 14,
    "global_concurrency": 3,
    "per_read_point_concurrency": 1,
    "task_timeout_ms": 1000
  },
  "visibility_policy": {
    "own_uncommitted_visible": true,
    "own_write_ignores_snapshot": true,
    "declaration": "Read-your-own-writes is explicit..."
  }
}
```

说明：

- `records` 也可替换为 `version_chains`，每项包含 `row_key` 和 `versions` 数组。
- 同一 `row_key` 对应唯一版本链；链内 `version_no` 必须唯一、严格递增。
- `prev_version` 必须为空或指向链内紧邻前一版本；引用成环、悬挂或跳跃均拒绝。
- `value_digest` 可省略，加载时由 `value_summary` 计算；若声明则必须匹配。
- 谓词支持 `gt/gte/lt/lte/eq/ne` 及符号形式，`mode` 为 `all` 或 `any`。
- 样本由整数种子生成：

```powershell
python samples\generate_samples.py --seed 20261122
```

同一 `--seed` 生成逐字节一致的样本。

## 5. 可见性规则

对每个查询，按读点 `snapshot_seq` 和读事务 `reader_txn_id`，沿版本链从最新版本回溯：

1. 写入事务为 `rolledback`：跳过，理由 `SKIP_ROLLEDBACK`。
2. 写入事务不是读事务自身：只有 `committed` 且 `commit_seq <= snapshot_seq` 才可见，否则记录 `SKIP_COMMIT_AFTER_SNAPSHOT`。
3. 写入事务是读事务自身：
   - 已提交且提交序号不晚于快照：可见，理由 `VISIBLE_OWN_COMMITTED`；
   - 快照时尚未提交但在后续提交：依据显式 read-your-own-writes 策略可见，理由 `VISIBLE_OWN_WRITE_POLICY`；
   - 策略不允许时跳过，理由 `SKIP_OWN_WRITE_BY_POLICY`。
4. 找到可见版本后，用 NumPy 对谓词条件做向量化布尔判定。
5. 所有被检查版本、跳过原因、可见原因和谓词结果都进入理由链；理由链规范 JSON 的 SHA-256 为 `reason_digest`。

每个判定还记录：

- 可见版本号、写入事务、值摘要；
- `steps_examined`、谓词列数、`cost_units`；
- 读点、快照序号、读事务、显式可见性策略声明。

版本链在加载时按 `(row_key, version_no)` 规范化，因此原始记录顺序不会改变可见版本或理由链摘要。

## 6. 强校验与异常码

事务校验：

- 事务标识唯一、非空；
- `start_seq`、`commit_seq` 为非负整数；
- `outcome` 只能是 `committed` 或 `rolledback`；
- 已提交事务必须有 `commit_seq >= start_seq`；
- 回滚事务不得带提交序号。

版本链校验：

- 行键成链且唯一；
- 版本号在链内唯一、严格递增；
- 写入事务必须存在；
- `prev_version` 必须存在、不得成环；
- 值摘要必须匹配。

读点与查询校验：

- 读点/查询标识唯一；
- 快照序号、谓词数值合法；
- 读点引用已存在事务；
- 查询引用已存在读点与行链。

违规整体拒绝，并定位首个违规标识：

| 异常码 | 场景 | CLI 退出码 |
|---|---|---:|
| `E_TXN` | 事务表违规，如提交序号倒退 | 2 |
| `E_CHAIN` | 版本链违规，如成环 | 3 |
| `E_QUERY` | 读点/查询/预算声明违规 | 4 |
| `E_REPLAY` | 快照声明或结论不一致，已整体替换 | 演进执行成功时为 0，错误字段中仍报 `E_REPLAY` |
| `E_SNAPSHOT` | 账本或快照内容摘要不符 | 6 |
| `E_UI` | 无 Tk 或无显示环境 | 7 |
| `E_TASK` | 判定任务超时/异常，本轮不提交 | 8 |

## 7. 成本预算与稳定分配

每条查询的整数代价：

```text
cost_units = 版本链实际回退检查步数 + 谓词条件列数
```

分配器按 `(priority, read_point_id, query_id)` 稳定排序，在 `round_cost_budget` 内贪心装入，并同时受 `global_cost_budget` 总上限约束。

- 单轮不超轮预算；
- 全轮合计不超全局预算；
- 装不下时进入下一轮；
- 预算耗尽/单查询超轮预算时记录延后查询、原因和估算代价；
- 同输入重复分配产生逐字节一致的规范 JSON。

样本冷启动计划为 3 轮、总代价 35：

- 第 1 轮：`Q01, Q03, Q05`，代价 11；
- 第 2 轮：`Q07, Q02`，代价 12；
- 第 3 轮：`Q04, Q08, Q06`，代价 12。

## 8. 并行调度

`QuotaScheduler` 使用真实 `ThreadPoolExecutor`：

- 线程池 worker 数等于 `global_concurrency`；
- 调度器只在全局运行数未达上限时提交；
- 每个读点单独计数，只在该读点运行数小于 `per_read_point_concurrency` 时提交；
- 被双配额阻挡的任务保留在队列，记录 `queue_enter_mono_ms` 与 `queue_exit_mono_ms`；
- worker 记录真实 `started_mono_ms`、`ended_mono_ms`、`wait_ms`、`worker_delay_ms`、`duration_ms`；
- 超时使用真实单调时钟截止时间，不以 sleep 模拟；
- 发生超时/错误时本轮不提交，未完成判定标为未知并要求重跑。

每轮调度轨迹写入 SQLite，并与该轮结论和游标同事务提交。

## 9. SQLite 账本与强制终止恢复

账本文件使用：

- `PRAGMA user_version = 1`；
- `meta.format_version = "1"`；
- WAL + `synchronous = FULL`；
- 每轮一行 `rounds` 记录，含整数格式版本、预算分配、调度轨迹、结论、内容摘要；
- 每次提交快照一行 `snapshots`，含声明版本、摘要、结论摘要、游标和内容摘要；
- `inflight` 在任务开始前提交；
- 强制终止后重启时，`RUNNING` 记录转为 `UNKNOWN_REQUIRES_RERUN`；
- `recover` 只展示已完整提交轮次，并明确 `report_available=false`；
- 重跑并提交该轮后，未知任务转为 `RERUN_COMPLETED`。

每轮持久化使用单个 `BEGIN IMMEDIATE ... COMMIT`，所以不会出现“结论已写但游标未写”的中间状态。

## 10. 快照比较、整体替换与 E_REPLAY

快照携带：

- 事务集合版本与摘要；
- 版本链版本与摘要；
- 预算参数版本与摘要；
- 读点集合版本、查询摘要；
- 结论摘要；
- 账本游标；
- 完整声明内容和内容摘要。

重新打开时：

1. 若声明与已提交快照一致，并且快照游标覆盖的结论可确定性重算一致：返回 `REPLAY_OK`，不写账本。
2. 若声明发生变化：从旧快照和当前声明推导受影响事务、行、读点、查询；计算变化前后可见版本；将当前视图切到新代际并完整重跑，不做增量合并。
3. 若声明相同但存储结论与确定性重算不一致：报 `E_REPLAY`，定位首个 `read_point_id`、`row_key`、`field`，并整体替换当前视图。
4. 未受影响查询逐项给出 `before_decision_bytes`、`after_decision_bytes` 和 `byte_equal`，用于人工复核。

样本演进：

- `T5` 从回滚变为提交，提交序号 9；
- 新增事务 `T11`；
- 向 `acct:1`、`acct:2` 追加版本；
- 受影响查询：`Q01, Q02, Q03, Q04, Q05, Q06`；
- 未受影响查询：`Q07, Q08`，理由链与可见版本逐字节一致；
- 首个不一致定位：读点 `RP01`、行 `acct:1`、字段 `txn_set_version`；
- 新代际为 generation 2，账本游标从 3 前进到 6；
- 第三次会话为 `REPLAY_OK`，游标保持 6。

## 11. Tkinter 诊断界面

`python -m mvcc_workbench.cli ui ...` 打开窗口，包含六个稳定排序标签页：

- Overview：状态、异常码、代际、账本版本、结论摘要、恢复状态；
- Chains：版本链、事务结局、提交序号、值摘要、读点快照线；
- Decisions：每条查询的可见版本、读点、理由链、摘要、谓词与成本；
- Schedule：每个任务的队列、开始、结束、等待、耗时与结局；
- Budget & Ledger：每轮预算、延后查询、已提交账本轮次；
- Affected：受影响读点/查询、变化前后版本、未受影响逐项字节校验。

“Refresh view”只读取当前快照和账本，不推进游标、不写轮次；“Run / resume”才执行并持久化。

## 12. 异常注入

三份变体文件，仅通过 CLI 参数选择：

| 参数 | 文件 | 期望异常码 | 实测退出码 |
|---|---|---|---:|
| `--kind chain_cycle` | `invalid_chain_cycle.json` | `E_CHAIN` | 3 |
| `--kind txn_commit_regress` | `invalid_txn_commit_regression.json` | `E_TXN` | 2 |
| `--kind snapshot_digest` | `invalid_snapshot_digest.json` | `E_SNAPSHOT` | 6 |

`snapshot_digest` 注入会在独立 scratch 数据库完成一次正常提交，再按样本中的错误摘要改写最新快照摘要，然后重新打开账本，由摘要校验报 `E_SNAPSHOT`。不会修改源码或正常账本。

## 13. 实际验证结果

本次环境执行：

```text
python3 -m unittest discover -v
Ran 17 tests in 2.952s
OK
```

`python3 scripts/demo.py` 实测摘要：

```json
{
  "ok": true,
  "ui_mode": "CLI_HEADLESS",
  "sessions": {
    "cold": {
      "status": "COLD_START_COMPLETE",
      "generation": 1,
      "ledger_cursor": 3,
      "conclusion_digest": "sha256:deeb296f97f1f0199862d4fa45718a35835b4070987de6c88e150f5bb7237d2a"
    },
    "evolution": {
      "status": "EVOLUTION_REPLACED",
      "code": "E_REPLAY",
      "generation": 2,
      "affected_query_ids": ["Q01", "Q02", "Q03", "Q04", "Q05", "Q06"],
      "unaffected_byte_equal": true
    },
    "replay": {
      "status": "REPLAY_OK",
      "generation": 2,
      "ledger_cursor": 6
    }
  },
  "crash_recovery": {
    "crash_exit": 87,
    "recovery_status": "RECOVERY_REQUIRED",
    "report_available_before_rerun": false,
    "resume_status": "RESUMED_COMPLETE",
    "unknown_after_resume": 0
  },
  "injections": {
    "chain_cycle": {"expected_code": "E_CHAIN", "exit_code": 3},
    "txn_commit_regress": {"expected_code": "E_TXN", "exit_code": 2},
    "snapshot_digest": {"expected_code": "E_SNAPSHOT", "exit_code": 6}
  }
}
```

演示产物位于 `artifacts/demo/`：

- `session1_cold.json`
- `session2_evolution.json`
- `session3_replay.json`
- `crash_recovery.json`
- `results.json`
- `events.jsonl`

结构化日志为逐行 JSON，含 `timestamp`、`event`、账本游标、轮摘要、异常码和定位。

## 14. 取舍与边界

- 判定任务本身是纯 CPU、短任务；线程池用于满足真实并行调度和配额语义。超时通过单调截止时间检测，发生超时后不提交该轮，避免把未完成结论写入账本。
- 预算按结论成本（回溯步数 + 谓词列数）计，不按 CPU 毫秒计，因此分配可逐字节复现；调度时间戳天然不可逐字节复现，故只比较调度上限与轨迹字段，不做跨运行字节比较。
- 演进替换使用新的 generation；旧 generation 的已提交轮次保留为审计历史，但当前视图与快照只读取新 generation，不增量拼接。
- 初始输入只允许 `committed` 与 `rolledback`，与强校验要求一致；读事务自己的“快照时尚未提交、后续提交”的版本通过显式 read-your-own-writes 策略建模，回滚版本永不可见。
- 当前容器无 Tk，因此窗口仅完成实现与无 Tk 降级验证；Windows 目标机需执行 `python -m tkinter` 做最终桌面确认。
