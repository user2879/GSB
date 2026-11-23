# 本地 SQL 逻辑计划编辑与一致性校验应用

这是一个仅使用 Python 标准库实现的 Windows 本地桌面应用：读取列式表与 SQL/计划编辑声明，生成稳定的逻辑计划树，以增量路径服务界面查询，并在后台用从零全量重编译逐字节校验。计划版本、编辑轨迹、真实延迟和校验结论写入同一份 SQLite 账本；会话快照支持内容摘要检查、整体替换和按账本重放。

## 交付内容

- `planner_app/`：SQL 解析、计划编译、增量编辑、SQLite 账本、快照、CLI 与 Tkinter 界面。
- `samples/workspace.json`：两张列式表、空值位图、两条分组聚合 SQL、5 个确定性编辑批次。
- `tests/`：28 项 `unittest` 自动化测试。
- `artifacts/verification-report.json`：一次完整冷启动演示的实际结果。
- `docs/DESIGN.md`：关键算法、事务边界、错误码和一致性不变量。
- `run_ui.bat`、`run_demo.bat`、`run_tests.bat`：Windows 双击/命令行入口。

## 环境与 Windows 启动

目标环境：Windows 10/11、Python 3.11 或更高、官方安装包中的 Tcl/Tk。

```bat
python --version
python -c "import tkinter; print(tkinter.TkVersion)"
```

若第二条命令失败，请重新运行 Python 安装程序并勾选 **Tcl/Tk and IDLE**。应用不依赖第三方包、网络或外部账号。

> 本次交付的自动验证运行在 WSL/Linux 容器中。该容器有 Python 3.11，但缺少 `libtk8.6.so`，因此 GUI 启动会明确返回 `E_TK`；解析、计划、账本、快照、CLI 和全部测试均已在本地文件与回环环境完成验证。Windows 标准 Python 只需具备 Tcl/Tk 即可启动同一界面。

## 快速开始

```bat
:: 1. 严格校验样例，不创建账本
python -m planner_app check --workspace samples\workspace.json

:: 2. 冷启动完整验收：5 个批次、逐批全量校验、权威重算、快照恢复、账本重放、崩溃意图恢复
python -m planner_app demo --reset-ledger ^
  --workspace samples\workspace.json ^
  --ledger data\demo.sqlite ^
  --snapshots data\snapshots ^
  --log logs\demo.jsonl ^
  --report artifacts\verification-report.json

:: 3. 启动 Tkinter 操作界面
python -m planner_app ui --workspace samples\workspace.json ^
  --ledger data\ui.sqlite --snapshots data\snapshots --log logs\ui.jsonl

:: 4. 全部测试
python -m unittest discover -s tests -v
```

也可以直接执行 `run_demo.bat`、`run_ui.bat`、`run_tests.bat`。

## Tkinter 操作界面

- **SQL 文本**：编辑当前查询并整体应用。SQL 版本号递增，旧计划版本整体标记陈旧，所有查询视图从当前声明全量重建。
- **列式表声明**：以 JSON 编辑整组表；表名/列名唯一、类型、空值位图和行数据全部强校验，失败时旧状态不变。
- **编辑操作**：提交 JSON 数组。默认勾选“后台执行从零全量校验”，界面先返回增量结果，后台线程完成全量比较后再刷新校验状态。
- **快速查询**：只读取增量计划缓存，返回计划摘要、逻辑行数、编译状态和真实单调时钟耗时；不会调用全量编译器。
- **权威重算**：从原始 SQL 和完整编辑轨迹重新编译，逐字节比较并写入账本。
- **刷新视图**：只重绘当前缓存视图，不追加任何账本记录。
- **计划树**：按计划先序、同层节点 ID 稳定排序；节点显示版本、逻辑行数、结构摘要和完整属性。
- **受影响/差异**：显示直接节点、被删节点、祖先传播依据、全部受影响节点、未变节点实例验证、新增/删除/摘要变化集合。
- **延迟与校验**：只从 SQLite 账本读取并展示真实测量值，标注 `mono_start_ns`、`mono_end_ns`、预算和真值来源。
- **保存会话 JSON**：把当前 SQL、表声明和单调版本原子写回本地声明文件；之后使用同一账本重启即可恢复界面内完成的参数修改。
- **快照**：保存版本、摘要和游标；恢复时先校验内容摘要和当前账本，再逐批次重放。

## JSON 声明格式

最小结构如下，完整样例见 `samples/workspace.json`。

```json
{
  "format": 1,
  "name": "Deterministic Retail Demo",
  "workspace_id": "deterministic-retail-demo",
  "scale_factor": 1.0,
  "interaction_budget_ms": 100.0,
  "versions": {"tables": 1, "sql": 1, "edits": 1, "scale": 1},
  "tables": [
    {
      "name": "orders",
      "columns": [
        {
          "name": "amount",
          "type": "float",
          "nulls": [false, true, false],
          "values": [10.5, null, 30.0]
        }
      ]
    }
  ],
  "queries": [
    {
      "name": "q_status",
      "sql": "SELECT status_code, COUNT(*) AS n, SUM(amount) AS total FROM orders WHERE amount >= 10 GROUP BY status_code ORDER BY total DESC"
    }
  ],
  "edit_batches": [
    {
      "name": "push_filter",
      "query": "q_status",
      "edits": [{"type": "push_filter", "target": "filter"}]
    }
  ]
}
```

约束：

- 表名、列名、查询名大小写不敏感且必须唯一。
- 列类型仅允许 `int`、`float`；`values`（兼容 `rows`）与 `nulls` 等长；每个表的各列等长。
- 空值位图为布尔数组；非空位置必须是合法数值。
- 查询只能引用已声明表/列；输出别名唯一。
- 任一结构或引用错误整体拒绝为 `E_SPEC`，并给出首个 JSON Pointer 位置；非法 SQL 语法为 `E_SYNTAX` 并给出字符位置。

## 支持的 SQL 子集

```sql
SELECT column | aggregate [AS alias], ...
FROM single_table
[WHERE predicate]
[GROUP BY column, ...]
[ORDER BY projected_name [ASC|DESC], ...]
```

- 聚合：`COUNT(*)`、`COUNT(column)`、`SUM`、`MIN`、`MAX`。
- 谓词：`=`、`!=`/`<>`、`<`、`<=`、`>`、`>=`、`IN`、`AND`、括号组合。
- 允许行注释 `--` 和末尾分号；不支持 JOIN、OR、HAVING、子查询。
- `AND` 条件会展平并按规范字节排序；`IN` 值去重排序；关键字、标识符、数值字面量均规范化。因此空白、冗余括号、大小写、`AND` 顺序和 `IN` 顺序不影响计划摘要。

## 编辑操作

```json
{"type": "push_filter", "target": "filter"}
{"type": "prune_columns", "target": "project", "columns": ["min_amount"]}
{"type": "replace_scan_order", "target": "scan", "columns": ["amount", "id", "..."]}
{"type": "modify_aggregate", "target": "aggregate", "name": "total_amount", "function": "max", "column": "amount"}
```

- `push_filter`：把 Filter 谓词合并进直接子 Scan，并删除 Filter 节点。
- `prune_columns`：删除 Project 的输出列；不能删空，也不能删除仍被 Sort 引用的输出。
- `replace_scan_order`：给出的列必须是当前 Scan 列集合的严格排列；也接受 `order` 作为 `columns` 的同义键。
- `modify_aggregate`：按输出名修改聚合函数和列；`name` 也接受 `alias`。

运行期缺失目标节点返回 `E_REF`；规格加载阶段发现同一问题则按整体规格校验返回 `E_SPEC`。所有编辑失败都不会提交批次或推进参数版本。

## 一致性、账本与恢复

- 计划节点不可变；节点摘要只覆盖节点自身、属性和子摘要。增量编辑只重建目标节点及祖先闭包，未受影响子树保持同一对象实例、摘要和版本。
- 每批编辑先写 `RUNNING` 意图，再增量编译；成功后在一个 SQLite 事务中写批次、计划字节、受影响集合、传播依据、编辑轨迹、延迟记录、进度游标和参数事件。
- 后台/同步全量校验从原始 SQL 和全部编辑重新编译，按规范化字节比较。不一致时按“先自身结构、再子树、最后派生摘要”的规则定位首个源头节点，并返回 `E_DIVERGE`。
- 进程在批次提交前终止时只会留下 `RUNNING` 意图。重启后它被改为 `UNKNOWN`，不参与恢复，状态从最后完整提交批次重建，并要求重跑；应用不生成虚假的退出报告。
- 界面快速查询只读缓存；权威重算和定时全量校验均写独立校验记录。预算超限使用真实单调时钟耗时标记 `E_LATENCY`，没有人工 sleep。

## 三类异常注入

以下命令分别产生 `E_SYNTAX`、`E_REF`、`E_SNAPSHOT`。命令退出码为 0 表示“按预期捕获了指定业务码”，stdout 为结构化结论，JSONL 日志包含单调时间、编辑序号和业务码。

```bat
python -m planner_app inject --kind syntax --reset-ledger ^
  --workspace samples\workspace.json --ledger data\inject-syntax.sqlite ^
  --snapshots data\snapshots --log logs\acceptance-injections.jsonl

python -m planner_app inject --kind bad-ref --reset-ledger ^
  --workspace samples\workspace.json --ledger data\inject-ref.sqlite ^
  --snapshots data\snapshots --log logs\acceptance-injections.jsonl

python -m planner_app inject --kind snapshot-digest --reset-ledger ^
  --workspace samples\workspace.json --ledger data\inject-snapshot.sqlite ^
  --snapshots data\inject-snapshot-dir --log logs\acceptance-injections.jsonl
```

## 实际验证结果

验证日期：2026-11-24。命令：

```text
python -m unittest discover -s tests -v
Ran 28 tests ... OK
```

冷启动 `demo` 的实际结果（详见 `artifacts/verification-report.json`）：

- 5 个编辑批次全部 `incremental_vs_full = MATCH`，每批纯增量编译约 `0.63–1.12 ms`，最终计划摘要 `efb2ed363477dda90a15583343996f7eb5c8d3afd97565ec372e3e7b44f7f080`。
- 两个初始快速查询分别约 `0.19 ms`、`0.20 ms`；最终快速查询约 `0.42 ms`，预算 `100 ms`，均未触发全量重编译。
- 权威重算约 `3.01 ms`，状态 `MATCH`，真值来源为账本延迟记录。
- 快照恢复与逐批次账本重放均为 `MATCH`；游标 7。
- 模拟未提交编辑意图被恢复为 `UNKNOWN`，恢复计划摘要与崩溃前已提交摘要完全相同，`uncommitted_edit_applied = false`，`exit_report_claimed = false`。
- 三类注入均返回预期业务码并写入 JSONL 日志。

## 已知边界

- 当前 SQL 子集面向任务要求，单查询单表，不支持 JOIN、OR、HAVING、子查询或 DISTINCT。
- 列类型按任务要求限定为整数和浮点；字符串类型未开放到规格表，但解析器保留字符串字面量能力。
- SQL 或表结构变化采用新的参数 epoch 并整体重建所有查询视图，已有编辑轨迹会重置；纯计划编辑批次则保留历史并做增量演进。
- Tkinter 是可选运行依赖；无 Tcl/Tk 时核心 CLI 与测试仍可完整运行。
