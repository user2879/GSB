# qplan：本地 SQLite 查询计划与索引选择诊断平台

这是一个面向本地 Windows 环境的确定性诊断平台。它从 JSON 声明生成 SQLite 样本数据，枚举查询访问路径，以可复核代价模型估算并搜索索引集合，再真实执行每个候选路径与全表扫描基线，最后把估算、选择、结果摘要、偏差归因、版本和批次摘要写入 SQLite 账本。

平台不访问网络、外部数据库或容器；核心验证使用 Python 3.11、内置 `sqlite3`、Tkinter 和 NumPy。

## 能力概览

- 严格声明校验：表/列唯一、整数或浮点类型、选择性提示、候选索引列序、过滤算子、查询引用与联接图；首个违规位置以 `E_SPEC` 整体拒绝。
- 确定性数据：SplitMix64 哈希生成可重复样本，原子发布 SQLite 数据文件；支持数百、数千到十二万行的混合规模。
- 四类路径：全表扫描、单列索引、多列联合索引、索引覆盖扫描；全部真实执行并输出排序无关的行集合 SHA-256 摘要。
- 可复核代价：访问行数、比较次数、回表次数、IO 页数四项整数权重合成总代价；同时记录提示选择性、精确/抽样直方图中间项和统计口径版本。
- 自适应统计：低于阈值全量采集，达到阈值确定性抽样；两条口径同时留档，输出样本量、置信区间和误差上界，误差重叠时按稳定规则决胜并标注影响。
- 索引搜索：候选数达到规则阈值时从穷举切换为贪心；稳定排序消除声明顺序影响；有穷举审计真值时，差异以 `E_SEARCH` 报告差异索引。
- 真实偏差：最优路径与全表扫描基线双向比对摘要、代价比、行数比；超过阈值以 `E_BIAS` 报告并归因到选择度、覆盖判定或联接顺序。
- 版本失效：行数、提示、统计口径、候选索引或权重变化时递增方案版本，列出受影响查询和失效依据，旧结论整体标记陈旧，当前视图只取新快照。
- 事务账本：每个查询批次在一个 SQLite 事务中提交估算中间项、选择、执行摘要、偏差、版本和内容摘要；强制终止后可恢复已提交批次并将缺失批次标为未知。
- 独立复核：`verify` 仅读取声明文件和 SQLite 账本（含归档统计）逐路径重算估算；不一致时定位首个查询和字段。
- Tkinter 桌面窗口：声明选择、运行、账本刷新、恢复、独立复核和结果分页查看。

## Windows 准备

1. 安装 Python 3.11 或更高版本。建议使用 python.org 官方安装包，并保留安装器中的 **Tcl/Tk and IDLE** 组件。
2. 在项目目录打开 PowerShell 或 `cmd`：

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import sqlite3, tkinter, numpy; print(sqlite3.sqlite_version, tkinter.TkVersion, numpy.__version__)"
```

3. 确认 Tk 窗口可启动：

```bat
python -m qplan.cli gui --spec examples\sample_spec.json --workspace workspace
```

如果 `import tkinter` 报 `_tkinter` 或 Tcl/Tk 动态库错误，请重新运行 Python 安装器并勾选 Tcl/Tk 组件，或安装与 Python 匹配的 Tk 发行包。当前自动化验证容器是 Linux/WSL 且缺少 `libtk8.6.so`，因此只验证了 GUI 的导入失败提示和代码语法；Windows 正式验收需按上述步骤确认窗口。

## 快速开始

也可以在 `cmd` 中直接运行：

```bat
run_sample.bat
```

等价命令：

```bat
python -m qplan.cli spec-check --spec examples\sample_spec.json
python -m qplan.cli run --spec examples\sample_spec.json --workspace workspace --batch-size 2 --out workspace\sample_report.json
python -m qplan.cli verify --spec examples\sample_spec.json --ledger workspace\qplan_ledger.sqlite --out workspace\sample_verify.json
python -m qplan.cli show --ledger workspace\qplan_ledger.sqlite
```

样本包含：

- `products` 350 行（精确统计口径）
- `customers` 20,000 行（抽样统计口径）
- `orders` 120,000 行（抽样统计口径）
- 4 条等值联接加过滤查询、11 个候选索引、27 条访问路径

## JSON 声明

完整示例见 `examples/sample_spec.json`，字段说明见 `docs/SPEC_SCHEMA.md`。最小结构如下：

```json
{
  "seed": 20261123,
  "stats": {
    "version": "stats-v1",
    "row_threshold": 5000,
    "sample_size": 2048,
    "confidence": 0.95,
    "histogram_buckets": 16,
    "value_count_limit": 4096,
    "page_rows": 100,
    "random_seed": 20261123
  },
  "cost_weights": {"rows": 1, "comparisons": 1, "lookups": 4, "io_pages": 2},
  "search": {"exhaustive_max_candidates": 6, "oracle_max_candidates": 12},
  "execution": {"warmup_runs": 1, "repetitions": 2},
  "bias": {"threshold": 1.75},
  "tables": [
    {
      "name": "items",
      "row_count": 1000,
      "columns": [
        {"name": "id", "type": "integer", "selectivity_hint": 0.001,
         "generator": {"kind": "unique_int", "start": 1}},
        {"name": "category", "type": "integer", "selectivity_hint": 0.1,
         "generator": {"kind": "categorical", "values": [1, 2, 3, 4]}},
        {"name": "price", "type": "real", "selectivity_hint": 0.2,
         "generator": {"kind": "uniform_float", "min": 1, "max": 100, "decimals": 2}}
      ],
      "candidate_indexes": [
        {"name": "idx_items_category", "columns": ["category"]},
        {"name": "idx_items_category_price_id", "columns": ["category", "price", "id"]}
      ]
    }
  ],
  "queries": [
    {
      "id": "Q_items",
      "tables": ["items"],
      "filters": [{"table": "items", "column": "category", "op": "eq", "value": 2}],
      "joins": [],
      "select": [
        {"table": "items", "column": "id"},
        {"table": "items", "column": "price"}
      ]
    }
  ]
}
```

支持的过滤算子：`eq`、`ne`、`lt`、`le`、`gt`、`ge`、`between`、`in`。

## CLI

| 命令 | 作用 |
|---|---|
| `spec-check` | 严格校验声明并输出摘要；非法时返回 `E_SPEC` 和首个位置 |
| `run` | 生成/复用数据、统计、估算、索引搜索、真实执行、偏差归因并按批次提交账本 |
| `show` | 只读显示当前快照、会话、批次和查询结论；`--full` 显示完整视图 |
| `recover` | 恢复已提交批次，标记强制终止后缺失的未知批次 |
| `verify` | 只读账本和声明，重算全部路径估算并逐项比对 |
| `gui` | 启动 Tkinter 桌面窗口 |

常用 `run` 选项：

- `--weights '{"rows":1,"comparisons":2,"lookups":4,"io_pages":1}'`：会话整数权重。
- `--row-threshold 5000`、`--sample-size 2048`：覆盖自适应统计切换与样本量。
- `--exhaustive-max-candidates 6`：覆盖穷举/贪心切换阈值。
- `--repetitions 2`：覆盖每路径实测重复次数。
- `--batch-size 2`：每个账本事务包含的查询数。
- `--expect-snapshot-version`、`--expect-snapshot-digest`：显式快照识别；不匹配时创建新版本并整体替换当前视图。
- `--force-new-version`：即使内容未变也创建新的方案版本。
- `--crash-after-batches N`：测试钩子，提交第 N 个批次后调用 `os._exit(75)` 模拟强制终止。

## 测量与估算口径

- **结果摘要**：对列名和按规范 JSON 编码、排序后的完整结果行集计算 SHA-256。行顺序不影响摘要；同路径重复执行摘要不同会抛 `E_EXEC`。
- **真实耗时**：`time.perf_counter_ns` 包围 `cursor.execute()` 和 `fetchall()`，预热后取多次实测中位数，字段 `real_elapsed_ms`。
- **累计统计耗时**：同路径所有实测重复的耗时之和，字段 `cumulative_ms`。
- **代价单位**：访问行数、比较次数、回表次数、IO 页数分别乘会话整数权重，字段 `cost` 为整数总代价。
- **偏差代价比**：先用全部查询的全表扫描基线做全局校准，得到 `unit_ms_per_cost`；将最优路径估算代价换算为毫秒，再除以该路径真实中位耗时。字段同时给出基线与最优路径的双向代价/时间比。
- **偏差阈值**：代价比或行数比超出 `[1/threshold, threshold]` 时发 `E_BIAS`。归因优先级为覆盖判定不一致、选择度误差、联接顺序偏差；每项均附实测依据。
- **全表扫描基线**：驱动表使用 `NOT INDEXED` 强制扫描；联接探测允许使用已声明物理索引，以避免 12 万 × 2 万行退化成数十亿次全扫描，同时保留“驱动表无索引访问”的基线语义。

## 统计、搜索与失效

- 精确统计与确定性系统抽样始终同时归档在账本快照中。
- 行数 `< row_threshold` 使用精确口径，否则使用抽样口径；若两口径选择性差不超过抽样误差界，仍采用自适应口径，但标记 `tiebreak_applied` 和 `error_affected`。
- 索引候选先按稳定键排序。候选数不超过穷举阈值时以穷举为主；不超过审计上限时同时运行穷举真值，若贪心结果不同，发 `E_SEARCH`，列出差异索引并采用穷举真值。
- 当前视图是“最新 current 快照 + 该快照最新会话”的整视图。旧快照和旧批次被标为 `stale`，不会与新结论增量合并。

## 账本与恢复

账本表包括：

- `snapshots`：快照版本、方案版本、规格/统计/权重摘要，完整声明与归档统计。
- `sessions`：批次计划、预期批次数、运行/完成/恢复状态、未知批次。
- `batches`：批次号、内容摘要和完整批次 JSON。每个批次包含该批查询的路径估算、选择、执行摘要和偏差归因。

`run` 在批次开始前先提交会话清单；每个批次单独事务提交。强制终止后：

```bat
python -m qplan.cli recover --ledger workspace\qplan_ledger.sqlite
python -m qplan.cli show --ledger workspace\qplan_ledger.sqlite
python -m qplan.cli verify --spec examples\sample_spec.json --ledger workspace\qplan_ledger.sqlite --allow-partial
```

恢复只返回已提交批次，缺失批次明确标记 `unknown`。未加 `--allow-partial` 时，`verify` 对未知批次返回 `E_RECOVERY`。

## 独立复核

`verify` 的输入只有声明文件和账本：

- 声明提供表、列、提示、候选索引和查询数据契约。
- 账本提供统计快照、会话权重、搜索配置、批次估算与执行摘要。
- 不打开生成的样本 SQLite 数据文件。

复核会先校验快照和批次摘要，再重算所有路径及搜索选择。首个不一致以如下结构定位：

```json
{
  "code": "E_VERIFY",
  "status": "MISMATCH",
  "first_mismatch": {
    "query_id": "Q_status_region",
    "field": "path_estimates.idx_orders_status:single.comparisons",
    "ledger_value": 12345,
    "recomputed_value": 12346
  }
}
```

## 测试

```bat
python -m unittest discover -s tests -v
```

测试覆盖声明校验、统计双口径、估算字节稳定性、候选顺序不变性、四类路径、`E_SEARCH`、三类偏差、版本失效、账本篡改替换、强杀恢复和部分复核。实际运行记录见 `docs/VALIDATION.md`。

## 目录

```text
qplan/                核心包与 CLI/GUI
examples/             确定性样本声明
tests/                单元与端到端测试
docs/                 设计、声明格式与验证记录
run_sample.bat        Windows 样本运行
run_gui.bat           Windows Tkinter 启动
recover_sample.bat    Windows 恢复与查看
```

## 设计边界

- 为保证每类候选路径都能真实执行，所有声明候选索引都会物理建立；全局选择集合决定查询可用的显式访问路径，而不是通过删索引来限制联接探测。
- 时间值本身不可确定复现；平台只要求结果摘要可复现，并通过多次执行检测摘要漂移。
- 代价模型是可独立重算的诊断模型，不等同于 SQLite 优化器内部代价；偏差归因同时保留 SQLite `EXPLAIN QUERY PLAN` 观测。
- 本平台不执行写查询、DDL 注入或外部数据访问。
