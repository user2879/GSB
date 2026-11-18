# 列式聚合管线：倾斜拆分、双预算调度与可重放账本

这是一个面向 Windows 本地环境的纯 Python 列式聚合组件。输入为本地 JSON 中声明的若干列式表与一条分组聚合查询；组件按固定行组生成扫描/聚合分片，在并行度与估算内存双重预算下调度，运行期识别倾斜分组并做一次确定性行区间拆分，最后按稳定规则合并为逐字节可复现的结果。

运行时不使用 Docker、数据库服务或网络端口。SQLite 来自 Python 标准库；`numpy` 与 `pytest` 由虚拟环境安装，聚合执行本身仅使用标准库整数语义。

## 快速开始（Windows CMD）

```bat
cd /d C:\path\to\project
python --version
scripts\setup_windows.cmd
.venv\Scripts\python.exe -m pytest -q
scripts\demo_windows.cmd
scripts\verify_anomalies_windows.cmd
```

`scripts\verify_windows.cmd` 会依次执行测试、三场景演示和异常业务码校验。

手工安装等价命令：

```bat
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install -e .
```

`requirements.txt` 固定安装 `numpy>=1.26,<3` 与 `pytest>=8,<9`。`setup_windows.cmd` 同时检查 Python 3.11+ 与 `sqlite3`。

## 输出位置

- 结果：`<out-dir>\final_result.json`，UTF-8、规范 JSON、末尾单个换行。
- 调度轨迹：`<out-dir>\trace.json`，包含任务时间、排队、拆分、倾斜判定和预算事件。
- 稳定摘要：`<out-dir>\summary.json`，不包含单调时钟时间，便于跨次比较。
- SQLite 账本：默认 `var\ledger\pipeline.sqlite`。
- 快照：默认 `var\snapshots\<run_id>.json`。
- 结构化 JSONL 日志：默认 `var\logs\pipeline.jsonl`。

演示固定使用 `var\demo`，并会在运行前安全地重建该目录；异常校验固定使用 `var\anomaly`。全过程不占用 TCP/UDP 端口。

## 数据格式

样例位于 `data\small.json` 与 `data\skew_amplified.json`。顶层结构：

```json
{
  "format_version": 1,
  "tables": [
    {
      "name": "sales_a",
      "version": 1,
      "row_count": 48,
      "columns": [
        {
          "name": "amount",
          "type": "fixed",
          "scale": 2,
          "null_bitmap": [0, 0, 1],
          "values": [1234, 5678, null]
        }
      ]
    }
  ],
  "query": {
    "from": ["sales_a", "sales_b"],
    "where": "(channel = 'online') AND (amount >= 100) AND (amount <= 900000)",
    "group_by": ["region"],
    "aggregates": [
      {"name": "row_count", "op": "count", "column": "*"},
      {"name": "amount_sum", "op": "sum", "column": "amount"},
      {"name": "amount_avg", "op": "avg", "column": "amount"}
    ]
  }
}
```

约束与语义：

- 支持列类型 `int`、`fixed`、`string`。`fixed` 必须声明非负整数 `scale`，列值本身就是定点整数，不使用浮点数。
- `fixed` 列的过滤字面量、分片求和与最终输出均使用同一 scale 的整数单位。
- `null_bitmap` 长度为 `row_count`，`1` 表示空值；空值位置的 `values` 必须为 JSON `null`。
- 表名、列名、聚合名必须是标识符且对应作用域内唯一；表版本为大于等于 1 的整数。
- `from` 可包含一张或多张表；每张表独立过滤，结果跨表合并。
- WHERE 支持等值 `=`、范围 `>`、`>=`、`<`、`<=` 以及 `AND` 和任意冗余括号。与空值比较结果为假。
- `group_by` 可包含多个列，顺序决定结果键序列；输出按空值优先、再按列的自然类型稳定排序。
- `count` 对 `*` 统计过滤后行数，对具体列统计非空值数；`sum`/`avg` 忽略空值，要求 `int` 或 `fixed` 列。
- 求和为精确整数定点结果；均值为 `floor(sum / nonnull_count)`，即向负无穷截断。误差声明为 `0 <= 精确均值 - 输出均值 < 1 个 scale 单位`。

任一数据或查询校验失败整体拒绝，返回 `E_DATA`，并给出如 `$.tables[0].columns[3].null_bitmap[7]`、`query.aggregates[1].column` 的位置。表内声明了不一致的 `digest` 时返回 `E_DIGEST`。

## CLI

```bat
.venv\Scripts\python.exe -m columnar_pipeline run ^
  --data data\skew_amplified.json ^
  --config data\params\skew.json ^
  --out-dir var\runs\skew ^
  --ledger var\ledger\pipeline.sqlite ^
  --snapshot-dir var\snapshots ^
  --log-file var\logs\pipeline.jsonl
```

常用参数也可直接覆盖配置文件：`--parallelism`、`--memory-budget-bytes`、`--budget-version`、`--split-threshold`、`--split-parts`、`--row-group-size`、`--ledger-batch-size`、`--split-strategy`、`--skew-safety-limit`。

```bat
REM 严格重放：逐分片比较状态、完成行数、聚合中间量、摘要与最终结果
.venv\Scripts\python.exe -m columnar_pipeline replay --snapshot var\snapshots\<run_id>.json --data data\skew_amplified.json --ledger var\ledger\pipeline.sqlite --out-dir var\runs\replay

REM 打开快照：兼容则重放；数据/表版本/预算参数/拆分参数变化则整体替换
.venv\Scripts\python.exe -m columnar_pipeline resume --snapshot var\snapshots\<run_id>.json --data data\skew_amplified.json --ledger var\ledger\pipeline.sqlite --snapshot-dir var\snapshots --out-dir var\runs\resume
```

## 核心机制

- **规范化查询标识**：WHERE 解析为 AND 展平、项排序的 AST；空白、大小写关键字与括号不影响 `query_id`。聚合按名称排序。分片计划摘要由表名、表版本、行区间和引用列集合生成。
- **分片与拆分**：基础分片按 `row-group-size` 划分。分片过滤后若某分组键行数达到 `split-threshold`，且该分片仍允许拆分，则按 `row_range` 策略切成 `split-parts` 个连续、首尾相接、互不重叠的子区间。实际拆分数不超过父区间行数，仅做一层二次拆分。热点并列时按分组键规范 JSON 升序稳定选择。
- **双预算调度**：任一时刻同时运行任务数不超过 `parallelism`，运行任务估算内存之和不超过有效内存预算。内存估算为 `64 + 行数 × 引用列宽度和`；数值列每行 9 字节，字符串列每行取最大 UTF-8 字节数加 1。首次因并行度或预算排队时记录 `queue_started_ms` 与 `queued` 事件，任务真实运行直到结束，不使用线程超时伪造终止。
- **确定性合并**：拆分父分片不计入最终合并，子分片恰好覆盖父区间一次。所有参与任务按 `task_id` 升序合并，分片内分组键稳定排序。计数和求和只用 Python 大整数；均值在全局合并后一次截断，避免分片均值误差累积。
- **账本与恢复**：每个账本批次持久化任务轨迹、倾斜判定、预算事件和当前聚合中间状态。快照保存表摘要、表版本、预算版本/摘要、拆分阈值/份数、账本摘要、最终结果和注入信息。恢复时版本不一致会创建全新会话并记录整体替换，绝不把旧分片状态增量并入。
- **严格重放**：重放按账本顺序重新执行每个已完成分片，依次比较 `status`、`rows_completed`、`aggregate`、`digest`、`included`，再比较最终结果与账本摘要。第一个差异会定位到 `task_id` 和 `field`，返回 `E_REPLAY`。

## 异常注入

注入只通过样本文件或参数完成，不修改源码：

```bat
scripts\verify_anomalies_windows.cmd
```

对应参数样本在 `data\params`：

- `inject_skew.json`：`inject_skew_amplify` 放大热点键；超过 `skew_safety_limit` 时返回 `E_SKEW`（退出码 20）。
- `inject_budget.json`：`inject_budget_shortfall` 把有效预算降到 1 字节；任务无法启动时返回 `E_BUDGET`（退出码 30）。
- `inject_digest.json`：`inject_digest_task: "first"` 对首个分片制造期望摘要不一致，返回 `E_DIGEST`（退出码 40）。
- 篡改账本分片后重放返回 `E_REPLAY`（退出码 50），并定位分片与字段。

## 样本与演示

- `data\small.json`：两张表、84 行的无倾斜小规模样本。
- `data\skew_amplified.json`：两张表、120 行，`east` 键占绝大多数行，用于验证热点拆分。
- `data\params\small.json`：小规模参数。
- `data\params\skew.json`：倾斜参数。
- `data\params\tight_budget.json`：同一倾斜数据的收紧预算参数。
- `data\params\inject_*.json`：三类异常注入参数。

`scripts\demo_windows.cmd` 以固定顺序执行：

1. 无倾斜小规模；
2. 倾斜放大并拆分；
3. 同一倾斜数据下收紧内存预算。

每个场景完成后立即重放，并比较场景 2 与场景 3 的最终结果文件是否逐字节一致。稳定摘要写入 `var\demo\demo_summary.json`。

## 测试

```bat
.venv\Scripts\python.exe -m pytest -q
```

测试覆盖：输入强校验与位置、查询规范化、参考实现结果、跨并行度/阈值逐字节一致性、倾斜拆分区间、内存预算上限、批次账本、快照恢复、整体替换、三类异常业务码和账本篡改的 `E_REPLAY` 定位。

详细设计见 `docs\DESIGN.md`，实际运行记录见 `docs\VALIDATION.md`。
