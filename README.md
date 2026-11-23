# JoinLab：Windows 本地连接顺序与代价模型实验台

JoinLab 是一个不依赖容器、外部数据源或网络服务的本地查询实验台。它从 JSON 工作负载读取列式样本表和多表内连接查询，完成规范化连接图、连接顺序与算法枚举、定点代价估算、双口径统计、三重约束并行执行、SQLite 账本、快照恢复和重放校验。

核心只使用 Python 标准库；NumPy 固定为 `1.26.4`，演示报告会记录实际版本。桌面面板使用 Python 官方安装包自带的 Tkinter，找不到 Tk 运行库时不影响 CLI 和领域组件。

## Windows 快速开始

推荐在 PowerShell 或 `cmd.exe` 中执行：

```bat
python --version
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m joinlab demo --reset --workload samples\demo_workload.json --db artifacts\demo_ledger.sqlite --out artifacts\demo_report.json
```

也可以直接双击或在命令行运行 `run_demo.bat`。启动可选桌面面板：

```bat
python -m joinlab gui --workload samples\demo_workload.json --db artifacts\joinlab_gui.sqlite
```

所有路径都是普通 Windows 本地文件；SQLite 使用标准库 `sqlite3` 的 WAL/FULL 同步模式，不使用 Docker、容器、网络或远程数据库。

## 目录

| 路径 | 作用 |
|---|---|
| `joinlab/loader.py` | JSON 强校验、首个非法表/列/字段定位 |
| `joinlab/sqlparse.py` | 查询 SQL 子集解析与规范化 AST |
| `joinlab/graph.py` | 规范化连接图、无笛卡尔积顺序、算法组合与显式上限 |
| `joinlab/stats.py` | 全量/确定性抽样双口径、Wilson 置信区间 |
| `joinlab/cost.py` | 定点代价、直方图连接选择度、逐算子误差界 |
| `joinlab/scheduler.py` | 全局并发、单表配额、内存预算三重准入 DAG 调度 |
| `joinlab/executor.py` | 哈希、排序归并、嵌套循环执行及结果一致性校验 |
| `joinlab/ledger.py` | 原子批次、内容摘要、整数格式版本、快照和完整性检查 |
| `joinlab/session.py` | 会话编排、选优、失效传播、整体视图替换、恢复重放 |
| `joinlab/cli.py` / `gui.py` | CLI 与可选 Tkinter 面板 |
| `samples/demo_workload.json` | 固定种子三表样本 |
| `tests/` | 29 项自动化测试 |

## 输入声明

完整样例见 `samples/demo_workload.json`。结构摘要：

```json
{
  "format_version": 1,
  "tuning": {
    "seed": 20240601,
    "fixed_point_scale": 1000,
    "exact_scan_threshold_rows": 100,
    "sample_size": 128,
    "max_candidates": 18,
    "global_concurrency": 4,
    "per_table_scan_quota": 1,
    "memory_budget_bytes": 67108864,
    "bias_threshold_ratio": 1000000000.0
  },
  "tables": [
    {
      "name": "orders",
      "columns": [
        {"name": "id", "type": "integer"},
        {"name": "amount", "type": "float"}
      ],
      "rows": [[1, 42.5], [2, null]],
      "null_bitmap": [[0, 0], [0, 1]],
      "histograms": {
        "id": {"version": 1, "bins": [{"lo": 1, "hi": 3, "count": 2}]},
        "amount": {"version": 1, "bins": [{"lo": 0, "hi": 50, "count": 1}]}
      }
    }
  ],
  "query": {
    "sql": "SELECT ... FROM a INNER JOIN b ON a.id = b.a_id WHERE ... GROUP BY ..."
  }
}
```

约束与校验规则：

- 表和列名必须匹配 `[A-Za-z_][A-Za-z0-9_]*`，不得使用 SQL 保留字；各自唯一。
- 类型只接受 `integer` 或 `float`。整数列样本值必须是整数；所有非空值必须有限。
- `null_bitmap` 是行数 × 列数的二维 0/1 矩阵。位为 1 时对应样本值必须为 `null`，位为 0 时不得为 `null`。
- 每列必须有直方图。区间采用半开形式 `[lo, hi)`，相邻区间必须满足前一 `hi` 等于后一 `lo`，不得重叠或留空隙。
- 直方图计数之和必须等于该列非空行数。整数列端点必须是整数。
- 查询目前支持 `SELECT`、两张以上表的 `INNER JOIN ... ON 等值谓词`、仅含 `AND`/括号的过滤、`GROUP BY`、数值聚合 `SUM/COUNT/AVG/MIN/MAX` 和可选 `ORDER BY`。不支持字符串、OR、子查询和表别名。
- 连接/过滤列必须存在；整数和浮点可比较；非聚合投影列必须出现在 `GROUP BY`。
- 任一错误整体抛出 `E_SCHEMA`，并给出首个违规的 `table`、`column`、`field`。

查询文本只作为输入载体；连接图摘要、计划 ID 和候选集合都来自规范化 AST，因此空白、大小写关键字和冗余括号不会改变摘要或候选集合。

## 候选与代价

连接图节点按表名排序，边和谓词按规范化 JSON 排序。枚举只扩展与已连接集合存在等值谓词的下一个表，产生笛卡尔积的前缀会记录为：

```json
{"prefix": ["orders"], "next_table": "regions", "reason": "CARTESIAN_WITHOUT_EQUALITY"}
```

每个合法顺序与每个连接步骤的 `HASH/MERGE/NESTED_LOOP` 组合形成稳定候选 ID。超过 `max_candidates` 时按固定字典序遍历截断，并列出被省略候选及 `CANDIDATE_LIMIT` 原因。

代价模型给出的每个字段都是定点整数。会话缩放因子 `fixed_point_scale=S` 下：

- 导出值 `fixed = trunc_toward_zero(value × S)`，负数夹到 0。
- 导出值超过 `2^63-1` 时饱和到 `2^63-1`，并标记 `saturated=true`。
- 内部估算使用高精度 `Decimal`，输出边界统一量化；每个算子每个量化字段的误差上界为 1 个定点单位，计划级上界为未饱和时 `算子数` 个定点单位；饱和时上界为 `2^63-1`。
- 逐算子输出访问/输出行数、比较次数、IO 页数、峰值内存、合成代价、直方图版本和统计口径版本。
- 过滤选择度来自自适应统计；等值连接选择度按直方图重叠区间的均匀密度积分计算，多谓词按独立选择度相乘。

## 双统计口径

每张表、每个过滤谓词同时保留：

- `exact-v1`：扫描全部行，给出精确选择度和扫描行数。
- `sample-v1-nK-sSEED`：用固定种子确定性抽取 `K` 行，给出抽样选择度、样本规模和 Wilson 置信区间。

当表行数 `< exact_scan_threshold_rows` 时最终估算选用全量口径；达到或超过阈值时选用抽样口径。最终估算逐算子声明所选口径及版本，报告中的 `caliber_comparison` 逐项列出两种口径的定点选择度差。

## 并行调度与真实执行

计划被拆为每表扫描任务和左深连接任务，任务依赖构成 DAG。任务只有同时满足以下条件才会提交到本地线程池：

- 当前运行任务数 `< global_concurrency`；
- 任一相关表的当前运行任务数 `< per_table_scan_quota`；
- 当前预留内存之和 `< memory_budget_bytes`；
- 所有依赖任务已经完成。

内存预留在任务提交前获取。扫描预留上界覆盖全表输出；连接预留使用相关表行数笛卡尔上界的两倍加固定开销，因此执行层可校验实测峰值不超过预留，实测超限会以 `E_RESOURCE` 拒绝。调度轨迹记录单调时钟纳秒与毫秒等待/运行时间、完成任务行数、预留和峰值内存。最终摘要同时报告观测到的全局、单表和内存最大值及三个 `*_respected` 布尔值。

真实执行实现了三种算法：

- `HASH`：右侧构建哈希桶，左侧探测。
- `MERGE`：两侧按键排序并归并等值组。
- `NESTED_LOOP`：左行逐右行比较。

输出统一为逻辑列名排序后的规范字节串；分组聚合使用 `Decimal` 累加，消除浮点加法顺序差异。最优计划与 SQL 原始顺序 + `NESTED_LOOP` 的基线逐字节比较原始连接行和分组结果；任何差异抛 `E_DIVERGE`，并报告首个分组键和字段。

代价/耗时比使用合成定点代价单位除以真实毫秒，作为实验指标。与 1 的偏离倍数超过 `bias_threshold_ratio` 时抛 `E_BIAS`，并报告贡献最大的算子及依据。

## 账本、快照和重放

SQLite 账本整数格式版本为 `1`。每次运行批次在同一个事务中写入：

- 全量候选估算及中间算子；
- 最优选择结论与稳定排名；
- 真实执行摘要、一致性结论、偏差结论；
- 调度轨迹；
- 进度游标；
- 完整会话快照和批次/快照内容摘要。

批次开始前会单独提交 `RUNNING` 尝试记录。进程终止后，已提交批次是唯一可恢复事实；遗留 `RUNNING` 尝试在重启时标为 `UNKNOWN` 并标记 `requires_rerun=true`。恢复从最新完整快照整体替换会话状态，只重跑缺失批次。

完整性检查同时验证批次/快照 `payload_json` 的 SHA-256 摘要，以及规范化明细行重建后是否与批次负载逐字段一致。

重放会恢复指定或最新快照，按账本记录的工作负载、统计版本和调度参数重新计算候选估算、真实输出摘要、任务完成行数与峰值内存、主导偏差算子和进度游标。首个不一致以 `E_REPLAY` 报告计划 ID、算子 ID 和字段路径。

## 变更集与整体替换

`ExperimentSession.apply_change_set(...)` 支持：

```json
{
  "table_row_counts": {"orders": 200},
  "histograms": {"customers": {"id": {"version": 2, "bins": [{"lo": 1, "hi": 61, "count": 60}]}}},
  "fixed_point_scale": 2000,
  "max_candidates": 12,
  "exact_scan_threshold_rows": 250,
  "sample_size": 256
}
```

行数缩小时确定性截断；扩大时按原行循环复制并重算直方图计数；空表不能扩大。若同时提供直方图，则以显式直方图为准。会话沿旧计划引用的表推导受影响候选，旧代估算和选择整体进入 `stale_conclusions`，当前视图一次性替换为新代，不是逐条修补。`view_diff` 稳定列出 `added`、`removed`、`retained` 及原因。

## CLI 与异常注入

```bat
python -m joinlab demo --reset --workload samples\demo_workload.json --db artifacts\demo_ledger.sqlite --out artifacts\demo_report.json
python -m joinlab replay --workload samples\demo_workload.json --db artifacts\demo_ledger.sqlite
python -m joinlab verify --workload samples\demo_workload.json --db artifacts\demo_ledger.sqlite
```

异常注入只改变输入文件或命令行：

```bat
python -m joinlab demo --workload samples\bad_overlap_histogram.json --db artifacts\bad.sqlite
python -m joinlab demo --fail-baseline --reset --db artifacts\diverge.sqlite
python -m joinlab demo --tamper-ledger replay_operator --reset --db artifacts\replay_bad.sqlite
python -m joinlab demo --interrupt-before-commit 1 --reset --db artifacts\resume.sqlite
python -m joinlab demo --changes samples\change_set.json --reset --db artifacts\changes.sqlite
```

错误码：`E_SCHEMA`、`E_DIVERGE`、`E_BIAS`、`E_RESOURCE`、`E_LEDGER`、`E_LEDGER_VERSION`、`E_RECOVERY`、`E_REPLAY`、`E_INTERRUPTED`。

## 领域组件集成

```python
from joinlab import ExperimentSession, load_workload
from joinlab.ledger import Ledger

workload = load_workload("samples/demo_workload.json")
session = ExperimentSession(workload)
analysis = session.analysis_report()

with Ledger("artifacts/integration.sqlite") as ledger:
    execution = session.run(ledger)
    replay = session.replay_check(ledger)
```

## 实际验证结果

验证环境：Linux 验证机（非容器），Python `3.11.0rc1`，NumPy `1.26.4`，SQLite 标准库；目标运行环境仍是 Windows 本地 Python 3.11/3.12。当前验证机缺少 `libtk8.6.so`，因此 CLI、领域层、线程池和 SQLite 全部实测，Tkinter 面板只做导入隔离与代码审查，未在本机打开窗口。

最近一次完整执行结果：

- `python -m pytest -q`：29 passed。
- 工作负载固定种子 `20240601`；图摘要 `645df06b7eccba5d81856968e96f5bb281392450070a29dc1d57a3c7fbbf84a6`。
- 合法算法组合 36 个，显式上限 18，输出 18 个候选、18 个被上限剪枝候选和 2 个无等值谓词的顺序前缀。
- `orders` 480 行采用抽样口径（128 行）；`amount >= 10` 精确选择度 `0.9270833333`，抽样选择度 `0.9296875`；`status = 1` 精确 `0.3333333333`，抽样 `0.4140625`，差异逐项留档。
- 最优计划为 `[customers HASH->regions HASH->orders]`，成本 `1864865` 定点单位（缩放 1000，即 `1864.865`）；基线为 SQL 顺序 + 嵌套循环。
- 最优与基线原始连接输出均为 118 行、4 个分组，规范字节摘要一致；一次实测调度最大全局运行数为 2（上限 4）、每表不超过 1、最大预留内存 `35,028,992` 字节（预算 `67,108,864`）。
- 两个执行批次分别提交快照版本 1、2；最新恢复版本 2，重放校验通过；强制中断测试中第 2 批初始为 `UNKNOWN` 并要求重跑，恢复后两批均完整。
- 账本摘要篡改返回 `E_LEDGER`；自洽账本中的算子字段篡改返回 `E_REPLAY`，首差异定位到 `candidate_id + join:1:regions + estimate.operators[4].output_rows_fx`；基线丢行注入返回 `E_DIVERGE`，首差异分组键为 `[2]`、字段为 `total_amount`。

完整 JSON 报告在运行 `run_demo.bat` 后写入 `artifacts/demo_report.json`，其中包含估算、调度轨迹、账本提交、快照恢复和重放结果；精简指标见 `artifacts/verification_summary.json`。
