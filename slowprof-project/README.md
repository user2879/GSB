# slowprof — 本地慢请求性能剖析与热点定位组件

面向 Windows 本地环境、纯标准库实现（NumPy 仅用于可选数值交叉校验）的可集成
Python 包。输入为按请求聚合的确定性 NDJSON 调用栈样本，输出同栈折叠、自/累计
耗时、稳定热点排名、调用上下文、定点占比与量化误差，并通过本地 SQLite 账本与
版本化快照支持强制终止恢复和历史重放校验。

## 1. 环境准备

- Python 3.11+；确认 `python -c "import sqlite3"` 可用。
- Tkinter：标准 Windows 安装包自带，用 `python -m tkinter` 应能弹窗。
  若解释器无 tkinter（如本项目所在的精简 Linux 沙箱），诊断命令自动回退到
  同一数据模型的只读文本渲染，业务逻辑不受影响。
- 可选：`pip install -r requirements.txt` 安装 NumPy 用于大规模数值校验。
- 不使用 Docker 或任何容器；样本全部本地确定性生成，不访问外部目标。

## 2. 输入契约（NDJSON，每行一个请求聚合样本）

| 字段 | 类型 | 约束 |
| --- | --- | --- |
| `request_id` | string | 非空 |
| `timestamp_ms` | int | 整数毫秒；同一 `service_id` 内单调不降 |
| `duration_ms` | int | 正整数毫秒 |
| `service_id` | string | 非空 |
| `weight` | int | 正整数采样权重 |
| `frames` | list[string] | 非空；每项匹配 `svc.method`，段名非空 |

帧名正则：`^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$`。栈按根→叶顺序书写，最后一个
帧为栈顶（leaf）。任何非法行以 `E_SAMPLE` 报告，携带物理文件行号与字段名。

## 3. 计算口径声明

- **同栈折叠**：完全相同的帧序列合并为一条，权重相加；结果与录入顺序无关。
- **自耗时（self）**：仅栈顶帧计入该样本权重；非栈顶帧自耗时为 0。
- **累计耗时（cumulative）**：帧只要出现在栈中即计入该样本权重；同一栈内帧
  重复出现也只计一次。
- **总权重**：观测窗口内全部样本权重之和，作为占比分母。
- 规范化输出见 `FoldResult.canonical_bytes()`，两条聚合路径对同一输入逐字节
  一致。

## 4. 双聚合路径与显式阈值

`select_path(unique_stacks, total_weight)` 的边界（默认值可通过参数覆盖）：

- `max_unique_stacks = 5000`
- `max_total_weight = 200000`

两者都**严格小于**阈值走 `full`（全量物化后折叠）；任一指标**达到或超过**
阈值走 `streaming`（单遍流式，仅保留折叠表和帧映射，内存随基数而非行数增长）。
输出中 `path_reason` 给出实际指标、阈值和选择理由。`cross-check` 与
`force_path` 可强制两路径并比较规范化字节。

## 5. 排名、上下文与定点误差

- 排名总序：`(-self_weight, -cumulative_weight, frame 字典序升序)`，取前 K。
- 调用上下文：基于折叠后的栈路径聚合直接调用者/被调用者的边权重与路径数，
  边按 `(-edge_weight, -call_count, counterpart)` 稳定排序。
- 定点：缩放因子 `S`（正整数，默认 10000），`q = floor(weight*S/total)`。
  单帧截断误差上界 `1/S`；前 K 集合合计误差上界 `K/S`。
- 真实 Top-K 成员由 `fractions.Fraction` 精确有理比较决定，定点值仅用于展示；
  当两帧定点值之差 `<= 1`（落在合成误差界 `1/S` 内）标注 `[ERROR-AFFECTED]`，
  排序仍由精确算术的稳定决胜规则决定。

## 6. 账本、恢复与快照

- SQLite 账本带整数 `ledger_format_version`/`schema_version` 与内容摘要。
- 每个样本批次在**同一事务**内提交批次游标和折叠增量；强制终止后只承认完整
  提交的批次，未提交尾部样本标记为 unknown，并从 `last_committed_line + 1` 重跑。
- 快照为单 JSON + SHA-256 侧车，记录样本摘要、区间参数版本、路径选择与全部
  结论。重开时样本摘要或区间版本不一致即**整体替换**，不做增量合并。
- 重放按记录的区间与路径重新加载样本重算，首个不一致的帧名/字段/期望值以
  `E_REPLAY` 报告。

## 7. 快速开始（PowerShell 或 cmd）

```bat
python -m slowprof.cli generate --rows 5000 --output data\samples.ndjson
python -m slowprof.cli cross-check --samples data\samples.ndjson
python -m slowprof.cli profile --samples data\samples.ndjson ^
  --ledger state\ledger.sqlite --snapshot-dir state\snap --print-report
python -m slowprof.cli diagnose --samples data\samples.ndjson ^
  --ledger state\ledger.sqlite --snapshot-dir state\snap
python -m slowprof.cli replay --samples data\samples.ndjson ^
  --ledger state\ledger.sqlite --snapshot-dir state\snap
run_demo.bat
run_tests.bat
```

Linux/macOS 等价命令使用 `python -m slowprof.cli ...`，演示为
`python scripts/demo.py`（加 `--quick` 跳过 15 万行规模段）。

## 8. 异常注入（只改输入或参数，不改源码）

```bat
python -m slowprof.cli inject-sample --samples data\samples.ndjson ^
  --output data\corrupt.ndjson --kind bad_weight --target-line 17
python -m slowprof.cli profile --samples data\corrupt.ndjson ^
  --ledger state\bad.sqlite --snapshot-dir state\bad_snap
REM -> {"code": "E_SAMPLE", "field": "weight", "line_number": 17, ...}

python -m slowprof.cli profile --samples data\samples.ndjson ^
  --ledger state\r.sqlite --snapshot-dir state\r_snap ^
  --start-ms 500 --end-ms 100
REM -> {"code": "E_RANGE", "field": "start_ms", ...}
```

注入类型：`bad_json`、`bad_weight`、`bad_duration`、`bad_frame`、
`missing_field`、`timestamp_regression`。

## 9. 错误码

| 码 | 触发场景 |
| --- | --- |
| `E_SAMPLE` | NDJSON 行解析/字段/单调性校验失败（含行号、字段名） |
| `E_RANGE` | 区间参数或缩放因子等会话参数非法 |
| `E_LEDGER` | 账本缺失、绑定冲突或快照摘要损坏 |
| `E_REPLAY` | 重放重算与持久化结论首个不一致项 |

## 10. 目录结构

```
slowprof/
  errors.py       错误码与 ReplayMismatch
  samples.py      NDJSON 强校验、区间切分、摘要
  folding.py      full / streaming 双引擎与规范化字节
  ranking.py      稳定排名与调用上下文
  fixedpoint.py   定点量化、误差界、真实 Top-K 判定
  ledger.py       SQLite 批次事务与恢复游标
  snapshot.py     版本化快照整体替换与重放比对
  session.py      端到端编排、强制终止与 resume
  diagnostics.py  Tkinter 窗口 + 只读文本渲染
  generator.py    确定性样本生成器
  cli.py          命令行入口（含注入子命令）
scripts/demo.py   可复现端到端演示
tests/            33 个 unittest（含 NumPy 交叉校验）
```
