# 实际验证记录

验证日期：2026-11-22（Asia/Shanghai）。所有样本、SQLite 和日志均位于本地工作区。

## 环境

| 项目 | 实测值 | 结论 |
|---|---|---|
| 执行平台 | WSL2 Linux 容器 | 目标为 Windows；代码仅使用 Python 可移植接口 |
| Python | 3.11.0rc1 | 满足 Python >= 3.11 |
| sqlite3 | 3.37.2 | 可用 |
| NumPy | 1.26.4 | 满足 >= 1.26 |
| tkinter | 不可用（系统缺少 `python3-tk`，当前账户无免密安装权限） | GUI 代码与视图数据测试通过，本容器无法实际打开窗口 |

`python -m alert_pipeline diagnose` 对 tkinter 返回
`tkinter_available=false`，进程退出码 10。Windows 官方 Python 安装并启用
Tcl/Tk 后应显示 `true`。

## 自动化测试

```text
python3 -m unittest discover -v
Ran 32 tests in 5.567s
OK
```

覆盖范围：

- v1/v2/v3 归一及 v1/v3 同核字段一致、默认补齐 provenance；
- 未知版本、缺字段、负窗口、悬空根因引用 `E_FIELD`；
- 重复服务、悬空依赖、有环、时间回退、非法级别/指纹、坏 JSON `E_SPEC`；
- 静默窗口边界、跨批次状态、到达顺序无关；
- 上下游候选、路径、稳定拓扑和其它输入顺序无关性；
- 全局/每服务双配额、排队时钟、峰值不超限；
- 并行与串行结论逐字节一致；
- 声明版本单调、整体替换、旧聚合/账本/批次陈旧；
- 强制终止、unknown 批次、已提交前缀重放、恢复续跑；
- 账本篡改 `E_REPLAY`、验证篡改 `E_VERIFY`；
- 三类 CLI 注入与结构化日志；
- 固定 DSN/GUI 视图所需的稳定行排序。

完整测试输出：`artifacts/test-run.log`。

## 基础样本 v1

样本包含 40 条混存告警，来源版本为 v1、v2、v3。manifest 声明版本 1，默认窗口
30000 ms，批大小 8，全局并发 4，每服务并发 1。

```text
business_code            OK
alert_count              40
aggregate_count          25
ledger_count             25
batch_count              5
suppressed_total         15
conclusion_digest        4a4d9a30d583f9654f34050d519b0347ad3efb059a2e4e28cd237e8ac7edf0eb
verification_digest      3ec7f79d89c7406203a3f1b3564543c1218bc228175d077db0cf016f6decc18e
最大批次全局峰值          2（上限 4）
最大每服务峰值            1（上限 1）
```

独立验证结果：`business_code=OK`，检查 40 条告警、25 条聚合、25 条账本、5 个
批次；来源版本集合为 `[1,2,3]`；首个不一致为空。

并行与串行各自使用独立 SQLite 和输出目录，导出的
`conclusions.canonical.json` 完全相同：

```text
parallel SHA-256  233e1defa90b63bcd70834dd844e7879e9f82082d71efee6163bc793708dfb6c
serial   SHA-256  233e1defa90b63bcd70834dd844e7879e9f82082d71efee6163bc793708dfb6c
byte_identical    true
```

证据文件：`artifacts/final-v1-run.json`、`artifacts/final-v1-verify.json`、
`artifacts/final-v1-check.json`。

## 演进样本 v2

在同一个 SQLite 中声明版本 1 → 2，同时变更默认静默窗口 30000 → 45000 ms、
依赖图版本 3 → 4、规则版本 2 → 3，并新增 `search` 服务。

```text
business_code            OK
aggregate_count          22
ledger_count             22
suppressed_total         18
replacement affected     25
conclusion_digest        f7cf7797ffa6b085662bb4e3b07c95b68a25c2d69e772be8be6a3004326d56a2
verification_digest      6abd27f31758baad516543ae4252eb54a0c49b2e7522f21fb1e9055f599d0569
```

代际与陈旧隔离：

```text
generation 1  version=1  complete  current=0  stale=1
generation 2  version=2  complete  current=1  stale=0

旧代际陈旧行      25 aggregates / 25 ledger / 5 batches
当前视图行        22 aggregates / 22 ledger / 5 batches
```

替换报告稳定列出 25 个受影响告警及前后结论。配置变化字段：

```text
default_silence_window_ms, graph_version, rules_digest, rules_version, services_digest
```

证据文件：`artifacts/final-v2-run.json`、`artifacts/final-v2-verify.json`、
`artifacts/final-v2-summary.json`。

## 强制终止、恢复与续跑

在第 3 个批次（索引 2）提交前执行强制终止，进程退出码为 91。此时：

```text
已提交批次        0, 1
提交游标          16
unknown 批次      2（start_cursor=16, end_cursor=24）
代际状态          building，非 current
```

`recover` 输出：

```text
business_code              OK
requires_rerun             true
exit_report_available      false
message                    uncommitted batch attempts are unknown and must be rerun
```

已提交前缀 0、1 已按声明版本、种子、原始文件摘要和批次快照逐项重放通过。随后
同一命令续跑：

```text
attempt 3 / batch 2   superseded
attempt 4 / batch 2   committed
attempt 5 / batch 3   committed
attempt 6 / batch 4   committed
alert_count           40
aggregate_count       25
ledger_count          25
conclusion_digest     4a4d9a30d583f9654f34050d519b0347ad3efb059a2e4e28cd237e8ac7edf0eb
```

续跑后的结论文件与无中断 v1 运行逐字节一致：

```text
uninterrupted SHA-256  233e1defa90b63bcd70834dd844e7879e9f82082d71efee6163bc793708dfb6c
resumed       SHA-256  233e1defa90b63bcd70834dd844e7879e9f82082d71efee6163bc793708dfb6c
byte_identical         true
```

证据文件：`artifacts/final-recover.json`、`artifacts/final-resume.json`、
`artifacts/final-crash-summary.json`。

## 重放与验证篡改

对最终数据库副本篡改一个批次的 `suppressed_count`：

```text
business_code  E_REPLAY
exit_code      6
alert_id       A-2d1dbb04acc1b527130b
field          aggregate.suppressed_count
batch_index    0
basis          declared version, seed, source streams and committed batch state
```

对另一副本篡改当前账本的候选服务：

```text
business_code  E_VERIFY
exit_code      7
alert_id       A-1929c81e60bd42b3f43f
field          ledger.candidate_services
evidence       raw graph/rule recomputation differs from the correlation ledger
```

证据文件：`artifacts/final-tamper-summary.json`。

## 异常注入

| 注入参数 | 业务码 | 退出码 | 首个位置 |
|---|---|---:|---|
| `--inject corrupt` | `E_SPEC` | 2 | line=41, field=`$` |
| `--inject unsupported-version` | `E_FIELD` | 3 | line=1, field=`version` |
| `--inject window-conflict` | `E_WINDOW` | 4 | line=1, field=`silence_window_ms` |

三项均写入结构化日志，最后一条事件的 `event=error`、`business_code` 与 CLI 输出
一致。证据文件：`artifacts/final-injection-summary.json`、
`logs/final-*.jsonl`。

## Tkinter 限制

本容器无法导入 tkinter：

```text
ModuleNotFoundError: No module named 'tkinter'
```

尝试安装 `python3-tk` 时因当前账户无免密 sudo 权限被拒绝。因此本次没有声称完成
GUI 窗口级实测。已完成：

- `gui.py` 的四个视图（告警、账本、时间线、批次）与稳定排序逻辑；
- `_load_view` 集成测试：连续两次读取行完全一致，告警行包含来源版本和候选路径；
- 无 tkinter 时返回明确 `E_STATE` 诊断，而不是静默失败。

在 Windows 官方 Python 中启用 Tcl/Tk 后，可直接运行 README 中的 `gui` 命令做
窗口级验收。
