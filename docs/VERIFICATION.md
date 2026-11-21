# 验证记录

## 环境

- Python：3.11.0rc1（目标要求 3.11+）。
- NumPy：1.26.4。
- SQLite：Python 标准库 `sqlite3`。
- 本轮执行环境：Linux/WSL2。当前 Linux Python 未安装 Tkinter；Windows 官方 CPython 安装包自带 Tkinter。Tkinter 模块只在实际启动 UI 时导入，不影响核心与测试。
- 未使用 Docker、容器、外部网络、账号或远程数据。

## 自动化测试

命令：

```powershell
py -3.11 -m unittest discover -s tests -v
```

本轮结果：`Ran 34 tests`，`OK`。覆盖：

- 声明唯一性、正整数、比例上限、聚合函数、首个违规字段、非法缩放因子、确定性摘要。
- 向零截断、64 位饱和、三段误差、均值界、主导段阈值。
- 极大定点值导致求和饱和时仍可写入 SQLite 且不溢出。
- 水位前后值、窗口定稿不可变、预算等号边界唯一接纳、随后拒绝。
- 打乱事件列表后的规范到达序归一、歧义稳定分组、规则版本陈旧隔离。
- 快照内容摘要与账本批次链篡改 `E_SNAPSHOT`、待定窗口恢复为 `UNKNOWN`、重放首差异 `E_REPLAY`、缩放因子清单差异。
- 四类 CLI 注入进程退出码与结构化日志业务码。

## 三场景演示

命令：

```powershell
py -3.11 -m streamagg.cli demo
```

本轮结果：`all_passed=true`，`business_code=OK_DEMO`。

| 场景 | 子进程退出码 | 业务码 | 关键结果 |
|---|---:|---|---|
| 1 有序定稿 | 0 | `OK_SCENARIO_1_ORDERED` | 36 事件、15 个定稿窗口、0 丢弃；源摘要可复现；账本重放一致 |
| 2 乱序与预算 | 0 | `OK_SCENARIO_2_LATE_BUDGET` | 420 事件、28 个定稿窗口、78 丢弃、26 次迟到接纳；同时出现条数耗尽、量耗尽、窗口已定稿三类原因 |
| 3 版本替换/崩溃/恢复 | 0 | `OK_SCENARIO_3_REPLACE_SNAPSHOT_REPLAY` | v1→v2 单调升级；旧视图陈旧；真实 `os._exit(17)` 后恢复；1 个待定窗口标记 `UNKNOWN`；重放一致 |

场景 2 的丢弃原因实测：

```text
LATE_BUDGET_AMOUNT_EXCEEDED: 2
LATE_BUDGET_COUNT_EXHAUSTED: 2
WINDOW_ALREADY_FINALIZED: 74
```

场景 2 的迟到接纳：26 条，均为 `LATE_WITHIN_BUDGET`。误差主导段分布中 15 个窗口为 `LATE_ADMISSION`，13 个为 `NONE`。歧义对 63 对。

## 异常注入

| 注入 | 期望业务码 | 实际业务码 | 实际退出码 | 结果 |
|---|---|---:|---:|---|
| 事件时间大幅倒退 | `E_ORDER` | `E_ORDER` | 3 | 通过 |
| 非法缩放因子 | `E_GRID` | `E_GRID` | 4 | 通过 |
| 账本批次内容摘要不符 | `E_SNAPSHOT` | `E_SNAPSHOT` | 5 | 通过 |
| 账本窗口结果篡改 | `E_REPLAY` | `E_REPLAY` | 6 | 通过 |

结构化日志位于 `artifacts/demo/logs/injection_*.jsonl`。

## 产物位置

- 总报告：`artifacts/demo/reports/demo_report.json`
- 场景报告：`artifacts/demo/reports/scenario1.json`、`scenario2.json`、`scenario3.json`
- SQLite 账本：`artifacts/demo/ledger.sqlite3`
- 快照：`artifacts/demo/snapshots/`
- 结构化日志：`artifacts/demo/logs/`
- 崩溃写入器独立进程退出码：17（见 `scenario3.json` 的 `forced_termination.crash_writer_exit_code`）

## 诊断界面

Windows 检查：

```powershell
py -3.11 -m tkinter
py -3.11 run_ui.py --ledger artifacts/demo/ledger.sqlite3
```

界面刷新使用 SQLite 只读 URI 与 `query_only=ON`，不修改账本。

本轮 Linux/WSL Python 未安装 Tkinter，因此只验证了只读数据装载与启动失败提示；核心引擎、账本和测试不导入 Tkinter。Windows 官方 CPython 安装包可按上面的 `python -m tkinter` 检查并直接启动界面。
