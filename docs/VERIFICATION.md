# 实际验证记录

验证日期：2026-11-21（Asia/Shanghai）

## 环境

```text
本机解释器: Python 3.11.0rc1
sqlite3: 3.37.2
tkinter: 当前 Linux 发行层缺失（ModuleNotFoundError）
```

由于当前 Linux Python 未编译 Tk，界面未进行窗口级点击测试；`deploy_workbench/gui.py` 已通过编译检查。目标 Windows 官方 Python 默认带 Tk，启动入口为 `run_workbench.cmd gui`。

## 自动化测试

执行：

```text
python3 -m unittest discover -s tests -v
```

结果：

```text
Ran 29 tests in 3.400s
OK
```

覆盖包括：

- 非法层级、重复 ID、非法数值类型、空键公式、缺失公式键均整体拒绝并带环境/路径。
- 缺失绑定源为 `E_BIND`，坏 JSON 为 `E_ENV`，非法缩放因子为 `E_GRID`。
- 环境、源、任务列表倒序后来源链与结果逐字节一致。
- `dev/headroom` 链为 `rt_dev(不满足绑定) -> file_dev(选中) -> base`。
- 邻近键比较在 `2/S` 合成界内使用键 ID 稳定决胜并标注。
- 版本变化后当前视图清空，旧 run 只读，新 run 整体替换。
- 强制中断模拟只恢复 dev 已提交记录，`staging`/`prod` 标为 `unknown`。
- 快照重放一致；篡改首个键后定位到 `dev/cpu_cores/effective_fixed` 并返回 `E_REPLAY`。

## 样本验收摘要

执行：

```text
python3 scripts/verify_acceptance.py
```

机器可读结果写入 `artifacts/acceptance_report.json`。

确定性裁决：

```text
环境数: 3
键数: 18
受误差界影响的比较: 1
dev/headroom 链: rt_dev(false) -> file_dev(true) -> base(false)
```

首轮版本与快照：

```text
run_id: 1
environment_version: 1
priority_version: 1
scale_version: 1
snapshot replay: verified
```

部分定点任务结果（所有 `actual_error <= error_bound`）：

| 环境 | 任务 | 定点结果 | 误差界 | 实际误差 | 主导段 |
| --- | --- | ---: | ---: | ---: | --- |
| dev | capacity | 14497 | 0.00291695 | 0.00291665 | formula_truncation |
| dev | floor_margin | -1375 | 0.0007 | 0.0003 | source_quantization |
| staging | capacity | 22994 | 0.00561695 | 0.00561665 | formula_truncation |
| prod | capacity | 39490 | 0.00979195 | 0.00979165 | formula_truncation |
| prod | latency_budget | 9875 | 0 | 0 | none |

版本整体替换：

```text
修改 dev/file_dev/headroom 后:
environment_version 1 -> 2
affected_environment_count: 1
affected_key_count: 4
旧 run stale=true, read_only=true
新 run 完成并写入新快照
```

重放差异定位：

```text
code: E_REPLAY
environment: dev
key: cpu_cores
field: effective_fixed
```

三类注入：

| 注入 | 业务码 | 退出码 | 路径样例 | JSONL 日志 |
| --- | --- | ---: | --- | --- |
| `--inject-env` | `E_ENV` | 2 | `$` | 有 |
| `--inject-bind` | `E_BIND` | 3 | `$.environments[0].bindings.cpu_cores.allowed_source_ids[0]` | 有 |
| `--inject-grid` | `E_GRID` | 4 | `$.scale_power` | 有 |

## 说明

- 上述记录由本地确定性样本生成，未访问网络或未授权目标。
- SQLite、日志、注入文件和快照均在命令行指定的工作区目录内创建。
- 强制中断恢复使用真实 SQLite 状态模拟“计算中未提交”边界；恢复报告明确不声称存在强制退出报告。
