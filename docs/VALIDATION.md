# 实际验证记录

## 环境

本次验证在当前工作区实际执行：

- Python `3.11.0rc1`
- SQLite `3.37.2`
- NumPy `2.4.6`
- pytest `8.4.2`
- 虚拟环境：`.venv`
- 未使用 Docker、容器、外部数据集或网络端口

当前执行环境为 WSL/Linux。已提供并静态检查 Windows CMD 脚本，但 WSL 未启用 Windows 可执行互操作；直接执行 `/mnt/c/Windows/System32/cmd.exe` 返回 `Exec format error`，因此 `scripts\*.cmd` 的 Windows 进程级运行需要在目标 Windows 主机上执行。Python 代码与仓库内命令在 Linux 上完整运行通过。

## 自动化测试

命令：

```bash
.venv/bin/python -m pytest -q
```

结果：

```text
......................                                                   [100%]
24 passed
```

覆盖项：

- 表/列唯一性、类型、空值位图、数值/空值一致性、分组/聚合列、过滤列与 JSON 语法错误的位置。
- 表声明摘要不一致的 `E_DIGEST`。
- WHERE 空白、括号、AND 顺序与聚合顺序规范化后相同 `query_id` 和计划摘要。
- 独立参考实现与小规模聚合结果对照。
- 并行度 1/3/4、拆分阈值 1/4/1000、拆分份数 2/4 下结果文件逐字节一致。
- 倾斜分片拆分区间连续、覆盖父区间、父任务不参与合并，预算事件不超限。
- 紧预算运行与正常预算运行结果逐字节一致。
- SQLite 任务、倾斜判定、预算事件和批次中间状态持久化。
- 快照恢复重放、数据版本变化后的整体替换审计。
- 篡改账本后 `E_REPLAY` 定位首个分片与 `rows_completed` 字段。
- `E_SKEW`、`E_BUDGET`、`E_DIGEST` 注入及结构化日志。

## 三场景演示

命令：

```bash
.venv/bin/python scripts/run_demo.py
```

固定顺序：

| 顺序 | 场景 | 数据 | 并行度 | 预算 | 阈值/份数 | 任务数 | 拆分数 | 结果 SHA-256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 无倾斜小规模 | `small.json` | 2 | 2048 | 24/2 | 6 | 0 | `dd33374b578967f7e5f677e786acc6ebea2d110172f42805cab8cf3625720e1a` |
| 2 | 倾斜放大 | `skew_amplified.json` | 3 | 4096 | 8/4 | 16 | 3（12 子片） | `a09703bf0464f814a0647b308731dda7d257ad9c395fdb8e8ee97b6faf915e84` |
| 3 | 预算收紧 | `skew_amplified.json` | 1 | 900 | 8/4 | 16 | 3（12 子片） | `a09703bf0464f814a0647b308731dda7d257ad9c395fdb8e8ee97b6faf915e84` |

场景 2 与场景 3 的 `final_result.json` 逐字节相同。三个场景均完成分片级重放，重放结果摘要与运行结果摘要一致；场景 3 观察到预算排队。

稳定演示摘要写入：

```text
var/demo/demo_summary.json
```

关键检查结果：

```json
{
  "all_replays_ok": true,
  "all_runs_ok": true,
  "paths_recreated_under_var_demo": true,
  "ports_used": [],
  "skew_and_tight_budget_result_bytes_identical": true,
  "skew_split_observed": true,
  "tight_budget_queue_observed": true
}
```

## 异常业务码校验

命令：

```bash
.venv/bin/python scripts/run_anomaly_checks.py
```

实际结果：

| 注入 | 业务码 | 退出码 | 定位 |
| --- | --- | --- | --- |
| 倾斜放大超安全上限 | `E_SKEW` | 20 | `injection.skew` |
| 预算持续不足 | `E_BUDGET` | 30 | 首个无法调度任务 |
| 分片摘要不一致 | `E_DIGEST` | 40 | 首个分片 ID |
| 篡改账本后重放 | `E_REPLAY` | 50 | 分片 ID + `rows_completed` |

本次 `E_REPLAY` 定位：

```json
{
  "task_id": "sales_a@v1:r000000000000-000000000016",
  "field": "rows_completed",
  "expected": 11,
  "actual": 10
}
```

稳定异常摘要与结构化日志：

```text
var/anomaly/anomaly_summary.json
var/anomaly/pipeline.jsonl
```

## 手工验收补充

- 倾斜样本中 3 个基础热点分片各拆为 4 个连续子区间；父/子范围首尾相接、无重叠。
- 正常预算轨迹峰值为 2304 字节（预算 4096）；所有预算事件均满足 `used_bytes <= budget_bytes`。
- 紧预算场景在并行度 1 下完成，轨迹包含真实启动、完成和排队记录，没有线程超时终止。
- 快照与当前数据版本不一致时，`resume` 返回新运行标识、`replaced=true`，比较原因包含 `data_digest/table_digests/table_versions`，并在 `snapshot_replacements` 写入整体替换记录。
- 账本分片完成行数被篡改后，重放在任何最终结果比较前定位首个任务字段并返回 `E_REPLAY`。
