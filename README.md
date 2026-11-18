# 应急预案演练工作台

面向本地 Windows 的确定性应急预案演练工作台。项目只使用 Python 3.10+ 标准库，
不依赖网络、Docker 或第三方包。演练预案、事件注入、逻辑时钟、SQLite 账本、快照恢复和
历史重放校验均可在命令行/脚本中运行，Tkinter 界面作为本地操作台。

## 快速开始（Windows）

```powershell
py -3.10 --version
py -3.10 -c "import tkinter, sqlite3; print(tkinter.TkVersion, sqlite3.sqlite_version)"
py -3.10 check_env.py

cd <项目目录>
py -3.10 demo.py
py -3.10 -m unittest discover -s tests -v
py -3.10 run_ui.py
```

`demo.py` 默认清理并重建 `artifacts/demo-data`，顺序执行异常注入、正常演练、失败事件演练、
强制终止后恢复三个场景，并把稳定顺序的机器可读摘要写入
`artifacts/demo-data/summary.json`。结构化 JSONL 日志位于
`artifacts/demo-data/logs/drill.jsonl`。

本仓库的 Linux 验收主机没有安装 Tkinter，因此本次实际验证覆盖核心引擎、CLI、演示脚本和
测试；`run_ui.py` 在该主机上会明确提示 `E_UI` 降级。Windows 官方 Python 通常自带 Tkinter。

## 目录

```text
emergency_drill/
  plan.py           预案/事件强校验、环检测、版本提升
  engine.py         逻辑时钟状态机、事件触发、失败策略、中止、强杀点
  repository.py     SQLite 事务、当前代、账本、快照注册、整代替换
  workbench.py      预案刷新、快照、恢复、重放、UI 渲染模型
  cli.py            命令行入口
  ui.py             Tkinter 操作台
  logging_utils.py  带 clock_source=logical 的结构化 JSONL 日志
samples/
  plan_valid.json   5 阶段、3 个注入事件的正常样本
  plan_corrupt.json 损坏字段样本（E_PLAN）
  plan_cycle.json   前置引用成环样本（E_CYCLE）
  plan_valid_v2.json 用于预案整体刷新的版本 2 样本
tests/test_workbench.py
demo.py
run_ui.py
check_env.py        Windows 环境自检（Python/Tkinter/SQLite）
data/               SQLite、快照、日志的默认运行目录
artifacts/          演示运行产物
```

## 预案格式

```json
{
  "plan_id": "core-link-drill",
  "name": "核心链路应急预案演练",
  "version": 1,
  "failure_policy": "continue_downstream",
  "stages": [
    {
      "seq": 1,
      "name": "告警确认",
      "estimated_seconds": 2,
      "depends_on": [],
      "actions": ["ack_alert", "isolate_node"],
      "success_criteria": ["alert_acknowledged"],
      "failure_policy": "continue_downstream"
    }
  ],
  "events": [
    {
      "event_id": "evt-node-lost-1",
      "type": "node_lost",
      "offset_ms": 500,
      "stage_seq": 1
    }
  ]
}
```

强校验规则：

- `stages[].seq` 必须从 1 开始连续且唯一；`depends_on` 必须引用存在阶段，阶段图不得成环。
- `estimated_seconds`、`offset_ms` 必须是非负整数；布尔值不会被当作整数接受。
- `actions`、`success_criteria` 必须是非空字符串数组；阶段级 `failure_policy` 可覆盖预案默认值。
- 事件类型只接受 `node_lost`、`switchover_failed`、`validation_timeout`，`stage_seq` 必须存在。
- 任一错误都整体拒绝，不写会话、不改旧状态。错误带 `stage_seq` 与 `field_path`，并汇总全部
  校验错误；普通错误为 `E_PLAN`，依赖环使用专用业务码 `E_CYCLE`，其
  `details.reject_code` 仍为 `E_PLAN`，表示整体拒绝。

事件是本地模拟扰动。阶段动作中对应的缓解动作可消除影响：

| 事件 | 缓解动作 | 缺少缓解动作时 |
|---|---|---|
| `node_lost` | `isolate_node` | 阶段失败，`E_EVENT` |
| `switchover_failed` | `force_switchover` | 阶段失败，`E_EVENT` |
| `validation_timeout` | `retry_validation` | 阶段失败，`E_EVENT` |

## 状态机与逻辑时钟

常规状态为 `pending -> running -> succeeded|failed`，待执行阶段也可迁移到 `skipped`。
恢复时额外使用展示状态 `unknown`，它不是正常演练迁移，只表示“running 已提交但无终态”，
并要求重跑。

- 逻辑时钟从 0 开始，单位整数毫秒；阶段时长由 `estimated_seconds * 1000` 推进。
- 事件仅按 `offset_ms` 和当前逻辑时钟裁定，不使用墙钟；到期事件在目标阶段窗口内触发。
- 若事件到期时目标阶段尚未启动，则在阶段启动点触发；若目标阶段已结束，则记录
  `W_EVENT_LATE` 并标记为 `ignored`，不回溯改变终态。
- 阶段失败后读取该阶段 `failure_policy`。`skip_downstream` 将所有传递下游 pending 阶段迁移为
  `skipped`，记录 `E_STAGE`、失败上游序号与原因；`continue_downstream` 强制继续。
- 每次状态迁移和账本行在同一 SQLite `BEGIN IMMEDIATE` 事务中提交；阶段迁移序号按会话单调。
- 事件注入、事件触发/取消、演练开始/完成/中止、恢复和快照操作也写入同一账本。

## 版本与整代替换

每次预案编辑发布都自动把版本提升到会话历史最大值 + 1，并切换到新“代”（generation）。
当前视图中旧版本的阶段状态、事件、判定与账本被整体替换；旧代仍留在 SQLite 审计表中，
但 `stale=1` 或以旧 generation 隔离，不参与当前展示和统计。

`Workbench.publish_plan()` / UI“编辑预案”返回稳定差分：

- `added_stage_seqs`
- `removed_stage_seqs`
- `changed_stages: [{seq, fields}]`
- 事件新增、删除、变化

## 快照、恢复与重放

快照文件位于 `data/snapshots/`，保存预案版本与完整预案、事件注入序列、阶段状态机、
已完成阶段集合、当前逻辑时钟、账本轨迹与逐阶段账本摘要，并以规范化 JSON 的 SHA-256
作为 `content_digest`。任何内容被改动但摘要未同步都会以 `E_SNAPSHOT` 拒绝打开。

恢复快照：

- 先比较快照预案版本/预案摘要与当前会话；不一致时整代替换预案、事件和阶段状态，
  不与当前状态增量合并。
- 快照的账本轨迹以 `imported_from_snapshot` 标记导入新代，原账本仍保留审计。
- 旧快照始终保留，注册表标记 `read_only=1`；UI 显示“只读历史”。

强制终止恢复：

- 演示脚本用子进程在提交下一阶段 `running` 后调用 `os._exit(137)`。
- 重启后仅依据完整提交的账本迁移恢复；最后仍为 `running` 的阶段标为 `unknown` 并要求重跑。
- 不完整账本尾行被隔离并丢弃，同时追加 `incomplete_tail_discarded` 审计行；不计入迁移和状态恢复。
- 恢复报告列出已恢复阶段数、未知阶段序号、待执行阶段和不完整尾行的处理方式，并明确
  `exit_report_available: false`，不声称生成强制终止退出报告。

历史重放：

- 校验快照摘要后，按快照预案、事件注入序列和逻辑时钟从零重放。
- 先比较当前预案版本、预案摘要、事件序列和上游阶段状态；不一致直接 `E_REPLAY`。
- 再逐阶段比较 `status`、`skip_reason`、`migration_count`；首个差异定位到
  `stage_states.<seq>.<field>`，不以“大致一致”替代。

## 命令行

```powershell
# 校验
py -3.10 -m emergency_drill.cli validate --plan samples/plan_valid.json

# 创建会话并运行
py -3.10 -m emergency_drill.cli init --session s1 --plan samples/plan_valid.json
py -3.10 -m emergency_drill.cli run --session s1

# 注入失败事件后运行
py -3.10 -m emergency_drill.cli inject --session s1 --event-id fail3 `
  --type switchover_failed --offset-ms 6000 --stage-seq 3
py -3.10 -m emergency_drill.cli run --session s1

# 快照、恢复、重放
py -3.10 -m emergency_drill.cli snapshot --session s1
py -3.10 -m emergency_drill.cli restore --session s1 --snapshot data/snapshots/s1-snap-0001.json
py -3.10 -m emergency_drill.cli replay --session s1 --snapshot data/snapshots/s1-snap-0001.json

# 强制终止子进程后恢复（137 为进程退出码）
py -3.10 -m emergency_drill.cli run --session s1 --crash-after-stage 2 --hard-crash
py -3.10 -m emergency_drill.cli recover --session s1
py -3.10 -m emergency_drill.cli resume --session s1
```

命令默认使用 `data/session.db`、`data/snapshots`、`data/logs/drill.jsonl`；每个子命令都接受
`--db`、`--snapshots-dir`、`--log`。业务错误输出稳定 JSON 并返回：
`E_PLAN=2`、`E_CYCLE=3`、`E_SNAPSHOT=4`、`E_REPLAY=5`、`E_LEDGER=6`、`E_STAGE=7`。

## Tkinter 操作台

```powershell
py -3.10 run_ui.py --session ui-session --plan samples/plan_valid.json
```

界面稳定展示：

- 当前预案名称/版本、generation、会话状态、逻辑时钟与 `clock_source=logical`
- 阶段序号、名称、依赖、状态、跳过原因、业务码、触发上游、未知标记
- 按逻辑偏移排序的事件时间轴、事件状态和来源
- 快照版本、逻辑时钟、内容摘要及“只读历史/当前写入”标记

按钮支持加载/编辑预案（自动升版）、注入事件、开始/中止演练、保存/整体恢复快照、
崩溃恢复和重放校验。仅 UI 阶段回调使用短暂展示延迟；所有状态判定仍只读取逻辑时钟。

## 测试与异常注入

```powershell
py -3.10 -m unittest discover -s tests -v
py -3.10 demo.py
py -3.10 demo.py --fault plan-corrupt --fault cycle --fault snapshot-digest
py -3.10 demo.py --skip-faults
```

异常注入只替换输入文件或参数，不改源码、不伪造账本：损坏预案与成环预案直接读取样本；
快照摘要异常通过修改快照副本内容而保留旧摘要产生。日志字段包含 `business_code`、
`stage_seq`、`field_path`、`logical_clock_ms` 与 `clock_source: "logical"`。

实际验证结果见 `docs/VERIFICATION.md` 和 `artifacts/demo-data/summary.json`。
