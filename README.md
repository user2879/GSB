# faultdiag — 本地故障域隔离与恢复诊断组件

面向 Windows 本地环境(同时兼容 Linux/macOS)的可集成 Python 包:以故障域
标记为核心,从本地 JSON 声明加载服务实例与故障域,按 NDJSON 故障事件流
推导健康与隔离状态,通过 127.0.0.1 回环 TCP 的 JSON 行协议向每个实例的
确定性模拟服务子进程下发隔离/恢复指令,并将全部决策与回执持久化到本地
SQLite 账本,支持快照整体替换恢复与恢复后逐事件重放校验。

- 运行要求: Python 3.10+,Tkinter(标准库自带,仅 UI 需要);无外部网络、
  账号、容器依赖。NumPy 非必需(演示环境已安装)。
- 原始任务书见 `docs/TASK.md`。

## 目录结构

- `faultdiag/spec.py` — 实例/故障域/事件流加载与强校验(E_SPEC)
- `faultdiag/state.py` — 实例健康与故障域隔离状态机
- `faultdiag/dispatch.py` — 并发受限的指令调度与子进程管理
  (E_PORT / E_HANDSHAKE / E_REPLY)
- `faultdiag/mock_server.py` — 确定性模拟服务子进程(可独立运行)
- `faultdiag/converge.py` — 回执与域级结论逐字节一致性判定(E_CONVERGE)
- `faultdiag/ledger.py` — SQLite 账本(格式版本 + 内容摘要 + 事务批次)
- `faultdiag/snapshot.py` — 会话快照版本识别与整体替换(E_SNAPSHOT)
- `faultdiag/replay.py` — 恢复后逐事件重放校验(E_REPLAY)
- `faultdiag/runner.py` / `cli.py` / `ui.py` — 编排、命令行、Tkinter 界面
- `samples/` — 演示输入(实例、故障域、NDJSON 事件流)
- `tests/` — 41 个 pytest 用例

## 快速开始

```bash
# 完整演示运行(决策 + 指令下发 + 账本 + 快照)
python -m faultdiag.cli run --instances samples/instances.json \
    --domains samples/domains.json --events samples/events.ndjson \
    --workdir demo/work

# 恢复后一致性重放校验
python -m faultdiag.cli replay --instances samples/instances.json \
    --domains samples/domains.json --events samples/events.ndjson \
    --workdir demo/work

# 异常注入(分别以 E_EVENT / E_LEVEL / E_SNAPSHOT 报告并写结构化日志)
python -m faultdiag.cli run ... --inject event
python -m faultdiag.cli run ... --inject level
python -m faultdiag.cli run ... --inject snapshot

# Tkinter 诊断界面(只读, 刷新不改账本)
python -m faultdiag.cli ui --workdir demo/work

# 测试
python -m pytest tests/ -q
```

每次命令向 stdout 输出一行结构化 JSON 结论;业务失败退出码为 2。

## 关键语义

- **声明校验(E_SPEC)**: 实例标识唯一、故障域引用存在、健康权重非负整数、
  副本序号正整数;域标识唯一、隔离等级 ∈ {0,1,2,3}、恢复冷却非负整数;
  事件序号连续唯一(1..N)、时间单调不减、实例存在、故障类型 ∈
  {degrade, crash, flap, recover}。任一不合法整体拒绝,`details` 给出首个
  出错行与字段。
- **状态机**: 实例 unknown/healthy/degraded/unhealthy;域
  normal/isolating/isolated/recovering。隔离阈值按等级推导:0 不隔离、
  1 任意不健康权重、2 不健康权重过半、3 全部权重。冷却窗口以事件流时间戳
  度量,未满不得转 recovering。每次迁移记录触发事件序号、原因与单调时钟
  毫秒。
- **调度**: 全局并发上限 + 每域配额,排队指令记录 enqueue/start/end 单调
  毫秒;握手与下发超时仅来自真实套接字超时;失败重试一次,重试回执以
  `attempt=2` 与首次区分记录。端口按实例排序序号确定分配
  (base_port + index),占用即 E_PORT。
- **收敛(E_CONVERGE)**: 按事件流与域状态重算每实例期望隔离状态,聚合成
  域级结论,与回执视图规范化后逐字节比对;不一致、未回执、序号乱序时
  稳定列出偏差实例、域、期望/实际摘要。
- **账本**: 每个事件批次在同一 SQLite 事务写入迁移记录、指令回执摘要、
  逐事件状态与进度游标;携带整数格式版本与内容摘要。强制终止时不生成
  退出报告;重启只依据已完整提交的记录恢复,未随批次提交的指令恢复时
  标记为 `unknown` 并要求重跑。
- **快照(E_SNAPSHOT)**: 持久化实例集合版本、故障域版本、事件流摘要、
  域/实例状态与账本摘要,并携带自身内容摘要。摘要损坏即 E_SNAPSHOT;
  版本或摘要与当前状态不一致时整体替换(非增量合并),旧快照保留为只读
  `.bak.json` 备份,并稳定列出被替换(状态发生变化)的域与实例。
- **重放(E_REPLAY)**: 从快照/账本恢复后按游标重放同一事件序列,逐事件
  比对实例健康、域隔离状态、期望隔离结论与回执摘要,差异定位到首个不
  一致的事件序号、实例与字段;故障域版本或事件流摘要不一致时标注为
  不可复现并说明原因。

## 实际验证结果(本机, Python 3.11)

- `python -m pytest tests/ -q` → **41 passed**(规格校验、状态机、并发
  调度、真实套接字超时、收敛判定、账本事务与恢复、快照替换、重放定位、
  端到端、三类注入)。
- 样本端到端: 10 事件、22 次迁移、8 条指令全部收敛;dom-a 在 seq1/6
  进入 isolating→isolated,seq7(ts=700)恢复时冷却 300ms 未满保持
  isolated,seq10(ts=1000)方转入 recovering→normal;dom-b 于 seq5
  过半触发隔离,seq9 冷却后恢复。
- 重放校验: `{"reproducible": true, "checked_events": 10}`;篡改
  event_states 后 E_REPLAY 精确定位 `event_seq=4, instance=s-a1,
  field=health`。
- 快照: 事件流摘要变化后重开同工作目录触发整体替换,旧快照
  `snapshot.bak.json` 为只读(权限 `-r--r--r--`)。
- 注入: `--inject event|level|snapshot` 分别以 E_EVENT/E_LEVEL/E_SNAPSHOT
  报告并写入 `logs/faultdiag.jsonl`。

## 已知边界

- 强制终止若发生在域已 isolating 但未提交确认之间,恢复后域保持
  isolating,相关指令按 unknown 重跑;`was_isolated` 集合不跨重启持久化,
  恢复路径依赖后续事件驱动。
- Tkinter 界面需在带显示环境的 Windows 上运行;本仓库 CI 环境无 tkinter,
  UI 数据层(`faultdiag.ui._load_view`)已单独验证。
