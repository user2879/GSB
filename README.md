# 混沌工程注入与恢复一致性验证管线（本地 / Windows）

面向 Windows 本地环境（同时兼容 Linux/macOS）的混沌实验管线：从 JSON 声明
加载实验，按整数随机种子生成确定性注入时间轴，通过真实回环连接与真实子
进程操作施加四类注入，把每条注入的声明、时间轴位置、生效/撤销时刻、观测
摘要与业务码写入本地 SQLite 账本，并支持强制终止后的恢复、未知标注、
整体替换与可复现重放校验。

## 环境准备（Windows）

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt   :: numpy + pytest，Tkinter 随标准库提供
```

样本声明 `samples/experiment.json` 中服务端口为 0，运行时由脚本在
127.0.0.1 上选择空闲高位端口并写入 `<账本>.resolved.json`（携带源声明
摘要，声明不变则跨崩溃/恢复复用，声明变化则重新分配）。

## 快速开始

```bat
set PYTHONPATH=.
:: 完整实验（6 条注入，并行上限 2）
python -m chaos.cli run --spec samples/experiment.json --ledger demo\ledger.sqlite
:: 强制终止模拟：提交 2 条后硬杀（退出码 137，不写退出报告）
python -m chaos.cli run --spec samples/experiment.json --ledger demo\ledger_kill.sqlite --kill-after 2
:: 恢复：标记 unknown、按原种子重做、继续剩余注入并重放校验
python -m chaos.cli resume --spec samples/experiment.json --ledger demo\ledger_kill.sqlite
:: 独立重放校验
python -m chaos.cli verify --spec samples/experiment.json --ledger demo\ledger.sqlite
:: 诊断界面（只读，刷新不写账本）
python -m chaos.cli ui --ledger demo\ledger.sqlite
```

## 架构

| 模块 | 职责 |
| --- | --- |
| `chaos/spec.py` | 声明加载与强校验，任一不合法整体拒绝（E_SPEC + 首个出错字段路径） |
| `chaos/timeline.py` | 种子驱动时间轴；稳定键 = 注入内容哈希，与书写顺序无关；同声明同种子逐字节一致 |
| `chaos/service_sim.py` | 目标服务模拟子进程：固定耗时、按种子规则（sha256(seed:scope:n) % 10000 < ppm）产生错误；控制面支持 SET_ERR/CLEAR_ERR/STATUS/PING |
| `chaos/injectors.py` | 四类真实注入与真实撤销验证：回环延迟代理、错误比例覆盖、进程挂起（SIGSTOP / NtSuspendProcess）、进程终止（kill / TerminateProcess） |
| `chaos/observe.py` | 固定间隔采样；逐请求记录为真值；整数定点统计（截断方向 toward_zero，误差上界 1/scale），样本不足标记不可判定 |
| `chaos/orchestrator.py` | 状态机（pending/active/observed/reverted/failed/unknown）+ 双预算调度 + 同目标串行化 + 排队等待时刻记录 + 观测隔离诊断 |
| `chaos/ledger.py` | SQLite 账本：单事务提交注入记录+观测+游标；整数格式版本 + 内容摘要（每次事务内刷新） |
| `chaos/recover.py` | 恢复：摘要校验（E_SNAPSHOT）、版本/声明漂移整体替换（非增量合并，差异稳定排序列出）、未提交 active 标记 unknown 并生成重做计划 |
| `chaos/replay.py` | 重放校验：按种子重算时间轴、按请求记录复算观测摘要，首个不一致定位到注入序号与字段（E_REPLAY） |
| `chaos/faults.py` | 三类异常注入（仅改输入文件或参数）：损坏声明→E_SPEC、种子不符→E_SEED、账本摘要篡改→E_SNAPSHOT |
| `chaos/ui.py` | Tkinter 诊断界面（只读连接）：时间轴、状态迁移、观测曲线与误差界、unknown 标注、版本与快照列表 |

## 业务码与退出码

| 业务码 | 含义 | 退出码 |
| --- | --- | --- |
| E_SPEC | 声明不合法（含 JSON 损坏），整体拒绝 | 2 |
| E_SEED | 种子/时间轴摘要与账本不一致 | 3 |
| E_SNAPSHOT | 账本内容摘要不一致（篡改/损坏） | 4 |
| E_REVERT | 撤销验证失败（证据保留在账本 evidence 列） | 5 |
| E_REPLAY | 重放复算不一致（定位首个注入序号与字段） | 6 |

## 关键设计

- **崩溃可恢复**：注入激活时先写 intent 行（committed=0，独立事务）；观测
  完成后在同一事务内写入注入记录、观测摘要、逐请求记录与进度游标
  （committed=1）。强制终止（`--kill-after N`，`os._exit(137)`）后重启只
  依据已完整提交的记录恢复；未提交的 active 行标记 `unknown` 并生成按原
  种子的重做计划。恢复报告列出已恢复条数、未知条数与重做计划，不伪造
  崩溃时的退出报告。
- **整体替换**：账本 meta 携带整数格式版本、声明摘要、种子、时间轴摘要；
  快照目录保存最近快照。重新打开时任一项与当前会话不一致即整体删除注
  入/迁移/请求/观测/游标并重建（差异按字段名稳定排序列出），绝不增量合并。
- **观测隔离**：同一目标服务的注入经 per-target 锁按稳定序号串行化；并行
  上限由工作线程数约束；排队等待开始/结束时刻入账。
  `check_observation_isolation` 诊断同目标 active 观测窗口重叠。
- **撤销即验证**：延迟代理关闭后探测端口必须拒绝连接；错误覆盖清除后
  STATUS 必须显示 override=-1；挂起恢复后进程状态非停止且控制面可通；
  终止后按声明重建（PING 验证）或确认不可重建（端口保持关闭）。失败以
  E_REVERT 报告并保留证据，不以睡眠代替验证。
- **定点统计**：成功率与延迟分位数以会话 scale（默认 10000）整数定点表
  示，截断方向 toward_zero，量化误差上界 1 个 scale 单位（分位数输入为
  整数毫秒，误差界为 0）；样本数低于 min_samples 时 decidable=0。

## 实际验证结果（本仓库样本，Python 3.11，WSL2 Linux；Windows 路径经同一代码分支）

- `pytest`：**29 passed**（声明校验 12、时间轴确定性/顺序无关 3、定点 4、
  账本恢复/整体替换/E_SNAPSHOT 4、端到端真实注入 3、异常注入 4 等）。
- 完整实验：6 条注入全部 `reverted`，观测隔离违规 0；延迟注入 active 窗口
  p50 由 8ms 升至 135ms（注入 120ms 延迟），错误比例注入成功率降至
  6470/10000，挂起注入 active 窗口样本不足被标记 `decidable=0`。
- 强制终止 + 恢复：`--kill-after 2` 退出码 137；恢复报告 recovered=2、
  unknown=1（seq 2）、redo_plan=[2]；恢复后 6 条全部重放一致
  （`replayed_injections=6`），账本含 `active->unknown` 与
  `unknown->active (redo)` 迁移记录。
- 异常注入：corrupt-spec→E_SPEC(rc=2)、seed-mismatch→E_SEED(rc=3)、
  ledger-digest→E_SNAPSHOT(rc=4)，均写入 `logs/run.jsonl` 结构化日志。
- 演示产物见 `demo/`（账本、快照、结构化日志、各阶段输出）。

## 测试

```bat
.venv\Scripts\python -m pytest tests -q
```
