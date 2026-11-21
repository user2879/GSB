# 本地延迟队列与重试中间件（delayqueue）

纯 Python 标准库实现的本地延迟队列/重试中间件，面向 Windows 本地环境
（Python 3.10+，仅依赖 `sqlite3`、`threading`、`tkinter` 等标准库，无任何
第三方或网络依赖）。覆盖延迟投递、受配额约束的并发领取、退避重试、死信、
幂等去重、SQLite 原子账本、强制终止恢复、格式版本识别/整体替换与一致性重放。

## 目录结构

```
.
├── delayqueue/
│   ├── __init__.py     # 包公共 API（open_queue/recover_queue/...）
│   ├── errors.py       # 异常体系与稳定错误码 E_SPEC/E_SNAPSHOT/...
│   ├── models.py       # 状态、显式迁移许可表、时钟、数据模型
│   ├── specio.py       # 主题声明 JSON 与 NDJSON 消息流强校验加载
│   ├── ledger.py       # SQLite 账本：schema/版本/摘要/批次/恢复/替换
│   ├── queue.py        # 调度器、领取配额、ack/nack、可见性超时回收
│   ├── policy.py       # 确定性消费策略（演示与重放共用）
│   ├── replay.py       # 离线一致性重放校验（E_REPLAY/不可复现）
│   └── ui.py           # Tkinter 诊断界面（惰性导入 tkinter）
├── samples/
│   ├── topics.json     # 主题声明（3 个主题、各自配额与退避序列）
│   └── messages.ndjson # 确定性消息流（重试、死信、重复幂等键）
├── tests/              # 25 个 unittest 用例
├── demo.py             # 命令行演示与异常注入
└── runtime/            # 运行期生成的账本、只读备份、恢复报告
```

## 环境准备

Windows 官方 Python 3.10+ 安装包默认包含 `sqlite3` 与 `tkinter`：

```bat
python --version
python -c "import sqlite3, tkinter; print('ok')"
```

当前验证环境为 WSL2 + Python 3.11，`tkinter` 不可用，因此界面模块采用惰性
导入：核心功能与测试不依赖桌面会话，仅在执行 `demo.py ui` 时才导入 Tk。

## 快速开始

```bash
python demo.py run                 # 正常演示：重试/死信/去重/重放
python demo.py crash               # 投递进行中 os._exit(42) 强制终止
python demo.py recover             # 重开账本、恢复 unknown、跑完并重放
python demo.py inject spec         # E_SPEC：消息流主题缺失
python demo.py inject snapshot     # E_SNAPSHOT：账本内容摘要被篡改→整体替换
python demo.py inject timeout      # E_TIMEOUT：可见性超时参数非法
python demo.py ui                  # Tkinter 诊断界面（需桌面会话）
```

## 包 API

```python
from delayqueue import (
    open_queue,        # 校验声明+消息流，建账本并入队
    recover_queue,     # 重开账本（校验版本/摘要）并执行崩溃恢复
    replay_ledger,     # 恢复后一致性重放校验
    launch_dashboard,  # Tkinter 诊断界面
    SpecError, SnapshotMismatchError, TimeoutConfigError,
    ReplayError, NotReproducibleError,
)

queue, inputs = open_queue("samples/topics.json",
                           "samples/messages.ndjson",
                           "runtime/queue.db")
delivery = queue.claim(timeout=5)     # 受全局上限+每主题配额约束
queue.nack(delivery, "reason")        # 退避重试或死信
queue.ack(delivery)                   # 成功，按幂等键写去重表
snapshot = queue.snapshot()           # 只读诊断视图，不修改账本
queue.close()

queue2, report = recover_queue("runtime/queue.db")
print(report.as_dict())               # 恢复/unknown/dead 计数
print(replay_ledger(queue2))          # 一致性重放
```

## 状态机（显式许可表）

状态固定为 `pending / delayed / ready / inflight / acked / dead`，迁移仅允许
许可表中的边，未列出的迁移一律抛 `E_TRANSITION`：

| 迁移 | 触发原因 |
| --- | --- |
| pending → delayed | `enqueue`（投递时间在未来） |
| pending → ready | `enqueue_now` |
| delayed → ready | `due` / `retry_backoff` |
| ready → inflight | `claim` |
| inflight → acked | `ack` |
| inflight → delayed | `nack_retry`（投递次数 +1，按退避序列） |
| inflight → dead | `nack_dead`（达到最大投递次数） |
| inflight → ready | `visibility_timeout`（**投递次数不增、超时记录保留**） |

每次迁移在同一 SQLite 事务内写入迁移前/后状态、原因、投递次数、单调时钟
毫秒与墙钟毫秒。

## 关键设计

### 调度与配额
- 全局 inflight 上限来自声明的 `global_concurrency`，每主题上限来自该主题
  `concurrency`；任一上限耗尽时 `claim` 在条件变量上排队，绝不绕过。
- 两个守护线程基于真实 `time.monotonic_ns()`：到期提升 delayed→ready；
  对超过主题可见性超时的 inflight 租约执行 inflight→ready。超时判定只依据
  单调时钟真实流逝，没有人为拨钟。
- `run_workers` 把领取真实投射到工作线程，并保证工作线程数不超过全局上限。

### 退避、死信与幂等
- 失败取主题 `backoff_ms[attempts-1]` 重新入 delayed，投递次数递增；达到
  `max_deliveries` 仍失败则迁入 dead，并记录首次与末次失败原因。
- 成功按幂等键写去重表；同一幂等键的重复投递不产生第二次业务效果，账本
  `dedup.suppressed_count` 与消息 `suppressed` 同步累加。

### 原子账本与恢复
- 每次领取/迁移在一个 `BEGIN IMMEDIATE` 事务内同时提交迁移记录、可见性
  租约记录与进度游标（`PRAGMA synchronous=FULL` + WAL）。
- 强制终止后重启只承认完整提交的批次；已提交 `claim` 但未提交 ack/nack 的
  inflight 消息被标注为 `unknown`（`unknown_count+1`），通过
  `visibility_timeout` 边回到 ready 重投，投递次数不变，旧租约行只关闭不
  删除。组件不生成“退出报告”，只生成基于磁盘事实的恢复报告。

### 版本识别与整体替换
- 账本 `meta` 携带整数 `format_version` 与内容摘要（对消息状态/次数/抑制/
  摘要及去重表规范化后 SHA-256）。重开时版本或摘要不符即抛 `E_SNAPSHOT`，
  报告版本差、消息数量差与将丢弃的迁移条数。
- `Ledger.replace` 把旧账本复制为只读备份（POSIX `0444` / Windows 只读
  属性），再整体重建消息状态与去重表，不做增量合并。

### 一致性重放
- `replay_ledger` 用虚拟时钟按账本声明版本与消息序列离散重放全部批次，
  逐条比较最终状态、投递次数、抑制次数与迁移次数，首个不一致即抛
  `E_REPLAY` 并给出消息标识、字段、期望值与实际值。
- 崩溃恢复产生的 `claim` 与恢复回投属于恢复事实而非可重放业务批次，比较
  迁移次数时按 `2 * unknown_count` 扣除。
- 主题声明版本/摘要或去重表版本不一致时单独抛 `E_NOT_REPRODUCIBLE`，
  归类为“不可复现”，不与数据差异混淆。

### 确定性样本语义
整数负载以最小大端字节存储；负载值 N 表示“前 N 次失败、之后成功”，
N 超过主题最大投递次数则进入死信。演示与重放共用 `delayqueue.policy`，
因此离线重放与在线工作线程推导的失败序列完全一致。`deliver_at` 为相对
队列建账时刻的非负偏移毫秒：0 表示立即可领取，正值进入延迟。

## 测试

```bash
python -m unittest discover -s tests -v
```

25 个用例覆盖：声明/消息流强校验与首违定位、状态机许可表、全局+每主题
配额、真实单调时钟可见性超时、退避死信、幂等抑制、原子迁移落盘、摘要
篡改 E_SNAPSHOT 与整体替换、重放分歧 E_REPLAY、版本不可复现、子进程
`os._exit` 硬杀后的 unknown 恢复与重放一致、快照只读与 Tk 惰性导入。

实际验证结果见 `artifacts/verification_run.txt`、
`artifacts/recovery_report.json`、`artifacts/replacement_report.json`。
