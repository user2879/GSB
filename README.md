# DistLock — 本地分布式锁与租约中间件

面向本地 Windows 环境、运行在 `127.0.0.1` 回环 TCP 上的分布式锁 / 租约中间件。
每个锁资源域由一个**独立仲裁子进程**托管，平台与仲裁进程之间使用 **JSON 行（JSONL）协议**
通信，按优先级、到达顺序与容量裁决持有权，并以仲裁进程侧的**真实单调时钟**判定租约过期。
申请 / 续约调用经**全局并发上限 + 每持有者并发配额**双约束的线程池调度，裁决结论写入
可整体替换恢复的 **SQLite 账本**和**会话快照**。

> 无 Docker / 容器依赖；不访问任何外部目标，全部使用本地确定性样本。

## 1. 环境要求与准备

- Windows 本地 Python **3.10+**，命令行可用 `python` 与 `pip`（本仓库在 Python 3.11 验证）。
- 依赖：`pip install -r requirements.txt`（安装 **NumPy 1.x**，用于定点水位向量化）。
- Tkinter：官方 Windows 安装包默认带 tcl/tk；如缺失请在安装时勾选 *tcl/tk and IDLE*。
  在无 Tcl/Tk 或无显示的环境（如最小化 Linux/WSL）中，诊断界面自动降级为等价的文本渲染。
- 回环端口检查（确认高位端口未被占用、安全软件不拦截回环监听）：

```bash
python scripts/port_check.py 39000 39030
```

- 首次运行前清理运行态（演示脚本 `reset_workdir` 也会自动清空 `workdir/`，保证可重复复现）：

```bash
python -m distlock demo --base-port 39000
```

## 2. 目录结构

```
distlock/
  errors.py            # E_* 结构化错误码（携带 文件/行号/字段/域/事件序号 上下文）
  clock.py             # MonotonicClock(真实) / FakeClock(确定性测试)
  models.py            # Domain/Holder/Event/Receipt/Transition/定点水位(NumPy)
  spec.py              # JSON 声明 + NDJSON 事件流强校验（拒绝整体加载）
  arbiter/state_machine.py  # 单域租约状态机、显式许可表、过期回收、稳定出队
  arbiter/process_entry.py  # 仲裁子进程入口
  protocol.py          # JSONL over TCP、握手超时、E_PORT/E_HANDSHAKE/E_REPLY、服务循环
  orchestrator.py      # 每域一个子进程的编排、强杀重启（不沿用旧持有权）
  scheduler.py         # 全局+每持有者信号量、等待计时、退避重试、E_EXHAUST
  ledger.py            # SQLite 单事务批次、格式版本、内容摘要、UNKNOWN 标注
  snapshot.py          # 版本/摘要识别、整体替换、账本重放逐项校验、E_REPLAY
  diagnostics.py       # Tkinter 仪表盘（缺失时文本降级，只读不写账本）
  engine.py            # 串联编排/调度/账本/快照，支持崩溃注入与恢复
  runner.py            # CLI：demo 三场景 + 三类异常注入
examples/<scenario>/   # 确定性 spec.json 与 events.ndjson（由脚本生成）
scripts/               # 端口检查、样本生成
tests/                 # 28 个 unittest（含真实子进程端到端用例）
```

## 3. 数据与协议

### 3.1 声明（`spec.json`）与事件流（`events.ndjson`）

- 域字段：`capacity`、`lease_duration_ms`、`renew_interval_ms`、`queue_limit`、`version`。
- 持有者：`holder_id`（唯一）、整数 `priority`、`target_domains`（引用必须存在）。
- 事件：`seq`（从 1 连续唯一）、`timestamp_ms`（单调不减整数毫秒）、`holder_id`、
  `domain_id`、`action` ∈ `申请/续约/释放`（也接受 `ACQUIRE/RENEW/RELEASE`）。
- 硬规则：`renew_interval_ms < lease_duration_ms`，否则整体拒绝。

### 3.2 JSONL 线协议（每行一个 UTF-8 JSON，以 `\n` 结束）

- 客户端→服务端：`HELLO`（握手）、`REQUEST`（裁决）、`STATUS`（只读诊断）、`PING`、`SHUTDOWN`。
- 服务端→客户端：`HELLO_ACK`、`RECEIPT`（含 `seq/domain_id/holder_id/verdict/remaining_ms`
  及本事件的全部许可表迁移 `transitions` / `drain_transitions`）、`STATE`、`PONG`、`ERROR`。
- 握手与请求均设置**真实 socket 超时**；端口占用 `E_PORT`、握手超时 `E_HANDSHAKE`、
  回执字段缺失 / 类型错误 `E_REPLY`。

## 4. 租约状态机与真实过期

许可表迁移：`(无)→granted`、`(无)→pending`、`pending→granted`（排空）、
`granted/renewed→renewed`、`granted/renewed→expired`、`*→released`。

- 每次处理请求前先执行 `reap()`：以仲裁进程单调时钟判定 `now >= expires_ms` 即过期回收。
- 续约仅对当前持有者有效且必须严格早于到期；同一时刻续约与过期不可能同时成立
  （先 reap 后裁决，过期判定为 `now >= expiry`），有专门单元测试守护。
- 每次迁移记录触发原因、事件序号、租约起止毫秒、域版本与单调时钟值。
- 容量耗尽时按 **优先级降序、同优先级按事件序号升序** 稳定入队；释放 / 过期腾位后按同一规则
  出队，排序键 `(-priority, seq)`，不依赖字典或内部插入顺序。

## 5. 双配额并行调度与重试

- 全局 `BoundedSemaphore(global_concurrency)` + 每持有者
  `BoundedSemaphore(per_holder_concurrency)`，配额不足时在信号量上排队，
  记录单调时钟的等待开始 / 结束与累计等待毫秒。
- 失败按声明的整数毫秒退避序列 `retry_backoff_ms` 重试；重试重发同一
  `(seq, holder, action)`，仲裁端对重复 seq 幂等，**不改变单次调用语义**。
- 重试预算耗尽以 `E_EXHAUST` 终结该事件（含尝试次数与最后错误）。

## 6. 定点水位（NumPy 1.x）

水位比例以会话缩放因子 `scale_factor` 做整数定点：
`ratio_fixed = floor(active * scale / capacity)`，截断方向为 **floor**，
量化误差上界严格 `< 1` 个定点单位（即真实比例误差 `< 1/scale`）。
`models.vector_watermark` 用 NumPy `int64` 向量化计算，逐域输出有效租约数、容量余量、
队列长度、水位比例与误差界。

## 7. 账本、强制终止与快照恢复

- 每个事件批次在**单个 SQLite 事务**内写入回执、租约迁移、水位与进度游标；
  账本携带整数 `format_version` 与对全部持久化行的 SHA-256 `digest`。
- 批次事务开启前先把事件登记到 `lease_inflight`。强制终止后重开账本，
  `recover_unknown()` 把没有对应已提交回执的在途行标记为 `UNKNOWN` 并要求从事件流重放；
  **不会伪造被杀死进程的退出报告**。
- 仲裁子进程被强杀后以**新一代空持有表**重启，旧持有权绝不沿用；恢复仅依据已提交账本，
  通过重放已提交事件重建持有表。
- 快照持久化域集合版本、缩放口径版本、各域有效租约与队列、水位、账本游标与摘要。
  重开时版本或摘要不符 → **整体替换**（非增量合并）并稳定列出差异字段；
  恢复后按账本游标继续推进，并把有效租约集合、队列次序、水位与误差界同账本逐项比较，
  首个不一致定位到 `seq + domain_id + field` 并报 `E_REPLAY`。

## 8. 错误码映射

| 代码 | 触发条件 |
| --- | --- |
| `E_SPEC` | 通用声明 / 事件流强校验失败（含行号、字段） |
| `E_TIMING` | `renew_interval_ms >= lease_duration_ms`（E_SPEC 的特化） |
| `E_REF` | 持有者目标域 / 事件域引用不存在（E_SPEC 的特化） |
| `E_PORT` | 回环端口占用 / 子进程无法绑定或未按时监听 |
| `E_HANDSHAKE` | 真实套接字握手超时或 HELLO_ACK 非法 |
| `E_REPLY` | 回执超时、信封错误或必填字段缺失 |
| `E_EXHAUST` | 退避重试次数达上限，终结该事件 |
| `E_SNAPSHOT` | 快照内容摘要不符 / 快照损坏 |
| `E_REPLAY` | 恢复后快照与账本逐项重放比较出现首个不一致 |

## 9. 使用方式

```bash
# 生成确定性样本（已随仓库生成，可重复执行）
python scripts/generate_samples.py

# 三场景顺序演示：申请争用 / 真实到期回收 / 强制终止恢复重放
python -m distlock demo --base-port 39000
python -m distlock demo --base-port 39000 --ui     # Windows 上打开 Tk 仪表盘

# 三类异常注入（分别报 E_TIMING / E_REF / E_SNAPSHOT，写结构化日志）
python -m distlock inject --inject-timing
python -m distlock inject --inject-ref
python -m distlock inject --inject-snapshot

# 测试
python -m unittest discover -s tests -v
```

结构化日志位于 `sample_logs/*.jsonl`，完整演示报告位于
`sample_logs/demo_report.json`，运行态位于 `workdir/<scenario>/`
（`ledger.sqlite3`、`snapshots/session.snapshot.json`、`runtime/ready_*.json`）。

## 10. 验证结果（本机实测）

- `python -m unittest discover -s tests`：**28 passed**。
  - 状态机：稳定优先级出队、队列满拒绝、续约 / 过期、同一时刻不双重判定、幂等重投、定点 floor。
  - 调度器：全局 / 每持有者在途峰值不超限、退避序列、`E_EXHAUST`、配额等待计时。
  - 规格：`E_TIMING`、`E_REF`、`E_SPEC`、序号缺口、时刻回退、非法动作。
  - 账本 / 快照：单事务批次与游标、未提交在途→UNKNOWN、摘要篡改 `E_SNAPSHOT`、
    版本不符整体替换、`E_REPLAY` 定位到 seq/domain/field。
  - 真实子进程端到端：`E_PORT` / `E_HANDSHAKE` / `E_REPLY`、强杀重启清空持有表且代数递增、
    STATUS 可 JSON 序列化、NumPy 向量化水位。
- `python -m distlock demo`：三场景全部成功；崩溃恢复后仲裁代数 1→2，
  在途事件 5–8 标记 UNKNOWN，快照重放校验 `status=OK`。
- 三类注入分别以 `E_TIMING` / `E_REF` / `E_SNAPSHOT` 拒绝并写入结构化日志。

### 平台说明

本仓库在 WSL/Linux + Python 3.11 + NumPy 1.26.4 实测，Tkinter 因该环境缺少
`libtk8.6.so` 而走文本降级路径（功能等价，数据同源）。在目标 Windows 上安装官方
Python（勾选 tcl/tk）后，`--ui` 将打开真正的 Tkinter 窗口；子进程使用
`CREATE_NEW_PROCESS_GROUP` 独立进程组，强杀 / 重启语义与本实现一致。
