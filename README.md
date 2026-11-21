# 事务消息中间件（Windows 本地回环实验实现）

这是一个仅使用 Python 标准库、面向 `127.0.0.1` 本地环境的事务消息中间件组件。重点不是做通用消息队列，而是把以下两件事做成可观察、可恢复、可复算的实现：

1. 半消息在显式结束指令与超时回查交织时的唯一终态裁决；
2. 以事务标识为幂等键的重复拉取去重，以及回查/拉取延迟预算判定。

核心代码不依赖第三方包，不使用 Docker。SQLite 是权威状态与事件账本，TCP 服务与回查都使用真实回环套接字。`tkinter` 只被可选桌面控制台导入；当前验证环境的 Python 3.11 缺少 tcl/tk，因此主验收使用无界面 CLI，缺失时 `python main.py tui` 会给出明确安装提示。

## 快速开始

```powershell
# Windows PowerShell / cmd，Python 3.10+
python --version
python -m unittest discover -s tests -v

# 运行确定性回环验收（自动选择高位端口、启动/释放监听）
python main.py scenario

# 真实强制终止、恢复、尾部截断和启动异常注入
python tools/crash_harness.py

# 一键执行全部检查并生成 VALIDATION.md
python tools/acceptance.py
```

服务端：

```powershell
python main.py serve --db data\mw.db --port 0 --port-file artifacts\port.json
```

`--port 0` 让操作系统分配一个当前空闲的高位端口。程序在 `127.0.0.1` 上监听，退出时关闭套接字。端口和恢复信息写入 `artifacts\port.json`。

生产者查询桩：

```powershell
python tools\query_stub.py --answers samples\query_answers.json --port 0 --port-file artifacts\query-port.json
```

单条请求：

```powershell
python main.py send --port 45678 --json "{\"op\":\"ping\"}"
```

## 目录

```text
main.py                       CLI 入口
txn_mw/
  codec.py                    TMW1 二进制事件帧编解码
  store.py                    SQLite 模式、事件序号、账本摘要
  engine.py                   状态机、回查线程池、仲裁、幂等拉取、恢复
  server.py                   127.0.0.1 JSON 行 TCP 服务
  ledger.py                   账本导出、整体替换、只读备份与后置校验
  verify.py                   按事务独立重放验证器
  scenario.py                 确定性样本执行器与查询桩
  cli.py                      命令入口
  tui.py                      可选 Tk 控制台
  logging.py                  结构化 JSONL 日志
samples/                      生产者、查询、消费、异常注入样本
tests/                        协议、状态机、恢复、账本与验证测试
tools/                        query_stub / crash_harness / acceptance
artifacts/                    实际运行结果、结构化日志与验收报告
```

## 协议

传输为回环 TCP 上的 UTF-8 JSON 行。每行一个对象，单行上限 8 MiB。所有服务端错误回复形如：

```json
{"ok": false, "code": "E_TXN", "message": "...", "path": "$.version", "details": {}}
```

### 入站操作

| `op` | 关键字段 | 说明 |
|---|---|---|
| `half` | `txn_id`, `version`, `body`, `txn_timeout_ms`, `query.host`, `query.port`, `query_max` | 持久化半消息；返回成功后才可进入回查队列 |
| `commit` | `txn_id`, 可选 `version` | 显式提交；回查进行中时延迟裁决 |
| `rollback` | `txn_id`, 可选 `version` | 显式回滚；回查进行中时延迟裁决 |
| `pull` | `txn_id`, `consumer_id`, 可选 `request_id` | 以事务标识为幂等键拉取 |
| `trace` | `txn_id` | 输出完整事件轨迹、仲裁依据、最终状态与延迟 |
| `inventory` / `status` | 无 | 事务、事件、幂等窗口、延迟汇总 |
| `stats` / `latency` | 可选 `txn_id`, `operation` | 整数定点延迟统计 |
| `ping` | 无 | 健康检查 |

`query` 端点必须是回环地址；非回环地址会以 `E_TXN` 拒绝。

### 出站回查

超时后服务端主动连接半消息中声明的生产者查询端点：

```json
{"op":"query_txn","txn_id":"txn-1","version":1,"attempt":1}
```

应答：

```json
{"txn_id":"txn-1","result":"committed"}
```

`result` 只允许 `committed`、`rolled_back`、`unknown`。连接失败、超时、非法应答都按 `unknown` 处理并消耗一次回查次数。

## 三态与终态裁决

持久化记录的状态严格限定为：

- `half`：已落盘的半消息；
- `committable`：可投递；
- `consumed`：已产生且只产生一次业务效果。

回滚不是第四种消息状态，而是 `half` 上的终态处置。`txns.disposition` 明确区分：

| `state` | `disposition` | 含义 |
|---|---|---|
| `half` | `pending` | 等待结束指令或回查 |
| `half` | `rolled_back` | 回滚终态，不可投递 |
| `half` | `manual_review` | 回查次数耗尽，待人工 |
| `committable` | `committable` | 可拉取 |
| `consumed` | `consumed` | 已有业务效果，重复拉取只返回原效果 |

### 固定仲裁规则

1. 显式提交命中半消息后写 `committable`；显式回滚把 `half` 锁定为 `rolled_back`。
2. 回查进行中到达的提交/回滚记录为 `pending_command`，不立即写终态；回查响应到达后合并裁决。
3. 回查已提交与回滚命令冲突时，固定采用 **提交优先**（`commit_precedence`），写入冲突表与 `conflict` 事件，并记录 `E_CONFLICT`。
4. 回查已回滚与提交命令冲突时同样采用提交优先。
5. 终态写入后，任何相反指令都不能改写状态；仍会留存证据并以 `E_CONFLICT` 回复。
6. 同一事务同时存在多条回查请求时，`query_inflight` 保证只有一个在途尝试；命令在途期间合并为 `pending_command`。
7. 查询进行中出现互相矛盾的提交/回滚命令时，立即保留冲突证据并报告 `E_CONFLICT`，合并后的待裁决命令按提交优先规则收敛。

裁决依据持久化在：

- `conflicts` 表；
- 事务事件轨迹中的 `command`、`query_result`、`conflict`、`decision` 事件；
- 结构化日志中的 `E_CONFLICT` 记录。

## 事件日志编码

每条事件使用 `TMW1` 二进制帧：

```text
magic           4 bytes   b"TMW1"
payload_length  uint32    大端，规范 JSON 字节长度
payload         N bytes   canonical JSON
digest          32 bytes  SHA-256(payload)
```

payload 必含：

```text
txn_id       非空字符串
version      正整数
state        half | committable | consumed
event_seq    正整数，事务内严格递增
body_digest  消息体规范化 JSON 的 SHA-256
type         事件类型
data         事件附加数据
```

解码检查：

- 长度字段与实际帧长一致；
- 尾部 SHA-256 与 payload 一致；
- `half` 事件中的 `body` 重新计算摘要必须等于 `body_digest`；
- 未知状态、缺失字段、类型错误、摘要不符均整体拒绝；
- 错误为 `E_TXN`，并包含精确字段路径，例如 `$.record.length`、`$.record.state`、`$.record.version`；
- `decode_records()` / 恢复/验证器额外检查同一事务的 `event_seq` 严格递增。

## 半消息确认与超时回查

`half` 的处理顺序是：

1. SQLite 事务写入 `half` 状态及 `half` 事件；
2. TCP 写入成功应答；
3. 再提交 `half_ack` 标记。

只有 `ack_sent=1` 的半消息会被回查调度器选中。若进程在应答发送前终止，半消息不会进入回查队列；若在发送后、`half_ack` 前终止，也会保守地保留为未确认半消息，恢复报告将其列为 `orphaned_unacked_half_messages_not_queued`。

超时判断使用 `time.monotonic_ns()`。调度器只处理真实到期事务，不用人为睡眠伪造超时。每次回查记录：

- `query_count`；
- `first_query_mono_ns`；
- 最后一次回查时刻、下一次计划时刻；
- 在途标志、最近结果；
- 每次 `query_started` / `query_result` 事件。

达到 `query_max` 后置为 `manual_review`，记录 `E_QUERY`，字段包含 `txn_id` 与 `query_count`。

## 幂等拉取与去重窗口

幂等键就是事务标识。首个成功拉取：

- 把 `committable` 原子迁移到 `consumed`；
- 生成确定性效果标识 `eff_<hash>`；
- 写入 `idempotency` 账本；
- 写入 `consume` 事件。

同一键的后续拉取返回原 `effect_id`，`duplicate=true`，消费计数保持为 1，并写入 `pull_duplicate` 事件。即使幂等窗口中的键被回收，`consumed` 事务仍保留效果标识，因此不会产生第二次业务效果。

窗口同时受两个整数维度约束：

- `--idem-max-entries`：最多保留键数；
- `--idem-ttl-ms`：按键最后一次访问计的整数毫秒 TTL。

回收顺序为：先 TTL 过期，再按最后访问时间最旧淘汰直到满足条数上限。每次回收写入 `idempotency_evictions` 表与 `idempotency_evict` 事件，依据为 `ttl_exceeded` 或 `capacity_exceeded`。

未完成拉取在崩溃恢复时被标记为 `re_pull_required`；如果效果已经完整提交而客户端没收到响应，重拉会命中幂等键并返回相同效果。

## 延迟观测与预算

真实来源是 SQLite `latency` 表的逐次记录；`truth_source` 字段固定为 `sqlite.latency.row`。每次查询/拉取记录：

```text
operation         query | pull
txn_id            事务标识
request_id        拉取请求标识（查询可空）
start_mono_ns     time.monotonic_ns()
end_mono_ns       time.monotonic_ns()
duration_us       整数微秒，向下取整
budget_us         声明预算
exceeded           0/1
```

查询预算同时作为真实套接字 `connect/send/recv` 超时使用；拉取预算只用于真实操作耗时判定，不添加人为延迟。

统计按事务和整体输出：

```text
mean_ms_x1000
p95_ms_x1000     最近秩法：sorted[ceil(0.95*n)-1]
max_ms_x1000
budget_exceeded_count
sample_count
```

`ms_x1000` 与 `us` 数值相同，表示三位小数毫秒的整数定点。纳秒到微秒截断误差小于 1 µs，均值取整误差小于 1 µs，因此声明量化误差上界为 **2 µs**。

单次超预算写 `E_LATENCY`，包含 `operation`、`txn_id` 和预算/实测值。

## 崩溃恢复

每次启动先扫描事件日志，再开启回查线程。恢复只依据 SQLite 中完整提交的行与可完整解码的事件帧，不会读取网络缓存。

恢复规则：

- 未完成回查：标记最近结果为 `unknown`，清除在途标志，重新进入回查队列，并追加 `recovery_query_unknown` 事件；
- 未完成拉取：`pulls.status` 改为 `re_pull_required`，报告请求与消费者；
- 未确认半消息：不进入回查队列；
- 尾部不完整/摘要错误的帧：移入 `quarantine`，原日志行标记 `valid=0`，报告 `tail_incomplete_records` 和字段路径；
- 回查次数、首次回查时间、幂等窗口都由事件按顺序重放；
- 内容摘要不一致会以 `E_LOG` 报告并从有效记录重建；
- 恢复报告明确 `shutdown_report_available=false`。强制终止无法生成有序退出报告，本实现不会伪造。

### 账本整体替换

```powershell
python main.py export-ledger --db data\mw.db --out data\ledger.json
python main.py inspect-ledger --ledger data\ledger.json
python main.py restore-ledger --db data\target.db --ledger data\ledger.json
python main.py verify --db data\target.db
python main.py verify --ledger data\ledger.json
```

账本 JSON 携带：

- 整数 `format_version`；
- 覆盖交易、事件、幂等、淘汰、拉取、延迟与冲突内容的 `content_digest`。

`restore-ledger` 不增量合并。它先：

1. 校验结构；
2. 将现有 SQLite 账本备份为 `*.pre-replace-<时间>-<纳秒>.readonly.db` 并尽力设为只读；
3. 删除并整体重建交易、事件、幂等与观测表；
4. 逐事务重新校验状态、回查次数、首次回查、效果标识、幂等键和淘汰证据；
5. 任一不一致以 `E_REPLACE` 报告 `txn_id` 与 `field`。

版本或内容摘要与当前账本不同也会触发整体替换，并写入结构化 `E_REPLACE` 说明，不进行合并。

## 独立验证器

```powershell
python main.py verify --db data\mw.db
python main.py verify --ledger data\ledger.json
```

验证器不使用运行中的内存状态，而是按事务独立重放 `half`、命令、回查、裁决、消费与幂等事件：

- 检查半消息唯一性；
- 检查事件序号；
- 检查回查次数/尝试编号；
- 检查消费只能从 `committable` 发生；
- 检查重复拉取效果标识一致；
- 比较重放终态与账本行。

首个不一致以 `E_VERIFY` 返回事务、字段、事件序号与说明。

## 异常注入

启动服务时可重复使用：

```powershell
python main.py serve --inject-fault log_tail_truncate
python main.py serve --inject-fault duplicate_txn
python main.py serve --inject-fault query_command_conflict
```

对应结构化错误码：

- 日志尾部截断：`E_LOG`
- 事务标识重复注册：`E_DUP`
- 回查已提交与回滚指令冲突：`E_CONFLICT`
- 回查耗尽：`E_QUERY`
- 延迟超预算：`E_LATENCY`
- 账本替换不一致：`E_REPLACE`
- 独立重放不一致：`E_VERIFY`
- 协议/状态错误：`E_TXN` / `E_STATE`

样本参数在 `samples/fault_injection.json`。实际冲突场景由 `samples/query_answers.json` 的延迟应答和生产者后到回滚共同触发，不使用中间件人为延迟。

## 确定性样本

- `samples/producer_half_sequence.json`：9 个事务，覆盖显式提交、显式回滚、超时回查提交、未知后提交、回查耗尽、查询/命令冲突、幂等淘汰；
- `samples/query_answers.json`：按事务与尝试次数返回查询结果；
- `samples/consumer_pulls.json`：12 路并发重复拉取、顺序重拉、回滚后拉取拒绝、淘汰后重拉；
- `samples/fault_injection.json`：三类异常和延迟/窗口参数；
- `samples/scenario.json`：场景入口。

## 当前环境验证结果

见 `VALIDATION.md` 与 `artifacts/validation/acceptance-report.json`。本环境实际结果：

- 23 个单元/集成测试全部通过；
- 确定性场景 9 个事务、18 次拉取、12 路并发拉取只产生 1 个效果；
- 18 次拉取均超过声明的 1 ms 预算并写入 `E_LATENCY`；
- 2 次幂等键容量淘汰、淘汰后重拉仍返回原效果且消费计数为 1；
- 独立事务重放与账本文件验证均通过；
- 真实 `SIGKILL` 后未生成退出报告，未完成回查被标记 unknown 并重入队列，重启后继续回查并提交；
- 恢复后查询/拉取延迟账本保留且新增样本可查询；
- 日志尾部截断进入隔离区并以 `E_LOG` 报告；
- 账本版本不一致时整体替换成功，旧 SQLite 账本备份权限为只读 `0444`。

当前自动化运行环境为 WSL2/Linux（Python 3.11.0rc1）；Windows 相关路径均使用
`pathlib`、标准 `socket`、`sqlite3` 与标准 GUI 可选导入，未使用 Unix 专属的运行依赖。
本容器缺少 tcl/tk，因此 `tui` 的缺失降级路径已实测返回安装提示；核心 TCP、SQLite、
回查和验收均在真实回环套接字上完成，而不是在 Docker 中运行。

## 安全与范围

- 只监听 `127.0.0.1`，查询目标也限制为回环地址；
- 无 TLS、无跨机认证；这是单机实验组件；
- 没有实现多进程同时写同一个空账本；并发模型是单服务进程内多线程 + SQLite 串行事务；
- 运行中的实时账本整体替换应先停止服务；`restore-ledger` CLI 以离线模式执行；
- 延迟统计反映本机和当前负载，不是跨机时钟同步结果。
