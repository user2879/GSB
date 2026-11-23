# 事件总线流式集成诊断台

面向 Windows 本地环境的确定性事件总线诊断工具。系统从本地 JSON/NDJSON
装载订阅者、发布通道和事件流，为每个订阅者启动真实回环 TCP 子进程，在全局
租约、每订阅者租约、每通道滑动窗口速率三重约束下派发批次，并用 SQLite 账本
保存三态迁移、租约归属、配额消耗、等待时长、进度游标和确认摘要。

核心代码只依赖 Python 标准库；Tkinter 是图形诊断台的可选运行依赖。仓库同时
保留了任务环境要求的 NumPy 依赖声明，但确定性协议与账本实现没有使用浮点随机
或依赖版本的随机数生成器。

## 目录

```text
eventbus_diag/          核心包、子进程服务、账本、调度、CLI、Tkinter 界面
samples/                确定性样本、损坏样本和声明演进样本
tests/                  25 项自动化测试
artifacts/              运行生成的 SQLite 账本和 JSONL 日志（已忽略）
pyproject.toml          可安装包与 eventbus-diag 命令
run_demo.bat            Windows 一键校验 + 运行 + 打开诊断台
recover_demo.bat        Windows 恢复演示
run_ui.bat              Windows 打开诊断台
run_demo.ps1            PowerShell 一键脚本
```

## 环境准备

1. 安装 Python 3.10 或以上版本。
2. 使用 Windows 官方安装包时确认 Tkinter 可用；也可执行
   `python -m tkinter` 检查图形运行库。
3. 安装依赖：`python -m pip install -r requirements.txt`。
4. 确认 `samples/spec.json` 中的高位回环端口 `47811`、`47812`、`47813`
   未被占用，并允许本地回环 TCP 通信。

```powershell
python -m eventbus_diag validate --spec samples/spec.json --events samples/events.ndjson
python -m eventbus_diag run --spec samples/spec.json --events samples/events.ndjson --db artifacts/ledger.sqlite --log artifacts/bus.jsonl --fresh
python -m eventbus_diag recover --spec samples/spec.json --events samples/events.ndjson --db artifacts/ledger.sqlite --log artifacts/bus.jsonl
python -m eventbus_diag ui --spec samples/spec.json --events samples/events.ndjson --db artifacts/ledger.sqlite --log artifacts/bus.jsonl
```

Windows 下可直接双击 `run_demo.bat`。如命令行已经安装过包，也可使用
`eventbus-diag` 替代 `python -m eventbus_diag`。

## 输入格式

### 声明 JSON

```json
{
  "declaration_version": 3,
  "global_lease_limit": 3,
  "handshake_timeout_ms": 3000,
  "subscribers": [
    {
      "subscriber_id": "s-alpha",
      "host": "127.0.0.1",
      "port": 47811,
      "processing_budget_ms": 80,
      "lease_limit": 2,
      "seed": 11,
      "topics": ["orders", "payments"]
    }
  ],
  "channels": [
    {
      "channel_id": "ch-orders",
      "topic": "orders",
      "batch_size": 2,
      "send_rate_limit": 5,
      "subscriber_quota": 2
    }
  ]
}
```

字段规则：

| 对象 | 字段 | 规则 |
| --- | --- | --- |
| 根 | `declaration_version` | 正整数；同一账本中必须单调递增 |
| 根 | `global_lease_limit` | 正整数，全局活跃租约上限 |
| 根 | `handshake_timeout_ms` | 可选正整数，默认 3000 |
| 订阅者 | `subscriber_id` | 非空字符串且全局唯一 |
| 订阅者 | `host` | 必须是显式字面量 `127.0.0.1` |
| 订阅者 | `port` | 1..65535 正整数，所有订阅者之间不重复 |
| 订阅者 | `processing_budget_ms` | 正整数处理预算 |
| 订阅者 | `lease_limit` | 正整数，每订阅者活跃租约上限 |
| 订阅者 | `seed` | 整数确定性种子 |
| 订阅者 | `topics` | 可选字符串数组；省略表示接受全部主题；支持 `*` |
| 通道 | `channel_id` | 非空字符串且全局唯一 |
| 通道 | `topic` | 非空字符串且只能由一个通道拥有 |
| 通道 | `batch_size` | 正整数 |
| 通道 | `send_rate_limit` | 正整数，每 1000 ms 滑动窗口内最多派发批数 |
| 通道 | `subscriber_quota` | 正整数，每通道选择的订阅者数量 |

订阅关系确定规则：每个通道先按 `subscriber_id` 字典序筛选接受该主题的订阅者，
再取前 `subscriber_quota` 个，因此与 JSON 对象顺序、线程顺序和派发顺序无关。
一个通道声明的主题必须至少有一个合格订阅者。

### 事件 NDJSON

每行一个事件：

```json
{"seq":1,"ts_ms":1000,"topic":"orders","payload_digest":"sha256:order-ok-1","expected_order":1}
```

| 字段 | 规则 |
| --- | --- |
| `seq` | 从 1 开始、逐行严格连续；也接受别名 `event_id` |
| `ts_ms` | 非负整数，全流单调不减；也接受别名 `timestamp_ms` |
| `topic` | 必须是已声明通道主题 |
| `payload_digest` | 非空负载摘要；也接受别名 `payload_summary` |
| `expected_order` | 非负整数，子进程按 `(expected_order, seq)` 确定性处理 |

`samples/corrupt/` 提供重复端口、非回环主机、事件缺号、时间戳倒退和未知主题等
损坏样本。第一个非法语义字段及 1 基源文件行号会写入 `E_SPEC`。

## 确定性批切分

对每个通道与每个被选订阅者，主题事件按 `seq` 升序后以 `batch_size` 切块。
批标识使用以下显式稳定序列化后的 SHA-256：

```json
{"schema":"eventbus.batch-id.v1","channel_id":"ch-orders","topic":"orders","batch_index":0,"subscriber_id":"s-alpha"}
```

序列化规则固定为 UTF-8、键排序、紧凑分隔符、不转义非 ASCII。批内容自身也按
键排序序列化并计算 `content_digest`；批内事件严格按 `seq` 升序。覆盖校验按
`(channel_id, subscriber_id)` 检查每个主题事件恰好出现一次，跨批不重复、不遗漏。

## 子进程与逐行协议

每个订阅者由一个 `python -m eventbus_diag.worker` 子进程服务，绑定声明中的
`127.0.0.1:port`。Windows 使用 `CREATE_NO_WINDOW`，父进程通过 stdin 管道
关闭触发子进程看门狗退出。

连接建立后服务先发送一行就绪问候：

```json
{"event":"ready","protocol_version":1,"subscriber_id":"s-alpha","port":47811}
```

总线发送一行批请求，服务返回一行确认。确认必须含：

```json
{"subscriber_id":"s-alpha","batch_id":"b1_...","event_ids":[1,2],"processing_ms":37}
```

本实现额外返回 `processed_order` 和 `ack_digest`，用于逐批摘要和重放校验。
端口占用、握手/连接/确认超时、确认字段缺失或事件集合不符分别产生
`E_PORT`、`E_TIMEOUT`、`E_ACK`，并同时写入结构化 JSONL 日志和 SQLite 错误表。

## 租约与三重预算

`LeaseManager` 在同一个条件锁内完成签发、续约、过期回收和配额计数。每次派发批
必须先取得租约；租约包含标识、批标识、订阅者、授予/到期单调毫秒和版本号。

* 全局活跃租约总数大于等于 `global_lease_limit` 时不得签发。
* 单订阅者活跃租约数大于等于其 `lease_limit` 时不得签发。
* 通道以真实单调时钟维护 1000 ms 窗口；窗口内派发数达到
  `send_rate_limit` 时等待最老时间戳滑出窗口。
* 等待记录包含入队单调毫秒、授予单调毫秒、等待毫秒和首个阻塞原因。
* 租约 TTL 使用订阅者 `processing_budget_ms`；守护线程每 10 ms 用单调时钟扫描
  过期租约，回收时输出 `revoked_mono_ms` 和受影响批标识。
* `renew(lease_id, subscriber_id, expected_version, extension_ms)` 同时校验当前
  持有者与版本号。版本不符返回 `E_LEASE` 且不修改到期时刻。

调度任务真实提交给 `ThreadPoolExecutor`。确认或失败后在 `finally` 中释放租约，
因此活跃租约数和自适应等待都不会绕过配额。

## SQLite 三态账本

账本使用整数格式版本 `1`，并保存整份当前内容摘要。主要表：

| 表 | 内容 |
| --- | --- |
| `ledger_info` | 整数格式版本、内容摘要 |
| `runs` | 声明版本、规范摘要、事件摘要、状态、游标、在途批数、替换差异 |
| `batches` | 批标识、事件集合、三态、租约版本、确认摘要、量化版本、等待/游标 |
| `transitions` | 每批显式状态迁移序列及单调/墙上时刻 |
| `leases` | 租约归属、授予/到期/回收单调时刻、状态 |
| `quota_usage` | 每次授予时的全局/订阅者/通道窗口配额快照 |
| `errors` | 结构化错误码、首个字段、行号、批标识和细节 |
| `replay_checks` | 恢复重放校验结果 |

状态许可表：

```text
NONE        -> DISPATCHING
DISPATCHING -> CONFIRMED | FAILED | UNKNOWN
FAILED      -> DISPATCHING
UNKNOWN     -> DISPATCHING | FAILED
CONFIRMED   -> (终态)
```

批在真正联网前提交 `DISPATCHING` 意图；完成时在同一个 SQLite 事务内写入最终
三态迁移、租约归属、配额快照、等待时长、确认摘要和进度游标，并更新运行
`in_flight_count`。`PRAGMA journal_mode=WAL` 与 `synchronous=FULL` 保证强制终止后
只恢复完整提交。

## 声明演进与陈旧防护

新声明版本必须严格高于账本中最新版本，且同一版本不能复用不同声明/事件摘要。
替换时：

1. 立即撤销内存中所有未到期租约并释放配额。
2. 旧 run、batch、transition、quota、lease、error、replay 记录整体标记 `stale=1`。
3. 当前视图只查询新 run，不合并任何旧统计或确认摘要。
4. 稳定排序输出新增、移除、字段升级的订阅者及变化通道。

Tkinter 诊断台每次刷新先清空表格再整体装载当前版本，不做增量合并。

## 强制终止恢复与 E_REPLAY

重启时使用账本记录的声明版本、规范摘要、事件摘要和确定性计划恢复：

* `CONFIRMED` 批逐项校验批标识、事件序号集合、内容摘要、确认摘要、租约归属和
  连续三态迁移；确认摘要还会用种子、预算和事件独立重算比对。
* 崩溃时遗留的 `DISPATCHING` 意图转为 `UNKNOWN`，没有任何提交记录的缺失批也
  补记 `UNKNOWN`，随后要求重跑。
* 强制终止可能留下正在响应 stdin EOF 的旧子进程；恢复启动对回环端口执行有界
  `RECOVERY_PORT_WAIT`/`RECOVERY_PORT_READY` 宽限，避免把短暂释放误报成
  `E_PORT`，同时仍对真正长期占用的端口报错。
* `FAILED` 与 `UNKNOWN` 批重新经真实套接字派发，成功后记录完整新生命周期。
* 首个差异返回 `E_REPLAY`，包含 `batch_id`、`mismatch_field` 和计划序，并在界面
  的错误/重放页展示。

## 诊断界面

`python -m eventbus_diag ui` 打开 Tkinter 诊断台，包含：

* 总览：声明版本、运行标识、状态、计划摘要、账本摘要、量化统计、替换差异。
* 订阅者：端口、预算、租约上限、种子、主题、活跃租约、进程健康状态。
* 发布通道：批量、速率、配额和稳定订阅者集合。
* 批次账本：三态、事件集合、等待/处理耗时、租约版本、确认摘要、游标。
* 租约：授予、到期、回收单调时刻和生命周期状态。
* 错误/重放：`E_SPEC`、`E_PORT`、`E_TIMEOUT`、`E_ACK`、`E_LEASE`、`E_REPLAY`。

如果没有图形运行库，命令会给出可读错误并以退出码 2 返回，不影响 CLI 验证和
恢复功能。

## 错误码

| 错误码 | 含义 |
| --- | --- |
| `E_SPEC` | 声明、JSON/NDJSON 或字段/行验证失败 |
| `E_PORT` | 回环端口被占用或子进程绑定失败 |
| `E_TIMEOUT` | 就绪握手、连接或确认读取超时 |
| `E_ACK` | 确认字段缺失、类型错误、身份不符或事件集合不符 |
| `E_LEASE` | 租约过期或续约持有者/版本校验失败 |
| `E_REPLAY` | 恢复重放首个不一致批及字段 |
| `E_LEDGER` | 账本格式版本不支持或持久化完整性错误 |

## 测试与验证

```powershell
python -m unittest discover -v
```

测试覆盖：规格首个错误字段/行号、事件连续性、确定性与反序输入一致的批计划、
覆盖完整性、全局/订阅者/速率三重预算、续约拒绝与真实过期、真实子进程确认、
`E_PORT`/`E_TIMEOUT`/`E_ACK`、声明版本替换和 `E_REPLAY`、实际强制终止后的
SQLite 恢复与未知批重跑。

最近一次本机验证结果见 [`VALIDATION.md`](VALIDATION.md)。
