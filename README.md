# HTTP/2 多路复用状态演进与恢复一致性验证工具

这是一个面向 Windows 本地环境的纯 Python 标准库项目。它不监听端口、不访问外部站点，而是在内存中确定性地推进 HTTP/2 帧、双层流控窗口、流生命周期与优先级依赖，并将结果写入可审计账本。项目还提供版本化快照、整体恢复、逐帧重放校验和 Tkinter 仪表盘。

## 快速开始

要求：Python 3.10 及以上；GUI 需要官方 Python 安装中的 Tcl/Tk 组件。

Windows 命令提示符：

```bat
run_demo.bat
run_gui.bat
run_cli.bat validate
```

或直接运行：

```text
py -3 run_demo.py
py -3 run_gui.py
py -3 run_cli.py run --declaration samples/session.json --script samples/frames.json
python -m unittest discover -v
```

CLI 三个主要命令：

```text
python run_cli.py validate --declaration D.json --script S.json
python run_cli.py run --declaration D.json --script S.json --strategy weighted_round_robin --snapshot-at 12 --snapshot stage.json --output run.json
python run_cli.py replay --declaration D.json --script S.json --snapshot stage.json --strategy weighted_round_robin --output replay.json
```

`validate` 严格加载；`run` 可在指定脚本事件后先落快照再继续；`replay` 从快照恢复并输出逐帧校验报告。错误以 JSON 写入标准错误，并以非零状态退出。

演示会依次执行三个共享同一份声明和主脚本的场景：

1. 小规模单策略发送；
2. 运行中切换策略后继续发送；
3. 同路径快照覆盖（保留只读 `.bak`）、整体恢复并从未完成位置逐帧重放。

所有结果写入 `artifacts/demo/`，包括账本、公平性诊断、两个场景摘要、最新快照、旧快照备份、恢复报告和尾部截断报告。程序不打开 socket，也不依赖网络。

## 目录

```text
h2mux/
  declaration.py   声明/脚本加载、字段路径校验、帧声明转帧
  hpack.py         HPACK 静态表与简化静态表编解码
  frames.py        六类 HTTP/2 帧模型、编解码和偏移错误
  priority.py      优先级树、份额、应得配额、重挂接
  ledger.py        帧、迁移、等待、调度轮次和诊断账本
  session.py       双层窗口、调度、状态机、诊断
  snapshot.py      版本/摘要快照、只读备份、整体替换恢复
  replay.py        基线重放、首差异定位、不可复现归因
  viewmodel.py     GUI/演示共用的只读稳定视图
  gui.py           Tkinter 仪表盘
  cli.py           本地 JSON 校验/运行/快照/重放 CLI
  demo.py          三场景确定性演示
samples/
  session.json     共享会话声明
  frames.json      完整确定性事件脚本
  frames_truncated.json  逻辑尾部截断变体
tests/             40 项标准库 unittest
artifacts/demo/    实际演示输出
```

## 会话声明

`samples/session.json`：

```json
{
  "schema_version": 1,
  "declaration_version": 7,
  "client": {"connection_window": 32768, "stream_window": 12000},
  "server": {"connection_window": 50000, "stream_window": 18000},
  "max_concurrent_streams": 3,
  "max_frame_size": 16384,
  "requests": [
    {"stream_id": 1, "weight": 32, "parent_id": 0, "body_bytes": 30000},
    {"stream_id": 7, "weight": 8, "parent_id": 3, "body_bytes": 12000}
  ]
}
```

含义：

- `client.*` 是本地接收方向窗口；`server.*` 是发送方向对端初始窗口。
- `parent_id: 0` 表示依赖树根；其余父流必须存在于请求集合。
- 客户端流标识必须为正奇整数且唯一；权重为 `1..256`；请求体字节数非负。
- 支持的帧大小集合为 `16384, 32768, 65535, 1048576, 16777215`。
- 所有窗口必须在 `0..2^31-1`；最大并发流必须为正整数。
- 依赖图必须无环。任一字段不合法都会整体拒绝加载并抛出 `E_DECL`，错误对象包含稳定的 JSON 字段路径，例如 `$.requests[3].parent_id`。

窗口字段也接受 `initial_connection_window` / `initial_stream_window` 别名；请求父字段接受 `depends_on` 别名。未知字段会被拒绝，避免拼写错误被静默忽略。

## 确定性事件脚本

脚本既可以是事件对象数组，也可以是顶层对象。每个事件是以下一种：

```json
{"event": "recv", "label": "peer SETTINGS", "frame": {"type": "SETTINGS", "stream_id": 0, "settings": [{"id": 4, "value": 18000}]}}
{"event": "schedule", "max_frames": 12}
{"event": "switch_strategy", "strategy": "window_priority"}
```

- `recv` 按脚本顺序把一个合法帧交给本地状态机；到达 SETTINGS 时自动产生 SETTINGS ACK。
- `schedule` 最多尝试发出指定数量的出站帧。一次调度可发 HEADERS、DATA，或因并发/窗口限制停止。
- `switch_strategy` 在脚本位置切换策略并写入控制账本。
- `expect_complete: true` 表示脚本应在结束前完成所有请求和流关闭；否则会归因为尾部截断。

帧 JSON 支持：

- `SETTINGS`: `settings: [{"id": 3, "value": 3}, ...]`、`flags: ["ACK"]`；
- `HEADERS`: `headers` 中可使用静态索引或字面字段，`priority` 可带依赖、权重和 `exclusive`；
- `DATA`: `data` 支持 `utf8`、`hex`、`repeat: {"byte": 122, "length": 300}`；
- `WINDOW_UPDATE`: `increment`；
- `RST_STREAM`: `error_code`；
- `GOAWAY`: `last_stream_id`、`error_code`、`debug`/`debug_hex`；
- `raw_hex` 可直接提供完整线上字节，用于编解码测试。

脚本加载时，每个声明帧都会先走一遍标准编码和严格解码；非法奇偶流、连接帧非零流、超限负载、非法静态索引等都会产生 `E_FRAME`，并带帧类型与偏移。

## 帧与 HPACK

实现固定 9 字节帧头和以下帧：

| 类型 | 负载 |
| --- | --- |
| DATA | 原始 DATA 字节 |
| HEADERS | 可选 PRIORITY 前缀 + HPACK 头块 |
| RST_STREAM | 4 字节错误码 |
| SETTINGS | 6 字节参数对；ACK 必须为空 |
| GOAWAY | last stream id、错误码、调试数据 |
| WINDOW_UPDATE | 4 字节增量，必须非零 |

简化 HPACK 支持 RFC 7541 的 61 项静态表、纯索引字段、带静态名索引的字面值、以及新名字字面值。动态表、Huffman、动态表尺寸更新和 never-indexed 表示不在支持范围内，遇到时以 `E_FRAME` 拒绝。

`decode_frame()` 保存原始字节；默认 `encode_frame(decoded)` 可逐字节恢复。显式传入 `preserve_raw=False` 时按规范化字段重新编码。所有偏移都是相对输入缓冲区的绝对字节偏移。

## 调度、窗口和状态机

发送方向使用服务端声明的连接/流初始窗口；接收方向使用客户端声明。每发一个 DATA：

```text
connection_send_window -= payload_bytes
stream_send_window     -= payload_bytes
0 <= both windows
payload_bytes <= max_frame_size
payload_bytes <= both available windows
```

HEADERS 不消耗流控窗口。接收 DATA 会同步扣减接收方向连接级与流级窗口；不足时审计为错误且不改变窗口。

策略：

- `weighted_round_robin`：按优先级树计算每个流的稳定份额，选择 `已发送字节 / 份额` 最小的流；平局依次按更高权重、较小流标识决定。
- `window_priority`：优先剩余流窗口最大者，再按权重与流标识稳定排序。

并发限制按非 closed 流计数。超过限制的请求按流标识排序排队，并写入开始/结束时间和原因。上游 `RST_STREAM` 或远端 END_STREAM 释放槽位后，下一次调度才打开排队流。closed 流上的后续帧记为 `E_STREAM`，不改变窗口或状态。

流状态机：

```text
idle -> open -> half_closed_local -> closed
             \-> half_closed_remote -> closed
open/half_closed_* --RST_STREAM--> closed
```

每次迁移记录帧序、虚拟毫秒、脚本位置、from/to 和触发原因。空等区段由窗口耗尽产生，记录受影响流、起止毫秒和关闭来源。时钟是确定性的虚拟单调毫秒：每个帧/控制事件推进 1 ms，从而重复运行可得到逐字节一致的账本。

## 优先级和公平性诊断

声明中的依赖关系构成以流 `0` 为根的树。一个流与其直接子流按各自权重竞争父级份额，递归得到所有流的稳定 `Fraction` 份额。诊断把总请求字节按份额做带流上限的加权 max-min 分配，输出：

- 应得字节、实得字节与偏差；
- 每个调度轮次的合格流和实得流；
- 连续至少 3 个有数据轮次未获配额的饥饿区段；
- 权重修改、依赖重挂接与原因。

父流收到 `RST_STREAM` 时，直接子流稳定地重挂接到父流的父节点；显式 PRIORITY 更新支持权重、依赖与 exclusive 重排。所有遍历均显式排序，不使用字典/哈希顺序作为业务顺序。

## 快照与整体替换

快照是带整数 `snapshot_format_version` 的 JSON 文档，内容包括：

- 声明版本、完整声明和声明摘要；
- 初始/当前策略及策略切换轨迹；
- 连接级、流级收发窗口；
- 所有流状态、已发/已收字节和排队请求；
- 优先级树与权重；
- 发送/接收账本、状态迁移、等待区段、调度轮次、优先级事件；
- 脚本摘要、脚本游标和脚本前缀摘要；
- 对除 `content_digest` 外全部内容计算的 SHA-256 内容摘要。

同路径再次保存时，旧文件复制为 `<name>.bak` 并设为只读。恢复时总是创建并填充一个新会话状态，不做增量合并。恢复报告稳定列出：

- 声明版本/摘要比较；
- 当前与快照声明字段的完整差异路径；
- 策略差异；
- 是否必须整体替换及具体原因。

## 逐帧重放校验

重放流程：

1. 从快照携带的声明、初始策略、策略切换轨迹和脚本前缀，重新生成快照时点基线；
2. 比较快照账本前缀、窗口、流、优先级树和等待状态；
3. 从快照 `script_index` 开始继续恢复会话；
4. 与不中断完整基线比较目标流、负载字节、连接/流双层窗口、流状态、业务码、状态迁移、优先级事件、调度轮次和最终诊断。

差异会定位到首个不一致的帧序号、流标识和字段。声明版本、声明内容或策略变化会在续跑前以 `E_REPLAY` + 具体 reason/basis 报告；脚本尾部截断单独返回 `SCRIPT_TAIL_TRUNCATED`，不会混成普通帧差异。

## Tkinter 仪表盘

`run_gui.py` 打开一个六页仪表盘：

- Frame timeline：帧收发顺序、负载、双层窗口与状态/业务码；
- Window curves：连接级收发窗口、每条流的发送与接收窗口曲线；
- State transitions：完整迁移轨迹；
- Priority & fairness：树深度、权重、份额、应得/实得/偏差及饥饿区段；
- Empty waits：窗口空等与并发排队区段；
- Snapshots：快照版本、策略、脚本游标、摘要和错误。

顶部可新建会话、分阶段运行、继续、切换策略、保存快照以及恢复并重放。刷新只调用 `build_viewmodel()` 生成排序后的只读投影，不修改账本；测试会校验刷新前后的账本摘要不变。

`Load JSON` 可从任意本地路径选择声明和脚本文件；加载失败会展示稳定错误 JSON。

## 错误码

- `E_DECL`：声明/脚本文档结构或声明字段不合法，带 `path`；
- `E_FRAME`：帧线上格式、长度、流标识、静态索引或帧语义不合法，带 `frame_type`、`offset`；
- `E_STREAM`：idle/closed 流收到不允许的帧，或非法流控/接收状态；
- `E_REPLAY`：不可复现、快照前缀差异或续跑帧/状态差异；
- `E_SNAPSHOT`：快照 JSON、格式版本、内容摘要或状态结构无效。

所有错误对象都可通过 `to_dict()` 生成稳定 JSON。

## 实际验证结果

本次验证环境为 Linux/WSL 上的 Python `3.11.0rc1`。当前解释器没有 Tkinter 模块，因此未做窗口渲染截图验证；`h2mux/gui.py` 已通过语法编译，GUI 数据源通过无头测试，`run_gui.py` 在缺少 Tkinter 时会明确提示并以 2 退出。Windows 官方 Python 自带 Tkinter 时可直接使用。

验证命令与结果：

```text
python -m unittest discover -v
Ran 40 tests ... OK

python run_demo.py
exit code 0, artifacts/demo/summary.json: all_checks_passed = true
```

最终演示摘要：

| 项目 | 结果 |
| --- | --- |
| 声明摘要 | `f566956ecca3899e37914676cf3f9de411f385927093a8db18ef599b360592f8` |
| 主脚本摘要 | `0b49ae9c16e76593e5601dd7e1a49163210ecd2f6f91378b35e65ab2b6e1cbf3` |
| 场景 1 | weighted，35 帧，35 条账本记录，完整关闭，序列摘要 `41d9670601bb89471d28308a501977f8428064c6df04b4ab0995bf3ca394ba8d` |
| 场景 2 | 第 10 个事件切到 window_priority，35 帧，35 条记录，序列摘要 `3143ab4f16a2b8708a05b1b565791db14580fe4894539f02d959c31e1600051c` |
| 场景 3 快照 | 脚本游标 15，内容摘要 `dd6c18ea60b9398869037d419053ca0772a608507035656401293de11fa7c8f7` |
| 场景 3 重放 | `MATCH`，11 条后续记录逐帧一致，首差异为空 |
| 旧快照备份 | `demo-session.json.bak`，模式 `0o400`（只读） |
| 尾部截断 | 单独归因 `SCRIPT_TAIL_TRUNCATED`，15 帧未完成 |

完整账本、诊断和场景输出已保存在 `artifacts/demo/`。重复运行主脚本会得到相同帧数、字段顺序和摘要；快照 JSON 按排序键写出，内容摘要不受运行环境时间影响。

## 范围与限制

- 这是离线状态演进验证器，不是完整的 HTTP/2 客户端或服务端，不监听、不连接真实端口。
- 只实现任务要求的六类帧；不支持 CONTINUATION、PUSH_PROMISE、动态表、Huffman、padding 或完整 TLS/HTTP 语义。
- “毫秒”是确定性虚拟单调时钟，用于保证相同输入产生相同等待区段；不是墙上时间。
- 接收窗口会被脚本数据消耗，但不会自动发送 WINDOW_UPDATE；脚本可显式补充发送方向窗口。
- 快照的 `content_digest` 用于发现意外损坏；若攻击者同时重写摘要，重放仍会通过逐帧前缀和后续状态比较定位差异。
