# HTTP/2 流控窗口突发更新诊断工作台

这是一个纯本地的 Windows 桌面诊断应用。它读取本地 JSON 会话声明，按事件脚本推进连接级/流级双层窗口，维护流状态机，并在突发窗口更新与千流规模下比较两种确定性调度策略。

项目不启动网络服务、不连接外部目标、不使用容器。运行计算只使用 Python 标准库与 NumPy；快照存放在本地 SQLite 数据库。

## 快速开始（Windows 10/11）

```powershell
py -3.11 -m pip install -r requirements.txt
py -3.11 -m h2burst validate samples\minimal.json
py -3.11 -m h2burst run samples\burst_demo.json --full
py -3.11 -m h2burst gui samples\minimal.json
```

也可以双击 `run_workbench.bat`，默认打开 `samples\minimal.json`。若 Tkinter 不可用，CLI 会返回明确的 `E_IO`，全部诊断仍可通过命令行与测试执行。

常用命令：

```powershell
py -3.11 -m unittest discover -s tests -v
py -3.11 -m h2burst verify
py -3.11 -m h2burst snapshot samples\minimal.json --at 1 --label checkpoint
py -3.11 -m h2burst snapshots
py -3.11 -m h2burst replay samples\minimal.json 1
py -3.11 -m h2burst restore samples\minimal.json 1
```

## 已实现能力

- 会话 JSON 强校验：流标识唯一且为奇数、优先级 `1..256`、非负请求体、窗口范围、受支持帧大小、连续事件序号、单调到达时刻、窗口增量及溢出检查。
- HTTP/2 简化帧编解码：`SETTINGS`、`HEADERS`、`DATA`、`WINDOW_UPDATE`、`RST_STREAM`、`GOAWAY`；9 字节定长头、24 位长度、31 位流标识。
- 静态表头编解码：RFC 7541 静态表 61 项，按显式静态索引排序，不使用字典遍历顺序；无 Huffman、无动态表。
- 双层窗口账本：连接窗口与每个流窗口分别更新、分别扣减；每个 DATA 帧记录扣减前后两个余量。
- 流状态机：`idle -> open -> half_closed_local -> closed` 等显式许可表，HEADERS、请求体完成、RST_STREAM、GOAWAY 均留下迁移记录。
- 两种调度策略：按优先级轮转，以及按剩余窗口比例分配帧槽；规模规则自动选择，阈值内使用稳定决胜规则。
- 突发抢占诊断：按整数最大余数法统计应得/实得字节，识别连续无配额流，并输出抢占来源事件序号。
- SQLite 快照：整数格式版本、声明版本、内容摘要、状态摘要、策略/切换历史、窗口、状态、优先级和帧轨迹；异会话恢复执行整体替换，旧快照只读保留。
- 逐帧重放：从快照的下一事件继续，比较帧号、目标流、字节、两层窗口前后值、策略、事件、业务码与状态迁移；首个差异抛 `E_REPLAY`。
- Tkinter 工作台：帧时序、双层余量、状态迁移、公平性、饥饿告警与快照列表，刷新只读取视图模型。

## 目录

```text
h2burst/
  model.py          会话声明、强校验、内容摘要
  frames.py         帧与静态表头编解码
  state_machine.py  显式流状态迁移表
  engine.py         双层账本、调度、公平性诊断、序列化
  snapshot.py       SQLite 快照、整体替换、稳定差异
  replay.py         快照续跑与逐帧比较
  viewmodel.py      无副作用 GUI 视图模型
  gui.py            Tkinter 界面
  cli.py            命令行入口
samples/            固定种子本地样本
tests/              77 项自动化测试
tools/verify.py     可重复验收工件生成器
artifacts/verification.json  最近一次实际验证结果
```

## 会话 JSON

最小结构如下；事件序号从 0（也兼容从 1）开始，必须连续。

```json
{
  "format_version": 1,
  "session_id": "minimal-two-stream",
  "declared_version": 1,
  "connection": {
    "initial_window": 65535,
    "initial_stream_window": 65535,
    "max_concurrent_streams": 10,
    "max_frame_size": 16384,
    "scheduling_quantum": 3
  },
  "streams": [
    {"stream_id": 1, "priority": 1, "request_body_bytes": 50000},
    {"stream_id": 3, "priority": 2, "request_body_bytes": 22000}
  ],
  "events": [
    {"seq": 0, "arrival_ms": 0, "type": "window_update", "scope": "connection", "increment": 10000},
    {"seq": 1, "arrival_ms": 4, "type": "window_update", "scope": "stream", "stream_id": 1, "increment": 8000},
    {"seq": 2, "arrival_ms": 10, "type": "strategy_change", "strategy": "proportional"},
    {"seq": 3, "arrival_ms": 20, "type": "rst_stream", "stream_id": 3, "error_code": 8}
  ]
}
```

事件类型：

| `type` | 必填字段 | 说明 |
|---|---|---|
| `window_update` | `scope`, `increment`；流级另需 `stream_id` | 连接或流窗口增量，允许 0，执行时不得使窗口超过 `2^31-1` |
| `rst_stream` | `stream_id`, 可选 `error_code` | 关闭流；已关闭流再次收到 RST 报 `E_STATE` |
| `goaway` | `last_stream_id`, 可选 `error_code` | 关闭编号大于 `last_stream_id` 的未关闭流，后续不再打开它们 |
| `priority_change` | `stream_id`, `priority` | 发送优先级可重复，数值越小优先级越高 |
| `strategy_change` | `strategy` | `round_robin` 或 `proportional`，显式钉住策略 |

所有非法声明在运行前整体拒绝，错误包含首个非法 `field`；事件错误同时包含 `event_seq`。

## 调度规则

初始化时为受最大并发限制的流发送 HEADERS，状态进入 `open`；请求体为零时立即进入 `half_closed_local`。每处理一个脚本事件执行一个调度轮，每轮最多发送 `scheduling_quantum` 个 DATA 帧，脚本结束后继续排空可发送数据。未显式指定时，量子按流数量取 3、5 或 12。

轮转策略：

1. 可调度流按 `(priority, stream_id)` 稳定排序。
2. 每轮给各流等量帧槽。
3. 帧槽按稳定顺序交错执行，不把同一流的槽位一次性连续发完。

比例策略：

1. 以轮开始时各流剩余窗口为权重。
2. 将 `scheduling_quantum` 个帧槽用整数最大余数法分配给各流，并受请求体/窗口可容纳帧数约束。
3. 余数相同时按 `(priority, stream_id)` 决胜。
4. 固定帧槽再按稳定顺序交错执行，因此 3:1 窗口在 4 个槽位下得到 3:1 槽位。

自动规模规则：

```text
stream_count >= 128 或 pending_frames >= 256      -> proportional
average_stream_window >= 3 * max_frame_size 且
stream_count >= 16                                -> proportional
否则                                               -> round_robin
```

若两种策略的预估帧数差不超过阈值（小规模 1，大规模 2），标记 `tie_affected=true`，保留规模规则策略作为稳定决胜。每次策略变化记录事件序号、原因、旧/新策略，以及每个受影响流的“切换前应得字节、切换后实得字节、差值”。

## 账本、状态与诊断

每个出站帧记录：

```text
frame_seq, event_seq, round_index, strategy, frame_type, stream_id,
length, data_bytes, connection_window_before/after,
stream_window_before/after, business_code, end_stream
```

状态许可表位于 `h2burst/state_machine.py`。非法迁移抛 `E_STATE`；当前事件不会继续修改窗口或帧轨迹。双重窗口不足时调度器不会发出负余额 DATA；窗口更新溢出抛 `E_FLOW`。

公平性诊断对每轮实际发出的 DATA 字节按策略权重做整数最大余数分摊，累计为各流 `expected_bytes`，再与实际 `granted_bytes` 比较。默认诊断参数：

- 连续 5 个调度轮无帧配额：判定饥饿。
- 相对偏差超过 0.5：满足抢占告警偏差阈值。
- 流级窗口更新增量不小于 `max_frame_size`：视作突发更新并保留来源事件序号。
- 高优先级受益流在饥饿区间内收到突发更新并被继续调度：输出 `window_burst_preemption`，包含 `source_event_seq`。

所有输出按帧号、流号、轮次与事件号稳定排序。调度器不依赖声明中流的遍历顺序；JSON 对象键顺序也不会改变内容摘要或结果。

## 快照与重放

快照数据库采用 SQLite，包含：

- `snapshot_format_version=1`、`engine_format_version=1`；
- 会话声明版本、会话内容摘要、状态摘要；
- 当前策略、显式钉住策略、选择理由、切换历史；
- 连接窗口、所有流窗口/状态/优先级/余量/累计分配；
- 全部 HEADERS/DATA 帧轨迹、状态迁移、告警；
- `next_event_seq` 与只读标志。

同摘要恢复直接恢复状态。声明版本或内容摘要与当前会话不一致时，不合并、不增量覆盖，而是用快照内的会话声明和状态整体替换当前工作区；替换前把当前状态保存为新快照，并将所有旧快照标为只读。差异以稳定路径列出 `added`、`removed`、`changed`。

重放时从快照记录的事件游标继续执行，与不中断运行比较目标流、数据字节、两层窗口前后值、策略、触发事件、业务码和状态迁移。首个差异抛出 `E_REPLAY` 并定位到 `frame_seq` 和字段。

## 错误协议

| 代码 | 场景 |
|---|---|
| `E_SPEC` | 会话声明或事件脚本非法 |
| `E_FRAME` | 帧长度、类型、流标识、负载或静态表头非法 |
| `E_FLOW` | 窗口更新溢出或窗口余额不足 |
| `E_STATE` | 非法流状态迁移或对已关闭流操作 |
| `E_REPLAY` | 快照续跑与完整运行出现首个逐帧差异 |
| `E_SNAPSHOT` | 快照格式、摘要或内容损坏 |
| `E_IO` | 本地文件、SQLite 或 Tk 启动问题 |

错误序列化为 `code/message/field/event_seq/frame_seq/details`。

## 界面

`python -m h2burst gui <session.json>` 打开 Tkinter 工作台。工具栏支持打开、运行、逐步事件、保存快照、恢复、重放和刷新。页签包括 Frames、Streams / windows、Transitions、Fairness、Starvation 与 Snapshots；顶部展示策略选择理由和规模参数，底部展示连接窗口与事件游标。刷新只重建表格，不推进事件或改写账本。

当前 Linux 验证机缺少 `libtk8.6.so`，因此验收工件如实记录 `tkinter.available=false`。Windows 上需确认 Python 安装包含 Tcl/Tk；界面逻辑由无副作用的 `WorkbenchViewModel` 和稳定渲染测试覆盖。

## 实际验证

执行：

```powershell
py -3.11 -m h2burst verify
```

最近一次结果写入 `artifacts/verification.json`，摘要如下：

| 样本 | 流 | 事件 | 帧 | 迁移 | 饥饿告警 | 抢占告警 | 结果 |
|---|---:|---:|---:|---:|---:|---:|---|
| `minimal.json` | 2 | 6 | 8 | 5 | 0 | 0 | 重放 3 帧、3 个迁移一致 |
| `burst_demo.json` | 28 | 36 | 132 | 57 | 33 | 13 | 6 次策略切换，重放 24 帧一致 |
| `scale_1000.json` | 1000 | 109 | 1353 | 1065 | 1190 | 731 | 自动选择 proportional，重放 146 帧一致 |

- 自动化测试：`77` 项，失败 `0`，错误 `0`。
- 三组样本重复运行摘要一致；快照恢复与逐帧续跑摘要一致。
- 连续两次执行 `h2burst verify` 生成的 `verification.json` 逐字节一致。
- 整会话替换探针：版本和摘要均不一致，`added=5`、`removed=5`、`changed=3`，备份快照存在且全部快照只读。
- 异常码探针全部命中：`E_SPEC`、`E_FRAME`、`E_FLOW`、`E_STATE`、`E_REPLAY`、`E_SNAPSHOT`。

样本生成器使用固定 NumPy 种子 `20261124` 与 `10002026`；运行期调度、窗口运算和摘要分摊全部使用整数逻辑，避免浮点漂移。
