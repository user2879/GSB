# 丢包观测归并与诊断工作台

这是一个面向本地 Windows 环境的确定性丢包诊断工作台。它把两类不同格式、不同时间基准的观测归并为同一内部模型，以发送序号为轴生成可复核的缺口结论，并区分真实丢包、重传和乱序。

交付包含：

- Tkinter 诊断界面：`lossdiag/gui.py`
- 无界面 CLI 与完整业务链路：`python -m lossdiag.cli`
- 固定本地样本：`samples/`
- 可重复演示与异常注入：`scripts/demo.py`
- 标准库自动测试：`tests/test_workbench.py`
- SQLite 快照、诊断账本、回环对端与强制终止恢复状态

## 1. 快速开始（Windows）

要求 Python 3.11+，使用 Windows 官方安装包并保留 `tcl/tk` 组件。

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# 重新生成固定样本（可选，仓库内已有固定样本）
python scripts\generate_samples.py

# 运行全部自动测试
python -m unittest discover -s tests -v

# 运行端到端演示（自动选择一个空闲的 47731..47830 回环端口）
python demo.py

# 启动 Tkinter 工作台
python main.py
```

当前仓库的 Linux 验证容器没有安装 Tkinter，因此 GUI 在本容器内只做了语法编译检查；核心、CLI、SQLite、TCP 回环与全部测试均为无界面实测。Windows 官方 Python 默认包含 Tkinter。

回环对端只绑定 `127.0.0.1`。样本默认端口为 `47731`，运行前可用下面任一命令确认：

```powershell
netstat -ano | findstr :47731
Get-NetTCPConnection -LocalPort 47731 -ErrorAction SilentlyContinue
```

`scripts/demo.py` 会自行从 `47731..47830` 中选择首个未占用端口，不访问任何非回环地址。

## 2. 一键复现

```powershell
python demo.py
```

演示依次验证：

1. 27 个紧凑二进制帧逐帧解码。
2. v1/v2/v3 清单归一与显式补齐字段。
3. 1000 ms 显式平移后的时钟对齐、真实丢包/重传/乱序归因。
4. 随机打乱帧和事件记录后，结论与统计的规范序列化逐字节一致。
5. 整数 ppm 丢包率、误差界内平局与 `flow_id_asc` 稳定决胜。
6. 当前会话 v2、快照 v3 时整体替换，不增量合并。
7. 快照摘要注入异常返回 `E_SNAPSHOT`。
8. 账本逐项重放一致；账本输入漂移时定位首个流、缺口与字段并返回 `E_REPLAY`。
9. 样本损坏返回 `E_WIRE`，清单版本冲突返回 `E_FORMAT`。
10. 回环重放一致；对端 SIGKILL 后只从 SQLite 持久状态恢复，并列出未确认字段。

最近一次实测：**29/29 单元与集成测试通过；14/14 演示检查通过**。演示报告写入 `artifacts/demo/demo_report.json`，主报告摘要为：

```text
863e9a29bf31e8a9ff488ac8e0b63049d5b5faf53ec3407d5ad81b0713ab269d
```

## 3. 目录

```text
lossdiag/
  wire.py        二进制帧编解码与 E_WIRE
  events.py      v1/v2/v3 清单读取、校验与归一
  alignment.py   五元组分流与仅平移的时钟基准对齐
  diagnosis.py   发送序号轴缺口识别与成因归因
  rates.py       整数毫秒窗口、定点 ppm、误差界
  pipeline.py    端到端会话与规范报告
  store.py       SQLite 快照、账本、整体替换恢复
  replay.py      账本重放与首个差异定位
  echo.py        127.0.0.1 回环对端、ACK 与崩溃恢复
  cli.py         命令行契约、异常注入
  gui.py         Tkinter 界面
samples/         固定二进制观测与 v1/v2/v3 清单
scripts/         样本生成、可重复演示
tests/           29 项回归测试
```

`docs/FORMATS.md` 给出逐字节帧布局与 JSON 字段矩阵，`docs/VALIDATION.md` 记录实际验证结果。

## 4. 二进制观测帧

小端定长头 42 字节，随后是负载：

```text
<4sHIIQIIHHBBIH
```

| 偏移 | 长度 | 字段 | 说明 |
|---:|---:|---|---|
| 0 | 4 | magic | 固定 `PKO1` |
| 4 | 2 | frame_len | 头 + 负载总长度 |
| 6 | 4 | flow_wire_id | 五元组规范串的 CRC32 |
| 10 | 4 | seq | 发送序号，uint32 |
| 14 | 8 | send_ms | 发送端单调时钟毫秒，uint64 |
| 22 | 4 | src_ip | IPv4 网络序整数 |
| 26 | 4 | dst_ip | IPv4 网络序整数 |
| 30 | 2 | src_port | uint16 |
| 32 | 2 | dst_port | uint16 |
| 34 | 1 | protocol | 6=TCP，17=UDP |
| 35 | 1 | flags | 合法集合 `0x01/0x03/0x05/0x07` |
| 36 | 4 | arrival_ordinal | 显式到达序数 |
| 40 | 2 | payload_len | 负载字节数 |
| 42 | N | payload | 原始负载 |

`arrival_ordinal` 是输入观测的一部分，不由文件遍历顺序推断，因此打乱帧列表不会改变乱序结论。

校验失败统一返回 `E_WIRE`，详情包含零基 `byte_offset` 和 1 基 `frame_index`：

- 截断头、截断载荷、magic 错误；
- `frame_len < 42` 或 `frame_len != 42 + payload_len`；
- 未知/非法标志位组合；
- 五元组非法或 `flow_wire_id` 不匹配；
- 不支持的协议号。

合法解码输出逐帧摘要、帧总数、原始文件摘要与仅依赖帧内容的 `wire_digest`。

## 5. 丢包事件清单

所有版本共享以下文件级字段：

```json
{
  "schema": "loss-events",
  "version": "v3",
  "source_id": "events-v3-local",
  "time_basis": {
    "clock": "collector_monotonic_ms",
    "unit": "ms",
    "epoch": "collector-boot-3"
  },
  "alignment": {
    "offset_to_frame_ms": 1000,
    "basis_version": "monotonic-offset-v1"
  },
  "normalization_defaults": {},
  "records": []
}
```

版本差异：

- `v1`：记录必须有 `event_seq`、`loss_count`。没有实测时间窗、链路、重传或乱序字段，必须显式声明：

  ```json
  "normalization_defaults": {
    "link_id": null,
    "time_window_ms": null,
    "retransmit": false,
    "reorder_depth": 0
  }
  ```

  `time_window_ms` 的默认值只允许 `null`，不得编造窗口。

- `v2`：增加必填 `link_id`、`time_window_ms`；`normalization_defaults` 必须显式给出 `retransmit` 与 `reorder_depth` 默认值。
- `v3`：同时要求 `link_id`、`time_window_ms`、`retransmit`、`reorder_depth`，不得补齐这些实测字段。

每条归一记录保留 `source_version`、`source_id`、`padded_fields`、原始窗口、对齐偏移和来源摘要。未知版本、必需字段缺失、负计数或非法默认值均返回 `E_FORMAT`，详情包含 1 基 `record_number` 与字段路径。

五元组可放在文件级 `five_tuple`，也可由记录级 `five_tuple` 覆盖，以支持一个文件包含多流。

## 6. 对齐语义

1. 先按方向五元组归入同一流。`flow_id` 形如：

   ```text
   10.10.0.1:40000->10.20.0.1:5000/tcp
   ```

2. 帧侧基准必须显式声明为 `sender_monotonic_ms`，单位 `ms`；清单侧必须显式声明为 `collector_monotonic_ms`，单位 `ms`。
3. 只允许：

   ```text
   frame_time = event_time + offset_to_frame_ms
   ```

   `offset_to_frame_ms` 必须由输入 `alignment` 声明。出现 `scale` 直接 `E_FORMAT`。
4. 缺失时间基准、缺失偏移、缺失清单侧窗口、基准版本冲突等都会进入 `unaligned_flow_ids`，并列出 `missing_basis`。不猜测窗口、不强行缩放。
5. v1 没有实测窗口，因此仍可在序号轴归因，但会明确标记为 `sequence_only` 并进入未对齐集合。

## 7. 缺口与归因

候选缺口来自：

- 事件清单的 `[event_seq, event_seq + loss_count - 1]`；
- 帧序号包络内的缺失序号；
- 重复序号、`RETRANSMIT` 标志、`REORDER` 标志；
- 基于显式 `arrival_ordinal` 的到达倒序。

归因优先级固定为：

```text
重传证据 > 乱序证据 > 真实丢包
```

证据包含事件版本标记、重传标记、乱序深度、帧标志、重复观测与到达倒序。相邻且同类的序号合并为稳定缺口区间。每流输出：

当旧版本清单缺少重传/乱序字段时，同一事件区间内由帧标志、重复观测或到达倒序得到的证据会提升到区间级，保证 v1/v2/v3 对同一可观测缺口给出一致的三类计数。

- 排序后的缺口区间与成因；
- `true_loss`、`retransmission`、`reordering`、`unwindowed_true_loss` 计数；
- 与遍历顺序无关的 `gap_digest`；
- 结论摘要 `report_digest`。

规范序列化使用 UTF-8 JSON、键排序、紧凑分隔符；流按 `flow_id`、缺口按起始序号、窗口按起始毫秒排序。

## 8. 整数窗口与误差界

- 窗口宽度由会话参数 `window_ms` 指定，必须为正整数。
- 比率单位为 ppm（每百万），使用整数定点，舍入策略为 `half_up`。
- 单率量化误差上界声明为 `1 ppm`，两率差值的合成误差界为 `2 ppm`。
- 两流差值落入合成误差界时，该组按 `flow_id` 升序稳定决胜，并在排名项标 `error_affected=true`，比较项标 `decision_affected=true` 与 `tie_breaker="flow_id_asc"`。
- 分母为窗口内观测到的唯一序号数加归入该窗口的真实丢包数。
- 实测缺口事件若有显式时间窗，整个缺口记入其**起始窗口**，并标记 `whole_gap_to_start_window`；没有实测时间窗的真实丢包记入 `unwindowed_true_loss`，不猜测其窗口。

样本中 A 流为 `200000 ppm`；B/C 流均为 `181818 ppm`，排序为 B 后 C，且二者都被标记受误差影响。

## 9. 快照与账本

SQLite 默认位置为 `artifacts/workbench.db`。

快照保存：

- 内部模型版本 `loss-model-v1`；
- 事件清单版本集合；
- 对齐基准版本；
- 完整归一观测模型、缺口结论、窗口统计；
- 输入摘要、报告摘要和内容摘要。

恢复时先重算快照内容摘要。摘要不符返回 `E_SNAPSHOT`。若当前会话的模型版本、事件版本集合、基准版本、窗口或摘要任一不同，恢复采用 `whole_payload_replace`，`incremental_merge=false`，并稳定列出 `field`、`difference_source`、快照值、当前值与替换原因。GUI 会把当前视图切换为快照权威状态。

账本保存每轮输入摘要、版本组合、缺口结论与统计结论。校验命令按账本输入重新解码、对齐、归因和聚合，比较顺序为：

1. 版本组合与对齐；
2. 按 `flow_id` 的流；
3. 按区间定位缺口；
4. 缺口字段、计数、摘要；
5. 窗口率、排名与报告摘要；
6. 输入摘要。

首个差异以 `E_REPLAY` 返回，详情定位到 `flow_id`、`gap_interval`、`field`、`expected`、`actual` 和账本 ID。

## 10. 回环对端与强制终止

对端仅监听 `127.0.0.1`，消息为 4 字节大端长度前缀的规范 JSON。状态机：

```text
prepared -> sent -> acknowledged
```

`prepared` 和 `sent` 在发送动作前后分别以 SQLite WAL + `synchronous=FULL` 持久化；只有收到客户端 ACK 才写 `acknowledged`。进程被 SIGKILL/`TerminateProcess` 后，恢复命令只读取持久状态，并报告：

- `recovery_source="sqlite_persisted_echo_state_only"`
- `complete_incident_reconstruction=false`
- 未确认作业的 `peer_process_liveness_at_termination`、`socket_delivery_completion`、`client_ack`

不会声称还原强制终止瞬间的完整现场。回环响应与诊断缺口逐项比较，首个不一致返回 `E_ECHO`，带流、缺口和字段。

## 11. CLI 速查

```powershell
# 逐帧解码
python -m lossdiag.cli inspect-wire `
  --wire samples\observations.bin `
  --manifest samples\wire_manifest.json

# 三版本归一
python -m lossdiag.cli inspect-events `
  --events samples\events_v1.json samples\events_v2.json samples\events_v3.json

# 完整诊断，同时落快照与账本
python -m lossdiag.cli diagnose `
  --wire samples\observations.bin `
  --manifest samples\wire_manifest.json `
  --events samples\events_v3.json `
  --window-ms 100 `
  --db artifacts\workbench.db `
  --snapshot --ledger

python -m lossdiag.cli snapshot-list --db artifacts\workbench.db
python -m lossdiag.cli snapshot-restore --db artifacts\workbench.db --snapshot-id <ID>
python -m lossdiag.cli ledger-list --db artifacts\workbench.db
python -m lossdiag.cli ledger-verify --db artifacts\workbench.db --ledger-id <ID>

# 启动/重放/恢复回环对端
python -m lossdiag.cli echo-serve --port 47731 --state-db artifacts\echo_state.db
python -m lossdiag.cli echo-replay --wire samples\observations.bin `
  --manifest samples\wire_manifest.json --events samples\events_v3.json `
  --window-ms 100 --port 47731
python -m lossdiag.cli echo-recover --state-db artifacts\echo_state.db
```

异常注入只改输入或运行参数，不改源码：

| 参数 | 业务码 |
|---|---|
| `--inject wire-corrupt` | `E_WIRE` |
| `--inject event-version-conflict` | `E_FORMAT` |
| `--inject snapshot-digest-mismatch` | `E_SNAPSHOT` |
| `--inject echo-mismatch` | `E_ECHO`（附加演示） |

业务失败退出码为 `2`，成功为 `0`，未预期内部异常为 `1`。所有输出为规范 JSON。

## 12. 限制与边界

- 本工作台不访问外部网络，只允许 `127.0.0.1` 回环。
- v1 清单无实测时间窗，因此只做序号轴诊断，并标记为未对齐。
- 缺口整体归入起始窗口是显式的保守规则，不把序列时间分布伪装成测量值。
- 强制终止恢复是持久状态恢复，不是完整现场快照。
- 数值关键路径使用 Python 整数运算以保证跨平台一致性；NumPy 作为任务环境依赖在 `requirements.txt` 中声明，不承担确定性关键路径。
