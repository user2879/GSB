# 技术设计

## 1. 模块边界

| 模块 | 责任 |
| --- | --- |
| `config.py` | JSON 严格读取、字段白名单、聚合校验、序号可达性、内容摘要 |
| `models.py` | 配置、事件、逐项结论、字段差异、稳定 JSON |
| `state_machine.py` | CLOSED → SYN_SENT → ESTABLISHED → FIN_WAIT/CLOSE_WAIT → CLOSED、拥塞状态推进 |
| `congestion.py` | 整数窗口增长、阈值调整、定点 SRTT/RTTVAR/RTO |
| `projection.py` | 十项物化结论、静态依赖图、全量重放、逐事件检查点 |
| `ledger.py` | SQLite 两阶段事件账本、逐项版本、只读历史、快照与版本登记 |
| `service.py` | 会话打开、恢复、增量事件、版本演进、快照 API |
| `snapshot.py` | 快照重放校验、首个差异定位、整集恢复、样本灌入 |
| `peer.py` | 127.0.0.1 确定性 TCP 对端与长度前缀 JSON 帧 |
| `runner.py` | 事件脚本装载、回环应答转事件 |
| `ui_model.py` / `ui.py` | 确定性视图模型和 Tkinter 工作台 |
| `structured_log.py` | JSON-lines 结构化日志 |

## 2. 配置与脚本校验

配置和脚本都有字段白名单。加载过程先校验全部结构，再校验跨字段语义；错误路径如 `$.budget.max_events`、`peer_script.responses[0].delay_ms` 会聚合排序后整体抛出 `E_CONFIG`，不存在部分加载。

关键语义约束：

- `initial_seq + 1 + budget.max_sent_bytes <= 2^32-1`，禁止未实现回绕。
- `max_sent_bytes` 必须为 MSS 的整数倍，`window_limit >= MSS`。
- 每个脚本 `for_seq` 必须落在 `[initial_seq+1, initial_seq+max_sent_bytes-MSS]`，且相对首数据序号按 MSS 对齐。
- 延迟必须为非负整数；`reorder` 必须为布尔值，丢包动作不能声明乱序。
- RTT 缩放因子固定为 8；RTO 下界、初值、上界必须自洽。

内容摘要使用排序键、无多余空白的规范 JSON SHA-256。配置和脚本分别维护版本号。

## 3. 状态机

支持迁移：

| 触发 | 源状态 | 目标状态 |
| --- | --- | --- |
| SYN | CLOSED | SYN_SENT |
| SYN_ACK | SYN_SENT | ESTABLISHED |
| FIN | ESTABLISHED | FIN_WAIT |
| FIN_ACK | FIN_WAIT | CLOSED |
| PEER_FIN | ESTABLISHED / FIN_WAIT | CLOSE_WAIT |
| CLOSE | CLOSE_WAIT | CLOSED |

每次迁移记录：事件序号、源/目标状态、触发报文类型、触发序号、单调毫秒、配置版本、脚本版本。

数据事件在 ESTABLISHED 或 CLOSE_WAIT 中推进；重复 ACK 计数、在途段、累计确认和重传标记都在同一状态对象中维护。旧于当前累计确认的迟到 ACK 不产生前移，只增加 stale 计数；显式重复 ACK 仍按重复确认规则计数。

## 4. 拥塞与 RTT 定点规则

- 窗口单位：字节；初始窗口 `initial_cwnd_mss * MSS`。
- 慢启动：每确认 1 字节，`cwnd += 1` 字节，受 `window_limit` 限制。
- 拥塞避免：`cwnd += max(1, MSS * acked_bytes // cwnd)`，整数向下截断；约每个 RTT 增长 1 MSS。
- 三次重复 ACK：`ssthresh = max(2*MSS, floor(bytes_in_flight/2/MSS)*MSS)`，`cwnd=ssthresh+3*MSS`，最老在途段标记重传；继续重复 ACK 时 `cwnd += MSS`。
- 超时：同一阈值规则，`cwnd=MSS`，重复 ACK 清零，RTO 乘 2 并夹在上下界；最老在途段标记重传。
- SRTT/RTTVAR 缩放 8，所有除法向下取整：
  - 首个样本：`SRTT=sample`，`RTTVAR=sample/2`。
  - 后续：`RTTVAR=(3*RTTVAR+|SRTT-sample|)/4`，`SRTT=(7*SRTT+sample)/8`。
  - `RTO=SRTT+4*RTTVAR`，夹在 `[min_rto_ms,max_rto_ms]`。
  - 首个样本前回退到 `initial_rto_ms`。
- Karn：被确认的段中只要包含重传段，本次 ACK 不产生 RTT 样本，只增加 `karn_excluded`。

## 5. 逐项结论与依赖

固定十项：

`connection_state`, `cwnd`, `ssthresh`, `dup_ack_count`, `sent_unacked`, `rtt`, `stats`, `budget`, `transitions`, `diagnosis`。

依赖关系：

```text
events -> connection_state/cwnd/ssthresh/dup_ack_count/
          sent_unacked/rtt/stats/budget/transitions
connection_state + cwnd + ssthresh + dup_ack_count +
sent_unacked + rtt -> diagnosis
```

每个 `ItemResult` 包含：

- `value_json` 与 `value_digest`；
- `revision`（项目版本，只在该项实际物化时增加）；
- `basis_event_seq`；
- `depends_on_items` 与 `depends_on_events`；
- `config_version`、`script_version`；
- `last_affected_event_seq`。

事件到达后，状态机在候选副本上推进，并对前后指标做规范 JSON 比较。只对变化项和恒变化的 `stats`/`budget` 执行 UPDATE；诊断内容不变时也不更新。因此未受影响行在 SQLite 中的值、版本、依据和依赖保持原字节。

全量重放使用完全相同的“事件推进 + 受影响项物化”路径，从空状态重新生成全部项目，因此可与增量结果逐字段比较。

## 6. 事件账本事务

```text
tx1: INSERT events(status='pending')
     COMMIT
tx2: derive candidate state + affected items
tx3: UPDATE events SET status='complete'
     UPSERT affected conclusions
     UPDATE session.current_event_seq
     COMMIT
```

若在 tx1 与 tx3 之间强制终止，重开时该事件为 pending，恢复过程改为 unknown。完整事件可从账本重新推导；unknown 事件阻止新事件摄入和退出报告。`retry` 会从账本读取原始 unknown 载荷与摘要，重新置为 pending，推导成功后提交。

正常恢复会把持久化结论与完整事件重放结果逐项比较；不一致抛 `E_REPLAY`。

## 7. 版本演进

版本历史表按 `(kind, digest)` 登记。相同摘要不能换版本；不同版本号不能指向不同摘要。内容变化时，`evolve` 要求对应版本严格大于当前会话版本。

演进步骤：

1. 在新配置/脚本下对全部完整事件做一次全量推导。
2. 把当前结论复制进 `conclusion_history`，固定 `stale=1`。
3. 清空当前结论表，插入完整的推导结果。
4. 更新会话版本与 epoch。
5. 对旧/新值做稳定字段扁平化和排序，生成 `affected_fields` 报告。

旧结论只存在于历史表，UI 和汇总只读取当前非 stale 结论。

## 8. 快照与重放定位

快照表持久化：配置/脚本原文、版本、内容摘要、当前状态、十项结论、完整事件序列、逐事件检查点、总体内容摘要。

校验算法：

1. 版本字段缺失 -> `E_VERSION`。
2. 重建快照载荷并比对内容摘要；不一致 -> `E_VERSION`。
3. 校验配置/脚本版本及摘要。
4. 按事件序列重放，并逐事件比较每个检查点的值、项目版本、依据事件、依赖事件、配置/脚本版本。
5. 最终再比较十项当前结论和完整状态。
6. 第一处差异抛出 `E_REPLAY`，context 包含 `first_event_seq`、`field`、`expected`、`actual` 和依据说明。

恢复时若快照版本/摘要与当前会话不同，执行整集替换：旧结论归档、会话切换、事件序列替换、完整结论集替换，并返回旧值/新值差异报告。

## 9. 回环对端

对端绑定 `127.0.0.1`，显式端口必须 >=1024；端口 0 由操作系统分配高位临时端口。帧格式为 4 字节大端长度 + UTF-8 JSON。

一批请求按脚本生成应答：

- `ack` / `dupack`：返回声明 ACK 号；
- `drop`：无应答，客户端生成超时事件；
- `delay_ms`：对端睡眠后返回；
- `reorder=true`：应答延后到本批普通应答之后发送，形成可观察乱序；
- 脚本耗尽：按 `default_policy=ack|drop` 和 `default_delay_ms` 应答。

连接/交换失败为 `E_CONNECT`；监听失败为 `E_PORT`；脚本加载或协议错误可注入为 `E_SCRIPT`。

## 10. 窗口与恢复约束

Tkinter 使用固定 `ITEM_ORDER` 构建表格；快照按 ID 倒序显示，日志按追加顺序取末尾。渲染模型不依赖 Tk 控件，因此可在无图形环境中做确定性文本验证。生产恢复不允许用 unknown 事件“猜测”结论，也不会生成退出报告；必须显式执行 `retry`。

