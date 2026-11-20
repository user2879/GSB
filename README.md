# 网关连接级配额水位组件

这是一个面向本地 Windows / Python 3.10+（实测 3.11）的确定性实验实现。它按 NDJSON 轨迹推进连接池并发水位，维护借用归属账本，输出定量告警，并把每个完整事件批次写入 SQLite。组件不访问外部网络，连接建立/关闭确认仅使用 `127.0.0.1` 回环 TCP 套接字，也可以用重放代理完全不打开套接字。

## 快速开始（Windows）

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -c "import numpy, sqlite3, socket, json; print(numpy.__version__)"
.\run_demo.ps1
```

命令提示符下也可执行 `run_demo.bat`。WSL / Linux 下执行 `./run_demo.sh`。

`run_demo.*` 会依次完成环境确认、输入强校验、单池满载、跨池借用与逾期归还、参数版本收紧、强制终止、完整批次恢复、剩余事件续跑、全量重放，以及三类异常注入。演示使用端口 `0` 申请操作系统分配的临时回环端口，并在启动前实际 bind 检查；也可在 CLI 中用 `--loopback-port` 指定固定端口。

## 交付内容

```text
gateway_quota/
  loader.py        JSON/NDJSON 加载与 E_SPEC 强校验
  engine.py        生命周期状态机、借用账本、告警与参数替换
  loopback.py      127.0.0.1 回环信号确认
  storage.py       SQLite 批次事务、恢复快照
  replay.py        逐事件重放比较与 E_REPLAY
  runner.py        批处理、强制终止与恢复编排
  cli.py           validate / run / replay 命令行
samples/
  pools.json       3 个连接池
  borrows.json     alpha -> beta 借用关系
  trace.ndjson     28 条确定性事件
  reconfig_v2.json 在 seq=22 后收紧告警阈值
  anomalies/       E_BORROW / E_GAP / E_VERSION 样本
tests/             12 项 unittest
demo.py            一键实验入口
artifacts/         实际运行生成的摘要、日志与 SQLite 账本
```

## 数据格式

### 连接池 `pools.json`

```json
{
  "version": 1,
  "alert_threshold": 1,
  "rate_window_ms": 1000,
  "percentage_scale": 1000,
  "pools": [
    {
      "id": "alpha",
      "max_connections": 3,
      "per_connection_request_limit": 2,
      "idle_release_ms": 500,
      "read_buffer_limit_bytes": 256
    }
  ]
}
```

`max_connections` 与 `per_connection_request_limit` 必须为正整数；空闲阈值与读取缓冲上限必须为非负整数。池标识必须唯一。

### 借用关系 `borrows.json`

```json
{
  "version": 1,
  "borrows": [
    {
      "lender_pool": "alpha",
      "borrower_pool": "beta",
      "max_borrowed": 1,
      "return_deadline_ms": 300
    }
  ]
}
```

`max_borrowed` 不得超过出借方 `max_connections`；关系图不得成环；同一出借/借入对不得重复。两个文件的 `version` 必须一致。

### 轨迹 `trace.ndjson`

```json
{"seq":13,"pool":"beta","connection":"b3","type":"establish","timestamp_ms":1800,"bytes":0}
{"seq":14,"pool":"beta","connection":"b3","type":"request","timestamp_ms":2150,"bytes":64}
{"seq":15,"pool":"beta","connection":"b3","type":"close","timestamp_ms":2250,"bytes":0}
```

事件类型为 `establish`、`request`、`idle_release`、`close`，同时接受中文别名 `建立`、`请求`、`空闲释放`、`关闭`。事件序号从 1 开始连续且唯一；同一连接必须先建立；同一池时间戳不得倒退。

### 参数演进 `reconfig_v2.json`

示例在 `seq=22` 后整体替换到版本 2，把告警阈值从 1 收紧为 0。文件包含完整的新参数快照与 `after_event_seq`，因此替换是整体视图重建，不是逐字段增量合并。

## 核心语义

每个池分别记录：

- `own_active`：使用本池基础槽位的活跃连接。
- `borrowed_in`：逻辑上属于本池、但由出借方授予容量的连接。
- `lent_out`：本池已借给其他池的槽位。
- `active_connections = own_active + borrowed_in`：逻辑活跃连接。
- `capacity_in_use = own_active + lent_out`：本池真实占用容量，始终 `<= max_connections`。
- `remaining_connections = max_connections - capacity_in_use`。

因此借用不会掩盖借出方容量饱和；借入池的逻辑活跃数可以临时超过其基础上限，但其 `capacity_in_use` 仍不超过基础配额，且出借方的 `own_active + lent_out` 始终受配额约束。这是同时满足“借入可用”和“出借方不得超卖”的可核验口径。

生命周期：

- `establish`：`missing -> established`，占用自有槽位或成功借用。
- `request`：仅允许 `established/idle`；成功后进入 `idle`，累计请求数，并检查字节缓冲上限。
- `idle_release`：仅允许已空闲达到 `idle_release_ms` 的活跃连接，进入 `released`。
- `close`：活跃连接进入 `closed`。
- `released/closed` 为终态；再次请求会记录 `E_LIFECYCLE` 且不计入。

当自有槽位不足时先记录 `E_POOLFULL`，再按 `(lender_pool, borrower_pool)` 稳定排序尝试借用。借用成功会登记单连接租约、借出方、借入方、截止毫秒与参数版本；借入连接关闭或空闲释放即归还。到截止时刻仍未归还的租约转为 `overdue` 并记录 `E_OVERDUE`，之后归还标为 `returned_late`。

## 定量诊断

每个事件后，对每个池计算：

```text
waterline_percentage_fixed =
    floor(capacity_in_use * percentage_scale / max_connections)
```

截断方向固定为 `floor_toward_negative_infinity`，由于计数非负，等价于向下取整。示例缩放因子为 1000，因此 2/3 输出 666（‰）。

当 `remaining_connections <= alert_threshold` 时输出 `W_APPROACHING`，包含：

- 事件序号、池标识、剩余连接数、容量水位与固定点百分比。
- 到达速率窗口 `(timestamp_ms - window_ms, timestamp_ms]`，窗口外不含、窗口端点含。
- 窗口内已接受的 `establish` 数量。
- 定点速率 `arrivals * percentage_scale * 1000 / window_ms`，向下取整。
- 耗尽预计 `ceil(remaining_connections * window_ms / arrivals_in_window)`。
- 推算窗口与保守误差上界，误差上界为完整滑动窗口 `window_ms`。

以下情况显式标注不可推算，不生成任意值：

- `WINDOW_WARMUP_INCOMPLETE`：会话尚未覆盖一个完整窗口。
- `NO_ESTABLISH_IN_WINDOW`：窗口内没有成功建立事件。
- 已经零剩余时直接给出精确 `eta_ms=0` 与 `ALREADY_AT_THRESHOLD`。

## 参数演进与陈旧防护

参数版本必须单调递增。版本切换时：

- 旧警告与其内嵌耗尽预计全部标记 `stale`。
- 旧租约全部标记陈旧；未归还租约状态变为 `stale` 并列入 `invalidated_borrows`。
- 当前连接集合按新配额和新借用关系重新确定性分配，整体重建当前水位视图。
- 稳定输出 `affected_pools`、`cleared_alert_records`、`invalidated_borrows`、`invalidated_borrow_relations`、`added_borrow_relations`、`reconstituted_loans` 和 `unknown_connections`。
- 旧版本水位不参与当前告警、统计或耗尽预计。

示例在 `seq=22` 后把阈值 1 收紧为 0，原先 `alpha`、`beta` 的两条告警被清除；33 条旧告警全部陈旧，当前告警数为 0。

## SQLite 账本、终止与恢复

每批事件开始前写入独立的 `pending_batches` 元数据标记；每批完成后，以下内容在同一 SQLite 事务中提交：

- 批次摘要与事件摘要。
- 每个事件的规范化事件和事件后审计快照。
- 各池 `own_active/borrowed_in/lent_out/capacity_in_use/remaining`。
- 全部连接原始归属、当前归属和生命周期。
- 借用租约、告警记录和参数版本。

强制终止时不会写退出报告。重启只读取最后一个完整批次；`pending_batches` 中的候选新建连接被列为 `unknown_connections`，要求由剩余轨迹重新建立。示例在 `seq=13` 强制退出，恢复点为 `seq=10`，未知连接为 `b1,b2,b3`，随后从 `seq=11` 重放并完成到 `seq=28`。

`replay` 从初始参数版本开始逐事件推进状态机，比较：

- 各池活跃数、真实容量占用、剩余连接数、固定点水位与告警结论。
- 每个活跃连接的原属池、当前归属池、租约与状态。
- 活跃数总和与连接归属条目总数。

首个差异会作为 `E_REPLAY` 报告，包含事件序号、池标识、字段、账本值、重放值与依据。服务关闭、空闲释放和借用归还都按原轨迹重放；重放使用 `NullSignalBroker`，不打开真实套接字。

## 命令行

```powershell
# 仅加载并整体校验
python -m gateway_quota.cli validate `
  --pools samples/pools.json --borrows samples/borrows.json --trace samples/trace.ndjson

# 正常运行；默认使用 127.0.0.1 临时端口
python -m gateway_quota.cli run `
  --pools samples/pools.json --borrows samples/borrows.json --trace samples/trace.ndjson `
  --reconfig samples/reconfig_v2.json --ledger artifacts/manual.sqlite3 `
  --batch-size 5 --log artifacts/manual.ndjson

# 仅重放账本
python -m gateway_quota.cli replay `
  --pools samples/pools.json --borrows samples/borrows.json --trace samples/trace.ndjson `
  --ledger artifacts/demo_ledger.sqlite3

# 指定固定回环端口（启动前会 bind 检查）
python -m gateway_quota.cli run ... --loopback-port 43119
```

`--no-loopback` 可关闭套接字信号；`--crash-after-seq 13` 会调用 `os._exit(137)` 模拟强制终止。

异常注入不修改源码：

```powershell
python demo.py --inject borrow-cycle
python demo.py --inject missing-close
python demo.py --inject version-mismatch
```

不传 `--inject` 时执行全部三类。结构化日志为逐行 JSON，异常记录包含 `code`、`business_code`、字段路径、行号或账本版本详情。

## 业务码

| 业务码 | 含义 |
| --- | --- |
| `E_SPEC` | 任一输入规格不合法，整批拒绝；更具体的借用/缺口子码见 `business_code` |
| `E_BORROW` | 借用关系成环等借用图错误，外层仍为 `E_SPEC` |
| `E_GAP` | 事件序列缺口，例如关闭事件被删除 |
| `E_VERSION` | 账本初始参数版本/摘要不一致或参数版本违背单调性 |
| `E_LIFECYCLE` | 连接当前状态不允许该事件 |
| `E_CONNLIMIT` | 每连接请求累计超限 |
| `E_POOLFULL` | 基础配额耗尽；记录借用成功或拒绝结果 |
| `E_OVERDUE` | 借用超过归还截止仍未归还 |
| `E_BUFFER` | 请求字节数超过读取缓冲上限 |
| `E_REPLAY` | 恢复重放与账本出现首个字段差异 |
| `E_LOOPBACK` | 本地回环信号确认失败 |

## 实际验证结果

在 workspace 中执行：

```text
python -m unittest discover -s tests -v
Ran 12 tests in 1.396s
OK
```

`python demo.py` 的实际稳定摘要（回环端口为动态环境字段）：

```text
version=1 pools=3 borrows=1 events=28 reconfigs=1
child_exit_code=137 (expected 137)
exit_report_produced=false
recovered_through_seq=10 pending_start_seq=11
unknown_connections=['b1', 'b2', 'b3']
recovery_replay_consistent=True checked=10
final_replay_consistent=True checked=28
continued_events=18 last_event_seq=28

single_pool_full=seq3(alpha capacity 3/3, remaining 0), seq7 issues=['E_POOLFULL']
borrow_return=loan-000001 beta->alpha overdue=14 status=returned_late
waterline_diagnostic=remaining=1 eta_ms=1000 window_ms=1000 error_bound_ms=1000
parameter_tightening=v1->v2 cleared=['alpha', 'beta'] stale_warnings=33 current_warnings=0

borrow-cycle: rejected=True business_codes=['E_BORROW']
missing-close: rejected=True business_codes=['E_GAP']
version-mismatch: rejected=True business_codes=['E_VERSION']
```

最终账本统计为 7 个完整批次、28 条事件审计、2 个参数版本、1 次参数迁移和 0 个未完成批次。完整机器可读结果见 `artifacts/demo_summary.json`，测试原始输出见 `artifacts/test_results.txt`，结构化日志见 `artifacts/demo_log.ndjson` 与 `artifacts/crash_run.ndjson`。

## 约束与说明

- 不使用 Docker、容器或外部服务；仅使用本地文件、SQLite 和 `127.0.0.1`。
- 回环套接字只模拟信号确认，不代表真实上游连接。
- 轨迹是确定性的；输入校验失败时不会处理任何事件，也不会产生部分水位。
- `missing_close` 样本通过删除关闭行并保留序号缺口触发 `E_GAP`；正常轨迹允许结束时尚有活跃连接。
- SQLite 账本保存逐批累积快照，因此跨历史批次会出现同一连接或租约的多个快照；恢复只读取最新完整批次，重放查询全部事件审计。
