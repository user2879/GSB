# probelease

面向 Windows 本地环境的确定性回环 UDP 探测会话、租约、暂停/恢复与账本恢复组件。项目不依赖 Docker、云服务或外部网络；全部探针、样本、日志、SQLite 账本和快照都位于本机。

## 交付内容

- `probelease/`：可安装的 Python 包，提供声明加载、探针子进程编排、真实 UDP 轮次、租约状态机、SQLite 原子账本、快照恢复、重放校验和拼接校验。
- `samples/config/`：确定性会话、探针参数与应答确认口径样本。
- `samples/broken/`：声明损坏、口径版本不符、依赖成环样本。
- `probelease/demo.py` 与 `examples/run_demo.py`：完整可重复演示。
- `examples/integration_example.py`：最小集成示例。
- `tests/`：25 个自动化测试。
- `artifacts/results/demo/demo_report.json`：最近一次实际验收报告。
- `artifacts/results/demo/demo.jsonl`：包含单调时钟的结构化事件日志。

## 环境

目标运行环境：

- Windows 10/11 x64。
- CPython 3.11 或更高版本。
- 建议使用虚拟环境。
- NumPy：项目声明 `numpy>=1.24`；本次验证使用 `numpy 1.26.4`。
- 标准库 `sqlite3`、`socket`、`subprocess`、`threading`、`json` 由官方 CPython 安装包提供。
- 样本端口 `18101` 至 `18105`。运行前需确认本机 `127.0.0.1` 的这些 UDP 端口未被占用。

本次实际验证环境为 WSL2/Linux、Python `3.11.0rc1`、NumPy `1.26.4`、SQLite `3.37.2`。Windows 分支已按 `CREATE_NEW_PROCESS_GROUP`、`OpenProcess`/`WaitForSingleObject` 和 `os._exit(137)` 实现，但当前执行环境无法直接启动原生 Windows 10/11，因此 Windows 专属系统调用没有在本次会话内实机执行。

Windows 准备命令示例：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
Get-NetUDPEndpoint -LocalPort 18101,18102,18103,18104,18105 -ErrorAction SilentlyContinue
```

## 快速运行

```bash
python -m pytest -q
python -m probelease.cli demo --root artifacts/results/demo
python examples/integration_example.py
```

演示会生成租约、暂停恢复、拼接、崩溃恢复、快照、重放和负向校验的完整报告。它使用固定样本端口，所以同一时刻不要并行运行两个演示。

## 声明格式

声明由三个文件组成，必须一起加载；任一文件不合法时不会返回部分声明。

### 会话文件

```json
{
  "format_version": 1,
  "session_set_version": 3,
  "ack_policy_version": 7,
  "sessions": [
    {
      "session_id": "alpha",
      "port": 18101,
      "probe_ids": ["alpha_ok_1", "alpha_ok_2", "alpha_drop"],
      "poll_period_ms": 220,
      "datagram_timeout_ms": 70,
      "ack_confirmation_floor_bp": 6000,
      "lease_miss_limit": 2,
      "depends_on": []
    }
  ]
}
```

`ack_confirmation_floor_bp` 使用万分比，范围为 `0..10000`。`lease_miss_limit` 是累计未续期轮数上限。所有端口固定绑定 `127.0.0.1`，不接受外部地址。

### 探针参数文件

```json
{
  "format_version": 1,
  "probe_parameter_version": 11,
  "probes": {
    "alpha_ok_1": {"seed": 1001, "behavior": "normal", "delay_ms": 0},
    "alpha_drop": {"seed": 1003, "behavior": "drop", "delay_ms": 0},
    "late_ok_but_late": {"seed": 3001, "behavior": "delay", "delay_ms": 180},
    "bad_sequence": {
      "seed": 4001,
      "behavior": "normal",
      "delay_ms": 0,
      "sequence_offset": 1
    }
  }
}
```

行为：

- `normal`：立即回送确定性的带序号应答。
- `drop`：不回送。
- `delay`：探针子进程真实等待 `delay_ms` 后再发送，可能自然晚于接收超时。
- `sequence_offset`：可选故障参数，用于稳定产生“应答序号不匹配”结局。

应答序号等于请求周期序号；固定整数种子用于生成确定性响应令牌。编排器从不伪造超时：判定只来自真实 UDP `sendto`/`recvfrom` 和套接字接收截止时间。

### 应答确认口径文件

```json
{
  "format_version": 1,
  "ack_policy_version": 7,
  "ratio_scale": 10000,
  "rounding": "half_up"
}
```

`sessions.ack_policy_version` 必须与口径文件一致，否则报告 `E_GRANT`。确认比例使用四舍五入到整数的万分比定点值：`(ACK 数 * 10000 + 探针数 / 2) // 探针数`。

## 强校验与业务码

加载时按固定顺序检查并指出首个非法字段路径：

- 会话标识非空且唯一。
- 端口为 `1..65535` 的整数且全局不重复；运行时只绑定 `127.0.0.1`。
- 轮询周期和数据报超时为正整数，且超时严格小于周期。
- 应答确认下限为 `0..10000` 的整数。
- `lease_miss_limit` 为正整数。
- 探针集合非空、同一会话内不重复，且每个探针都有参数定义。
- 依赖目标存在；缺失目标属于 `E_SESSION`。
- 依赖图无环；检测到环时使用 `E_DEPEND`。`DependencyError` 同时保留 `outer_code = E_SESSION` 语义。
- 三个文件格式版本一致且为整数 `1`。
- 口径版本不一致时使用 `E_GRANT`。

业务码：

| 业务码 | 含义 |
| --- | --- |
| `E_SESSION` | 声明结构、字段、版本或依赖目标非法 |
| `E_GRANT` | 应答确认口径版本不匹配，或账本/快照描述符冲突 |
| `E_DEPEND` | 会话租约依赖成环 |
| `E_LEASE` | 暂停/终止会话不得签发新租约 |
| `E_FORMAT` | 账本或快照格式版本不支持 |
| `E_LEDGER` / `E_SNAPSHOT` | 账本或快照损坏、读写失败 |
| `E_REPLAY` | 真实 UDP 重放与账本检查点首个字段不一致 |
| `E_SPLICE` | 拼接路径与不中断路径的逻辑账本字节首个差异 |

## 租约状态机

每个会话初始租约版本为 `1`，状态为 `ACTIVE`，周期游标为 `0`。

1. 一轮结束后，ACK 数按万分比计算确认比例。
2. 比例不低于声明下限：续期，租约版本加 `1`，累计未续期轮数清零。
3. 低于下限：版本不变，累计未续期轮数加 `1`。
4. 累计未续期轮数达到 `lease_miss_limit`：该轮落账为 `PAUSED`、原因 `LEASE_EXPIRED`，同事务写入暂停事件。
5. 暂停/终止会话不再签发新租约；暂停期间执行轮次会返回 `E_LEASE`。
6. 下游会话只有在所有上游状态为 `ACTIVE` 且上游最近确认比例不低于其下限时才可推进。上游失效时下游记录 `DEPENDENCY_LOST` 暂停；上游恢复健康时，依赖型暂停可自动恢复并记录 `DEPENDENCY_RESTORED`。

过期判定只使用已提交轮次计数、累计未续期轮数和单调时钟事件记录，不使用墙钟时间差。

## 暂停、恢复与周期游标

暂停有两种入口：

- 已提交轮次之间调用 `pause_session()`：周期游标保持最后一个完整轮次，暂停事件登记下一次待执行周期。
- 发送后接收前调用 `execute_round(..., pause_after_send=True)`：所有已发送数据报以 `DANGLING_UNKNOWN` 落入账本，保留所属探针和真实发送单调时间，不构成完整轮次，不更新周期游标，也不计入任何比例。

恢复不会消耗额外周期，也不改写已持久化的租约版本。恢复后的第一轮从持久化周期游标的下一个周期开始，携带当前租约版本；该轮成功后按正常续期规则产生下一版本。这样暂停本身不改变逻辑轮次序列、租约版本序列或确认比例。

设计说明：暂停/恢复的控制面事件与数据面轮次分开存放。拼接校验比较数据面逻辑账本；暂停原因由 `pause_events` 单独稳定记录并通过演示报告列出。物理发送/接收单调时间、事务 ID 和套接字端口不会进入拼接字节投影，否则两次独立运行的临时端口和时间必然不同。

## 真实 UDP 探针

每个会话启动一个 `probelease.probe_runner` 子进程，在该会话声明的 UDP 端口复用多个探针标识。编排器为每轮创建临时 UDP 套接字，向每个探针发送一条 JSON 数据报，然后以真实接收截止时间收集回复。结局包括：

- `ACK`
- `NO_RESPONSE`
- `SEQ_MISMATCH`
- `PROTOCOL_ERROR`
- `SEND_ERROR`
- `DANGLING_UNKNOWN`
- `UNKNOWN`（强制终止后的未提交残留）

只有 `ACK` 计入确认比例。`NO_RESPONSE` 和 `SEQ_MISMATCH` 使用不同结局和不同理由文本，例如 `timeout:no_response` 与 `sequence_mismatch:expected=1:received=2`。

探针子进程每 100ms 检查父进程是否仍存在。父编排器被强制终止后，探针也会退出，释放 UDP 端口。

## SQLite 原子账本

账本使用 WAL、`synchronous=FULL`、外键和 5 秒 busy timeout。

公开账本表：

- `meta`：格式版本、会话集合版本、探针参数版本、口径版本、内容摘要、声明定义。
- `sessions`：租约版本、周期游标、累计未续期轮数、最近比例、状态和原因。
- `rounds`：完整轮次的确认比例、续期结果、版本、状态与游标结果。
- `datagrams`：已提交的逐探针发送/接收记录和悬空/未知记录。
- `round_attempts`：已完成、暂停或恢复时判定未知的尝试历史。
- `pause_events`：暂停、恢复、终止事件。
- `recovery_events`：进程重启后的未提交轮次恢复事件。

未完成轮次先写入独立的 `_stage_attempts` 和 `_stage_datagrams`。完整轮次的落账在一个 SQLite 事务中同时完成：

- 插入 `round_attempts` 完成记录。
- 插入 `rounds`。
- 插入该轮全部 `datagrams`。
- 更新会话租约、周期游标、状态和原因。
- 写入该轮自动暂停事件。
- 删除对应 staging 行。

这些最终记录在 `COMMIT` 前对其他连接不可见，提交后作为整体可见。空中暂停、崩溃恢复和快照写入也使用单事务。

## 强制终止与重启

CLI 的 `--crash-before-commit-cycle N` 会在第 N 轮真实发送/接收结束、最终事务开始前调用 `os._exit(137)`。该进程不会输出完成报告，也不会生成“强制终止时的退出报告”。

重启后：

- 只读取已完整提交的 `rounds` 和最终 `datagrams`。
- staging 中状态为 `RUNNING` 的尝试被整体转为 `UNKNOWN`。
- 每个 staging 数据报在公开账本中记录为 `UNKNOWN`，理由 `uncommitted_round_unknown_rerun_required`。
- 写入 `recovery_events`，业务语义为 `E_RERUN_REQUIRED`。
- 周期游标不前进，调用方必须用下一尝试号重跑同一周期。

本次演示中强制终止时 stdout/stderr 均为空、退出码为 `137`，重启后第 2 轮 3 个探针数据报均被标记为未知。

## 快照与整体替换

快照包含：

- 整数 `snapshot_format_version = 1`。
- `format_version`、`session_set_version`、`probe_parameter_version`、`ack_policy_version`。
- 声明内容摘要 `content_digest`。
- 会话与探针定义。
- 完整账本状态及 `state_digest`。

恢复规则：

1. 快照描述符与当前声明不一致：返回 `E_GRANT`，不混合加载。
2. 快照描述符匹配当前声明，但磁盘账本的版本、内容摘要或状态摘要不同：整体替换会话租约、轮次历史、数据报、暂停记录、恢复记录和周期游标，不做增量合并。
3. 替换前通过 SQLite backup API 生成旧账本备份，并设置为只读。
4. 报告稳定排序的新增、移除和变化的会话及探针。
5. 快照和磁盘状态完全一致时不重复替换。

## 重放校验

`replay_verify()` 会启动新的确定性探针子进程，按账本中已提交的周期重新执行真实 UDP 轮次。每一轮比较：

- 每个探针的最新 ACK 序号。
- 每个探针当前结局和理由。
- 确认比例定点值。
- 请求租约版本和结果租约版本。
- 会话状态和暂停原因。

首个差异抛出 `E_REPLAY`，包含会话标识、探针标识、周期、字段、期望值、实际值和依据。重放前必须停止占用样本端口的编排器。演示中篡改第 2 轮比例后，得到 `sessions[alpha].ratio_bp` 差异和 `E_REPLAY`。

## 拼接校验

`compare_splice_ledgers()` 比较两个账本的稳定逻辑 JSONL 字节投影。投影逐轮包含：

- 会话、周期、尝试号。
- ACK 数、总数、确认比例。
- 请求/结果租约版本、是否续期。
- 轮后状态、暂停原因。
- 每个探针的结局、理由、应答序号。

投影排除物理单调时间、随机事务 ID 和临时源端口。比较使用规范 JSON（键排序、无多余空白）并计算 SHA-256；差异抛 `E_SPLICE` 并定位首个会话/周期/探针/字段。

本次验证中，不中断 4 轮与“2 轮 → 暂停 → 重启进程 → 恢复 → 2 轮”的 1692 字节逻辑账本完全一致：

```text
545fb0a339ffab5a5ed3629f5e558e969b8199bc0cfdbf858f2ebc6cd4560cf4
```

篡改拼接副本第 3 轮比例后返回 `E_SPLICE`，字段路径为 `sessions[alpha].cycles[3].ratio_bp`。

## Python API

```python
from probelease import LeaseOrchestrator, load_declarations

declarations = load_declarations(
    "samples/config/sessions.json",
    "samples/config/probes.json",
    "samples/config/ack_policy.json",
)

with LeaseOrchestrator(declarations, "artifacts/my-ledger.sqlite3") as orchestrator:
    first = orchestrator.execute_round("alpha")
    orchestrator.pause_session("alpha", "MAINTENANCE")
    orchestrator.resume_session("alpha")
    second = orchestrator.execute_round("alpha")
    lease = orchestrator.query_lease("alpha")
    orchestrator.save_snapshot("artifacts/my-snapshot.json")
```

其他公开入口：

- `ProbeSupervisor(declarations).start()/stop()`：单独管理探针子进程。
- `LeaseOrchestrator.restore_ledger(snapshot)`：关闭探针和账本后整体恢复并重新打开。
- `restore_from_snapshot(...)`：无编排器时的底层整体替换接口。
- `replay_verify(...)`：真实 UDP 重放校验。
- `compare_splice_ledgers(...)`：逻辑账本字节比对。

## CLI

```bash
# 校验
python -m probelease.cli validate \
  --sessions samples/config/sessions.json \
  --probes samples/config/probes.json \
  --ack-policy samples/config/ack_policy.json

# 运行 alpha 4 轮并保存快照
python -m probelease.cli run \
  --sessions samples/config/sessions.json \
  --probes samples/config/probes.json \
  --ack-policy samples/config/ack_policy.json \
  --ledger artifacts/manual/ledger.sqlite3 \
  --session alpha --cycles 4 \
  --snapshot artifacts/manual/snapshot.json

# 重放
python -m probelease.cli replay \
  --sessions samples/config/sessions.json \
  --probes samples/config/probes.json \
  --ack-policy samples/config/ack_policy.json \
  --ledger artifacts/manual/ledger.sqlite3 --session alpha

# 拼接比较
python -m probelease.cli splice \
  --sessions samples/config/sessions.json \
  --probes samples/config/probes.json \
  --ack-policy samples/config/ack_policy.json \
  --left artifacts/baseline.sqlite3 --right artifacts/spliced.sqlite3

# 快照恢复
python -m probelease.cli restore \
  --sessions samples/config/sessions.json \
  --probes samples/config/probes.json \
  --ack-policy samples/config/ack_policy.json \
  --ledger artifacts/manual/ledger.sqlite3 \
  --snapshot artifacts/manual/snapshot.json
```

三类故障注入：

```bash
python -m probelease.cli validate ... --inject corrupt-session   # E_SESSION
python -m probelease.cli validate ... --inject ack-version       # E_GRANT
python -m probelease.cli validate ... --inject dependency-cycle  # E_DEPEND
```

所有 CLI 结果都会写入 `--log` 指定的 JSONL。每条日志包含 `event`、`code`、`monotonic_ns`、UTC 时间和上下文字段。

## 实际验证结果

最近一次完整演示命令：

```bash
python -m probelease.cli demo --root artifacts/results/demo
```

结果摘要（详细值见 `artifacts/results/demo/demo_report.json`）：

- 自动化测试：`25 passed`。
- 正常续期：`alpha` 连续 3 轮确认比例 `6667 bp`，租约版本依次 `2 → 3 → 4`。
- 租约过期：`lossy` 两轮比例均为 `5000 bp`，低于 `7500` 下限；第 1 轮累计 miss=1 仍 `ACTIVE`，第 2 轮达到上限后 `PAUSED / LEASE_EXPIRED`，版本保持 `1`。
- 依赖派生：`child` 因 `lossy` 失效进入 `PAUSED / DEPENDENCY_LOST`，未签发新租约。
- 延迟应答：`late` 的真实发送到超时跨度至少 55ms，结局 `NO_RESPONSE / timeout:no_response`，随后过期暂停。
- 错序应答：`mismatch` 收到序号 `2`（期望 `1`），结局 `SEQ_MISMATCH`，理由 `sequence_mismatch:expected=1:received=2`，随后过期暂停。
- 强制终止会话：`terminate_session()` 后状态 `TERMINATED`，不得续租。
- 空中暂停：3 条数据报为 `DANGLING_UNKNOWN`，含真实发送单调时间，游标保持 `0`；恢复后同周期以 attempt=2 重跑，只有新轮次计入 `6667 bp`。
- 跨进程拼接：1692 字节逻辑账本 SHA-256 完全一致，见上方摘要。
- 强制进程终止：退出码 `137`，无 stdout/stderr，不生成退出报告；重启只看到第 1 轮完整提交，第 2 轮 3 个数据报标为 `UNKNOWN` 并要求重跑。
- 快照恢复：整体替换成功，生成只读备份；恢复后从第 2、3 轮继续。
- 重放：3 轮、9 个探针检查点全部一致。
- 负向重放：篡改后 `E_REPLAY`，定位 `sessions[alpha].ratio_bp`，周期 `2`。
- 负向拼接：篡改后 `E_SPLICE`，定位 `sessions[alpha].cycles[3].ratio_bp`。
- 故障注入：`E_SESSION`（`sessions`）、`E_GRANT`（`sessions.ack_policy_version`）、`E_DEPEND`（`sessions.sessions[4].depends_on[0]` 或注入形成的等价首个依赖边）。

## 运行约束

- 同一账本只允许一个编排器写入；读取器可使用独立 SQLite 连接。
- 使用固定样本端口时不要并行运行多个探针组。
- Windows 定时器精度和系统负载可能影响亚 20ms 超时；样本使用 60/70ms 超时与 180ms 延迟，留出稳定边界。
- 真实 UDP 只发往 `127.0.0.1`，没有远程目标或外网访问代码路径。
