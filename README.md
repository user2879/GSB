# TCP 连接状态与拥塞诊断工作台

面向 Windows 本地环境的确定性教学/诊断工作台。它在 `127.0.0.1` 的高位 TCP 端口运行脚本化对端，用本地 JSON 驱动确认、重复确认、丢包、延迟和乱序；客户端侧以整数定点推进连接状态、拥塞窗口、慢启动阈值、重复确认计数、在途字节数和 RTT/RTO。

工作台的核心不是“重新算一遍所有东西”，而是把每个结论做成带版本、带依赖事件、带配置版本的可校验记录：

- 事件先以 `pending` 写入 SQLite，推导成功后才与受影响结论一起提交。
- 新事件只更新实际变化的状态项、统计和预算；未受影响项的 `value_json`、版本号、依据事件和依赖列表逐字节不变。
- 配置或脚本换代时，旧结论整体归档为只读历史，新结论集整体替换，绝不与旧值增量合并。
- 快照保存逐事件重放点，恢复校验能定位到第一个不一致的事件序号和字段，并以 `E_REPLAY` 报明依据。

## 环境准备（Windows）

要求 Python 3.11 或更高版本；Windows 官方 Python 已自带 Tkinter。

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

也可直接运行 `scripts/setup_windows.ps1`。不依赖 Docker、WSL、外部站点或网络账号。

当前交付验证容器是 Linux，系统 Python 缺少 `tkinter` 软件包；因此 Tkinter 窗口无法在该容器内实际弹出，界面逻辑通过无界面渲染模型验证。Windows 官方 Python 的标准库包含 Tkinter，不额外安装依赖。

## 快速开始

```powershell
# 1. 强校验配置与脚本
python -m tcp_workbench.cli validate --config samples/session.json

# 2. 新建账本，逐条灌入确定性事件，并做增量/全量逐项比对
python -m tcp_workbench.cli run `
  --db var\tcpdiag.sqlite3 `
  --config samples\session.json `
  --events samples\events.json `
  --snapshot-name baseline `
  --reset

# 3. 真实 127.0.0.1 TCP 回环对端会话
python -m tcp_workbench.cli loopback `
  --db var\loopback.sqlite3 `
  --config samples\session.json `
  --segments 7 `
  --snapshot-name loopback `
  --reset

# 4. 无界面查看工作台模型；在 Windows 上去掉 --headless 打开 Tkinter
python -m tcp_workbench.cli ui --db var\tcpdiag.sqlite3 --headless
python -m tcp_workbench.cli ui --db var\tcpdiag.sqlite3

# 5. 恢复检查
python -m tcp_workbench.cli recover --db var\tcpdiag.sqlite3

# 6. 全量重算复核
python -m tcp_workbench.cli compare --db var\tcpdiag.sqlite3
```

一键无界面验收：

```powershell
python scripts\verify_acceptance.py
python -m pytest -q
```

## 样本

- `samples/session.json`：ISN、窗口上限、MSS、预算和拥塞参数。
- `samples/peer_script.json`：ACK、重复 ACK、丢包、延迟 10 ms、乱序应答。
- `samples/events.json`：20 条确定性事件，覆盖握手、慢启动、三次重复确认、快速重传、超时、Karn 排除和五个连接状态。
- `samples/snapshots/*.json`：两个历史版本的快照种子；`seed-snapshots` 会重放事件后写入 SQLite。
- `samples/versions/*`：配置版本 2 与脚本版本 2。
- `samples/invalid/*`：强校验和 `E_SCRIPT` 注入样本。

快照样本本身不是 SQLite 文件，而是可审计的种子描述。灌库命令：

```powershell
python -m tcp_workbench.cli seed-snapshots --db var\tcpdiag.sqlite3 --directory samples\snapshots
```

## 命令与业务码

| 命令 | 作用 |
| --- | --- |
| `validate` | 加载并强校验配置/脚本，任何字段错误整体拒绝 |
| `run` | 灌入事件脚本，增量推导，最后与全量重放逐项比较 |
| `loopback` | 启动真实 `127.0.0.1` TCP 对端并把应答转成事件 |
| `recover` | 重开账本，pending 事件标为 unknown，仅用完整事件恢复 |
| `retry` | 按账本中保留并校验过的 unknown 事件载荷重新推导 |
| `evolve` | 对单调递增的新配置/脚本版本整体替换结论集 |
| `compare` | 检查当前逐项结论与全量重算是否完全一致 |
| `snapshot save/list/verify/restore` | 快照保存、列表、逐事件重放校验、整集恢复 |
| `seed-snapshots` | 导入历史版本快照样本 |
| `ui` | Tkinter 工作台；`--headless` 输出稳定文本视图 |
| `inject script/port/version` | 分别注入 `E_SCRIPT`、`E_PORT`、`E_VERSION` 并写结构化日志 |

稳定业务码：`E_CONFIG`、`E_CONNECT`、`E_SCRIPT`、`E_PORT`、`E_VERSION`、`E_REPLAY`、`E_STATE`、`E_BUDGET`。命令行失败时 stderr 输出 JSON，同时追加 JSON-lines 到 `logs/tcpdiag.jsonl`（可用全局 `--log` 改变路径）。

异常注入示例：

```powershell
python -m tcp_workbench.cli --log logs\inject.jsonl inject script `
  --config samples\session.json --script samples\invalid\peer_script_corrupt.json

python -m tcp_workbench.cli --log logs\inject.jsonl inject port `
  --config samples\session.json --port 43177

python -m tcp_workbench.cli --log logs\inject.jsonl inject version `
  --db var\tcpdiag.sqlite3 --snapshot-id 1
```

## 界面内容

Tkinter 工作台按固定顺序显示：

1. 连接状态、拥塞窗口、慢启动阈值、重复确认计数、在途字节/段数、SRTT 与 RTO、当前 `C{config}/S{script}` 版本、恢复状态。
2. 十项逐项结论：值、项目版本号、依据事件序号、本次是否受影响、配置/脚本版本、依赖事件序号。
3. 快照列表：ID、名称、版本、事件数、内容摘要。
4. 结构化日志末尾。

按钮支持刷新、保存快照、校验所选快照、整集恢复快照和增量/全量比对。

## 设计要点

- **真值来源**：SQLite `events` 表；每条事件含载荷摘要，状态为 `pending/complete/unknown`。
- **时钟**：生产使用 `time.monotonic_ns() // 1_000_000`；确定性样本使用受控的单调毫秒计数器，不使用墙钟。
- **两阶段提交**：`begin_event` 先提交 pending，结论事务成功后再把事件改为 complete；强制终止后不会把半成品当完整事件。
- **增量投影**：状态机先推进候选状态，再只物化实际变化项；未变化结论在数据库中不执行 UPDATE。
- **RTT 定点**：缩放因子 8，除法向下截断；首次样本 `RTTVAR=SRTT/2`；RTO=`SRTT+4*RTTVAR` 并夹在配置上下界。重传段确认按 Karn 规则排除。
- **版本防护**：内容摘要与版本号同时登记；相同内容禁止换版本，变更内容必须提高对应版本。
- **整体替换**：旧结论复制进 `conclusion_history` 并固定 `stale=1`；当前表被新一轮完整结论集替换。
- **快照重放**：快照保存当前结论、逐事件重放点、事件序列、配置/脚本原文与内容摘要；校验会逐个事件比较值、项目版本、依赖事件和配置/脚本版本。

更细的依赖图、定点约定、事务时序和重放算法见 `docs/TECHNICAL.md`。实际运行结果见 `docs/VALIDATION.md`。

## 测试

```powershell
python -m pytest -q
```

测试覆盖：配置聚合错误、脚本可达性、回环 ACK/重复 ACK/丢包/延迟/乱序、慢启动与拥塞避免、三次重复确认、超时回退、Karn、事件级增量/全量一致性、未受影响数据库行逐字节不变、版本替换与只读归档、真实子进程强制终止恢复、快照缺失版本、`E_REPLAY` 首差异定位、三类异常结构化日志、稳定 UI 渲染顺序。

## 教学边界

回环对端是运行在真实 TCP Socket 上的确定性应用层协议封装；连接与拥塞状态机、窗口推进与 RTT 估计由本工具显式模拟，不依赖操作系统内核的拥塞控制实现。脚本中的延迟使用本地睡眠模拟，乱序通过把 `reorder=true` 的应答推迟到同批末尾实现，因此行为可重复、可审计。
