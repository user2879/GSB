# 实际验证记录

验证日期：2026-11-23（Asia/Shanghai）  
工作目录：`/home/zsl/feedback/uidfc5909d3-default-136`

## 环境

| 项目 | 实际值 |
|---|---|
| 平台 | Linux 6.18.33.2-microsoft-standard-WSL2 x86_64（WSL2） |
| Python | 3.11.0rc1 |
| SQLite | 3.37.2（Python 标准库 `sqlite3`） |
| numpy | 1.26.4 |
| tkinter | 当前 WSL 缺少 `libtk8.6.so`，不可导入 |
| 网络范围 | 仅 `127.0.0.1` 高位端口；无外部网络目标 |

说明：本环境不是 Windows 原生会话，因此 `CREATE_NO_WINDOW`、`SO_EXCLUSIVEADDRUSE` 等 Windows 分支已实现但未在 Windows 内核上执行；Linux/WSL 分支、子进程、真实 TCP、强杀和恢复均已执行。目标环境使用官方 CPython Windows 安装包时应自带 Tk；当前 WSL 的最小 Python 安装缺少 Tk 动态库，所以可选的 `driftwatch.gui` 未做窗口级验证。核心组件、CLI、账本、代理和测试均不导入 Tk。

```text
$ python examples/check_environment.py
{
  "core_ready": true,
  "numpy": "1.26.4",
  "platform": "Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.35",
  "python": "3.11.0rc1",
  "python_executable": "/home/zsl/.feedback/python-base/bin/python3",
  "sqlite": "3.37.2",
  "tkinter": "unavailable: libtk8.6.so: cannot open shared object file: No such file or directory"
}
```

## 自动化测试

命令：

```text
python3 -m pytest -q
..........................                                               [100%]
26 passed in 8.39s
```

覆盖点：

- 快照序号重复、父缺失、环、容差越界和多违规聚合。
- 代理非字面量回环地址、重复端口、非整数种子和未知主机。
- 真实代理子进程启动、确定性重复观测。
- 端口占用 `E_PORT`、真实握手超时 `E_TIMEOUT`、缺字段 `E_RECORD`。
- 全局/每代理并发上限、排队时间戳、首次与重试独立登记。
- 权威值稳定排序、去重来源、稀疏拒绝、定点接受/拒绝/边界决胜和饱和。
- 代理增删、口径变化、快照新增的传播。
- SQLite 未提交批次恢复、重采清除、版本换代和会话摘要损坏换代。
- 端到端多代理采集、归并、判定、重放。
- 强杀协调进程后仅恢复已提交批次、未提交批次零观测、未知项重采和重放。
- 篡改账本字段后 `E_REPLAY` 定位到首个快照/主机/指标/字段。

## 完整示例 v1/v2

命令：

```text
python3 examples/run_demo.py --db <temporary.sqlite3> --reset
```

实际结果：

| 阶段 | 代号 | 换代 | merged | drift | ACCEPT | REJECT | UNKNOWN/E_SPARSE | error_affected | 重放 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| v1 三快照、三代理 | 1 | 否 | 27 | 27 | 16 | 2 | 9 | 1 | 未请求 |
| v2 代理/口径/快照演进 | 2 | 是 | 36 | 36 | 34 | 2 | 0 | 1 | 未请求 |
| v2 已有账本重放 | 2 | 否 | 36 | 36 | 34 | 2 | 0 | 1 | 4 批 / 36 行一致 |

v2 换代实际产生 36 个受影响主机指标，差异中稳定包含：

- `agent_added`: `agent-delta`
- `agent_removed`: `agent-charlie`
- `aggregation_version_changed`: `agent-bravo: mean-v1 -> mean-v2`
- `snapshot_added`: 快照 `4`

v1 的 `edge-c` 仅由 `agent-bravo` 覆盖，低于 `min_sources=2`，因此 3 快照 × 3 指标共 9 行为 `E_SPARSE`，权威值为空。v2 中 `edge-c` 由 `agent-bravo` 和 `agent-delta` 覆盖，稀疏消失。

## 强制终止恢复

命令：

```text
python3 examples/crash_recovery_demo.py --db <temporary.sqlite3>
```

实际关键输出：

```text
batches_before_recovery:
  snapshot 1  COMMITTED
  snapshot 2  RUNNING

batches_after_recovery:
  snapshot 1  COMMITTED
  snapshot 2  ABORTED

recovered_unknown: 9
recovery_source: ledger_declarations
replayed_committed_batches:
  batch_count: 1
  matched_rows: 9
  ok: true

final_view:
  merged_rows: 27
  drift_rows: 27
  accepted: 16
  rejected: 2
  unknown: 9
  sparse: 9
final_snapshots: [1, 2, 3]
exit_report_claimed: false
```

强杀发生在快照 2 的 `batch_begun` 之后。数据库只留下快照 1 的 `COMMITTED` 和快照 2 的 `RUNNING`；快照 2 未提交观测数为 0。恢复把 9 个计划主机指标标记为必须重采，重放快照 1 的 9 行一致，随后重采快照 2、3 并重建活动视图。

## 重放不一致定位

测试主动将首个漂移行的 `decision/status` 改为与确定性重算不符。`replay_ledger()` 返回 `E_REPLAY`，定位字段为：

```text
snapshot_seq = 1
host_id = host-a
metric_id = <首个字典序指标>
field = drift.decision 或 drift.status
```

测试确认错误详情包含 `expected`、`actual` 和 `batch_id`。

## 未验证项

- 未在原生 Windows 上执行：PowerShell 脚本、Windows 控制台窗口隐藏、Windows `OpenProcess`/`WaitForSingleObject` 父进程检测、Windows 独占端口语义。
- 未打开 Tk 窗口：当前 WSL 缺少 `libtk8.6.so`。GUI 模块在缺少 Tk 时返回结构化 `E_GUI`，不影响核心功能。
