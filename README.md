# DriftWatch Local

面向本地 Windows 环境的部署漂移观测归并诊断组件。它在 `127.0.0.1` 高位端口上编排若干确定性观测代理子进程，以 JSON 行协议采集 CPU 配额、内存页数与进程数观测，把多代理记录归并为每主机每指标的权威值，再用部署快照容差做定点漂移判定，并将批次、结论、误差界、传播信息、进度和会话快照写入 SQLite。

本仓库同时包含可集成的 Python 领域组件、CLI、可选 Tk 查看器、固定端口示例、强杀恢复演示和自动化测试。核心只依赖 Python 标准库；`numpy` 与 `tkinter` 仅作为环境检查/可选桌面查看器能力使用。

## 快速开始（Windows PowerShell）

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e . pytest numpy
python examples/check_environment.py
python -m pytest -q
python -m driftwatch validate `
  --snapshots examples/snapshots.json `
  --agents examples/agents_v1.json `
  --criteria examples/criteria.json `
  --check-ports
python examples/run_demo.py --reset
python examples/crash_recovery_demo.py
```

官方 CPython Windows 安装包自带 Tcl/Tk；若在精简 Linux/WSL 中运行，需要系统包 `python3-tk`/`tk`。核心采集、归并、账本和测试不依赖 GUI。示例端口为 `39411`、`39412`、`39413`、`39414`，执行前会做真实绑定检查。

## 目录

- `driftwatch/config.py`：快照链、代理清单和判据的加载与聚合校验。
- `driftwatch/protocol.py`：JSON 行协议、真实套接字超时、字段校验。
- `driftwatch/deterministic.py`、`agent_runtime.py`、`agent_main.py`：确定性代理与整数聚合。
- `driftwatch/processes.py`：Windows/跨平台子进程编排、端口预检和握手。
- `driftwatch/scheduler.py`：全局并发、每代理并发、排队与重试登记。
- `driftwatch/merge.py`：多源权威值、来源、口径版本和偏离量。
- `driftwatch/fixedpoint.py`：十进制缩放定点比较、三段误差界、饱和与决胜规则。
- `driftwatch/propagation.py`：代理/口径/快照演进的影响传播。
- `driftwatch/ledger.py`：SQLite 原子批次、会话快照、恢复和差异记录。
- `driftwatch/engine.py`：采集编排、提交、恢复重放。
- `driftwatch/cli.py`、`gui.py`：CLI 与可选 Tk 查看器。
- `examples/`：固定种子示例配置、完整演示和强杀恢复演示。
- `docs/TECHNICAL.md`：算法、事务边界和协议细节。
- `docs/VERIFICATION.md`：本环境实际验证记录。

## CLI

```powershell
# 只校验声明；--check-ports 会尝试绑定每个代理端口
python -m driftwatch validate --snapshots <snapshots.json> --agents <agents.json> --criteria <criteria.json> --check-ports

# 采集、归并、判定并原子提交批次；--verify-replay 会先重放所有已提交批次
python -m driftwatch run --snapshots <snapshots.json> --agents <agents.json> --criteria <criteria.json> --db driftwatch.sqlite3 --verify-replay

# 查看账本状态 / 当前活动视图
python -m driftwatch status --db driftwatch.sqlite3
python -m driftwatch show --db driftwatch.sqlite3 --drift-only

# 仅重放已提交批次；使用账本内保存的声明
python -m driftwatch replay --db driftwatch.sqlite3

# 使用账本内声明恢复：中止 RUNNING、重放已提交、重采未知项并提交
python -m driftwatch recover --db driftwatch.sqlite3

# 可选 Tk 查看器
python -m driftwatch gui --db driftwatch.sqlite3
```

`run --strict` 在出现 `REJECT` 或 `UNKNOWN` 行时返回退出码 `1`。配置错误返回 `2`，运行期端口/超时/记录错误返回 `3`，重放不一致返回 `4`。

## 领域组件 API

```python
from driftwatch import DriftEngine, Ledger, load_inputs

chain, agents, criteria = load_inputs(
    "snapshots.json",
    "agents.json",
    "criteria.json",
)

with DriftEngine(chain, agents, criteria, "driftwatch.sqlite3") as engine:
    report = engine.run(verify_replay=True)
    print(report.to_dict())

with Ledger("driftwatch.sqlite3") as ledger:
    current = ledger.active_view()
    affected = ledger.affected(current["generation"])
```

`DriftEngine.run()` 返回活动代、完整换代差异、受影响项、恢复项、已提交批次、重放结果、游标和视图摘要。原始观测、逐次尝试、权威来源与偏离、漂移/误差界可从 SQLite 账本或 `Ledger.batch_results()`、`Ledger.session_snapshot()` 读取。

## 输入声明

### 部署快照链

```json
{
  "schema_version": 1,
  "snapshots": [
    {
      "seq": 1,
      "parent_seq": null,
      "hosts": {
        "edge-a": {
          "expected": {"cpu_quota": 120000, "memory_pages": 70000, "processes": 600},
          "tolerance_bps": {"cpu_quota": 1000, "memory_pages": 800, "processes": 1000}
        }
      }
    },
    {
      "seq": 2,
      "parent_seq": 1,
      "hosts": {
        "edge-a": {
          "expected": {"cpu_quota": 103000, "memory_pages": 64000, "processes": 512},
          "tolerance_bps": 500
        }
      }
    }
  ]
}
```

`tolerance_bps` 可写一个整数（对本主机三个指标相同），也可写三指标映射。校验会整体收集违规项并以 `E_SNAPSHOT` 拒绝：非整数/非正期望值、容差为负或大于 `10000`、序号重复、父引用缺失、父子环、未知指标等。

### 观测代理清单与判据

```json
{
  "schema_version": 1,
  "criteria": {
    "global_concurrency": 3,
    "per_agent_concurrency": 1,
    "min_sources": 2,
    "scale_exp": 3,
    "socket_timeout_ms": 2000,
    "handshake_timeout_ms": 5000,
    "max_retries": 2,
    "retry_backoff_ms": 25,
    "boundary_rule": "tolerance_floor"
  },
  "agents": [
    {
      "agent_id": "agent-alpha",
      "host": "127.0.0.1",
      "port": 39411,
      "seed": 101,
      "aggregation": {"kind": "median", "window": 7, "version": "median-v1"},
      "hosts": ["edge-a", "edge-b"],
      "metrics": ["cpu_quota", "memory_pages", "processes"],
      "per_agent_concurrency": 1,
      "profiles": {
        "edge-a": {
          "cpu_quota": {"base": 106000, "spread": 1200},
          "memory_pages": {"base": 62200, "spread": 700},
          "processes": {"base": 508, "spread": 8}
        }
      }
    }
  ]
}
```

校验以 `E_AGENT` 整体拒绝：非 `127.0.0.1` 字面量、端口不在高位范围或全局重复、重复代理标识、非整数种子、未知主机/指标、非法聚合窗口或判据参数等。代理支持 `median`、`mean_floor`、`trimmed_mean`、`minimum`、`maximum` 整数聚合。

### 观测记录

代理回传记录至少包含：

```json
{
  "agent_id": "agent-alpha",
  "observation_seq": 101,
  "host_id": "edge-a",
  "metric_id": "cpu_quota",
  "value": 106320,
  "aggregation_version": "median-v1",
  "monotonic_ms": 43149554
}
```

`host_id` 是对“按主机归并”必要的显式扩展。缺失、类型错误、序列/主机/指标不匹配都报 `E_RECORD`；真实套接字超时报 `E_TIMEOUT`，且不会产生观测。

## 核心语义

- 权威值按 `observation_seq` 降序、再按 `agent_id` 字典序升序选择。所有来源值、聚合版本、来源序号和相对权威值的带符号偏离量都保留。
- 有效来源按代理去重；低于 `min_sources` 时整行标为 `SPARSE`/`E_SPARSE`，权威值和漂移判定均为空，绝不以单条观测代替。
- 资源期望是上限，采用单边上行漂移：低于期望不会触发拒绝。
- 会话缩放 `S = 10^scale_exp`。漂移量为整数差，容差额度使用向下取整除法；有符号报告比例向零截断；所有固定点运算在 `±(2^63-1)` 饱和。
- 三段误差界为观测量化误差 `S`、容差比较误差 `1`、归并偏离误差 `max|source-authority|*S`；合成为三者之和。
- `excess > allowed + composite` 明确拒绝；`excess + composite <= allowed` 明确接受；其余为受误差影响边界。`tolerance_floor` 在边界按 `excess <= allowed` 决胜，`conservative_reject` 直接拒绝。
- 代理集合、代理规格/覆盖、聚合口径、快照新增/内容或容差、缩放/决胜口径、`min_sources` 变化都会推导受影响的主机指标并在旧代上标 `stale`。新声明创建新的活动代，当前视图从空开始完整重算，不做增量修补。
- 每个快照采集是一个批次：先独立事务登记 `RUNNING`，再一次性原子写入观测、尝试记录、权威归并、漂移/误差界和游标并改为 `COMMITTED`。强杀后只恢复已提交批次；`RUNNING` 批次转为 `ABORTED`，其计划主机指标登记为必须重采。
- 重放只比较确定性语义字段；代理单调时钟 `monotonic_ms` 不作为可复现字段。首个不一致以 `E_REPLAY` 返回 `snapshot_seq`、`host_id`、`metric_id`、`field`、`expected`、`actual`。

## 示例与验证

`examples/run_demo.py` 展示：

1. v1 三个快照、三代理；`edge-c` 只有单源，因此 9 行为 `E_SPARSE`；快照 2 有一个明确拒绝和一个受误差影响的边界拒绝。
2. v2 新增代理、移除代理、改变 `agent-bravo` 聚合版本并增加快照 4；输出 36 个受影响主机指标，创建第 2 代并完整重算。
3. 对 v2 的 4 个已提交批次重放 36 行，结论摘要一致。

`examples/crash_recovery_demo.py` 在快照 2 批次登记后强制杀死协调进程，再演示只基于已提交账本恢复、9 个未知项重采和已提交批次重放。它明确输出 `exit_report_claimed: false`：强杀没有退出报告，只有账本事实。

当前仓库的实际执行结果见 `docs/VERIFICATION.md`。完整算法和误差公式见 `docs/TECHNICAL.md`。
