# MeshGov 本地服务网格流量治理工作台

这是一个面向 Windows 本地环境、完全离线可运行的确定性服务网格工作台。它从 JSON 加载服务/路由规则，在全局并发预算与每服务配额下并行调度本地回环上游，执行确定性权重选择、重试和熔断，并将审计轨迹写入 SQLite 与 JSONL，最后发布完整结果快照并通过 Tkinter 展示。

核心实现只依赖 Python 3.11+ 标准库，便于无依赖验收；目标 Windows 环境预装的 `numpy` 与 Tkinter 由 `doctor` 命令检查。当前仓库已在 WSL/Python 3.11 上完成真实回环端到端验收，Tk 界面因容器无 Tk/显示服务而以同一视图模型做无头验证。

## 交付内容

- `meshgov/rules.py`：严格 JSON 规则加载、字段路径校验、原始内容 SHA-256 版本。
- `meshgov/selector.py`：按端口排序的显式 Smooth Weighted Round-Robin，按请求输入顺序与尝试层级预规划。
- `meshgov/budget.py`：全局/服务配额原子获取、排队计时、后台合并预算持久化。
- `meshgov/circuit.py`：关闭/打开/半开状态机、真实单调时钟冷却、单探针。
- `meshgov/scheduler.py`：并行尝试、重试预算、熔断快速失败、完整尝试记录。
- `meshgov/ledger.py`：SQLite WAL 账本、请求检查点、配额状态、熔断状态、恢复查询。
- `meshgov/verify.py`：终态/并发/熔断冷却/重试/权重证据校验及首个稳定偏差。
- `meshgov/snapshot.py`：不可变完整快照、原子替换 `latest.json`、版本与路由差异。
- `meshgov/gui.py` + `viewmodel.py`：Tkinter 工作台与稳定排序视图模型。
- `meshgov/upstream.py`：确定性回环 HTTP 上游（成功/失败/抖动/慢响应）。
- `meshgov/acceptance.py`：整套自动验收与 JSON/Markdown 报告。
- `samples/`、`tests/`、`scripts/`：样本、21 个单元测试及运行辅助脚本。

## Windows 环境

1. 安装官方 Python 3.11 或更高版本，确认安装时包含 Tcl/Tk。
2. 确认 `numpy` 可导入：`python -c "import numpy; print(numpy.__version__)"`。
3. 在仓库根目录执行：

```powershell
python --version
python -m meshgov doctor --workspace . --report artifacts/reports/doctor.json
```

`doctor` 输出 Python、numpy、Tkinter、SQLite、工作区写权限与平台检查。Linux/WSL 验收环境缺少 numpy/Tk 时二者显示 `warn`，但核心命令与无头视图模型仍可完整运行；Windows 目标环境应显示 `pass`。

## 快速运行

### 一键样本（运行后打开工作台）

```powershell
.\run_demo.bat
```

或在 PowerShell 中分步执行：

```powershell
python scripts/check_ports.py --config samples/upstreams.json
python -m meshgov run `
  --rules samples/rules.json `
  --requests samples/requests.json `
  --db artifacts/mesh.db `
  --log artifacts/logs/events.jsonl `
  --snapshots artifacts/snapshots `
  --report artifacts/reports/run.json `
  --round-id round-v1 `
  --upstream-config samples/upstreams.json `
  --start-upstreams
python -m meshgov gui --snapshot artifacts/snapshots/latest.json
```

`--start-upstreams` 让模拟上游与调度器同生命周期运行。也可开两个终端，分别启动长期上游和调度：

```powershell
# 终端 1
python scripts/start_upstreams.py --config samples/upstreams.json

# 终端 2（去掉 --start-upstreams）
python -m meshgov run --rules samples/rules.json --requests samples/requests.json `
  --db artifacts/mesh.db --log artifacts/logs/events.jsonl `
  --snapshots artifacts/snapshots --report artifacts/reports/run.json
```

样本端口为 `28101/28102/28111/28112/28121`，慢上游端口为 `38201`。运行前检查命令会在 Windows 使用独占绑定语义。若端口被占用，预检和自动上游启动均返回 `E_UPSTREAM`；也可用 `--inject upstream_unreachable` 专门验证该链路。

## 规则与请求格式

```json
{
  "schema_version": 1,
  "limits": {"global_concurrency": 6, "default_service_quota": 2},
  "services": [
    {
      "name": "users",
      "concurrency_quota": 3,
      "instances": [
        {
          "name": "users-a",
          "host": "127.0.0.1",
          "port": 28101,
          "weight": 3,
          "failure_threshold": 2,
          "cooldown_ms": 300
        }
      ]
    }
  ],
  "routes": [
    {
      "prefix": "/api/users",
      "service": "users",
      "timeout_ms": 300,
      "max_retries": 1
    }
  ]
}
```

请求文件：

```json
{
  "requests": [
    {"id": "req-0001", "method": "GET", "path": "/api/users/1"}
  ]
}
```

强校验规则：服务名唯一；实例地址必须严格为 `127.0.0.1`；端口 `1..65535` 且全局唯一；权重为非负整数且服务权重和大于零；路由前缀以 `/` 开头且唯一；超时为 `1..600000ms` 整数；重试预算为 `0..10` 整数；目标服务必须存在；重复 JSON 键、布尔冒充整数、损坏 JSON 均整体拒绝。错误返回 `E_RULE` 与 `$.services[0].instances[1].port` 形式字段路径。

## 命令清单

| 命令 | 用途 |
|---|---|
| `python -m meshgov run ...` | 执行一个治理轮次并发布快照 |
| `python -m meshgov verify --snapshot PATH` | 校验快照轨迹，失败输出 `E_VERIFY` 与首个偏差 |
| `python -m meshgov view --snapshot PATH --format text` | 无头渲染稳定视图 |
| `python -m meshgov gui --snapshot PATH` | 启动 Tkinter 工作台 |
| `python -m meshgov diff --old A --new B` | 对比规则版本与变更路由 |
| `python -m meshgov recover --db PATH` | 强杀后恢复项/未知项清单 |
| `python -m meshgov doctor` | 环境与写权限检查 |
| `python -m meshgov acceptance` | 完整自动验收并生成报告 |
| `python -m meshgov upstream --config PATH` | 独立运行模拟上游 |
| `python scripts/check_ports.py` | 上游端口占用预检 |
| `python -m unittest discover -v` | 单元测试 |

三种故障注入：

```powershell
python -m meshgov run --inject rule_corruption ...       # E_RULE, exit 2
python -m meshgov run --inject upstream_unreachable ...   # E_UPSTREAM, exit 3
python -m meshgov run --inject snapshot_write_failure ... # E_SNAPSHOT, exit 4
```

也可使用业务码本身作为参数：`--inject E_RULE`、`--inject E_UPSTREAM`、`--inject E_SNAPSHOT`。

## 核心语义

- **确定性选择**：服务实例先按端口排序；使用显式 Smooth Weighted Round-Robin。轮次开始前按“请求输入顺序 + 尝试层级”规划所有可能尝试目标，因此并发线程完成顺序不会改变目标序列；零权重实例永久排除。
- **预算**：每次真实上游尝试原子获取全局和服务许可，两者不足即等待并在 `queue_ms` 记录。熔断快速失败不占预算。每次重试重新获取许可，单次尝试始终使用路由超时。
- **熔断**：连续失败达到实例阈值后打开；冷却期内目标请求直接写入 `E_CIRCUIT`、`selected=0`；冷却按单调时钟到期后仅允许一个半开探针；探针失败重开、成功关闭。迟到的在飞成功不能绕过冷却。
- **验证**：从已记录许可区间扫描全局/服务并发；核对终次尝试和终态码；核对冷却区间内没有真实 dispatch；核对尝试序号/重试预算和权重目标；按请求索引与检查项稳定排序首个偏差。
- **快照**：每个轮次绑定单一规则版本；先写不可变文件，再原子整体替换 `latest.json`，不合并旧字段；旧文件只读保留。轮次中规则文件改动会记录 `rules_changed_during_round=true`、轮次版本和观测到的新版本；下一次轮次使用新版本发布完整替换快照。
- **恢复**：请求、尝试、预算占用、熔断状态由 WAL/SQLite 事务持久化。强杀后只承认完整记录；正在运行但未完成的请求标为 `unknown_requires_rerun`，恢复报告固定声明 `exit_report_supported=false`。

## 实际验收结果

完整验收报告见 `artifacts/reports/acceptance.json` 与 `artifacts/reports/acceptance.md`，自动检查 **13/13 PASS**。最近一次验证包括：

- 21 个 `unittest` 全部通过。
- 样本 46 请求：全局实际峰值 `5/6`（上限内），`users` 峰值 `3/3`，`search` 峰值 `2/2`，`legacy` 峰值 `1/1`；验证器 PASS。
- 两次相同规则/请求运行的尝试目标序列逐项相同（59 条实际尝试）。
- v1 快照被 v2 完整替换；旧 v1 文件 SHA-256 不变；差异识别 `/api/users/admin` 新增及 `/api/users`、`/api/search` 变更。
- 轮次中替换规则文件后检测到版本变化，快照仍绑定该轮旧版本，同时记录新观测版本。
- 注入业务码实测：`E_RULE=2`、`E_UPSTREAM=3`、`E_SNAPSHOT=4`，均已写入 JSONL。
- 强制终止退出码 `137`；恢复命令列出 6 个未知重跑请求、1 条未完成尝试及持久化配额，拒绝生成退出报告。
- Tk 视图模型无头验证实例/路由稳定排序与单一规则版本绑定；本容器缺少 Tk 显示组件，具体视觉渲染需在 Windows 上启动 `gui`。

重新生成报告：

```powershell
python -m meshgov acceptance --workspace . --report artifacts/reports/acceptance.json
```

## 产物目录

- `artifacts/mesh.db`：默认 SQLite 账本（WAL 模式）。
- `artifacts/logs/events.jsonl`：结构化事件日志。
- `artifacts/snapshots/latest.json`：当前完整快照；同目录保留旧版本不可变文件。
- `artifacts/reports/`：运行、校验、差异、恢复与验收报告。
- `artifacts/acceptance/<run-id>/`：每次自动验收的完整隔离证据。

更详细的状态机、账本事务与验证算法见 `docs/TECHNICAL.md`。
