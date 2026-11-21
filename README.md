# 告警降噪与根因关联管线

这是一个面向 Windows 本地环境的确定性告警处理与结果验证工具。它把混存的
NDJSON 告警 v1/v2/v3 归一为统一表示，按“服务 + 指纹键 + 静默窗口”降噪，
沿服务依赖图推导根因候选，并把批次、聚合状态、关联账本和进度游标原子写入
本地 SQLite。工具支持并行调度、代际整体替换、强制终止恢复、重放校验和
Tkinter 诊断界面。

实现只使用 Python 标准库与 NumPy，不依赖 Docker、消息队列或远程服务。

## 快速开始

Windows PowerShell：

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m alert_pipeline diagnose

.venv\Scripts\python -m alert_pipeline generate --output samples
.venv\Scripts\python -m alert_pipeline run --manifest samples\manifest.json
.venv\Scripts\python -m alert_pipeline verify --manifest samples\manifest.json
```

也可以安装为命令：

```powershell
.venv\Scripts\python -m pip install -e .
alert-pipeline run --manifest samples\manifest.json
```

`diagnose` 会检查 Python、sqlite3、NumPy 与 tkinter。Python 官网安装包默认带
Tcl/Tk；如果 `tkinter_available=false`，请在 Windows 安装器中勾选
“tcl/tk and IDLE”，或重新运行 Python 安装器修复该组件。

## 核心命令

```text
generate   生成本地确定性样本、基础/演进依赖图与 manifest
run        加载、归一、降噪、关联、分批持久化并导出结论
verify     独立从原始告警流重算并与当前账本逐字段比对
recover    重放已完成批次，报告 unknown 批次与是否可生成退出报告
gui        打开 Tkinter 诊断界面
diagnose   输出本机环境能力
```

常用操作：

```powershell
# 串行模式；结论文件应与并行模式逐字节一致
python -m alert_pipeline run --manifest samples\manifest.json --mode serial `
  --global-concurrency 1 --per-service-concurrency 1

# 声明版本 2：依赖图、规则、静默窗口整体演进
python -m alert_pipeline run --manifest samples\manifest_v2.json
python -m alert_pipeline verify --manifest samples\manifest_v2.json

# 模拟第 3 批提交前被强制终止；进程退出码为 91
python -m alert_pipeline run --manifest samples\manifest.json `
  --crash-before-commit-batch 2

# 恢复检查：unknown 批次会出现，exit_report_available=false
python -m alert_pipeline recover --manifest samples\manifest.json

# 使用相同声明版本和种子重跑未完成批次
python -m alert_pipeline run --manifest samples\manifest.json

# 三类受控异常注入
python -m alert_pipeline run --manifest samples\manifest.json --inject corrupt
python -m alert_pipeline run --manifest samples\manifest.json --inject unsupported-version
python -m alert_pipeline run --manifest samples\manifest.json --inject window-conflict
```

使用独立数据库做实验时加 `--database data\experiment.db`、`--log` 与
`--output-dir`。显式命令行相对路径按当前目录解析；manifest 内相对路径按
manifest 所在目录解析。

## 输入格式

### 服务依赖图

```json
{
  "version": 3,
  "services": [{"service_id": "gateway"}, {"service_id": "payment"}],
  "dependencies": [{"from": "gateway", "to": "payment"}]
}
```

边 `from -> to` 表示 `from` 依赖 `to`。因此 `upstream` 沿依赖方向到达被依赖
服务，`downstream` 沿反向到达调用方。服务标识必须唯一，边引用必须存在，图
必须无环。拓扑排序使用稳定最小堆，不依赖输入或哈希遍历顺序。

### 根因规则

```json
{
  "version": 2,
  "defaults": {
    "directions": ["upstream", "downstream"],
    "max_depth": 2,
    "include_self": true
  },
  "rules": [{
    "rule_id": "checkout-both",
    "services": ["checkout"],
    "directions": ["upstream", "downstream"],
    "max_depth": 3,
    "include_self": true
  }]
}
```

`rules` 中的服务必须出现在依赖图中，一个服务最多被一条显式规则覆盖。没有显式
规则的服务使用 `defaults`。账本保存 `rule_id`、规则版本与依赖图版本。

### 告警 v1/v2/v3

规范字段名为：

| 版本 | 必需字段 | 版本新增/补齐 |
|---|---|---|
| v1 | `version,timestamp_ms,service_id,level` | 指纹按 `derived:<service_id>:<level>` 补齐；静默窗口用 manifest 默认值；根因引用为空；策略为 `legacy-default` |
| v2 | v1 字段 + `fingerprint,silence_window_ms` | 指纹和窗口来自记录 |
| v3 | v2 字段 + `root_candidates,strategy_id` | 根因引用必须是依赖图中的服务 |

所有版本均可带 `tags`，规整为按键排序的字符串标签对象。为便于互操作，加载器
接受有限别名，例如 `timestamp`、`service`、`alert_level`、
`fingerprint_key`、`silence_ms`、`root_candidate_refs` 和
`denoise_strategy`；同一语义字段的别名若值冲突则报 `E_FIELD`。

统一内部表示包含来源格式版本、声明版本、内容摘要，以及每字段 provenance。
v1 补齐字段会记录 `source=default` 与具体默认规则，例如
`v1.fingerprint=derived:<service_id>:<level>`。因此 v1 与使用相同派生指纹的
v3 记录在时间戳、服务、级别、指纹和标签上可逐字段一致。

### manifest

```json
{
  "declaration_version": 1,
  "format_version": 3,
  "default_silence_window_ms": 30000,
  "seed": 20261122,
  "batch_size": 8,
  "global_concurrency": 4,
  "per_service_concurrency": 1,
  "services_path": "services.json",
  "rules_path": "rules.json",
  "alerts_path": "alerts.ndjson",
  "database_path": "../data/sample.db",
  "log_path": "../logs/pipeline.jsonl",
  "output_dir": "../artifacts/sample-run"
}
```

## 处理语义

### 校验与业务码

加载在进入处理前整体完成；任一错误都会拒绝整份输入并报告首个违规行/字段。

| 业务码 | 场景 | CLI 退出码 |
|---|---|---:|
| `E_SPEC` | 重复服务、悬空依赖、图有环、时间戳回退、非法级别/指纹、根因规则引用不存在、损坏 NDJSON | 2 |
| `E_FIELD` | 未知格式版本、必需字段缺失、窗口为负、v3 根因引用悬空、字段类型错误 | 3 |
| `E_WINDOW` | 时间戳与静默窗口超出安全毫秒整数范围，无法形成合法窗口 | 4 |
| `E_VERSION` | 声明版本回退，或语义配置变化但未递增声明版本 | 5 |
| `E_REPLAY` | 已完成批次重放后与账本/摘要不一致 | 6 |
| `E_VERIFY` | 独立验证与当前结论不一致 | 7 |
| `E_STATE` | 本地状态或调用条件不满足 | 8 |
| `E_IO` | 文件不存在或不可读 | 9 |

所有错误同时写入结构化 JSONL 日志，包含业务码、行号、字段和证据。

### 降噪

分组键为 `(service_id, fingerprint)`，组内按 `(timestamp_ms, line_number)`
排序。设当前聚合首条时刻为 `first`、首条所用窗口为 `window`：

```text
timestamp_ms - first < window   => 抑制，计入 suppressed_count
timestamp_ms - first >= window  => 放行，开启新聚合
```

每条聚合记录首末时刻、被抑制数、总条数、参与行集合、来源格式版本、所用窗口
版本与完整来源 provenance。跨批次状态保存在聚合快照中，后续批次继续沿用窗口
锚点。并发完成顺序不会改变排序后的业务结果。

### 根因关联

对每条聚合告警使用服务规则沿上下游 BFS，最多到 `max_depth`。候选按以下稳定
键排序：

```text
(到告警服务的距离, 依赖图层级, 稳定拓扑秩, 服务标识)
```

每条候选保存方向、路径和规则证据。v3 中声明的根因引用保存在账本中作为输入
证据，但最终候选仍由依赖图和规则推导，避免输入引用绕过图约束。

### 并发调度

线程池同时受 `global_concurrency` 与 `per_service_concurrency` 两个条件约束。
任务只有在两个配额都可满足时才从队列进入运行；等待开始/结束、运行区间、峰值
全局占用和每服务峰值均使用单调时钟毫秒。批次时间线按任务 ordinal 稳定输出。

串行与并行的耗时时间线必然不同，但聚合、关联账本、批次摘要和
`conclusions.canonical.json` 逐字节一致。

### 版本演进与陈旧隔离

`declaration_version` 必须严格递增。下列任一变化都改变语义摘要，必须递增声明
版本：

- `format_version`
- 默认静默窗口
- 种子或批大小
- 服务依赖图或其版本
- 根因规则或其版本
- 原始输入文件内容

新代际创建时，旧代际、旧聚合、旧账本和旧批次统计全部标记 `is_stale=1`，且不再
出现在 `current_aggregate_view`、`current_ledger_view`、`current_batch_view`。
新结果写入独立代际，不做增量覆盖。完成后记录受影响告警、变化字段和替换前后
完整结论差异。

### 原子批次与恢复

每个批次开始前先提交一条 `unknown` 尝试记录。批次完成后，在同一个 SQLite
事务中写入：

1. 批次摘要与业务码；
2. 当前聚合中间状态快照；
3. 当前关联账本快照；
4. 进度游标与 committed 尝试状态。

进程被强制终止时，未提交事务回滚，进度仍停留在上一已提交批次；`unknown`
尝试保留并要求重跑。`recover` 只重放已完整提交的批次，按代际记录的声明版本、
种子、原始文件摘要重新计算并逐项比较。存在 unknown 时报告
`exit_report_available=false`，不生成或宣称退出报告。

### SQLite 结构

主要表与视图：

- `generations`：代际、声明版本、语义摘要、种子、状态、current/stale
- `batches`：批次摘要、峰值并发、时间线、结论摘要
- `batch_attempts`：`unknown/committed/superseded` 意图与提交状态
- `progress`：每代际进度游标
- `aggregates` / `batch_aggregates`：当前结论与每批中间快照
- `ledger` / `batch_ledger`：关联账本与每批关联快照
- `replacements` / `replacement_runs`：整体替换差异
- `current_aggregate_view` / `current_ledger_view` / `current_batch_view`：当前视图

## 验证与诊断界面

`verify` 不信任 SQLite 中的聚合结果。它从原始服务图、规则和 NDJSON 独立重算
归一、降噪与关联，再检查：

- 归一来源格式版本与字段 provenance；
- 静默窗口抑制数、参与行和窗口版本；
- 根因候选、推导路径、规则/图版本；
- SQLite 内容摘要；
- 当前代际一致性及陈旧隔离。

首个差异以 `E_VERIFY` 报告 `alert_id`、字段、期望值、实际值与证据。

GUI 启动：

```powershell
python -m alert_pipeline gui --manifest samples\manifest.json
```

界面包含“告警列表、关联账本、并发时间线、账本批次”四个稳定排序视图，并提供
“运行独立验证”按钮。

## 项目结构

```text
alert_pipeline/
  spec.py         JSON/NDJSON 强校验、v1/v2/v3 编解码与归一
  dedup.py        指纹/窗口聚合与跨批次状态
  correlate.py    稳定拓扑、上下游根因推导
  scheduler.py    双配额线程调度与单调时钟时间线
  engine.py       批次内归一→降噪→关联三阶段
  storage.py      SQLite 事务、代际替换、恢复、重放
  pipeline.py     运行编排与产物导出
  verifier.py     独立重算验证
  gui.py          Tkinter 四视图诊断界面
  samplegen.py    NumPy 确定性样本与演进样本
  cli.py          命令、结构化输出与异常注入
samples/          v1/v2/v3 混存样例与两代 manifest
tests/            32 项自动化测试
docs/             架构与实测验证记录
```

测试：

```powershell
python -m unittest discover -v
```

## 本次实测环境

- 执行环境：WSL2 Linux 容器（目标运行环境为 Windows 本地，代码未使用 Unix 专属接口）
- Python：3.11.0rc1，满足 `>=3.11`
- SQLite：3.37.2，`sqlite3` 可用
- NumPy：1.26.4，满足 `>=1.26`
- Tkinter：本容器未安装系统 `python3-tk`，且当前账户无免密系统安装权限，因此未能在本容器启动 GUI。`gui.py` 已完整实现并有稳定视图数据测试；Windows 官方 Python 补齐 Tcl/Tk 后可直接运行

正式样本与各项业务码、字节摘要、恢复/重放结果见
[`docs/VERIFICATION.md`](docs/VERIFICATION.md)。架构与确定性不变量见
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)。
