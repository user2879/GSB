# 云原生分布式追踪样本裁决组件

这是一个面向本地 Windows 环境、不使用 Docker 的确定性分布式追踪裁决样本。系统把跨节点、不同传播规则口径下的 NDJSON 跨度流裁决为唯一根跨度与采样节点集合，并写入可版本化、可逐条校验的 SQLite 账本。

权威真值来源是“汇聚服务返回的裁决摘要”，不是 Tkinter 快速视图使用的 SQLite 缓存。快速视图只读当前账本快照；权威复核必须重新启动本地汇聚服务并逐字节比较裁决摘要。

## 环境准备

要求 Windows 上可用 `python` 命令调用 Python 3.10 及以上版本。

```powershell
python --version
python -m pip install -r requirements-dev.txt
python -c "import tkinter; print(tkinter.TkVersion)"
python -c "import numpy; print(numpy.__version__)"
```

Tkinter 随 Windows 标准 Python 发行版提供。NumPy 按任务环境准备，但采样分桶、整数统计和摘要比较均使用确定性标准库整数运算；NumPy 不在账本关键路径上。

工作路径不要包含空格或中文。示例端口为 `39101`、`39102`、`39103`；运行前应确认这些高位回环端口未被占用。

## 快速验证

```powershell
python -m pytest -q
python scripts\run_demo.py --workdir demo_output
```

PowerShell 也可运行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_demo.ps1 -Workdir demo_output
```

演示会依次执行规则集 1 裁决、权威复核、快照重放、缓存视图、规则集 2 整体替换、再次复核和重放，并生成 `validation_report.json`。

## 手工命令

首轮裁决：

```powershell
python -m trace_arbiter adjudicate `
  --nodes sample_data\nodes_v1.json `
  --spans sample_data\spans.ndjson `
  --db demo_output\ledger.sqlite3 `
  --log demo_output\events.log.jsonl `
  --global-concurrency 3 `
  --per-node-concurrency 1 `
  --batch-size 2
```

权威复核、重放与快速视图：

```powershell
python -m trace_arbiter review --db demo_output\ledger.sqlite3 --log demo_output\events.log.jsonl
python -m trace_arbiter replay --db demo_output\ledger.sqlite3 --log demo_output\events.log.jsonl
python -m trace_arbiter view --db demo_output\ledger.sqlite3 --gui
python -m trace_arbiter view --db demo_output\ledger.sqlite3 --json
python -m trace_arbiter verify-ledger --db demo_output\ledger.sqlite3
```

规则演进并整体替换当前视图：

```powershell
python -m trace_arbiter adjudicate `
  --nodes sample_data\nodes_v2.json `
  --spans sample_data\spans.ndjson `
  --db demo_output\ledger.sqlite3 `
  --log demo_output\events.log.jsonl `
  --global-concurrency 3 `
  --per-node-concurrency 1 `
  --batch-size 2

python -m trace_arbiter view --db demo_output\ledger.sqlite3 --json
```

异常注入：

```powershell
python -m trace_arbiter adjudicate --nodes sample_data\nodes_v1.json --spans sample_data\spans.ndjson --db demo_output\bad1.sqlite3 --inject corrupt-message
python -m trace_arbiter adjudicate --nodes sample_data\nodes_v1.json --spans sample_data\spans.ndjson --db demo_output\bad2.sqlite3 --inject parent-missing
python -m trace_arbiter replay --db demo_output\ledger.sqlite3 --inject snapshot-mismatch
```

分别返回 `E_SPEC`、`E_CHAIN`、`E_SNAPSHOT`，并把结构化错误写入 JSONL 日志。

强制终止后恢复：

```powershell
python -m trace_arbiter recover --db demo_output\ledger.sqlite3 --log demo_output\recovery.log.jsonl
```

恢复会先找出最近未完成运行，把 `pending/running/failed` 批次标为 `unknown` 并保留审计标志，只复用状态为 `committed` 的批次，然后重跑未知批次、完成账本提交并执行快照重放。系统不生成“退出报告”；强制终止时只能依据已提交记录恢复，并由恢复后的重放结果证明一致性。

## 输入格式

节点声明 JSON：

```json
{
  "format_version": 1,
  "ruleset_version": 1,
  "nodes": [
    {
      "node_id": "api",
      "host": "127.0.0.1",
      "port": 39101,
      "propagation_rule_version": 1,
      "sample_ratio_bps": 10000,
      "random_seed": 101
    }
  ]
}
```

NDJSON 每行一个跨度：

```json
{"trace_id":"trace-cross","span_id":"tc1","parent_span_id":null,"node_id":"api","timestamp_ms":1710000001000,"duration_us":500,"sampled":true}
```

强校验规则：

- 节点 ID 唯一，主机必须是字面量 `127.0.0.1`，端口为 `1024..65535` 且全局唯一。
- 节点按声明顺序的传播规则版本单调不减；同一传播规则版本必须使用相同采样比例和随机种子。
- 采样比例是 `0..10000` 的整数；随机种子为整数；时间戳、耗时和版本均为非负/正整数。
- 每行字段类型严格，`sampled` 必须是 JSON 布尔值。
- 同一追踪内跨度 ID 唯一、父引用存在、无环、恰有一个根跨度，父跨度时间戳不得晚于子跨度。
- 判定不依赖 NDJSON 行顺序；加载后会按 `(trace_id, timestamp_ms, span_id)` 规范化。
- 节点规则集合、比例或种子变化时，`ruleset_version` 必须递增；某节点比例或种子变化时，该节点 `propagation_rule_version` 也必须递增。
- 首个失败项返回结构化 `path`，跨度流错误同时返回原始 `line`。

## 裁决语义

稳定分桶键为：

```text
SHA-256(seed | rule_version | trace_id | span_id | node_id)
```

取摘要前 8 字节转无符号整数后对 `10000` 取模。`bucket < sample_ratio_bps` 时为采样。

每个追踪：

- 只输出一个根跨度。
- 逐跨度记录所属节点、规则版本、分桶值、采样结论、输入采样标记是否与本地裁决一致。
- 未采样跨度的 `downstream_visible=false`，但父链和根归属不变。
- 对追踪参与节点的每个不同规则版本重新评估全部跨度。若同一跨度在不同版本下采样结论不同，则标记 `rule_difference.detected=true`，并输出涉及的节点、版本、采样跨度集合与受影响节点。
- 规则差异不是报文非法；只有字段、图结构、父引用、环、时间戳或端口/节点声明问题才进入 `E_SPEC`/`E_CHAIN`。

同批次请求重复发送时，返回的 `decision_summary` 字符串逐字节一致；`processing_us` 是独立整数测量值，不参与摘要。服务端不跨批次保存裁决状态。

## 组件结构

- `trace_arbiter/spec.py`：节点与 NDJSON 强校验、规范化与内容摘要。
- `trace_arbiter/adjudicate.py`：纯函数采样、跨版本差异、根归属与 canonical 摘要。
- `trace_arbiter/service.py`：绑定 `127.0.0.1` 的线程化 TCP 汇聚服务，JSON 行协议、握手、父进程看门狗。
- `trace_arbiter/pool.py`：Windows 子进程池、真实 socket 超时与同字节重试、全局/每节点并发调度。
- `trace_arbiter/engine.py`：批次编排、权威复核、快照重放、恢复和延迟采样。
- `trace_arbiter/ledger.py`：SQLite 事务、不可变账本条目、快照、翻转和陈旧标记。
- `trace_arbiter/ui.py`：Tkinter 快速视图，只读缓存。
- `trace_arbiter/cli.py`：命令行入口和结构化错误日志。

所有摘要使用同一 canonical JSON：UTF-8、键排序、无多余空格、禁止 NaN。摘要为 SHA-256 十六进制字符串。

## SQLite 账本

主要表：

- `snapshots`：规则集版本、节点/报文规范内容、内容摘要、当前/陈旧标记。
- `ledger_entries`：不可变逐追踪账本条目；含格式版本、根跨度、参与节点、采样节点/跨度、规则差异、内嵌裁决摘要与内容摘要。
- `snapshot_traces`：快照到账本条目的映射，并记录该追踪所用规则版本集合。
- `flips`：`changed/added/removed` 变更记录，含翻转前后规则版本、差异字段、前后条目摘要和陈旧标记。
- `runs` / `run_batches`：运行清单、每批请求摘要、响应摘要、归属上下文、尝试次数、队列等待、进度游标和未知标志。
- `latency_samples` / `latency_stats`：界面、复核、重放等操作的定点延迟样本与聚合结果。

账本条目不包含规则集修订元数据；规则版本通过快照映射记录。因此同一追踪的裁决内容在规则演进后完全相同时，可直接复用原不可变条目，其 JSON 字节和条目摘要保持不变。快照、旧翻转、旧延迟样本和旧统计通过 `is_stale=1` 排除在当前视图之外。

每完成一个服务批次，在同一个 `BEGIN IMMEDIATE` 事务中写入：

1. 批次裁决摘要和完整响应；
2. 归属上下文；
3. 重试次数、处理耗时和排队单调时钟区间；
4. 连续提交前缀进度游标。

## 并发与超时

- 每个节点各有独立线程池，上限为每节点并发配额。
- 所有请求再经过全局信号量，上限为全局并发配额。
- 调度器记录每个批次的排队开始/结束单调纳秒、等待微秒与毫秒字段，并统计实际最大全局/节点运行数。
- 每个 TCP 请求设置真实 socket 超时；超时后使用完全相同的 canonical 请求字节重试。服务无状态，因此重复处理返回相同摘要。
- 观察到任何上限被突破时以 `E_PROTOCOL` 终止。

## 快速视图与权威复核

`view --gui`/`view --json` 只读取当前 SQLite 快照，显示根跨度、采样节点集合、规则版本、条目摘要有效性和翻转记录。它每次整体替换列表，不做增量合并。

`review` 不读取界面缓存作为真值。它按快照节点规则和报文序列重新启动汇聚服务，重新计算全部批次和最终归属，逐追踪：

- 校验不可变条目摘要；
- 校验条目顶层字段与其内嵌裁决摘要一致；
- 逐字节比较服务返回的裁决摘要；
- 比较快照映射的规则版本集合。

失败返回 `E_DIVERGE`，带首个差异 `trace_id` 与 `field`。输出明确声明 `truth_source=service_decision_summary`。

## 定点延迟统计

所有耗时以 `time.perf_counter_ns()` 单调时钟差值计算，向下截断为整数微秒。统计值仍是整数：

- `scale_factor=1000`：1 ms 等于 1000 个定点单位，即 1 单位 = 1 us。
- `rounding=floor_toward_negative_infinity`。
- 纳秒到微秒截断的量化误差上界为 `999 ns`，保守整数上界为 `1 us`。
- P95 使用最近秩：排序后索引为 `ceil(0.95*n)-1`。
- 均值使用整数向下截断。

统计按操作、节点、`ALL`（节点请求整体）和 `END_TO_END`（界面/复核命令端到端）输出，均包含 `sample_count/mean_us/p95_us/max_us`。规则集整体替换后，旧快照下的统计为陈旧且不参与当前展示。

## 错误码

| 错误码 | 退出码 | 含义 |
|---|---:|---|
| `E_SPEC` | 2 | 节点声明或 NDJSON 字段/规则非法 |
| `E_PORT` | 3 | 回环端口绑定失败或占用 |
| `E_SPAWN` | 4 | 子进程无法启动或在握手前退出 |
| `E_HANDSHAKE` | 5 | 握手超时或协议不符 |
| `E_DIVERGE` | 6 | 权威服务摘要与账本不一致 |
| `E_REPLAY` | 7 | 恢复重放无法复现账本 |
| `E_SNAPSHOT` | 8 | 快照内容摘要不符 |
| `E_CHAIN` | 9 | 注入或服务侧父链错误 |
| `E_TIMEOUT` | 10 | socket 重试耗尽 |
| `E_PROTOCOL` | 11 | 协议、并发上限或摘要结构错误 |
| `E_RECOVERY` | 12 | 存在未完成运行或缺少可恢复快照 |
| `E_IO` | 13 | 本地文件读写失败 |
| `E_UI` | 14 | Tkinter 不可用 |

所有 CLI 失败都会写入 `<db>.log.jsonl`（或显式 `--log`）中的 JSONL 记录。

## 已验证结果

当前自动化验收：

```text
python -m pytest -q
18 passed
```

覆盖内容包括：字段/图/时间戳拒绝；文档化三类注入；输入逆序摘要一致；跨版本规则差异；重复批次摘要逐字节一致；`E_PORT`/`E_SPAWN`/`E_HANDSHAKE`；真实 socket 超时同字节重试；全局与节点并发上限；规则演进翻转；未变化条目字节稳定；`E_DIVERGE`；`E_REPLAY`；`E_SNAPSHOT`；强制终止后的 `unknown` 标记、恢复与重放；定点均值/P95/最大值；CLI 结构化日志。

本地样本规则集 1 得到 5 条追踪、7 个批次；规则集 2 产生 4 条 `sampled_node_ids` 翻转，`trace-static` 账本条目在两次规则集之间逐字节不变，旧视图与旧统计被整体排除。演示脚本会重新生成本地 `validation_report.json`。

本次 WSL2/Linux 验证环境未提供 Tkinter，因此 GUI 窗口未实际打开，数据面使用 `view --json` 完整验证；在 Windows 标准 Python 上可用 `view --gui` 打开快速视图。未使用 Docker、外部网络、账号或线上服务。
