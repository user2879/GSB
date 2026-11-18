# 推理网关灰度快照重建与交互诊断工作台

这是一个仅使用本地回环网络的确定性推理网关示例。它从 JSON 声明加载模型后端、路由规则和灰度阶段，通过 JSON 行协议接收请求，以 NumPy 模拟前向推理，并把命中账本、快照重建轨迹、交互延迟和一致性结论持久化到 SQLite。项目同时提供 Tkinter 诊断界面、pytest 测试和三段可重复演示。

核心约束是：**重建快照时只读取启动瞬间的单一账本版本，新请求继续准入和分流；展示统计必须与按账本逐请求权威重放的结果逐字节一致。**

## 快速开始

### Windows 本地环境

1. 安装 Python 3.11 或更高版本，并保留官方安装包中的 `tcl/tk and IDLE` 组件。
2. 在项目根目录执行：

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pytest
python scripts\demo.py
python -m graygate.ui --spec samples\gateway_spec.json --db runtime\workbench.db
# 或：python -m graygate --spec samples\gateway_spec.json --db runtime\workbench.db
```

### Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest
python scripts/demo.py
python -m graygate.ui --spec samples/gateway_spec.json --db runtime/workbench.db
# 或：python -m graygate --spec samples/gateway_spec.json --db runtime/workbench.db
```

Linux 需要安装对应 Python 的 Tk 组件。当前系统的 `/usr/bin/python3.11` 未编译 Tk，因此核心与演示使用该解释器验证；另用自带 Tk 9.0 的 CPython 3.12.16 完成了真实窗口创建、8 行统计渲染、快照/重建列表渲染和自动销毁检查。Windows 官方 Python 不需要额外安装 Tk。

## 项目结构

```text
graygate/
  spec.py           声明加载、强校验、E_SPEC 字段路径
  routing.py        稳定哈希与阶段、去向决策
  backend.py        NumPy 确定性前向、输出摘要、整数耗时
  network.py        127.0.0.1 JSONL 后端、批调度、后端集群
  gateway.py        网关 JSONL 准入与记账
  ledger.py         SQLite WAL 账本、原子提交、恢复与延迟记录
  metrics.py        全量/流式统计、逐字节比较、定点延迟摘要
  snapshot.py       快照、权威重放、E_SNAPSHOT、E_REPLAY
  service.py        重建线程、快照替换、恢复、诊断服务
  ui.py             Tkinter 工作台
  demo.py           三场景演示
  demo_worker.py    强制终止演示子进程
tests/              22 项 pytest（Tk 用例无显示时自动跳过）
samples/gateway_spec.json
scripts/demo.py
runtime/            默认 SQLite 目录
demo-output/        演示 SQLite 与 JSON 结论
```

## 声明格式

样本见 `samples/gateway_spec.json`。支持两种阶段表写法：

- 每个路由内提供 `stages`，每项含 `stage_id`、`start_request`、`end_request`、`gray_weight_bp`。
- 顶层提供全局 `stages`，路由使用其 `gray_weight_bp` 作为该路由默认权重；也可在阶段上提供 `weights: {"rule-id": 2500}` 覆盖指定路由。样本采用第一种写法。

路由规则要求：

- `rule_id`、`match_prefix` 唯一；请求的 `feature` 以匹配前缀区分路由。
- `stable_version`、`canary_version` 必须引用已声明且不同的模型版本。
- `gray_weight_bp` 是 `0..10000` 的整数；阶段权重必须非降。
- `batch_size` 是正整数。真实网关按 `(规则, 目标后端, 批大小)` 聚合请求，最多取该批大小。

模型与端口要求：

- 模型版本标识唯一。
- `model_dim`、`hidden_dim` 为正整数，`seed` 为整数。
- `host` 只能是字面量 `127.0.0.1`。
- 网关端口和所有模型端口均在 `1..65535` 内且全局不重复。

阶段请求序号必须首尾相接，例如 `1..120`、`121..240`。任一校验失败都会整体拒绝加载，返回 `E_SPEC` 和首个错误字段路径，例如：

```text
E_SPEC $.routes[0].stages[1].start_request: must equal previous end_request + 1 (121)
```

## 网关协议

网关监听声明中的 `gateway.port`。每个 TCP 连接发送一行 JSON 请求，并读取一行 JSON 响应。所有监听地址强制为 `127.0.0.1`。

请求：

```json
{"client_key":"client-17","feature":"alpha:checkout","sample":[0.1,0.2,0.3,0.4]}
```

成功响应：

```json
{
  "ok": true,
  "ledger_version": 42,
  "request_seq": 42,
  "rule_id": "alpha-route",
  "stage_id": "alpha-0",
  "stage_weight_bp": 0,
  "hash_bucket": 7312,
  "side": "stable",
  "backend_version": "alpha-stable-v1",
  "batch_seq": 10,
  "output_digest": "sha256:...",
  "elapsed_us": 37
}
```

分流实现为 `SHA-256(rule_id + 0x1F + client_key) % 10000`，桶值小于阶段权重则走灰度，否则走稳定版。因此在同一规则、同一阶段下，同一客户端键的结论只由稳定哈希决定，不依赖前序请求分布或批调度顺序。请求序号只负责选择放量阶段。

模型后端也是独立回环 JSONL 服务。网关把批次发送给目标模型端口；后端用模型种子构造固定权重，执行 `tanh(x @ W1 + b) @ w2`，把输出 `float64` 字节的 SHA-256 作为摘要，并返回整数微秒耗时。

## 快照与一致性

快照按 `规则 × 阶段` 聚合：

- 稳定/灰度命中数、实际灰度占比（万分比，量化误差上界 1 bp）。
- 稳定/灰度输出摘要集合（排序去重）。
- 稳定/灰度整数耗时总和与最大值。
- 整数格式版本、内容 SHA-256、源账本版本、观察上界、被排除命中数。

重建流程：

1. 在启动时捕获 `source_ledger_version`，写入 `RUNNING` 轨迹。
2. 独立线程重新读取 `ledger_version <= source_ledger_version` 的全部命中；不在旧快照上增量修改。
3. 重建期间网关继续使用同一 SQLite WAL 账本写入，账本版本继续增长。
4. 完成时读取 `observed_ledger_version` 与区间内排除命中数，原子插入新快照并把重建标为 `COMPLETE`，然后整体替换界面当前快照。
5. 界面列出相对上一快照发生变化的规则、阶段与字段。

权威校验会逐命中重算路由桶、阶段、去向、后端版本与 NumPy 输出摘要，再按同一聚合定义生成统计。统计以规范 JSON 比较；不一致时返回 `E_SNAPSHOT`，定位首个 `rule_id/stage_id/field`。

全量与流式聚合共用完全相同的整数统计定义。命中数或阶段数低于 `stream_threshold` 时走排序分组，达到阈值时走单遍累加；pytest 验证两条路径的规范化统计逐字节一致。

## 延迟诊断

- 时钟基准：`local_monotonic_ms`，即单调时钟整毫秒差。
- 真值来源：SQLite 账本中的逐次命中与观测记录。
- 每次重建、每次界面查询均写入 `latency_observations`。
- 均值使用 `scale=1000` 的定点整数；P95 使用最近秩，最大值为精确整数；均值量化误差上界为 1 个定点单位，即 0.001 ms。
- 界面查询超过 `query_latency_budget_ms` 时写入并返回 `E_LATENCY`，包含预算、实际值、超出量和影响范围；不影响网关准入与分流。

## 持久化、强制终止与恢复

SQLite 使用 WAL、`synchronous=FULL` 和短事务。命中行含整数格式版本及链式内容摘要；每次追加会同时更新账本链摘要。以下数据分别持久化：

- `hits`：逐请求命中账本。
- `rebuilds`：重建轨迹、源版本、状态、时限、排除数。
- `snapshots`：规范快照正文、内容摘要、声明参数、聚合路径。
- `latency_observations`：单调时钟起止、预算、超出量与影响范围。
- `consistency_checks`：权威校验和恢复重放结论。

若进程在重建中被强制终止，重启只会看到已完整提交的命中；遗留 `RUNNING` 重建被标为 `UNKNOWN`，`E_RECOVERY` 要求重新触发，系统不会生成或声称存在退出报告。恢复命令按快照保存的账本版本、阶段参数和声明摘要重新构建，先校验正文摘要，再逐项比较统计；差异返回 `E_REPLAY` 并定位首个规则、阶段与字段。

## 错误码

| 错误码 | 含义 |
|---|---|
| `E_SPEC` | 声明不合法，整体拒绝并给出字段路径 |
| `E_SNAPSHOT` | 快照统计与权威逐请求重放不一致 |
| `E_LATENCY` | 重建或界面查询超过显式预算 |
| `E_REPLAY` | 恢复后按保存参数重放快照不一致 |
| `E_RECOVERY` | 遗留 `RUNNING` 重建被恢复流程标为未知 |
| `E_STAGE` / `E_REQUEST` / `E_TRANSPORT` | 请求阶段、输入或回环传输错误 |

## 演示

`python scripts/demo.py` 顺序执行三个独立数据库场景，并写入 `demo-output/`：

1. **cold_start_ramp**：真实回环网关并发准入 480 个请求，覆盖四个阶段；重建快照并权威校验。
2. **rebuild_during_traffic**：在 120 条已提交命中上启动慢重建，同时准入 40 条新请求；确认新请求继续按阶段分流、快照只含 120 条、完成时排除 40 条，再校验一致性。
3. **forced_termination_recovery**：子进程先提交快照，再启动长重建；父进程在实际收到就绪标记后强制终止它，新进程恢复出 `UNKNOWN` 轨迹，重放旧快照，再重新触发并校验。

本次实际运行结果：

```text
SCENARIO cold_start_ramp ok=True
  snapshot source=480 digest=sha256:be336103fe25e3039711beafc4a12a686dbf3f8b806d4c72e375dce550e781b3 consistency=OK
SCENARIO rebuild_during_traffic ok=True
  snapshot source=120 observed=160 excluded=40 hits=120 routing_unchanged=True
  consistency_during=OK consistency_final=OK
SCENARIO forced_termination_recovery ok=True
  unknown_rebuilds=1 replay=OK retrigger=OK exit_report_claimed=False
```

测试结果：

```text
Python 3.11.0rc1 + NumPy 2.4.6 + pytest 9.2.0: 21 passed, 1 skipped (no Tk)
Python 3.12.16 + Tk 9.0.4 + NumPy 2.5.5 + pytest 9.2.0: 22 passed
Tk UI smoke: title="Gray Snapshot Rebuild Workbench", stats_rows=8, snapshot_rows=1, rebuild_rows=1
```

演示生成的 SQLite、JSON 和日志被 `.gitignore` 忽略，但保留在工作区中供复核。
