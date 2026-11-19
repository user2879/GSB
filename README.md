# 本地配置生效结果账本与观测合并诊断台

这是一个只在本机运行的 Python 3.11+ / Tkinter / SQLite 诊断台。它不连接真实集群，也不改写任何配置声明；它回答的问题是：

> 同一个配置键在多个组件节点上实际报告了什么值，这些观测如何按声明口径归并为唯一账本结论？

所有网络请求都限定为字面量 `127.0.0.1` 与高位端口，观测端由仓库内的确定性模拟器提供。运行时不需要第三方包，Tkinter 与 SQLite 来自 Windows 标准 Python 发行版。

## 快速开始

在仓库根目录使用 Windows `py` 启动器（或直接使用 `python`）：

```bat
py -3.11 -m config_ledger demo --runtime runtime\demo --all-anomalies
```

该命令会自动启动 3 个本地观测进程，执行基础样本和三组异常注入，输出 JSON 报告，并写入：

- `runtime\demo\*\ledger.sqlite`：SQLite 轮次账本。
- `runtime\demo\*\structured.jsonl`：逐条结构化日志。
- `runtime\demo\*\spec.json`：若样本端口被占用，自动改用本机其他高位端口后的实际声明。

自动改端口只写运行时副本，仓库内 `samples\` 下的原始声明不会被修改。

运行交互式诊断台：

```bat
py -3.11 tools\run_gui.py
```

`run_gui.py` 会自动启动分歧样本的本地模拟进程。也可以连接已经手工启动的模拟器：

```bat
py -3.11 -m config_ledger gui ^
  --spec samples\divergence\spec.json ^
  --ledger runtime\gui\ledger.sqlite ^
  --log runtime\gui\structured.jsonl
```

手工执行一轮 CLI 观测时，先为每个节点各开一个终端。例如：

```bat
py -3.11 -m config_ledger.simulator --scenario samples\divergence\simulator.json --node-id node-a --port 43871
py -3.11 -m config_ledger.simulator --scenario samples\divergence\simulator.json --node-id node-b --port 43872
py -3.11 -m config_ledger.simulator --scenario samples\divergence\simulator.json --node-id node-c --port 43873
```

然后执行：

```bat
py -3.11 -m config_ledger run ^
  --spec samples\divergence\spec.json ^
  --ledger runtime\manual\ledger.sqlite ^
  --log runtime\manual\structured.jsonl ^
  --replay-after
```

校验声明而不启动任何节点：

```bat
py -3.11 -m config_ledger validate --spec samples\divergence\spec.json --log runtime\validate.jsonl
```

运行测试：

```bat
py -3.11 -m unittest discover -v
```

## 声明格式

声明文件同时包含节点观测通道、逐键期望值和运行预算。以下为缩减示例；完整样本见 `samples/divergence/spec.json`。

```json
{
  "name": "divergence-and-arbitration",
  "parameter_version": 1,
  "global_concurrency": 2,
  "observation_timeout_ms": 500,
  "ui_latency_budget_ms": 50,
  "observation_window": "divergence-window-1",
  "arbitration_rule": "majority_then_latest",
  "keys": {
    "app.mode": {"expected": "stable", "quorum_bp": 6000},
    "feature.enabled": {"expected": 1, "quorum_bp": 6667}
  },
  "nodes": [
    {"id": "node-a", "host": "127.0.0.1", "port": 43871, "quota": 1},
    {"id": "node-b", "host": "127.0.0.1", "port": 43872, "quota": 1},
    {"id": "node-c", "host": "127.0.0.1", "port": 43873, "quota": 1}
  ]
}
```

强校验规则：

- 节点 ID 非空且全局唯一；节点集合非空。
- `host` 必须精确等于字面量 `127.0.0.1`，不解析别名、不做 DNS。
- 端口为 `1..65535` 的整数且全局唯一；`quota` 为正整数。
- 每个键的 `expected` 只能是整数（明确拒绝 JSON boolean）或字符串。
- `quorum_bp` 是 `0..10000` 的整数万分比。
- 未知字段、重复 JSON 对象键、非法 JSON 都会被拒绝。
- 所有问题一次收集，以 `E_SPEC` 整体拒绝；错误项包含节点标识和字段路径，例如 `keys["array.expected"].expected`。加载器不会返回半成品，重新加载失败也不会替换内存中的旧声明。

## 处理管线

1. `config_ledger/spec.py`：严格加载、校验、规范化声明，并生成参数指纹。
2. `config_ledger/observer.py`：在全局并发上限与逐节点配额同时生效时调度拉取；每个请求使用真实 `http.client.HTTPConnection` 套接字超时。
3. `config_ledger/merge.py`：按稳定规则从多节点、多份观测中选候选值、计算法定观测数、仲裁分歧。
4. `config_ledger/latency.py`：以整数微秒保存逐次耗时，计算均值、最近秩 P95 和最大值。
5. `config_ledger/ledger.py`：SQLite 事务账本、参数版本、陈旧隔离、内容摘要、强杀恢复和重放记录。
6. `config_ledger/service.py`：端到端编排、UI 查询预算和重放首差异定位。
7. `config_ledger/ui.py`：Tkinter 诊断界面；只在该命令执行时导入 Tkinter。
8. `config_ledger/simulator.py`：确定性回环观测端，支持延迟、重复观测和缺失键。

### 双重并发预算

`ObservationFetcher` 自己调度任务，而不是仅把任务扔给无界线程池：

- 任一时刻活动请求总数不超过 `global_concurrency`。
- 任一节点活动请求数不超过该节点 `quota`。
- 节点配额已满的任务留在待调度队列，记录 `queue_wait_start_ms`、`queue_wait_end_ms`、等待毫秒数和 `queued` 标志。
- 返回的 `budget_summary` 还会给出本轮 `peak_global` 与 `peak_by_node`，测试直接断言峰值未超限。

套接字超时不是人为 sleep：模拟器可以延迟响应，而客户端由 `HTTPConnection(..., timeout=...)` 产生真实的读超时。超时、连接拒绝、HTTP 404、非法 JSON、非法载荷分别记录状态；所有缺失都会在轮次中留下 `E_OBSERVE` 记录，连接拒绝还会额外记录 `E_PORT` 诊断码。

## 归并与仲裁口径

对每个配置键：

1. 每个节点先取自身最新观测；排序键为 `observation_seq DESC, timestamp_ms DESC, node_id ASC, canonical_value ASC`。
2. 有效节点数 = 至少有一份合法最新观测的节点数。重复观测不会虚增法定观测数。
3. 候选值 = 所有节点最新观测中按上述时序规则最靠前的一条。
4. 法定观测数判定使用纯整数比较：`valid_nodes * 10000 >= quorum_bp * total_nodes`。
5. 若存在多个取值：
   - 先按“取值计数多数”仲裁。
   - 计数并列时，按最高观测序号、最高时间戳、节点 ID 的稳定顺序采用时间最新值。
6. 所有节点对照行记录生效值、观测序号、时间戳、重复份数、是否被采纳和理由；候选值与仲裁胜出值可能不同，二者会同时写账。

状态与归因是分开的：

- 状态：`OK`、`EXPECTED_MISMATCH`、`DIVERGED`、`INSUFFICIENT_OBSERVATIONS`、`NO_VALUE`。
- 归因：`OBSERVATION_MISSING`、`OBSERVATION_TIMEOUT`、`VALUE_DIVERGENCE`，按固定顺序输出。
- 低于法定观测数时，`effective_value` 为空；仲裁细节仍保留为诊断信息，节点采纳理由带有 `TENTATIVE_QUORUM_NOT_MET`。

示例 `divergence` 中，`app.mode` 的最新候选是 `canary`，但 2/3 节点报告 `stable`，所以多数仲裁胜出为 `stable`；`tie.rule` 的 `alpha`/`beta` 为一比一并列，`beta` 因观测序号更高而胜出。同一输入重复拉取与归并的 `deterministic_bytes()` 完全一致。

## 延迟统计与预算

- 每个拉取请求和每次 `query_dashboard()` 都记录单调时钟起点、终点与整数微秒耗时。
- 界面查询的测量边界是“进入查询处理器”到“当前视图组装完成”；随后立即把该样本写入账本并重算展示聚合，避免测量自身写入造成自指循环。
- 单次记录写入 SQLite；拉取样本随轮次事务提交，界面查询样本写入 `ui_latency_samples`。
- 账本保存逐节点与整体拉取聚合；界面查询保存整体聚合。所有聚合都基于完整样本集，不剔除超预算样本。
- 均值使用整数四舍五入；P95 使用 `ceil(0.95*n)-1` 的最近秩方法；最大值为样本最大值。
- 统计同时输出整数微秒与整数 `fixed6_ms`（1 us 表示为 `1000`，即六位小数毫秒定点）。
- 从纳秒取整到微秒以及均值舍入的最坏量化误差上界为 `1.5 us = 0.0015 ms`，账本中记录为 `quantization_error_bound_us_halves=3` 和 `quantization_error_bound_fixed6_ms=1500`。
- 界面查询超过 `ui_latency_budget_ms` 时写入 `E_LATBUDGET`，并给出 `overrun_us`/`overrun_fixed6_ms`；该样本仍进入当前版本的均值、P95 与最大值。

## 账本、版本与恢复

SQLite 账本格式版本为整数 `1`，同时保存在 `ledger_info.format_version` 与 `PRAGMA user_version`；`meta` 中保存当前参数版本、当前轮次、内容摘要和当前参数指纹。每轮在同一个事务中写入：

- 每个节点/键的观测摘要（含失败状态与原始观测数组）。
- 逐键结论与仲裁对照。
- 逐请求延迟样本、节点/整体聚合。
- 轮次游标与 `COMMITTED` 尝试状态。
- 轮次内容摘要；摘要覆盖参数声明、观测、结论、对照、拉取延迟和进度游标。

参数版本规则：

- 参数指纹覆盖节点集合、逐键期望值、合并口径、观察窗口、超时、并发和界面预算。
- 指纹变化时，新版本为 `max(当前版本+1, 声明中 parameter_version)`，绝不回退。
- 旧轮次被标记 `is_stale=1`，旧界面延迟样本同样标记陈旧；当前视图只读取新版本/当前轮次，并整体替换为空，直到新一轮提交。
- 最后完整提交的轮次与版本单独持久化；即使参数切换后、新轮次提交前重启，下一次提交仍能生成旧/新整体替换对照。
- `replacement_events` 稳定列出参数差异、受影响键、旧结论、新结论以及判定翻转。
- 重新打开账本时重算当前轮内容摘要；若与轮次或 `meta` 摘要不符，写入 `E_SNAPSHOT`，陈旧化当前轮、清空当前指针，并把游标改为 `RERUN_REQUIRED`。

强制终止恢复：

- 开始拉取前先独立提交 `IN_PROGRESS` 尝试标记。
- 完整轮次数据与 `COMMITTED` 状态在同一个 SQLite 事务中提交。
- 重启时只能恢复已提交轮次；遗留 `IN_PROGRESS` 被改为 `UNKNOWN` 并要求重跑。
- 系统不伪造退出报告，也不把未提交观测当成已完成数据。`tests/test_ledger_service.py` 使用子进程强杀验证这一点。

## 重放校验

`service.replay()` 使用账本中保存的参数版本、声明和观察窗口重新拉取并归并，然后按固定顺序比较：

1. 逐键结论字段。
2. 逐键、逐节点对照字段。
3. 账本内逐次延迟样本重算结果与已提交聚合。

首个差异以 `E_REPLAY` 记录，包含 `key`、`node_id`、`field`、账本值和重放值。键级字段的 `node_id` 使用 `*`，延迟差异使用 `key="<latency>"`。

需要说明的是：新一次网络拉取的墙钟耗时本来就不可能与旧轮次逐微秒相同。因此重放对“延迟统计”的验证是检查账本逐次样本重算后是否仍等于已提交聚合，而不是要求两次真实墙钟完全相同；实时重放耗时另行展示。这样既能检测账本篡改，也不会用不可复现的时钟制造假失败。

## 异常注入与业务码

| 注入 | 触发方式 | 预期业务码 |
| --- | --- | --- |
| 非法 JSON | `demo --inject invalid_json`，或模拟器 `--invalid-json` | `E_OBSERVE` |
| 节点端口不可达 | `demo --inject port_unreachable`（不启动 node-b） | `E_PORT`，同时该缺失仍记为 `E_OBSERVE` |
| 账本内容摘要不符 | `demo --inject snapshot_mismatch`（提交后篡改 `meta` 摘要再重开） | `E_SNAPSHOT` |
| 界面延迟预算超限 | 将 `ui_latency_budget_ms` 设为 `0` 后查询；自动化测试已覆盖 | `E_LATBUDGET` |
| 重放差异 | 篡改已提交结论/延迟聚合后执行 replay | `E_REPLAY` |

结构化日志为 UTF-8 JSONL，每条至少包含 `ts_utc`、`level`、`code`、`message`，异常记录还带节点、键、状态、原因和量化后的耗时。

## 确定性样本

| 目录 | 覆盖内容 |
| --- | --- |
| `samples/divergence` | 3 节点、多数仲裁、最新值并列仲裁、重复观测、缺失键、非数字字符串期望值、期望不匹配 |
| `samples/missing` | 缺失键、真实套接字超时、达到/未达到法定观测数 |
| `samples/string_expected` | 合法的非数字字符串期望值 |
| `samples/invalid_expected` | boolean/array 期望值，必须整体拒绝并输出 `E_SPEC` |

模拟器场景中的 `observations` 固定写入 `observation_seq` 与 `timestamp_ms`，因此重复执行不依赖当前时间。模拟器还支持：

- 节点级 `default_response_delay_ms` 和逐键 `response_delays_ms`。
- 逐键 `duplicate_counts`。
- `missing_keys`。
- 命令行 `--delay-ms`、`--duplicate-count`、`--missing-key` 覆盖项。

## 界面

Tkinter 界面包含固定标签页：

- `逐键结论`：期望值、生效值、法定观测数、状态、归因。
- `节点对照`：每个键/节点的值、序号、时间戳、采纳状态与理由。
- `分歧仲裁`：最新候选、多数/最新规则胜出值、并列值、期望匹配和依据。
- `延迟统计`：拉取逐节点/整体和界面查询的计数、均值、P95、最大值、量化误差上界。
- `账本/版本`：格式版本、参数版本、轮次摘要、游标、版本历史、整体替换事件和 UNKNOWN 尝试。

所有列表按稳定键插入，不依赖字典遍历顺序。顶部按钮支持运行一轮、重放校验、重新加载声明和刷新查询。刷新查询本身会计入界面查询延迟预算。

当前自动化验证环境是 Linux 容器且未安装 Tkinter，所以界面只能做语法编译检查；目标 Windows 标准 Python 自带 Tkinter。CLI、SQLite、模拟器和全部业务测试不依赖图形环境。

## 目录

```text
config_ledger/       核心包、CLI、模拟器、Tkinter 界面
samples/             确定性声明与模拟场景
tests/               unittest 验收测试（无第三方依赖）
tools/               Windows 友好启动器与强杀恢复子进程
runtime/             运行时生成，已加入 .gitignore
VALIDATION.md        实际验证记录
```

## 范围与限制

- 不访问真实集群、公网或未授权目标；只允许精确回环字面量。
- 不修改配置声明，只读取声明并把观测结论落账。
- 期望值仅支持整数或字符串，与题目口径一致。
- 进程强杀时无法得知当时内存中的最后动作；系统只承认已提交轮次，并明确要求重跑未知轮次。
- 重放需要同一确定性观察窗口的模拟端仍然可用；账本本身不会虚构观测。
