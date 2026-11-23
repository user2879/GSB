# 配置中心热更新中间件诊断台

面向 Windows 本地环境的配置中心诊断工作台。项目不使用网络服务、外部账号或容器，所有样本、SQLite 账本、快照、只读备份和结构化日志都位于当前工作区。

核心能力：

- 本地 JSON 强校验：键名、整数取值、单调版本、TTL、派生引用、派生公式与环检测。
- SQLite 原子账本：键记录、内容摘要、快照、视图和变更游标在同一事务提交；同一操作重试幂等。
- 稳定拓扑失效传播：仅重算直接与间接受影响键，未受影响键逐字节不变。
- 单调时钟 TTL 回收：过期键立即退出派生态，下游同步失效。
- 订阅视图整体替换：集合版本或派生口径版本变化后，旧视图与旧失效轨迹统一标记陈旧。
- 增量路径与全量重建路径逐键比较，支持 `E_DIVERGE` 诊断。
- 强制终止恢复、变更重放、逐键字段和失效轨迹复核，支持 `E_REPLAY`。
- 三类命令注入异常：`E_SPEC`、`E_CYCLE`、`E_SNAPSHOT`，全部写入 JSONL 日志。
- Tkinter 诊断界面：键表、依赖图、传播链、陈旧标记、账本版本与快照列表。

## 环境准备（Windows）

要求 Python 3.11 或以上，官方安装包自带 Tkinter。

```bat
cd /d 工作区路径
py -3.11 -m pip install -r requirements.txt
py -3.11 -m config_diag validate
py -3.11 -m config_diag init --reset
```

也可以使用 `python`：

```bat
python -m pip install numpy
python -m config_diag validate
```

本项目的核心只使用 Python 标准库；`numpy` 用于带固定整数种子的本地点选择，采用 `numpy.random.RandomState(seed + round * 2654435761)`，相同样本与轮次可复现。

## 快速运行

```bat
REM 校验样本并展示稳定拓扑序
python -m config_diag validate

REM 初始装载
python -m config_diag init --reset

REM 应用单键变更
python -m config_diag apply --sequence 1

REM 应用键组变更
python -m config_diag apply --sequence 2

REM 编辑派生公式，派生口径版本递增
python -m config_diag apply --sequence 3

REM 缩短 TTL 后等待并真实计时回收
python -m config_diag apply --sequence 4
python -m config_diag expire --wait-ms 250

REM 增量/全量逐键比较
python -m config_diag compare

REM 恢复后逐键重放
python -m config_diag recover

REM Windows Tkinter 诊断台
python -m config_diag gui
```

所有 CLI 输出都是 UTF-8 JSON。默认文件：

| 类型 | 默认路径 |
|---|---|
| SQLite 账本 | `runtime/ledger.sqlite3` |
| SQLite WAL/SHM | `runtime/ledger.sqlite3-wal`、`-shm` |
| 只读备份 | `runtime/backups/` |
| 结构化日志 | `logs/diagnostics.jsonl` |
| 本地样本 | `samples/` |
| 验收结果 | `artifacts/` |

可通过全局参数覆盖：`--workspace`、`--samples`、`--db`、`--log`。

## 样本

- `samples/config_keys.json`：11 个键，包含两条独立分支和一条五层定价派生链。
- `samples/subscription_clients.json`：3 个固定整数种子的订阅客户端。
- `samples/change_sequence.json`：4 个确定性变更操作。
  - `op-001`：单键修改基础费用。
  - `op-002`：键组同时修改税率与批次大小。
  - `op-003`：修改 `reporting.rounded` 派生公式。
  - `op-004`：将 `cache.region_limit` 的 TTL 改为 180 ms，供真实计时回收演示。

初始派生结果：

| 键 | 取值 |
|---|---:|
| `pricing.net_fee` | 950 |
| `tax.amount` | 76 |
| `checkout.total` | 1026 |
| `reporting.rounded` | 102 |
| `ops.batch_value` | 75 |
| `cache.shard_span` | 100 |

`op-001` 后，受影响集合固定为：

```text
pricing.base_fee
pricing.net_fee
tax.amount
checkout.total
reporting.rounded
```

其余 6 个键的键记录字节、内容摘要和键级版本保持不变。

## 派生公式

所有取值和权重都是整数；定点缩放由 `divisor` 表达。除法统一向零截断，避免 Windows/Linux 或 Python 版本差异。

支持三种公式：

```json
{"op": "identity"}
```

```json
{
  "op": "linear",
  "terms": [
    {"ref": "pricing.base_fee", "weight": 1000},
    {"ref": "pricing.base_discount", "weight": -1000}
  ],
  "divisor": 1000,
  "bias": 0
}
```

```json
{
  "op": "product",
  "terms": [
    {"ref": "pricing.net_fee", "weight": 1},
    {"ref": "tax.rate_permille", "weight": 1}
  ],
  "divisor": 100,
  "bias": 0
}
```

计算口径：

```text
linear  = trunc_toward_zero(sum(value(ref) * weight) / divisor) + bias
product = trunc_toward_zero(prod(value(ref) * weight) / divisor) + bias
```

每个键的 `refs` 必须与公式 `terms` 的引用集合完全一致。根键只能使用 `identity` 且 `refs=[]`；派生键的取值只读，修改口径时同时提供 `refs` 与 `formula`。

## 账本与事务

`ledger_meta` 保存：

- `format_version`：当前为 `1`。
- `ledger_version`：每次已提交状态变更递增。
- `config_set_version`：每轮配置集合整体提交递增。
- `derivation_version`：派生口径公式变更时递增。
- `change_cursor`：最近完整提交的 journal id。
- `content_digest`：规范 JSON + SHA-256 账本摘要。

`ledger_keys` 按主键 `key` 只保留一条当前记录，包含原始值、生效值、键级版本、引用、公式、TTL、单调时钟过期时刻、状态、派生依据和逐键内容摘要。

写入流程：

1. 独立事务提交 `pending` 变更意图。
2. `BEGIN IMMEDIATE` 开启账本事务。
3. 仅 UPSERT 失效集合内键。
4. 同一事务更新 `ledger_meta`、视图、journal 状态和快照。
5. `COMMIT` 后所有读取者整体可见。

WAL 模式下，提交前其他连接只能看到旧键记录和 `pending` 意图，不能看到半成品摘要。

## 失效传播与视图

1. 将直接编辑键作为变化源。
2. 沿 `ref -> dependent` 边求全部下游。
3. 在受影响子图上使用最小堆 Kahn 拓扑排序，键名作为稳定平局规则。
4. 每个传播步骤输出 `basis=direct|dependency`、`via`、`edge`、`fields` 和动作。
5. 只对受影响集合重算值、派生依据、逐键摘要和键级版本。

每次提交从全部当前键记录重新构造视图，不逐键修补旧视图。旧视图状态改为 `stale`；旧 journal 轨迹记录 `stale_since_view`。订阅模拟只从 current view 读取值和计算统计。

## 过期与回收

过期时刻保存为 `time.monotonic_ns()` 基准的整数纳秒，判定不依赖墙上时钟。

```bat
python -m config_diag expire
python -m config_diag expire --wait-ms 250
```

回收时：

- 所有到期键按 key 稳定排序。
- 过期键状态设为 `expired`，不在 current view 的 active 集合中。
- 沿依赖图求下游，统一设为 `invalid` 并清空生效值。
- journal 记录回收单调时刻、回收键集合、下游集合和传播链。
- 同一输入状态重复扫描结果稳定；已回收键下一轮不会重复返回。

## 增量/全量一致性

```bat
python -m config_diag compare
python -m config_diag compare --inject-diverge
```

`compare` 在冻结的同一单调时刻创建两个全新账本：

- 增量路径：每步只重算受影响子集。
- 全量路径：每步执行全键拓扑重算，同时保留相同的键级版本语义。

比较字段：`value`、`content_digest`、`version`、`basis`、`status`、`refs`、`formula`、`expires_mono_ns`。此外用全量路径每步前后的键记录字节差异独立核对增量失效集合。首个差异以 `E_DIVERGE` 输出 key、field、两侧值和原因。

## 强制终止与恢复

变更意图先提交为 `pending`，账本主体后提交。进程在以下位置被强杀时：

- `--crash-point after-intent`：只有待处理意图，账本仍为上一完整版本。
- `--crash-point before-commit`：键写入与摘要尚未提交，SQLite 回滚，只保留意图。
- `--crash-point after-commit`：账本完整提交，但进程不会生成退出报告。

重启打开账本时，`pending` 被标记为 `unknown`：

```json
{
  "code": "E_RECOVERY",
  "requires_rerun": true,
  "exit_report_available": false
}
```

`recover` 只读取已完整提交的记录和游标，在临时账本中从初始声明重新装载并重放已提交变更。未知事务不参与恢复，必须以原 `operation_id` 重跑。`recover` 在仍有未知事务时返回退出码 `2`。

重放逐项比较：

- 每键取值
- 内容摘要
- 键级版本
- 派生依据
- 状态、公式、引用、过期时刻
- 每轮失效集合和传播轨迹
- current view 摘要与集合

差异以 `E_REPLAY` 定位首个键、字段、账本侧值和重放侧值。

## 账本自校验与只读备份

打开已有账本时检查：

1. SQLite `PRAGMA integrity_check`。
2. `format_version`。
3. 每个键的内容摘要。
4. 由全部规范键记录和元数据重算的账本摘要。

版本或摘要不一致时：

1. 先 checkpoint，再复制账本到 `runtime/backups/`。
2. 将备份设为只读。
3. 从最新“自身摘要可验证”的快照整体 `DELETE + INSERT` 替换，而不是逐键合并。
4. 写入 `E_SNAPSHOT` 结构化日志，输出备份路径、快照 id 和恢复模式。

## 订阅模拟

```bat
python -m config_diag poll --rounds 3
```

每个客户端使用固定 `seed`、`poll_interval_ms`、键列表和 `sample_size`。当缓存视图的集合版本、派生口径版本或视图摘要变化时：

- 将旧缓存整体标记陈旧并记录 `whole_view_replace`。
- 用完整 current view 替换，不逐键修改缓存。
- 只使用 current view 的 active 条目生成观测和统计。
- 过期/失效条目列入 `unavailable`，不参与可见计数或求和。

## 注入与错误码

```bat
python -m config_diag inject --kind spec
python -m config_diag inject --kind cycle
python -m config_diag inject --kind snapshot
```

| 代码 | 含义 |
|---|---|
| `E_SPEC` | 声明、客户端或变更样本结构/取值/版本非法 |
| `E_CYCLE` | 派生依赖图存在环；`context.spec_category` 同时标记为 `E_SPEC` 类别 |
| `E_SNAPSHOT` | 账本格式或内容摘要不符，已备份并触发快照整体替换 |
| `E_DIVERGE` | 增量路径与全量重建路径首个差异 |
| `E_REPLAY` | 恢复重放与账本首个差异 |
| `E_PENDING` | 存在未知事务，必须先重跑 |
| `E_VERSION` | 键或操作版本不满足单调递增 |
| `E_STATE` | 当前账本状态不允许该操作 |
| `E_GUI` | 当前 Python 无法加载 Tkinter |

所有诊断写入 `logs/diagnostics.jsonl`，每行包含时间、PID、代码、事件、级别、消息和结构化 context。

## Tkinter 诊断台

```bat
python -m config_diag gui
```

界面稳定展示：

- 配置键列表：key、取值、键级版本、active/invalid/expired 状态、摘要。
- 稳定分层派生依赖图，箭头方向为引用键到下游键。
- 最近一轮失效集合和逐步传播链。
- 当前视图版本与摘要、陈旧视图数、陈旧轨迹数。
- 账本格式/版本/集合版本/派生口径版本/内容摘要。
- 快照、视图、journal 列表。
- 初始化、应用下一条变更、计时回收、模拟轮询、恢复重放按钮。

所有列表和图中同层节点均按 key 排序；旧视图条目和旧结论不参与界面展示统计。

## 测试与验收

```bat
python -m unittest discover -v
python scripts\run_demo.py
```

测试覆盖 27 个场景，包括：

- 非法键/字段、缺少引用、成环、客户端非法引用、版本回退。
- WAL 下提交前不可见、提交后整体可见。
- 操作重试幂等、无重复键记录。
- 稳定拓扑传播、未受影响键字节与版本不变。
- 单调时钟过期、下游失效、重复回收为空。
- current/stale 视图、陈旧轨迹、整视图替换、确定性轮询。
- 增量/全量逐键一致及 `E_DIVERGE` 注入。
- 快照摘要不符、只读备份、整体替换。
- 强杀后 unknown、需重跑、重跑后逐键重放。
- `E_SPEC/E_CYCLE/E_SNAPSHOT` 三类注入与结构化日志。

实际运行结果见：

- `artifacts/validation_results.md`
- `artifacts/demo_output.json`
- `artifacts/test_output.txt`
- `artifacts/demo_run_output.json`
- `artifacts/diagnostics.jsonl`

本次验收环境为 Linux WSL，提供了 `libpython` 与 `numpy`，但缺少 `libtk8.6.so`，因此只能完成 GUI 源码编译和无头业务验收；Windows 官方 Python 安装包应直接提供 Tk。若 Windows 上运行 `gui` 报 `E_GUI`，请使用官方 python.org 安装包并勾选 Tcl/Tk。
