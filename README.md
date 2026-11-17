# 多集群配置分发诊断台

这是一个完全本地、可复现的多集群配置分发诊断台。它在 Windows 回环 TCP 端口上为每个集群启动确定性模拟服务，把每个集群每个配置修订的分发结果提交到 SQLite 结果账本，并同时提供：

- Tkinter 增量视图：只刷新受当前修订影响的集群节点，但根绑定整体切换。
- 权威整树重算：每次从账本根修订重建完整版本树，逐节点重新调用本地模拟服务。
- 查询/重算延迟观测：单调纳秒时钟、整数定点统计、显式预算与 `E_LATENCY`。
- 崩溃恢复与快照重放：未提交事务不进入账本，未知节点要求重跑。
- 三类异常注入：修订环、端口占用、账本内容摘要损坏。

样例、时钟基准、聚合真值和异常注入均在本机完成，不访问真实集群、外部数据或线上服务。

## 环境

- Python 3.11 或更高版本。
- Windows 官方 Python 安装包默认包含 Tkinter；可用 `python -c "import tkinter; print(tkinter.TkVersion)"` 检查。
- NumPy（固定种子样本与分组聚合）。

Windows PowerShell 示例：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -c "import tkinter, numpy; print(tkinter.TkVersion, numpy.__version__)"
```

项目不依赖 Docker、容器、数据库服务或网络目标。

## 快速开始

在项目根目录执行：

```powershell
python -m cluster_diag validate --spec examples/spec.json
python -m cluster_diag run --spec examples/spec.json --workdir runtime
python -m cluster_diag gui --spec examples/spec.json --workdir runtime --latency-budget-ms 50
```

`run` 会依次提交到最新修订、执行权威整树重算，并从快照恢复后重放。首次运行会创建：

- `runtime/results.sqlite3`：结果版本树账本。
- `runtime/session_snapshot.json`：会话快照。
- `runtime/logs/events.jsonl`：结构化 JSON Lines 日志。
- `runtime/logs/service-<cluster>.jsonl`：每个模拟子进程的日志。

## 规格格式与强校验

样例见 `examples/spec.json`。规格是一个完整 JSON 文档：

```json
{
  "schema_version": 1,
  "seed": 20261118,
  "clusters": [
    {"id": "edge-a", "host": "127.0.0.1", "port": 45201, "nodes": ["a-1", "a-2"]}
  ],
  "tasks": [
    {
      "id": "core-network",
      "config_keys": ["net.timeout_ms"],
      "targets": ["edge-a"],
      "aggregation": "mean",
      "sample_count": 256
    }
  ],
  "revisions": [
    {"version": 1, "parent": null, "config": {"net.timeout_ms": 1200}, "task_ids": ["core-network"]}
  ]
}
```

约束如下；任何一项失败都会整体拒绝并返回 `E_SPEC` 及字段路径，不会跳过非法条目继续运行：

- `schema_version` 必须为 `1`，`seed` 为非负 64 位整数。
- 集群标识唯一；`host` 只能是回环字面量 `127.0.0.1` 或 `::1`；端口范围 `1024..65535` 且全局唯一；节点列表非空且元素唯一。
- 修订版本必须为正整数、在文件中严格递增、形成单链；首修订 `parent=null`，后续父修订必须存在且引用前一修订。环会单独识别。
- 任务标识唯一；配置键和目标集群列表非空；目标集群必须存在；聚合口径为 `sum/mean/min/max/count`；样本数为 `1..1,000,000`。
- 每个修订列出的任务必须存在；任务引用的每个配置键必须存在于该修订的 `config` 中。
- 第一修订必须覆盖所有集群，保证初始版本树完整。
- 同一修订中，一个集群只能被一个任务命中。因为账本主键是 `(cluster_id, revision)`，该约束保证“每集群每修订一个已提交节点”语义明确。
- 规格中未知字段、非有限 JSON 数值、类型错误等均被拒绝。

普通规格错误使用 `E_SPEC`；明确由 `--inject-cycle` 触发或识别到的修订环使用 `E_REV`。

## 确定性与结果内容

每个集群由一个独立 Python 子进程模拟，使用 JSON 行协议：

- 请求：`{"command":"distribute","revision":2,"task_id":"core-network",...}`
- 响应：规范 JSON 聚合摘要、摘要摘要值、整数 `compute_ms` 和会话令牌。

样本使用 NumPy `PCG64` 生成，种子由全局 seed、任务、集群、修订、配置键和版本化样本标签共同决定。数值先映射为 `1e-6` 定点整数，再执行分组聚合，避免浮点统计口径漂移。同一请求重复发送时，`aggregate_summary_canonical` 与 `aggregate_summary_digest` 逐字节一致；`compute_ms` 是独立观测值，不参与内容摘要。

节点内容摘要覆盖任务、集群、修订、父修订和规范聚合摘要，不包含完成时刻或耗时。因此重复计算可稳定核对内容。

## 账本与根不变量

SQLite 使用 WAL、`synchronous=FULL` 和显式 `BEGIN IMMEDIATE` 事务。一次修订提交在同一事务中完成：

1. 校验当前根等于预期父修订。
2. 写入该修订受影响的全部集群节点。
3. 重建新根的有效节点集并校验所有声明集群均完整。
4. 写入提交记录与整树快照摘要。
5. 切换单例根指针。
6. 提交事务。

因此根指针任意时刻都对应完整、可回溯的树。相同 `(cluster_id, revision)` 受主键保护。旧根可通过旧修订号从账本重建，且不会因新修订写入而改变。

未受新修订影响的集群不写新节点；新根中的该集群仍指向其最近已提交节点。这里没有把新旧节点“合并成新树”：整棵新根有效集由账本统一重建，视图随后整体换绑。

## 增量视图、整体换根与陈旧排除

Tkinter 视图模型维护：

- 当前根绑定的不可变 `TreeSnapshot`。
- 每集群有效的 `ViewItem`。
- 历史根快照，只读保留。

提交新修订后：

- 根绑定一次性替换为账本重建的新快照。
- UI 只更新 `affected_clusters`/实际变化集群的节点数据行；未变化节点的内容摘要不重算，仅更新其单一根绑定标签。
- 全部旧根行被标记为陈旧并形成稳定排序的陈旧节点集合。
- 当前延迟统计只纳入 `bound_root_revision == 当前根` 且未陈旧的记录；旧根记录仍保留为账本真值，但不混入当前统计。
- 换根报告稳定列出受影响集群、实际变化集群、旧/新修订及摘要摘要值、陈旧节点集合。

权威路径不依赖视图模型。它总是从当前根修订重建完整树，并逐个历史节点调用对应集群模拟服务重新计算聚合与内容摘要，同时检查提交链、父子引用、根指针、任务归属和根快照摘要。首个差异以 `E_VIEW` 返回，包含集群标识、修订版本与字段。

## 延迟口径

- 时钟基准：`time.monotonic_ns()`。
- 真值来源：SQLite `latency_events` 表逐次查询记录。
- 报告单位：`ms_x1000`，即 1 个整数单位 = `0.001 ms`。
- 量化：纳秒跨度舍入到整数微秒，再直接作为 `ms_x1000`；量化误差上界保守声明为 `1` 个 `ms_x1000` 单位。
- 聚合：分组键为 `(kind, cluster_id, revision)`；均值四舍五入到整数单位，P95 使用 nearest-rank，最大值直接取整数样本。
- 每次 Tk 查询和每次权威重算都记录单调起止值与耗时。
- `--latency-budget-ms` 是显式 UI 查询预算。超预算仍返回真实查询结果，同时把事件标记 `E_LATENCY` 并记录超出量、阈值和当前根。
- 不使用人为延迟，不使用后台线程超时来伪称查询完成；GUI 操作同步执行并只在真实完成后更新状态。

权威重算的延迟事件没有显式 UI 预算，`cluster_id` 为 `null`，以 `kind=authoritative` 区分。

## 快照、崩溃恢复与重放

会话快照保存：

- 根修订与根快照摘要。
- 根内全部历史节点的内容摘要集合。
- 延迟统计口径版本与按当前根聚合的统计量。

重新打开时只比较账本当前根和节点摘要集合。若根、节点集合、快照结构或延迟口径版本不一致，整个会话状态从账本重建并整体替换，不做增量合并。

SQLite 保证强杀时未提交事务回滚。重启后：

- 只加载账本中已提交节点和根指针。
- 规格中位于当前根之后的修订节点标记为 `unknown`，`required_action` 为 `rerun`。
- 不生成也无法生成“强杀瞬间退出报告”。

恢复后的一致性重放会按根重建树，逐节点在本机模拟服务重算，逐项比较节点摘要、父子引用、根指针和延迟统计。任何不可复现项返回 `E_REPLAY` 并说明 `basis`、集群与修订（适用时）。

## 命令

```text
python -m cluster_diag validate --spec SPEC [--workdir DIR]
python -m cluster_diag commit   --spec SPEC [--workdir DIR] [--revision next|latest|N]
python -m cluster_diag query    --spec SPEC [--workdir DIR] --cluster ID [--revision N] [--latency-budget-ms 50]
python -m cluster_diag verify   --spec SPEC [--workdir DIR]
python -m cluster_diag recover  --spec SPEC [--workdir DIR] [--no-replay]
python -m cluster_diag run      --spec SPEC [--workdir DIR] [--revision latest] [--no-verify] [--no-recover]
python -m cluster_diag gui      --spec SPEC [--workdir DIR] [--latency-budget-ms 50]
```

所有命令输出一个 JSON 文档。错误输出含稳定 `code`、`message`、`field_path`/`details`；异常同时写入 `workdir/logs/events.jsonl`。

## 异常注入

```powershell
# E_REV：将最后一个修订的 parent 指向自身后加载
python -m cluster_diag validate --spec examples/spec.json --workdir inj-rev --inject-cycle

# E_PORT：先用参数持有首个集群端口，再启动模拟服务
python -m cluster_diag run --spec examples/spec.json --workdir inj-port --inject-port-conflict

# E_LEDGER：已有账本后篡改一个节点 content_digest，再执行权威路径
python -m cluster_diag run --spec examples/spec.json --workdir inj-ledger
python -m cluster_diag verify --spec examples/spec.json --workdir inj-ledger --inject-ledger-corruption
```

注入只由命令行参数触发，不访问外部服务。

## 错误码

| 错误码 | 含义 |
| --- | --- |
| `E_SPEC` | 规格整体非法，含字段路径 |
| `E_REV` | 修订环（注入/专项识别） |
| `E_PORT` | 回环端口占用、绑定或连接失败 |
| `E_LEDGER` | 账本节点内容摘要、快照或事务不变量损坏 |
| `E_VIEW` | 权威路径发现的视图/重算差异 |
| `E_REPLAY` | 快照恢复后的重放不可复现 |
| `E_LATENCY` | UI 查询超过显式预算的非致命告警 |
| `E_SERVICE` | 模拟子进程或协议失败 |
| `E_ENV` | 运行环境缺失，例如 Tkinter 不可用 |

## 测试与验证

```powershell
python -m unittest discover -v
python scripts/verify.py
```

测试覆盖：规格整体拒绝、环、端口冲突、确定性摘要、原子提交、增量视图、换根陈旧排除、分权重重算、强杀回滚、未知节点、快照整体替换、`E_VIEW`、`E_REPLAY`、延迟定点统计、三类 CLI 注入和 Tkinter 冒烟。

`scripts/verify.py` 会运行测试并生成 `artifacts/verification.json`，记录本次实际根修订、权威重算节点数、恢复重放节点数、环境版本与测试结果。

## 文件布局

```text
cluster_diag/
  spec.py         规格模型与强校验
  determinism.py  定点样本与确定性分组聚合
  service.py      回环 JSONL 模拟服务子进程
  transport.py    子进程管理与客户端
  ledger.py       SQLite 结果版本树账本
  view.py         增量视图与整体换根模型
  runtime.py      提交、查询、恢复编排
  authority.py    权威整树重算与恢复重放
  latency.py      整数定点延迟统计
  snapshot.py     会话快照识别与整体替换
  gui.py          Tkinter 诊断台
  cli.py          命令入口
examples/spec.json
tests/
scripts/verify.py
docs/ARCHITECTURE.md
```

更细的事务、状态转换与不变量说明见 `docs/ARCHITECTURE.md`。
