# 本地模型配置分发调度工作台

面向 Windows 本地 Python 3.10+ 的确定性模型配置分发、优先级抢占、失败重试、回执衰减、SQLite 账本恢复与 Tkinter 诊断工作台。

系统完全运行在 `127.0.0.1` 回环 TCP 上，不依赖 GPU、真实模型、外部站点或容器。每个推理节点是独立子进程，通过 JSON 行协议接收带整数版本号的配置切片，并使用 NumPy 整数前向计算生成确定性摘要。

## 快速开始

```powershell
# 在项目根目录
py -3.10 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt

# 校验示例
.venv\Scripts\python -m modeldash validate --spec examples\demo_v1.json --json

# 两次完整演示：抢占、重试、回执衰减、整体替换、实时回放、逐字节比较
.venv\Scripts\python demo.py --data-dir runtime\demo --repeat 2

# 单次调度与实时回放
.venv\Scripts\python -m modeldash run --spec examples\demo_v2.json --data-dir runtime\manual --json
.venv\Scripts\python -m modeldash replay --spec examples\demo_v2.json --data-dir runtime\manual --json

# 只读诊断文本或 Tkinter GUI
.venv\Scripts\python -m modeldash diagnose --data-dir runtime\demo\run-1
.venv\Scripts\python -m modeldash gui --db runtime\demo\run-1\ledger.sqlite3

# 测试
.venv\Scripts\python -m pytest -q
```

`demo.py` 会从 `47831..47999` 选择三个空闲高位回环端口，将两个示例规格物化到运行时目录。两次运行使用相同配置、种子和端口，并对以下内容做规范化 JSON 的逐字节比较：

- 每任务结局、输出摘要与分片输出摘要；
- 每个回执的输出、状态、衰减量和不可采信分；
- 每节点不可采信排序；
- 抢占、暂停与抢占决策序列。

运行产物默认位于 `runtime/`（已被 `.gitignore` 忽略）：

```text
runtime/<session>/
  ledger.sqlite3
  logs/workbench.jsonl
  logs/node-<id>.log
  snapshots/
```

## 命令行

| 命令 | 作用 |
|---|---|
| `python -m modeldash validate --spec FILE` | 强校验规格；失败输出首个违规字段 `E_SPEC`，且不创建账本状态 |
| `python -m modeldash run --spec FILE --data-dir DIR` | 启动节点子进程、执行调度、事务提交并在启动/抢占/结束时创建快照 |
| `python -m modeldash replay --spec FILE --data-dir DIR` | 从快照识别剩余任务，校验账本轮次，并实时重发回执比对摘要；`--offline` 仅做账本复算 |
| `python -m modeldash diagnose --data-dir DIR` | 输出稳定排序的只读诊断视图 |
| `python -m modeldash gui --db FILE` | 打开 Tkinter 诊断界面 |
| `python -m modeldash snapshot --spec FILE --data-dir DIR` | 从最近已提交状态创建快照 |

## 规格与强校验

示例见 `examples/demo_v1.json`、`examples/demo_v2.json`。顶层结构：

```json
{
  "format_version": 1,
  "spec_version": 2,
  "seed": 20261122,
  "input": {"size": 12, "scale": 37},
  "nodes": [
    {"id": "n1", "host": "127.0.0.1", "port": 47831, "quota": 1,
     "shard_delay_ms": 90, "faults": []}
  ],
  "configs": [
    {"version": 3, "priority": 5, "target_nodes": ["n1"], "shards": 3,
     "items": {"alpha": 21, "beta": 3.5, "gamma": "v3"}}
  ],
  "budget": {
    "global_concurrency": 2,
    "timeout_ms": 3000,
    "retry_backoff_ms": [40, 80],
    "max_retries": 2
  },
  "tasks": [
    {"id": "t1", "config_version": 3, "node": "n1", "priority": 9,
     "release_after_ms": 0}
  ]
}
```

强校验规则包括：

- `format_version == 1`；`spec_version` 为正整数；配置版本严格递增且唯一；
- 配置项键匹配 `^[A-Za-z][A-Za-z0-9_.-]{0,63}$`；值仅支持有符号 64 位整数、有限浮点、布尔和最长 256 字符的字符串；
- 目标节点必须存在；主机必须是字面量 `127.0.0.1`；端口处于 `1024..65535` 且全局唯一；
- 任务优先级为非负整数；分片数为正整数；任务节点必须属于对应配置的目标节点；
- 任一违规整体拒绝为 `E_SPEC`，并在 `field` 中给出首个违规路径。CLI 在创建账本前完成校验。

规格版本演进时，若 `spec_version` 未增大、旧配置版本被复用但内容改变、或当前配置版本倒退，会在写账本前拒绝。

## 调度语义

### 配额与排队

- 同时运行的**任务数**受全局并发上限限制；同节点运行任务数受节点 `quota` 限制；
- 就绪任务按 `(-priority, task_id)` 排序；
- 任务等待配额时记录单调时钟毫秒窗口；任务完成或在退避时记录重试次数、已用配额时长和输出检查点；
- 每次分片下发都有真实 `socket` 连接/读写超时。

### 抢占与挂起

显式抢占规则：

1. 只有严格更高优先级任务可抢占低优先级任务；
2. 先选择持有目标节点配额的较低优先级任务；
3. 若仍被全局配额阻塞，再选择优先级最低的较低优先级运行任务；
4. 同级按已完成分片数少者、再按任务标识排序；
5. 被选中任务置为 `PREEMPT_REQUESTED`，完成**当前分片**后在分片边界写入检查点并进入 `PAUSED`；
6. 恢复时仅从未完成分片继续，已完成分片不会丢失或重复。

每次抢占记录被抢占任务、发起任务、请求时/挂起时的分片号、占用的全局/节点资源、最后输出摘要和状态。若任务恰好在最后一个分片结束，抢占记录为 `RETIRED_AT_COMPLETION`，不会产生空挂起。

### 重试与退避交互

- 失败后第 `i` 次重试等待 `retry_backoff_ms[i]`；达到 `max_retries` 后以 `E_EXHAUST` 终结；
- 退避等待期间释放全局和节点配额；重试使用相同语义请求和稳定 `request_id`，节点按请求摘要做幂等缓存；
- 高优先级任务被配额阻塞时，会逐任务记录较低优先级退避任务的裁决：
  - `WAIT_EXEMPT / BACKOFF_DOES_NOT_HOLD_QUOTA`：退避任务不占配额，无需抢占；
  - `TERMINAL_NOT_PREEMPTABLE`：已终结；
  - `ALREADY_SUSPENDED_NO_QUOTA`：已挂起；
  - `PREEMPTABLE`：运行中且持有所需资源。

## 回执衰减与不可采信度

每个成功分片产生一个回执，绑定：

- 节点、任务、配置版本、分片号；
- 输入摘要、输出摘要；
- 生成版本、衰减计数、依赖引用、信任状态、不可采信分。

对当前配置版本 `V`，算法为纯函数并始终按稳定键排序：

- 同一 `(节点, 分片, 输入摘要, 分片数)` 存在当前版本重算回执时，比较输出摘要；
- 输出不同：`UNTRUSTED_MISMATCH`；
- 同节点、共享配置项依赖且受直接失配影响的旧回执：`UNTRUSTED_DEPENDENCY`；
- 无当前版本重算：保守记为 `UNTRUSTED_UNVERIFIED`；
- 当前版本重算一致：`VERIFIED_UNDER_CURRENT`；
- 同一比较槽位出现互相矛盾的当前版本输出：`UNTRUSTED_INCONSISTENT`。

整数规则：

```text
impact = 0(verified) | 1(dependency/unverified) | 2(direct mismatch)
version_gap = current_version - generated_version
decay_count = version_gap + 2 * impact
untrust_score = 100 * impact + 10 * version_gap + 3 * decay_count
node_score = sum(untrusted receipt scores on that node)
```

节点排序为 `(-node_score, node_id)`，排名写入账本。输入相同的执行顺序不会改变结果。

## 版本替换与恢复

- `spec_version` 增大后，新会话整体替换当前视图；旧轨迹、任务、抢占、回执、不可采信度、快照全部标记 `is_stale=1`；
- 替换摘要稳定列出受影响节点、任务、旧结论计数与汇总摘要；
- 每完成一个调度事件，在一个 SQLite `BEGIN IMMEDIATE` 事务中写入轮次记录、进度游标、全部任务状态、重试、抢占、回执和节点不可采信度；
- `rounds` 与 `sessions.state_json` 都带内容摘要，读取时重新计算校验；
- 强制终止后只恢复最后完整事务。最后事务中仍为运行态的任务被标记未知分片，清除在途状态并从最后提交的分片边界重跑；
- 快照格式版本或内容摘要不一致时标记陈旧，并以空的新会话整体替换，不增量合并。

## 回放校验

`ReplayVerifier` 检查：

- 轮次序号和进度游标严格递增；
- 已完成分片连续且不重复；
- 每任务聚合摘要等于分片摘要聚合；
- 等待总时长等于单调时钟窗口之和；
- 抢占挂起位置有效；
- 回执状态、衰减量、不可采信分和节点排序可以逐项复算；
- 实时模式下从 `session-start` 快照识别剩余任务，并重新向节点下发账本中的回执请求，比较输入/输出摘要。

首个差异以 `E_REPLAY` 报出，并携带 `first_task_id`、`node_id`、`field`、`expected`、`actual`。

## Tkinter 诊断界面

六个只读页签：

1. 任务优先级队列与检查点；
2. 抢占记录与占用资源；
3. 抢占/退避裁决；
4. 回执摘要账本；
5. 每节点不可采信排序；
6. 快照列表。

刷新只执行 `SELECT`，排序固定为优先级降序/任务标识、排名/节点标识、节点/任务/分片。当前交付验证容器未安装 Tkinter，因此 GUI 窗口未在该容器启动；数据查询、文本渲染和“刷新不改账本”已由测试覆盖。Windows 官方 Python 安装自带 Tkinter。

## 实际验证结果

验证环境：Python 3.11.0rc1、NumPy 1.26.4、SQLite 3.37.2（Linux 验证容器；目标运行环境为 Windows Python 3.10+）。

```text
python3 -m pytest -q
36 passed in 8.22s
```

`python3 demo.py --data-dir runtime/demo_final2 --repeat 2` 的实际结果：

```text
DEMO_OK repeats=2 byte_identical={'v1': True, 'v2': True}
v1 preemptions=2
v1 WAIT_EXEMPT: high=t-v2-n1 victim=t-v1-n3 reason=BACKOFF_DOES_NOT_HOLD_QUOTA
v1 ranking=[
  {'node_id': 'n1', 'rank': 1, 'score': 675, 'untrusted_receipts': 3, 'receipt_count': 6, 'max_decay': 5},
  {'node_id': 'n3', 'rank': 2, 'score': 675, 'untrusted_receipts': 3, 'receipt_count': 6, 'max_decay': 5},
  {'node_id': 'n2', 'rank': 3, 'score': 357, 'untrusted_receipts': 3, 'receipt_count': 3, 'max_decay': 3}]
v2 activation=replaced ranking=[
  {'node_id': 'n1', 'rank': 1, 'score': 0, ...},
  {'node_id': 'n2', 'rank': 2, 'score': 0, ...},
  {'node_id': 'n3', 'rank': 3, 'score': 0, ...}]
replay={'ok': True, 'code': 'REPLAY_OK', 'redispatch_count': 9,
        'remaining_task_ids': ['t-v3-n1', 't-v3-n2', 't-v3-n3'],
        'checks': 105}
```

关键确定性摘要（同一报告可在 `runtime/demo_final2/demo_report.json` 查看）：

> 本次实际运行报告同时固化在 `docs/demo_report.json`，便于在未保留 `runtime/` 的环境中复核。

```text
v1 deterministic_digest = 74b87ef9e8f5b51b595c41f868b133e01d17ec2dbcee216125c05c9b6d47a67b
v1 node_output_digest  = 11e60eb410a9193c6b096a8f49566b65d6def2465814855e1bb6e8d8b98cb873
v1 untrust_digest      = 1eeb8b75278920a7164479b9c963506aca3f569834ef6a8ffacf6a65817ec605
v2 deterministic_digest = 46c84ef99f98ae91e7b4a9268a5fd4e55d024b740d916ece1b643a573a45a72f
v2 node_output_digest  = d2457f99cfa50e5ed23c946649d61753bf1ea47722bbe59220f1f5fe5b09a70c
v2 untrust_digest      = 11dd1829a870d442d4c8a47b3eb5f87360d33c6adf77b948c5ae5905179869de
```

测试同时覆盖：真实套接字超时、端口占用与握手失败分别报错、节点子进程确定性、全局/节点配额上限、边界抢占与断点恢复、退避不占配额、`WAIT_EXEMPT`、`E_EXHAUST`、快照摘要损坏整体替换、账本复算、首个 `E_REPLAY` 差异和强制终止后的未知分片重跑。

## 代码结构

```text
modeldash/
  spec.py          规格数据模型与强校验
  protocol.py      JSON 行协议、真实套接字超时、端口检查
  inference.py     确定性整数 NumPy 前向与摘要
  node_server.py   节点子进程
  nodes.py         子进程监管、启动重试与调度客户端
  scheduler.py     并发调度、抢占、退避、检查点、恢复
  receipts.py      回执衰减、依赖传播、节点不可采信排序
  storage.py       SQLite 事务账本、版本替换、快照
  replay.py        一致性回放与首个差异定位
  gui/             Tkinter 界面与稳定文本视图
  cli.py           命令行入口
demo.py            双运行端到端演示
examples/          确定性示例规格
tests/             36 项单元/集成测试
docs/DESIGN.md     更详细的算法、事务与不变量说明
docs/demo_report.json  最近一次两次运行与实时回放的详细验证报告
```
