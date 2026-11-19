# ShardRep：本地分片副本修复中间件

ShardRep 是面向 Windows 本地教学与排障演示的确定性中间件。它以本地 JSON 声明分片组，在镜像/拼接关系图上增量传播失效，在全局槽位与节点配额约束下并行重打包，并把修复决策、调度轨迹、SQLite 提交点与版本化快照全部保存在工作区内。

组件不访问网络、不使用容器、不依赖外部服务。NumPy 仅在安装时作为字节缓冲的快速物化层；核心输出与纯标准库路径逐字节一致。Tkinter 诊断界面使用 Python 官方 Windows 安装包自带的 Tk。

## 快速开始

在 Windows PowerShell 中于项目根目录执行：

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m shardrep init-sample
python -m shardrep validate --spec samples/spec.json
python -m shardrep demo --root .
python -m shardrep gui --spec samples/spec.json
```

也可直接运行 `run_demo.bat`、`run_gui.bat` 或 `run_tests.bat`。所有路径默认相对工作区，不访问回环端口。

`demo` 会按固定顺序完成三个阶段：

1. 注入截断损坏，推导失效链并执行完整修复与全量重打包。
2. 再注入字节翻转，仅对受影响分片 `s4 → s5 → s7` 增量重打包，同时校验 5 个未受影响分片的产物文件、摘要和版本全部不变，并与独立全量重打包结果逐项比较。
3. 保存快照；制造磁盘状态漂移后整体替换恢复；在干净工作区按快照记录的注入序列与预算参数重放并逐项比较。

结果写入 `demo-run/demo_report.json`，稳定事实写入 `demo-run/stable_facts.json`，结构化日志写入 `demo-run/logs/events.jsonl`，SQLite 账本为 `demo-run/ledger.sqlite3`。同样输入重复运行时 `stable_digest` 不变。

## 声明模型

```json
{
  "format": "shardrep-spec",
  "format_version": 1,
  "group_id": "classroom-repair",
  "group_version": 1,
  "mirror_graph_version": 1,
  "content": {"engine": "shake256-v1", "seed": 20261120},
  "nodes": [
    {
      "node_id": "node-a",
      "directory": "nodes/node-a",
      "pack_quota": 2,
      "repair_enabled": true,
      "shards": [
        {
          "shard_id": "node-a:s0",
          "logical_id": "s0",
          "seq": 0,
          "bytes": 1024,
          "digest": "sha256:...",
          "file": "s00.bin"
        }
      ]
    }
  ],
  "blocks": [
    {"block_id": "block-alpha", "seq_start": 0, "seq_end": 2, "checksum": "sha256:..."}
  ],
  "relations": [
    {"kind": "mirror", "source": "s1", "target": "s2"},
    {"kind": "join", "sources": ["s3", "s4"], "target": "s5"}
  ]
}
```

- `shard_id` 是跨规格唯一的物理副本标识；`logical_id` 表示跨节点复制、关系和失效传播的逻辑分片。
- 同一逻辑分片在不同节点必须声明相同的 `seq`、`bytes` 与 `digest`。
- 块区间必须从 0 开始、首尾相接、不重叠、不越界，并覆盖全部逻辑序号。
- `mirror`/`join` 的所有来源和目标必须存在，关系图必须是 DAG。
- 任一非法声明整体抛出 `E_SPEC`，携带 `shard_id` 与 JSON 字段路径，不返回部分装载对象。成环原因同时给出 `E_CYCLE`。

完整模式说明见 `docs/SPEC_SCHEMA.md`。

## 命令行

```text
python -m shardrep init-sample [--root .] [--keep]
python -m shardrep validate --spec samples/spec.json
python -m shardrep inject --spec samples/spec.json --shard-id node-b:s4 --kind flip
python -m shardrep run --spec samples/spec.json --inject node-b:s4=flip --repair
python -m shardrep verify --spec samples/spec.json --incremental-pack demo-run/packs
python -m shardrep snapshot-restore --spec samples/spec.json \
  --snapshot demo-run/snapshots/session.json --state demo-run/state/current_state.json
python -m shardrep recover --db demo-run/ledger.sqlite3 --run-id crash-run
python -m shardrep anomaly --kind cycle --spec samples/spec.json
python -m shardrep anomaly --kind integrity --spec samples/spec.json --shard-id node-a:s2
python -m shardrep anomaly --kind snapshot --snapshot demo-run/snapshots/session.json
```

注入类型为 `truncate`、`flip`、`replace`、`delete`。`replace` 默认使用固定种子生成内容；脚本可用 `--replacement-hex` 指定确定性替换内容。注入只写物理分片文件，不修改 JSON 源码。

## Python API

```python
from shardrep import (
    PackScheduler,
    PackStore,
    ReplicaRegistry,
    apply_and_report,
    build_pack_tasks,
    load_spec,
    propagate_invalidation,
)

spec = load_spec("samples/spec.json")
registry = ReplicaRegistry.load(spec)
injection = registry.inject(shard_id="node-b:s4", kind="flip")
propagation = propagate_invalidation(spec, [injection.logical_id])
registry.mark_affected(propagation.affected)

quotas = {node.node_id: node.pack_quota for node in spec.nodes}
tasks = build_pack_tasks(spec, propagation.affected, quotas=quotas)
result = PackScheduler(
    store=PackStore("run/packs", spec),
    global_slots=2,
    node_quotas=quotas,
    ledger=None,
    run_id="api-example",
).run(tasks)

report = apply_and_report(spec, registry)
```

## 关键保证

- **确定性摘要**：SHAKE-256 生成样本字节，SHA-256 计算内容摘要；NumPy 有无不影响结果。
- **稳定传播**：拓扑排序使用稳定堆序，轨迹为每个种子到目标的最短且字典序最小路径；与节点顺序、关系声明顺序和字典遍历顺序无关。
- **增量正确性**：仅生成受影响逻辑分片的打包任务；未受影响产物文件不被重写。与全量重建比较摘要或块副本摘要失败时抛出 `E_INCR`，报告首个分片与字段。
- **双重预算**：`ThreadPoolExecutor(max_workers=global_slots)` 限制全局并发；提交前另行检查节点运行数，节点配额不足时任务留队。调度记录排队、开始、结束、时长、完成字节、版本和被跳过原因。
- **真实超时**：工作线程按 `time.monotonic()` 截止时间分块检查；超时任务提交 `timeout` 账本行后按 `max_retries` 重试，重试仍占用同一槽位与节点配额。
- **修复三分法**：完全修复、部分修复、不可修复计数与逐块副本状态、修复任务结果和副本总数自洽；不可修复块记录缺失序号与全部已尝试副本。
- **快照整体替换**：快照含整数格式版本、快照版本、内容摘要、组版本、图版本、副本摘要、打包清单、调度轨迹、修复报告、注入序列与预算。恢复时发现规格/组/图/磁盘摘要差异即整体替换状态，不做增量合并；旧快照复制为只读备份。
- **崩溃一致性**：任务先写 `running` claim；成果、字节数、预算消耗、进度游标与 committed 状态在一个 SQLite 事务中提交。强制终止后未提交 claim 标记为 `unknown_requires_rerun`，恢复只依据完整提交行，绝不生成伪造退出报告。

## 日志与账本

每个 JSONL 事件都含 `code`、`shard_id`、`version`、`timestamp`。关键业务码包括 `E_SPEC`、`E_CYCLE`、`E_INTEGRITY`、`E_INCR`、`E_UNRECOVERABLE`、`E_SNAPSHOT`、`E_REPLAY`。

SQLite 表：

- `event_log`：结构化事件。
- `task_claims`：任务运行声明。
- `task_commits`：原子提交的任务结局、完成字节、预算消耗与进度游标。
- `run_progress`：每次提交同步更新的累计字节和游标。

## 测试与文档

```powershell
python -m unittest discover -s tests -v
```

- `docs/DESIGN.md`：组件结构、传播/调度/恢复算法和事务边界。
- `docs/SPEC_SCHEMA.md`：JSON 字段与全部强校验规则。
- `docs/VALIDATION.md`：实际运行命令、测试项、确定性摘要与环境说明。

当前验收环境为无 Tkinter、无 NumPy 的 headless Linux Python 3.11；因此 GUI 只做编译验证，核心链路使用纯标准库回退完整运行。目标 Windows 官方 Python 3.10+ 可直接使用 Tkinter，并按 `requirements.txt` 安装 NumPy。

