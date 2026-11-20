# EdgeCache Consistency Workbench

面向本地 Windows 的确定性边缘缓存一致性与诊断工作台。项目使用一个内容源子进程和至少三个边缘节点子进程，在 `127.0.0.1` 高位端口上以 JSONL over TCP 交换数据；节点缓存、容量逐出、派生失效、双版本账本、SQLite 持久化、快照整体替换与 Tkinter 诊断界面均由本地确定性样本驱动，不访问网络或外部服务。

## 能力概览

- 强校验本地规格：路径唯一、正整数 TTL、唯一缓存键字段、派生引用存在、派生图无环、节点容量与逐出规则合法。
- 真实子进程编排：内容源、三个边缘节点、端口预检、套接字超时握手、强制终止与持久化恢复。
- 版本化缓存条目：ETag、单调时钟写入基准、剩余 TTL、条目版本号；命中递减 TTL，过期回收。
- 确定性容量逐出：`stable_fifo`、`smallest_key`、`nearest_expiry`；访问顺序不会改变同一组条目的逐出集合。
- 稳定派生失效：沿依赖图计算全部直接/间接下游，使用堆式稳定拓扑序；轨迹可逐字节复现。
- 单轮版本绑定：每个账本轮次只绑定一个 `content_version` 和一个 `policy_version`；跨轮条目均标记为陈旧。
- SQLite 轮次事务：条目、逐出、传播轨迹和进度游标在同一事务中提交；中断轮次重启后标记为 `unknown` 并要求重跑。
- 快照与重放：保存版本集合、内容摘要、缓存条目和传播轨迹；版本集合不一致时整体替换，旧快照只读保留。
- 可复核恢复：节点或内容源被强制终止后，只从最新完整账本恢复；强制终止不提供伪造退出报告。
- Tkinter 工作台：稳定排序展示条目、ETag、内容版本、剩余 TTL、逐出、失效链与账本版本。
- 三类异常注入：`E_CONTENT`、`E_CYCLE`、`E_SNAPSHOT`；端口冲突与恢复差异分别报告 `E_PORT`、`E_REPLAY`。

## Windows 环境准备

1. 安装 Python 3.11 或 3.12 官方发行包，并勾选 Tcl/Tk。
2. 在项目根目录确认标准库可用：

```powershell
py -3.12 -c "import sqlite3, socket, subprocess, tkinter; print(sqlite3.sqlite_version, tkinter.TkVersion)"
```

3. 核心运行时仅依赖标准库。若要重新生成确定性数值样本，按官方 wheel 安装 NumPy：

```powershell
py -3.12 -m pip install --only-binary=:all: -r requirements-samples.txt
py -3.12 tools\generate_samples.py
```

样本生成器固定使用 `numpy.random.default_rng(20261121)`，运行时不依赖 NumPy。项目不使用 Docker、容器编排或外部服务。

## 快速开始

```powershell
py -3.12 -m edgecache.cli validate --spec samples\demo_spec.json
py -3.12 -m edgecache.cli demo --workspace runtime\demo --port-base 46000
py -3.12 -m edgecache.cli demo --workspace runtime\demo --port-base 46000 --gui
```

演示顺序固定为：

1. 冷启动填充：启动内容源与 `edge-a`、`edge-b`、`edge-c`，各节点拉取内容并按容量逐出。
2. 内容版本更新：`/data/root` 从 v1 演进到 v2，稳定失效 `/data/left`、`/data/right`、`/page/home`。
3. 快照恢复重放：保存 v2 快照，再演进到 v3；加载 v2 快照时检测版本集合不一致，创建新的整体替换轮次并重放传播轨迹。

结果写入 `runtime\demo\report.json`。`--gui` 在演示结束后打开只读诊断界面。

### 异常注入

```powershell
py -3.12 -m edgecache.cli demo --workspace runtime\bad-content --port-base 46100 --inject content_truncation
py -3.12 -m edgecache.cli demo --workspace runtime\bad-cycle --port-base 46100 --inject cycle
py -3.12 -m edgecache.cli demo --workspace runtime\bad-snapshot --port-base 46100 --inject snapshot_digest_mismatch
```

这些命令以退出码 `2` 和稳定 JSON 报告：

- `content_truncation` -> `E_CONTENT`，定位到 `/data/root` 的 `content_b64`。
- `cycle` -> `E_CYCLE`，同时以 `details.spec_rejection = E_SPEC` 表明加载层整体拒绝。
- `snapshot_digest_mismatch` -> `E_SNAPSHOT`，定位到 `content_digest`。

`--port-base` 必须对应至少四个连续空闲的高位回环端口。端口已占用时，在启动任何子进程前报告 `E_PORT`。

### 查看账本与工作台

```powershell
py -3.12 -m edgecache.cli view --db runtime\demo\ledger.sqlite3
py -3.12 -m edgecache.cli gui --db runtime\demo\ledger.sqlite3
```

工作台只读取当前账本视图。刷新不会追加账本版本、不会修改条目或轨迹；历史版本显示为 `stale=true`，并且不参与当前统计。

## 本地规格格式

完整示例见 `samples\demo_spec.json`。核心字段如下：

```json
{
  "schema": "edgecache.spec/v1",
  "policy": {
    "version": "policy-v1",
    "key_fields": ["node_id", "path", "etag"],
    "nodes": [
      {"node_id": "edge-a", "capacity": 4, "eviction_rule": "stable_fifo"},
      {"node_id": "edge-b", "capacity": 5, "eviction_rule": "smallest_key"},
      {"node_id": "edge-c", "capacity": 3, "eviction_rule": "nearest_expiry"}
    ]
  },
  "content": [
    {
      "path": "/data/root",
      "etag": "root-v1",
      "ttl_ms": 30000,
      "content_b64": "Uk9PVDoyMDA6djE=",
      "content_sha256": "<64 hex characters>",
      "derived_from": [],
      "derivation": null
    },
    {
      "path": "/page/home",
      "etag": "page-v1",
      "ttl_ms": 50000,
      "content_b64": "...",
      "content_sha256": "...",
      "derived_from": ["/data/left", "/data/right", "/api/config"],
      "derivation": {"rule": "concat_v1", "separator": "\n"}
    }
  ]
}
```

约束：

- `content[].path` 必须唯一且为绝对路径。
- `ttl_ms` 必须是正整数。
- `policy.key_fields` 非空、字段允许 `node_id`、`path`、`etag`，且不得重复。
- 至少声明三个节点；`node_id` 唯一；容量为正整数。
- `derived_from` 必须是已声明路径，且派生图无环。
- 内容摘要必须匹配；派生路径的字节必须等于其依赖内容按声明规则组合的结果。
- 变更内容字段时必须同时变更该路径 ETag；单个内容演进轮次不允许混入策略变更或无触发源变化。

## 账本与快照语义

`ledger.sqlite3` 中的每一轮包含：

- 单调递增 `ledger_version`、单一 `content_version`、单一 `policy_version`。
- 该轮完整的节点条目、逐出事件、传播轨迹和进度游标。
- `pending`、`complete`、`unknown` 状态；只有 `complete` 轮次可作为恢复依据。

完成一轮时，条目/事件/轨迹/游标/当前版本指针在一个 SQLite 事务内提交。进程在提交前终止时，重启会将 `pending` 轮次标记为 `unknown`，并恢复最后一个 `complete` 轮次。

快照包含账本版本、内容版本集合、策略版本与摘要、完整内容记录、派生图、缓存条目、逐出与传播轨迹。加载快照时先验证内容摘要和逐条目版本归属，再按依赖图重放轨迹；任一不一致报告 `E_REPLAY` 并给出首个触发源路径与受影响路径。若快照版本集合与磁盘当前状态不同，恢复会创建新的账本版本，仅包含快照条目与派生图，不进行增量合并。先前快照文件被设为只读备份。

## 子进程与协议

编排器使用当前 Python 解释器执行 `python -m edgecache.worker`：

```powershell
py -3.12 -m edgecache.worker source --spec samples\demo_spec.json --host 127.0.0.1 --port 46000
py -3.12 -m edgecache.worker node --node-id edge-a --source-host 127.0.0.1 --source-port 46000 --port 46001
```

也可使用配套脚本 `scripts\run_demo.py`、`scripts\run_source.py`、`scripts\run_node.py`；环境自检与端口查找分别见 `tools\windows_check.py`、`tools\find_free_ports.py`。

常见 JSONL 操作：

- 内容源：`hello`、`snapshot`、`status`、`reload`、`shutdown`。
- 边缘节点：`hello`、`configure`、`sync`、`apply_propagation`、`restore`、`state`、`get`、`shutdown`。

节点在 `sync` / `apply_propagation` 时主动连接内容源拉取快照，再回报条目与逐出事件。所有握手均使用真实套接字超时；强制终止通过 `TerminateProcess`/`SIGKILL` 语义完成，恢复报告明确为 `exit_report=null`。

## 测试

```powershell
py -3.12 -m unittest discover -s tests -v
```

覆盖规格校验、TTL、访问顺序无关逐出、确定性传播、版本绑定、陈旧标记、待定轮次未知化、端口冲突、真实套接字超时、强制终止恢复、策略演进、快照只读备份、整体替换、`E_REPLAY`/`E_SNAPSHOT` 定位以及 CLI 异常码。

当前执行环境的实际验证记录见 `docs\VERIFICATION.md`，设计与不变量见 `docs\DESIGN.md`。
