# MeshDiag：本地服务网格流量治理诊断工作台

面向 Windows 本地环境的服务网格路由诊断实现。它把 v1、v2、v3 三代路由修订归一为同一内部模型，编译依赖图与执行计划，在全局/每服务并发预算下通过真实回环 HTTP 套接字分发请求，并把治理结论、尝试轨迹、快照与历史重放结果持久化到 SQLite 账本。

运行时只使用 Python 3.10+ 标准库。Tkinter 随 Windows Python 提供；不需要 NumPy、Docker 或容器环境。所有上游地址强制为 `127.0.0.1`。

## 快速开始（Windows）

```bat
cd /d <工作区目录>
python --version
python -c "import tkinter, sqlite3; print(tkinter.TkVersion, sqlite3.sqlite_version)"
python demo.py
python -m unittest discover -s tests -v
python -m meshdiag ui --workspace samples\workspace.json
```

也可以直接运行：

- `run_demo.bat`：完整可重复演示。
- `run_tests.bat`：30 项自动化测试。
- `run_ui.bat`：Tkinter 诊断工作台。

`demo.py` 每次清理并重建 `var/`，启动 `127.0.0.1:18181..18188` 的确定性模拟上游，依次执行：

1. v1/v2/v3 三格式归一与依赖图输出。
2. revision 1 冷启动治理。
3. 全部规则升级为 v3 的 revision 2 热轮次，并展示旧轮次 `stale=1`。
4. 故障注入轮次的快照重放校验。
5. 子进程强制终止后的原子快照恢复，未完成请求标记 `unknown` 并要求重跑。
6. 实例集合漂移负控，`E_REPLAY` 给出 `instance_set` 字段及具体依据。

## 样本与数据格式

`samples/workspace.json` 声明服务、实例、并发配额、默认兼容预算和文件路径。示例包含四个服务、八个回环实例：

```json
{
  "workspace_id": "local-mesh-demo",
  "global_concurrency": 4,
  "revision_file": "revision-r1.json",
  "requests_file": "requests-cold.json",
  "defaults": {"timeout_ms": 500, "retries": 0, "cooldown_ms": 1000},
  "services": [
    {
      "id": "catalog",
      "max_concurrency": 2,
      "instances": [
        {"id": "catalog-a", "host": "127.0.0.1", "port": 18181},
        {"id": "catalog-b", "host": "127.0.0.1", "port": 18182}
      ]
    }
  ]
}
```

顶层 `revision` 必须单调递增。每条规则独立声明 `format_version`：

```json
{"id":"r1","format_version":1,"prefix":"/catalog","target_service":"catalog"}
```

```json
{
  "id":"r2","format_version":2,"prefix":"/checkout","target_service":"checkout",
  "instance_weights":{"checkout-a":3,"checkout-b":1},"timeout_ms":180
}
```

```json
{
  "id":"r3","format_version":3,"prefix":"/orders","target_service":"orders",
  "instance_weights":{"orders-a":1,"orders-b":1},"timeout_ms":120,
  "retries":1,"circuit_threshold":2,"cooldown_ms":5000,
  "depends_on":["rule:r2","service:catalog"]
}
```

`depends_on` 使用 `rule:<id>` 或 `service:<id>`。未声明实例权重视为 0；权重不得为负；目标服务权重和必须大于 0。路由按最长前缀匹配，路径边界为 `prefix` 本身或 `prefix/...`。

请求文件：

```json
{"requests":[{"id":"req-1","method":"GET","path":"/catalog/ok"}]}
```

## 诊断命令

```bat
python -m meshdiag normalize --workspace samples\workspace.json --revision samples\revision-r1.json
python -m meshdiag normalize --workspace samples\workspace.json --revision samples\revision-r1.json --json
python -m meshdiag validate --workspace samples\workspace.json --revision samples\revision-r1.json
python -m meshdiag run --workspace samples\workspace.json --revision samples\revision-r1.json --requests samples\requests-cold.json --round-id manual-r1
python -m meshdiag snapshots --workspace samples\workspace.json
python -m meshdiag view --workspace samples\workspace.json
python -m meshdiag recover --workspace samples\workspace.json --snapshot snap-cold-r1
python -m meshdiag replay --workspace samples\workspace.json --snapshot snap-cold-r1 --revision samples\revision-r1.json
python -m meshdiag ui --workspace samples\workspace.json
```

手工运行前若已执行演示，可先删除 `var/`，或使用新的工作区副本。模拟上游可单独启动：

```bat
python mock_upstream.py --ports 18181,18182,18183,18184,18185,18186,18187,18188
```

## 核心实现

| 模块 | 职责 |
|---|---|
| `meshdiag/models.py` | 三格式解码、字段路径校验、归一模型、摘要、依赖图与确定性环检测 |
| `meshdiag/selection.py` | SHA-256 桶的显式加权选择，零权重不进入候选桶 |
| `meshdiag/scheduler.py` | 固定全局工作线程数 + 每服务配额；饱和服务排队，准入顺序确定 |
| `meshdiag/http_client.py` | 单次尝试真实 `socket.create_connection/recv`，超时由套接字产生 |
| `meshdiag/circuit.py` | 实例级连续失败计数、熔断、`monotonic_ns` 冷却与单探针 half-open |
| `meshdiag/ledger.py` | SQLite WAL 单事务轮次提交、当前物化视图整体替换、历史 `stale` |
| `meshdiag/snapshot.py` | 校验和 + 临时文件 + `os.replace` 的原子快照 |
| `meshdiag/engine.py` | 路由匹配、执行、检查点、恢复、版本演进与变更清单 |
| `meshdiag/replay.py` | 逐请求/逐尝试/账本重放比较与首个差异定位 |
| `meshdiag/ui.py` | Tkinter 进度、服务、依赖边、规则来源、熔断、结局与快照视图 |

### 归一与确定性

- 归一模型排除仅影响来源展示的 `format_version`；语义相同的 v1 默认等权与 v2 显式等权可得到相同摘要。
- 规范化 JSON 使用 `sort_keys=True, separators=(",", ":")` 和 UTF-8 SHA-256；规则、实例、节点、边、错误均稳定排序。
- 环检测只遍历规则依赖边，按规则 ID 排序 DFS；输出环旋转到字典序最小节点，例如 `a -> b -> c -> a`。
- 删除仍被引用的规则时，规范化前先检查原始新修订，`E_REF` 的 `referrers` 稳定列出引用者。

### 调度、重试与熔断

- 全局并发预算由固定工作线程数强制；每服务配额在准入条件中强制，配额满时请求保留在队列。
- 每次尝试都重新按确定性权重选择实例；重试序号写入 `X-Mesh-Attempt`，方法、路径与单次超时预算不变。
- 成功、HTTP 失败、套接字超时、熔断快速失败分别记录 `outcome`、`reason`、单调时钟起止毫秒和状态码。
- 熔断状态按实例保存；冷却结束只允许一个试探请求，其他请求在冷却/探针占用期间快速失败。
- 加权选择按实例 ID 排序后构造累积权重，桶号来自 `SHA-256(service_id, request_id, attempt_no)`；结果不依赖哈希遍历顺序，零权重实例不占桶。

### 账本与恢复

- 每轮结果在一个 `BEGIN IMMEDIATE ... COMMIT` 事务内写入修订、请求、尝试、结局和预算。
- `current_requests/current_attempts` 是当前物化视图；新一轮提交时先整体删除再写入。历史轮次保留审计但 `stale=1`，不进入当前视图。
- 同版本号但规则摘要或实例指纹变化会以 `E_VERSION` 拒绝；变更清单按规则 ID 和受影响服务稳定排序。
- 快照包含修订文档、归一摘要、依赖图版本、实例指纹、请求序列、已完成轨迹、进行中预算/配额、熔断状态和最终账本行。
- 快照写入先写 `.tmp`、`fsync`、再 `os.replace`；读取时验证 schema 与 SHA-256。未完整落盘的内容不会成为可恢复记录。
- 恢复时若快照修订/依赖图与当前修订或账本不一致，则以快照修订整体替换账本活动状态；运行中遗留请求标记 `unknown`、`rerun_required=1`。

### 重放比较

完成快照使用记录修订、初始熔断状态和当前本地实例集合重新分发同一序列，比较：

- 每请求 `outcome`、`rule_id`、`service_id`、`instance_sequence`、`attempt_count`；
- 每次尝试的实例、结局、原因、状态码；
- SQLite 账本请求行和尝试行的语义字段。

单调时钟、真实经过时间和预算毫秒天然不可跨轮次逐纳秒复现，因此作为明确列出的排除字段；其余字段不一致时 `E_REPLAY` 定位首个请求、规则和字段，并给出 expected/actual/basis。修订、依赖图或实例指纹不一致时在分发前拒绝并说明具体依据。

## 诊断码

| 代码 | 含义 |
|---|---|
| `E_RULE` | 未知格式版本、缺失/非法字段、负权重、零权重和等；整份修订拒绝并给字段路径 |
| `E_REF` | 目标服务、依赖规则/服务不存在，或删除仍被引用的规则；包含 `referrers` |
| `E_CYCLE` | 规则依赖成环；`cycle` 为稳定环序列 |
| `E_VERSION` | 修订版本回退，或同版本下规则摘要/实例集合变化 |
| `E_SNAPSHOT` | 快照 schema、校验和或内部摘要不一致 |
| `E_REPLAY` | 重放前置条件不可满足或逐项比较首个差异 |
| `E_LEDGER` | 账本轮次不存在等持久化错误 |

## 实际验证

验证环境为 Linux 容器（Python 3.11）模拟 Windows 可移植路径；所有网络仅访问 `127.0.0.1` 高位端口。

```text
python3 demo.py
... DEMO_OK
```

演示实际结果摘要：

```text
冷启动 r1: success=8 failure=0 timeout=2 circuit_open=0 unknown=0, max_global=4
热轮次 r2: success=8 failure=2 timeout=0 circuit_open=2 unknown=0
升级变更: changed=[catalog-route, checkout-route], affected_services=[catalog, checkout]
故障注入重放: requests=12 ledger_requests=12 ledger_attempts=16 -> REPLAY_MATCH
强制终止: CRASH_EXIT=96, completed=3 unknown=9 rerun_required=9
恢复完成快照: CONSISTENT, ledger_replaced=True, 重放 12 请求匹配
实例漂移负控: E_REPLAY field=instance_set
```

自动化测试：

```text
python3 -m unittest discover -s tests -v
Ran 30 tests in 9.097s
OK
```

覆盖内容包括三格式归一/摘要稳定、错误字段路径、依赖环与删除引用、权重确定性、真实套接字超时与重试头、全局/服务配额、熔断冷却与 half-open、版本演进整体替换、快照版本不一致替换、强制终止恢复、未知请求重跑、账本重放与首个差异定位。

当前验证容器没有 Tkinter 和图形显示，因此仅对 UI 做了 `py_compile` 语法验证，无法在此环境实际打开窗口；Windows Python 自带 Tkinter 时使用 `run_ui.bat` 启动。若缺少显示环境，程序会返回明确的 `E_TKINTER`/`E_TK_DISPLAY`，不会影响核心 CLI、账本与演示功能。
