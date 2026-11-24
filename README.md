# 跨节点分片下发配额账本组件

面向 Windows 本地环境、无容器依赖的可重复演示。协调进程在 `127.0.0.1` 上启动多个独立 Python 子进程作为计算节点，通过真实 TCP JSON 行协议下发分片批次；节点使用 NumPy 的 `int64` 做整数定点分组聚合。协调器负责依赖就绪、三重配额、租约唯一归属、版本化配置失效传播、SQLite 提交账本、强杀恢复与逐字段重放校验。

## 交付内容

- `quota_ledger/`：可复用实现。
- `demo.py`：演示入口，支持样本路径、工作目录、Tk 诊断界面开关和异常注入。
- `generate_sample.py`：确定性样本生成器，不下载外部数据。
- `samples/demo/`：已由生成器生成的本地样本（3 节点、10 分片、修订链、事件流）。
- `tests/`：真实套接字、租约、校验、调度、失效、强杀恢复、异常注入测试。
- `README.md`：运行、设计、验证记录。

## 环境准备（Windows）

要求 Python 3.11+、pip、NumPy；`sqlite3` 和 Tkinter 随标准库提供。

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

端口占用检查（高位端口，无需管理员权限）：

```powershell
Test-NetConnection 127.0.0.1 -Port 41125
```

若默认端口被占用，重新生成样本时使用另一段空闲端口：

```powershell
.\.venv\Scripts\python.exe demo.py generate --out samples\demo --seed 20261125 --base-port 45125
```

## 快速演示

```powershell
# 完整运行；打开只读 Tk 诊断界面
.\.venv\Scripts\python.exe demo.py --sample samples\demo --workdir work\demo --gui

# 无界面运行
.\.venv\Scripts\python.exe demo.py --sample samples\demo --workdir work\demo
```

Linux/macOS 下把解释器换成 `python3` 即可；本轮自动化验证在 Linux 容器中完成，`--gui` 需要系统 Tcl/Tk 动态库。

## 输入文档

样本目录包含四份强校验文档：

- `nodes.json`：`global_budget`、`default_lease_ttl_ms` 和节点声明。
- `shards.json`：分片 ID、绑定节点、依赖、资源单位、样本数、种子偏移。
- `revisions.json`：严格递增的修订版本、父修订、全局预算、聚合参数和节点容量配置。
- `events.json`：严格递增到达序号的分片下发、取消、修订和配额事件。

节点声明示例：

```json
{
  "id": "node-b",
  "host": "127.0.0.1",
  "port": 41126,
  "concurrency": 2,
  "lease_limit": 1,
  "seed": 20261327,
  "capacity_version": 1
}
```

加载时会一次性验证全部文档，且在启动任何节点子进程之前完成。非法输入抛出 `BusinessError`，包含业务码、位置、字段和首个非法条目：

- `E_SPEC`：JSON/字段类型/唯一性/端口/正整数/修订链/事件序号等。
- `E_GRAPH`：依赖引用不存在、自依赖、重复依赖或依赖环。
- 环检测输出稳定路径，例如 `s01 -> s10 -> s01`。

## 核心设计

### 节点协议与真实网络错误

协调器用 `subprocess.Popen([sys.executable, "-m", "quota_ledger.node_server", ...])` 启动独立节点。Windows 下设置 `CREATE_NO_WINDOW`；子节点监控父 PID，父进程被强杀后自行退出，避免占用端口。

协议是每行一个 JSON 对象，核心操作：

- `ping`：握手。
- `configure`：在配置屏障后更新节点侧并发/租约上限与容量版本。
- `lease_acquire` / `lease_renew` / `lease_release`：节点侧租约。
- `compute`：携带分片 ID、配置版本、租约版本、批次样本、定点比例与分组数。
- `snapshot`：诊断快照。

`compute` 回执包含节点 ID、分片 ID、配置版本、聚合摘要和单调时钟毫秒耗时。错误区分为：

- `E_PORT`：启动前端口探测失败、端口已占用或节点进程绑定失败。
- `E_TIMEOUT`：`socket.settimeout(...)` 触发的真实套接字超时；没有人为构造的业务延迟。
- `E_REPLY`：回执缺字段、身份/版本不匹配、摘要结构非法。

### 租约

每个租约包含 `lease_id`、`shard_id`、`node_id`、授予/到期单调毫秒、版本号和配置版本。规则：

- 同一分片最多一个活跃租约；节点侧也再次拒绝重复分片租约。
- 续约必须携带期望版本；版本不符返回 `E_LEASE`，不延长到期时刻。
- 过期扫描使用 `time.perf_counter_ns()` 派生的单调毫秒，不依赖墙钟。
- 续约和过期在同一把锁下竞争，状态只允许一次从 `active` 迁移，因而不会同时计入“续约成功”和“到期回收”。

### 三重配额与线程池

调度器同时检查：

1. 全局在途资源单位 `sum(units) <= global_budget`。
2. 单节点活跃分片数 `<= concurrency`。
3. 单节点活跃租约数 `<= lease_limit`。

所有分片按稳定 ID 排序后尝试准入；配额不足时进入排队并记录真实单调时钟的 `wait_start_ms` / `wait_end_ms`。准入后由 `ThreadPoolExecutor` 工作线程执行真实套接字调用。SQLite 的 `reserve_dispatch` 会在事务内再次校验三项约束，避免调度器或事件更新造成超限。

### 依赖、取消与事件屏障

分片只有在全部依赖状态为 `completed` 且摘要可用时才进入就绪集合。控制流采用配置屏障，避免完成时序竞争：

- 同一屏障内先应用所有下发/取消事件，再并行执行。
- 取消在下发执行前生效，持久化为 `cancelled_shards`，之后不得重试。
- 修订/配额事件作为屏障；先排空上一批，再应用新配置并重算失效集合。
- 就绪、排队、取消依据和失效集合均按分片 ID 或稳定拓扑序输出，不依赖字典遍历顺序。

### 失效传播与增量重算

修订事件比较父修订的实际生效配置；配额事件比较节点容量变化。直接失效来源包括：

- 修订全局预算或聚合参数。
- 修订节点的并发/租约/容量版本。
- 配额事件改变的节点。
- 配额事件改变的全局预算。

随后沿依赖图向后代传播，记录 `direct`、`reason` 和 `topo_rank`。例：

```text
revision:2:node:node-b
dependency:s02->s06
quota:13:node:node-c
```

仅失效集合会被重新请求；未失效分片的结果行不会被更新。每个失效事件在同一事务内记录未失效分片的结果摘要前后值，运行中即时校验；差异会以 `E_REPLAY` 报告首个分片与字段。`demo.py verify` 可比较增量结果与空白全量重建账本。

### 账本、恢复与摘要

SQLite 账本格式版本为整数 `1`。每完成一个分片，在同一事务中写入：

- 分片结局与聚合摘要。
- 节点、租约 ID、租约版本、配置版本、资源单位。
- 进度游标与完成快照。
- 内容摘要更新。

内容摘要覆盖业务字段：结果、租约归属/状态、节点配额配置、全局预算、失效证据、事件日志、游标和输入摘要。它刻意排除单调时钟、墙钟、线程名、尝试次数和运行耗时，因此同一输入多次运行的“账本摘要”可逐字节一致；上述运行数据仍完整保存在账本/报告但不参与确定性摘要。

强杀语义：

- 分片完成以前只存在已提交的派发尝试；重启把 `in_flight` 尝试标为 `unknown`，把未完成活跃租约标为 `unknown`，重置运行时占用，并要求重跑。
- 强杀路径使用 `os._exit(70)`，不会写退出报告，也不声称拥有退出报告。
- 可处理的 `SIGINT`/`SIGTERM` 或 `--graceful-after-commits` 会停止新派发、排空运行任务、写入完整退出报告并设置 `clean_shutdown=1`。
- 恢复完成后可按分片逐字段比较结局、摘要、租约归属、配置版本、资源单位，以及节点/全局配额占用；首个差异以 `E_REPLAY` 指明分片与字段。

## 命令

```text
python demo.py run --sample SAMPLES --workdir WORK [--gui] [--inject KIND]
python demo.py verify --baseline FULL_WORKDIR --candidate CANDIDATE_WORKDIR
python demo.py snapshot-check --workdir WORK [--inject]
python demo.py inject --kind corrupt-nodes|cycle|snapshot --sample SAMPLES --workdir WORK
python demo.py generate --out SAMPLES [--seed N] [--base-port N]
```

`demo.py` 也接受省略 `run` 的简写：

```powershell
.\.venv\Scripts\python.exe demo.py --sample samples\demo --workdir work\demo --gui
```

## 可重复验收步骤

以下命令在仓库根目录执行。强杀与恢复：

```bash
python3 demo.py run --sample samples/demo --workdir work/full
python3 demo.py run --sample samples/demo --workdir work/recovered --crash-after-commits 2
sleep 1
python3 demo.py run --sample samples/demo --workdir work/recovered
python3 demo.py verify --baseline work/full --candidate work/recovered
```

三类异常注入：

```bash
python3 demo.py inject --kind corrupt-nodes --sample samples/demo --workdir work/inject-spec
python3 demo.py inject --kind cycle --sample samples/demo --workdir work/inject-graph
python3 demo.py snapshot-check --workdir work/full --inject
```

每次注入均返回结构化 JSON，并追加到对应工作目录的 `structured.jsonl`；记录含 `monotonic_ms`、`shard_id`、`code`、`position`、`field`。`inject` 命令把预期注入视为演示成功并返回 `0`，但输出中的 `code` 分别是 `E_SPEC`、`E_GRAPH`、`E_SNAPSHOT`。

## 实际验证记录

验证日期：2026-11-25。环境：Python 3.11.0rc1、NumPy 1.26.4、SQLite 3.37.2。

```text
python3 -m unittest discover -v
Ran 30 tests in 12.768s
OK
```

覆盖结果：

- 真实 TCP 超时（含节点握手超时）、端口占用、缺失回执字段。
- 节点子进程租约版本乐观校验；本地真实过期只回收一次；取消已占配额只释放一次。
- 依赖环/缺失引用/重复 ID/端口/事件序号/修订父链校验。
- 同输入两次运行账本摘要完全一致。
- 三重配额峰值：全局资源 `4/4`；节点活跃分片峰值 `node-a=1,node-b=1,node-c=2`；节点租约峰值 `node-a=1,node-b=1,node-c=2`；发生真实排队等待记录。
- 修订与配额两次失效按拓扑序输出；未失效分片增量检查全部 `unchanged=1`。
- 强制终止后退出报告为空；恢复报告标出未知分片并重跑；恢复账本与同输入不中断账本逐字段比较一致。
- 可处理退出写出包含未完成分片的完整退出报告，随后可恢复至同一摘要。
- 三类注入分别输出 `E_SPEC`、`E_GRAPH`、`E_SNAPSHOT` 并落结构化日志。

内置样本在默认端口 `41125-41127` 下的一次运行：

```text
source_digest  = 9f42d96aea9b48f1881006d4f23ab48c32c9eb194f175616e4de0e0cebb9ed60
ledger_digest  = 0fdbf8247ec41774c3dd6f4114c41bd526380618987f0d080e12e80ff94d0477
completed       = 9
cancelled       = s10
snapshots       = 19
```

同输入第二次运行得到同一 `ledger_digest`。以 `--crash-after-commits 2` 强杀后，账本无退出报告；恢复时未知分片为 `s03`，恢复完成后的摘要是同一 `0fdbf8...d0477`，`verify` 比较 9 个分片全部一致。

本验证镜像缺少 `libtk8.6.so`，因此无法在该 Linux 容器中实际弹出 Tk 窗口；Tk 代码已作为可选运行入口提供，Windows 标准库 Tkinter 可直接使用。诊断界面所需的节点/租约/资源/排队/失效/快照报告数据已由实际运行生成并验证；本次仅未执行窗口渲染层。

## 已知边界

- 仅允许 `127.0.0.1` 字面量，不解析 `localhost` 或 IPv6。
- 样本默认端口需空闲；冲突会明确返回 `E_PORT`，可用 `--base-port` 重新生成。
- 确定性摘要排除运行时钟/线程等易变字段，以支持同输入逐字节比较；完整时序仍保留在账本行、报告和 JSONL 日志中。
- 不使用 Docker、容器编排、外部下载或未授权网络目标。
