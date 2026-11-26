# 租约仲裁首版技术说明

本交付在 `127.0.0.1` 回环 TCP 上用独立子进程模拟本地存储节点，提供声明强校验、JSON-line 协议、代数 CAS、状态机、线程池调度、三重配额、定点误差裁决、失效传播、SQLite 哈希账本和独立重放校验。

## 快速运行

```bash
bash scripts/demo.sh
```

等价的分步命令：

```bash
virtualenv -p python3 .venv
source .venv/bin/activate
python -m pip install 'numpy>=1.24' pytest
python -m pip install -e .

lease-arbitrator env-check
lease-arbitrator validate --spec samples/spec.json --events samples/events.ndjson
lease-arbitrator demo
lease-arbitrator verify --spec samples/spec.json --events samples/events.ndjson \
  --ledger ledger/demo.sqlite3
pytest -q
```

Tk 运行库不在系统标准路径时：

```bash
source scripts/tk-env.sh
lease-arbitrator ui --report reports/demo-report.json
```

`lease-arbitrator env-check` 也会自动尝试 `TK_ROOT`、`./.tk-runtime` 和 `~/.local/tkroot`。

## 代码结构

- `src/lease_arbitrator/spec.py`：JSON/NDJSON 加载、`E_SPEC` 首非法路径校验。
- `src/lease_arbitrator/node_server.py`：节点子进程、JSON-line TCP、节点侧代数 CAS。
- `src/lease_arbitrator/protocol.py`：真实 socket 连接、握手、读写超时。
- `src/lease_arbitrator/cluster.py`：子进程拉起、就绪探测、停止和日志。
- `src/lease_arbitrator/model.py`：状态集合、显式许可表、单调时钟迁移记录。
- `src/lease_arbitrator/domain.py`：实时调度与离线重放共用的容量/状态/裁决核心。
- `src/lease_arbitrator/engine.py`：`ThreadPoolExecutor`、条件变量排队、续约/释放/过期/撤销。
- `src/lease_arbitrator/fixedpoint.py`：整数定点水位、误差界和 NumPy 归因。
- `src/lease_arbitrator/invalidation.py`：节点、版本、租约和事件引用图及稳定传播。
- `src/lease_arbitrator/ledger.py`：SQLite 原子事务、游标、格式版本和哈希链。
- `src/lease_arbitrator/replay.py`：仅依赖账本与原始声明的独立重放和 `E_VERIFY` 定位。
- `src/lease_arbitrator/cli.py`：`validate/run/demo/verify/invalidate/water-report/ui/env-check`。

## 声明与 E_SPEC

样例声明：`samples/spec.json`；样例事件：`samples/events.ndjson`。

校验顺序覆盖：

- `node_id`、`lease_id` 唯一且非空。
- 节点 `host` 必须是字符串字面量 `127.0.0.1`，端口为 1–65535 且 `(host, port)` 全局唯一。
- `quota`、`max_active`、`global_budget` 为非负整数；布尔值不被当作整数。
- 租约 `units` 为正整数，`priority` 必须在 `priority_range` 内。
- `quota_versions[*].quota_version` 单调不减。
- 事件 `seq` 唯一且严格递增，事件引用的租约和版本必须存在。

错误输出形如：

```json
{"ok": false, "error": {"code": "E_SPEC", "path": "$.nodes[1].port"}}
```

CLI 在 `load_bundle()` 成功后才构造 `NodeCluster`，因此声明非法时不会启动任何节点子进程。

## TCP 与代数标签

每个节点由以下形式启动：

```bash
python -m lease_arbitrator.node_server \
  --node-id node-a --host 127.0.0.1 --port 21881 --seed 1701
```

协议为每行一个 JSON 对象。客户端先发送 `hello`，在真实 socket 超时内读取握手回执，再发送 `apply`、`renew`、`release` 或 `revoke`。

代数规则：

- 新租约 apply 从代数 0 推进到 1。
- 续约/释放/撤销请求必须携带节点当前公认标签 `g`。
- 匹配时节点把当前标签推进到 `g+1`，回执含 `generation_before=g`、`generation_after=g+1`、`generation_used=g`。
- 不匹配时返回 `E_STALE` 和 `current_generation`，节点代数不变。
- 仲裁层另用 `generation.py` 审计 `g -> g+1` 链；跨代跳变或回退链边报 `E_GEN`，归属不变。

样例中的 `seq=7` 故意在续约到 2 后仍携带旧标签 1，因此得到 `E_STALE`，随后携带 2 的请求仍可成功。

## 状态机

显式许可表在 `model.py`：

- `declared -> queued | revoked`
- `queued -> granted | revoked | failed`
- `granted -> renewed | released | expired | revoked`
- `renewed -> renewed | released | expired | revoked`
- `released/expired/revoked` 为终态。

每条 `Transition` 记录租约、迁移前状态、目标状态、原因、`time.monotonic_ns()` 换算的单调毫秒、事件序号、使用代数和前后代数。

## 三重约束调度

容量判定同时检查：

```text
global_used + units <= global_budget
node_used[node] + units <= node_quota[node]
node_active[node] < node_max_active[node]
```

apply 事件先在领域状态中登记为 `queued`，再提交到真实 `ThreadPoolExecutor`。工作线程在 `threading.Condition` 上等待；容量预占也计入 `global_used/node_used/node_active`，因此节点 TCP 调用在途时同样不会破坏约束。释放、过期或撤销后唤醒并重选。

等待记录使用单调时钟：

- `wait_start_ms`：apply 登记时间。
- `wait_end_ms`：成功预占容量时间。
- `wait_ms`：二者差，写入账本并进入报告。

样例中 `L-A-WAIT` 因 `node-a.max_active=1` 排队，直到 `L-A-LONG` 在 95ms 事件处释放；最终运行通常记录约 100ms 等待，测试以 `>= 60ms` 防止脆弱断言。

## 定点水位与误差归因

会话缩放因子 `Q = 10^scale`，`scale ∈ {0,1,2,3}`。

- 定点量：`q(u) = u * Q`。整数配额输入的直接量化误差为 0。
- 水位定点值：`level_fp = floor(used * Q / quota)`。
- 所有容量和裁决比较只用整数乘法、整除和取余；NumPy 仅用于误差归因占比。
- 截断方向：非负量向下取整。
- 溢出规则：兼容 signed-64 定点口径，上限 `2^63-1`；乘法/加法结果超过上限时饱和到该值并标记 `saturated=true`。

误差界：

- 单租约直接量化误差：0 个配额单位。
- 单租约派生水位的比率截断误差：`< 1 level_fp / Q`；换算配额单位为 `< quota/Q`。
- 两节点水位比较：两次 floor，保守界 `2 level_fp`。
- 全局聚合：`N` 个节点截断加一次全局 floor，保守界 `N+1 level_fp`。
- 候选节点水位与全局水位合计使用四次 floor，因此合成界为：

```text
B_ulp = 4 + scheduling_order_ulp + generation_reorder_ulp
B_units = B_ulp / Q
```

归因段为量化截断、调度次序和代数重排；NumPy 计算三段占比，任一段占比 `>= 0.60` 才声明为主导段，否则为 `mixed_below_threshold`。

边界判定：

```text
abs(remaining - demand) * Q <= B_ulp
abs(water_index_a - water_index_b) <= B_ulp
```

进入误差界的候选按 `(quota_version 降序, priority 降序, lease_id 升序)` 选择，并在 `decision.error_affected=true`、`error_bound` 和 `tie_rule` 中保留依据。样例里 `L-C-HIGH` 与 `L-C-LOW` 同节点、同需求、同定点水位，最终高优先级的 `L-C-HIGH` 入选。

可随时重新生成该时刻报告：

```bash
lease-arbitrator water-report \
  --spec samples/spec.json --events samples/events.ndjson \
  --at-ms 0 --out reports/water-adjudication.json
```

## 失效传播与整体替换

`invalidation.py` 建立 `node:*`、`lease:*`、`event:*`、`version:*` 节点和有向传播边，边依据包括：

- `node_membership`
- `quota_version_binding`
- `lease_reference`
- `generation_reference_reverse`
- `event_reference` / `event_lease_binding`

输出集合固定按 `(node_id, lease_id)` 排序，传播步骤也稳定排序。编辑样本：

```bash
lease-arbitrator invalidate --spec samples/spec.json \
  --events samples/events.ndjson \
  --edit samples/edit-delete-lease.json \
  --out reports/invalidation-delete.json
```

删除 `L-C-LOW` 会稳定得到 `['L-C-LOW', 'L-DEPENDS']`；删除 `node-c` 会得到该节点上的全部租约。

配额升级时：

1. 在同一把仲裁条件锁内生成旧版本撤销集并冻结新预留。
2. 对旧版本仍活跃租约向节点发起真实 `revoke` CAS。
3. 取消旧版本排队项并释放预占。
4. 整体替换 global budget、node quota 和 max active，不与旧策略合并。
5. 旧版本归属、水位和轨迹进入 `stale_view`，当前 `current_view` 只展示当前版本。

## SQLite 账本与独立重放

账本路径：`ledger/demo.sqlite3`。表：

- `meta`：spec/events 摘要、整数 `format_version=1`。
- `cursors`：全局进度游标。
- `entries`：事件类型、ok/错误码、租约/节点、前后状态、三个代数字段、全局/节点余量、等待起止、游标、格式版本、内容摘要、前摘要和本行哈希。

`Ledger.append()` 在 `BEGIN IMMEDIATE` 事务中同时写 `entries` 与推进 `cursors`，并提交哈希链。

独立校验：

```bash
lease-arbitrator verify \
  --spec samples/spec.json \
  --events samples/events.ndjson \
  --ledger ledger/demo.sqlite3
```

`replay.py` 不读取报告，也不信任账本中的状态结论；它只读取原始声明/事件和 SQLite 账本，用确定性领域核心与内存 `NodeCore` 重推申请、排队、授予、续约、过期/回收、撤销和版本替换，再按租约逐项比较状态、代数和配额余量。首个不一致以 `E_VERIFY` 给出 `lease_id`、`field`、`generation` 和路径。哈希链损坏也统一包装为 `E_VERIFY`。

## 实际验证记录（2026-11-27）

- 虚拟环境：`.venv`，Python `3.11.0rc1`（满足 >=3.10）。
- SQLite：标准库 `sqlite3`，引擎版本 `3.37.2`。
- NumPy：`2.4.6`（满足 >=1.24）；pytest：`9.2.0`。
- Tkinter：`8.6`；系统未配置 `python3-tk`/`libtk8.6.so`，本次用本机已有的 `/home/zsl/.local/tkroot` 用户态运行库验证，`Tk()` 根窗口创建、`update()` 和销毁成功。`scripts/tk-env.sh` 与 `env-check` 可自动启用。
- `pytest -q`：`21` 个测试全部通过；连续重复 5 轮均通过。
- `bash scripts/demo.sh`：退出码 0；真实启动 3 个 `127.0.0.1` 子进程并完成演示。
- 演示账本：`15` 条 entries，进度游标 `15`；独立 `verify` 返回 `"ok": true`。
- 最终一次样本复核：`L-A-LONG=released/gen4`，`L-A-WAIT=expired/gen2/wait_ms=105`（多次运行约 100–110ms），`L-B-HOLD=revoked/gen2`，`L-C-HIGH=revoked/gen2`，`L-C-LOW=revoked/gen0(排队取消)`，`L-V2-NEW=released/gen2`。
- 运行期三重约束违例：`[]`。
- 删除传播样本：`L-C-LOW -> L-DEPENDS`；节点删除传播：`L-C-HIGH, L-C-LOW, L-DEPENDS`。

## 后续可集成点

- CI 若以无显示用户运行，Tk 测试会在 `Tk()` 失败时自动 skip；核心包不依赖显示。
- 若目标机可使用系统包，执行 `sudo apt-get install python3-tk` 后可直接获得系统 Tkinter；当前无免密 sudo，因此使用已验证的用户态 Tk root。
- 生产化时可把 `NodeCluster` 的日志接入集中日志，并为 SQLite 增加定期 checkpoint/备份；当前 WAL 文件与主库均在 `ledger/`。
