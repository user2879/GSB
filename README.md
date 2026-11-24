# CacheLease 多节点缓存租约与容量水印诊断器

这是一个可独立运行的 Python 3.11+ 本地组件。它把每个缓存节点实现为独立子进程，在
`127.0.0.1` 高位 TCP 端口上使用 JSON Lines 协议通信；控制器负责确定性内容摘要、跨节点租约仲裁、
预占与宽限回滚、配额版本演进、并发调度、SQLite 账本、崩溃恢复，以及增量/全量重建比对。

组件不使用 Docker、云服务、外部账号或网络数据。所有样本均由固定种子的生成器在本地创建；只有
NumPy 是第三方依赖，SQLite 使用标准库。

## 快速开始（Windows）

```powershell
python -m pip install --upgrade pip
python -m pip install numpy
python -m pip install -e .

python -m cachelease generate --out samples/sample --seed 243
python -m cachelease validate --config samples/sample/sample.config.json --events samples/sample/events.ndjson
python -m cachelease port-check --config samples/sample/sample.config.json

python -m cachelease run `
  --config samples/sample/sample.config.json `
  --events samples/sample/events.ndjson `
  --ledger reports/sample.sqlite `
  --work-dir reports/sample-nodes `
  --report reports/sample-run.json
```

Linux/macOS 下把续行符 `` ` `` 换成 `\` 即可。`port-check` 必须显示所有声明端口可用；如果端口被占用，
修改配置中的 `port` 后重新生成或编辑端口即可。

## 交付内容

- `cachelease/`：协议、子进程节点、仲裁、调度、SQLite 账本、恢复、投影比对和 CLI。
- `samples/sample/`：固定种子样本的节点声明、事件流、前缀事件和内容摘要清单。
- `tests/`：22 个自动化测试，覆盖规范、内容摘要、真实节点、排队、账本、恢复和增量/全量一致性。
- `scripts/acceptance.py`：一键执行样本全量、增量校验、强制终止恢复和测试，并写出验收摘要。
- `reports/validation.json`：最近一次实际验收结果。

## 声明格式

节点、流水线阶段和制品清单位于一个 JSON 文件中：

```json
{
  "format_version": 1,
  "session": {"scale": 1000},
  "runtime": {
    "global_concurrency": 4,
    "handshake_timeout_ms": 4000,
    "reply_timeout_ms": 4000,
    "grace_ms": 160
  },
  "nodes": [
    {
      "node_id": "node-a",
      "host": "127.0.0.1",
      "port": 43001,
      "capacity_slots": 5,
      "lease_limit": 6,
      "seed": 101,
      "quota_version": 1,
      "request_concurrency": 2
    }
  ],
  "pipeline": {
    "stages": [{"stage": "build", "concurrency": 3}],
    "artifacts": [
      {"key": "core", "stage": "build", "refs": [], "shards": {"count": 3, "bytes": 96}},
      {"key": "api", "stage": "build", "refs": ["core"], "shards": {"count": 2, "bytes": 64}}
    ]
  }
}
```

约束包括：节点标识唯一、端口全局唯一且为 `1024..65535`、`host` 必须是字面量
`127.0.0.1`、容量/租约/配额版本/并发数均为正整数、制品引用存在且无环、阶段存在。
JSON 重复键也会被拒绝。

事件文件是 NDJSON，每行一个事件。`seq` 单调不减，`batch` 连续且单调；`release`/`quota`
批次必须只含一个事件，`write`/`read` 批次可以并行执行。

```jsonl
{"seq":1,"batch":1,"type":"write","artifact":"core","lease_ms":60000}
{"seq":2,"batch":2,"type":"read","artifact":"core","lease_ms":60000}
{"seq":3,"batch":3,"type":"release","artifact":"core"}
{"seq":4,"batch":4,"type":"quota","node":"node-a","quota_version":2,"capacity_slots":5,"lease_limit":6,"policy":"renew"}
```

`write` 可带诊断用的 `hold_ms`：控制器真实等待该毫秒数后再提交。如果超过 `grace_ms`，节点会先由
单调时钟自动回滚，随后提交返回 `E_EXPIRED`，用于验证“宽限到期释放容量”。

任何规范错误均输出 `E_SPEC`，并携带第一个违规的 `location`。

## 确定性内容摘要

节点只根据声明的 `(node_id, artifact_key, seed)` 生成分片。每个分片的 PCG64 种子是：

```text
sha256("CACHELEASE-CONTENT-SEED-v1" || lp(node_id) || lp(artifact_key) || lp(decimal(seed)))
```

`lp(x)` 是 4 字节大端长度前缀加 UTF-8 字节。分片按整数序号升序生成。分片摘要规则为：

```text
shard_i = sha256(
  "CACHELEASE-ARTIFACT-SHARD-v1" || lp(node_id) || lp(artifact_key)
  || u32(i) || u32(len(bytes_i)) || bytes_i
)

artifact_digest = sha256(
  "CACHELEASE-ARTIFACT-MANIFEST-v1" || lp(node_id) || lp(artifact_key)
  || order_i(u32(i) || shard_i)
)
```

所有拼接都按声明的整数序号进行，不依赖字典、集合或文件加载顺序。相同声明在不同进程、不同加载顺序
与不同运行中产生逐位一致的摘要。

## 租约、预占和仲裁

每个节点在锁内维护一个确定性状态机：

1. `reserve` 先从整数容量槽位中扣减一个临时槽，并从每节点租约余量中预留一个租约许可；此时不写入内容。
2. `commit` 在宽限内把临时槽转为内容条目与活跃租约，或合并到已有内容后释放临时槽。
3. `rollback` 显式释放临时槽；后台回收线程也会按节点本地单调时钟在宽限到期时自动执行回滚。
4. 租约到期由本地单调时钟驱逐；最后一个租约离开后释放内容槽。

控制器对每个写/读事件生成完整候选集，排序规则是：

```text
(内容摘要匹配优先, 剩余容量槽位降序, node_id 字典序)
```

内容摘要匹配表示节点已持有该节点本地确定性摘要。其余候选保留在 `arbitrations.candidates_json`，
并记录 `selected`、`digest_match`、`free_slots`、`lease_remaining`、并发原因与否决理由
（如 `lower_priority`、`no_free_slot`、`lease_limit`、`node_concurrency_limit`、`digest_missing`）。
每个事件最多向一个节点发出成功授予，同一制品键在同一批次内由制品锁串行决策。

## 容量水印与配额版本

对每个事件，账本为每个节点写入一行水印：

```text
used_slots = 内容条目数 + 尚未提交的预占槽
free_slots = capacity_slots - used_slots
active_leases = 当前活跃租约数
lease_remaining = lease_limit - active_leases - 活跃预占

watermark_units = round_half_up(used_slots * scale / capacity_slots)
quantization_error_upper = ceil(scale / (2 * capacity_slots))
```

`quantization_error_upper` 是四舍五入定点表示的保守上界。批次的所有事件共享批次结束时的账本水位；
每个节点自己的协议回执同时包含操作当时的即时水位。

`quota` 事件必须递增版本。节点先处置旧版本预占，再按显式 `policy` 处理旧租约：

- `renew`：逐笔续接并生成新版本租约 ID；若容量或租约上限缩小，超出部分按确定顺序撤销并记录原因。
- `revoke`：逐笔撤销旧版本租约。

账本会把旧版本的租约、预占和水印整体置为 `stale=1`；当前统计与投影只读取新版本。每一笔续接/撤销
都写入 `leases` 或 `reservations`，带 `parent_lease_id` 和处置依据。

## 真实并发调度与排队

请求通过本地 `ThreadPoolExecutor` 执行，并使用条件变量同时约束：

- `runtime.global_concurrency` 全局在途上限；
- 每个节点的 `request_concurrency`；
- 制品所属阶段的 `pipeline.stages[].concurrency`。

配额不可用时线程在条件变量上等待，不使用睡眠模拟排队。许可获得前后分别读取单调时钟，写入
`queue_observations.wait_start_ms`、`wait_end_ms`、`wait_ms`，并记录当时的全局/节点/阶段在途数。
因此 `wait_ms` 与账本记录逐项相等。

## 账本、强制终止与恢复

SQLite 账本包含整数 `format_version`、配置摘要、内容摘要、滚动账本摘要和完整业务记录。每个批次分两步：

1. 独立事务写入 `batches.status='open'`，作为“本批次已经开始”的耐久标记。
2. 所有网络操作完成后，用单个事务写入事件、租约授予/续约/释放、预占/回滚、容量变更、逐节点水印、
   排队观测、状态快照、滚动摘要和 `progress`，并把批次改为 `committed`。

进程在步骤 2 之前被强制终止时，`progress` 不前进，开批次在恢复时标记为 `unknown`，并写入
`UNKNOWN_UNCOMMITTED_BATCH` 恢复动作；节点中以 `apply_seq` 超过已提交游标的租约/预占会被
`reconcile` 撤销并记录为未知。恢复只从最后一个完整提交的批次快照装载节点状态，然后重放账本校验：

- 每个事件的存储摘要与事件负载一致；
- 每个节点每个事件都有水印记录，水印观察值与摘要未变；
- 批次滚动摘要链和 `progress.ledger_digest` 一致；
- 快照中的水印、预占账和内容摘要与当前版本记录一致。

任何不一致以 `E_REPLAY` 定位到第一个可识别事件序号。恢复后从 `progress` 的下一个事件继续。

```powershell
python -m cachelease recover --config ... --events ... --ledger reports/sample.sqlite
python -m cachelease run --resume --config ... --events ... --ledger reports/sample.sqlite
```

## 增量与全量重建校验

`verify` 在检查点执行三条路径：

1. 从空白开始处理检查点前缀，建立基础账本；
2. 复制基础账本，恢复节点快照，只推进检查点之后的事件；
3. 从空白开始处理完整事件流。

随后比较检查点前缀快照字节、进度，以及逐节点、逐制品的租约归属、租约 ID、内容摘要、配额版本、
已用/空闲槽、活跃租约、租约余量、定点水印比例与量化误差上界。第一处差异以：

```text
E_DIVERGE node_id / artifact_key / field
```

报告。增量路径不修改任何已提交前缀快照，测试会逐字节验证。

```powershell
python -m cachelease verify `
  --config samples/sample/sample.config.json `
  --events samples/sample/events.ndjson `
  --checkpoint-seq 15 `
  --work-dir reports/verify `
  --report reports/verify.json
```

检查点必须落在批次边界；检查点前缀必须与已有账本一致。对检查点之后的任意尾部编辑，增量路径只重算
后缀批次，前缀状态保持不变。

## CLI

```text
python -m cachelease generate --out samples/sample --seed 243
python -m cachelease validate --config CFG --events EVENTS [--report R]
python -m cachelease port-check --config CFG [--report R]
python -m cachelease run --config CFG --events EVENTS --ledger DB
                           [--work-dir DIR] [--until-seq N] [--resume] [--report R]
python -m cachelease recover --config CFG --events EVENTS --ledger DB
                             [--work-dir DIR] [--report R]
python -m cachelease verify --config CFG --events EVENTS --checkpoint-seq N
                            [--work-dir DIR] [--report R]
```

错误码：`E_SPEC`、`E_HANDSHAKE`、`E_REPLY`、`E_DIVERGE`、`E_REPLAY`。节点业务拒绝（如
`E_CAPACITY`、`E_LEASE_LIMIT`、`E_EXPIRED`）记录在事件结果中，不会伪造成功。

## 验证

```powershell
python -m pytest -q
python scripts/acceptance.py
```

`acceptance.py` 会实际启动回环端口和子进程，执行全量样本、增量/全量校验、强制终止恢复与续跑，
最后再次运行测试，把结果写入 `reports/validation.json`。最近一次实际结果见该文件；摘要如下：

本次实跑环境为 Python 3.11 / NumPy 1.26 / Linux（当前任务容器）；Windows 专用路径包括
`CREATE_NO_WINDOW` 子进程标志和 `ctypes` 父进程存活检测。监听、客户端、SQLite 和测试逻辑均为跨平台实现。

```text
样本: 3 节点 / 4 制品 / 25 事件 / 检查点 seq=15
全量: progress=(ordinal=25, seq=25), replay=25 事件/75 条节点水印
内容账本摘要: 441d681d2c24688c68ff19464635152910b1f4779dad13715e9f2adc5bcf20ff
重复制品: seq1 与 seq5 均授予 node-a；seq5 摘要匹配优先，其余节点记录 lower_priority
预占回滚: seq14 返回 E_EXPIRED，账本保留 1 条 rolled_back 预占，未遗留容量槽
配额演进: node-a/node-b 到版本 2；20 条旧版本水印被标记 stale 并从当前统计排除
排队: 20 条真实观测；每条 wait_ms == wait_end_ms - wait_start_ms
增量: 前缀 15 事件、仅推进后缀 10 事件；前缀快照逐字节一致；投影与全量一致
恢复: crash exit=137；batch 5 标记 unknown；恢复后重放 7 个已提交事件，续跑至 seq 25；
      恢复投影与全新全量投影逐字段一致
测试: 22 passed
```

## 说明

- Tkinter 未被使用，因此当前 Linux 验证环境缺少 Tk 系统库不影响组件；Windows 官方 Python 可直接运行。
- 组件不尝试连接声明之外的地址，不包含容器化配置。
- 端口由声明文件决定；`port-check` 只探测这些显式本地回环端口。
- 长租约跨进程恢复时使用账本快照中的剩余租约毫秒数；到期仍由恢复后节点的本地单调时钟重新计时。
