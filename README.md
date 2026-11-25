# 边缘缓存内容分片租约调度器（首版）

本工作区已从空目录落地一套可运行的首版实现：每个缓存节点是独立 Python 子进程，在 `127.0.0.1` 上通过 TCP JSON Lines 接收分片批次；调度器用工作线程池并行连接节点，使用 NumPy 按节点/内容声明的整数种子生成并定点校验分片；租约、续约、过期、配额替换和崩溃恢复全部写入本地 SQLite 账本，并提供独立只读重放器和 Tkinter 诊断界面。

首版针对两个核心问题给出的答案是：

- **同一容量槽位不会双重归属**：每个租约只能产生一次 claim（SQLite 主键约束），配额事件先做单调时钟到期裁决，再撤销旧版本租约；活跃槽位另有部分唯一索引。每次归属变化记录 `claim_kind`、`basis`、`event_seq` 和 `quota_version`。
- **强杀后可恢复出一致结论**：租约在发送网络批次之前先以 `OPEN` 批次和 `RESERVED` 租约原子入库。子进程被强杀时，未提交批次整体标为 `UNKNOWN`、释放槽位和字节预算；重跑使用新的 attempt。实测重跑后的节点/分片/槽位/摘要与无故障全量运行完全一致，且独立重放通过。

## 环境与快速验证

要求 Python 3.11+、NumPy；SQLite 与 Tkinter 来自标准库。Windows PowerShell 示例：

```powershell
python --version
python -m pip install numpy

# 检查演示端口是否空闲（任一有输出即被占用，请修改 sample 中对应端口）
Get-NetTCPConnection -LocalPort 19183,19184,19185,19186 -ErrorAction SilentlyContinue

python -m edgecache spec-check --sample samples\demo.json
python -m edgecache run --sample samples\demo.json --db runs\demo.sqlite
python -m edgecache replay --sample samples\demo.json --db runs\demo.sqlite
python -m edgecache ui --db runs\demo.sqlite
```

Linux/WSL 只需把路径分隔符换为 `/`。也可以安装为命令：`python -m pip install -e .`。

完整强杀恢复演示：

```powershell
python -m edgecache run --sample samples\crash.json --db runs\crash-clean.sqlite
python -m edgecache run --sample samples\crash.json --db runs\crash-injected.sqlite `
  --crash-node edge-a --crash-after-grants 2
# 第一步按设计返回 3/PARTIAL；重新打开同一账本并要求重跑
python -m edgecache run --sample samples\crash.json --db runs\crash-injected.sqlite --retry-unknown
python -m edgecache replay --sample samples\crash.json --db runs\crash-injected.sqlite
```

命令成功返回 `0`；`E_SPEC/E_LEASE/E_PORT/E_TIMEOUT/E_REPLAY` 等结构化错误返回 `2`；仍有队列未排空时运行返回 `3`。所有错误均以 JSON 输出。

## 目录

```text
edgecache/
  spec.py        声明加载、强校验、首个非法行/字段定位
  content.py     确定性分片生成、SHA-256 摘要、NumPy 定点校验
  protocol.py    回环 TCP JSON Lines 客户端
  worker.py      独立节点子进程及 JSON Lines 服务
  scheduler.py   事件调度、线程池、租约/队列/过期/配额/崩溃重跑
  ledger.py      SQLite 原子账本、checkpoint、快照替换、未知恢复
  replay.py      仅读账本与原始声明的独立重放校验
  dashboard.py   稳定只读快照与 Tkinter 界面
  cli.py         spec-check/run/replay/ui/dashboard
samples/
  demo.json                 到期、续约、队列、配额与节点集合演进
  crash.json                强杀未提交批次与重跑
  invalid/duplicate_port.jsonl
tests/                      20 项单元/集成/恢复/重放测试
runs/                       运行账本与验证输出（不提交）
```

## 声明格式

标准 JSON 或 JSONL 均可。标准 JSON 顶层为 `format_version/config/nodes/contents/events`；JSONL 每行以 `kind` 标记 `config/node/content/event`，可获得逐行精确错误定位。

```json
{
  "format_version": 1,
  "config": {
    "sample_scale": 64,
    "lease_ttl_ms": 450,
    "global_lease_limit": 3,
    "global_byte_budget": 32768,
    "socket_timeout_ms": 3000,
    "handshake_timeout_ms": 4000,
    "max_workers": 8,
    "drain_timeout_ms": 3000
  },
  "nodes": [{
    "node_id": "edge-a", "host": "127.0.0.1", "port": 19183,
    "slots": 1, "lease_limit": 1, "seed": 1101, "quota_version": 1
  }],
  "contents": [{
    "content_key": "clip-001", "total_bytes": 16384,
    "shard_count": 4, "content_seed": 9001
  }],
  "events": [
    {"seq": 1, "type": "apply", "node_id": "edge-a", "content_key": "clip-001", "shards": [0]},
    {"seq": 2, "type": "renew", "lease_id": "L000001-S0000", "expected_version": 1, "extend_ms": 1800},
    {"seq": 3, "type": "release", "lease_id": "L000001-S0000"},
    {"seq": 4, "type": "quota", "quota_version": 2, "node_ids": ["edge-a"]}
  ]
}
```

强校验规则包括：节点标识/内容键唯一；`host` 必须是字面量 `127.0.0.1`；端口全局唯一且在 `1..65535`；容量槽位、每节点租约上限、全局限制、TTL 为正整数；`shard_count` 整除 `total_bytes`；种子为整数（布尔值不算整数）；事件序号从 1 连续；事件类型仅限 `apply/renew/release/quota`；申请引用当前配额视图中的节点及合法分片；配额版本严格递增。校验先收集问题，再按源文件行号选择首个错误，返回 `E_SPEC`、`line`、`field`、`path`。

`delay_ms` 是可选的本地事件投递节奏，仅用于样本中演示真实到期，不参与调度规则。

## 分片字节与摘要规则

对节点 `n`、内容 `c`、分片 `s`：

1. 规范化 JSON 头为 `{node_id, node_seed, content_key, content_seed, shard_index}`（键排序、紧凑编码）。
2. `seed64 = LE_U64(SHA256("edgecache-shard-v1\0seed\0" || header)[0:8])`。
3. 使用 `numpy.random.Generator(numpy.random.PCG64(seed64))` 生成 `sample_count = ceil(shard_bytes/4)` 个 `[-32768, 32767]` 整数样本。
4. 以 `(raw * sample_scale + 16384) // 32768` 做整数定点量化，写入 little-endian `int32`，截断到 `shard_bytes`。
5. 摘要为 `SHA256("edgecache-shard-v1\0digest\0" || header || 0x00 || shard_bytes)` 的十六进制值；因此绑定节点标识、内容键、分片序号和最终字节。
6. 节点同时计算固定点校验和 `sum(q[i] * ((i % 17) - 8))`，父进程与子进程结果必须一致。

所有派生仅依赖声明，不依赖加载顺序、线程调度或批次切分；节点端与重放器使用同一纯函数，但通过各自进程独立计算。

## 租约、调度与唯一裁决

- 租约字段：`lease_id/node_id/content_key/shard_index/slot_index/bytes/granted_ms/expires_ms/lease_version/quota_version/digest/checksum`。
- 槽位按节点内最小空闲整数分配；`(node_id, slot_index)` 的活跃所有权由 SQLite 部分唯一索引保护。
- 三类容量同时生效：全局活跃租约数、单节点活跃租约数（同时受 `slots` 限制）、全局活跃字节预算。任一不足时不签发，申请进入队列。
- 队列记录 `wait_start_ms/wait_end_ms`（单调时钟整数毫秒）；释放、到期或配额回收后由过期线程/事件线程重新泵送。
- 续约先检查本地真实到期，再向节点发送 `expected_version`；节点版本不符返回 `E_LEASE`，账本只写拒绝事件，不修改到期时刻。成功续约才增加版本并延长 TTL。
- 过期以 `time.monotonic_ns()//1_000_000` 判定，由后台过期线程和事件边界共同推进。
- 每个分片校验作为独立任务提交到 `ThreadPoolExecutor`，每个任务建立真实回环 TCP 连接；不是进程内伪并发。

配额事件的固定裁决顺序是：**先处理 `expires_ms <= now` 的到期 claim，再处理版本撤销**。一个租约只能写入一次 `slot_claims`，因此即使到期与配额回收同时出现，也只有一个最终归属。到期依据为 `monotonic_deadline<=now; expiry_precedes_quota`；配额依据为 `quota_version_advanced; expiry_precedes_quota`。

## 配额演进与陈旧结论

配额版本递增或节点集合变化时：

1. 先做到期裁决；
2. 取消旧版本下尚未签发的排队申请（`CANCELLED_STALE`）；
3. 对旧版本仍活跃租约逐条写 `QUOTA_REVOKED` claim，释放槽位/字节；
4. 将旧当前视图、水位和结论整体写入 `stale_views`；
5. 在 `DELETE FROM current_view` 后整体替换当前视图，而不是增量合并；
6. 在 `quota_history` 中稳定记录新增/移除节点及新增/撤销/保留租约集合。

旧视图不会出现在当前租约、当前容量或当前展示中；界面只单列陈旧结论元数据。当前实现中旧版本租约全部撤销，因此该次转换的“保留租约集合”为空，这是显式且可重放的结论。

## SQLite 账本与崩溃恢复

账本包含 `journal/checkpoints/batches/leases/current_view/slot_ownership/queue_waits/slot_claims/quota_history/stale_views/state/meta`。`journal` 使用自增 `ledger_seq` 记录规范事件；`checkpoints` 与对应 journal 行在同一事务中写入容量余量、节点租约集合、进度游标、整数 `format_version=1` 与内容摘要。

授予、续约、claim、配额替换等每次状态变更都在一个 `BEGIN IMMEDIATE` 事务中完成。批次顺序是：

```text
BATCH_OPEN + RESERVED grants + slot ownership  -> 网络并行校验 -> BATCH_COMMIT/ACTIVE
                                                \-> 失败/强杀 -> LEASE_UNKNOWN + 槽位先回收
```

重新打开账本时：

- 若 `format_version` 或规范化声明 `manifest_digest` 不一致，旧库复制为 `*.stale-<timestamp>` 后整体新建，当前视图不会混入旧结论。
- 若存在 `OPEN` 批次，其中 `RESERVED` 租约改为 `UNKNOWN`，写入 claim，释放容量，`requires_rerun=1`。
- `--retry-unknown` 会重启死亡节点，以新 attempt 重跑未知申请；成功后写 `RERUN_RESOLVED`。失败或跨旧配额版本的未知项仍留在界面并保持需重跑状态。

## 独立重放

```powershell
python -m edgecache replay --sample samples\demo.json --db runs\demo.sqlite
```

重放器只以只读 URI 打开 SQLite，并读取原始声明文件。它按 `ledger_seq` 重新执行 grant/renew/claim/unknown/quota 事件，重新计算每个分片摘要与定点校验和，并逐条比较：

- checkpoint 中每个节点的租约集合；
- 全局/节点容量余量与已用槽位；
- 分片摘要、校验和；
- claim 唯一性、裁决依据、事件序号与配额版本；
- 最终 `current_view`、配额版本、活跃节点集合和 `requires_rerun`。

不一致时返回 `E_REPLAY`，并携带首个不一致的 `event_seq/node_id/field`。对当前视图篡改的测试已验证会定位到对应租约的 grant 事件和字段。

## 节点协议

每个连接先发送 hello，再发送一个操作；每行一个规范 JSON 对象。

```json
{"op":"hello","node_id":"edge-a","client":"edgecache-dispatcher/1","quota_version":1}
{"op":"validate_batch","batch_id":"B...","quota_version":1,"items":[{"lease_id":"L...","content_key":"clip-001","shard_index":0,"expected_digest":"..."}]}
{"op":"renew","lease_id":"L...","expected_version":1,"quota_version":1}
{"op":"release","lease_id":"L...","quota_version":1}
{"op":"set_quota","quota_version":2}
```

节点返回 `quota_version`、`manifest_digest`、pid、单调毫秒耗时、摘要、校验和与版本。端口占用在父进程预检和子进程 bind 两层报告 `E_PORT`；握手/套接字超时报告 `E_TIMEOUT`。

## 诊断界面

```powershell
python -m edgecache ui --db runs\demo.sqlite
python -m edgecache dashboard --db runs\demo.sqlite   # 无图形环境的稳定快照
```

界面固定展示租约水位、活跃租约、排队申请、未知租约、陈旧结论、配额转换和账本批次。所有查询按主键/事件序稳定排序；刷新只读，不写账本。当前验证机缺少 `libtk8.6.so`，因此图形窗口未实际打开，但同一数据路径通过 `dashboard` 和只读哈希测试验证；Windows 标准 Python 自带 Tkinter。

## 实际验证结果

本机环境：Python `3.11.0rc1`、NumPy `1.26.4`、SQLite `3.37.2`、Linux/WSL。Tkinter 因缺少系统库 `libtk8.6.so` 无法创建窗口，已按上述方式降级验证。

```text
python -m unittest discover -s tests -v
Ran 20 tests ... OK
```

覆盖：声明唯一性/行号、整除约束、序号连续性、确定性摘要、声明顺序不变性、真实到期、排队等待、配额撤销、E_LEASE 乐观版本拒绝、E_PORT、E_TIMEOUT、未知批次恢复、账本整体替换、重复打开续跑不重签、强杀一致性、E_REPLAY 篡改检测、只读界面稳定性。

`samples/demo.json` 实测：

```text
status=OK
claims=7
  EXPIRED: L000002-S0000, L000002-S0001, L000007-S0003, L000008-S0002
  QUOTA_REVOKED: L000001-S0000
  RELEASED: L000011-S0000, L000015-S0003
stale_views=3, quota_transitions=3
final active leases=0, global active bytes=0
replay=OK, journal_records=42, claim_count=7, exit=0
second run (same ledger): status=OK, events skipped, leases=0, claim_count=7
```

`samples/crash.json` 实测：

```text
clean run: 4 active leases, status=OK
crash phase 1: exit=3, status=PARTIAL, edge-a attempt 0 = UNKNOWN (2 leases),
               requires_rerun=true; edge-b = COMMITTED
crash phase 2 (reopen same ledger with --retry-unknown):
               status=OK, active leases=4, requires_rerun=false,
               edge-a batch attempt 1 = COMMITTED
normalized final conclusion (node/content/shard/slot/digest/quota) == clean run
replay=OK, journal_records=19, active_lease_count=4, exit=0
```

这些结果由 `runs/final-verification-*-report.json`、`runs/final-verification-*-replay.json` 留下；运行产物被 `.gitignore` 排除，可随时重跑。

## 边界与说明

- 所有网络仅使用本机 `127.0.0.1`，不访问外部目标，不使用 Docker 或容器。
- 账本绑定规范化声明摘要；任何影响语义的声明变化都会整体换新账本。
- 单调时钟毫秒适用于同一启动周期内的真实到期判断；重启后的“当前时间”仍使用系统单调时钟，账本历史中的到期事件不重新解释。
- `slot` 是节点内本地容量槽位，唯一性是 `(node_id, slot_index)`；跨节点不存在共享物理槽位。
- 当前实现侧重确定性、可重放与崩溃恢复，不包含鉴权、TLS 或跨主机传输；这些不属于首版回环演示范围。
