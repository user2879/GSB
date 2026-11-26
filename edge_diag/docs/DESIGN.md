# 边缘缓存分片下发诊断工作台 — 技术说明

## 1. 目标与范围

对同一批边缘缓存分片，在多节点租约水位反复编辑、配额版本换代、子进程被
强制终止之后，验证结论能否被原样复核。所有组件运行在本地，节点为独立
子进程，经 `127.0.0.1` 回环 TCP 的 JSON 行协议通信，不访问任何外部网络。

## 2. 模块结构

| 模块 | 职责 |
| --- | --- |
| `edge_cache/spec.py` | 节点声明/内容清单/事件流 JSON 加载与强校验（E_SPEC） |
| `edge_cache/determinism.py` | 确定性分片字节、FNV-1a 混合种子、NumPy 整数定点校验摘要 |
| `edge_cache/clock.py` | 单调时钟封装，统一整数毫秒 |
| `edge_cache/protocol.py` | JSON 行协议、真实套接字超时（E_PORT/E_TIMEOUT） |
| `edge_cache/node.py` | 缓存节点子进程：真实监听、并发连接、租约版本乐观校验 |
| `edge_cache/ledger.py` | SQLite 原子账本、格式版本/摘要整体替换、快照与游标 |
| `edge_cache/ledger_queue.py` | 专用账本写入线程，解决 SQLite 线程亲和性 |
| `edge_cache/engine.py` | 调度、租约生命周期、双路回收唯一归属裁决、崩溃恢复 |
| `edge_cache/replay.py` | 独立只读重放校验（E_VERIFY 首点定位） |
| `edge_cache/samplegen.py` | 确定性样本生成 |
| `edge_cache/cli.py` / `__main__.py` | 命令行入口 |
| `edge_cache/ui.py` | Tkinter 只读水位界面 |

## 3. 确定性与定点校验

- 分片字节仅由 `(内容键, 内容种子, 分片序号, 分片字节数)` 决定，使用
  `numpy.random.default_rng(派生种子)` 的 PCG64 生成，与节点、加载顺序、
  批次切分无关，因此任意批次切分下字节逐位一致。
- 种子由 64 位 FNV-1a 对字段显式混合，顺序敏感、可复现。
- 节点以自身种子派生样本，`floor(uniform * session_scale)` 量化为 `int64`
  整数后由 NumPy 求定点加总 `fp_sum` 与平方和 `fp_sqsum`。
- 摘要 `sha256(node|content|shard|byte_hash|fp_sum|fp_sqsum|n|scale)`，
  绑定节点标识、内容键、分片序号与内容字节。
- 主进程对每个回执独立重算同一摘要并逐字段比对，漂移即 `E_CHECKSUM`。

## 4. 租约与真实单调时钟

租约字段：`lease_id, node_id, shard_id, grant_ms, expire_ms, renew_count,
lease_version, quota_version, slot_index, state`。

- 时间戳全部来自 `time.monotonic_ns()//1_000_000`，到期判定仅比较
  `now_ms >= expire_ms`，不使用 `sleep` 或线程超时冒充回收。
- 续约在节点侧做乐观校验：携带 `expected_version`，版本不符或已到期回
  `E_LEASE` 且不延长到期；成功则版本号自增、续约次数累加。

## 5. 双路回收的唯一归属裁决

同一份槽位 `slot_key=(node_id, slot_index)` 同一时刻至多被一个活跃租约
持有。到期回收（`expired`）与配额换代回收（`quota_recycle`）互斥：

- 配额变更前先做一次真实到期扫描，到期归属优先落账。
- `_reclaim` 幂等，锁内二次确认租约仍为 `active` 才记账，先到者写入
  `ownership`（含裁决依据 `basis`、事件序号、配额版本、单调时钟时刻），
  后到者直接返回，不重复记账、不重复释放名额。
- 槽位释放回空闲池，节点容量、节点租约数、全局租约数三重水位同步更新。

## 6. 三重约束并行调度

- 全局活跃租约数、单节点活跃租约数、累计下发字节预算（授予即预占、不
  返还）三重闸门，全部在 `Scheduler` 的条件变量内裁决。
- 工作真实投射到 `ThreadPoolExecutor`；等待区间记录真实
  `wait_start_ms/wait_end_ms`（见账本 `waits` 表）。
- 预算耗尽在授予前拒绝并落 `rejected/E_BUDGET`，不绕过限制继续下发。
- 写任务先取调度名额再做账本/TCP I/O，避免“持账本等名额、持名额等账本”
  的循环等待；事件循环不被排队写入阻塞，使释放/配额事件能解锁排队者。

## 7. SQLite 原子账本与整体替换

- 每次授予/续约/回收在单事务内写 `event_records`、`leases`、`node_water`、
  `ownership` 与 `progress` 游标，要么全可见要么全不可见（WAL + FULL
  synchronous）。
- `meta` 携带整数 `format_version` 与声明 `spec_digest`；重开时版本或摘要
  不一致即 DROP 全部业务表整体重建，绝不增量合并。
- 未确认批次先写 `unknown_batches`；恢复时只依据完整提交记录重建，未确认
  批次先回收其占用再要求重跑。
- SQLite 连接有线程亲和性，所有写操作经 `LedgerThread` 在创建连接的同一
  线程串行提交；读操作同样经该线程派发。

## 8. 强制终止恢复

`crash-demo` 在清晰提交边界 `SIGKILL` 节点，随后用全新引擎：

1. 打开账本，版本/摘要校验通过后按 `leases/node_water` 重建内存态；
2. 恢复点用真实时钟裁决跨进程停机期间到期的租约（保留负 `seq` 段记账，
   避免与正常事件主键冲突）；
3. 回收未知批次占用；
4. 重启被杀节点并回放其已提交租约表；
5. 依据 `progress.last_committed_seq` 续跑未提交事件；
6. 独立重放校验逐字段一致。

租约/批次 id 计数器在恢复时推进到账本最大值之后，避免新进程复用旧 id。

## 9. 错误码

| 代码 | 含义 | 定位 |
| --- | --- | --- |
| E_SPEC | 声明/清单/事件非法 | 首个非法字段路径或 event#seq |
| E_PORT | 端口占用/连接拒绝/套接字错误 | host:port |
| E_TIMEOUT | 真实套接字连接/读超时 | host:port |
| E_LEASE | 续约版本冲突/目标不存在/已到期 | event#seq.node.shard |
| E_BUDGET | 全局字节预算耗尽 | event#seq |
| E_CHECKSUM | 分片摘要字段不一致 | event#seq.node.shard.field |
| E_VERIFY | 重放与账本不一致 | event#seq.node.field |
| E_LEDGER | 账本版本/元数据异常 | meta 字段 |

## 10. 运行方式

```bash
python3 -m venv .venv && source .venv/bin/activate   # 或 virtualenv
pip install -r requirements.txt
cd edge_diag
bash scripts/demo.sh           # 一键全流程
python -m edge_cache.cli gen-sample --spec samples/demo_spec.json
python -m edge_cache.cli run   --spec samples/demo_spec.json --db state/ledger.sqlite3
python -m edge_cache.cli crash-demo --crash-seq 3 --crash-node node-b
python -m edge_cache.cli verify
python -m edge_cache.cli ui
```

Tk 运行时：系统缺失 `python3-tk` 时，项目内置 `.tk-python`（tkinter 包+
扩展）与 `.sysroot`（libtk/tcl 脚本），`source scripts/tk-env.sh` 后即可
建窗；无显示时可用 `scripts/start-xvfb.sh` 在用户命名空间内启动本地 Xvfb。
