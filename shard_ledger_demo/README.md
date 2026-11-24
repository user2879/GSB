# 跨节点分片下发配额账本组件（本地确定性可重复演示）

面向计算引擎多节点分片任务的本地产物。节点是独立 OS 子进程，在 `127.0.0.1` 回环 TCP
上以 JSON 行协议通信，使用 NumPy 对固定整数种子样本做整数定点分组聚合；调度器在
全局资源预算、每节点租约上限与并发配额三重约束下并行下发，所有裁决写入带格式版本与
链式内容摘要的 SQLite 账本，支持强制终止恢复、一致性重放、失效子集重算与结构化异常注入。

## 环境准备

- Windows 安装官方 CPython 3.11+，命令行可直接调用 `python` 与 `pip`。
- 安装依赖：`pip install -r requirements.txt`（仅依赖 NumPy；Tkinter 随官方安装包自带）。
- 不使用 Docker；所有网络操作仅限 `127.0.0.1`，样本由声明字段确定性生成。
- 一键演示：`run_demo.cmd`；一键测试：`run_tests.cmd`。

## 目录结构

```
shard_ledger_demo/
  shard_ledger/
    errors.py         # E_SPEC/E_DEPEND/E_CYCLE/E_PORT/E_TIMEOUT/E_FIELD/E_LEASE/E_REPLAY/E_SNAPSHOT
    clock.py          # time.monotonic_ns 单调毫秒时钟
    spec.py           # JSON/NDJSON 加载、强校验、稳定拓扑排序
    compute.py        # NumPy 整数定点确定性聚合与修订参数口径
    node.py           # 节点子进程：回环 TCP JSON 行服务
    protocol.py       # 子进程编排与真实套接字客户端（E_PORT/E_TIMEOUT/E_FIELD）
    lease.py          # 租约签发、乐观续约、真实过期释放
    engine.py         # 三重约束调度、依赖就绪、单次语义、失效传播、恢复重放
    invalidation.py   # 沿 DAG 与节点引用的稳定失效传播
    ledger.py         # SQLite 账本、格式版本、链式 SHA-256 内容摘要
    diagnostics.py    # Tkinter 诊断界面（无 Tk 时无头文本回退）
    fault_inject.py   # 仅改输入文件/参数的三类异常注入
    logger.py         # 结构化 JSONL 日志
    demo.py           # 命令行入口 run/recover/verify/inject
  samples/manifest.json, samples/events.ndjson
  tests/              # 29 项 unittest（单元 + 真实子进程集成）
  run_demo.cmd / run_tests.cmd / requirements.txt
```

## 声明与样本

`samples/manifest.json` 声明：3 个节点（N1/N2/N3，端口 9101-9103，各自并发配额、租约
上限、整数种子、容量版本）、7 个 DAG 分片（依赖、整数资源单位、目标节点集合）、修订链
v1（基线）→ v2（仅影响 N2 的定点口径 `scale/shift/sample_count`）。全局预算 12，
租约 TTL 500ms。`samples/events.ndjson` 按到达序号给出提交、取消、配额变更与修订事件。

## 业务要求到实现的映射

1. 强校验（`spec.py`）：节点/分片标识唯一、`127.0.0.1` 字面量与端口全局唯一、配额与租约
   上限为正整数、依赖/目标引用存在、Kahn 算法无环、修订版本严格递增且父修订指向上一版；
   非法输入抛出 `E_SPEC`/`E_DEPEND`/`E_CYCLE`，异常体带首个非法实体与字段。
2. 子进程与逐行协议（`node.py`/`protocol.py`）：按端口 spawn `python -m shard_ledger.node`，
   真实 socket 握手 + 批次 JSON 行，回执含节点/分片标识、配置版本、聚合摘要、单调毫秒耗时；
   端口占用 `E_PORT`、真实 socket 超时 `E_TIMEOUT`（`socket.settimeout`，非人为延迟）、
   回执缺字段 `E_FIELD`。
3. 租约（`lease.py`）：每次下发先签发含租约标识、分片、节点、授予/到期单调毫秒与版本号的
   租约；续约凭版本号乐观校验，不符抛 `E_LEASE` 且不延长原到期；清扫线程用单调时钟判定真实
   过期，过期即释放资源单位与节点租约计数。
4. 三重约束并行调度（`engine.py`）：在途资源单位之和 ≤ 全局预算，单节点活跃租约数 ≤
   `min(并发配额, 租约上限)`；耗尽则排队并记录等待起止单调毫秒；真实线程池执行，
   约束不满足绝不签发租约、绝不下发。
5. 依赖就绪与单次语义：分片仅在全部依赖完成且摘要可用后进入就绪集合；每轮调度每个分片至多
   下发一次；取消后不再重试；就绪与排队均按分片标识序稳定输出（排序处理，不依赖字典/哈希序）。
6. 失效传播与子集重算（`invalidation.py`）：修订到达后先等在途排空，再按“目标节点受影响”
   求直接失效集，沿 DAG 反向边求全部间接失效分片，按稳定拓扑序输出集合与逐步依据；仅重算
   失效集合，未失效分片不产生新一代记录，其摘要/版本/轨迹在账本中逐字节保留并可重放校验。
7. 账本版本/恢复/重放（`ledger.py`/`engine.py`）：每个结局在单个 SQLite 事务内写分片结局、
   摘要、资源占用、租约归属与进度游标；`meta` 存整数格式版本，`records` 用
   `SHA256(prev_digest|kind|canonical_json)` 形成链式内容摘要。强制终止后重启只恢复已完整
   提交记录；有事件游标但无完整结局的在途分片标记为 unknown 并要求重跑；`verify` 用确定性
   计算重放同一事件序列，首个不一致以 `E_REPLAY` 定位到分片标识与字段。
8. 诊断界面与异常注入（`diagnostics.py`/`fault_inject.py`）：界面按稳定顺序展示节点租约/
   资源、就绪/排队、失效集与依据、未知分片、账本版本与快照，只读刷新不改账本；
   `inject --inject corrupt-spec|cycle|snapshot` 仅修改输入文件或账本字节，分别报
   `E_SPEC`/`E_CYCLE`/`E_SNAPSHOT` 并写结构化日志，不改源码。

## 命令行用法

```bash
# 正常演示（产出 runs/<wd>/ledger.sqlite、events.log.jsonl、report.json）
python -m shard_ledger.demo run --manifest samples/manifest.json \
    --events samples/events.ndjson --workdir runs/demo
# 在某分片下发瞬间强制终止（os._exit(9)），模拟 kill -9
python -m shard_ledger.demo run --manifest samples/manifest.json \
    --events samples/events.ndjson --workdir runs/crash --crash-dispatch S2
# 恢复：识别 unknown 分片并仅重跑这些分片
python -m shard_ledger.demo recover --manifest samples/manifest.json \
    --events samples/events.ndjson --workdir runs/crash
# 重放校验：逐分片比对确定性摘要，不一致报 E_REPLAY
python -m shard_ledger.demo verify --manifest samples/manifest.json \
    --events samples/events.ndjson --workdir runs/demo
# 三类异常注入
python -m shard_ledger.demo inject --manifest samples/manifest.json \
    --events samples/events.ndjson --workdir runs/fault --inject corrupt-spec
python -m shard_ledger.demo inject ... --inject cycle
python -m shard_ledger.demo inject ... --inject snapshot
# Tkinter 界面（无 Tcl/Tk 的环境自动回退为无头文本快照）
python -m shard_ledger.demo run ... --ui
```

## 确定性与“逐字节不变”如何保证

- 样本由 `PCG64(sha256(节点种子|分片标识|生效参数集))` 生成，参数全为整数；聚合中间量
  全部 `int64`，相同输入在任意机器/任意调度时序下产出相同摘要。
- 节点仅在自身被某修订 `affects` 时更换口径；未受影响节点在修订前后摘要内容完全一致。
- 账本中的摘要以排序键、紧凑分隔符的 canonical JSON 入库；重放比较前剔除非确定性的
  `duration_ms`，因此耗时波动不会污染一致性判定，而业务字段任何字节差异都会触发 `E_REPLAY`。

## 已执行的验证结果

- `python -m unittest discover -s tests`：29 项全部 OK（含真实子进程/套接字集成测试）。
- 端到端演示：S1/S2/S3/S5/S7 在 v2 修订后产生 gen2 重算记录；仅目标 N1 的 S4 与已取消的
  S6 不重算，S4 保留 gen1/v1 单条轨迹；`verify` 对全部已完成分片重放通过。
- 下发中途 `os._exit(9)` 后恢复：S2 被标记 `incomplete at forced termination` 的 unknown
  分片并仅重跑 S2，随后重放校验通过。
- 异常注入：损坏声明 → `E_SPEC`（定位到 JSON 行列）、依赖成环 → `E_CYCLE`（定位首个实体
  与环成员）、篡改账本摘要 → `E_SNAPSHOT`（定位到 record 序号与 digest 字段），均落结构化日志。

## 说明与限制

- 本沙箱为 Linux 且缺少 Tk 动态库，UI 走无头回退；目标 Windows 官方 CPython 自带 Tkinter，
  `--ui` 直接打开窗口。节点额外带 8 秒空闲自动退出，便于崩溃演示后回收回环端口。
- 演示端口固定在样本中，重复运行前请确认 9101-9103 未被占用；崩溃留下的节点可由
  `protocol.kill_orphans` 或等待空闲超时回收。
