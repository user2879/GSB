# 架构与技术说明

## 1. 模块总览

| 模块 | 职责 |
| --- | --- |
| `faultisolation/spec.py` | 声明/事件流/会话加载与强校验（`E_SPEC`，含字段与行号） |
| `faultisolation/clock.py` | 物理单调时钟 `MonotonicClock` 与逻辑时钟 `LogicalClock` |
| `faultisolation/engine.py` | 租约状态机、槽位唯一归属、双重并发约束、确定性引擎 |
| `faultisolation/coordinator.py` | 每域 FIFO 排队协调器（引擎与回环服务共用同构组件） |
| `faultisolation/planner.py` | 三路径工作量估算、显式规模规则、阈值与稳定决胜 |
| `faultisolation/fixedpoint.py` | 整数定点比例、截断方向、单值/比较误差界 |
| `faultisolation/budget.py` | 当前/峰值槽位占用口径（纯整数采样） |
| `faultisolation/ledger.py` | SQLite 账本、事务、格式版本与 SHA-256 摘要、恢复 |
| `faultisolation/protocol.py` | TCP JSON 行协议（固定字段顺序，逐字节可比较） |
| `faultisolation/server.py` | 每域子进程仲裁服务（真实回环、物理 RTT、逻辑仲裁时刻） |
| `faultisolation/process.py` | 子进程编排、线程池实时调度、回执核对 |
| `faultisolation/replay.py` | 独立重放、三路径逐字节校验、`E_DIVERGE` 定位 |
| `faultisolation/recovery_demo.py` | 强制终止（kill）与账本恢复演示 |
| `faultisolation/gui.py` | Tk 工作台，缺 Tk 时自动降级无头控制台 |
| `faultisolation/cli.py` | 统一命令行入口 |

## 2. 关键设计决策

### 2.1 双时钟模型
- **物理单调时钟**（`time.monotonic_ns`）：仅用于服务端为请求打戳并测量真实回执往返 RTT。
- **逻辑单调时钟**：取值锚定事件 `arrival_ms`，租约到期判定、槽位再分配、等待时长全部基于它，
  因此离线重放与真实回环执行的确定性结论逐字节一致。
- 平台不使用任何线程 `sleep` 模拟租约超时；`Condition.wait` 仅用于可选节拍对齐，默认关闭。

### 2.2 唯一事实来源
三条仲裁路径（`event`/`batch`/`merge`）调用同一个 `Engine` 与同一套状态机，
仅事件组织方式不同，状态机推进严格保持规范序
`(arrival_ms, priority, instance_id, seq)`，因此结论天然逐字节一致。
`replay.py` 作为独立进程只读原始声明、事件流与账本，自行重放后比对，不依赖主流程内存。

### 2.3 双重并发约束
- 全局并发租约上限：引擎计数 + 实时线程池 `BoundedSemaphore`。
- 每域租约上限：`DomainArbiter` 准入判断 + 每域信号量。
- 容量不足进入每域 FIFO 队列，只允许队首获准入，优先级不能插队；
  等待开始/结束单调毫秒与等待时长落账本。

### 2.4 槽位唯一归属
- 统一分配规则：始终取当前最小的 `demand` 个空闲槽位（升序）。
- 授予前校验 `slot_owner`，重复持有立即抛 `E_ALIAS` 并报告冲突双方租约标识。
- 回收即释放槽位，再分配只发生在回收之后，保证同一槽位任一时刻只有一个活跃租约。

### 2.5 乐观续约
续约必须携带当前版本号；版本不符抛 `E_LEASE` 且**不延长**到期时刻、版本号不自增。

### 2.6 定点误差界
设缩放因子 `K=10^p`（p∈{0,1,2,3}），量化 `q=used*K//total`（向零截断）：
- 单值量化误差界 `1/K`；
- 两比例比较合成误差界 `2/K`；
- 量化差落入界内时按 `(量化值降序, 标识升序)` 稳定决胜并标注“受误差影响”。

### 2.7 账本与恢复
- 每批次在**同一 SQLite 事务**写结局、迁移、槽位、等待、游标与摘要。
- 摘要为全量结局 canonical JSON 的 SHA-256，校验器可独立复算。
- 未提交批次因事务原子性不存在，恢复游标 `unknown_from_seq` 指向需重跑的首个事件。
- 格式版本或摘要不一致：整体删除重建，绝不增量合并。

## 3. 真实回环执行
每个故障域由独立 Python 子进程在声明的 `127.0.0.1` 端口监听；客户端经线程池发起 TCP JSON 行请求。
服务端内嵌与引擎同构的 `DomainCoordinator`，FIFO 排队/泵取在真实服务内复现。
实时驱动把每个事件的逻辑时刻随请求注入，服务端据此做到期判定；物理 RTT 单独记录用于审计。
运行结束用真实快照核对服务端槽位归属与确定性计划完全一致。
