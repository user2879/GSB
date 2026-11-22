# MeshBench 架构与一致性设计

面向本地 Windows、仅依赖回环 127.0.0.1 的服务网格流量治理与错误预算诊断工作台。
无容器、无外部服务；模拟上游是真实 TCP 监听，超时与连接错误都来自真实套接字。

## 模块地图

| 模块 | 职责 |
| --- | --- |
| `meshbench/config.py` | JSON 规则加载与强校验，非法声明整体抛 `E_RULE`，不产生分发记录 |
| `meshbench/hashing.py` | SHA-256 稳定哈希 → `[0,10000)` 分桶 → 累计权重区间选实例 |
| `meshbench/upstream.py` | 线程化确定性模拟上游，真实回环 TCP，按请求键哈希决定错误/慢响应 |
| `meshbench/netutil.py` | 端口占用预检与上游可达性预检（仅 127.0.0.1） |
| `meshbench/breaker.py` | CLOSED/OPEN/HALF_OPEN 熔断状态机，真实单调时钟计时 |
| `meshbench/dispatcher.py` | 真实 socket 尝试、信号量并发预算、重试、熔断联动 |
| `meshbench/budget.py` | 定点 SLI 与逐窗口错误预算回执（纯函数、顺序无关、字节稳定） |
| `meshbench/ledger.py` | SQLite 账本：单事务批次写入、哈希链内容摘要、在途游标 |
| `meshbench/replay.py` | 权威重放/重算，逐字节比对，`E_PREVAIL` / `E_REPLAY` |
| `meshbench/snapshot.py` | 会话快照、摘要校验、版本识别整体替换 |
| `meshbench/engine.py` | 编排：缓存视图、规则演进整体替换、计时、快照 |
| `meshbench/view.py` / `tk_dashboard.py` | 无头聚合视图与懒加载 Tk 仪表盘 |
| `meshbench/cli.py` | `validate` / `demo` / `recover` / `inject` 入口 |

## 确定性分发

- 命名空间 `meshbench/v1 | rule_version | route_id | client_key` 经 SHA-256 取 8 字节，
  对 10000 取模得到整数分桶。不使用 Python 随机化的内置 `hash`。
- 实例按 `instance_id` 显式排序后累加权重；切点 `point = bucket * total_weight // 10000`
  使用精确整数地板除，零权重实例不进入候选集合，因而永不被选中。
- 实例选择在整次请求内固定，重试只在报文里携带递增 `attempt`，
  `attempt_timeout_ms` 每次尝试都重新设置为同一值，不被重试放大。

## 熔断状态机

- 计时只用 `time.monotonic()`（真实本地单调时钟），不使用可能被虚拟化的窗口时钟。
- 连续失败达到阈值 → OPEN 并把连续失败计数清零；OPEN 冷却内 `allow_request()`
  直接返回拒绝，分发层抛 `E_CIRCUIT` 快速失败且**不发起任何连接**。
- 冷却结束进入 HALF_OPEN，按 `half_open_probes` 显式放行试探；试探失败立即重开
  并重置计数，试探全部成功才回到 CLOSED。所有迁移产出事件并随批次落库。

## 四类结局与定点回执

结局互斥：`success`（成功且未超慢阈值）、`slow`（成功但真实耗时 ≥ 阈值）、
`failure`（上游错误/真实连接错误/熔断快速失败）、`timeout`（真实 socket 超时）。

- 缩放因子 `SCALE = 1_000_000`，所有比例以整数微单位存储。
- 所有定点除法显式**地板截断**，单个比例量化绝对误差严格 `< 1/SCALE = 1e-6`，
  回执内以 `quant_error_max` 显式声明分子/分母上界。
- 预算：`allowed_bad_scaled = total * budget_permyriad * SCALE // 10000`，
  `remaining_scaled = allowed_bad_scaled - bad * SCALE`。
- 样本数 `< min_samples` 时 `insufficient_sample=true` 且绝不判定预算耗尽。
- 回执是计数器与路由参数的纯函数，不含任何到达顺序或时间戳字段，
  通过 `canonical_dumps`（排序键、固定分隔符）保证逐字节可复现。

## 账本、哈希链与恢复

- 每个批次在**一个 SQLite 事务**内写：请求、逐次尝试、熔断事件、熔断状态、
  窗口回执、进度游标、pending 状态与哈希链。
- 哈希链：`entry = BLAKE2b(prev_head || batch_digest)`，账本头保存最新链摘要，
  `meta` 保存整数 `format_version`。
- 强制终止时未提交事务中的请求只会残留在 `pending_requests(state=in_flight)`，
  恢复时报告为 `unknown` 并要求重跑，不虚构退出报告。
- 恢复仅依据已提交记录重放窗口，重算回执与存储的 `canonical_bytes` 逐字节比较：
  缓存视角差异报 `E_PREVAIL`，恢复重放差异报 `E_REPLAY`，账本摘要不符报 `E_SNAPSHOT`。

## 规则演进与整体替换

- 新版本号必须严格单调递增；通过 dataclass 规范化摘要比较路由与服务，
  服务变化会传播到所有引用它的路由。
- 受影响窗口的旧回执与熔断状态整体置 `stale=1`，不参与当前视图与汇总；
  当前视图整体替换而非增量合并，替换报告稳定列出窗口、旧/新版本与受影响实例。

## 时钟与真值来源

- 熔断与所有界面查询/权威重算的耗时使用 `time.monotonic()` 毫秒差，写入 `view_timings`。
- 演示用 `VirtualClock` 仅用于跨统计窗口的确定性窗口编号；真值来源始终是
  账本中的逐次请求记录，权威重算只读取这些记录。

## Tkinter 说明

`tk_dashboard.py` 在调用时才 `import tkinter`，无 Tcl/Tk 的环境仍可使用全部
领域组件与无头视图。Windows 安装 Python 3.11 时勾选 Tcl/Tk 即可启用仪表盘。
