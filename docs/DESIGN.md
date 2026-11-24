# 技术说明：分层滑动聚合组件 obsflow

## 1. 分层架构

```
观测源声明 JSON / 窗口参数 JSON
        │
记录流 v1/v2/v3 ──► codec 强校验与归一 ──► NormalizedRecord（含来源版本/补齐集合/补齐依据）
        │
aggregation 纯函数定点量化与滑动窗口（与批次、线程无关）
        │
propagation 失效推导 ──► scheduler 双配额线程池 ──► lease 租约
        │
ledger SQLite 单事务账本（窗口/租约/传播/游标 + 版本与摘要）
        │
diagnostics（Tk 界面 / 无头视图模型 / 本地 socket）+ cli 异常注入
```

- `errors.py`：业务码层级。`E_ORDER` 继承 `E_FORMAT`，因此加载期序号倒退
  既满足“以 E_FORMAT 拒绝并指明行号字段”，注入链路又能给出更具体的 E_ORDER。
- `codec.py`：JSONL 头声明版本，逐行强校验未知版本、缺字段、类型、序号倒退、
  负时刻、非法缩放因子；归一只补齐缺失字段并记录依据，绝不覆盖原值。
- `aggregation.py`：无副作用纯函数。全局顺序固定为
  `(seq, source_id, metric_key, line_no)`，任何批次切分与推进节奏产出相同字节。

## 2. 定点语义与误差上界

记录整数 `raw_value` 在 `record_scale` 下，会话缩放因子为 Q：

```
q = truncate(raw_value * Q / record_scale)      # 向零截断
```

- 单值严格误差上界：`|value - q| < 1`（量化整数单位）。
- 长度 n 窗口求和严格上界：`|sum_error| < n`，结果字段
  `value_error_bound_q=1`、`sum_error_bound_q=count`。
- 均值为 `trunc(sum_q / count)` 向零；量化值与求和按 int64 钳位饱和，
  `saturation` 字段显式记录 `none / value_or_sum_clamped / mean_clamped`。
- 当 record_scale 恰为 Q 的整数倍时误差为 0，上界仍按严格不等式声明。

## 3. 失效传播与仅子集重算

`propagation.py` 沿 观测源 → 指标键 → 窗口 引用关系推导：

- 格式版本 / 默认缩放变化：该源喂入的指标键全部窗口失效；
- 指标映射变化：源键与目标键两侧窗口失效；
- 窗口长度 / 步长变化：所有指标键的窗口网格整体重建；
- 输出按稳定顺序编号，每个事件带 reason、变更字段、旧值新值、受影响窗口与依据。

引擎只重算受影响（或输入摘要变化、或 unknown）窗口；其余窗口连同
`window_version` 逐字节保留。窗口版本仅在输入变化或强制失效时递增，
重算结果携带 `input_digest` 与 `result_digest`（SHA-256 规范化 JSON）。

## 4. 双配额并发调度

`scheduler.py` 使用真实 `ThreadPoolExecutor`，全局信号量与每观测源信号量
均以非阻塞获取做配额判定，没有任何人为 sleep。任务排队的
`enqueued / wait_start / wait_end / run_start / run_end` 全部取单调时钟；
测试用 `ManualClock` 让等待时间可确定断言。聚合结果与调度节奏无关，
线程只改变执行时机。

## 5. 租约

`lease.py` 的租约含租约 id、指标键、窗口起止序号、授予/到期毫秒与版本号。
续约凭版本号乐观校验，版本不符抛 `E_LEASE` 且不延长；过期由
`time.monotonic_ns`（或测试时钟）真实判定，过期租约被回收，其窗口状态
不参与后续聚合。重算窗口前先吊销旧租约再签发新租约。

## 6. 账本、快照与强制终止恢复

`ledger.py` 每个批次在单个 SQLite 事务内写 batches/windows/leases/
propagation/cursors，并在 meta 记录整数账本格式版本与内容摘要。

- 重开先比较观测源集合版本、格式版本集合、缩放口径版本；版本演进触发
  整体替换并把旧库复制为只读备份（chmod 0444），禁止增量合并。
- 同会话下 `verify_snapshot` 单独校验内容摘要，损坏即 `E_SNAPSHOT`。
- 强制终止只可能留下 `in_progress` 批次；恢复仅信任已提交数据，未提交
  批次触及的窗口标记 `unknown`、吊销其租约并要求重跑，不生成退出报告。
- `replay_verify` 按账本记录的格式版本与缩放因子重新推进，逐字段比较
  count/sum/mean/误差上界/窗口版本/状态/摘要，首个差异定位到指标键、
  窗口与字段并报 `E_REPLAY`。

## 7. 异常注入

`injection.py` 只改输入文件或参数，不改引擎逻辑：截断流 → E_INTEGRITY，
序号倒退 → E_ORDER，篡改摘要 → E_SNAPSHOT；结构化日志写 JSONL。
`process_runner.py` 用独立 `sys.executable` 子进程留下未提交批次后被
真实 SIGKILL，用于崩溃恢复演示。
