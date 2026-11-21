# 设计说明：图遍历可达距离定点估算与误差传播诊断台

## 1. 模块结构

```
graphdiag/
  core/
    errors.py       # E_GRAPH / E_SCALE / E_REPLAY / E_STATE 错误码
    models.py       # 不可变领域模型、规范 JSON、SHA-256 内容摘要
    loader.py       # JSON 强校验加载（首个冲突即整体拒绝）
    fixedpoint.py   # 定点量化、饱和加法、误差界合成、主导误差段
    traversal.py    # 确定性有界 Dijkstra、剪枝、预算钩子、上限归因
    scheduler.py    # 并发上限 + 共享访问预算 + 单调时钟排队
    persistence.py  # SQLite 遍历账本 / 快照 / 事件审计
    engine.py       # 版本号、陈旧防护、整体替换差异、快照恢复重放
  gui/app.py        # Tkinter 专业界面（无 tk 时优雅降级）
  cli.py            # 命令行入口
  sample_data.py    # NumPy 固定种子确定性样本生成
scripts/demo.py     # 可重复端到端演示
tests/              # 27 个 unittest 用例
data/               # tiny_graph.json（手算可核）与 sample_graph.json（NumPy 生成）
```

## 2. 定点整数口径（显式声明，无隐式浮点）

- 原始边权与上限均为非负整数“微单位”，`1_000_000` 微单位 = 1.0 逻辑单位。
- 会话缩放因子 `S = 10 ** exponent`，`exponent ∈ [0, 6]`。
- 量化：`q = floor(weight_micro * S / 1_000_000)`，即向零截断（权重非负，等同向下取整）。
- 1 个定点单位（ulp）= `10 ** (6 - exponent)` 微单位。
- 累加全部为整数加法；溢出采用有符号 64 位最大值 `FP_MAX = 2^63 - 1` 的**饱和加法**。
- 会话启动前做域检查：可达的最大整数为 `quantize(cap) + quantize(最大边权)`，若超过
  `FP_MAX` 立即以 `E_SCALE` 拒绝，饱和仅作防御性守卫，不产生静默结果。

## 3. 误差传播与上界

单边截断余数 `r_i = weight_micro_i - q_i * micro_per_ulp`，满足 `0 <= r_i < 1 ulp`。

- 精确上界（exact）：路径上余数之和 `Σ r_i`（微单位）。
- 保守上界（conservative）：`m 条边 × 1 ulp`，随边数线性增长，逐边输出中间项。
- 主导误差段：余数最大的边；余数相同取路径上最早出现者（稳定决胜）。
- 多候选路径决胜键（全序，逐字节稳定）：
  `(距离 fp, 精确余数上界, 保守 ulp 上界, 节点路径字典序)`。
  误差界仅在距离相等后参与，绝不改变最短距离结论。

## 4. 上限比较与超界归因

对每个目标输出 `上限 cap / 距离 d / 误差界 e` 三元组，分类为：

- `within`：`d + e <= cap`，可确定在限内。
- `confirmed_over`：`d > cap`，累计距离本身确实超界。
- `indeterminate_error_cross`：`d <= cap < d + e`，仅因误差上界跨越上限，无法判定，
  并标注 `error_affected=true`。
- `unreachable`：限内无路径到达（含预算耗尽未到达）。

exact 与 conservative 两种口径同时计算并持久化，当前视图按会话 `error_mode` 选用。

## 5. 确定性保证

- 邻接表在加载时按 `(weight_micro, target, source)` 预排序；遍历只走排序后的元组。
- 堆键为显式全序元组，含单调 `counter` 与显式路径，不依赖 dict/set/hash 顺序。
- 同一图与同一查询在任意节点/边插入顺序下，访问序列与结论逐字节一致
  （`tests/test_traversal.py::test_byte_identical_under_insertion_shuffle`）。
- 剪枝记录 `cap_exceeded` / `saturated` 及候选值、上限值。

## 6. 预算与并行调度

- `max_workers` 控制全局并发；`TraversalBudget` 为共享访问节点预算，每次节点扩张前
  必须 `acquire()` 一个令牌，预算耗尽不得继续遍历。
- 预算耗尽时在途查询以 `budget_exhausted` 停止，排队查询在条件变量上等待补充，
  并用 `time.monotonic_ns()` 毫秒记录等待开始/结束、查询开始/结束时刻、访问节点数、
  剪枝分支数与预算消耗。

## 7. 版本、陈旧防护与整体替换

- 图/查询/缩放分别做内容寻址（规范 JSON 的 SHA-256）。图内容或缩放口径变化 →
  `graph_version` 单调递增；查询集合变化 → `query_version` 单调递增。
- 任何变更后旧 `ConclusionView` 立即置空（陈旧结论不参与展示与统计），新结论以
  整体替换方式生成，绝不增量合并；`diff_views()` 稳定列出查询标识、目标、距离变化量
  与判定变化。

## 8. 快照与一致性重放

- SQLite 持久化 `ledger`（追加账本）、`snapshots`（全量快照）、`snapshot_events`（审计）。
- 快照携带 `integer_format_version=1`、图/查询摘要、版本号、缩放口径、账本与全部结论，
  并对“除格式版本字段外的规范 JSON”计算 SHA-256。
- 恢复时先校验摘要与格式版本，再以快照内图**单线程重放全部查询**，逐目标比较距离、
  误差界、判定、主导段与路径；首个不一致定位到 `(query_id, target, field)` 并抛 `E_REPLAY`。
  只有重放通过才整体采用快照状态。

## 9. 已知环境限制

当前 Linux 沙箱解释器未安装 Tkinter 扩展（无法无密码提权安装 `python3-tk`），
GUI 模块在导入时被守卫并给出友好提示，CLI 与全部核心功能、测试不受影响。
Windows 官方 Python 安装包自带 Tkinter，可直接 `python -m graphdiag` 启动界面。
