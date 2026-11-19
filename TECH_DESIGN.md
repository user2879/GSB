# 消息中间件实验工作台 — 技术设计说明

面向 Windows 本地环境的**确定性**分区聚合与水位观测工作台。不实现消息投递语义，
只对本地 NDJSON 确定性事件流做分区聚合、整数毫秒水位定稿、定点计算与交互延迟观测。
全部数据来自本地样本，无外部服务、无容器。

## 1. 模块地图

| 模块 | 职责 |
| --- | --- |
| `bench/mw/config.py` | 规则/规格/NDJSON 强校验、定点编码与饱和、规范化 JSON 序列化 |
| `bench/mw/engine.py` | 纯内存、确定性的分区水位状态机与聚合引擎 |
| `bench/mw/latency.py` | 定点均值/P95/最大值、误差上界、单调时钟测量 |
| `bench/mw/ledger.py` | SQLite 权威账本：原始事件、增量缓存、查询账本、权威重放 |
| `bench/mw/session.py` | 会话编排、查询、缓存-权威比对、参数版本替换、快照保存/恢复 |
| `bench/mw/snapshot.py` | 原子快照、整数格式版本、SHA-256 内容摘要 |
| `bench/mw/recovery.py` | 恢复后游标续传、E_REPLAY 校验、E_ORDER 倒退检测、崩溃 unknown |
| `bench/mw/fault_inject.py` | 仅改输入文件/参数的三类异常注入（不改源码） |
| `bench/mw/numpy_check.py` | numpy 独立第二实现交叉校验 |
| `bench/mw/gui.py` | 可选 Tkinter 实时仪表盘（无 tkinter 时优雅降级） |
| `bench/mw/cli.py` | 命令行入口，稳定错误码与退出码 |
| `bench/acceptance.py` | 一键可复现验收，产出结构化 JSON 总结 |

## 2. 八项业务要求的实现映射

### 要求 1：加载与强校验（E_SPEC）
- `load_spec` / `load_events` 逐字段校验：分区键唯一且匹配 `^[A-Za-z0-9_.-]{1,64}$`；
  同分区 `seq` 从 0 起连续严格单调；`ts_ms` 为非负整数；`out_of_order_ms`、
  `latency_budget_ms` 为正整数；`scale` 为 0..6。
- 整文件事务式校验：任一不合法即抛 `SpecError(E_SPEC)`，带**首个出错行号 `line`
  与字段 `field`**，不向账本提交任何数据。未知字段同样拒绝。

### 要求 2：整数毫秒水位定稿（E_LATE / E_WATER）
- 每分区独立维护 `max_ts_seen` 与单调 `watermark_ms = max(watermark, max_ts_seen -
  out_of_order_ms)`。事件 `ts_ms < watermark` 判为迟到，登记 `late_log`（E_LATE）、
  累加 `late_count`，**绝不进入已定稿集合**；否则入 pending，在
  `ts_ms <= watermark` 时定稿。
- 水位只增不减：`request_watermark` 遇回退抛 `WatermarkError(E_WATER)` 并保留原水位。
- 每次推进产生 `watermark_log` 步骤（旧/新水位、触发 `seq`、原因）。

### 要求 3：确定性定点聚合
- 引擎按分区 `seq` 规范序应用事件（批内分桶+排序），跨分区交错、批次大小、
  到达顺序都无法影响单分区结论；显式水位请求不越过事件推导边界时结论不变。
- 求和用 Python 任意精度整数累加，物化时一次性乘 `10**scale` 并按 **int64 饱和**
  （`sum_saturated` 标记），避免逐次饱和的非结合性。输入为整数，定点乘法无截断，
  单值量化误差为 0；视图经 `canonical_dumps`（排序键、无空白、禁 NaN）保证
  **逐字节一致**。

### 要求 4：增量缓存 vs 权威重放（E_VIEW）
- 缓存路径读 `partition_state` 维护表；权威路径 `authoritative_replay` 从
  `raw_events` 逐事件重建（完全不读缓存）。
- `verify_views` 逐分区逐字段比较，再比较两条路径的规范化字节串；首个差异以
  `ViewMismatchError(E_VIEW)` 报告分区与字段。任何展示结论均可由权威重算复现，
  缓存永不替代权威结论。

### 要求 5：交互延迟观测与预算（E_LAT）
- `time.monotonic_ns()` 起止差地板取整为整数毫秒；支持 `--inject-delay-ms`
  确定性制造超预算。真值来源声明为账本 `query_log` 逐次查询记录。
- 按分区与整体给定点 `mean/p95/max`：均值地板除（误差 `< 10**-scale`），
  P95 用 nearest-rank 精确次序统计；超预算写 E_LAT 与查询账本。延迟统计只读
  查询账本，不回写任何聚合状态。

### 要求 6：参数变更整体替换与陈旧防护
- 水位/预算/缩放因子变更后 `param_version` 单调 +1，旧版本置 `stale` 并从当前
  展示/汇总中隐藏；当前视图是**整体替换**的全新空状态，不做增量合并。
- 替换结果稳定列出每个受影响分区、旧/新水位、旧/新 scale 与预算。CLI 复连时若
  规格摘要未变则复用当前活动版本，避免无谓版本膨胀。

### 要求 7：快照版本识别与强制终止恢复
- 快照含整数 `format_version`、`param_version`、规则摘要、每分区聚合/水位/迟到、
  查询账本，以及 SHA-256 `content_digest`；写入走 temp+fsync+rename 原子替换。
- 重开比较快照参数版本与摘要；与当前会话不一致即按快照**整体替换**分区状态与账本。
- 强制终止后仅依据已完整提交（事务内 fsync）的记录恢复；提交游标之后的事件标记
  `unknown_reload_required`，要求重新加载，**不生成虚构退出报告**。

### 要求 8：恢复重放校验与异常注入（E_REPLAY / E_ORDER / E_SPEC / E_SNAPSHOT）
- 恢复后按快照游标续传剩余事件，`verify_against_snapshot` 逐项比较水位、定稿计数、
  迟到计数、定点和与查询账本；首个分歧定位到分区键、`seq`、字段并抛 E_REPLAY，
  附快照摘要依据。
- 三类注入只改输入/参数：事件文件损坏→E_SPEC；恢复续传时时间戳倒退超过水位
  余量→E_ORDER；快照内容摘要不一致→E_SNAPSHOT。均写结构化日志。

## 3. 错误码与退出码

| 码 | 含义 | CLI 退出码 | 性质 |
| --- | --- | --- | --- |
| E_SPEC | 规格/事件非法 | 2 | 硬错误，整体拒绝 |
| E_WATER | 水位回退 | 3 | 硬错误，保留原水位 |
| E_VIEW | 缓存与权威重放不一致 | 4 | 硬错误 |
| E_ORDER | 时间戳倒退超余量 | 5 | 硬错误 |
| E_SNAPSHOT | 快照版本/摘要不符 | 6 | 硬错误+整体替换 |
| E_REPLAY | 恢复重放分歧 | 7 | 硬错误 |
| E_LATE | 迟到事件登记 | — | 观测记录（不抛异常） |
| E_LAT | 查询超延迟预算 | — | 观测记录（入查询账本） |

## 4. 定点与误差口径

- 编码：`fixed = value * 10**scale`，输入为整数 → 无截断、单值绝对误差 0。
- 饱和：有符号 int64 边界钳制，`sum_saturated=true` 显式标注；`raw_sum` 以 TEXT
  存任意精度，保证可审计，饱和只发生在定点物化层。
- 延迟均值：地板除，绝对误差严格小于 `10**-scale`（报告 `error_bound_decimal`）；
  P95/max 为精确次序统计，误差 0；测量本身为整数毫秒地板取整（<1ms）。

## 5. 确定性论证（可复现性）

1. 事件按分区 `seq` 规范序处理，分区之间状态独立，跨分区交错不影响结论。
2. 水位是事件集与整数余量的纯函数，与批次划分、推进步长无关。
3. 任意精度累加 + 一次性饱和，消除加法顺序敏感性。
4. 所有视图经排序键、无空白、禁 NaN 的规范化 JSON 序列化，结构相等即字节相等。
5. 测试 `DeterminismTests` 用固定随机种子洗牌、多组批次大小、中间水位步长，
   断言视图与序列化字节完全一致。

## 6. 样本与验收

- `samples/basic`：3 分区、含乱序与迟到，附 `expected.json` 预期定稿聚合。
- `samples/resume`：两段式续传样本，附快照边界与全量预期。
- `samples/invalid`：非法规格/事件样本。
- `python -m bench.acceptance` 产出 6 组检查的结构化总结到
  `artifacts/acceptance_summary.json`；`python -m unittest discover -s tests` 共 31 个用例。
