# 定点特征管线诊断工具 — 技术设计

## 1. 范围与约束
- 纯 Python 3.11 标准库实现（`sqlite3/json/struct/argparse/hashlib`），不依赖网络、容器或外部数据集。
- 全部数值路径为**整数定点运算**，浮点 `float` 仅出现在测试对照（`math.log`）中，不进入管线。
- Windows 原生运行：仅使用 `pathlib/os.path` 与相对路径，无 shell 专有语法。

## 2. 定点口径（显式声明）
- 表示：带符号 32 位整数，尺度指数 `e∈{0..6}`，实值 `v = x / 10**e`。
- 除法舍入：统一 `round half away from zero`（`fixedpoint.div_half_away`），与符号无关、确定。
- 溢出：饱和到 `[-2^31, 2^31-1]`（`fixedpoint.sat32`），饱和行单独打标，绝不回绕。
- 空值：以 `None` 表示并逐阶段传播；空值位图由独立空值种子的 LCG 确定性生成。
- 对数：`fixedpoint.ilog` 用整数位分解 + 整数泰勒级数（内部 30 位二进制分数），
  非正输入与空值输出 `None`；输出误差上界常量 `LOG_ROUTINE_BOUND=16`（实测最大 12 ulp）。

## 3. 阶段变换（全部整数）
| 阶段 | 整数口径 | 关键规则 |
|---|---|---|
| standardize | `z=sat32(div_half_away((x-mean)*10^scale, sd))` | `sd>0`，否则 `E_PARAM` |
| bucket | 整数边 `x<=edge` 归左桶，输出桶中心整数 | 边界归属不确定行打标 |
| log | 整数 ln/log2/log10 | 非正→空值 |
| product | 定点乘 `mul_quant`，先精确大整数乘积再缩放 | 记录放大与量化误差 |
| truncate | 按本阶段尺度声明的上下界钳制 | 钳制行打饱和标 |

所有阶段都是无状态逐元素运算；块切分（`block_size`）与两种调度
（`stage_major`/`pipeline_major`）产出逐字节一致，见 `tests/test_determinism.py`。

## 4. 误差上界传播（errorchain.py）
两遍法，严格按声明的阶段顺序：
1. 值遍：先算出各阶段整数输出与每块 `max_abs`；
2. 界遍：用观测幅度合成紧致上界（输出 ulp）。
- 量化误差 `quant`：缩放除法最大残差 `10^d/2`。
- 分桶误差 `bucket`：上游界 + 最大桶半宽，按尺度换算。
- 交互放大 `product`：`|x|·eb + |y|·ea`（二阶小项舍弃）+ 输出量化。
- 主导段阈值 `DOMINANT_SHARE_BPS=4000`（40%），并列时取最早阶段（稳定决胜）。
- 沿 `refs` 单父链/双父（product）传播，只依赖声明顺序，不依赖 dict 遍历顺序。

## 5. 统计与漂移（stats.py，全部整数）
- 均值：先整数求和再 `div_half_away(sum,n)`（和可结合 → 分组顺序无关）。
- 分位桶分布：以**参考窗**派生的固定整数十分位边，同时用于两窗，直方图为纯计数。
- 缺失率：`bps_ratio(nulls,total)`，万分比整数。
- 漂移三分量（万分比）：均值相对漂移、固定边直方图总变差(TV)、缺失率差；复合取 max。
- 每分量携带误差界：均值界 `b/|ref.mean|*1e4`；TV 界仅由边界不确定行数驱动；缺失界为 0。

## 6. 不可区分裁决（稳定决胜）
对复合漂移 `d` 与误差 `e`、阈值 `T`：
- `d-e > T` → `drift_confirmed`（确超）；
- `d+e < T` → `stable`（确未超）；
- 否则 `indistinguishable`（误差不可区分），标注 `error_affected=true`。
等号情形（区间恰好触碰阈值）保守归入不可区分，永不判超。

## 7. 失效传播与增量（orchestrator.py）
- 规格 `version` 必须单调递增；变更阶段/数据源经 `refs` 前向不动点传播得到受影响集合。
- `verify` 命令在隔离目录做从零全量影子重算，与持久化增量结果逐特征比对；
  首个不一致阶段/字段抛 `E_DIVERGE`。

## 8. 账本、快照与恢复（ledger.py）
- SQLite 四张表：`stage_batches/feature_stats/error_chains/drift_verdicts`，
  每行带 sha256 内容摘要，账本带整数 `LEDGER_FORMAT_VERSION`。
- 批次为阶段级全有或全无事务；强制终止只可能留下 `status='unknown'` 的未提交批次。
- 快照持久化规格版本、规格摘要、数据源摘要、尺度口径版本与账本游标；
  重开身份不一致 → **整体替换**而非增量合并；快照摘要被篡改 → `E_SNAPSHOT`。
- 恢复仅信任 committed 批次，重算并逐项重放比对，首个不一致抛 `E_REPLAY`。

## 9. 异常注入与业务码
| 注入 | 仅改输入/参数 | 业务码 | 退出码 |
|---|---|---|---|
| `data_corrupt` | 翻转 `.col` 缓存字节致摘要不符 | E_DATA | 11 |
| `param_overflow` | 运行期 `sd=0` 越界 | E_PARAM | 12 |
| `snapshot_digest` | 篡改快照 digest 字段 | E_SNAPSHOT | 15 |

另有 E_SPEC(10) 规格非法、E_DIVERGE(13)、E_REPLAY(14)。所有事件写入
`events.log.jsonl` 结构化日志（规范 JSON、单调 seq）。

## 10. CLI
`init-sample | validate | run | recover | crash | verify | inject`，
见 `python run_demo.py --help` 或 `python -m fppdiag.cli --help`。
