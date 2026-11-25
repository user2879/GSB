# 技术说明：定点观测流水线原型

## 结构

| 模块 | 职责 |
| --- | --- |
| `obs_pipeline/errors.py` | `E_SPEC` / `E_FIELD` / `E_VERSION` 三类错误，携带阶段标识、字段名、行号 |
| `obs_pipeline/spec.py` | 规格 JSON 加载与强校验（需求 1），拓扑排序 |
| `obs_pipeline/records.py` | v1/v2/v3 编解码与归一（需求 2），定点量化口径 |
| `obs_pipeline/gen.py` | 整数种子确定性样本生成、空值位图、乱序到达（需求 3） |
| `obs_pipeline/stages.py` | 四类阶段整数定点变换与显式截断口径（需求 4） |
| `obs_pipeline/error_model.py` | 沿依赖链的误差上界推导与逐指标归因（需求 5） |
| `obs_pipeline/drift.py` | 参考/观测窗口统计与三分类漂移判定（需求 6） |
| `obs_pipeline/scheduler.py` | 全局预算 + 阶段配额双约束线程池调度（需求 7） |
| `obs_pipeline/ledger.py` | SQLite 账本、事务批次提交、快照整体替换、强杀恢复（需求 8） |
| `obs_pipeline/pipeline.py` | 编排：执行、失效传播、仅失效子集重算 |
| `demo.py` / `tools/crash_writer.py` | 端到端演示与强制终止模拟 |

## 关键口径（显式声明）

- **定点量化**：`Decimal(value) * scale`，`ROUND_HALF_EVEN` 取整，单值量化误差界为记录 scale 的半 ulp。
- **阶段截断口径**（`stages.SEMANTICS`）：标准差分无除法、int64 饱和、负值保留；滑窗均摊 floor 除法（向负无穷）、运行和 int64 饱和；分位桶归并 floor 除法、桶下标钳制 `[0, buckets-1]`；阈值归一 floor 除法、输出钳制 `[0, scale]`、负值饱和到 0。
- **空值规则**（`stages.NULL_RULES`）：差分传播（前驱或自身为空则空，窗口首记录为空）；均摊跳过（空值不计入和与计数，全空窗口才输出空）；分位桶与阈值归一传播。
- **误差合成**（`error_model`）：差分 `e→2e`；均摊 `e→e+1+e/w`（截断 1 ulp + 滑窗边界归属项 `e/w`）；分位桶 `e→e·b/(hi−lo)+1/2`（桶边界归属半桶）；阈值归一 `e→e·scale/span+1`。全部用 `Fraction` 精确运算，归因份额 = 各阶段新增误差（换算到末级 ulp）/ 末级累计界；主导阶段 = 份额最大且 ≥ `dominance_threshold_ppm`（显式阈值，默认 400000 ppm）。
- **漂移判定**：`drift−bound > 阈值` → 确超阈值；`drift+bound ≤ 阈值` → 确未超阈值；否则为误差不可区分，按稳定决胜规则（裸漂移量 > 阈值则判超）给出结论并置 `error_affected=True`。聚合顺序声明为 `obs_seq-ascending`。
- **调度**：任务按 `(stage_order, window_order)` 确定序派发；在途点数受全局预算与阶段配额双重约束，超限排队并以单调时钟毫秒记录等待起止；任务点数超过预算/配额直接报错而非死等。
- **账本**：批次行先以 `committed=0` 落库，阶段摘要、逐指标统计、误差归因、进度游标与 `committed=1` 在同一事务提交，携带整数 `FORMAT_VERSION` 与内容摘要。快照标识（spec_version + 内容摘要）不一致时当前状态整体替换；`error_attributions` 按 `(metric, spec_version)` 键保留旧版本对照，不属于被替换的快照状态。强杀后未提交批次分类为 unknown 并要求重跑，不生成退出报告；恢复重放用 `verify_stage_summaries` 定位首个不一致的 (stage, window, field)。

## 实测验证结果（本机，Python 3.11）

- `python3 -m pytest`：**48 passed**（覆盖需求 1–8）。
- `python3 demo.py`：`DEMO OK`。要点：
  - 三格式归一：v1 补齐 `{source_id, timestamp, metric_key, scale}`，v2 补齐 `{metric_key, scale}`，v3 无补齐，均带补齐依据。
  - 误差归因（seed=42 样本）：三个指标的主导阶段均为 `s_bucket`（分位桶归并），份额约 98.8%（如 m0 为 200000/202481），主导阈值 400000 ppm 显式输出——**定点误差在分位桶归并一级被放大**（桶边界归属半桶误差在末级 ulp 下占绝对主导）。
  - 规格演进（`s_avg` 窗口 4→6、scale 变更，spec_version 1→2）：失效传播仅命中 `s_avg/s_bucket/s_norm`，`s_diff` 五个窗口摘要逐字节不变；旧版本误差界保留并对照输出（m0：202481/2400 → 1518607/18000 ulp）。
  - 强制终止恢复：崩溃子进程（exit 137）留下 unknown 批次，重放校验 `mismatch=None`（逐字节一致）。
