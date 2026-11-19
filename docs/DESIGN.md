# 技术设计说明

## 1. 数据流

```text
JSON spec ──强校验──> normalized Spec
                     │
                     ├── dependency graph ── topo / cycle / refs
                     │
local seed/file ─────┴── one row traversal per source
                               │
                               ├── null fill
                               ├── Decimal fixed-point quantization
                               ├── bisect_right bucket id
                               └── counts / null counts / error bound
                                        │
                                        ├── ordered pack plans
                                        │       │
                                        │       └── global budget + per-source quota thread pool
                                        │
                                        ├── per-pack deterministic summaries
                                        ├── additive merge → source/global stats
                                        └── SQLite ledger / recovery
```

## 2. 规范化与确定性

- 规格中的数据源、规则、统计按 ID 排序；列按列名排序；边界保留升序数组。
- JSON 输出统一 `sort_keys=True`、紧凑分隔符、ASCII 转义，禁止 NaN/Infinity。
- 对象摘要使用 SHA-256；桶版本、包摘要和账本内容摘要均基于规范化 JSON。
- 调度时钟是唯一非确定性输出，位于独立 `schedule` 区域，不进入内容摘要。
- 包摘要中的桶版本是包行切片版本，保证未受影响包逐字节稳定。

## 3. 单趟展开

每个数据源按行迭代一次。对每行按拓扑顺序计算该数据源所有桶规则：

1. 源列输入：若空值位图为 1，使用 `null_fill`；否则使用原始值。
2. 桶列输入：使用上游桶 id，并继承上游空值位。
3. `quantize` 生成 int64 定点值和误差。
4. `bisect_right` 计算桶号。
5. 同一次行访问内更新桶计数、空值计数、饱和计数和最大误差。

增量时仍按一次行遍历处理需重算规则，未失效桶列直接复用已有元组。

## 4. 失效传播

依赖图的边为 `上游桶列 → 下游桶规则`。统计引用作为终止引用参与删除校验，但不形成桶列环。

触发集合：

- 值变更触发所有直接读取该源列的规则；
- 空值调整同上；
- 边界调整触发规则本身；
- 规则删除触发被删规则。

沿 `children` 做稳定堆序 BFS 得到全部下游。受影响包：

- 行级变更：仅包含变更行的既有包；
- 边界变化：该数据源所有包；
- 删除导致行宽/规则集合变化：该数据源全部重新切包；
- 未涉及的列变更：可能只更新数据源摘要，不重算任何桶列或包。

## 5. 增量与全量比对

`deterministic_view(result)` 去除 `schedule`、`recompute` 和自身 `content_digest` 后做规范化比较。比较范围包含：

- 图节点/边/拓扑；
- 每源桶值、空值位、边界、量化误差和整列桶版本；
- 每包行区间、桶计数、空值计数、列宽、成本、切片桶版本、内容摘要和包摘要；
- 合并统计、直接统计、派生统计和成本账本；
- 口径策略。

不一致抛 `E_DIVERGE`，并尝试给出首个差异包和桶列。

## 6. 账本状态机

```text
RUNNING ──commit pack*──> RUNNING ──all expected committed──> COMPLETE
   │                          │
   └── crash ────────────────┴── recover: missing → UNKNOWN → rerun → COMPLETE

format/content mismatch ──> replace whole ledger ──> E_REPLAY
```

表：

- `ledger_meta`：格式版本、批次/整账本内容摘要；
- `batches`：批次格式、规格摘要、快照摘要、状态；
- `expected_packs`：批次包清单与期望摘要；
- `packs`：逐包摘要、宽度、成本、时钟、内容摘要、`COMMITTED/UNKNOWN`。

SQLite 使用 `BEGIN IMMEDIATE`、`synchronous=FULL`、WAL（文件模式）和 busy timeout。每包插入与批次摘要更新处于同一事务。

## 7. 成本口径

`width_used` 为行数 × 每行固定槽位宽度。成本为：

```text
cost_micros = width_used × unit_cost_micros
```

账本记录每包成本与总计；`within_budget` 比较 `budget_micros`。样例预算明显高于实际成本，验证报告中同时输出可核验总计。

## 8. GUI 与无桌面约束

Tk 相关导入仅在 `bucketpack/gui.py`。`bucketpack/view.py` 不依赖 Tk，可独立测试视图模型。无桌面时 `python -m bucketpack gui` 返回 `E_GUI`，其他命令行功能不受影响。
