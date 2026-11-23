# 带状定点增量 DTW 与序列对齐诊断

本项目从零实现一个本地、确定性、可复现的 DTW 诊断组件。它在 Sakoe-Chiba 带状可行域内做定点整数对齐，输出路径、总代价、逐单元回溯记录、分段代价，并在每批编辑后只复用未失效的持久化单元格。

## 快速开始

```bash
python3 -m pip install -r requirements.txt
python3 scripts/generate_samples.py
python3 -m dtw_diag demo
python3 -m pytest -q
```

`demo` 会完成增量编辑、逐批快照重载、最终全量重建对照、串行/并发对照，并把报告写到 `artifacts/demo/report.json`。本次实际运行结果见 `artifacts/demo/VALIDATION.md`。

## 项目结构

```text
dtw_diag/                 核心包
  config.py               YAML 配置解析与业务校验
  storage.py              JSONL 序列库、编辑计划校验
  band.py                 带状可行域
  quantization.py         确定性定点量化与残差
  align.py                定点 DP、回溯、路径长度策略、分段
  invalidation.py         单元格级精确失效规划
  engine.py               批次调度、预算、线程池、快照、全量重建
  cli.py                  demo/run/generate 命令
samples/                  固定种子生成的样本
scripts/generate_samples.py
tests/                    15 项自动化测试
docs/technical.md         设计与一致性证明
artifacts/demo/           实际验证报告、账本与最终结果
```

## 输入契约

YAML 配置除任务要求的字段外，增加了两项完成完整 DTW 所需的显式参数：

```yaml
reference_sequence_id: S00   # 库内所有序列都与该参考序列对齐；缺省时自对齐
path_length_policy: reject   # reject | truncate
```

核心字段：

- `step_mode`: `symmetric` 或 `asymmetric`
- `bandwidth_permille`: `0..1000` 整数
- `fixed_point.fraction_bits`: 定点小数位宽，`1..30`
- `fixed_point.quantization_step`: 以最小定点单位计的正整数档位，且不超过 `2**fraction_bits`
- `cell_recompute_budget`: 每批最多重算的单元格数
- `thread_pool_workers`: 线程池上限
- `snapshot_interval_batches`: 不可变归档快照间隔；`latest.json` 每批原子提交

序列库为 JSONL，每行：

```json
{"id":"S01","observations":[0.1,null,0.3],"timestamp_interval":11,"segment_ids":[0,0,1]}
```

`null` 或配置的 `missing_marker` 表示缺失点。编辑计划支持：

- `sequence_replace`
- `point_update` / `point_correction`
- `sequence_append`
- `bandwidth_update`
- `step_mode_update` / `step_mode_switch`

## 算法约定

### 带状可行域

```text
radius = ceil(max(n, m) * bandwidth_permille / 1000)
center_i = round_half_up(i * (m - 1) / (n - 1))
row_i = [max(0, center_i - radius), min(m - 1, center_i + radius)]
```

首行首列与末行末列被强制包含；单行/单列输入扩展到可连通。`RowBand.empty` 可显式表示空区间。结果中同时输出每行区间、可行单元格数、带外单元格数及 `out_of_band_cells` 清单。

### 定点量化

```text
fixed_units = round_half_up(value * 2**fraction_bits / quantization_step)
              * quantization_step
residual    = value - fixed_units / 2**fraction_bits
```

量化使用 `Decimal(str(value))` 与 `ROUND_HALF_UP`，同一配置下逐点可重复；残差以精确十进制字符串输出。

### DP 与前驱优先级

- 局部距离：`distance_coefficient * abs(fixed_row - fixed_col)`；任一点缺失时使用 `missing_penalty`。
- 对称模式前驱顺序固定为 `DIAG, UP, LEFT`。
- 非对称模式固定为两条前驱：行数不少于列数时 `DIAG, UP`，否则 `DIAG, LEFT`。
- 严格使用 `<` 更新最小值，因此并列时选择固定顺序中的第一条。
- 起点代价为该单元局部代价；不可达单元保留 `reachable: false`。

路径超限时，`reject` 返回 `E_PATH_LENGTH_REJECTED`；`truncate` 取前 `max_path_length` 个单元并记录 `W_PATH_TRUNCATED`。

### 分段汇总

沿路径把连续相同的 `(query_segment_id, reference_segment_id)` 切为一段。每段输出路径起止索引、行列边界、段内单元数和局部代价；求和显式按 `(row, col)` 升序执行。

### 内容指纹与精确失效

缓存链从下到上为：

1. 每点内容键：原值、量化整数、残差、档位、位宽、缺失状态。
2. 每单元 `local_key`：两侧点内容键、距离系数、缺失罚分。
3. 每单元 `cache_key`：单元坐标、`local_key`、行带/步进键、前驱内容键。
4. 每单元 `content_key`：代价、回溯指针、前驱、局部内容。
5. 序列结果键：两条记录、两侧量化内容、带指纹、对齐参数、路径长度策略。
6. `path_key` 与 `segment_key`：分别从单元内容清单/路径及分段汇总派生。

失效规划先标记引用变动点、带外/新可行单元、前驱集合变化或行带键变化的单元，再沿新可行图的后续单元传播。可复用单元直接来自磁盘快照中的 `CellProduct`，不会重算。

### 预算与并发不变性

主线程先按“上批待办序列优先，再按序列 ID”的稳定顺序分配预算，随后只把互不依赖的序列任务提交给线程池。每个任务内部按 `(row, col)` 顺序消费已冻结的配额。因此线程调度顺序、并发度和预算耗尽时点都不会改变：

- 序列结果顺序与路径
- 逐单元回溯顺序
- 失败清单顺序
- 预算、失效和复用账本

## 输出与业务码

主要产物：

- `incremental_result.json`: 最终增量结果
- `full_rebuild_result.json`: 从空缓存全量重建的参考结果
- `equivalence.json`: 逐字段等价结论与两侧指纹
- `budget_ledger.json`: 每批冻结预算、消耗、待办余量与逐序列分配
- `invalidation_ledger.json`: 每批失效原因、移除单元、复用数
- `reuse_ledger.json`: 逐序列复用/重算统计
- `failure_list.json`、`edit_failure_ledger.json`: 活动异常与编辑历史异常
- `incremental_state/snapshots/latest.json`: 最近原子快照；按间隔另存归档快照

主要业务码：

`E_STEP_MODE`、`E_BANDWIDTH`、`E_QUANT_LEVEL`、`E_MAX_PATH_LENGTH`、`E_DUPLICATE_ID`、`E_EMPTY_SEQUENCE`、`E_OBS_VALUE`、`E_TIMESTAMP_INTERVAL`、`E_SEGMENT_LENGTH`、`E_INDEX_RANGE`、`E_SEQUENCE_NOT_FOUND`、`E_NO_PATH`、`E_PATH_LENGTH_REJECTED`、`E_CELL_BUDGET_PENDING`。

活动失败按 `(sequence_id, code, stage, 位置)` 稳定排序。编辑历史异常单独记录，避免历史失败污染最终状态对照。

## 实测结果

- `python3 -m pytest -q`: `15 passed`
- `python3 -m dtw_diag demo`: `PASS`
- 样本：7 条序列、7 批编辑、合计 11 个执行批次（含预算待办续跑）
- 增量/全量：`sequences`、`failures`、`pending_sequences` 逐字段一致
- 两个最终结果 JSON 字节级一致（SHA-256 `d90b3b50…`）
- 串行/并发：结果、预算账本、失效账本、复用账本一致
- 最终待办：0 条序列、0 个单元格；活动异常：0

详细证明与账本字段说明见 `docs/technical.md`。
