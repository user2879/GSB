# 召回分片合并与重排工作台

面向搜索/推荐候选集的多路召回分片工作台。项目只依赖 Python 3.11+ 标准库，支持确定性样本生成、分片滚动更新、跨分片合并、通道内去重、定点融合、窗口多样性重排、内容指纹缓存、重算预算、阶段快照与增量/全量一致性校验。

## 快速开始

```bash
python3 tools/generate_samples.py

# 增量滚动：step 0 为初始分片集，再依次应用计划 step 1..4
python3 -m workbench roll --samples samples --output runs/incremental

# 以最终分片集合和最终配置做一次空白全量重建
python3 -m workbench rebuild --samples samples --output runs/full --step 4

# 逐字段比较业务规范件
python3 -m workbench compare \
  --left runs/incremental/canonical.json \
  --right runs/full/canonical.json \
  --json-out runs/comparison.json

# 一键执行全部验收场景并生成 demo/verification_report.json
python3 tools/verify.py

# 标准库测试
python3 -m unittest discover -s tests -v
```

常用单步命令：

```bash
python3 -m workbench run --samples samples --output runs/incremental --step 0
python3 -m workbench run --samples samples --output runs/incremental --step 1
python3 -m workbench run --samples samples --output runs/incremental --step 2 --serial
python3 -m workbench run --samples samples --output runs/budget --step 4 --budget 3
```

## 目录

- `workbench/loader.py`：计划物化、JSONL 分片解析、配置/属性/覆盖区间/维度校验。
- `workbench/merge.py`：全局时间轴扫描、通道内权威来源与去重、跨通道保留。
- `workbench/scoring.py`：通道归一化、定点量化、融合打分、预算调度。
- `workbench/rerank.py`：窗口内贪心多样性重排、相似度与折减审计。
- `workbench/engine.py`：阶段缓存、指纹失效、快照提交、报告与规范件。
- `workbench/cli.py`：`run`、`roll`、`rebuild`、`compare` 命令。
- `tools/generate_samples.py`：固定种子样本与滚动/注入配置生成器。
- `tools/verify.py`：增量/全量、并发/串行、预算、陈旧快照的一键验收。
- `tests/test_workbench.py`：8 个端到端与边界测试。
- `samples/`：确定性输入；`demo/`：实际验收输出。

## 输入格式

### 候选分片 JSONL

每个文件为一个或多个逻辑分片；每行含：

```json
{
  "request_id": "R001",
  "channel": "lexical",
  "item_id": "I001",
  "score": 0.95,
  "features": [0.1, -0.2, 0.3, 0.4],
  "shard_seq": 1,
  "coverage_start_ms": 100,
  "coverage_end_ms": 200
}
```

校验规则：

- `request_id`、`channel`、`item_id` 非空；`shard_seq` 非负且在通道内唯一。
- 起止毫秒为整数，满足 `start < end`，且均整除 `timeline_alignment_ms`。
- 同通道分片覆盖区间不得重叠；分片内所有行必须声明相同区间。
- `score` 与稠密特征均为有限数，特征长度等于 `feature_dim`。
- 任一非法行或非法分片整体输出 `E_SHARD_INVALID`，进入排序稳定的失败清单，且不参与后续阶段。

### 重排配置

`samples/rerank_config.json` 包含通道权重、归一化方式、融合温度/偏置/评分区间、相似度阈值、窗口大小、多样性折减系数、定点小数位宽、量化档位、重算预算、线程池并发数、快照提交间隔等。非法配置返回 `E_CONFIG_INVALID`。

### 物品属性与滚动计划

`items.json` 为物品 ID 到 `{category, price_band}` 的映射。`roll_plan.json` 的步骤支持：

- `append`：追加分片。
- `replace`：按通道与覆盖区间移除旧分片并加入新分片。
- `revoke`：按通道和 `shard_seq` 撤销分片。
- `update_config`：深合并配置补丁。

`injections.json` 声明注入项；报告中的 `injection_hits[].id` 给出命中编号。样本覆盖非法分片、裁剪边界、融合并列、预算待办、相似度阈值等于边界、折减和快照陈旧探针。

## 确定性算法

### 合并与权威来源

分片先按 `(channel, coverage_start_ms, coverage_end_ms, shard_seq)` 串行扫描。同一 `(request_id, channel, item_id)` 内按以下顺序选权威来源：

1. 通道内得分降序；
2. 分片序号降序；
3. 覆盖起点降序；
4. 行号降序。

非权威来源写入 `overridden`，并记录是否与权威来源区间重叠。每条合并记录保留 `occurrences_count`、`max_score`、权威特征、引用分片集合和通道集合。跨通道相同物品保留为不同合并记录，不互相去重。最终顺序与按通道名、覆盖起点串行扫描一致。

### 归一化与定点融合

归一化按请求和通道独立执行：

- `minmax`：`(x-min)/(max-min)`；分母为零时全置零并记录 `denominator_zero`。
- `max`：`x/max`；零或负分母固定置零并记录。
- `none`：裁剪到 `[0,1]`。

归一化值先量化到 `[0, quantization_levels-1]`，再映射到 `10^fixed_decimal_places` 定点整数。所有舍入采用半远离零规则，不使用环境相关浮点格式。融合按通道名字典序固定求和：

```text
raw_fixed = round_div(sum(weight_fixed * normalized_fixed), temperature_fixed) + bias_fixed
fusion_fixed = clamp(raw_fixed, score_min_fixed, score_max_fixed)
```

融合分恰好位于裁剪边界、被裁剪、或多候选融合分相等时，报告分别记录 `fusion_score_exact_clip_boundary`、`fusion_score_clipped`、`fusion_score_tie_item_id_order`；并列时物品 ID 字典序优先。

### 窗口多样性重排

融合分按 `(-fusion_fixed, item_id)` 排序后切成固定大小窗口，仅在窗口内重排。贪心选择时，计算候选与已选集合的最大相似度：

```text
similarity = category_weight * [category equal] + price_band_weight * [price_band equal]
```

严格大于阈值时，以 `score * diversity_discount_factor` 的定点整数结果扣减并重新插入。每条折减记录候选、参照候选、相似度、扣减前后得分与扣减量。相似度恰等于阈值不折减，记录 `similarity_equal_threshold_no_discount`；折减后同分按物品 ID 决胜并记录边界判据。

### 内容指纹与缓存

各阶段缓存键分别派生自：

- 合并：分片内容指纹、覆盖参数、请求/通道/物品。
- 归一化：相关合并键、归一化配置指纹。
- 融合：通道归一化键、通道权重/温度/偏置/评分区间指纹。
- 重排：请求内融合键顺序、窗口参数指纹、物品属性表指纹。

分片变化只会改变引用它的合并键，并沿归一化、融合、重排依赖向下失效。不相关候选直接命中已有阶段产物；报告逐阶段给出 `invalidated_keys`、`reused_keys`、`added_keys` 与计算计数。

## 预算、快照与一致性约定

### 重算预算

每个缺失的融合候选消耗 1 个重算点。调度顺序为：上一轮待办优先，再按 `(request_id, item_id)` 字典序。超预算候选保留为 `E_RECOMPUTE_BUDGET_PENDING`；下一轮优先消费。每轮扣减序列、剩余预算和待办数写入 `budget.rounds` 审计。低预算分多轮完成的候选，其融合定点结果与高预算一次完成完全相同。

### 阶段快照

对齐、合并、归一化、融合、重排分别写 `alignment.json`、`merge.json`、`fusion.json`、`rerank.json`；另有 `manifest.json`、`state.json`、`cache.json`、`canonical.json`。所有文件先写入 `<output>/.tmp/snapshot-*`，再整体替换 `<output>/live`。启动时比对：

- schema 版本；
- 基础样本指纹；
- 已应用计划前缀指纹；
- 配置哈希；
- 各阶段版本与内容哈希。

任一不一致返回 `E_SNAPSHOT_STALE` 并写 `error_report.json`，不覆盖已有快照。阶段内容未变化时复用版本号，变化时仅递增对应阶段版本，避免逐文件盲目覆盖。

### 增量与全量等价边界

`canonical.json` 是业务一致性契约，包含：完整分片集、全局时间轴、合并记录、归一化、融合记录、重排前后顺序、折减记录、边界样本、待办、失败清单与当前注入命中。它不包含运行模式、全局版本、运行期缓存命中列表或累计预算历史，因为这些必然依赖执行路径。

因此验收比较的是所有确定性业务字段和失败顺序；运行期 `report.cache` 的失效/复用遥测单独审计，用于证明未受影响阶段被复用。缓存命中时直接读取阶段产物，且测试同时验证“增量最终规范件 == 空白全量最终规范件”与“并发线程池 == 串行”。

## 输出报告

`report.json` 主要包括：

- `summary`：请求数、唯一候选数、失败数、缓存失效/复用/计算计数、当前轮预算。
- `requests[]`：逐请求候选数、通道贡献分解、合并记录、融合候选、重排前后对比。
- `timeline`：配置对齐粒度及每通道分片覆盖轴。
- `discount_records`：每处折减的候选、参照、扣减量。
- `boundary_samples`：裁剪边界、融合并列、相似度阈值相等、重排同分等判据。
- `failure_list`：按来源、分片、行号稳定排序的 `E_SHARD_INVALID` 等失败。
- `cache`：逐阶段失效/复用键清单。
- `budget.rounds`：每轮预算扣减序列和待办。
- `injection_hits`：当前及历史注入命中编号与证据。
- `snapshots.stage_versions`：阶段版本、内容哈希、本次是否变化。

## 实际验证结果

验证日期：2026-11-21；环境：Linux/WSL 本地，Python `3.11.0rc1`，无网络访问。

```text
python3 -m unittest discover -s tests -v
Ran 8 tests ... OK

python3 tools/verify.py
ok: true
```

样本结果：

- 21 个唯一候选，4 个请求，最终 8 个有效分片，1 个注入非法分片。
- 增量 step 0→4 的最终规范件哈希与空白全量重建一致。
- 线程池 4 并发与 `--serial` 的规范件哈希一致。
- 预算 3/轮时，待办数从 18 逐轮降到 0，共 7 轮；最终规范件与高预算全量一致。
- 陈旧快照探针观察到 `E_SNAPSHOT_STALE`。
- 边界命中：裁剪边界 5、融合并列 2、相似度等于阈值 1；折减记录 3 条。
- 注入命中：`INJ-001`、`INJ-002`、`INJ-003`、`INJ-005`、`INJ-006`。

精确哈希和逐轮扣减序列见 `demo/verification_report.json`；最终业务报告见 `demo/incremental/report.json`、`demo/full/report.json` 和同名 `canonical.json`。

## Windows 路径说明

所有路径均通过 `pathlib.Path` 处理；目录提交使用 `os.replace`。若目标目录已存在，先把旧目录改名为备份，再整体替换新目录，失败时回滚。输出目录必须位于当前工作区内，且不能直接是工作区根目录。
