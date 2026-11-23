# 本地确定性搜索精排/重排实验台

这是一个从零实现的离线、确定性搜索精排与重排实验台。它覆盖候选加载校验、定点归一化与融合、窗口多样性重排、内容指纹缓存、线程池预算调度、交互过期语义、阶段化原子快照、故障注入隔离，以及增量/冷启动一致性验证。

核心约束是：**规范报告不包含墙钟时间、绝对路径、线程完成顺序或浮点运算结果**。同一输入在同一 Python 环境中重复运行，报告可逐字节一致；串行和并行、增量缓存复用和冷缓存重算得到相同的逐请求结果。

## 1. 环境与离线安装

要求 Python 3.11+、numpy、PyYAML。依赖版本见 `requirements.txt`：

```text
numpy==1.26.4
PyYAML==6.0.2
```

离线安装示例（先将 wheel 下载到本地 `wheels/`，或配置本地索引）：

```powershell
py -3.11 -m pip install --no-index --find-links .\wheels -r .\requirements.txt
py scripts\check_env.py
```

```bash
python3.11 -m pip install --no-index --find-links ./wheels -r requirements.txt
python3 scripts/check_env.py
```

`check_env.py` 只读取本地已安装包，不访问网络。

## 2. 快速运行

生成固定种子样本：

```bash
python scripts/generate_samples.py
```

运行默认样本（`python main.py` 与 `python -m rerank_lab` 等价）：

```bash
python -m rerank_lab \
  --sample-dir samples/default \
  --output-dir out \
  --clean
```

`--clean` 默认开启：运行前删除已知的旧快照、临时提交目录和旧报告，但不会删除输出目录中的其他文件。保留旧输出可显式传入 `--keep-output`。

输出同时写入 `out/report.json` 和标准输出。`--injections` 可指定独立 JSON/YAML 注入清单；如果样本目录中存在 `injections.json`，未显式传参时也会自动加载。`--no-snapshots` 可关闭快照提交，`--report-name` 可改报告文件名。

一次完整验收：

```bash
python scripts/validate_run.py
```

它会跑默认、预算、注入、串并行和低延迟样本，输出并写入 `artifacts/acceptance/acceptance.json`。

自动化测试：

```bash
pytest -q
```

当前本机实际结果：`23 passed`，验收脚本 `status: pass`。

## 3. 样本目录

一个样本目录包含：

```text
candidates.jsonl      候选集，一行一个 JSON 对象
config.json           精排配置
feature_groups.json   特征分组表
commands.jsonl        交互命令流
injections.json       可选独立注入清单
```

### 3.1 候选 JSONL

最小结构：

```json
{"request_id":"r-001","item_id":"c-001","recall_scores":{"bm25":88,"ann":91},"dense_features":[-5,20,10,30],"sparse_features":{"brand":1,"color":3}}
```

支持别名（如 `requestId`、`itemId`、`denseFeatures`、`sparseFeatures`）。校验规则：

- `request_id`、`item_id` 必须是非空字符串；同一请求内 `item_id` 唯一。
- 召回分数、稠密特征必须是有限数。
- 稀疏计数必须是非负整数，布尔值不算整数。
- 特征名由目录统一为 `recall.<key>`、`dense.<index>`、`sparse.<key>`。
- 未声明特征、重复候选、非法数值进入失败清单；合法候选按 `(request_id, item_id)` 排序。

### 3.2 配置

```json
{
  "feature_weights": [0.15, 0.2, 0.4, 0.3, 0.2, 0.1, 0.05, 0.1],
  "normalization": "minmax",
  "fusion_temperature": 1.0,
  "score_clip_interval": [-0.4, 0.75],
  "fixed_decimal_places": 4,
  "quantization_step": 1,
  "rerank_window_size": 4,
  "diversity_penalty": 0.15,
  "concurrency": 3,
  "global_mac_budget": 2000,
  "per_request_quota": 500,
  "cache_residency_budget": 30,
  "snapshot_commit_interval_commands": 2,
  "interaction_latency_budget_ms": 10,
  "min_command_batch": 1,
  "latency_units_per_ms": 1000,
  "similarity_threshold": 0.8
}
```

约束：

- `feature_weights` 长度必须等于特征维度；支持数组或按特征名完整的对象。
- `normalization` 取 `minmax` 或 `zscore`。
- 定点小数位宽、量化档位、重排窗口、并发数、缓存块数、快照间隔、最小命令批为正整数；并发数硬上限为 32。
- 全局 MAC 预算、单请求配额、交互延迟预算为正。
- 融合温度为正；打分区间不低于；多样性惩罚非负。
- 结构性/数值错误使用 `E_INPUT_INVALID`，预算、并发、缓存类错误使用 `E_BUDGET_INVALID`。

### 3.3 特征分组表

```json
{
  "feature_order": ["recall.bm25", "recall.ann", "dense.0", "dense.1"],
  "groups": [
    {"name": "recall", "features": ["recall.bm25", "recall.ann"], "conflict": true},
    {"name": "dense", "features": ["dense.0", "dense.1"], "conflict": false}
  ],
  "similarity_features": ["dense.0", "dense.1"],
  "similarity_threshold": 0.8
}
```

`feature_order` 是固定运算顺序。分组表用于校验维度、标注量纲冲突、提供相似度特征子集，并进入内容指纹。

### 3.4 命令流

支持五类命令，JSONL 或 JSON 数组均可：

```json
{"command":"set_weights","weights":[0.1,0.2]}
{"command":"set_normalization","normalization":"zscore"}
{"command":"set_quantization","fixed_decimal_places":3,"quantization_step":2}
{"command":"set_rerank","rerank_window_size":3,"diversity_penalty":0.2}
{"command":"add_candidate","candidate":{...}}
{"command":"remove_candidate","request_id":"r-001","item_id":"c-002"}
```

别名包括 `modify_weights`、`switch_normalization`、`adjust_quantization`、`modify_rerank` 等。添加已存在候选按覆盖处理；删除不存在候选产生命令失败，不改变状态。版本号按命令序号递增，非法命令也占一个版本并记录失败。

## 4. 精确定点算法

所有输入 JSON 浮点用 `Decimal` 解析，再转换成 `Fraction`。核心路径不使用二进制浮点。

### 4.1 归一化

对每个请求、每个特征单独统计：

- `minmax`: `(x - min) / (max - min)`；分母为 0 时固定置 0，并在 `zero_denominators` 中记录 `zero_denominator_to_zero`。
- `zscore`: `(x - mean) / std`，使用总体方差；标准差用 80 位 Decimal 计算，方差为 0 时固定置 0，并记录 `zero_variance_to_zero`。

设 `S = 10^fixed_decimal_places`、`q = quantization_step`：

```text
fixed_feature = half_away_from_zero(normalized * S / q)
```

### 4.2 融合

```text
weight_fixed = half_away_from_zero(weight * S)
temp_fixed   = half_away_from_zero(temperature * S)

contribution_i = half_away_from_zero(feature_i_fixed * weight_i_fixed / temp_fixed)
raw_score      = sum(contribution_i)          # 严格按 feature_order
score_fixed    = clamp(raw_score, score_min_fixed, score_max_fixed)
```

逐候选输出 `normalized_fixed`、`weight_fixed`、`contribution_fixed`，因此贡献整数和严格等于 `raw_score_fixed`。融合排序键为 `(-score_fixed, item_id)`。

最小最大和 z 分数统计量均在请求内计算，不跨请求共享。

## 5. 窗口多样性重排

1. 按融合得分降序、`item_id` 字典序取前 `rerank_window_size` 个候选。
2. 只为窗口内候选预计算相似度对，乘加成本为 `C(window, 2) * feature_dimension`。
3. 相似度用配置的 `similarity_features` 在定点向量上计算余弦：

```text
dot^2 >= threshold^2 * norm_a * norm_b   # 恰好等于阈值时命中
```

4. 每轮对仍待选、且与任一已选候选过近的候选最多折减一次：

```text
new_score = max(score_min_fixed, old_score - diversity_penalty_fixed)
```

5. 折减后重新插入，按 `(-score_fixed, item_id)` 排序，然后选出当前队首。
6. 记录每轮的 `pending_before`、被折减候选、折减量、命中相似度的已选候选和每次折减缓存键。
7. 窗口外候选保持融合排序追加到末尾。

因此每轮每个候选最多折减一次，轮数等于窗口大小；同分和阈值边界均由整数/精确分数比较决定。

## 6. 缓存键与精确失效

每个缓存键由 `stage + upstream_fingerprint + config_fingerprint + quantization` 的 SHA-256 规范 JSON 派生：

- `normalization`: 请求候选内容、特征顺序、分组、归一化方式、量化参数。
- `fusion`: 规范化键、候选内容指纹、权重、温度、裁剪区间、量化参数。
- `rerank`: 融合键集合、窗口、惩罚、相似度阈值/特征、量化参数。
- `final_sort`: 重排键与稳定排序策略。
- `rerank_reduction`: 每轮折减的候选、已选近邻、轮次和惩罚参数。

每个结果同时记录 `cache_refs`，其中包含 `feature_dimension` 和完整 `candidate_ids`。

失效传播：

| 命令 | 统计量 | 融合 | 重排/最终排序 |
|---|---|---|---|
| 修改权重/温度/区间 | 复用 | 重算 | 重算 |
| 切换归一化 | 重算 | 重算 | 重算 |
| 修改定点位宽/量化档位 | 重算 | 重算 | 重算 |
| 修改窗口/惩罚 | 复用 | 复用 | 重算 |
| 增删某请求候选 | 该请求重算 | 该请求重算 | 该请求重算 |
| 其他请求 | 复用 | 复用 | 复用 |

实际缓存是块级 LRU，容量由 `cache_residency_budget` 控制。协调线程按稳定顺序执行 `get/put`，因此线程完成顺序不会改变淘汰顺序。淘汰键为 `(last_access, block_key)`，同访问序按块键字典序决胜。被淘汰块再次引用时必须重算，结果与首次计算一致。

## 7. 并发、预算和调度

- 使用 `ThreadPoolExecutor(max_workers=concurrency)`，并发数不会超过配置和硬上限 32。
- 统计量、融合候选、重排请求均以稳定键提交；融合任务键由 `(request_id, item_id)` 派生并写入 `schedule.candidate_tasks`。
- 预算按 `request_id` 升序分配，与线程调度无关。
- 单请求成本：

```text
fusion_misses * feature_dimension
+ (rerank 未命中 ? C(min(candidate_count, window), 2) * feature_dimension : 0)
```

  统计量不计 MAC，完全命中的最终结果成本为 0。

- 成本大于 `per_request_quota`：请求中止，`E_REQUEST_BUDGET_EXHAUSTED`。
- 全局剩余预算不足：当前及后续需要计算的请求进入待办，`E_GLOBAL_BUDGET_PENDING`；已完整缓存的请求仍可复用。
- 失败或待办请求不进入已完成统计；失败请求从进度分母中剔除。
- 串行与并行只影响执行方式，不影响排序、任务键、预算分配和输出。

## 8. 交互响应语义

每次命令后生成一个响应摘要，包含完成请求数、当前排序首位、受影响请求、进度和错误码。

为了满足“重复运行逐字段一致”，规范报告不记录墙钟时间。实验台用确定性 MAC 成本模型把工作量换算为毫秒：

```text
estimated_ms = sampled_mac_units / latency_units_per_ms
```

若超过 `interaction_latency_budget_ms`：

- `response.status = "stale"`；
- `error_codes` 包含 `E_RESULT_STALE`；
- 返回最近一次已完成版本的摘要；
- `version_source` 指旧版本，`requested_version` 指当前版本；
- `computed_version` 明确标出后台已计算的版本，绝不冒充当前版本结果。

当请求数少于 `min_command_batch` 时，测量口径标为 `min_command_batch`，并按最小命令批放大采样工作量。报告中的 `measurement.wall_clock_used` 明确为 `false`。

## 9. 阶段快照与恢复判定

阶段为 `validation -> normalization -> fusion -> rerank`。每次提交：

1. 在 `.tmp-snapshot/` 完整写入 `validation.json`、`normalization.json`、`fusion.json`、`rerank.json`、`manifest.json`。
2. 将旧 `snapshot/` 改名为 `.snapshot-old/`。
3. 原子改名新临时目录为 `snapshot/`。
4. 成功后删除备份；失败时回滚。
5. 运行开始时清理临时目录，`--clean` 还会清除旧快照。

清单记录：

- 全局版本号；
- 配置哈希；
- 特征分组表哈希；
- 候选集哈希；
- 已完成请求游标和列表；
- 四阶段各自版本与内容哈希；
- 每请求最终结果缓存键；
- 清单哈希。

提交前比较版本号、配置哈希、分组哈希、候选哈希和游标；完全相同时 `skip_unchanged`，否则整体替换。非 clean 运行时还会输出恢复判定：哈希匹配为 `resume_metadata`，否则为 `resume_incompatible`。

## 10. 故障注入

注入清单：

```json
{
  "injections": [
    {"id":"INJ-001","request_id":"r-002","stage":"load","fault":"non_finite_feature"},
    {"id":"INJ-002","request_id":"r-003","stage":"fusion","fault":"weight_length_mismatch"},
    {"id":"INJ-003","request_id":"r-004","stage":"budget","fault":"quota_below_minimum"}
  ]
}
```

- `non_finite_feature`: 记录非有限稠密特征在打分前被拒绝，产物中给出候选和字段证据。
- `weight_length_mismatch`: 记录期望/注入权重长度，在融合阶段隔离。
- `quota_below_minimum`: 有效配额小于最小融合 MAC 量，在预算阶段隔离。

注入编号写入 `injection_hits`，同时出现在对应失败项。单个请求在任一阶段异常时记录 `request_id + stage + code`，继续处理其他请求；失败请求不进入未完成结果统计。

## 11. 报告要点

`report.json` 顶层字段：

- `summary`: 完成/失败/待办数、进度、首位候选、状态和错误码。
- `requests[]`: 逐候选定点分、贡献分解、融合顺序、重排轮次与折减、最终顺序、缓存键与引用、预算及复用遥测。
- `schedule`: 并发、预算策略、请求计划、融合候选稳定任务键。
- `commands[]`: 每命令版本、受影响请求、缓存复用/重算遥测、交互响应和快照记录。
- `failures[]`: 稳定排序的输入、预算、注入和阶段失败。
- `pending[]`: `E_GLOBAL_BUDGET_PENDING` 待办队列。
- `cache`: 驻留块、累计淘汰、最终复用/重算请求。
- `snapshot`: 恢复判定、提交记录、最新清单。
- `injection_hits[]`: 注入命中证据。
- `consistency`: 增量结果与冷缓存重建投影哈希是否一致。

## 12. 验收结果

本机运行：

```text
python3 scripts/check_env.py       -> status ok, Python 3.11, numpy 1.26.4, PyYAML 6.0.2
pytest -q                          -> 23 passed
python3 scripts/validate_run.py    -> status pass
```

`artifacts/acceptance/acceptance.json` 记录：

- 默认样本状态 `ok`，重复运行报告字节等价，增量/冷重建投影一致，无定点不变量违例。
- 预算样本完成 1 个、待办 3 个，剩余 12 MAC，待办码均为 `E_GLOBAL_BUDGET_PENDING`。
- 注入样本命中 `INJ-001/002/003`，仅完成 1 个请求，3 个失败请求不计入进度分母。
- 并发 1 与并发 8 的请求结果和 MAC 成本完全一致。
- 低延迟样本所有交互响应均为 `stale`，带 `E_RESULT_STALE`，并标注最小命令批口径。

生成样本、验收报告均为 UTF-8 + LF、排序键 JSON，不使用时间戳或绝对路径，可在 Windows 和 Linux 上重复产生相同规范字段。
