# 设计说明

## 1. 模块边界

| 模块 | 职责 |
| --- | --- |
| `search_proto/config.py` | 配置加载、枚举/正数/正整数校验、配置指纹 |
| `search_proto/loader.py` | 文档/查询 JSONL、词表、停用词、编辑命令逐行校验 |
| `search_proto/index.py` | 正排/倒排构建、停用词排除、并发文档统计、词项引用 |
| `search_proto/fixed.py` | 定点量化、IDF、长度归一、饱和、阶段误差 |
| `search_proto/scoring.py` | 查询-文档定点与浮点参考打分、反事实阶段分数 |
| `search_proto/ranking.py` | TopK、并列、边界清单、次序翻转与归因 |
| `search_proto/engine.py` | 增量索引、缓存账本、批次预算、线程池调度、状态序列化 |
| `search_proto/snapshot.py` | 临时目录落盘、版本目录、原子指针、校验与恢复 |
| `search_proto/report.py` | 阶段产物、报告与增量/全量投影比较 |
| `search_proto/generator.py` | 固定整数种子的本地合成样本 |
| `search_proto/cli.py` | `generate/validate/index/run` 命令 |

## 2. 输入校验与失败清单

所有 JSONL 文件逐行读取：

- 空行、JSON 解析失败、非对象行均产生失败记录。
- 文档标识重复、查询标识重复均保留首个合法项并记录重复失败。
- 文档版本必须是非负整数（布尔值不算整数）。
- 字段名仅允许 `title/body`，且必须属于配置字段。
- 文档与查询词项必须属于受控词表。
- 查询字段偏好必须非空且属于配置字段。
- 字段权重、`k1`、`b` 必须有限且大于零。
- 定点位宽与量化档位必须命中允许枚举。
- TopK、并发数、批次预算、单查询配额必须为正整数。
- 编辑命令在加载阶段校验结构，在应用阶段校验目标文档是否存在。

业务失败统一使用 `E_INDEX_INPUT`；最终失败清单按 `(source, line, entity, code, detail, message)` 稳定排序并去重。

## 3. 确定性索引

文档统计按 `doc_id` 升序提交到线程池，任务键为 `document:<doc_id>:field_stats`。合并时仍然按 `doc_id` 升序、`title` 后 `body`、字段内位置升序。

每个字段统计包含：

- 有效字段长度（移除停用词和文档标记后）；
- 被排除位置、词项与原因（`stopword_table` / `document_mark`）；
- 每个词项的词频与原始位置列表。

倒排记录 `term -> doc -> field` 保存 `tf`、`positions`、`field_length`、文档版本、内容指纹与字段统计指纹。词项文档频率和平均字段长度仅由这些倒排记录派生。

索引指纹由内容指纹、语义配置指纹、停用词指纹与 schema 版本共同派生。并发观测（实际活动线程数、任务键）不进入确定性索引产物。

## 4. 定点打分

设 `S = 10**decimal_places`，所有整数都以 `1/S` 为单位，`q` 是量化档位（同样以 `1/S` 为单位），舍入采用半值远离零。

固定顺序为：

```text
for term in sorted(unique(query.terms)):
    idf_fixed = quantize(ln(...) * S)
    for field in ("title", "body") in configured/preference order:
        length_ratio_fixed = quantize(dl * S * S / avgdl_fixed)
        length_norm_fixed = quantize(b_fixed * length_ratio_fixed / S)
        denominator = tf*S + quantize(k1_fixed * (S - b_fixed + length_norm) / S)
        saturation = quantize(tf * (k1_fixed + S) * S / denominator)
        contribution = quantize(weight_fixed * idf_fixed * saturation / S / S)
        total += contribution
```

每一步使用整数/Python `Fraction` 运算；IDF 先用精度 60 的 Decimal 计算自然对数，再量化。浮点参考路径独立使用同一公式的 IEEE-754 实现，只作为参考，不参与定点 TopK 判定。

每个字段保留所有原分量、定点值、浮点值和单元误差：

- `inverse_document_frequency`
- `field_length_normalization`
- `term_frequency_saturation`
- `field_weight`
- `accumulation`

首次绝对值超过配置容差的环节写入得分与边界文档。另计算四类单阶段量化反事实分数，用于把排序翻转归因到具体环节。

## 5. TopK、边界与翻转

固定排序键：`(-fixed_score_units, doc_id, doc_version)`。浮点参考也使用同样的并列键，仅把首项换成浮点分。

若文档数为 `N`：

- `N >= K` 时，边界分是第 K 名的定点分；
- `N < K` 时，边界分是最低分，并标注 `no_truncation_minimum`；
- `2 * |doc_fixed - boundary_fixed| <= quantum` 的文档均进入边界清单，包括 TopK 之外的近边界候选。

边界文档保存定点分、浮点参考、边界差、定点/浮点位置、是否在 TopK、阶段误差、首次超容差环节与翻转原因。

成对翻转先比较定点关系和浮点关系。若两文档定点分相等，原因是 `tie_break_rule`；否则依次检查饱和、字段长度、IDF、字段权重的单阶段反事实排序，若都无法解释则记为 `accumulation_quantization`。

## 6. 增量缓存与精确失效

### 文档级

缓存键包含文档内容指纹、配置字段集合和停用词表版本。文档替换只重建该文档；未命中停用词或文档标记的文档不重算内容，只按新命名空间重派生缓存键并记为 `reused`。

### 词项级

缓存键包含该词项全部倒排记录（文档 ID、版本、内容指纹、字段统计指纹、词频、位置、字段长度）。文档替换只影响旧/新文档涉及的词项；停用词修改只影响实际包含该词项的文档及其词项。

### 查询-文档得分级

缓存键只包含本文档实际命中的查询词项统计，不使用整个语料所有词项。因此一个无关词项的 DF 变化不会使本文档得分失效。零贡献文档不携带内容指纹，仅保留 ID、版本、定点与配置命名空间。

### 排序级

缓存键包含查询指纹、全部单文档得分键、TopK、位宽/量化与容差。只有得分集合或排序配置变化才使排序结论失效。

账本每个实体只保留最新决策，避免历史重复记录影响最终审计。增量运行先建立基线缓存，再施加编辑，因此可观察到真实复用。全量运行从空缓存开始，决策必然全部为 `rebuilt`。因此一致性对照比较活动缓存实体、最终缓存键与结果指纹；执行时状态不属于最终语义。

## 7. 调度

每个查询维护：

- 剩余文档 ID 列表；
- 已消费点数；
- 当前得分与得分缓存键。

每批按“待办查询优先，再按 query_id”的顺序处理。文档块大小取以下三者最小值：

```text
(单查询剩余配额 // 查询词数)
(本批剩余预算 // 查询词数)
剩余文档数
```

批次预算耗尽时保留进度并写 `E_POINT_BUDGET_PENDING`。无法再满足单查询配额时写 `E_QUERY_BUDGET_EXCEEDED`。若批次预算小于单个文档所需点数，产生零进展，调度器标记 `scheduler_stalled` 并退出，避免无限空转。

文档统计与查询打分均通过 `ThreadPoolExecutor` 执行；实际并发数取配置且不允许超过。每个任务键为 `document:<id>:field_stats` 或 `query:<id>:score:<doc_id>`，结果按输入稳定顺序合并。

## 8. 快照与恢复

一次提交包含：

```text
manifest.json
state.json
index.json
term_stats.json
scores.json
ranking.json
failures.json
cache_ledger.jsonl
report.json
```

流程：

1. 全部文件写入 `root/.tmp/stage-<uuid>/`。
2. 写入每个文件 SHA-256 与状态指纹。
3. `fsync` 文件与目录。
4. 将临时目录原子改名为 `versions/vNNNNNNNN/`。
5. 旧 `CURRENT` 写入 `LAST_GOOD`。
6. 原子替换 `CURRENT` 指针。
7. 保留最近若干版本。

恢复时校验指针、清单 schema、全部产物校验和、状态 JSON 指纹。`CURRENT` 损坏时回退 `LAST_GOOD`。续跑判定同时检查源文件指纹、最终配置哈希、停用词版本和可选全局版本；任何不匹配即重建。

## 9. 一致性投影

`report.comparison_projection()` 包含：

- 索引/内容/配置/停用词指纹；
- 查询-词项、词项-文档依赖；
- 全部单文档得分（定点、浮点、中间量、误差）；
- 全部查询排序、边界与翻转证据；
- 稳定失败清单；
- 活动缓存实体的最终键和结果指纹；
- 完成/中止/待办查询集合。

该投影不包含并发观测、全局版本或执行时缓存命中状态，因此适用于增量/全量与并发/串行的语义等价核对。

