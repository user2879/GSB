# 确定性跨语言融合召回工程

本目录是一个可离线运行的 Python 3.11 工程，覆盖样本生成、JSONL/配置/命令校验、双语正排与倒排、定点融合打分、TopK/边界诊断、内容指纹缓存、增量复用、预算调度、并发一致性、阶段快照与崩溃恢复。

## 一键命令

```bash
python3 --version
python3 -m pip install -r requirements.txt

export PYTHONPATH=src
python3 -m fusion_recall.cli doctor
python3 -m fusion_recall.cli generate
python3 -m fusion_recall.cli run-sample
python3 -m fusion_recall.cli verify
python3 -m pytest -q
```

不设置 `PYTHONPATH` 时，可用根目录入口：

```bash
python3 main.py verify
```

也可以执行 `python3 -m pip install -e .` 后直接使用 `fusion-recall ...`。

`run-sample` 会重建示例工作区 `workspace/`，写入：

- `workspace/ledger.json`：完整诊断账本与倒排条目。
- `workspace/cache/`：词项桶、字段词频、融合分、排序结论的持久化 JSON 产物。
- `workspace/staging/`：阶段快照临时目录。
- `workspace/releases/`：不可变快照发布目录。
- `workspace/CURRENT.json`：当前快照指针。

`verify` 会把机器可读验收结果写入 `runs/verification_report.json`。

## 输入模型

- `configs/vocab.json`：固定种子 `360` 生成的 24 个受控概念，每个概念含 `original`（zh-CN）与 `translation`（en-US）词形和别名。
- `configs/pipeline.json`：语言视图、字段权重、TopK、饱和参数、定点位宽、量化档位和预算。
- `samples/documents.jsonl`：12 条有效文档和 5 条故意非法记录。
- `samples/queries.jsonl`：10 条有效查询和 2 条非法查询。
- `samples/commands.jsonl`：14 条有效编辑/配置命令和 2 条非法命令。

文档结构示例：

```json
{
  "doc_id": "D001",
  "version": 2,
  "views": {
    "original": {"language": "zh-CN", "fields": {"title": ["C05"], "body": ["C05", "C08"]}},
    "translation": {"language": "en-US", "fields": {"title": ["index"], "body": ["index", "language"]}}
  }
}
```

查询的 `terms` 可写概念 ID、原语言词形、译语言词形或已登记别名；`language_preference` 支持正整数、小数字符串或 `numerator/denominator` 精确有理数字符串。

## 校验与失败清单

`fusion_recall/loading.py` 逐行解析文档和查询，`fusion_recall/commands.py` 逐行解析编辑流。以下问题均输出 `E_INPUT_INVALID`、保留行号并跳过该条：

- JSON 语法错误或根对象类型错误。
- 重复文档/查询 ID。
- 语言标识不在 `allowed_languages`。
- 词项无法解析到受控词表。
- 文档版本不是非负整数；布尔值不会被误判为整数。
- 查询词项为空。
- 字段权重不是正整数。
- `frac_bits` 不在 `(8, 16, 32)`。
- `quant_tier` 不在 `(floor, nearest, ceil)`。
- 未知操作、未知字段/视图、对不存在文档执行替换/删除等。

样本当前共有 9 条输入/命令失败：5 条文档、2 条查询、2 条命令。

## 确定性索引

`fusion_recall/index.py` 按以下固定顺序构建：

1. 文档 ID 升序。
2. 配置中的语言视图顺序：`original` 然后 `translation`。
3. 字段顺序：`title` 然后 `body`。
4. 词项在字段中的位置升序。

每个桶 ID 为 `<view>:<concept_id>`，载荷包含：

- `term_id`、`surface_term`、`view`。
- `document_frequency` 与升序 `doc_ids`。
- 每篇文档的版本、字段 TF 和位置。
- 由完整 postings 规范化 JSON 派生的 `bucket_fingerprint`。
- 由桶指纹和索引结构指纹派生的 `cache_key`。

## 定点融合公式

设 `Q = 2 ** frac_bits`，量化函数按 `floor`、Half-Up `nearest` 或 `ceil` 做整数除法。

```text
sat_q(tf)  = quant_div(tf * Q, tf + saturation_k)
idf_q      = quant_div(Q * (N + 1), df + 1)
field_q    = quant_div(field_weight * sat_q * idf_q, Q)
semantic_q = Σ field_q                  # 字段配置顺序
lang_q     = quant_div(Q * global_weight_num * preference_num,
                       global_weight_den * preference_den)
contrib_q  = quant_div(semantic_q * lang_q, Q)
fusion_q  += contrib_q
```

累加嵌套顺序固定为查询词项序 → 语言视图固定顺序 → 桶内文档 ID 升序。切换 `frac_bits` 或 `quant_tier` 会改变 `sat_q`、`idf_q`、`lang_q`、`contrib_q` 和最终 `fusion_score_q`，不是显示层换算。

## TopK、并列与边界

排序键唯一定义为：

```text
(-fusion_score_q, doc_id ascending, version ascending)
```

截断到 `top_k`。当前整数定点域的一个 ULP 为 1 q，量化半步为 `1/2 q`；由于分数差是整数，边界组即与截断分数相等的候选。每个边界文档输出：

- 当前分数、截断分数、距离。
- 更高一位允许位宽（8→16，16→32，32→40）的参考分。
- 参考分缩放回当前量程后的值与差值。
- 高位宽下的排名和距参考截断线的差值。

`fusion_recall/diagnostics.py` 用高位宽工作域和混合档位反事实排序区分：

- `tf_quantization`：词频/语义量化导致。
- `language_weight_rounding`：语言权重舍入导致。
- `truncation_tie` / 并列上下文：截断并列决断参与。
- `mixed_tf_and_language_rounding`：两类量化共同导致。

合成夹具已实际复现 TF 翻转和语言舍入翻转。

## 指纹缓存与增量范围

缓存键均使用 SHA-256 规范化 JSON：

- 字段 TF：`fieldtf(content_fingerprint, index_schema_fingerprint)`。
- 词项桶：`bucket(bucket_fingerprint, index_schema_fingerprint)`。
- 融合分：`fusion(query_fingerprint, document_count, referenced_bucket_ids, referenced_bucket_fingerprints, scoring_config_fingerprint, language_weight_fingerprint, fixed_point_fingerprint)`。
- 排序：`rank(fusion_cache_key, top_k, tie_policy, ordered_rows_hash)`。

替换 D004 的实际样本范围：

- 31 个相关桶重算，17 个无关注入桶直接校验并复用 `workspace/cache/buckets/`。
- 预热后再次替换 D004：Q01–Q07、Q09、Q10 重算，Q08 的融合分和排序结论命中持久化缓存。
- 修改 `translation` 语言权重：48 个桶全部复用，10 个查询融合结论全部因语言权重指纹变化而重算。
- 不改配置再跑一批：10 个融合分和 10 个排序结论全部复用。

新增/删除文档会改变 `document_count`，因此 IDF 对所有查询的影响是真实影响，所有融合键随之失效；桶级仍只增删/更新实际 postings 来源。

## 预算调度

点数定义严格采用：

```text
cost_points = len(query_terms) * len(participating_union_documents)
```

调度器按查询 ID 稳定顺序处理；未消耗的待办 ID 保持在队列头部，下一批优先。样本预算 `batch_points=40`、`query_quota_points=40` 的实测：

- 第一批：Q01 消耗 36 点；Q02(48)、Q03(60) 记 `E_QUERY_QUOTA_EXCEEDED`；Q04–Q10 因剩余 4 点不足记 `E_POINT_BUDGET_PENDING` 并保留待办。
- 第二批 24 点：优先处理 Q04，消耗 24 点；Q05–Q10 继续待办。

## 一致性与并发

`Engine.parity_material()` 对最终文档、正排、倒排账本、字段统计、逐查询分数、边界清单、缓存键、失败清单和调度状态生成规范化材料。验收比较三个独立工作区：

1. 增量命令序列 + 单线程。
2. 从最终文档/配置空白全量重建。
3. 同一增量序列 + 4 线程打分。

三者 SHA-256 相同。随机测试还使用 8 组种子化的追加、替换、删除、语言/字段权重、饱和参数、位宽和档位序列复核同一不变量。

并发只并行执行无共享的纯打分函数，结果按稳定查询顺序回主线程合并和写缓存，因此账本顺序确定。

## 快照与恢复

每次提交依次写：

```text
01_validation.json
02_index.json
03_statistics.json
04_scores.json
05_ranking.json
scheduler.json
report.json
manifest.json
```

所有文件先落在 `workspace/staging/txn-*`，随后整体移动到 `workspace/releases/snapshot-v...`，最后用临时指针文件和 `os.replace` 原子切换 `workspace/CURRENT.json`。提交前校验 `expected_global_version` 与配置哈希，冲突时抛 `E_VERSION_CONFLICT`。

恢复检查会构造 v1 已提交快照、一个不完整事务和一个完整 prepared v2 事务；`SnapshotStore.recover()` 清理不完整事务，并在版本基线匹配时提升 v2。实测 `recovered_to_v2=true`、`incomplete_cleaned=true`、`decision=promoted_prepared`、收尾 `can_exit=true`。

## 实际验证结果

最终一次完整结果以 `runs/verification_report.json` 为准；其中包含实际哈希、预算明细、复用清单、恢复结果和合成翻转归因。自动化测试结果为：

```text
8 passed
```

环境自检为 `ready=true`：Python `3.11.0rc1`、NumPy `1.26.4`、`127.0.0.1:0` 回环绑定成功、`workspace/staging` 可写。全过程只使用本地文件和确定性样本，不依赖 Docker、外部账号或线上服务。
