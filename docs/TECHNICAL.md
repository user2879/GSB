# Technical Design

## 1. Pipeline Shape

```text
JSONL inputs
  -> normalization and record validation (batched snapshots)
  -> language identification
  -> per-language BM25 inverted index (batched snapshots)
  -> stable budget plan
  -> thread-pool query retrieval and bridge expansion (batched snapshots)
  -> evaluation (batched snapshots)
  -> deterministic JSON report
```

四个阶段为 `normalize`、`index_build`、`retrieval_scoring`、`evaluation`。每个阶段都读取 `snapshot.batch_size`，每批成功后先写临时目录，再将目录原子重命名为版本化产物；最后原子替换 `manifest.json`。若进程在产物重命名后、清单替换前退出，清单仍指向旧产物，续跑不会使用未提交的半批。

## 2. Input Normalization and Validation

归一化顺序：

1. 按配置执行 Unicode NFKC，覆盖全角/半角折叠。
2. 执行 Unicode casefold。
3. 将连续空白折叠为单个空格并去除首尾空白。
4. 使用配置的正则 `normalization.token_pattern` 分词。

加载层检查：

- 文档 ID、查询 ID 非空且唯一；重复 ID 的后续记录记 `E_RECORD_INVALID`。
- 语言标签位于 `allowed_languages`；翻译语言位于 `index_languages`。
- 标题、正文、查询、翻译词项归一化后非空。
- 翻译置信度是有限数值且位于 `[0, 1]`。
- 相关等级是非负整数且不超过 `max_relevance_grade`。
- qrel 必须引用已加载的查询 ID 和文档 ID。
- JSON 语法错误、非对象记录、空行均进入失败清单。

单条非法记录只生成一个失败项，不影响后续记录。最终失败清单按阶段、标识、业务码、异常类型、消息、来源和行号稳定排序。

## 3. Language Identification

每个词项按脚本分类：

- 含 Han 字符：`zh`
- 含拉丁字母：`en`
- 数字或符号：neutral

证据包括 Han/Latin 词项数、比例、混合比例、各语言翻译覆盖率和决策规则。LID 开启时：

- 两种脚本均存在且 `min(han_ratio, latin_ratio) >= mixed_ratio_threshold`：判为 `mixed`。
- 否则取脚本比例较高的一侧。
- 无脚本信号时，用翻译覆盖率作为决策依据。

LID 关闭时使用声明语言，但仍输出完整证据。`mixed` 查询按每个词项的脚本路由到对应语言索引，这是混合查询能够同时检索中英文档的关键。

## 4. BM25 and Stable Candidate Order

每个索引语言维护独立倒排表、文档长度与文档频率：

```text
idf(t) = log(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
score(t,d) = idf(t) * tf(t,d) * (k1 + 1)
             / (tf(t,d) + k1 * (1 - b + b * dl(d) / avgdl))
```

标题词按 `bm25.title_repeat` 重复计入词频。查询词项按 `(脚本语言, 词项)` 排序；每个词项的 posting 文档按 ID 排序。最终候选按融合得分降序、单语得分降序、文档 ID 字典序决断。

每个查询任务携带 `query_id|query|retrieval_scoring`，每个候选携带 `query_id|doc_id|retrieval_scoring`，每个词项分解携带包含角色、语言和词项的稳定键。

## 5. Cross-Language Bridge

对每个查询词项：

1. 先在词项自身脚本语言的索引中查找 posting。
2. 有 posting 时只做单语计分，不扩展。
3. 无 posting 时在翻译表查找目标语言条目。
4. 同一源词项按置信度降序、目标词项字典序、目标语言字典序选第一条，其余写入 `suppressed`。
5. 桥接贡献为 `bm25(target_term, doc) * confidence * bridge_weight * query_tf`。
6. 无翻译条目不扩展，写入 `missing_translation_terms` 诊断。

翻译抑制清单、扩展清单和缺失清单都按固定键排序。

## 6. Fusion, Fixed Point and Grades

对每个候选文档：

```text
mono_norm   = mono_raw   / (mono_raw   + temperature)
bridge_norm = bridge_raw / (bridge_raw + temperature)
fused       = (mono_weight * mono_norm + cross_weight * bridge_norm)
              / (mono_weight + cross_weight)
score_fixed = round(fused * fixed_point.scale)
```

定点整数被限制在配置位宽的有符号正数范围内。报告同时记录 `fixed_width`、`fixed_scale`、分数整数、浮点展示值、单语整数和桥接整数。

预测等级由排序后的 `relevance_thresholds` 下界决定。等级比较使用定点反量化值，避免平台浮点展示差异影响阈值。

## 7. Parallel Budget Scheduler

预算计划在启动线程前按查询 ID 字典序生成：

```text
allocation = min(per_query_quota, global_remaining)
global_remaining -= allocation
```

全局预算耗尽后的查询保留稳定位置并标记 `E_BUDGET_EXHAUSTED`。已获配额的查询逐候选检查预算：

- 候选发现阶段只读取 posting 的文档 ID，不计算 BM25 得分。
- 候选评分严格按文档 ID 顺序进行。
- 在评分第 `allocation + 1` 个候选前立即抛出 `E_QUERY_BUDGET_EXHAUSTED`。
- 该查询不返回部分排序，消耗点数记为已用配额。

线程池最大工作数来自 `threads.max_workers`，每批实际提交任务数不超过该值。线程写入仅限各自结果对象；主线程按提交顺序收集，并按查询 ID 重排后提交快照，因此线程完成顺序不会改变任何输出字段或预算扣减序列。

## 8. Snapshot and Resume

`manifest.json` 记录：

- `global_version`：已完成逻辑批次数。
- `pipeline_version`、`config_hash`、`translation_hash`、`injection_hash`。
- 文档、查询、翻译、qrel 的原始输入哈希。
- 每阶段已完成批次号。
- 每阶段最新产物版本、相对路径、最后批次和提交版本。

启动时比较完整身份。任何差异都会输出 `E_CONFIG_MISMATCH` 并拒绝续跑，且不修改清单。`--fresh` 是显式重建状态目录的唯一方式。阶段产物使用 `vNNNNNN` 名称；续跑只执行清单中未完成的批次。

## 9. Fault Injection

独立注入文件不影响默认运行。支持的故障：

| Kind | Stage | Real fault |
| --- | --- | --- |
| `nonfinite_confidence` | retrieval | 将实际翻译条目的置信度构造成非有限值并通过生产校验器校验 |
| `translation_term_missing` | retrieval | 强制查询扩展查找一个翻译表中不存在的词项 |
| `recall_limit_zero` | retrieval | 将单查询召回上限真实设置为 0 |
| `relevance_grade_out_of_range` | evaluation | 将目标查询的实际评估等级改为 `max_relevance_grade + 1` |

每个异常只终止对应查询的目标阶段。失败项记录查询 ID、阶段、业务码、异常类型、消息和注入 ID；其他查询继续调度。报告为每个注入项给出 `hit`、`hit_stage` 和实际失败证据。

## 10. Evaluation

评估只使用成功完成检索的查询。支持：

- `precision_at_k`
- `recall_at_k`
- `ndcg_at_k`
- `mrr_at_k`
- `grade_mae`
- `grade_accuracy`

`k` 由 `evaluation.k` 配置。排序指标使用 qrel 中的实际等级；等级误差和准确率比较预测等级与已判定 qrel。评估失败查询不计入聚合分母，但保留在逐查询评估段和全局失败清单。

## 11. Report and Determinism Hash

报告包含：

- 逐查询位置、状态、LID 证据、翻译扩展/抑制/缺失、候选和逐词项分解。
- 相关性得分、定点分数、预测等级和排序。
- 预算分配、实际消耗、剩余预算和稳定扣减序列。
- 评估指标与逐查询评估。
- 失败清单、注入命中、索引统计、阶段耗时和快照清单。

确定性摘要排除时间戳、阶段耗时、观测线程数、`threads.max_workers` 和快照身份哈希，保留所有业务结果。并发/串行以及全量/增量验证均比较该摘要和逐字段可比内容。

## 12. Actual Verification

执行命令：

```bash
python -m unittest discover -s tests -v
python run_pipeline.py verify --sample-dir samples/default \
  --config samples/default/config.json \
  --injections samples/default/injections.json \
  --output artifacts/verification.json
```

验证覆盖：非法记录隔离、LID 证据、桥接、翻译抑制、无翻译诊断、定点字段、同分决断、并发/串行一致、全量/增量一致、可处理停止退出码 0、全局/单查询预算、四类注入命中与隔离、配置哈希不一致拒绝续跑。

