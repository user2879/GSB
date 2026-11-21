# 中文新闻摘要与重写诊断工作台

本项目在本地完成中文新闻的句子切分、定点打分、摘要选择、风格化重写和结果汇总，重点验证反复编辑、参数调整、并发执行、优雅中断及强制终止后的状态一致性。

项目只读取本地受控样本，不访问网络，不依赖 Docker 或 GPU。运行环境为 Python 3.11、NumPy 1.26 和 PyYAML。打分使用 `Decimal` 固定运算顺序，NumPy 作为环境依赖预装，避免浮点数导致跨运行漂移。

## 快速开始

```bash
python3 scripts/generate_samples.py
python3 -m news_diag run --clean-snapshots
python3 -m news_diag run                    # 无变更增量：应复用全部阶段产物
python3 -m news_diag verify                 # 逐命令比较增量与空白全量
python3 -m pytest -q
python3 scripts/run_acceptance.py            # 一键生成汇总验收报告
```

默认输入位于 `samples/`，默认快照位于 `.workbench/snapshots/`，输出位于 `artifacts/`。

```text
samples/news.jsonl          固定种子生成的新闻 JSONL
samples/config.yaml         摘要与重写配置
samples/commands.jsonl      编辑命令流
samples/injections.yaml     五类真实输入故障注入
samples/invalid_documents.jsonl
samples/invalid_config.yaml
samples/invalid_commands.jsonl
artifacts/report.json       最近一次运行诊断报告
artifacts/output.json       最近一次运行的规范结果投影
artifacts/acceptance_report.json  汇总验收报告
```

## 输入契约

### 新闻文档

`news.jsonl` 每行一个文档，也接受一个 JSON 数组。核心结构：

```json
{
  "document_id": "news-0001",
  "title": "标题",
  "paragraphs": ["第一段。", "第二段。"],
  "entities": [{"id": "e1", "text": "云海市发改委", "type": "ORG"}],
  "events": [{"subject": "云海市发改委", "predicate": "发布", "object": "年度计划"}]
}
```

校验包括文档 ID 唯一、标题和每个段落非空、实体有有效 `id/text/type`、实体 ID 不重复、事件三元组三个字段完整。非法条目进入 `load` 阶段失败清单，业务码为 `E_DOC_INVALID`，不会进入后续阶段。

### 配置

`config.yaml` 的主要字段：

```yaml
max_summary_sentences: 3
scoring:
  word_frequency_weight: "0.45"
  entity_coverage_weight: "0.35"
  position_decay_weight: "0.20"
  entity_coverage_min: "0.16"
  position_decay_base: "0.96"
fixed_decimal_places: 4
thread_pool_size: 4
snapshot_commit_interval_docs: 2
rewrite:
  style_template_id: official
  synonym_table_version: v1
  max_chars: 62
synonym_tables:
  v1:
    templates:
      official: {prefix: "【要闻】", suffix: "", max_chars: 60}
    entries:
      - {template_id: official, source: "表示", candidates: ["指出", "强调"]}
```

配置校验覆盖正整数摘要句数、权重与实体覆盖下限、0..9 定点位宽、1..64 并发数、提交间隔、长度上限、模板声明和每个同义替换条目的模板引用。配置非法为 `E_CONFIG_INVALID`。

### 编辑命令

`commands.jsonl` 每行一条命令。支持：

- `replace_document`：替换或新增完整文档。
- `add_paragraph` / `remove_paragraph`：按 `index` 增删段落，禁止删除唯一段落。
- `set_max_summary_sentences`：调整摘要长度。
- `set_style_template`、`set_synonym_table_version`、`set_rewrite_style`：调整重写风格参数。
- `set_scoring`、`set_fixed_decimal_places`：调整打分参数。
- `append_batch`：追加带 `batch_id` 的命令批次。
- `rollback_last_batch` / `rollback_batch`：恢复最新批次之前的完整文档与配置状态，批次历史随快照持久化。

无效命令进入 `command` 阶段失败清单，业务码为 `E_COMMAND_INVALID`；同一条命令流中的其余命令继续处理。

## 五阶段处理

1. **切分**：按中文句末标点和段落边界切分；句子保存 `paragraph_index`、全局 `start/end` 与原文片段。偏移基于段落以 `\n` 连接后的稳定原文。
2. **打分**：以配置词表做最长匹配分词。固定运算顺序为
   `word_frequency_weight * 词频占比 + entity_coverage_weight * 实体覆盖率 + position_decay_weight * 位置衰减`，
   最后按 `ROUND_HALF_UP` 转为定点整数。实体覆盖为零时，`score_sources` 固定不包含 `entity_coverage` 并设置 `entity_coverage_zero=true`。
3. **选择**：只选择实体覆盖达到下限的句子；按得分降序、起点升序、终点升序、句子 ID 升序稳定排序。
4. **重写**：按替换表条目顺序扫描；同一位置仅采用首个匹配条目，候选列表固定采用索引 0。超长文本硬截断到有效上限，记录 `truncation_offset` 和原始长度。
5. **汇总**：只聚合成功文档，输出摘要、逐句来源、实体集合、替换记录、截断记录与统计。失败文档不进入汇总。

## 精确失效与一致性

每个阶段产物都包含：

```json
{"cache_key": "...", "fingerprint": "...", "payload": {}}
```

缓存键由阶段名、文档 ID、上游内容指纹、该阶段配置指纹以及适用的同义替换表版本派生。具体依赖如下：

| 变更 | 失效范围 |
| --- | --- |
| 替换文档、增删段落 | 该文档切分、打分、选择、重写、汇总 |
| 调整摘要句数或实体覆盖下限 | 选择、重写、汇总 |
| 调整风格模板、替换表版本或最大长度 | 重写、汇总 |
| 调整词频/实体/位置权重或定点位宽 | 打分、选择、重写、汇总 |

命中缓存时直接复用 payload，不调用对应阶段函数；运行报告的 `invalidation_reuse` 会逐项标记 `reused/recomputed/failed`。`output.json` 中的 `invalidation_reuse` 是规范化的阶段键清单，不含运行时状态，因此增量与全量可逐字段比较。

验证命令会对每条命令后的最终状态执行：

- 同一快照目录继续增量运行；
- 独立快照目录忽略旧缓存、从空白应用同一命令前缀；
- 比较完整 `output.json`，包括文档顺序、切分与偏移、逐句得分、选择顺序、摘要文本、替换/截断记录、失败清单顺序、缓存键和阶段键顺序。

## 快照、恢复与信号

`SnapshotStore` 采用代际提交：

1. 新状态写入 `snapshots/.tmp-*` 临时目录并 `fsync`；
2. 临时目录原子重命名为唯一的 `gen-00000000000N/`；
3. 原子替换 `CURRENT` 指针，使其指向完整代。

重启只读取 `CURRENT` 指向的 `state.json`。未完成临时目录、未发布代和损坏代均忽略。快照记录全局版本、配置哈希、基础文档哈希、同义替换表版本、命令前缀哈希、已完成文档游标、有效配置覆盖、批次历史和五个阶段的产物版本。

启动时会校验配置哈希、基础文档哈希和已应用命令前缀。若指定 `--expect-version`，还会校验全局版本。不匹配时报告 `E_SNAPSHOT_STALE`，拒绝复用旧缓存并从当前输入重建。

安装 `SIGTERM` / `SIGINT` 处理器。信号只设置停止事件；正在处理的文档完成各阶段后，在文档边界提交快照并以整数 `0` 退出。`--stop-after-docs N` 提供可重复的模拟信号路径。强制 `SIGKILL` 不会留下可见半成品，重启从最后完整代继续。

## 异常注入

`samples/injections.yaml` 针对指定文档和阶段修改真实局部输入，再调用生产校验器：

| 编号 | 故障 | 阶段 | 业务码 |
| --- | --- | --- | --- |
| INJ-001 | 空段落 | split | E_DOC_INVALID |
| INJ-002 | 非法实体 | scoring | E_DOC_INVALID |
| INJ-003 | 摘要长度为零 | selection | E_CONFIG_INVALID |
| INJ-004 | 定点位宽越界 | scoring | E_CONFIG_INVALID |
| INJ-005 | 替换表条目引用缺失模板 | rewrite | E_CONFIG_INVALID |

报告写入注入编号、命中状态、实际业务码和 `code_match`。单文档失败被隔离，其余文档继续运行和汇总。

## CLI

```bash
# 增量或首次全量
python3 -m news_diag run \
  --docs samples/news.jsonl \
  --config samples/config.yaml \
  --commands samples/commands.jsonl \
  --snapshot-dir .workbench/snapshots \
  --report artifacts/report.json \
  --output artifacts/output.json

# 忽略旧缓存并清理快照目录
python3 -m news_diag run --rebuild --clean-snapshots

# 故障注入
python3 -m news_diag run --injections samples/injections.yaml \
  --snapshot-dir .workbench/injection-snapshots \
  --report artifacts/injection_report.json \
  --output artifacts/injection_output.json

# 逐步比较增量与空白全量
python3 -m news_diag verify
```

`run` 对业务条目失败仍返回 0，并把错误写入报告；配置无法加载等无法启动的情况返回 2。`verify` 全部一致返回 0，否则返回 1。

## 实际验收结果

`python3 scripts/run_acceptance.py` 生成的 `artifacts/acceptance_report.json` 记录了以下实测值：

- 全量运行：10 篇成功摘要、1 条预期 `E_COMMAND_INVALID`、41 个阶段项；20 条摘要句、35 条替换、13 条截断。
- 无变更增量：41/41 阶段项复用，规范结果与全量完全相同。
- 逐命令比较：0..11 共 12 个状态全部逐字段一致。
- 异常注入：5/5 命中，5 个注入的业务码全部匹配，5 篇故障文档被隔离，其余 5 篇进入汇总。
- 模拟优雅中断：收到可处理停止事件后退出 0，恢复运行结果与全量一致。
- 真实 `SIGTERM`：退出码 0，报告记录 `handled_signal=15`，恢复结果与全量一致。
- 真实 `SIGKILL`：从 `gen-000000000001` 完整代恢复，8 个已有阶段产物复用，最终结果与全量一致。
- 陈旧快照：`--expect-version` 不匹配时产生 `E_SNAPSHOT_STALE`，复用数为 0。
- 自动化测试：11 项全部通过。
