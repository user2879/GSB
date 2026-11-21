# 中文新闻摘要与风格重写诊断工作台

该项目面向中文新闻摘要链路，提供确定性句子切分、定点重要性打分、实体覆盖约束选择、模板化风格重写、扰动注入归因、并发一致性、失败隔离、断点恢复和逐句 JSON 诊断。

## 快速开始

```bash
python -m pip install -r requirements.txt
python scripts/generate_samples.py --output samples
python run.py --news samples --config samples/config.yaml \
  --injections samples/injections.json --report reports/sample_report.json
python scripts/verify_samples.py
python -m unittest discover -s tests -v
```

Windows PowerShell 等价命令：

```powershell
py -3.10 -m pip install -r requirements.txt
py -3.10 scripts/generate_samples.py --output samples
py -3.10 run.py --news samples --config samples/config.yaml --injections samples/injections.json --report reports/sample_report.json
py -3.10 scripts/verify_samples.py
py -3.10 -m unittest discover -s tests -v
```

也可以安装为 console script：`pip install -e .` 后运行 `news-diagnose`。所有输入输出均按 UTF-8 处理。

## 项目结构

```text
src/news_diag/
  cli.py          CLI、早期信号处理、报告落盘
  loader.py       JSONL/YAML/注入清单加载与结构校验
  splitter.py     中文句界、原文偏移、词频/实体输入
  scoring.py      定点分项量化与 float64 参考路径
  selection.py    覆盖下限约束的确定性选择
  rewrite.py      模板替换、冲突抑制、缺失模板保留
  engine.py       文档流水线、扰动归因、线程池、状态恢复、报告
  models.py       数据模型与配置
  io_utils.py     原子写、稳定 JSON、进度追加
scripts/
  generate_samples.py  固定种子样本生成器
  verify_samples.py    并发/串行/重复运行一致性验收
samples/               生成后的新闻、配置、替换表、注入清单
tests/                 16 个单元与集成测试
reports/               实际运行报告与断点状态
docs/RUN_REPORT.md     本次实际验证记录
```

## 输入

`--news` 可指向 JSONL 文件，也可指向目录。目录中优先使用 `news*.json{,l}`，也支持由单行单文档 JSON 文件组成的目录。每篇文档至少包含：

```json
{
  "doc_id": "DOC-001",
  "title": "新闻标题",
  "paragraphs": ["第一段。", "第二段。"],
  "entities": [
    {"entity_id": "E1", "name": "市交通委", "aliases": ["交通委"], "type": "ORG"}
  ],
  "events": [
    {"event_id": "V1", "subject": "E1", "predicate": "召开", "object": "E1"}
  ]
}
```

校验规则包括文档 ID 唯一、标题和段落非空、实体 ID/名称/类型及别名结构、事件 triples 完整且引用已声明实体。非法条目不会被静默丢弃，而会生成 `E_CONFIG_INVALID` 并进入 `failures`。

关键配置字段：

```yaml
max_summary_sentences: 3
word_frequency_weight: 0.40
entity_coverage_weight: 0.38
position_decay_coefficient: 0.22
entity_coverage_min: 0.80
style_template_id: formal_v1
synonym_table_version: v1
synonym_table_path: synonym_table.yaml
fixed_decimals: 6
quantization_levels: 5
max_workers: 4
```

`quantization_levels` 必须整除 `10**fixed_decimals`，以保证量化档位可精确映射到定点整数。`max_workers` 是线程池的硬上限。

支持四类注入：

| 类型 | 作用 |
| --- | --- |
| `frequency_delta` | 对目标句增加/减少词频命中数 |
| `entity_coverage_delta` | 对目标句增加实体覆盖输入值 |
| `illegal_entity` | 在目标段落上下文注入非法实体，预期触发文档加载失败 |
| `missing_template_ref` | 将指定替换规则改绑到未声明模板 |

每个注入项都有唯一 `injection_id`。命中项进入 `injections.applied`，未命中项附判定依据进入 `injections.not_applied`。

## 数值规则

句子按 `。！？!?；;` 和段落边界切分，记录文档级字符半开区间 `[start, end)`。标题不参与句子打分；段落之间以单个换行连接。词频词表来自文档实体名称、别名和事件谓词，句内命中数除以文档总命中数得到词频归一值。实体覆盖比例是句内命中的文档实体数除以文档实体总数。位置服从 `1 - sentence_index/(sentence_count-1)`，单句文档取 1。

固定路径按以下固定顺序计算：

```text
component = raw_component * weight
quantized_component = round_half_up(component * quantization_levels) / quantization_levels
fixed_component_integer = quantized_component * 10**fixed_decimals
fixed_sentence_integer = sum(fixed_component_integer for 3 components)
```

参考路径使用 numpy float64，以相同项顺序计算三项加权和，不量化。每句报告均包含：

- 词频、实体、衰减三项的归一值、权重、加权值、定点整数、定点值、舍入残差与参考值；
- 定点总分、参考总分、两者差值和总舍入残差；
- 扰动前后两条路径分数和选中状态。

样本使用 `fixed_decimals=6`、`quantization_levels=5`，量化步长为 `0.2`，用于稳定触发数值边界。

## 选择与边界

固定路径按 `(-fixed_score, start, sentence_index)` 排序，参考路径按 `(-reference_score, start, sentence_index)` 排序。依次选取不超过 `max_summary_sentences` 的句子并累计实体覆盖，达到 `entity_coverage_min` 后停止补充。累计覆盖恰好等于下限时写入 `coverage_equal_to_threshold_stop` 事件，并计入 `coverage_boundary_hit_count`。

同一句在固定路径中被选/被弃、而在参考路径中结论相反时，标记为边界句。边界句清单输出固定分数、参考分数、差值与两条路径的选择结果。边界句占比的分母是该文档全部句子数。

## 重写与错误链

只使用 `style_template_id` 对应的规则。候选替换按键长降序、规则表条目序、起点升序处理；接受区段与已接受区段重叠时只保留优先级更高的一次替换，并记录 `suppressed_replacements`。

未声明模板引用的规则不参与替换，但会先保留其匹配区段，防止其他短规则改写该区段。报告中同时输出：

- `rewrite_table_warnings`：表级 `E_TEMPLATE_MISSING`；
- `rewrite_errors`：命中保留区段及原文片段；
- 被保留区段压制的候选替换记录。

单个文档的切分、打分、选择或重写异常只生成该文档的 `E_STAGE_FAILED` 失败记录，不终止其他文档。`E_CONFIG_INVALID` 用于配置、加载和注入清单校验失败。

## 并发、确定性与状态

文档任务按 `doc_id` 排序提交，线程池大小取 `min(config.max_workers, 待处理文档数)`；`--serial` 只使用一个线程。每个文档、句子和阶段都有由文档 ID、句子序号、阶段名派生的稳定键。文档、摘要句、边界句、替换记录、注入项和失败清单最终均显式排序，因此并发与串行报告逐字节一致。

`--state-dir` 将每篇完成文档写成独立原子 JSON，并维护小型清单 `run_state.json` 与追加式 `progress.jsonl`。恢复只读取该目录中的已持久化产物。收到可处理的 `SIGINT`/`SIGTERM`/`SIGBREAK` 时，运行器停止启动未开始文档、等待活动文档完成并落盘，随后以退出码 0 返回；强制终止后的恢复只依赖原子状态文件，不承诺生成退出报告。

## 报告入口

报告顶层字段包括 `dependencies`、`input_fingerprint`、`config`、`documents`、`failures`、`rewrite_table_warnings`、`rewrite_errors`、`injections` 和 `completion`。每篇文档包含 `summary`、逐句 `score_before`/`score_after`、`competition_order`、`boundary_sentences`、`replacement_records`、`suppressed_replacements`、`diagnostic_errors` 与 `statistics`。

同一输入和配置的两次完整运行会生成逐字节相同的 JSON。当前样本的实测哈希、命令和结果见 `docs/RUN_REPORT.md`。

## 实际环境说明

本次执行环境是 WSL Ubuntu 22.04（Linux），Python 3.11.0rc1、numpy 1.26.4、PyYAML 6.0.2。实现仅使用标准库、numpy 和 PyYAML，不依赖 Docker、GPU、网络服务或平台专属路径；Windows 10/11 可按上方 PowerShell 命令运行，但本次未在原生 Windows 解释器上重复执行。
