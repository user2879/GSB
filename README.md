# 本地确定性中文 NER 编辑与操作应用

这是一个仅使用 Python 标准库的单机应用。它面向固定种子生成的字级中文 NER 语料，在用户连续编辑词表、语料、模型权重、定点位宽与量化档位时，同时提供：

- 配置预算内的交互摘要；超预算时返回最近一次已完成摘要和 `E_LATENCY_BUDGET_EXCEEDED`。
- 基于内容指纹的阶段级增量计算；未失效的特征、发射、解码、合并产物直接复用。
- 与空白缓存全量计算逐字段一致的业务结果。
- 分段原子快照、可恢复快照、信号中断后的安全退出。
- 完整 JSON 诊断报告与可重复的单元/集成测试。

## 快速开始

```bash
python3 --version                 # 需要 3.10+
python3 main.py --workspace . --verify-full --report artifacts/report.json
```

也可使用模块入口：

```bash
python3 -m ner_editor --workspace . --verify-full --report artifacts/report.json
```

常用选项：

```text
--corpus PATH          字级语料 JSONL，默认 samples/corpus.jsonl
--model PATH           模型参数包 JSON，默认 samples/model.json
--config PATH          交互配置 JSON，默认 samples/config.json
--commands PATH        命令流 JSON，默认 samples/commands.json
--report PATH          报告路径，默认 <workspace>/artifacts/report.json
--run-id ID            快照运行标识，默认 default
--resume               从 CURRENT 快照恢复并只执行尚未处理的命令
--max-commands N       本次最多执行 N 条命令后稳定并提交快照
--budget-ms MS         覆盖配置中的交互延迟预算
--verify-full          最终状态再用空白缓存全量重算并比较
--no-snapshot          禁用快照
--quiet                不输出逐命令 JSONL，只写报告
```

正常完成返回 `0`；可恢复信号 `SIGINT`/`SIGTERM` 在完成当前命令快照后分别返回 `130`/`143`；模型或配置致命非法返回 `2`。

Windows 可直接运行：

```powershell
python main.py --workspace . --verify-full --report artifacts\report.json
```

## 目录

```text
main.py                    入口脚本
ner_editor/app.py          CLI、信号、恢复、全量对照
ner_editor/loader.py       四类输入解析与业务码校验
ner_editor/features.py     字符/前后字/位置/词表命中长度特征哈希
ner_editor/emission.py     定点嵌入聚合与发射打分
ner_editor/decoder.py      整数对数域 BIESO 维特比
ner_editor/merging.py      片段修复、类型裁决、边界探测
ner_editor/engine.py       内容寻址缓存、线程池、预算摘要、账本
ner_editor/snapshot.py     分代目录快照与原子 CURRENT 指针
samples/                   固定种子样本、参数、配置、命令流
samples/invalid/           非法输入演示
tests/                     标准库 unittest 验收测试
TECHNICAL.md               算法、缓存键、快照与确定性规则
VALIDATION.md              实际运行结果
```

## 输入契约

### 字级语料 JSONL

每行一个对象。字符、BIESO 标签、实体类型都可用空格分隔字符串或数组：

```json
{"id":"s001","chars":"张 三 在 北 京","labels":"B E O B E","types":"PER PER O LOC LOC"}
```

支持字段别名：

- 样本 ID：`id` / `sample_id`
- 字序列：`chars` / `tokens` / `text`
- 标签：`labels` / `bieos` / `tags`
- 类型：`types` / `entity_types` / `ner_types`
- 也接受 `B-PER` 形式的组合标签。

校验项：样本 ID 唯一、字与标签等长、标签限于 `B/I/E/S/O`、实体从 `B` 开始并以 `E` 结束（允许 `B E` 两字实体）、`I/E` 不得孤立、片段不越界、同一片段类型一致。非法行产生 `E_CORPUS_INVALID`，且不会进入推理阶段；其他合法行继续运行。

### 模型参数包

核心字段：

```json
{
  "model_version": "local-hash-bieso-v1",
  "feature_dim": 12,
  "hash_buckets": 97,
  "labels": ["B", "I", "E", "S", "O"],
  "char_embeddings": {"张": [0.1, -0.2, "..."], "@BOS": ["..."]},
  "emission_weights": [["12 行，每行 5 列"]],
  "transition_weights": [["5x5 对数域转移"]],
  "start_weights": [0, 0, 0, 0, 0],
  "default_entity_type": "ORG",
  "vocabulary": [{"id": "v001", "text": "张三", "type": "PER"}]
}
```

约束：`feature_dim`、`hash_buckets` 为正整数；每个嵌入向量长度等于 `feature_dim`；发射矩阵行数等于 `feature_dim`、列数等于标签数；转移矩阵为标签数乘标签数；词表条目 ID 唯一、类型在 `LOC/ORG/PER/TIME` 内。结构/维度非法输出 `E_MODEL_PARAM_INVALID`。非有限数值允许载入，但在发射阶段输出 `E_EMISSION_INVALID`。

### 交互配置

```json
{
  "interaction_latency_budget_ms": 600,
  "fixed_decimals": 4,
  "quantization_level": 1,
  "quantization_levels": [1, 2, 8],
  "thread_pool_workers": 4,
  "snapshot_commit_interval_commands": 3,
  "boundary_score_tolerance": 4.50006,
  "temp_dir": "artifacts/tmp",
  "default_entity_type": "ORG"
}
```

### 命令流

命令流可以是顶层数组、`{"commands":[...]}` 或 JSONL。支持别名，规范命令如下：

| `type` | 必需字段 |
|---|---|
| `vocab_add` | `entry:{id,text,type}` |
| `vocab_delete` | `entry_id` |
| `vocab_replace` | `entry_id`, `entry:{id,text,type}` |
| `corpus_replace` | `sample_id`, `sample:{id,chars,labels,types}` |
| `corpus_delete` | `sample_id` |
| `emission_adjust` | `target:{dimension,label}`, `delta`/`value`/`weight` |
| `transition_adjust` | `target:{from,to}`, `delta`/`value`/`weight` |
| `fixed_point_set` | `fixed_decimals` |
| `quantization_set` | `quantization_level` |
| `noop` / `quit` | 无 |

命令类型不合法、目标缺失或权重字段缺失输出 `E_COMMAND_INVALID`；命令携带的语料非法输出 `E_CORPUS_INVALID`；越界权重目标输出 `E_MODEL_PARAM_INVALID`。失败命令不会修改状态。

## 输出与确定性

报告包含：

- `samples`：逐样本实体片段、路径标签/总分、逐位置候选得分、修复依据、边界记录、发射量化细节、阶段版本。
- `failures` / `current_stage_failures`：按固定顺序排列。
- `code_counts`：业务码统计，包括延迟预算码。
- `reuse_recompute_ledger`：每样本每阶段的缓存键、输入/输出指纹、版本、引用词表 ID 和 `reuse/recompute/failed`。
- `command_responses`：逐命令首响摘要、进度和状态。
- `snapshot`：当前快照代次、清单、变化阶段。
- `equivalence`：`--verify-full` 时的空白缓存全量对照结果。

同一内容、配置与量化档位下，特征索引、发射得分、维特比路径、候选得分、实体片段和失败顺序均逐位一致。线程数只影响执行方式，不影响业务投影。

## 测试

```bash
python3 -m unittest discover -s tests -v
```

测试覆盖：三类加载错误、非有限参数、定点残差、标签/类型并列决断、片段修复、边界翻转、阶段级失效、逐命令增量/全量一致、1/4 线程一致、低预算首响、快照硬链接复用、损坏快照回退、正常恢复与 `SIGINT` 恢复。

详细算法与快照协议见 [TECHNICAL.md](TECHNICAL.md)，实际结果见 [VALIDATION.md](VALIDATION.md)。
