# mtcomp：确定性增量机器翻译组件

`mtcomp` 从零实现了一个面向本地固定样本的句级机器翻译组件。它把词典、打分参数和源句都视为可增量变更的状态，以内容寻址的三阶段缓存只重算真正受依赖影响的句子，并用版本目录快照、稳定排序预算调度和线程池保证可复现性。

项目不访问网络、不使用容器、不依赖随机数或真实翻译服务。运行前只需要 Python 3.10+、NumPy 和 PyYAML；所有输出、缓存和快照目录都由首次运行创建。

## 快速开始

```bash
python3 -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
python -m pip install -r requirements-dev.txt

# 生成固定种子样本（已随仓库提供 examples/sample，可重复生成）
python -m mtcomp generate-sample --output examples/sample

# 首次增量运行；自动创建 examples/out、阶段快照和全量对照报告
python -m mtcomp run --input examples/sample --output examples/out --reset

# 生成、运行、恢复续跑、全量对照和注入诊断的一体化验收
python -m mtcomp verify

# 单元/集成测试
python -m pytest -q
```

Windows 控制台由 CLI 主动切换为 UTF-8；快照使用同盘临时目录、`fsync` 与 `os.replace`，最终以清单文件原子切换版本。

## 输入目录

默认文件名如下，也由 `mtcomp.loader.DEFAULT_FILES` 固定。

```text
compile.yaml        编译与调度配置
params.json         定点打分参数
lexicon.tsv         双语词典
sentences.jsonl     源句
edits.json          增量编辑命令流
injections.json     独立异常注入配置
```

### `sentences.jsonl`

每行一个 JSON 对象。`tokens` 可以是空格分隔字符串或字符串数组。

```json
{"id":"S001","tokens":"the red cat runs fast","batch":0}
```

校验规则：

- 句 ID 必须非空且唯一；
- token 必须全部属于 `compile.yaml` 的 `controlled_vocabulary`；
- `batch` 必须是非负整数，同批有效句数不得超过 `max_sentences_per_batch`；
- 句子不得为空，且不得超过 `max_sentence_tokens`；
- 非法行输出 `E_SENT_INVALID` 并从可运行状态隔离。

### `lexicon.tsv`

严格四列：`源词 目标词 词频 对齐置信度`。

```text
cat 猫 95 0.98
cat 猫咪 35 0.93
```

词频必须为正有限数，置信度必须为 `[0,1]` 内有限数；重复的 `(源词,目标词)` 会输出 `E_LEX_INVALID`。词典内部按稳定键排序，文件行序不影响结果。

### `params.json`

```json
{
  "alignment_weight": 1.25,
  "order_weight": 0.85,
  "length_penalty": 0.18,
  "fusion_temperature": 0.92,
  "confidence_threshold": 0.75,
  "fixed_point_bits": 12,
  "quantization_tier": 16,
  "allowed_fixed_bits": [8, 12, 16, 20, 24],
  "allowed_quantization_tiers": [1, 2, 4, 8, 16, 32]
}
```

`fixed_point_bits` 与 `quantization_tier` 必须位于允许枚举内。量化步长为 `max(1, 2^bits / tier)`；权重、分数、置信度最终均换算为整数。

### `compile.yaml`

```yaml
ngram_order: 3
max_sentences_per_batch: 3
thread_pool_concurrency: 4
decode_points_per_batch: 40
snapshot_commit_interval_batches: 1
max_sentence_tokens: 12
candidate_limit: 6
controlled_vocabulary: [the, red, cat, runs, fast]
```

并发数、预算和快照间隔只影响调度，不进入句级结果缓存键；`ngram_order` 与 `candidate_limit` 进入结果编译指纹。

### `edits.json`

命令数组，也接受 `{"commands":[...]}`。支持的操作：

| `op` | 必需字段 | 说明 |
|---|---|---|
| `lex_add` | `source,target,frequency,alignment_confidence` | 新增词典条目 |
| `lex_delete` | `source,target` | 删除条目 |
| `lex_frequency` | `source,target,value` 或 `delta` | 设置/调整词频 |
| `lex_confidence` | `source,target,value` 或 `delta` | 设置/调整对齐置信度 |
| `param_set` | `field,value` | 修改打分参数、位宽或量化档位 |
| `sent_replace` | `sentence` | 替换同 ID 源句 |
| `sent_delete` | `id` | 删除源句 |
| `batch_append` | `sentences` | 追加新批次/新句 |
| `noop` | - | 不改变状态 |

命令失败输出 `E_EDIT_INVALID`，历史失败保存在失败清单，但标记为隔离且不进入活动失败集；不会部分修改状态。

### `injections.json`

```json
{
  "injections": [
    {"id":"INJ-LEX-001","type":"lexicon_invalid","line":"x y 0 1.2"},
    {"id":"INJ-WEIGHT-002","type":"nonfinite_weight","field":"alignment_weight","value":"NaN"},
    {"id":"INJ-SENT-003","type":"overlong_sentence","sentence_id":"INJ-LONG-SENTENCE","token":"the","token_count":99}
  ]
}
```

诊断在独立副本/临时文件中执行，不修改引擎状态。报告包含 `item_number`、`item_id`、实际业务码、`isolated`、`control_state_valid` 和 `effect_on_engine_state:"none"`。

## 确定性翻译算法

### 1. 候选生成

对句子的每个起点和 `1..ngram_order` 的连续源跨度，若跨度内每个源词都有词典项，则生成该跨度的目标短语候选。每词候选按以下顺序使用：

1. 对齐置信度降序；
2. 词频降序；
3. 目标词字典序升序。

重叠跨度候选再交给维特比解码决断。句子引用的依赖集合包含句子中每个 token 对应的全部词典条目，因此同源词新增/删除候选也能正确失效。

### 2. 定点打分与维特比

所有项均先量化到整数定点域，按源词序号升序累加：

- 对齐项：各词 `log2(max(conf, epsilon))` 的均值乘 `alignment_weight`；
- 词序项：`log2(span_length) / log2(ngram_order+1)` 乘 `order_weight`，奖励更长的连续短语覆盖；
- 长度项：源/目标字符长度差的相对值乘 `length_penalty` 后扣除；
- 融合项：总和除以定点 `fusion_temperature`。

DP 使用 NumPy `int64` 数组保存最优/次优分数。路径同分时依次比较目标词元组字典序、目标字符总长度、片段长度序列、源跨度序列和引用条目键。线程调度、字典插入顺序和命令顺序都不会改变该排序。

### 3. 置信度与边界

路径置信度由所选条目的平均对齐置信度和最优/次优路径边距组成，并量化到配置档位：

```text
confidence = quantize((3 * mean_alignment_confidence + margin_ratio) / 4)
margin_ratio = margin / (margin + 2^bits)
```

判定固定为：

```text
confidence > threshold  -> ACCEPT
confidence == threshold -> REVIEW（边界样本）
confidence < threshold  -> REJECT
```

每个边界样本会以 `fixed_point_bits + 1` 重算对齐、维特比与置信度，输出参考整数、千分比、差值和参考状态。若两档位状态翻转：

- 路径片段不同：`ALIGNMENT_QUANTIZATION_RESIDUAL`；
- 路径相同但边距/舍入导致翻转：`DECODE_ROUNDING_ACCUMULATION`。

句级输出同时保留整数千分比与定点整数/尺度。

## 精确失效与缓存

三个阶段的键都由规范化 JSON 的 SHA-256 派生：

```text
alignment_key  = H(content_hash, referenced_entry_fingerprints,
                   params_fingerprint, result_compile_fingerprint,
                   fixed_bits, quantization_tier, algorithm_version)
decode_key     = H(alignment_key, ...)
confidence_key = H(decode_key, ...)
```

`referenced_entry_fingerprints` 是句子中所有 token 的词典桶条目的稳定排序集合。单条词频/置信度变化只改变引用它的句子的键；未引用句子的三个阶段全部命中，重算计数增量为 0，扣点为 0。参数变化改变全部句子的参数指纹，触发全量重算。

词典、参数和源句最终集合决定结果；文件行序、批次划分、并发度、预算和快照间隔不改变逐句译文、置信度、边界详情、引用集合和结果缓存键。

## 预算调度与并发

- 调度稳定键为 `(batch, sentence_id)`；
- 每批每轮都有独立 `decode_points_per_batch` 预算，后续轮次重新获得预算；
- 静态解码点数为 `max(1, token 数 × ngram_order)`；三阶段缓存全命中时为 0；
- 预算不足时保留待办，输出 `E_DECODE_BUDGET_PENDING`，下一轮优先处理待办；
- 扣点按稳定键串行预保留，然后由最多 `thread_pool_concurrency` 个工作线程执行；
- 结果按稳定键统一提交，候选、账本、失败与累计点数顺序和单线程一致。

运行账本记录每句动作、静态点数、本轮扣点、全局/批次累计点数、剩余预算以及首次耗尽位置。

## 快照与恢复

输出目录结构：

```text
out/
├── manifest.json
├── result.json
├── sentences.jsonl
├── ledger.jsonl              # 运行历史账本
├── canonical_ledger.jsonl    # 结果派生的规范账本
├── failures.json
├── active_failures.json
├── diagnostics.json
├── parity.json
└── stages/vN/
    ├── alignment.json
    ├── decode.json           # 解码 + 置信度
    └── ledger.json           # 状态、记录、失败、运行账本
```

提交流程：在临时目录写三个阶段文件并逐文件 `fsync`，整体重命名到新版本目录，最后原子替换清单。清单包括：

- schema/global 版本号；
- 当前词典哈希、输入词典哈希、参数与配置指纹；
- 已完成批次游标、命令游标与命令前缀哈希；
- 每个产物的文件名、SHA-256、字节数和阶段版本。

启动时校验版本、输入文件哈希、命令前缀以及产物哈希。任一不一致时输出 `E_SNAPSHOT_STALE`，不续跑旧产物，并从当前输入安全重建。

## 增量与全量一致性

`parity.json` 对同一最终状态做两项独立运行：

1. 按命令流推进、保留真实复用历史的增量引擎；
2. 从空白缓存一次性计算的最终状态全量引擎。

严格比较并报告：

- 逐句 `translation`、片段序列、整数千分比/定点置信度、阈值状态、边界详情、引用集合、结果缓存键、错误；
- 活动失败清单；
- 规范账本顺序与全部静态字段。

增量账本中的 `last_action`、`recompute_count`、`reuse_count` 和实扣点数属于运行历史，必然与从空白全量重算不同；这些字段只存在于 `ledger.jsonl`/`runtime_ledger`，不混入可重建的 `canonical_ledger.jsonl`。翻译结果、活动失败和规范账本仍逐字段一致。

## 业务码

| 业务码 | 含义 |
|---|---|
| `E_SENT_INVALID` | 源句 JSON、ID、token、批次或长度非法 |
| `E_LEX_INVALID` | 词典行、词频、置信度或重复条目非法 |
| `E_CONFIG_INVALID` | 编译配置或打分参数非法/非有限 |
| `E_DECODE_BUDGET_PENDING` | 本轮预算不足，句子保留待办 |
| `E_NO_ALIGNMENT` | 删除词典后句子已无完整对齐路径 |
| `E_EDIT_INVALID` | 单条编辑命令非法；状态不变，失败隔离 |
| `E_SNAPSHOT_STALE` | 快照版本、输入哈希、命令前缀或产物哈希不一致 |
| `E_INJECTION_UNKNOWN` | 注入类型未知 |

## 项目结构

```text
mtcomp/
├── alignment.py     n-gram 候选与定点对齐打分
├── cache.py         三阶段内容指纹键
├── canonical.py     规范化 JSON、定点算术
├── cli.py           CLI 与 UTF-8 控制台
├── commands.py      增量命令及原子编辑
├── confidence.py    阈值判定、边界/翻转归因
├── decoder.py       定点维特比与稳定同分规则
├── diagnostics.py   三类注入隔离诊断
├── engine.py        缓存记录、预算调度、线程池、账本
├── loader.py        JSONL/TSV/JSON/YAML 加载校验
├── model.py         数据模型、指纹与失败类型
├── runner.py        增量/全量运行编排和验收报告
├── samplegen.py     固定种子样本生成
└── snapshot.py      阶段快照、原子提交与恢复判定
```

## 实际验证结果

本轮在 Python 3.11.0rc1 的项目虚拟环境中执行，安装版本为 NumPy 2.4.6、PyYAML 6.0.4、pytest 9.2.0；系统解释器中也已验证 NumPy 1.26.4：

```text
python -m pytest -q
20 passed

python -m mtcomp verify
checks_pass: true
seed: 20261122
sentence_count: 7
status_counts: ACCEPT=3, REJECT=3, REVIEW=1
boundary_count: 1
pending_count: 0
completed_batch_cursor: 3
run_count: 9
total_charged_points: 258
full_rebuild_parity: true
resume_payload_identical: true
injections_isolated: true
```

样本中唯一边界句为 `S006`：`confidence=750‰`，更高一位宽参考置信度同为 `750‰`，差值为 `0‰`。三个注入项分别得到 `E_LEX_INVALID`、`E_CONFIG_INVALID`、`E_SENT_INVALID`，且控制状态保持有效。
