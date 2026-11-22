# 本地语料配比与去重工作台

面向本地小型预训练语料的可重复数据配比、MinHash 近似去重、预算调度、阶段快照与审计工作台。项目只依赖 Python 3.10+、`numpy` 与 `pytest`，不需要容器、数据库、外部服务或网络访问。所有输入、缓存、快照、报告和结果路径都使用仓库内相对路径，可直接在 Windows 上执行。

## 快速开始

```bash
python -m pip install -r requirements.txt
python -m corpus_workbench generate-demo --root .
python examples/run_demo.py
pytest -q
```

Windows PowerShell 中可将 `python` 换成 `py -3`。示例会自行重新生成确定性数据，在 `output/` 下执行串行/并行一致性、预算中断续跑、快照失效和阈值缓存复用验证。

单次 CLI 运行：

```bash
python -m corpus_workbench run --run-id demo --reset
python -m corpus_workbench show --run-id demo
```

预算不足后，用更大的调度预算续跑同一个快照：

```bash
python -m corpus_workbench run --run-id demo --scheduler configs/scheduler_tight.json --reset
python -m corpus_workbench run --run-id demo --scheduler configs/scheduler_expanded.json --resume
```

## 交付内容

```text
corpus_workbench/
  config.py       配置模型、校验、稳定配置哈希
  loader.py       JSONL 字节级加载、UTF-8 校验、注入故障追踪
  sampling.py     来源权重最大余数分配与 SplitMix64 确定性采样
  hashing.py      token/shingle、固定置换族 MinHash、缓存键
  cache.py        签名缓存与重复组决断缓存、复用审计
  dedup.py        并查集归组、代表文档决断、相似度记录
  scheduler.py    稳定批次、线程池、来源配额、全局预算、待办游标
  snapshots.py    完整快照目录原子替换、manifest 与陈旧判定
  pipeline.py     阶段流水线、恢复、报告组装
  demo_data.py    确定性样本、配置与异常注入生成器
  cli.py          `python -m corpus_workbench` 入口
examples/run_demo.py
tests/            15 个端到端与单元测试
data/corpus/      确定性 JSONL 样本（含真实非 UTF-8 注入行）
configs/          配比、去重、调度、注入及变体配置
output/           运行时生成的快照、缓存、报告与结果
```

## 输入格式

### 文档 JSONL

数据目录下所有 `*.jsonl` 按相对路径升序读取，每行一个对象：

```json
{"source":"books","doc_id":"books-0001","text":"...","metadata":{"topic":"novel"}}
```

加载顺序与业务码：

1. 原始行按严格 UTF-8 解码；失败记 `E_ENCODING_INVALID`。
2. 非 UTF-8 行会从字节中提取 `source`、`doc_id`、`metadata.injection_id` 作为审计线索。
3. JSON 结构非法记 `E_ENCODING_INVALID`；必需字段类型非法记 `E_CONFIG_INVALID`。
4. 来源不在 `source_weights` 中记 `E_SOURCE_UNKNOWN`。
5. 文档 ID 全局重复时，后出现项记 `E_DUP_DOC_ID`。
6. 文本不是非空 UTF-8 字符串（包括空文本）记 `E_ENCODING_INVALID`。
7. 单条失败不会终止文件或后续文件加载。

### 配比配置

```json
{
  "source_weights": {"books": 0.30, "papers": 0.20, "web": 0.50},
  "target_sample_count": 42,
  "sampling_salt": "local-corpus-demo-v1"
}
```

权重必须为有限正数，目标采样数必须为正整数，盐必须为非空字符串。也兼容 `weights`、`target`、`salt` 字段别名。

### 去重配置

```json
{
  "shingle_length": 3,
  "signature_dim": 64,
  "jaccard_threshold": 0.72,
  "representative_rule": "source_weight_desc_doc_id_asc"
}
```

`shingle_length`、`signature_dim` 必须为正整数；阈值必须在 `[0, 1]`；代表规则支持 `source_weight_desc_doc_id_asc` 及 `source_weight_then_doc_id`、`weight_then_doc_id` 别名。

### 调度配置

```json
{
  "max_workers": 4,
  "global_shingle_budget": 5000,
  "source_quotas": {"books": 5000, "papers": 5000, "web": 5000},
  "batch_size": 8,
  "snapshot_interval_batches": 2
}
```

并发数、全局点数预算、批大小、快照间隔必须为正整数；单来源配额为非负整数。省略某来源配额时，该来源只受全局预算约束。

### 注入配置

```json
{
  "injections": [
    {
      "injection_id": "inj-non-utf8",
      "source": "books",
      "doc_id": "books-injected-non-utf8",
      "fault": "non_utf8",
      "business_code_hint": "E_ENCODING_INVALID"
    }
  ]
}
```

生成器实际写入四类输入故障：`empty_text`、`non_utf8`、`unknown_source`、`duplicate_doc_id`。报告中的 `injection.items[*].matched`、`observed_code`、`observed_phase` 与 `location` 证明注入命中及业务码映射。

## 确定性规则

### 加权采样

1. 只考虑实际存在文档且权重为正的来源，按来源名升序稳定处理。
2. 使用最大余数法把目标采样数分配到各来源；余数相同时来源名升序决胜。
3. 每个来源的随机种子由 `source + "\\0" + sampling_salt` 的 BLAKE2b 哈希派生，再由平台无关的 SplitMix64 产生 `randbelow`。
4. 来源内文档按 `doc_id` 升序后等概率有放回采样，因此允许同一文档重复出现。
5. 输出按 `(source, sample_index)` 排序，并赋全局 `order`；`sample_index` 与 `order` 均从 0 开始。

### MinHash

1. 使用 Unicode 单词正则分词并 `casefold`。
2. 连续 `shingle_length` 个 token 组成 shingle；短于长度的文档保留其唯一完整 token 序列。
3. shingle 先经 BLAKE2b 稳定映射为 64 位整数，再求唯一 shingle 点数 `point_cost`。
4. 签名使用 `2^61-1` 模素数上的固定仿射置换族 `(a*x+b) mod p`；系数由固定算法版本与维度位置派生。
5. 以签名位置一致率作为 MinHash 相似度估计，`>= jaccard_threshold` 的文档对用并查集连通，连通分量即重复组。
6. 代表文档先按来源权重降序、再按 `doc_id` 字典序升序决断。被剔除文档记录代表文档来源/ID、估计相似度和度量名。

### 预算、配额与待办

- 预算单位是上文定义的唯一 shingle 点数。
- 待办任务按 `(source, doc_id, stage)` 稳定键排序，每批取全局稳定序前 `batch_size` 个；线程池只影响计算完成先后，不影响准入、扣减、分组或报告顺序。
- 签名缓存命中不消耗预算，因为它复用了已有签名；缓存未命中才按唯一 shingle 点数扣减。因此增量复用会实际改变有限预算下的最终保留集合。
- 某来源累计点数将超过 `source_quotas[source]` 时，该来源立即停止，记 `E_SOURCE_QUOTA_EXCEEDED`，其余任务进入待办。
- 下一签名需要点数大于全局剩余预算时，记 `E_GLOBAL_BUDGET_EXHAUSTED`，当前与后续未处理来源进入待办。
- `pending_queue`、`cursor.next_pending_stable_key`、`cursor.remaining_by_source` 与 `batch_count` 一起持久化。提高调度配置预算后可从同一快照续跑。
- 预算耗尽时报告仍会给出已处理子集的临时去重结果，但 `complete=false`；只有待办清空后才提交 `complete` 结果。

### 并发一致性

文档任务稳定键为 `stage + "\\0" + source + "\\0" + doc_id`。调度器先在主线程按稳定顺序完成缓存查询与预算准入，再提交线程池计算；结果按稳定键回填。只要输入和缓存初始状态相同，`max_workers=1` 与任意并发度的以下内容逐字段一致：

- 采样序列与来源计数；
- 预算扣减序列、来源点数、已消耗点数；
- 失败清单及顺序；
- 重复组、代表文档、剔除文档及相似度；
- 最终保留集合与稳定结果摘要 `deterministic_result_digest`。

预算扣减记录最终按 `(source, doc_id, stable_key)` 规范化并重算 `remaining_global`，因此恢复后的历史扣减与新扣减也保持同一规范顺序。

测试中串行与 8 线程使用独立冷缓存运行，稳定核心对象完全相等。缓存热状态本身是运行输入的一部分，复现对比时应在相同缓存状态下比较。

## 增量失效与缓存审计

签名缓存键包含：

```text
内容指纹 + shingle_length + signature_dim + 算法版本 + 分词版本
```

因此：

- 只改 `jaccard_threshold`：签名全部复用，0 点预算，只重算并查集分组和代表决断。
- 改 `shingle_length` 或 `signature_dim`：所有相关签名键变化，全部签名失效。
- 改单篇文本：只有该文档内容指纹变化，其他来源/文档继续命中缓存。
- 重复组决断键额外包含成员内容指纹、来源权重、shingle 长度、签名维度和阈值；阈值变化时决断缓存自然失效。

报告 `cache_audit.signatures` 给出 `hits`、`misses`、`invalidated_by_params`、`invalidated_by_content`、按来源命中/重算计数、复用来源与重算来源。`cache_audit.group_decisions` 给出重复组决断缓存命中与未命中键。

## 阶段快照与恢复

阶段顺序为 `validation -> samples -> minhash -> dedup -> write`。每个快照版本写入独立临时目录，完成后用目录重命名整体替换，再原子更新 `CURRENT` 指针。布局如下：

```text
output/
  snapshots/<run_id>/
    CURRENT
    versions/v000001/
      manifest.json
      validation.json
      samples.json
      minhash.json
      dedup.json
      result.json
  cache/signatures/...
  cache/group_decisions/...
  reports/<run_id>.json
  reports/<run_id>-v000001.json
  results/<run_id>.json
  tmp/snapshots/<run_id>/...
```

`manifest.json` 记录全局版本号、配比与去重配置哈希、调度配置哈希、语料摘要、完整采样式列表、已消耗点数、待办队列、游标、各阶段产物版本/状态和产物哈希。语料摘要用于审计；真正的逐文档失效由采样文档指纹摘要与签名内容键完成。

恢复判定：

- 版本号与 `--expected-version` 不一致、配比哈希变化或去重哈希变化时，输出 `E_SNAPSHOT_STALE`，不读取旧阶段产物，不更新旧快照。
- 调度配置不在陈旧判定中，因此可以只增加全局预算或来源配额后继续同一快照。调度哈希仍写入 manifest 供审计。
- 去重参数变化导致快照陈旧后，签名缓存仍可跨运行复用；以新 `run_id` 启动新鲜流水线即可只重算必要签名和分组。
- 内容变化不会整体废弃快照：恢复时重建当前校验阶段，比较确定性采样序列，并核验每篇采样文档的指纹。未变文档直接复用签名；已变文档的旧点数会回收并重新计算，其他来源不受影响。若采样序列本身改变，则从采样阶段之后重置调度状态，但签名缓存仍按内容键复用未变文档。

## 异常隔离与业务码

| 业务码 | 触发位置 | 含义 |
| --- | --- | --- |
| `E_SOURCE_UNKNOWN` | 加载 | 来源不在权重字典 |
| `E_DUP_DOC_ID` | 加载 | 文档 ID 重复 |
| `E_ENCODING_INVALID` | 加载/MinHash | 非 UTF-8、空文本、无 token 或分片准备失败 |
| `E_CONFIG_INVALID` | 配置/结构 | 配置或文档结构不合法 |
| `E_SOURCE_QUOTA_EXCEEDED` | 调度 | 来源点数将超过单来源配额 |
| `E_GLOBAL_BUDGET_EXHAUSTED` | 调度 | 全局点数预算不足 |
| `E_SNAPSHOT_STALE` | 恢复 | 版本或配比/去重配置不一致，拒绝复用 |
| `E_SAMPLE_FAILED` | 采样 | 单来源采样记录异常，其他任务继续 |
| `E_MINHASH_FAILED` | MinHash | 单文档签名计算异常，其他文档继续 |

失败清单按阶段、来源、文档 ID、行号、业务码稳定排序。签名失败或输入失败文档不会进入去重结果。

## 审计报告

每次运行写入 `output/reports/<run_id>.json` 与 `output/results/<run_id>.json`；完成版本还写入对应版本报告。主要字段：

- `sampling`：目标数、采样记录、各来源采样数、最大余数分配。
- `documents`：加载有效/失败数、唯一采样文档数、去重前后数、签名失败数。
- `dedup`：重复组、成员、代表文档、被剔除文档及相似度。
- `budget`：全局预算、已消耗点数、剩余点数、来源配额/点数、完整预算扣减序列、待办数、游标。
- `failures`、`injection`：业务失败与注入命中映射。
- `stage_timings_ms`、`snapshot`、`cache_audit`：阶段耗时、快照版本/阶段状态、增量复用审计。
- `deterministic_core` 与 `deterministic_result_digest`：去掉计时和运行标识后的稳定结果，用于串行/并行逐字段核对。

## 实际验证

完整结果记录见 `VALIDATION.md`。当前仓库实测：

- 15 个 pytest 用例全部通过。
- 默认样本：42 条采样记录、36 篇唯一采样文档、2 个重复组、34 篇最终保留、576 个 shingle 点。
- 紧预算：160 点后停止，10 篇已处理、26 项待办，同时命中来源配额与全局预算业务码；扩容后从快照恢复为完整结果。
- 单线程与 8 线程运行的 `deterministic_result_digest` 相同，预算扣减序列、失败清单和重复组逐字段相同。
- 仅改阈值：36 个签名全部命中，签名未重算、消耗 0 点；仅改 shingle 长度/维度：36 个签名全部失效。
- 恢复时只改一篇采样文档：回收该文档旧预算点，仅该文档签名重算；其他已提交签名直接复用。
- 配置或版本不一致恢复：输出 `E_SNAPSHOT_STALE`，`CURRENT` 版本保持不变。
