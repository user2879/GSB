# 本地电商查询意图理解诊断工作台

这是一个无网络、固定种子的本地工作台，用于验证查询理解状态在命令编辑、并发执行和中断恢复后的确定性。项目从查询归一化、实体槽位抽取、意图分类、改写建议到阶段缓存、快照恢复和诊断报告形成完整闭环。

运行环境为 Python 3.10+，唯一运行时第三方依赖是 NumPy。本次实际验证使用 Python 3.11.0rc1 与 NumPy 1.26.4。

## 快速开始

```bash
python3 -m pip install numpy
python3 -m search_intent_workbench.cli generate --out data --seed 20261123
python3 -m search_intent_workbench.cli run \
  --data data/base \
  --injections data/injected/injections.json \
  --snapshots .snapshots \
  --out artifacts \
  --workers 4
python3 -m unittest discover -s tests -v
```

也可以一次完成生成、演进验证、故障注入和快照恢复检查：

```bash
python3 -m search_intent_workbench.cli demo
```

主要产物：

- `artifacts/diagnostic_report.json`：完整诊断报告。
- `artifacts/diagnostic_report.html`：无外部资源依赖的可视化报告。
- `artifacts/command_evolution.json`：8 条命令后的增量/全量一致性及失效集合。
- `artifacts/serial_concurrency_check.json`：串行、并发与超限请求的结果一致性。
- `artifacts/recovery_check.json`：正常恢复、陈旧判定和中断续跑判定。
- `artifacts/fault_report.json`：I01-I04 故障注入证据。
- `.snapshots/versions/v000001` 至 `v000009`：每个状态的阶段化快照。

## 目录结构

```text
search_intent_workbench/
  loader.py         查询、配置、规则、命令加载与校验
  normalization.py  大小写/全半角/空白归一化与切分
  entities.py       最大匹配、重叠消解、抑制清单
  intent.py         特征权重、位置衰减、稳定 softmax、量化
  rewrite.py        意图改写规则与实体占位符
  pipeline.py       四阶段缓存键、线程池与稳定排序
  state.py          五类命令的原子状态变更
  snapshot.py       暂存目录、整体替换、manifest、恢复判定
  report.py         JSON 诊断与 HTML 可视化
  runner.py         命令演进、并发和恢复的端到端验证
  faults.py         独立故障样本验证
scripts/generate_samples.py  固定种子样本生成器
data/base/                   合法基础样本
data/injected/               I01-I04 独立故障样本
tests/                       29 个 unittest 用例
```

## 输入数据

`data/base/queries.jsonl` 每行包含：

```json
{"query_id":"q001","terms":["Apple","iPhone","15","手机","壳"],"session_id":"s1","turn":1,"intent_label":"product_search"}
```

`data/base/intent_config.json` 包含受控词表、意图类别及特征词权重、实体类型、实体词典、归一化规则引用、打分参数和最大并发数。

`data/base/normalization_rules.json` 包含归一化规则表和改写模板表。规则表与配置分离，规则版本和修订号参与缓存键。

`data/base/commands.json` 包含 8 条命令，覆盖新增查询、删除查询、修改归一化规则、调整意图权重和替换实体词典条目。

固定种子 `20261123` 用于确定性地打乱受控词表和 JSONL 行顺序；语义标签和词项保持显式、可审计。`data/dataset_manifest.json` 记录样本计数和每个文件的 SHA-256。

## 校验与异常隔离

加载层逐行解析 JSONL。单行格式错误或业务校验失败输出 `E_QUERY_INVALID` 并进入失败清单，其他合法行继续执行。

已校验的查询约束：

- 查询 ID 唯一。
- 每个词项精确落在受控词表内。
- 会话内轮次是连续的 `1..N`，无重复、无空洞。
- 意图标注属于已配置类别。

已校验的配置和规则约束：

- 意图类别、实体类型、受控词表、规则 ID、改写规则 ID 不重复。
- 意图总分权重、特征权重、实体优先级均为正有限数。
- 归一化规则引用存在且不重复。
- 实体词典 `entry_id` 不重复，归一化后的 `(term, entity_type)` 不重复。
- 每个意图有改写模板。
- 小数位宽、量化档位和最大并发数合法。

单查询执行异常会变成 `E_QUERY_EXEC`，不会中止其他查询。失败清单最终按 `query_id / command_id / code / stage / message` 稳定排序。

## 四阶段算法

### 1. 归一化

按规则表顺序执行：

- `width_fold`：Unicode NFKC，全角转半角。
- `case_fold`：Unicode casefold。
- `whitespace_collapse`：合并连续空白并去除首尾空白。
- `token_split`：按空白切分为词项。

输出保留每个 token 对应的原始词项索引。归一化缓存键只纳入对该查询可能生效的规则修订快照，因此无关规则变化不会使查询失效。

### 2. 实体槽位

将实体词典词面按同一规则归一化后，在 token 序列上做连续最大匹配。候选排序为：

```text
token 长度降序 -> 词典优先级降序 -> 起始位置升序 -> 词典顺序 -> entry_id
```

按该顺序贪心接受互不重叠候选。与已接受槽位重叠的候选进入抑制清单，保留 `entry_id`、原始位置、长度、优先级及重叠槽位 ID。

### 3. 意图分类

特征词匹配支持多 token 词面。每个匹配贡献：

```text
intent_weight * feature_weight * token_length * (1 / (1 + position_decay * token_start))
```

原始分求和后使用最大值减法稳定 softmax：

```text
exp(raw - max(raw)) / sum(exp(raw - max(raw)))
```

得分按配置的定点小数位宽和 quantum 使用 `ROUND_HALF_EVEN` 量化。意图排序为量化分降序、类别名升序；主意图取首项。输出包含贡献分解、`score_units`、六位定点字符串和量化元数据。

### 4. 改写建议

按主意图选择优先级最高、ID 字典序最小的改写模板，替换 `{entity:type}`、`{normalized_query}`、`{query}` 和 `{intent}`。缺失实体类型稳定回退为“相关商品”。

## 阶段缓存与精确失效

每个查询的四个阶段都生成稳定缓存键。键材料按阶段选择实际依赖：

- 归一化：查询内容指纹、规则表版本、适用规则修订快照。
- 槽位：内容指纹、归一化键、规则表版本、当前查询可命中的词典条目快照。
- 意图：内容指纹、归一化键、槽位键、词表特征、位置衰减、量化参数、实际命中意图的权重。
- 改写：内容指纹、归一化键、槽位键、意图键、所选改写模板。

阶段结果同时记录引用词项、规则、实体条目、意图和改写规则。某实体词条只进入能够匹配该查询的槽位键，因此替换一个词条只使实际命中的查询及其下游重算。权重调整也只影响命中相应意图的查询。

实际失效结果：

| Step | Command | 类型 | 增量=全量 | 各阶段重算数 |
|---:|---|---|:---:|---|
| 1 | cmd001 | 新增 q017 | 是 | 1/1/1/1 |
| 2 | cmd002 | 删除 q006 | 是 | 0/0/0/0 |
| 3 | cmd003 | 关闭 case_fold | 是 | 7/7/7/7 |
| 4 | cmd004 | 调整 product/support 权重 | 是 | 0/0/14/14 |
| 5 | cmd005 | 替换 iPhone 15 条目优先级 | 是 | 0/3/3/3 |
| 6 | cmd006 | 替换 AirPods Pro 条目词面 | 是 | 0/2/2/2 |
| 7 | cmd007 | 新增 q018 | 是 | 1/1/1/1 |
| 8 | cmd008 | 关闭 whitespace_collapse | 是 | 1/1/1/1 |

表中每列依次为归一化、槽位、意图、改写。

## 增量与全量一致性

每条命令后都执行两条路径：

1. 使用上一状态阶段缓存做增量运行。
2. 使用相同最终状态的新 Workbench 做无缓存全量运行。

比较内容包括查询顺序、归一化产物、槽位与抑制清单顺序、各意图得分、贡献分解、主意图、改写建议、缓存键、输出指纹和失败清单顺序。8 条命令全部逐字段一致。

## 并发与稳定排序

使用 `ThreadPoolExecutor`，每个查询一个任务，任务内部按阶段产生 `query_id::stage` 稳定排序键。结果在收集后按查询 ID 排序，意图得分按量化分与类别名排序，失败清单再按固定键排序。

并发参数取 `min(配置 max_workers, 请求 workers)`，下限为 1。实际验证：

- 串行有效并发 1，观测峰值 1，结果一致。
- 请求 4，有效并发 4，观测峰值 4，结果一致。
- 请求 20，被配置上限截为 4，观测峰值 4，结果一致。

## 阶段化快照与恢复

每次状态提交先在 `.staging-*` 临时目录写入：

- `state.json`：查询、配置、规则、失败清单。
- `stage_normalization.jsonl`
- `stage_slots.jsonl`
- `stage_intent.jsonl`
- `stage_rewrite.jsonl`
- `manifest.json`

每个阶段文件采用原子写；全部完成后整个目录原子改名为 `versions/vNNNNNN`，再原子更新 `CURRENT` 指针。manifest 包含全局版本、配置哈希、规则表指纹、词典哈希、各阶段文件哈希与条数、已完成阶段和已完成查询游标。

恢复判定：

- 版本与配置哈希完全匹配且已提交：`RESUME`，可复用 4 个阶段文件。
- 匹配版本的暂存目录处于 `in_progress/crashed_in_progress`：`RESUME_STAGING`，返回已完成阶段和游标。
- 版本或配置哈希不一致：`E_SNAPSHOT_STALE`，`reusable=false`，可复用产物列表为空。
- 无任何快照：`FRESH`。

测试包含真实子进程在写完 slots 阶段后执行 `os._exit(87)`。重启检查得到 `RESUME_STAGING`、`completed_stage=slots`、`completed_query_cursor=17`；哈希不匹配时得到 `E_SNAPSHOT_STALE` 且不返回任何产物。

## 诊断报告与故障注入

JSON 报告包含：

- 意图分布。
- 实体槽位接受/抑制统计。
- 最小分差的边界意图样本。
- 逐查询归一化文本、槽位、抑制候选、完整意图得分、贡献分解与改写。
- 失败清单、阶段复用/重算清单、命令演进、并发与恢复检查。
- 独立故障注入命中证据。

四个独立故障文件：

| ID | 故障 | 观察结果 |
|---|---|---|
| I01 | 意图权重为 Infinity | `E_CONFIG_INVALID`，配置执行前被拒绝 |
| I02 | 实体词典条目重复 | `E_CONFIG_INVALID`，重复词面与类型被识别 |
| I03 | 会话轮次 1 -> 3 | `E_QUERY_INVALID`，坏查询入失败清单，其余 16 条继续 |
| I04 | 查询词项“量子”越界 | `E_QUERY_INVALID`，坏查询入失败清单，其余 16 条继续 |

## 实际验证结果

执行日期：2026-11-23。最终状态：

```text
status=PASS
global_version=9
valid_queries=17
failures=0
commands=8
all_commands_equal=true
serial_concurrent_equal=true
recovery_ok=true
fault_hits=I01,I02,I03,I04
config_hash=27af258c745401e116557fb4cad0bf80ff417eb128a1b74409f3a87034b49692
rule_fingerprint=b137b126ded48884895f6c77787ca0797b347397b980750ba082e64cf540f24c
dictionary_hash=c31303d541c039209f8e1edaa0577647b57190a839bdd278d94e9315d8787f49
```

意图分布为 `product_search=12`、`browsing=2`、`support=2`、`scenario_recommendation=1`。接受槽位 34 个，抑制候选 7 个。

测试命令：

```text
python3 -m unittest discover -s tests -v
```

结果：`Ran 29 tests in 2.707s`，全部通过。`python3 -m pyflakes search_intent_workbench scripts tests` 无告警。

## 设计边界

- 样本和词典均保存在本地，不访问网络。
- 改写建议是确定性模板渲染，不调用外部生成模型。
- 缓存作用于单次进程内上一状态；快照持久化阶段产物用于重启复用。
- 进入暂存目录后若进程被强制终止，未提交版本不会被当作正式快照。
