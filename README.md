# 确定性倒排召回与定点排序原型

这是一个完全本地、无在线服务、无外部数据集的 Python 3.11 原型，用于核对旧搜索引擎迁移中的倒排召回、字段加权、停用词、定点 BM25 风格打分、TopK 边界决断、精确增量复用与中断恢复。样本由固定整数种子生成，所有中间产物只写入工作目录。

## 环境与安装

```bash
cd /home/zsl/feedback/uidfc5909d3-default-312
.venv/bin/python --version
.venv/bin/python -c 'import numpy, pytest; print(numpy.__version__, pytest.__version__)'
```

本机基础解释器缺少 `ensurepip`，因此实际用本机 `virtualenv -p python3.11` 创建 `.venv`，再用其 pip 安装 `numpy` 与 `pytest`。在具备 `python3.11-venv` 的环境中也可直接：

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install numpy pytest
```

## 快速开始

```bash
# 1. 生成固定种子样本、受控词表、停用词、配置与编辑流
.venv/bin/python -m search_proto generate --out-dir samples --seed 20261126 --documents 24 --queries 8

# 2. 严格校验全部输入；任何坏行都会产生 E_INDEX_INPUT
.venv/bin/python -m search_proto validate \
  --config samples/config.json \
  --vocabulary samples/controlled_vocab.json \
  --stopwords samples/stopwords.json \
  --documents samples/documents.jsonl \
  --queries samples/queries.jsonl \
  --edits samples/edits.jsonl

# 3. 增量运行，并与空白全量重建逐字段对照
.venv/bin/python -m search_proto run \
  --config samples/config.json \
  --vocabulary samples/controlled_vocab.json \
  --stopwords samples/stopwords.json \
  --documents samples/documents.jsonl \
  --queries samples/queries.jsonl \
  --edits samples/edits.jsonl \
  --out-dir artifacts/acceptance-final \
  --snapshot-dir artifacts/acceptance-final-state \
  --mode both

# 4. 测试
.venv/bin/pytest -q
```

`--mode both` 会输出 `incremental/`、`full/` 与 `comparison.json`。`comparison.json` 中 `equal=true` 表示排序、定点/浮点分数、边界文档、失败清单顺序、依赖集合以及活动缓存键与结果指纹逐字段一致。

## 输入格式

### 文档 JSONL

```json
{"doc_id":"D0001","version":0,"fields":{"title":["alpha","the","alpha"],"body":["beta","delta"]},"field_stopword_marks":{"body":[1]}}
```

- `doc_id` 唯一且非空。
- `version` 为非负整数。
- 字段名只能是 `title`、`body`，且须在配置字段集合内。
- 所有词项必须属于受控词表。
- `field_stopword_marks` 是按位置声明的额外字段级停用标记；配置停用词和文档标记都会记录被排除位置与原因。

### 查询 JSONL

```json
{"query_id":"Q0001","terms":["alpha","beta"],"field_preferences":["title","body"]}
```

- `query_id` 唯一，`terms` 非空。
- `field_preferences` 只能引用配置字段。
- 打分时查询词项稳定升序、字段按固定顺序 `title, body` 处理，不依赖集合/数组归约顺序。

### 其他输入

- `controlled_vocab.json`：`{"version":"vocab-v1","terms":[...]}`。
- `stopwords.json`：`{"version":"sw-v1","global":[],"by_field":{"title":[...],"body":[...]}}`。
- `config/prototype.json`：字段权重、BM25 参数、定点枚举、TopK、并发上限、批量预算、单查询配额与阶段容差。
- `edits.jsonl`：支持 `append_doc`、`replace_doc`、`delete_doc`、`set_field_weights`、`set_bm25`、`add_stopword`、`remove_stopword`、`set_fixed`、`set_top_k`。

任何非法输入都不会被静默跳过，而是写入失败清单，业务码为 `E_INDEX_INPUT`，并带来源文件、行号、实体与细分原因。

## 打分口径

对查询词项 `t`、字段 `f`、文档 `d`：

```text
idf(t)       = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
avgdl(t, f)  = 含 t 于字段 f 的文档有效字段长度均值
norm         = b * dl(d, f) / avgdl(t, f)
saturation   = tf(d, t, f) * (k1 + 1) / (tf + k1 * (1 - b + norm))
score(d, q)  = Σ_t Σ_f field_weight(f) * idf(t) * saturation
```

定点路径每个中间环节都量化为整数最小单位：

1. `idf` 用高精度 Decimal 计算后量化。
2. 字段长度比与长度归一项量化。
3. 饱和项量化。
4. 字段权重、IDF、饱和项按固定乘法顺序量化并累加。

`decimal_places` 决定每分的小数位与内部 `scale=10**decimal_places`，`quantum` 是内部整数域的最小量化步长。切换位宽或量化档位会改变真实整数分与中间量，而不是只改显示格式。

每条得分同时保留：

- `fixed_score_units` 与格式化 `fixed_score`；
- `float_score` 浮点参考；
- 每个词项/字段的 IDF、长度归一、饱和、字段权重、贡献定点与浮点值；
- 分阶段最大绝对误差、首次超过容差的环节；
- 单阶段反事实分数，用于把排序翻转归因到饱和舍入、字段长度量化、IDF 量化、字段权重量化、累加量化或并列决断。

TopK 先按定点分降序，再按 `doc_id` 升序、`doc_version` 升序决断。与截断边界差落在半个量化步长内的文档会进入边界清单，并附带浮点参考分、差值、位置、翻转原因和阶段误差。并列导致的翻转显式标记为 `tie_break_rule`。

## 增量与缓存

缓存分为四层，全部使用稳定 JSON 指纹：

- 文档字段统计：文档内容指纹、字段集合、停用词表版本。
- 词项统计：该词项当前引用的全部文档倒排记录。
- 查询-文档得分：查询指纹、文档实际命中的查询词项统计、字段/BM25/定点配置。
- 查询排序：全部得分缓存键、TopK、位宽/量化与容差。

替换单篇文档时，只移除并重建该文档的倒排项及相关词项统计；查询仅在该文档命中的词项统计发生变化时重算得分。修改字段权重、BM25 参数或位宽只重算打分与排序。停用词条目只影响实际含该词项的文档，并记录未命中文档的键重派生与结果复用。每层账本都记录 `reused`、`rebuilt`、`invalidated`、旧/新键、依据和结果指纹。

增量运行会先建立基线缓存，再施加编辑并重跑，因此未受影响部分有真实复用。全量运行没有前一缓存，决策状态自然是 `rebuilt`；等价对照比较的是活动缓存实体、缓存键与结果指纹，以及最终语义产物。

## 调度、快照与恢复

- 每批维护独立打分预算；点数按 `查询词数 × 本次参与文档数` 计。
- 待办查询优先于新查询，按 `query_id` 稳定处理。
- 单查询累计所需点数超过配额时记 `E_QUERY_BUDGET_EXCEEDED` 并中止该查询。
- 批次预算耗尽时保留查询游标、已消费点数和部分得分，记 `E_POINT_BUDGET_PENDING`，下一批优先继续。
- 若单批预算连一份文档都容纳不了，调度器停止并标记 `scheduler_stalled`，绝不空转。

索引、词项统计、得分、排序、失败、缓存账本、报告与完整状态先写入工作目录内临时目录，`fsync` 后整体改名为版本目录，再原子切换 `CURRENT` 指针；上一版本写入 `LAST_GOOD`。启动时校验清单、产物 SHA-256、配置哈希、停用词版本、源指纹和全局版本，决定续跑或重建。`--crash-after-batch 1` 会在快照提交后调用 `os._exit(137)`，可验证真实中断恢复。

## 产物

运行目录包含：

- `index.json`：正排、倒排、排除位置、依赖关系与索引指纹。
- `term_stats.json`：词项 DF、字段 DF、平均字段长度与引用文档。
- `scores.json`：定点/浮点分、完整中间量、阶段误差与反事实分数。
- `ranking.json`：逐查询 TopK、边界文档、次序翻转证据与原因。
- `failures.json`：稳定排序的业务失败清单。
- `cache_ledger.jsonl`：逐条缓存失效/复用证据。
- `report.json`：TopK、边界数、翻转数、失败码计数、缓存决策与并发观测。

更多实现说明见 `docs/DESIGN.md`，实际验证结果见 `docs/VERIFICATION.md`。
