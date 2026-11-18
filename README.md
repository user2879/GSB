# 本地电商知识库对话式检索与生成式答案应用

纯本地、确定性的多轮检索-抽取-引用拼装应用。仅依赖 Python 3.11 标准库与
NumPy 2.x，不访问网络服务、不加载模型权重；所有样本由固定种子生成，所有产物
写入工程内 `artifacts/` 目录。

## 1. 环境准备

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt   # numpy 2.x + pytest
```

若系统 Python 裁剪了 `ensurepip`，可先用官方 `get-pip.py` 引导（本工程即按此
方式在 Python 3.11.0rc1 + NumPy 2.4.6 上验证）。

一键验收（清空产物 → 生成样本 → 全量 → 增量重放 → 快照恢复 → 测试）：

```bash
./run.sh
```

## 2. 目录结构

```
rag_app/
  errors.py           业务码常量（与题面一一对应）
  models.py           Document / QueryTurn / Command / RetrievalConfig / Failure
  normalize.py        全半角折叠、大小写折叠、空白合并与分词
  rules.py            领域规则表：同义词、类目扩展、改写模板
  loader.py           JSONL 逐行加载与 E_REC_INVALID 校验
  fingerprint.py      SHA-256 内容指纹与缓存键
  retrieval.py        NumPy 向量化 BM25 + 融合特征，字典序同分决断
  extraction.py       答案窗口抽取、引用补足、E_CITATION_MISSING
  engine_keys.py      召回/抽取/引用三阶段缓存键
  engine_pipeline.py  单轮三阶段流水线（缓存命中/失效/快照持久化）
  engine_state.py     交互命令与受影响轮次/下游闭包计算
  engine.py           引擎主体：线程池、首响预算、版本化快照
  snapshot.py         临时文件 + os.replace 整体替换、manifest 恢复判定
  replay.py           增量重放 vs 空白全量的逐字段比对
  generator.py        固定种子样本/配置/规则/故障注入生成
  report.py / cli.py  JSON 报告与命令行入口
sample/               生成的确定性样本（可重复复现）
tests/                31 个 unittest 风格 pytest 用例，覆盖 8 条业务要求
artifacts/            快照与报告（验收前由 run.sh 清空）
```

## 3. 输入与业务码映射

- `sample/documents.jsonl`：`doc_id / title / body / source`
- `sample/queries.jsonl`：`session_id / turn_no / query / is_followup / context_ref`
- `sample/commands.jsonl`：`seq / op / args`，操作码
  `set_bm25`、`set_recall_top_k`、`set_fusion_weights`、`set_template`、
  `update_template`、`set_context`、`replace_turn_query`
- `sample/retrieval_config.json`：归一化、BM25、召回条数、答案窗口、引用下限、
  融合权重、首响预算、并发上限、快照提交间隔
- `sample/domain_rules.json`：`version / synonyms / category_expand / templates`
- `sample/fault_injection.json`：异常注入开关（字段名与业务码链路对应）

| 业务码 | 触发场景 | 代码位置 |
| --- | --- | --- |
| `E_REC_INVALID` | 文档/查询/命令结构、唯一性、轮序空洞、追问引用非法 | `rag_app/loader.py`、`rag_app/replay.py` |
| `E_CITATION_MISSING` | 召回为空或强制无引用，答案标记不可用 | `rag_app/extraction.py`、`rag_app/engine.py` |
| `E_ANSWER_BUDGET_PENDING` | 预算内无法得到最新参数答案，返回中间状态 | `rag_app/engine.py` |
| `E_CONFIG_MISMATCH` | 快照 manifest 配置哈希/规则指纹不一致 | `rag_app/snapshot.py` |

## 4. 关键设计

### 4.1 确定性检索（要求 2）

归一化顺序固定为全半角折叠 → 大小写折叠 → 空白合并 → 空白分词，再套用同义词
与类目扩展。BM25 使用 NumPy 矩阵一次性打分（稳健 IDF，避免负值），融合
`title_weight / overlap_weight` 特征后按 `(-score, doc_id)` 排序，同分时严格
按文档 ID 字典序决断；召回条数受 `recall_top_k` 上限约束，零分文档不召回。

### 4.2 三阶段指纹缓存与失效（要求 4）

- 召回键：查询指纹 + BM25/融合/召回参数指纹 + 索引指纹 + 规则版本
- 抽取键：父召回键 + 命中文档集合 + 查询词项 + 窗口/融合参数 + 规则版本
- 引用键：父抽取键 + 片段指纹 + 引用下限 + 模板名与模板体 + 规则版本

每轮记录引用文档集合。命令只标记受影响轮及其追问下游闭包：
`set_template/update_template` 仅令 citation 阶段失效（检索与抽取零重算），
其余命令令召回→抽取→引用三级失效。未失效轮在提交前即被跳过，不会重新检索。

### 4.3 首响预算（要求 5）

`ThreadPoolExecutor`（并发数取 `max_concurrency` 且不超过该上限）按稳定排序键
`session_id#turn_no` 提交任务；每条命令在 `first_response_budget_ms` 内驱动
重算，超时立即返回携带 `E_ANSWER_BUDGET_PENDING` 的中间摘要（已完成轮数、
当前引用列表、受影响轮、pending 轮与进度），后续命令或 `drain` 继续消费
in-flight future，绝不阻塞等待全量、也不会把延迟结果标记为完成。

任务在**提交时**快照索引、配置、规则与模板（不可变 `TurnContext`），并携带
轮次纪元 `epoch`：命令使某轮失效时递增纪元并取消该轮旧任务，任务完成时若
纪元已落后则丢弃结果。因此无论线程调度如何交错，跨命令遗留任务都不可能用
过期参数覆盖新状态，并发度 1/2/4/8 的产物与单线程逐字段一致。

### 4.4 增量与全量逐字段一致（要求 6、7）

`replay` 先在同一引擎上按命令流增量执行，再用最终输入在空白快照上一次性全量
执行；对每轮的有效查询、召回候选与得分、抽取片段、答案文本、引用顺序、补足
来源及失败清单做逐字段比对。并发度 1/2/4/8 的结果与单线程完全一致（测试
`test_concurrency_levels_produce_identical_results`）。

### 4.5 版本化快照（要求 8）

阶段产物按缓存键内容寻址，写入走 `*.tmp` + `os.fsync` + `os.replace`，外部
永远观察不到半成品；`manifest.json` 记录全局版本号、配置哈希、规则指纹、
已完成轮游标与各阶段产物版本。恢复时先比对版本/哈希，不一致直接返回
`E_CONFIG_MISMATCH`，一致则从快照加载并全部缓存命中（零检索）。

## 5. CLI

```bash
.venv/bin/python -m rag_app.cli generate           # 重新生成确定性样本
.venv/bin/python -m rag_app.cli validate           # 仅做输入校验
.venv/bin/python -m rag_app.cli full               # 空白快照全量执行
.venv/bin/python -m rag_app.cli replay             # 增量 vs 全量一致性
.venv/bin/python -m rag_app.cli resume             # 快照恢复判定/续跑
```

报告输出：`artifacts/reports/full_report.json`、`replay_report.json`，包含逐轮
答案与引用、缓存命中/失效/跳过计数、每条命令耗时与是否超预算、快照版本号与
失败清单。

## 6. 实测结果（本机记录）

- `pytest tests -q`：31 passed，连续多轮稳定无竞态
- `replay`：`consistent=true`、`diffs=[]`、`budget_pending_seen=true`
- 模板切换命令：检索阶段 `stage_skips > 0`、`retrievals_executed` 不增加
- 快照恢复：15 个流水线轮次三阶段全部缓存命中、`retrievals_executed=0`
- 故障注入：`E_REC_INVALID`、`E_CITATION_MISSING`、`E_ANSWER_BUDGET_PENDING`、
  `E_CONFIG_MISMATCH` 均有对应用例覆盖并通过
