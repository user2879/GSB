# GSB 确定性分片倒排索引

本目录实现一个无容器、纯本地、可复核的分片倒排索引原型。样本由固定整数种子生成；每个分片是独立的 Python 子进程，通过 `127.0.0.1` TCP JSONL 协议接收文档批次。中心进程负责校验、修订仲裁、定点召回、预算调度、内容指纹复用、快照发布与恢复诊断。

## 环境

实际验证环境：

- Python `3.10.12`
- NumPy `2.2.6`
- pytest `9.2.0`
- 虚拟环境：`.venv/`
- 回环端口由样本生成器在 `127.0.0.1` 上探测后分配；子进程由测试或构建代码自行拉起并回收

本机的默认 `python3` 缺少 `ensurepip`，因此这里用系统 Python 3.10 和已有的 `virtualenv` 建环境：

```bash
virtualenv --clear -p /usr/bin/python3.10 .venv
.venv/bin/python -m pip install -r requirements.txt
```

验证命令：

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/verify.py
```

最近一次结果：`24 passed`；验收脚本全部检查为 `true`，详细结果见
`artifacts/verification/report.json`、`artifacts/verification/summary.txt`。

## 目录

```text
gsb_index/
  config.py          # 加载与逐对象校验，失败不中断其余对象
  models.py          # 文档、分片、压缩、预算、修订模型
  revisions.py       # 显式父修订链、回滚、回执仲裁
  indexing.py        # 分片侧确定性倒排摘要与内容指纹
  shard_service.py   # 真实回环 TCP JSONL 分片子进程
  cluster.py         # 子进程启动、就绪探测、失败隔离与回收
  scheduler.py       # 稳定顺序配额、并发派发、待办队列
  scoring.py         # 定点 TF/IDF、量化、并列决断、边界标记
  invalidation.py    # 分片/桶级精确失效与快照复用
  ledger.py          # 中心账本合并与仲裁审计
  snapshot.py        # 临时目录暂存、原子指针提交、恢复判定
  engine.py          # 端到端编排
  edits.py           # 可序列化编辑序列
  sample.py          # 固定种子样本与配置生成
  verify.py          # 验收场景脚本
samples/seed-17/     # 首版确定性样本、分项配置、编辑序列
tests/               # 24 个单元/集成测试
artifacts/verification/
```

## 样本与运行

生成样本：

```bash
.venv/bin/python -m gsb_index generate --out samples/seed-17 --seed 17 --port-start 19200
```

校验配置（`--config` 可为聚合 `project.json`，也可为只含分项 JSON 的目录；不启动服务）：

```bash
.venv/bin/python -m gsb_index validate --config samples/seed-17/project.json
```

全量构建：

```bash
.venv/bin/python -m gsb_index build \
  --config samples/seed-17/project.json \
  --workdir artifacts/run-full \
  --mode full --workers 4 --global-version 1
```

增量构建：

```bash
.venv/bin/python -m gsb_index build \
  --config samples/seed-17/project.json \
  --workdir artifacts/run-full \
  --mode incremental --workers 4 --global-version 2
```

模拟提交前终止（退出码 `9`，留下 `.staging-*`，不切换 `CURRENT.json`）：

```bash
.venv/bin/python -m gsb_index build \
  --config samples/seed-17/project.json \
  --workdir artifacts/run-full \
  --mode incremental --simulate-termination
```

## 配置语义

`project.json` 聚合以下分项，样本目录同时保留独立文件：

- `vocabulary.terms` / `vocabulary.stopwords`：受控词表与停用词。
- `documents[]`：`doc_id`、正整数 `version`、`shard_id`、`fields.title`、`fields.body`、`field_stopwords.title/body`。
- `layout.shards[]`：`shard_id`、`port`、`term_bucket_capacity`、`concurrency_limit`、`seed`、`revision`。
- `compression`：`bucket_count`、`bucket_weights`、`encoding`。
- `budget`：`global_term_entry_budget`、`per_shard_term_entry_limit`。
- `revisions`：显式 `version`/`parent` 链以及当前生效 `current_revision`；回滚只需把当前版本指向祖先。
- `scoring` 与 `queries[]`：定点位宽、量化步长、字段权重、TF 饱和常数和查询。

校验规则包括：文档 ID 唯一、词项受控、字段仅允许 `title/body`、停用词标记为布尔、分片 ID 与回环端口唯一、容量和并发上限为正整数、修订版本单调不减、桶容量必须被生效桶数整除、权重必须有限且非负、修订父引用有效。失败写入排序稳定的 `failure_list`，合法对象继续构建。

## 分片协议

每个连接发送一行 JSON：

```json
{"type":"index_batch","request_id":"...","shard_id":"shard-a","revision":2,"documents":[...],"compression":{...},"entry_limit":27,"debug_delay_ms":0}
```

成功回执：

```json
{"ok":true,"type":"batch_result","request_id":"...","shard_id":"shard-a","served_revision":2,"compression":{...},"elapsed_ms_monotonic":0,"summary":{...}}
```

摘要包含按桶排序的倒排词项、标题/正文 TF、标题/正文 DF、文档级 posting、所选编码的载荷、桶内容指纹、文档指纹、分片内容指纹及条目数。分片的并发槽位用非阻塞信号量获取；已占满时不排队，直接回 `E_SHARD_CONCURRENCY_EXCEEDED`。

## 修订仲裁

每个修订显式引用父修订，可替换压缩域或通过 `weight_updates`、`add_buckets`、`remove_buckets`、`encoding` 继承修改。回执的 `served_revision` 与当前修订不一致时：

- 相同修订：计入账本。
- 旧修订在当前祖先链上且回执使用的全部桶域未被编码、桶数或权重改变：以 `accepted_compatible` 计入。
- 改动触及回执桶域、回执来自未来修订或链不连通：拒绝计入，但完整回执仍保存在 `receipt_arbitrations[].raw_receipt`，并记录 `basis` 与 `covered_by`。

业务码示例：`E_RECEIPT_STALE_COMPATIBLE`、`E_RECEIPT_STALE_INCOMPATIBLE`、`E_RECEIPT_FUTURE_REVISION`、`E_RECEIPT_PROFILE_MISMATCH`。

## 定点打分

评分全链路只使用整数：

1. `tf_fp = (tf << fraction_bits) // (tf + saturation_k)`。
2. 用整数重复平方计算定点 `log2((N+1)/(df+1))`，再加一个定点单位。
3. 按查询词项顺序依次计算标题/正文加权项，累加并对当前位宽饱和。
4. 按 `quantum_step` 四舍五入到量化格，排序键为“量化得分降序、`doc_id` 升序、`version` 升序”。
5. 与截断边界原始分差不超过半个量化步长，或落在同一量化值的文档标记 `boundary=true`。

切换 `bit_width` 会改变中间整数饱和点和排序；切换 `quantum_step` 会改变量化结果与边界集合，测试对此有直接断言。

## 预算与调度

调度器按 `shard_id` 稳定顺序维护全局剩余条目预算。每个分片的精确唯一词项数在派发前计算，配额取“分片所需条目、每分片上限、全局剩余”的交集；配额不足的分片不静默成功，而是进入下一批待办队列：

- 全局限额不足：`E_BUDGET_PENDING`。
- 超过每分片条目上限：`E_SHARD_ENTRY_LIMIT`。
- 待办队列保存在快照 manifest 的 `pending_shards`，下一批优先派发；并发只发生在不同分片之间，单个分片不会因调度器而超过自身并发上限。

## 内容指纹与复用

- 文档指纹：规范化文档对象哈希。
- 桶指纹：桶索引、权重、编码、词项与 posting 的哈希。
- 分片指纹：分片 ID、修订、压缩指纹、文档指纹集合、桶指纹集合的哈希。

单文档追加/替换/删除只标记该文档旧词项和新词项触达的桶；压缩桶数/权重/编码变化会标记相关分片的全部桶。未受影响分片直接从已提交快照读取，未受影响桶对象逐字节复用；受影响桶重算后再重算分片指纹。每个动作按稳定顺序写入 `reuse.records`，包含 `reuse/recompute`、依据、来源快照和受影响桶。

## 快照与恢复

每次构建先写 `snapshots/.staging-*`：逐分片摘要、`ledger.json`、`recall.json`、`report.json`、`manifest.json`，并 fsync 文件与目录。完成后先把暂存目录重命名为不可变快照目录，再用原子 `os.replace` 切换 `CURRENT.json` 指针：

```text
snapshots/
  CURRENT.json
  snap-00000001-<config-prefix>/
  .staging-snap-...        # 强制终止时留下的孤儿
```

manifest 记录全局版本、配置哈希、索引修订、压缩域、已完成文档游标、已完成/待办分片、已消耗预算、文档指纹、分片规格、评分与查询配置。启动时：

- 无当前快照：`E_RECOVERY_NO_SNAPSHOT`，重建。
- 配置哈希变化：`E_RECOVERY_CONFIG_CHANGED`，重建。
- 版本变化或文档变化：决定续跑并记录 `E_RECOVERY_VERSION_CHANGED` / `E_RECOVERY_RESUME`。
- 发现暂存孤儿：保留诊断并记录 `E_RECOVERY_ORPHAN_TEMP`，不会把未发布内容当账本。
- 完全匹配：`E_RECOVERY_CURRENT`。

## 一致性与验收

`engine.equivalence_projection` 比较增量与空白全量重建的业务字段：倒排桶内容与指纹、分片/总条目数、预算消耗、失败清单及顺序、召回候选/选中顺序、被拒绝回执的裁决依据。时间戳、实际网络回执数量、以及“本次从快照复用了哪些桶”的执行跟踪单独审计，不污染业务账本。

最近一次本地真实验收结果：

- 基线增量与全量等价：`true`，总条目 `84`。
- 6 步编辑序列（追加、替换、删除、容量调整、压缩档位切换、修订回滚）逐步等价：`true`。
- 单文档替换观察到桶级复用和未受影响分片复用：`true`。
- `workers=1` 与 `workers=8` 逐字段业务等价、失败顺序相同：`true`。
- 预算不足：`E_BUDGET_PENDING` 共 `2` 次，保留 `shard-b/shard-c` 待办。
- 每分片硬上限：`E_SHARD_ENTRY_LIMIT` 共 `3` 次。
- 过期修订回执：`E_RECEIPT_STALE_INCOMPATIBLE` 共 `1` 次，`covered_by=[2]`，原始回执保留。
- 强制终止恢复：发现孤儿暂存目录，判定 `resume`。

完整数据见 `artifacts/verification/report.json`；运行输出见
`artifacts/verification-run.out`，pytest 输出见 `artifacts/pytest.txt`，
业务码汇总见 `artifacts/verification/business-codes.json`，分片业务日志位于各构建目录的
`services/<shard_id>/shard.log`。
