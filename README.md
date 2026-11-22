# iidx：确定性倒排索引构建、压缩档位调度与可交互诊断

`iidx` 是一个纯本地、无容器、无网络依赖的 Python 应用。它从固定种子样本目录读取文档流与查询流，分批构建倒排块，使用两种真实的变长整数差值编码，按令牌预算与分片配额并发调度，并用内容指纹缓存、阶段化快照和交互式延迟预算完成可重复演示。

## 快速开始

```bash
python3 --version                 # 需要 >= 3.10
python3 -c 'import numpy; print(numpy.__version__)'
python3 -m iidx doctor

python3 -m iidx validate --data sample
python3 -m iidx run --data sample --state .iidx
python3 -m iidx verify --work-dir .iidx/verification
python3 -m iidx interactive --data sample --state .iidx
```

`sample/` 已包含固定种子生成的可重复语料；`python3 -m iidx demo --data sample --force` 可重新生成后再运行。

## 环境与实际验证

- Python：`3.11.0rc1`
- NumPy：`1.26.4`，`python3 -m iidx doctor` 返回 `python_ok=True`、`numpy_ok=True`
- 自动化测试：`21 passed`（`pytest -q`）
- 完整验收：`python3 -m iidx verify --work-dir .iidx/verification`
- 验收报告：`reports/verification-report.json`

验收套件实测结果：

| 场景 | 结果 |
|---|---|
| 初始构建、单文档修正、分片追加、文档删除、档位切换、分块大小变更、预算/档位策略变更 | 7/7 步的增量结果与干净全量重建逐字段一致 |
| 最终构建 | 40 个块、12 个查询；两档字节长度全部不同；全部解码校验通过；档位 1 和 2 均被选中 |
| 并发 | 构建、压缩、查询三个阶段实测最大并发均为 3，配置线程池 3、配额 4 |
| 优雅终止与续跑 | 中断退出码 0，检查点续跑与全量重建一致 |
| 全局预算耗尽 | 产生 `E_BUDGET_PENDING`、待办分片清单与单调递减账本 |
| 非法输入隔离 | 成功保留 36 文档/12 查询，并输出 6 类业务失败码 |

最终验收 canonical digest：`72d4738a6861fa86713771274ce7f15a842c21a906dc9153001a8460c90c6b61`。

## 输入格式

默认文件名如下；应用也支持少量常见别名文件名。

```text
sample/
  documents.jsonl
  queries.jsonl
  vocabulary.txt
  query_versions.json
  config.json
```

文档流每行一个对象：

```json
{"doc_id":"doc-0001","title_terms":["Alpha","ＩＮＤＥＸ"],"body_terms":["cache","alpha"],"shard":"shard-a"}
```

查询流每行一个对象：

```json
{"query_id":"q-001","query_terms":["alpha","index"],"query_stream_version":"v1"}
```

受控词表支持 UTF-8 文本一行一词，也支持 JSON 字符串数组或 `{"terms": [...]}`。查询流兼容表支持数组或映射形式，映射键即声明版本；例如：

```json
{"declared_versions":["v1","v2"],"compatibility":{"v1":"stable","v2":"stable"}}
```

每个文档必须且只能声明一个非空分片标签；多个文档可以共享同一分片。分片标签会做 NFKC、大小写折叠与空白合并，作为稳定分片名。

## 配置语义

| 字段 | 含义 | 约束/默认 |
|---|---|---|
| `chunk_size` | 每个倒排块的词项数 | 正整数，默认 5 |
| `compression_tier` | `auto`、`1` 或 `2` | 默认 `auto` |
| `compression_switch_points` | 全局块序号处切换档位；严格递增，在每个点翻转档位 | 默认 `[3,6]` |
| `batch_max_docs` | 调度批最大文档数 | 正整数 |
| `thread_pool_size` | 线程池大小 | 正整数，不得超过 `concurrency_quota` |
| `concurrency_quota` | 并发配额上限 | 正整数 |
| `global_token_budget` | 全局词项处理令牌预算 | 非负整数 |
| `per_shard_token_quota` | 单分片累计令牌配额 | 非负整数 |
| `snapshot_commit_interval_batches` | 每 N 个调度批提交检查点 | 正整数 |
| `interactive_latency_budget_ms` | 交互命令等待最新结果的毫秒预算 | 非负整数 |
| `fixed_decimal_bits` | 查询定点分数小数位宽 | 1..30 |
| `quantization_tiers` / `quantization_tier` | 量化步长列表与索引 | 默认 `[1,2,4,8]`、索引 1 |
| `low_frequency_threshold` | 档位二过滤的文档频率阈值 | 非负整数 |
| `query_top_k` | 查询默认召回上限 | 正整数 |

配置错误不会终止整个流程：加载器记录 `E_CONFIG_*` 并使用安全回退值。若 `thread_pool_size` 超过配额，会记录 `E_CONFIG_CONCURRENCY_EXCEEDS_QUOTA` 并夹到配额上限。

`auto` 档位按全局块序号工作：块序号依次经过一个切换点时翻转档位。例如切换点 `[3,6]` 表示块 0..2 使用档位 1、块 3..5 使用档位 2、块 6 之后回到档位 1。显式 `compression_tier: 1|2` 会覆盖自动策略。

## 处理阶段

1. **加载校验**：逐行解析 JSONL，严格拒绝重复 JSON 字段、重复 ID、空词项、词表外词项、未声明查询版本与非法配置项；成功条目继续处理。
2. **归一化与调度**：NFKC 全半角折叠、Unicode casefold、空白合并；分片与文档按稳定顺序排列，按批扣减全局和单分片令牌预算。
3. **倒排块构建**：每分片并行构建词项字典序块；块内词项、文档 ID、位置均升序；块指纹由词项、文档与位置内容派生。
4. **压缩写盘**：每块真实生成档位 1 与档位 2 两份二进制，解码逐字段比较，并记录字节数、校验码与过滤计数。
5. **查询召回**：使用选中档位解码后的倒排块并发执行查询，计算定点量化词频分数，按分数降序与文档 ID 升序稳定排序。
6. **快照提交**：所有阶段先写临时目录，完成后整体替换；最终清单记录版本、配置哈希、完成分片、令牌消耗和阶段产物版本。

位置从 `0` 开始，按“标题词项 + 正文词项”的连接顺序编号。

## 压缩格式

两种档位都使用 LEB128 变长整数，文档 ID 按块内字典序分配局部序号并做差值编码，位置也从 `-1` 基点做差值编码。块头包含 magic、格式版本、档位、32 字节块指纹、词项/文档计数与局部文档字典。

- **档位 1**：全部词项使用差值编码。
- **档位 2**：文档频率 `<= low_frequency_threshold` 的词项先从主词项表过滤，块头写入阈值与过滤计数；为保证“解码后逐字段还原”的要求，被过滤词项完整保存在无损恢复附表中。因此两档长度始终不同，且档位 2 不会因过滤而丢失任何召回数据。

每次编码后立即解码，并检查块指纹、词项、文档 ID 与位置列表。失败产生 `E_DECODE_MISMATCH`，带块序号、档位和首个不一致路径；成功记录 `encoded_bytes` 与 `decoded_checksum`。

## 令牌预算与稳定并发

调度单位是稳定顺序下的 `(分片, 批次)`。批次令牌数等于批内文档归一化词项总数。账本逐次记录：

```json
{"deduction_index":1,"stable_order":1,"shard":"shard-a","batch_index":0,"tokens":65,"remaining":1135,"remaining_before":1200}
```

若下一批会超过单分片配额，立即停止该分片并记录 `E_SHARD_QUOTA_EXHAUSTED`；若会超过全局预算，则当前与后续未完成分片进入待办队列，记录 `E_BUDGET_PENDING`。报告的 `pending_queue` 还会列出每个待办分片尚未处理的文档与令牌数。已完成的批次仍会被索引，账本保持单调递减和稳定扣减位次。

块构建、两档压缩、查询召回都使用线程池。每个任务有由分片名、块序号或查询 ID 派生的稳定键；最终结果按稳定键重排，因此线程调度不会改变输出顺序。报告中 `diagnostics.concurrency` 记录实际线程池大小、配额和观测最大并发。

## 缓存、失效与增量一致性

内容寻址缓存位于 `.iidx/cache/`：

- 归一化结果：原始文档指纹 + 词表指纹 + 归一化版本。
- 倒排块：块内容指纹 + 分片 + 索引配置指纹。
- 每份压缩产物：块缓存键 + 档位 + 压缩配置指纹。
- 查询结果：查询定义 + 引用块的编号/指纹/档位 + 相关文档长度 + 查询配置指纹。

缓存键不含运行时间或线程调度信息。未失效块直接读取持久化块与二进制产物；命中块不会调用编码器，报告中的 `encoding_calls` 保持为 0。文档修正只会让位置/文档集合确实变化的块重算；分块大小或档位变更会让布局或编码键变化。

复用与重算清单按新块序号排序，失效旧块按旧块序号排序，查询清单按查询 ID 排序。增量运行与干净全量重建的 canonical 比较覆盖：

- 逐块词项、文档 ID、位置、文档数、位置总数、块序号、块指纹
- 两档字节长度、选中档位、过滤计数、解码校验码、块与压缩产物缓存键
- 逐查询召回顺序、定点分数、匹配词项、引用块集合、结果校验码与查询缓存键
- 压缩档位决策、预算账本、失败清单

缓存复用/失效的“来源”本身依赖运行前的缓存状态：干净全量运行没有可复用缓存，而增量运行可能有。因而验收将缓存来源清单排除在 canonical digest 之外，但要求由这些清单驱动得到的最终索引、查询、账本与失败结果逐字段一致；这一边界在验收报告中有明确记录。

## 快照与恢复

```text
.iidx/
  cache/                    # 内容寻址的归一化、块、压缩、查询产物
  snapshots/v00000001/      # 仅完整提交的快照
    stage-00-loading/
    stage-01-build/
    stage-02-compress/
    stage-03-query/
    manifest.json
    report.json
  checkpoints/              # 原子提交的调度检查点
    current.json
  reports/latest.json
  current.json              # 原子替换的当前快照指针
```

每个阶段与最终快照都在同文件系统临时目录中完成后再替换。当前指针校验版本、目录名、清单状态与 manifest 哈希。配置哈希不一致、指针/清单不一致或显式 `--expected-version` 不匹配时，报告记录 `E_SNAPSHOT_STALE` 并重建受影响阶段。阶段产物版本相同的阶段标记为可复用。

调度器按 `snapshot_commit_interval_batches` 提交检查点。CLI 收到 `SIGTERM`/`SIGINT` 时设置停止事件；调度器完成当前批次、提交检查点并以 `0` 退出。下次同配置、同输入指纹启动时从已完整替换的检查点继续，最终账本、块与查询结果与干净全量运行一致。

## 交互命令

```bash
python3 -m iidx interactive --data sample --state .iidx
```

```text
status
run
tier auto|1|2
chunk-size 4
budget 1800
quota 650
threads 3
batch-size 4
switch-points 3,6
latency 180
append-doc {"doc_id":"doc-9999","title_terms":["alpha"],"body_terms":["index"],"shard":"shard-a"}
update-doc doc-0002 {"body_terms":["alpha","index","query"]}
delete-doc doc-0007
query q-001
blocks
blocks 3
ledger
compression
failures
report
quit
```

修改档位、分块大小、预算等参数后会启动后台重建。命令最多等待 `interactive_latency_budget_ms`；若最新参数下结果尚未完成，立即返回最近一次完整摘要，并附带 `E_INTERACT_BUDGET` 与 `pending=true`。摘要包含完成分片数、块数、档位和受影响分片。后台完成后，下一次 `status` 即返回最新结果。

交互修改会持久化到 `.iidx/runtime_config.json`，后续 CLI 运行默认沿用；可用 `--ignore-runtime` 忽略。

## CLI 与报告

```bash
python3 -m iidx validate --data sample --json
python3 -m iidx run --data sample --state .iidx --tier 2 --chunk-size 4 --json
python3 -m iidx run --data sample --expected-version 3
python3 -m iidx verify --work-dir .iidx/verification --json
python3 -m iidx doctor --json
```

每次正常或优雅中断运行都写入 `.iidx/reports/latest.json`，完整快照内另有 `report.json`。主要字段：

- `blocks[]`：块序号、分片、词项、文档数、位置总数、块指纹、两档字节数、解码校验码、过滤计数、缓存键与复用状态。
- `queries[]`：逐查询分数、召回文档、引用块、定点量化参数、结果校验码与缓存状态。
- `compression_decisions[]`：选中档位、自动切换点或覆盖原因、两档长度差与过滤数。
- `budget_ledger[]`：稳定扣减位次、分片、批次、令牌数、扣减前后剩余。
- `reuse`：稳定排序的复用、重算与失效块/查询清单及复用率。
- `failures[]`：按 `stage, stable_key, code, message` 排序的业务失败清单。
- `diagnostics`：快照事件、检查点、实际并发、缓存指标、阶段复用、输入指纹与可复现语义。

常用失败码：`E_DOCUMENT_ID_DUPLICATE`、`E_DOCUMENT_TERM_OUT_OF_VOCAB`、`E_DOCUMENT_TERM_EMPTY`、`E_SHARD_TAG_AMBIGUOUS`、`E_QUERY_ID_DUPLICATE`、`E_QUERY_VERSION_UNSUPPORTED`、`E_CONFIG_*`、`E_SHARD_QUOTA_EXHAUSTED`、`E_BUDGET_PENDING`、`E_DECODE_MISMATCH`、`E_SNAPSHOT_STALE`、`E_INTERACT_BUDGET`。

退出码：正常完成、条目级失败被隔离、优雅中断检查点提交均为 `0`；不可恢复的命令行/运行异常为 `2`；验收套件失败为 `1`。

## 测试

```bash
pytest -q
```

测试覆盖：配置失败码与回退、归一化、JSONL 非法条目隔离、版本声明、两种压缩往返与长度差异、解码 mismatch 路径、预算耗尽、分片配额、账本稳定顺序、重复运行全复用、档位切换零重编码、单文档修正部分复用、文档长度变化导致的查询失效、增量/全量一致、阶段快照清单、优雅停止续跑和交互延迟预算。
