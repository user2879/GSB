# 设计与算法说明

## 1. 范围与不变量

系统只处理本地确定性输入，不建立网络连接，也不模拟真实 broker。核心不变量是：

1. 三种历史格式归一后具有同一语义模型。
2. 结论只依赖记录集合语义，不依赖 JSONL 物理行序或消费者组遍历顺序。
3. 主流程与独立校验器从相同原始输入得到逐字节相同的规范 JSON 输出。
4. 每个主题批次在 SQLite 中原子提交；未提交主题不产生可信结论。
5. 参数版本变化时，旧视图整体陈旧化，新视图整体替换，不逐条修补。

## 2. 模块边界

| 模块 | 职责 |
|---|---|
| `mwlab/config.py` | 严格读取主题、组、会话、确认策略与路径；生成参数指纹 |
| `mwlab/records.py` | v1/v2/v3 解码、强校验、来源行号保留、归一模型摘要 |
| `mwlab/semantics.py` | 组细胞推导、全局汇总、冲突归因、稳定排序 |
| `mwlab/stats.py` | 整数定点比例、误差界、主题比较与稳定决胜 |
| `mwlab/artifacts.py` | 规范 JSON/JSONL 产物、摘要、预期摘要比对 |
| `mwlab/ledger.py` | SQLite 账本、主题事务、参数版本、恢复重放 |
| `mwlab/snapshot.py` | 快照摘要、只读备份、整体恢复 |
| `mwlab/independent.py` | 不导入主记录/语义/统计模块的独立归一与校验 |
| `mwlab/cli.py` | 命令、业务码、结构化错误日志 |

## 3. 记录格式与归一

JSONL 每行使用信封：

```json
{"format_version":"v2","consumer_group":"billing","record":{...}}
```

信封的 `consumer_group` 是 v1 的组上下文；v2/v3 的载荷还必须带相同组名。这样 v1 载荷仍严格只含 `message_id`、`topic`、`acknowledged`，同时可以在混合文件中归属消费者组。

| 版本 | 必需载荷字段 |
|---|---|
| v1 | `message_id`, `topic`, `acknowledged` |
| v2 | v1 + `delivery_time_ms`, `retry_count`, `consumer_group` |
| v3 | v2 + `idempotency_key`, `delivery_sequence_range`, `semantic_verdict` |

强校验在解码阶段完成：

- 版本未知或与组声明不一致：`E_FORMAT`。
- 字段缺失、多余、类型错误：`E_FORMAT`，详情含 `line` 和 `field`。
- `retry_count < 0` 或区间 `start < 0`：`E_FORMAT`。
- 区间 `start > end`：`E_FORMAT`。
- 消息标识不匹配主题的序号模板：`E_FORMAT`。
- 非法 v3 语义标记：`E_FORMAT`。

内部模型保留 `source_format`、`source_group` 和 `source_line`。模型摘要按规范语义字段排序，刻意排除行号，因此原始行乱序不改变模型摘要；行号只用于回溯。

## 4. 语义推导

### 4.1 组细胞

对每个 `(topic, message_id, consumer_group)`：

```text
record_count = 组内记录数
retry_max    = max(retry_count)
attempts     = max(record_count, retry_max + 1)
upper        = attempts - 1
```

v3 区间先裁剪到 `[0, upper]`、排序并合并。只有至少存在一个区间时才计算未覆盖区间；v1/v2 缺少区间字段不等于“区间未覆盖”。

精确一次候选要求（由主题确认策略控制是否启用）：

- 至少一条确认记录；
- 至少一个非空幂等键；
- 投递区间完整覆盖 `[0, upper]`。

否则：

- 有确认、重投或真实未覆盖区间：`at_least_once`；
- 单次、未确认、无区间数据：`at_most_once`。

若策略指定 v3 语义标记为权威且组内标记唯一，则使用该标记，并在依据中记录 `marker_applied=true`。

### 4.2 全局汇总

组结论列表按组名排序后汇总：

- 所有组均 `exactly_once`：全局 `exactly_once`；
- 所有组均 `at_most_once`：全局 `at_most_once`；
- 其他情况：全局 `at_least_once`。

全局结论依据包含组结论列表、来源格式、来源组、重投最大值、幂等键和投递区间。它不包含行号或输入文件路径，因此行序变化不改变字节。

### 4.3 冲突

- `ACK_WITH_UNCOVERED_RANGE`：确认存在、存在区间数据、未覆盖区间非空，且策略不允许该情况。
- `RETRY_LIMIT_EXCEEDED`：组内最大重投次数超过主题策略上限。
- `DUP_IDEMPOTENCY_ACROSS_GROUPS`：同一幂等键出现在多个消费者组；每个涉及消息生成一条归因。
- `SEMANTIC_MARKER_MISMATCH`：v3 标记与推导出的组结论不一致。

冲突排序键为 `(message_seq, conflict_type, topic, message_id)`；所有集合先排序再序列化，不依赖字典或哈希遍历顺序。

## 5. 定点统计与误差

给定非负整数 `n`、`d` 和缩放因子 `S=10^k`：

```text
fixed = floor(n*S/d)
0 <= (n/d)*S - fixed < 1
```

因此单值比例误差严格小于 `1/S`。两个定点值 `a`、`b` 的量化误差差值绝对值严格小于 2 个定点单位；比较判定采用保守界：当 `abs(a-b) <= 2` 时标记受误差影响。此时按主题字典序决胜。

主题排名先按区间不确定性排序：若两主题定点差大于 2，则高值在前；否则按主题字典序。输出保留 `stable_rank`、`error_affected` 和每个受影响主题对。

## 6. SQLite 账本

主要表：

- `parameter_versions`：单调版本号、参数指纹、参数文档。
- `executions`：一次运行、参数版本、模型摘要、状态、内容摘要。
- `batches`：主题批次摘要和陈旧位。
- `records`：按主题提交的归一记录。
- `conclusions` / `group_conclusions` / `conflicts`：结论和归因。
- `group_stats` / `topic_stats`：定点值、误差界、冲突计数。
- `cursors`：主题最后消息序号和批次游标。
- `view_diffs`：相对上一当前视图的新增、删除、变更条目。

`commit_topic` 在单个事务中写本主题记录、结论、组结论、冲突、统计、批次和游标。`start_execution` 在开启新当前执行时把旧批次置为陈旧，并创建新的参数版本或复用未变化的版本。参数版本只在当前参数指纹变化时递增，且一旦变化绝不回退。

内容摘要忽略执行 ID、时间戳、展示用视图差异和批次编号，保留参数版本、格式集合、归一记录、结论、冲突和统计。这样相同语义状态可跨执行比较。

## 7. 快照与恢复

快照目录包含：

- `snapshot.json`：状态、内容摘要、格式版本集合、参数版本。
- `ledger.sqlite3`：一致性 SQLite 备份。

保存时使用 SQLite backup API，不直接复制正在写入的文件。同名快照覆盖前，旧目录复制到 `backup/` 并去除写权限。

恢复流程：

1. 校验 `snapshot.json` 的状态摘要；篡改或注入摘要不一致立即 `E_SNAPSHOT`。
2. 读取当前账本的格式集合和内容摘要；差异写入 `mismatch_basis`。
3. 将当前账本备份到只读目录。
4. 原子替换 SQLite 文件，复制整个快照状态，不做增量合并。
5. 重新打开并验证恢复后的内容摘要。

## 8. 强制终止与重放

`run --crash-after-topic N` 在提交前 N 个主题后直接调用 `os._exit(91)`，不写最终产物。账本中只会出现完整主题批次；当前执行状态保持 `running`。

`recover` 只重推已提交主题的账本记录，列出缺失主题及原始记录中对应消息为 `unknown_uncommitted_topic`。若状态未完成，`exit_report_available=false`。完成后重推会核对：

- 归一模型摘要（仅完整执行）；
- 全局结论；
- 组结论；
- 冲突；
- 组统计和主题统计。

任一差异抛出 `E_REPLAY`，包含首个消息标识、字段、两侧值和“基于账本记录与参数版本重推”的依据。

## 9. 独立校验

`mwlab/independent.py` 不导入 `records.py`、`semantics.py`、`stats.py`。它独立读取原始 JSONL，完成自己的严格归一、组细胞、全局汇总、冲突与定点统计，然后比较：

- 归一模型摘要；
- `conclusions.jsonl` 规范字节；
- 组结论、冲突和统计的规范 JSON。

摘要不同说明归一阶段分歧，返回 `origin=format_normalization`；摘要一致但结论/统计不同说明语义阶段分歧，返回 `origin=semantic_derivation`。首个差异包含消息标识、字段和两侧值。

## 10. Windows 注意事项

- 使用 `pathlib` 与 `os.replace`，不依赖 POSIX 路径。
- 文件摘要与输出统一 UTF-8、`\n` 和规范 JSON。
- 只读备份通过去除写权限位实现；Windows 会映射为只读属性。
- SQLite 使用 `journal_mode=DELETE` 与 `synchronous=FULL`，避免依赖 WAL 旁路文件。
- `tkinter` 只在 `selfcheck` 中导入；核心实验与测试不依赖显示服务器。

