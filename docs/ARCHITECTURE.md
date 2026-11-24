# 技术设计

## 1. 本地文件与二进制段

工作区包含：

- `spec.json`：列、行组、块声明、样本值、查询。
- `segment.seg`：确定性二进制段。
- `ledger.sqlite3`：批次账本、任务轨迹、计划与查询结论。
- `workspace.json`：预算与单调版本。
- 会话快照：独立 JSON 文件。

段文件布局：

```text
MAGIC(8) | header_len(u32-le) | canonical_json_header
repeat: record_len(u32-le) | block_header_len(u32-le) | block_header |
        payload_len(u32-le) | payload
sha256(all preceding bytes)(32)
```

块头字段：

```json
{
  "block_id": "amount@rg1",
  "column": "amount",
  "row_group": "rg1",
  "encoding": "dictionary",
  "compressed": true,
  "row_count": 5,
  "uncompressed_bytes": 20,
  "compressed_bytes": 17,
  "payload_digest": "..."
}
```

`block_digest = sha256(canonical_header || stored_payload)`。读取时逐层校验 magic、长度前缀、块头摘要、payload 摘要、编码枚举、行数以及段尾整体摘要。

## 2. 编码稳定性

- `raw`：有符号 64 位小端整数序列。
- `dictionary`：`字典项数` + 按数值升序的字典项 + 行索引。显式排序，不使用 `set`/`dict` 遍历顺序。
- `rle`：`游程数` + 每个 `(行数, 值)`。相同输入产生完全相同的字节串。
- `compressed=true` 时仅对编码 payload 做固定级别 zlib；块头分别记录压缩前与压缩后长度。

同一输入重复编码会得到相同 `block_digest` 与 `record_bytes`。增量更新重写段时，未失效块复用原 `record_bytes`，因此其长度前缀后的块记录逐字节不变。

## 3. 计划与稳定序号

查询 DSL 示例：

```text
region IN (1, 3) AND (amount >= 10)
```

解析器支持 `=`/`==`/`!=`/`>`/`>=`/`<`/`<=`、`IN`、`AND` 和括号。AND 子节点扁平化、去重并按规范 JSON 排序，因此空格与冗余括号不影响 AST 或计划摘要。

计划包含：

- 输出列与谓词列；
- 被裁剪列；
- 每个行组的匹配行数与候选块；
- 稳定推导依据（`column_clip:*`、`row_group:*`、`predicate:and:*`）；
- 计划摘要。

块序号采用“潜在槽位”规则：对每个查询×行组×引用列都保留固定序号，即使当前谓词把该槽位裁掉。这样候选块增减不会让未失效块重编号。

## 4. 三重预算调度

调度器在提交到线程池之前串行执行准入：

1. `len(running) < global_parallelism`
2. `segment_running[segment] < segment_concurrency`
3. `column_memory[column] + task.decode_memory <= column_memory_budget_bytes`

满足后才 `submit` 到 `ThreadPoolExecutor`。worker 无权扩大资源，因此不存在绕过预算路径。任务从进入队列到获准运行记录 `queue_started_ms`/`wait_ended_ms`，并记录运行起止与持续时间；所有时钟来自 `time.monotonic_ns()`。

## 5. 增量失效传播

变更集可修改指定块的 `values`、`encoding`、`compressed`。算法：

1. 直接失效块集合只包含被修改块，仅这些块重新编码。
2. 从查询的 `select` 与谓词列索引找到引用变更列的查询。
3. 对每个受影响查询与变更行组比较旧/新候选状态：
   - 都候选：只重扫直接变更列对应的块任务，其余块任务复用账本结果；
   - 由无到有：为新候选块创建任务；
   - 由有到无：旧任务标陈旧并移除；
   - 仍为空：不创建任务。
4. 仅重算受影响查询×行组结论；其他任务与查询结论完全复用。
5. 输出传播步骤、直接/间接受影响块、重编码块、重扫任务、未失效块/任务/查询结论逐项保持校验。

编码方式改变会递增 `encoding_selection_version`。为满足“旧版本轨迹整体陈旧”，会建立新生成并克隆未失效结论；克隆不重新编码或扫描，因此仍满足仅失效子集重算。

## 6. 账本与权威重放

SQLite 表：

- `generations`：生成、版本、当前/陈旧标记、游标。
- `batches`：完整提交批次、原因、游标范围、版本与段摘要。
- `scan_ledger`：任务、块序号、完成行数、解码内存、四个单调时钟值、块摘要、结果摘要、JSON 统计。
- `query_results`：查询×行组行集合与摘要。
- `plans`：当前查询计划。

一个批次的所有插入和生成游标更新位于一个 SQLite 事务。中途失败会回滚，因而恢复只能看到完整批次。

权威重放不读取账本的摘要用于计算，而是：

1. 按计划顺序逐块从段文件解码；
2. 重新运行同一谓词并计算行掩码、统计、块摘要与结果摘要；
3. 逐项比较完成行数、解码内存、块摘要、结果摘要、结果 JSON 与统计；
4. 普通权威校验的首个差异抛出 `E_DIVERGE`；快照恢复重放使用同一权威算法但抛出 `E_REPLAY`。两者均携带 `block_ordinal`、`field`、账本值与重放值；
5. 再独立合并查询×行组行集合并比较结果摘要与行集合。

## 7. 版本与整体替换

- 修改全局并行度、每列内存预算或段配额：`budget_params_version += 1`。
- 修改编码选择：`encoding_selection_version += 1`。
- 新生成成为唯一 `is_current=1`；旧生成 `status='stale'`，旧批次、轨迹、块结果与查询结果均不参与当前统计。
- 预算变更会从游标 0 重新调度全部任务；编码策略变更仅重编码/重扫受影响块，并克隆未失效结论到新生成。
- 替换前后按稳定结论 ID 分类：任务结果 `task:<task_id>`，查询行组结果 `query:<qid>:<rg>`；输出排序后的 `added/deleted/changed/unchanged`。

## 8. 快照与恢复

快照包含：

- 段集合摘要与每段 `file_digest`/`spec_digest`/格式版本/编码版本；
- 预算参数版本与预算值；
- 扫描序列、完整计划 AST、当前生成、任务轨迹、账本批次；
- 块摘要、结果摘要、查询行组结论与统计；
- `task_index` 游标；
- `content_digest = sha256(canonical_json(除 content_digest 外全部内容))`。

恢复先验证内容摘要，然后在一个账本事务内把当前活动视图整体标陈旧并导入快照生成，不存在增量合并。若段集合、格式/编码版本或扫描序列不一致，导入后以 `E_SNAPSHOT` 返回 `state_replaced=true` 与不匹配明细；预算版本可随快照恢复，随后按该预算与游标继续。

恢复后 `scan_pending` 从快照游标取同一序列后缀继续调度，完成后权威重放比较每个块与查询结论，不一致以 `E_REPLAY` 定位首个块。
