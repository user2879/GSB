# 技术设计

## 模块边界

| 模块 | 职责 |
| --- | --- |
| `dataset.py` | 读取 JSON、强校验、表摘要、确定性倾斜放大 |
| `query.py` | WHERE 词法/语法解析、类型校验、查询规范化与 `query_id` |
| `execution.py` | 分片计划、逐行过滤、分片聚合、固定点结果合并 |
| `scheduler.py` | 并行度/内存双预算调度、单调时钟轨迹、热点判定与二次拆分 |
| `ledger.py` | SQLite 模式、批次持久化、语义账本摘要、替换审计 |
| `snapshot.py` | 快照构造、ID 校验、版本比较、原子写入 |
| `replay.py` | 分片级重执行、首个差异定位、最终结果比较 |
| `runner.py` | `run`、`resume`、`replay` 的端到端编排 |
| `cli.py` | Windows 可调用的命令行与业务退出码 |

## 输入校验

加载过程先构造完整不可变 `Table`/`Query`，任何失败都不会产生部分运行状态。校验顺序覆盖：

1. JSON 语法与根类型；JSON 错误位置保留 `file:line:column`。
2. 表名/列名标识符与唯一性、表版本、行数、列集合。
3. 类型、`fixed.scale`、空值位图长度/取值、值数组长度、空值与类型一致性。
4. 源表存在且不重复，分组列与聚合列在每张源表中存在，跨表类型/scale 一致。
5. WHERE 列存在，字面量类型与比较运算兼容。
6. 聚合函数与列类型兼容：`sum/avg` 只允许 `int/fixed`，`*` 只允许 `count`。

业务错误对象包含 `code`、`message`、可选 `location`、`task_id`、`field` 和结构化 `details`。

## 查询规范化

WHERE 文法：

```text
expr        := term (AND term)*
term        := '(' expr ')' | comparison
comparison  := identifier ('=' | '>' | '>=' | '<' | '<=') literal
literal     := integer | 'string'
```

解析结果中 AND 递归展平并按规范 JSON 排序，因此 `((a=1) AND (b>=2))`、`b >= 2 AND a = 1` 和额外括号得到相同 AST。查询标识的规范对象由排序后的源表名、WHERE AST、按原顺序的 `group_by`、按名称排序的聚合列表组成，再做 SHA-256。

分片计划不含查询文本，只含：

- 稳定任务 ID：`<table>@v<version>:r<start:012d>-<end:012d>`；
- 行区间、表版本、表摘要、排序后的引用列；
- 基础任务是否允许拆分。

计划摘要对以上基础任务规范 JSON 做 SHA-256。

## 分片执行

每个任务只读取 `row_start..row_end`。任务状态：

```text
groups[key] = {
  aggregate_name: {"count": integer, "sum": integer?}
}
```

- `count *` 增加 `count`；
- `count column` 仅在值非空时增加 `count`；
- `sum/avg` 仅在值非空时增加 `count` 与整数 `sum`。

任务摘要对任务 ID、表版本/摘要、行区间、扫描行数、完成行数、规范聚合状态做 SHA-256。任务结果不包含浮点数或执行时间，因此摘要可重放。

## 倾斜判定与拆分

过滤完成后统计该分片内完整分组键的行数。所有 `count >= split_threshold` 的键都记录到判定依据，触发键按 `(-count, canonical_key_json)` 排序取首项，保证并列时确定。

只有基础分片可再拆一次。拆分策略固定为显式 `row_range`：

1. `k = min(split_parts, parent_row_count)`，要求 `k >= 2`；
2. 将父区间均分，余数从前到后各多 1 行；
3. 子任务 ID 追加 `/s01`、`/s02` 等；
4. 运行时断言首子区间从父起点开始、相邻子区间端点相等、末子区间止于父终点。

父分片状态为 `split_parent` 且 `included=false`；子分片状态为 `completed` 且参与合并。若子分片仍被识别为热点，只记录 `observed_after_split`，不递归拆分。若热点数超过 `skew_safety_limit`，记录 `rejected_safety` 后抛 `E_SKEW`。

## 双重预算调度

主调度线程维护 `pending`、`active`、`used_bytes`。启动条件同时满足：

```text
len(active) < parallelism
used_bytes + task.estimate_bytes <= effective_memory_budget
```

内存估算：

```text
estimate = 64 + row_count * sum(width(column) for column in referenced_columns)
width(int|fixed) = 9
width(string) = 1 + max(UTF-8 byte length of non-null values)
```

任务通过线程池真实执行。调度器只等待 Future 完成，无执行超时。每次启动/完成都写预算事件。任务首次因并行度或预算不能启动时记一次排队，写入 `queued` 预算事件和 `queue_started_ms`，并令 `queue_wait_ms = worker_start_ms - queue_started_ms`。`start_ms`、`end_ms`、`queue_started_ms` 均以调度器启动为原点的单调毫秒。若没有活动任务且最小待运行任务仍超过预算，则任务永远无法运行，立即以 `E_BUDGET` 终止。

## 聚合合并与精度

参与任务按 `task_id` 升序合并。所有分组先按键稳定排序。计数与定点求和是整数加法，分片边界不影响结果。

最终结果：

```text
count = sum(count_i)
sum   = sum(sum_i)                     # 仅非空值
avg   = floor(sum / count_nonnull)     # 全局一次截断
```

`floor` 指向负无穷。对每个分组，均值的绝对误差上界小于 1 个 scale 单位。结果 JSON 使用 `sort_keys=True`、无多余空白、UTF-8、末尾单个换行，因此同一输入在不同并行度和拆分阈值下逐字节一致。

## SQLite 账本

主要表：

- `sessions`：查询标识、数据摘要、表摘要/版本、预算版本/摘要、拆分参数、配置、计划、最终结果。
- `tasks`：任务来源、区间、引用列、拆分父/子关系、状态、完成行数、聚合 JSON、摘要、开始/结束/排队开始时间、排队次数、估算字节。
- `skew_decisions`：动作、触发键、热点计数、阈值、策略、请求/实际拆分数、子区间和全部热点。
- `budget_events`：启动、完成、拒绝事件及当时的活动任务数、占用和预算。
- `batch_state`：批次边界上的聚合中间状态、峰值内存、拆分计数。
- `snapshot_replacements`：快照不兼容时的旧/新会话、原因与比较结果。
- `snapshot_index`：快照 ID、路径、账本摘要和结果摘要。

账本摘要对忽略会话更新时间后的语义字段、任务、倾斜判定和批次状态做规范 SHA-256。快照重放依赖该摘要检测外部篡改。

## 快照与整体替换

快照格式版本为 1，快照 ID 是除 `snapshot_id` 外全部内容的规范 SHA-256。兼容性比较：

- 数据摘要、逐表摘要、逐表版本；
- 查询标识；
- 预算版本、预算摘要、有效内存；
- 拆分阈值、拆分份数、行组、并行度；
- 完整运行配置摘要。

`resume` 发现任一差异时不会复用旧任务或聚合，而是以新运行标识重新加载当前数据、重新生成计划、重新调度，并写入 `snapshot_replacements`。旧会话保留为审计记录，新会话状态独立。

## 重放差异定位

重放从账本读取每个任务的定义和参数字段，使用快照中的预算参数版本和拆分阈值重新执行。比较字段顺序固定：

1. `status`
2. `rows_completed`
3. `aggregate`
4. `digest`
5. `included`
6. 最终 `result_json` / `result_digest`
7. 账本摘要

首个差异立即以 `E_REPLAY` 返回，同时携带 `task_id`、`field`、`expected`、`actual`。因此篡改中间聚合、完成行数或状态都能定位。

## 确定性与并发

线程完成顺序和单调时钟值不参与结果。最终合并只依赖稳定任务 ID、稳定分组键排序和整数运算。调度轨迹中的时间/排队时长允许因机器调度不同而变化，但任务集合、拆分判定、拆分区间、分片摘要、最终结果和恢复比较均稳定。
