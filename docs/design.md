# vscan 技术设计与确定性规则

## 1. 规范与校验

规范根对象只允许 `format_version`、`segments`、`queries`、`execution` 四个字段。
`format_version` 固定为整数 `1`。任何未知字段、缺失字段、错误类型、重复段标识、
重复列名、错误编码、非法行数、非法批行数/缩放因子/预算/并发数均抛出 `E_SPEC`。
错误结构始终包含：

```json
{"code":"E_SPEC", "segment_id":"s1", "field":"rows", "message":"..."}
```

查询引用的列若不存在，错误定位到排序后第一个缺列的目标段。分组列必须属于声明投影。
聚合源列可作为隐式扫描投影，不要求重复写入 `project`。JSON 对象重复键也会被拒绝。

规范化顺序：

- 段按 `id`、列按 `name`、查询按 `id` 排序。
- 投影、分组列去重排序；聚合按名称排序。
- `and`/`or` 递归展平、按规范 JSON 排序并去除重复子表达式；`in` 值去重升序。
- 计划结构摘要不包含 `batch_rows`，因此改变批行数不改变结构摘要。

## 2. 值生成与编码

每段有固定整数 `seed`。每列以 `SplitMix64(mix64(segment.seed XOR sha256(column.name)[:8]))`
为默认种子；列可显式覆盖 `seed`。每行依次消耗两个 64 位随机数：一个用于空值判定，
一个用于在 `[value_min, value_max]` 闭区间取整数值。相同声明总是得到相同值序列。

所有列文件共享 32 字节头：

```text
magic "VSC1" | u16 version=1 | u8 encoding | u8 flags=0
| u64 rows | u64 non_null_count | u64 payload_len
```

最后 32 字节是 `SHA-256(header || payload)`。整数一律 little-endian signed int64。

- `raw`：所有行的 int64 值（空值写 0），随后是 LSB-first 有效性位图。
- `dict`：`u32 dict_count | u8 code_width | 3 bytes zero | 升序 int64 字典 | code 数组`。
  代码宽度为 1/2/4/8 字节；最大值保留为空值代码。字典顺序是显式数值升序。
- `rle`：`u64 run_count`，随后每个游程 `int64 value | u64 length | u8 is_null`。
  游程严格按原始行序和最长相邻相等序列生成，空值也形成游程。

`read_rows` 不展开整列：raw 按索引取值和位，dict 只映射选中 code，RLE 对游程结束位置
二分定位。因此谓词列按批读取，投影/聚合列只读取过滤后候选行。

列摘要覆盖名称、编码、文件摘要、行数和非空数；段摘要覆盖段标识、行数、种子和有序列摘要；
快照摘要覆盖有序段摘要。加载顺序和批切分不会影响这些字节或摘要。

## 3. 批级计划与裁剪

每段行区间为 `[k*batch_rows, min((k+1)*batch_rows, rows))`，首尾相接、无重叠、无遗漏。
每批：

- `output_columns = declared_project ∪ aggregate_source_columns`
- `decode_columns = output_columns ∪ predicate_columns`
- `pruned_columns = segment_columns - decode_columns`

计划摘要给出每段解码列、裁剪列、谓词规范表达式和来源依据。结构摘要与谓词/投影书写顺序、
批行数无关；实际批边界由段标识序和批序稳定产生。

## 4. NumPy 过滤与定点语义

谓词在每个批上一次性计算为布尔 `np.ndarray`，再整体过滤。比较与空值规则：

- 普通比较遇到 `NULL` 为假；`is_null`/`is_not_null` 直接检测位图。
- `and`/`or` 对所有叶节点求布尔数组后合并；`not` 取反布尔数组。
- 空值分组键作为一个显式分组参与聚合。

所有数值是 signed int64 定点整数；`scale` 是定点单位元数据，属于
`{1,10,100,1000,10000}`，不做隐式浮点换算。

- `count(*)`：行数；`count(col)`：非空数。
- `sum`：跳过空值，内部以 Python 任意精度整数精确合并，输出时饱和到 int64。
- `min`/`max`：跳过空值；该组没有非空值时值为 `null`。
- `avg`：跳过空值，`trunc_div_toward_zero(sum, non_null_count)`，再饱和到 int64。
  向零截断，例如 `-5/2 -> -2`。
- 全空或空组的 `sum` 与 `avg` 为 0；`count` 为 0；`min/max` 为 null。
- 每个聚合状态显式记录 `count_rows`、`non_null_count` 和 `null_count`。

批局部结果按组键规范排序后生成摘要；查询合并时再次按组键规范排序。因此线程完成顺序和批
遍历顺序不改变聚合结果或摘要。

## 5. 收益归因

解码字节单位固定为每单元格 9 字节：8 字节 int64 + 1 字节有效性。对每批：

```text
U = rows * all_columns * 9                         # 未裁剪基线
P = rows * decode_columns * 9                      # 仅列裁剪
A = rows * predicate_columns * 9
  + passed_rows * (decode_columns - predicate_columns) * 9   # 列裁剪 + 下推

列裁剪节省 = U - P
谓词下推节省 = P - A
批次划分节省 = 0
```

谓词下推的“候选行”为过滤前批行数 `rows` 和过滤后 `passed_rows`；比较次数为
`候选行数 * 谓词叶节点数`。另输出实际向量比较次数 `rows * leaf_count`。
批划分不直接减少字节或比较次数，它改变工作集和调度并行度，因此贡献显式记为 0。

贡献比例使用整数 ppm，不使用浮点。输出两套口径：

- `floor`：逐项向下取整。
- `largest_remainder`：最大余数法，总和精确为 1,000,000。

稳定规则选择 `floor`。若两套口径最大差值不超过
`execution.attribution_tolerance_ppm` 且差值非零，设置 `tolerance_affected=true` 并保留
两套原始整数结果。

## 6. 并发与内存准入

批任务真实提交给 `ThreadPoolExecutor`。调度循环同时满足：

- 在途批次数 `<= max_concurrency`。
- 所有在途批的保守准入解码字节之和 `<= segment_decode_memory_bytes`。
- 每个段的在途批解码字节之和也 `<= segment_decode_memory_bytes`。
- 同一调度轮每个批键最多提交一次。
- `ready=false` 的批不会进入线程池。

准入字节是批内 `decode_columns * rows * 9` 的最坏值；实际执行可利用下推只解码部分行。
任务排队时记录单调时钟毫秒的 `wait_start_monotonic_ms`、`wait_end_monotonic_ms` 和差值，
并随批结果原子写入账本。统计结果给出并发峰值、总解码字节峰值和逐段峰值。

## 7. 增量失效与重算

批指纹覆盖批计划摘要、批边界、缩放因子、段行数和每个解码列在本批行区间上的值/空值位图摘要。
因此单列单值改变只改变包含该行的批指纹，而不是整列所有批。

传播链：

1. 列值改变 -> 该段该列文件/列摘要改变。
2. 查询引用该列且包含该段 -> 查询失效。
3. 编辑行所在批 -> 具体批失效。
4. 从失效批局部聚合重建查询聚合与归因。

投影/谓词改变只失效目标查询；批行数改变失效所有旧/新批。被编辑但没有任何查询引用的列只改变
列摘要，不重算批。报告按段标识、批号稳定输出失效集和每步依据。

`Session.verify_against_full_rebuild()` 在临时目录按同一规范、覆盖值和空批切分重建，比较：

```text
column.encoding, column.non_null_count, column.column_digest, column.file_digest
batch.start_row, end_row, plan_digest, passed_rows, decoded_bytes,
      aggregate_digest, aggregate_json, attribution
query.aggregate_digest, query.attribution
```

首个不一致抛出 `E_DIVERGE`，detail 包含 `field`、`segment_id`、`column`、`batch_no`。

## 8. SQLite 账本、快照替换与恢复

账本模式包含 `meta` 与 `batches` 两张表。每批完成时在同一个 `BEGIN IMMEDIATE ... COMMIT`
事务中写入：

- 批结局、计划摘要和批指纹；
- 通过行数、实际解码字节、聚合摘要/JSON；
- 等待起止与等待毫秒、进度序号和进度游标；
- 仅由已提交行计算的账本内容摘要。

内容摘要不包含单调时钟和墙钟时间，避免非确定性值破坏重放比较；等待数据仍完整落盘。
账本使用 WAL 和 `synchronous=FULL`。

打开账本时校验整数 `format_version`、快照摘要和内容摘要。任一不匹配即关闭旧库、复制出
`ledger.sqlite.backup-<id>`、设为 `0444`，然后以新库整体替换，绝不增量合并。

新批会在执行前写入 `pending`；已提交批重跑时保留旧提交直到新提交原子覆盖。恢复时：

- 缺失、`pending`、非 `committed` 或批指纹不匹配的记录均标为 `unknown` 并要求重跑。
- 指纹匹配的已提交记录可保留，未受影响批不会因其他批中断而丢失。
- 重算批与崩溃前已提交记录的边界、计划摘要、批指纹、通过行数、解码字节和聚合摘要逐项比较。
- 任一差异抛出 `E_DIVERGE`，定位首个段、批和字段。

## 9. 样本与实测

`examples/spec.json` 含 `s_east` 47 行、`s_west` 71 行；每段五种列覆盖 raw/dict/rle，
查询 `q_region` 有双谓词与 count/sum/min/max/avg，`q_product` 有单谓词与
count/sum/min/max。批行数 16，并发上限 3，段解码预算 1200 字节，归因容差 2 ppm。

实现不监听网络端口，也不依赖套接字服务；强制终止验证只启动本地 Python 子进程操作 SQLite，
不涉及容器或外部目标。

实际 `python -m vscan verify` 结果：

```text
batch_count=16
max_observed_inflight=3
max_observed_decoded_bytes=1152
full_rebuild_verification.consistent=true
segments_checked=2
columns_checked=10
batches_checked=16
queries_checked=2
```

实际测试：

```text
$ .venv/bin/pytest -q
46 passed
```
