# 声明文件格式

声明文件是一个 UTF-8 JSON 对象。平台先完成完整解析与校验，再生成任何数据；任何违规均以 `E_SPEC` 拒绝，并在 `location` 中给出第一个违规位置，例如 `tables[0].columns[1].type`、`queries[2].filters[0].op`。

## 顶层字段

| 字段 | 必填 | 默认 | 说明 |
|---|---:|---|---|
| `spec_version` | 否 | `1` | 正整数声明版本 |
| `name` | 否 | `local-workload` | 工作负载名称，仅用于标识 |
| `seed` | 否 | `20261123` | 非负整数，确定性数据种子 |
| `tables` | 是 | - | 非空表声明数组 |
| `queries` | 是 | - | 非空查询数组 |
| `stats` | 否 | 见下 | 统计与抽样口径配置 |
| `cost_weights` | 否 | `rows=1,comparisons=1,lookups=4,io_pages=2` | 正整数/零整数代价权重 |
| `search` | 否 | 穷举 6、审计 12 | 索引搜索切换配置 |
| `execution` | 否 | 预热 1、实测 2 | 真实执行计时配置 |
| `bias` | 否 | `threshold=1.75` | 偏差报告阈值，必须 `>= 1` |

`stats`：

| 字段 | 默认 | 约束 |
|---|---:|---|
| `version` | `stats-v1` | 非空字符串，进入统计口径版本 |
| `row_threshold` | `5000` | `>= 1`；小于它精确采集，否则抽样 |
| `sample_size` | `2048` | `>= 1`；实际为 `min(行数, sample_size)` |
| `confidence` | `0.95` | `(0.5, 1.0)` 开区间 |
| `histogram_buckets` | `16` | `>= 1` |
| `value_count_limit` | `4096` | 不同值不超过上限时保留完整频数 |
| `page_rows` | `100` | 每数据页行数假设 |
| `random_seed` | `20261123` | 非负整数，确定性抽样种子 |

`search`：

| 字段 | 默认 | 说明 |
|---|---:|---|
| `exhaustive_max_candidates` | `6` | 候选数 `<=` 此值以穷举为主，否则贪心 |
| `oracle_max_candidates` | `12` | 不超过此值时额外计算穷举审计真值；必须 `>=` 穷举阈值 |

## 表声明

```json
{
  "name": "orders",
  "row_count": 120000,
  "columns": [],
  "candidate_indexes": []
}
```

- `name` 非空，不得含 NUL/控制字符；同一声明中表名唯一。
- `row_count` 为正整数。
- `name` 不能使用平台保留名 `qplan_meta`，也不能以 SQLite 保留前缀 `sqlite_` 开头。
- `columns` 非空；同一表内列名唯一。
- `candidate_indexes` 可为空；索引名全局唯一，且不能与任何表名冲突。

## 列声明

```json
{
  "name": "status",
  "type": "integer",
  "selectivity_hint": 0.166667,
  "nullable": false,
  "null_fraction": 0.0,
  "generator": {
    "kind": "categorical",
    "values": [0, 1, 2],
    "weights": [0.5, 0.3, 0.2]
  }
}
```

- `type` 支持 `integer`、`float`、`real`；`float` 规范化为 `real`。
- `selectivity_hint` 必须 `> 0` 且 `<= 1`，进入选择性估计的 25% 先验混合。
- `nullable` 默认 `false`；`null_fraction` 取值 `[0, 1)`，非空列必须为 0。
- `generator` 可省略；省略时按类型生成确定性默认值。

生成器类型：

| `kind` | 字段 | 说明 |
|---|---|---|
| `unique_int` | `start` | `start + row_index`，适合主键/外键目标 |
| `uniform_int` | `min`, `max` | 闭区间确定性均匀整数 |
| `categorical` | `values`, `weights` | 无权重时均匀；权重非负且总和 `> 0` |
| `uniform_float` | `min`, `max`, `decimals` | 确定性浮点，`decimals <= 12` |
| `foreign_key` | `table`, `column` | 目标必须是整数 `unique_int` 列，生成合法键值 |
| `constant` | `value` | 常量列 |

所有值由 `seed + 表名 + 列名 + 行号` 经 SplitMix64/BLAKE2b 组合生成，不依赖 Python 随机状态。

## 索引候选声明

```json
{"name": "idx_orders_status_customer", "columns": ["status", "customer_id"]}
```

- `columns` 非空，列必须存在于当前表。
- 同一索引内列名不得重复，数组顺序即索引列序。
- 索引名全局唯一，且进入稳定的候选排序。

## 查询声明

```json
{
  "id": "Q_status_region",
  "tables": ["orders", "customers"],
  "filters": [
    {"table": "orders", "column": "status", "op": "eq", "value": 2},
    {"table": "customers", "column": "region", "op": "eq", "value": 3}
  ],
  "joins": [
    {"left_table": "orders", "left_column": "customer_id",
     "right_table": "customers", "right_column": "id"}
  ],
  "select": [
    {"table": "orders", "column": "id"},
    {"table": "customers", "column": "id"}
  ]
}
```

- `id` 非空且全局唯一。
- `tables` 非空、无重复，所有表必须在顶层声明中存在。
- `filters` 可为空；过滤器表和列必须在查询表集合中。
- 支持算子：`eq`、`ne`、`lt`、`le`、`gt`、`ge`、`between`、`in`。
- `between` 值必须是两个有序同类值；`in` 值必须是非空数组。
- `joins` 中两端表和列必须存在；不支持无别名的自联接；多表查询的联接图必须连通。
- `select` 省略时输出所有查询表的所有列；提供时必须非空且不能重复列。

## 首个违规位置示例

| 违规 | `location` |
|---|---|
| 第 2 个表列类型非法 | `tables[1].columns[0].type` |
| 索引重复列 | `tables[0].candidate_indexes[0].columns[1]` |
| 查询引用不存在表 | `queries[2].tables[0]` |
| 不支持的过滤算子 | `queries[0].filters[1].op` |
| 重名表 | `tables[2].name` |
| JSON 语法错误 | `$` |

