# 类别交叉编码复核报告

- 状态：`completed`
- 配置/表指纹：`bundle:6769dc8348d3e2b98480d0da127d6012fb16d3ba729aab613cfa49752f30fe95`
- 位宽档位：`{"encoding_width_bits": 8, "fixed_point_bits": 16, "quantization_levels": 16}`
- 快照版本：global_version=`1`, command_cursor=`1`
- 业务码计数：`{}`

## 逐组合裁决

| # | 组合 | 阶数 | 唯一码 | 碰撞 | 点预算 | 裁决 | 指纹 |
|---:|---|---:|---:|---:|---|---|---|
| 0 | `channel` | 1 | 5 | 67 | `admitted` | `CARDINALITY_RETAINED` | `combination:6876524ec114` |
| 1 | `city` | 1 | 6 | 66 | `admitted` | `CARDINALITY_RETAINED` | `combination:b03ab998f28f` |
| 2 | `device` | 1 | 5 | 67 | `admitted` | `CARDINALITY_RETAINED` | `combination:ad73f4fd50eb` |
| 3 | `membership` | 1 | 4 | 68 | `admitted` | `CARDINALITY_RETAINED` | `combination:4c86d65f49ce` |
| 4 | `channel|city` | 2 | 22 | 50 | `admitted` | `CARDINALITY_TRUNCATED` | `combination:f931c510ab9f` |
| 5 | `channel|device` | 2 | 22 | 50 | `admitted` | `CARDINALITY_TRUNCATED` | `combination:771d076cb0ad` |
| 6 | `channel|membership` | 2 | 15 | 57 | `admitted` | `CARDINALITY_RETAINED` | `combination:573627653f58` |
| 7 | `city|device` | 2 | 26 | 46 | `admitted` | `CARDINALITY_TRUNCATED` | `combination:b1ff726d1a5c` |
| 8 | `city|membership` | 2 | 18 | 54 | `admitted` | `CARDINALITY_RETAINED` | `combination:cecff116df9a` |
| 9 | `device|membership` | 2 | 13 | 59 | `admitted` | `CARDINALITY_RETAINED` | `combination:12288432e3eb` |
| 10 | `channel|city|device` | 3 | 48 | 24 | `admitted` | `CARDINALITY_TRUNCATED` | `combination:5a84420362ec` |
| 11 | `channel|city|membership` | 3 | 43 | 29 | `admitted` | `CARDINALITY_TRUNCATED` | `combination:28a242d61a2d` |
| 12 | `channel|device|membership` | 3 | 38 | 34 | `admitted` | `CARDINALITY_TRUNCATED` | `combination:e2ba84bee888` |
| 13 | `city|device|membership` | 3 | 46 | 26 | `admitted` | `CARDINALITY_TRUNCATED` | `combination:612d1bad4b22` |

## 失效与复用统计

- 总计：`{"recomputed": 56}`
- `enumeration`: `{"recomputed": 14}`
- `coding`: `{"recomputed": 14}`
- `cardinality`: `{"recomputed": 14}`
- `adjudication`: `{"recomputed": 14}`

## 翻转归因

- `channel`: `{"hash_salt_change": 4, "point_budget_ruling_order_change": 1}`
- `city`: `{"hash_salt_change": 4, "point_budget_ruling_order_change": 1}`
- `device`: `{"hash_salt_change": 4, "point_budget_ruling_order_change": 1}`
- `membership`: `{"hash_salt_change": 4, "point_budget_ruling_order_change": 1, "truncation_bucket_count_change": 1}`
- `channel|city`: `{"hash_salt_change": 4, "point_budget_ruling_order_change": 1, "truncation_bucket_count_change": 1}`
- `channel|device`: `{"hash_salt_change": 4, "point_budget_ruling_order_change": 1, "truncation_bucket_count_change": 1}`
- `channel|membership`: `{"hash_salt_change": 4, "point_budget_ruling_order_change": 1, "truncation_bucket_count_change": 1}`
- `city|device`: `{"hash_salt_change": 4, "point_budget_ruling_order_change": 1, "truncation_bucket_count_change": 1}`
- `city|membership`: `{"hash_salt_change": 4, "point_budget_ruling_order_change": 1, "truncation_bucket_count_change": 4}`
- `device|membership`: `{"hash_salt_change": 4, "point_budget_ruling_order_change": 1, "truncation_bucket_count_change": 4}`
- `channel|city|device`: `{"hash_salt_change": 4, "point_budget_ruling_order_change": 1, "truncation_bucket_count_change": 1}`
- `channel|city|membership`: `{"hash_salt_change": 4, "point_budget_ruling_order_change": 1, "truncation_bucket_count_change": 1}`
- `channel|device|membership`: `{"hash_salt_change": 4, "point_budget_ruling_order_change": 1, "truncation_bucket_count_change": 1}`
- `city|device|membership`: `{"hash_salt_change": 4, "point_budget_ruling_order_change": 1, "truncation_bucket_count_change": 1}`
