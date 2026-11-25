# 类别交叉编码复核报告

- 状态：`completed`
- 配置/表指纹：`bundle:346b77dd285ca9be2d748aa4d7da26d0773507049883a7e10d2660e1ece29e40`
- 位宽档位：`{"encoding_width_bits": 8, "fixed_point_bits": 16, "quantization_levels": 16}`
- 快照版本：global_version=`2`, command_cursor=`2`
- 业务码计数：`{}`

## 逐组合裁决

| # | 组合 | 阶数 | 唯一码 | 碰撞 | 点预算 | 裁决 | 指纹 |
|---:|---|---:|---:|---:|---|---|---|
| 0 | `channel` | 1 | 5 | 67 | `admitted` | `CARDINALITY_RETAINED` | `combination:182d4658dae0` |
| 1 | `city` | 1 | 6 | 66 | `admitted` | `CARDINALITY_RETAINED` | `combination:fe9449ae6316` |
| 2 | `device` | 1 | 5 | 67 | `admitted` | `CARDINALITY_RETAINED` | `combination:c079099bb64f` |
| 3 | `membership` | 1 | 4 | 68 | `admitted` | `CARDINALITY_RETAINED` | `combination:870cee359102` |
| 4 | `channel|city` | 2 | 21 | 51 | `admitted` | `CARDINALITY_TRUNCATED` | `combination:79c2bf200580` |
| 5 | `city|device` | 2 | 24 | 48 | `admitted` | `CARDINALITY_TRUNCATED` | `combination:d45c59d77435` |
| 6 | `device|membership` | 2 | 16 | 56 | `admitted` | `CARDINALITY_RETAINED` | `combination:30e16caabd9f` |
| 7 | `channel|city|device` | 3 | 47 | 25 | `admitted` | `CARDINALITY_TRUNCATED` | `combination:58fc11b841a1` |
| 8 | `city|device|membership` | 3 | 46 | 26 | `admitted` | `CARDINALITY_TRUNCATED` | `combination:977974008527` |

## 失效与复用统计

- 总计：`{"reuse_hit": 36}`
- `enumeration`: `{"reuse_hit": 9}`
- `coding`: `{"reuse_hit": 9}`
- `cardinality`: `{"reuse_hit": 9}`
- `adjudication`: `{"reuse_hit": 9}`

## 翻转归因

- `channel`: `{"hash_salt_change": 4, "point_budget_ruling_order_change": 1, "truncation_bucket_count_change": 1}`
- `city`: `{"hash_salt_change": 4, "point_budget_ruling_order_change": 1, "truncation_bucket_count_change": 1}`
- `device`: `{"hash_salt_change": 4, "point_budget_ruling_order_change": 1, "truncation_bucket_count_change": 1}`
- `membership`: `{"hash_salt_change": 4, "point_budget_ruling_order_change": 1}`
- `channel|city`: `{"hash_salt_change": 4, "point_budget_ruling_order_change": 1, "truncation_bucket_count_change": 1}`
- `city|device`: `{"hash_salt_change": 4, "point_budget_ruling_order_change": 1, "truncation_bucket_count_change": 1}`
- `device|membership`: `{"hash_salt_change": 4, "point_budget_ruling_order_change": 1, "truncation_bucket_count_change": 2}`
- `channel|city|device`: `{"hash_salt_change": 4, "point_budget_ruling_order_change": 1, "truncation_bucket_count_change": 1}`
- `city|device|membership`: `{"hash_salt_change": 4, "point_budget_ruling_order_change": 1, "truncation_bucket_count_change": 1}`
