# 类别交叉编码复核报告

- 状态：`completed_with_pending`
- 配置/表指纹：`bundle:ae1c587ff67bc123bae71c3d9e65b9b9925ce007cabb424bc7ca6ff6b609018e`
- 位宽档位：`{"encoding_width_bits": 8, "fixed_point_bits": 16, "quantization_levels": 16}`
- 快照版本：global_version=`1`, command_cursor=`1`
- 业务码计数：`{"E_POINT_BUDGET_PENDING": 4}`

## 逐组合裁决

| # | 组合 | 阶数 | 唯一码 | 碰撞 | 点预算 | 裁决 | 指纹 |
|---:|---|---:|---:|---:|---|---|---|
| 0 | `channel` | 1 | 5 | 67 | `admitted` | `CARDINALITY_RETAINED` | `combination:182d4658dae0` |
| 1 | `city` | 1 | 6 | 66 | `admitted` | `CARDINALITY_RETAINED` | `combination:fe9449ae6316` |
| 2 | `device` | 1 | 5 | 67 | `admitted` | `CARDINALITY_RETAINED` | `combination:c079099bb64f` |
| 3 | `membership` | 1 | 4 | 68 | `admitted` | `CARDINALITY_RETAINED` | `combination:870cee359102` |
| 4 | `channel|city` | 2 | 21 | 51 | `admitted` | `CARDINALITY_TRUNCATED` | `combination:79c2bf200580` |
| 5 | `city|device` | 2 | - | - | `pending` | `POINT_BUDGET_PENDING` | `combination:d45c59d77435` |
| 6 | `device|membership` | 2 | - | - | `pending` | `POINT_BUDGET_PENDING` | `combination:30e16caabd9f` |
| 7 | `channel|city|device` | 3 | - | - | `pending` | `POINT_BUDGET_PENDING` | `combination:58fc11b841a1` |
| 8 | `city|device|membership` | 3 | - | - | `pending` | `POINT_BUDGET_PENDING` | `combination:977974008527` |

## 待办队列

5. `city|device` -> `E_POINT_BUDGET_PENDING`
6. `device|membership` -> `E_POINT_BUDGET_PENDING`
7. `channel|city|device` -> `E_POINT_BUDGET_PENDING`
8. `city|device|membership` -> `E_POINT_BUDGET_PENDING`

## 失效与复用统计

- 总计：`{"pending": 8, "recomputed": 28}`
- `enumeration`: `{"recomputed": 9}`
- `coding`: `{"pending": 4, "recomputed": 5}`
- `cardinality`: `{"pending": 4, "recomputed": 5}`
- `adjudication`: `{"recomputed": 9}`

## 翻转归因

- `channel`: `{"hash_salt_change": 4, "point_budget_ruling_order_change": 2, "truncation_bucket_count_change": 1}`
- `city`: `{"hash_salt_change": 4, "point_budget_ruling_order_change": 2, "truncation_bucket_count_change": 1}`
- `device`: `{"hash_salt_change": 4, "point_budget_ruling_order_change": 2, "truncation_bucket_count_change": 1}`
- `membership`: `{"hash_salt_change": 4, "point_budget_ruling_order_change": 2}`
- `channel|city`: `{"hash_salt_change": 4, "point_budget_ruling_order_change": 2, "truncation_bucket_count_change": 1}`
