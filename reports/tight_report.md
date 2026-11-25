# 类别交叉编码复核报告

- 状态：`completed_with_pending`
- 配置/表指纹：`bundle:d3567bb2a0a60866dc5cac97cc977c950a64cab178de5785f46d2f7a1cb8cb7a`
- 位宽档位：`{"encoding_width_bits": 8, "fixed_point_bits": 16, "quantization_levels": 16}`
- 快照版本：global_version=`1`, command_cursor=`1`
- 业务码计数：`{"E_POINT_BUDGET_PENDING": 9}`

## 逐组合裁决

| # | 组合 | 阶数 | 唯一码 | 碰撞 | 点预算 | 裁决 | 指纹 |
|---:|---|---:|---:|---:|---|---|---|
| 0 | `channel` | 1 | 5 | 67 | `admitted` | `CARDINALITY_RETAINED` | `combination:6876524ec114` |
| 1 | `city` | 1 | 6 | 66 | `admitted` | `CARDINALITY_RETAINED` | `combination:b03ab998f28f` |
| 2 | `device` | 1 | 5 | 67 | `admitted` | `CARDINALITY_RETAINED` | `combination:ad73f4fd50eb` |
| 3 | `membership` | 1 | 4 | 68 | `admitted` | `CARDINALITY_RETAINED` | `combination:4c86d65f49ce` |
| 4 | `channel|city` | 2 | 22 | 50 | `admitted` | `CARDINALITY_TRUNCATED` | `combination:f931c510ab9f` |
| 5 | `channel|device` | 2 | - | - | `pending` | `POINT_BUDGET_PENDING` | `combination:771d076cb0ad` |
| 6 | `channel|membership` | 2 | - | - | `pending` | `POINT_BUDGET_PENDING` | `combination:573627653f58` |
| 7 | `city|device` | 2 | - | - | `pending` | `POINT_BUDGET_PENDING` | `combination:b1ff726d1a5c` |
| 8 | `city|membership` | 2 | - | - | `pending` | `POINT_BUDGET_PENDING` | `combination:cecff116df9a` |
| 9 | `device|membership` | 2 | - | - | `pending` | `POINT_BUDGET_PENDING` | `combination:12288432e3eb` |
| 10 | `channel|city|device` | 3 | - | - | `pending` | `POINT_BUDGET_PENDING` | `combination:5a84420362ec` |
| 11 | `channel|city|membership` | 3 | - | - | `pending` | `POINT_BUDGET_PENDING` | `combination:28a242d61a2d` |
| 12 | `channel|device|membership` | 3 | - | - | `pending` | `POINT_BUDGET_PENDING` | `combination:e2ba84bee888` |
| 13 | `city|device|membership` | 3 | - | - | `pending` | `POINT_BUDGET_PENDING` | `combination:612d1bad4b22` |

## 待办队列

5. `channel|device` -> `E_POINT_BUDGET_PENDING`
6. `channel|membership` -> `E_POINT_BUDGET_PENDING`
7. `city|device` -> `E_POINT_BUDGET_PENDING`
8. `city|membership` -> `E_POINT_BUDGET_PENDING`
9. `device|membership` -> `E_POINT_BUDGET_PENDING`
10. `channel|city|device` -> `E_POINT_BUDGET_PENDING`
11. `channel|city|membership` -> `E_POINT_BUDGET_PENDING`
12. `channel|device|membership` -> `E_POINT_BUDGET_PENDING`
13. `city|device|membership` -> `E_POINT_BUDGET_PENDING`

## 失效与复用统计

- 总计：`{"pending": 18, "recomputed": 38}`
- `enumeration`: `{"recomputed": 14}`
- `coding`: `{"pending": 9, "recomputed": 5}`
- `cardinality`: `{"pending": 9, "recomputed": 5}`
- `adjudication`: `{"recomputed": 14}`

## 翻转归因

- `channel`: `{"hash_salt_change": 4, "point_budget_ruling_order_change": 2}`
- `city`: `{"hash_salt_change": 4, "point_budget_ruling_order_change": 2}`
- `device`: `{"hash_salt_change": 4, "point_budget_ruling_order_change": 2}`
- `membership`: `{"hash_salt_change": 4, "point_budget_ruling_order_change": 2, "truncation_bucket_count_change": 1}`
- `channel|city`: `{"hash_salt_change": 4, "point_budget_ruling_order_change": 2, "truncation_bucket_count_change": 1}`
