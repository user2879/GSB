# 多路召回与定点融合误差诊断实验报告

## 可复现元数据

- 全局版本号：`1`
- 配置哈希：`a8c2af126b9dc33e4ac7276f6b43b9890a3a6b5bb927c264034ab9f487fbad9f`
- 固定种子：`20261120`
- 主位宽 / 参考位宽：`5` / `9` 位小数
- 量化步长：`0.03125` / `0.001953125`
- 舍入方式：`half_even`；容差：`0.01`
- 结果摘要哈希：`3288c9d3f797172789f58cee27246d1828c22e824da97fc9e99839f770bf5d0d`

## 执行与快照

- 执行模式：`thread_pool`，状态：`completed`，退出码：`0`
- 配置最大并发：`4`；并发上界：`4`
- 用户数：`48`

| 阶段 | 产物版本 | 已完成用户数 | 是否完成 | 游标 |
|---|---:|---:|---|---|
| recall | 10 | 48 | True | u0047 |
| fusion | 10 | 48 | True | u0047 |
| evaluation | 10 | 48 | True | u0047 |

## 输入校验与失败清单

- 非法记录数：`11`；业务码分布：`{"E_RECORD_INVALID": 11}`
- 业务码目录：`E_RECORD_INVALID, E_CONFIG_INVALID, E_CONFIG_MISMATCH, E_SNAPSHOT_INVALID, E_INPUT_NOT_FOUND`

| 业务码 | 来源 | 行 | 字段 | 原因 |
|---|---|---:|---|---|
| E_RECORD_INVALID | interactions.jsonl | 508 | user_id | must be a non-empty string |
| E_RECORD_INVALID | interactions.jsonl | 509 | item_id | item id not present in catalog: i999 |
| E_RECORD_INVALID | interactions.jsonl | 510 | event_ms | must be a parseable non-negative millisecond timestamp |
| E_RECORD_INVALID | interactions.jsonl | 511 | event_type | event type not configured: share |
| E_RECORD_INVALID | interactions.jsonl | 512 | item_id | must be a non-empty string |
| E_RECORD_INVALID | interactions.jsonl | 513 | $ | record must be a JSON object |
| E_RECORD_INVALID | items.jsonl | 73 | item_id | must be a non-empty string |
| E_RECORD_INVALID | items.jsonl | 74 | terms | must be a non-empty list |
| E_RECORD_INVALID | items.jsonl | 75 | terms[1] | must be a non-empty string |
| E_RECORD_INVALID | items.jsonl | 76 | item_id | duplicate item id: i000 |
| E_RECORD_INVALID | items.jsonl | 77 | $ | invalid JSON: Expecting property name enclosed in double quotes |

## 三路召回

- 用户数：`48`；冷启动用户：`3`；零候选权重重分配用户：`3`
- 通道候选均值：`lexical`=13.12；`collaborative`=13.12；`popularity`=14.00

## 逐级误差传播

- 排序位置变化用户数：`47`；位置变化数：`518`
- 排序变化归因：`{"feature_rounding_accumulation": 376, "normalization_residual": 142}`

| 环节 | 最大差值 | 超过容差候选数 | 首次超差计数 |
|---|---:|---:|---:|
| normalization | 0.015625 | 520 | 520 |
| channel_weighted_sum | 0.050781 | 546 | 311 |
| feature_add | 0.070312 | 820 | 308 |
| clipping | 0.056641 | 744 | 0 |

### 位宽切换导致的排序变化

| 用户 | 物品 | 主位宽名次 | 参考位宽名次 | 归因 |
|---|---|---:|---:|---|
| u0000 | i065 | 8 | 10 | feature_rounding_accumulation |
| u0000 | i050 | 9 | 8 | feature_rounding_accumulation |
| u0000 | i051 | 10 | 9 | feature_rounding_accumulation |
| u0000 | i026 | 11 | 12 | feature_rounding_accumulation |
| u0000 | i019 | 12 | 11 | feature_rounding_accumulation |
| u0000 | i003 | 13 | 16 | feature_rounding_accumulation |
| u0000 | i020 | 14 | 13 | feature_rounding_accumulation |
| u0000 | i000 | 15 | 14 | normalization_residual |
| u0000 | i013 | 16 | 17 | normalization_residual |
| u0000 | i043 | 17 | 15 | normalization_residual |
| u0000 | i007 | 20 | 22 | normalization_residual |
| u0000 | i014 | 21 | 20 | normalization_residual |
| u0000 | i053 | 22 | 23 | normalization_residual |
| u0000 | i040 | 23 | 24 | normalization_residual |
| u0000 | i058 | 24 | 25 | normalization_residual |
| u0000 | i004 | 25 | 21 | normalization_residual |
| u0001 | i002 | 7 | 8 | feature_rounding_accumulation |
| u0001 | i054 | 8 | 7 | feature_rounding_accumulation |
| u0001 | i038 | 19 | 20 | feature_rounding_accumulation |
| u0001 | i005 | 20 | 19 | feature_rounding_accumulation |
| u0001 | i028 | 26 | 27 | feature_rounding_accumulation |
| u0001 | i067 | 27 | 26 | feature_rounding_accumulation |
| u0002 | i060 | 3 | 4 | normalization_residual |
| u0002 | i000 | 4 | 3 | feature_rounding_accumulation |
| u0002 | i069 | 5 | 6 | feature_rounding_accumulation |
| u0002 | i052 | 6 | 5 | normalization_residual |
| u0002 | i063 | 11 | 12 | feature_rounding_accumulation |
| u0002 | i004 | 12 | 13 | feature_rounding_accumulation |
| u0002 | i032 | 13 | 11 | feature_rounding_accumulation |
| u0002 | i055 | 19 | 20 | feature_rounding_accumulation |
| u0002 | i042 | 20 | 19 | feature_rounding_accumulation |
| u0002 | i064 | 22 | 23 | feature_rounding_accumulation |
| u0002 | i051 | 23 | 22 | feature_rounding_accumulation |
| u0002 | i054 | 24 | 25 | feature_rounding_accumulation |
| u0002 | i011 | 25 | 24 | feature_rounding_accumulation |
| u0002 | i002 | 27 | 28 | feature_rounding_accumulation |
| u0002 | i033 | 28 | 27 | feature_rounding_accumulation |
| u0002 | i028 | 29 | 30 | feature_rounding_accumulation |
| u0002 | i067 | 30 | 29 | feature_rounding_accumulation |
| u0003 | i071 | 3 | 4 | feature_rounding_accumulation |

### 边界敏感候选

| 用户 | 物品 | 位宽角色 | 位宽 | 名次 | 相邻分差 | 原因 |
|---|---|---|---:|---:|---:|---|
| u0000 | i070 | primary | 5 | 1 | 0.000000 | adjacent_margin_within_tolerance |
| u0000 | i049 | primary | 5 | 3 | 0.000000 | adjacent_margin_within_tolerance |
| u0000 | i059 | primary | 5 | 7 | 0.000000 | adjacent_margin_within_tolerance |
| u0000 | i065 | primary | 5 | 8 | 0.031250 | rank_changed_between_widths |
| u0000 | i050 | primary | 5 | 9 | 0.000000 | rank_changed_between_widths, adjacent_margin_within_tolerance |
| u0000 | i051 | primary | 5 | 10 | 0.000000 | rank_changed_between_widths, adjacent_margin_within_tolerance |
| u0000 | i026 | primary | 5 | 11 | 0.031250 | rank_changed_between_widths |
| u0000 | i019 | primary | 5 | 12 | 0.031250 | rank_changed_between_widths |
| u0000 | i003 | primary | 5 | 13 | 0.000000 | rank_changed_between_widths, adjacent_margin_within_tolerance |
| u0000 | i020 | primary | 5 | 14 | 0.000000 | rank_changed_between_widths, adjacent_margin_within_tolerance |
| u0000 | i000 | primary | 5 | 15 | 0.000000 | rank_changed_between_widths, adjacent_margin_within_tolerance |
| u0000 | i013 | primary | 5 | 16 | 0.000000 | rank_changed_between_widths, adjacent_margin_within_tolerance |
| u0000 | i043 | primary | 5 | 17 | 0.031250 | rank_changed_between_widths |
| u0000 | i001 | primary | 5 | 18 | 0.000000 | adjacent_margin_within_tolerance |
| u0000 | i007 | primary | 5 | 20 | 0.000000 | rank_changed_between_widths, adjacent_margin_within_tolerance |
| u0000 | i014 | primary | 5 | 21 | 0.000000 | rank_changed_between_widths, adjacent_margin_within_tolerance |
| u0000 | i053 | primary | 5 | 22 | 0.000000 | rank_changed_between_widths, adjacent_margin_within_tolerance |
| u0000 | i040 | primary | 5 | 23 | 0.000000 | rank_changed_between_widths, adjacent_margin_within_tolerance |
| u0000 | i058 | primary | 5 | 24 | 0.000000 | rank_changed_between_widths, adjacent_margin_within_tolerance |
| u0000 | i004 | primary | 5 | 25 | 0.187500 | rank_changed_between_widths |
| u0000 | i016 | primary | 5 | 26 | 0.000000 | adjacent_margin_within_tolerance |
| u0000 | i033 | primary | 5 | 27 | 0.000000 | adjacent_margin_within_tolerance |
| u0000 | i070 | reference | 9 | 1 | 0.000000 | adjacent_margin_within_tolerance |
| u0000 | i050 | reference | 9 | 8 | 0.005859 | rank_changed_between_widths, adjacent_margin_within_tolerance |
| u0000 | i051 | reference | 9 | 9 | 0.003906 | rank_changed_between_widths, adjacent_margin_within_tolerance |
| u0000 | i065 | reference | 9 | 10 | 0.015625 | rank_changed_between_widths |
| u0000 | i019 | reference | 9 | 11 | 0.011719 | rank_changed_between_widths |
| u0000 | i026 | reference | 9 | 12 | 0.019531 | rank_changed_between_widths |
| u0000 | i020 | reference | 9 | 13 | 0.017578 | rank_changed_between_widths |
| u0000 | i000 | reference | 9 | 14 | 0.003906 | rank_changed_between_widths, adjacent_margin_within_tolerance |
| u0000 | i043 | reference | 9 | 15 | 0.003906 | rank_changed_between_widths, adjacent_margin_within_tolerance |
| u0000 | i003 | reference | 9 | 16 | 0.003906 | rank_changed_between_widths, adjacent_margin_within_tolerance |
| u0000 | i013 | reference | 9 | 17 | 0.015625 | rank_changed_between_widths |
| u0000 | i001 | reference | 9 | 18 | 0.000000 | adjacent_margin_within_tolerance |
| u0000 | i014 | reference | 9 | 20 | 0.003906 | rank_changed_between_widths, adjacent_margin_within_tolerance |
| u0000 | i004 | reference | 9 | 21 | 0.005859 | rank_changed_between_widths, adjacent_margin_within_tolerance |
| u0000 | i007 | reference | 9 | 22 | 0.007812 | rank_changed_between_widths, adjacent_margin_within_tolerance |
| u0000 | i053 | reference | 9 | 23 | 0.007812 | rank_changed_between_widths, adjacent_margin_within_tolerance |
| u0000 | i040 | reference | 9 | 24 | 0.000000 | rank_changed_between_widths, adjacent_margin_within_tolerance |
| u0000 | i058 | reference | 9 | 25 | 0.173828 | rank_changed_between_widths |
| u0001 | i063 | primary | 5 | 1 | 0.000000 | adjacent_margin_within_tolerance |
| u0001 | i014 | primary | 5 | 2 | 0.000000 | adjacent_margin_within_tolerance |
| u0001 | i068 | primary | 5 | 3 | 0.000000 | adjacent_margin_within_tolerance |
| u0001 | i002 | primary | 5 | 7 | 0.000000 | rank_changed_between_widths, adjacent_margin_within_tolerance |
| u0001 | i054 | primary | 5 | 8 | 0.093750 | rank_changed_between_widths |
| u0001 | i000 | primary | 5 | 10 | 0.000000 | adjacent_margin_within_tolerance |
| u0001 | i055 | primary | 5 | 11 | 0.000000 | adjacent_margin_within_tolerance |
| u0001 | i007 | primary | 5 | 14 | 0.000000 | adjacent_margin_within_tolerance |
| u0001 | i042 | primary | 5 | 15 | 0.000000 | adjacent_margin_within_tolerance |
| u0001 | i059 | primary | 5 | 16 | 0.000000 | adjacent_margin_within_tolerance |

## 真值评估

| 位宽 | 用户数 | 命中率 | Precision@K | 平均位置偏差 | 平均 Top-K 重合 | 未命中真值候选数 |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 48 | 0.981771 | 0.981771 | 0.516857 | 7.854167 | 7 |
| 9 | 48 | 0.997396 | 0.997396 | 0.045078 | 7.979167 | 1 |

两档位评估是否真实不同：`True`（评估使用各自位宽的实际排序结果）。

## 产物

- 清单：`/home/zsl/feedback/uidfc5909d3-default-308/runs/verification/manifest.json`
- 召回快照：`/home/zsl/feedback/uidfc5909d3-default-308/runs/verification/stages/recall.json`
- 融合快照：`/home/zsl/feedback/uidfc5909d3-default-308/runs/verification/stages/fusion.json`
- 评估快照：`/home/zsl/feedback/uidfc5909d3-default-308/runs/verification/stages/evaluation.json`
- 完整报告 JSON 内含每用户逐候选误差、通道贡献和未命中归因。
