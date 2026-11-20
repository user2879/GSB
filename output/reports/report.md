# RoPE 外推诊断报告

- 全局版本: 1
- 配置哈希: `7f66b692d3233626c8cf0a3f`
- 注入清单哈希: `652817c45ee00de4aae7fafe`
- 剪枝口径指纹: `57541f35a0705233db6a6d41`

## 一致性校验
- serial_vs_concurrent: PASS
- reproducible_rerun: PASS
- incremental_vs_full: PASS
- incremental_reused_artifacts: PASS
- pending_marked: PASS
- injection_codes_present: PASS

## 误差追踪汇总（首次超差环节计数）
- ntk: 外推键 20 个, 首次超差分布 {'position_vector': 8, 'score': 12, 'weight': 0, 'none': 0}
- standard: 外推键 8 个, 首次超差分布 {'position_vector': 0, 'score': 8, 'weight': 0, 'none': 0}
- linear: 外推键 20 个, 首次超差分布 {'position_vector': 8, 'score': 12, 'weight': 0, 'none': 0}

## 分层打分（距离档位数）
- ntk: in_train 距离档 56 个, extrapolation 距离档 20 个
- standard: in_train 距离档 29 个, extrapolation 距离档 8 个
- linear: in_train 距离档 56 个, extrapolation 距离档 20 个

## 敏感性不可用档位
- standard: base x1000000000.0 -> E_DIM_TRUNCATION
- linear: base x1000000000.0 -> E_DIM_TRUNCATION
- ntk: base x1000000000.0 -> E_DIM_TRUNCATION

## 预算与失败
- 点数预算 430, 已消耗 415, 待算 2 项 (E_POINT_BUDGET_PENDING)
- 失败 linear/S001/load_validate: E_NONFINITE_FEATURE (注入 INJ-001)
- 失败 linear/S002/load_validate: E_KEY_POS_OOB (注入 INJ-002)
- 失败 linear/S003/position_encoding: E_BASE_ZERO (注入 INJ-003)
- 失败 linear/S004/position_encoding: E_SCALE_NONPOSITIVE (注入 INJ-004)
- 失败 ntk/S001/load_validate: E_NONFINITE_FEATURE (注入 INJ-001)
- 失败 ntk/S002/load_validate: E_KEY_POS_OOB (注入 INJ-002)
- 失败 ntk/S003/position_encoding: E_BASE_ZERO (注入 INJ-003)
- 失败 ntk/S004/position_encoding: E_SCALE_NONPOSITIVE (注入 INJ-004)
- 失败 standard/S001/load_validate: E_NONFINITE_FEATURE (注入 INJ-001)
- 失败 standard/S002/load_validate: E_KEY_POS_OOB (注入 INJ-002)
- 失败 standard/S003/position_encoding: E_BASE_ZERO (注入 INJ-003)
- 失败 standard/S004/position_encoding: E_SCALE_NONPOSITIVE (注入 INJ-004)

## 剪枝决策
- 选中策略: P_PRUNE_32
- 依据: policy P_PRUNE_32 scores highest (101.250000) for dim_block_size=32 and prune_tolerance=0.001; its dim_range and score_chunk are applied to the scoring path
