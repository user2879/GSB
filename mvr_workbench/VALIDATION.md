# 实际验证结果（本地确定性样本）

运行环境：Python 3.11.0rc1，NumPy 2.4.6，Pillow 13.0.0，pytest 9.2.0，无 GPU/网络。

## 一键验收结论（scripts/acceptance.py，退出码 0）

| 验收项 | 结果 |
| --- | --- |
| concurrency_serial_equals_parallel | True |
| incremental_equals_clean_full_rebuild | True |
| no_report_on_forced_kill | True |
| recovery_matches_clean_rebuild | True |
| 强制终止退出码 | 137 (137=SIGKILL 式 os._exit) |
| 注入命中项 | INJ-001, INJ-002, INJ-003, INJ-004 |
| 注入业务码 | E_NON_FINITE_FEATURE, E_PARALLAX_OUT_OF_RANGE, E_VIEW_INVALID |

## 重建规模
- 基线全量：点云代表点 31，消耗预算 140
- 编辑后最终状态：点云代表点 28
- 基线各视图特征数：[40, 40, 40, 40, 40, 40]
- 基线各视图深度格数：[23, 28, 31, 33, 35, 27]
- 基线匹配对（接受/内点/点数）：[('vp-0001-0002', True, 28, 25), ('vp-0002-0003', True, 27, 25), ('vp-0003-0004', True, 30, 27), ('vp-0004-0005', True, 31, 31), ('vp-0005-0006', True, 32, 32)]

## 测试套件
```
37 passed in 16.41s
```

## 诊断报告位置
- 全量/并发对照：`acceptance_runs/full_w1/`、`acceptance_runs/full_w8/`
- 增量序列：`acceptance_runs/incremental/report_*.json`
- 最终状态全量对照：`acceptance_runs/full_final/`
- 注入故障：`acceptance_runs/injection/report.json`
- 强制终止与恢复：`acceptance_runs/kill/`（无 kill 时报告，resume_report.json 与全量一致）
