# 验证结果（实际运行留档）

生成时间: 2026-11-23T09:59:03
Python: 3.11.0rc1  NumPy 验证模块: statistics_view.py

## 1. 单元测试

- 命令: `python run_tests.py`
- 结果: Ran 43 tests in 1.054s，全部 OK（0 failures, 0 errors）

## 2. 定点误差界（scale_exponent=2, 量子 u=0.01）

- bound_interval_real: 0.02
- bound_latest_real: 0.01
- bound_majority_real: 0.02
- note: majority absolute intermediate term grows linearly with the sum of weights; after normalization the decision bound stays at 2u.
- quantum_real: 0.01
- single_observation_quantization_error: 0.005

## 3. 多环境仲裁权威取值

| 视图 | 版本 | 键 | 口径 | 权威值(q) | 权威值(real) | 界(q) | 相邻差(q) | 误差标记 | 归因 |
|---|---|---|---|---|---|---|---|---|---|
| view-all-fleets | v1 | maintenance_window | interval_intersection | None | None | 2 | None | False | policy_conflict,high_rejected_ratio |
| view-all-fleets | v1 | pool_size | latest_wins | 400 | 4.0 | 1 | 0 | True | error_indistinguishable,high_rejected_ratio |
| view-all-fleets | v1 | timeout_ms | weighted_majority | 13233 | 132.33 | 2 | 33 | False | high_rejected_ratio |
| view-production | v1 | maintenance_window | interval_intersection | 700 | 7.0 | 2 | 100 | False | - |
| view-production | v1 | pool_size | weighted_majority | 722 | 7.22 | 2 | 22 | False | high_rejected_ratio |
| view-production | v1 | rate_limit | weighted_majority | 10044 | 100.44 | 2 | 6 | False | - |
| view-production | v1 | timeout_ms | latest_wins | 13000 | 130.0 | 1 | 0 | False | high_rejected_ratio |

## 4. 异常注入（exit code 均为 2，结构化日志见 events.jsonl）

| 注入参数 | 错误码 | 首个违规定位 |
|---|---|---|
| --inject spec_corruption | E_SPEC | environments[env-prod-eu].observations[timeout_ms].seq（重复序号） |
| --inject grid_violation | E_GRID | scale_exponent（42 超出 [-6,6]） |
| --inject version_mismatch | E_VERSION | snapshot.environment_version（环境集合版本不符） |

## 5. 强制终止恢复

- `--crash-view view-production`：写入结论行后未提交即终止，重开账本
  清理孤儿行，该视图进入 unknown；重跑后 `unknown_views=[]`，两视图全部提交。
- 见 demo_results.txt 中 `recovery_detected_unknown` 与 `run_completed` 事件。

## 6. 顺序无关性

- tests/test_determinism.py 在环境、观测、视图、env_ids、policies 全
  反序后断言规范化快照逐字节一致；重复打开账本结论摘要保持稳定。