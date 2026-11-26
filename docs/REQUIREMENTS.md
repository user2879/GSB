# 需求追溯矩阵

| 业务要求 | 实现模块 | 关键测试 |
| --- | --- | --- |
| 1 加载校验 E_AUDIO_FORMAT/E_INPUT_INVALID | loading.py, pcmio.py | test_loading.py |
| 2 窗切分/跨边界引用/确定性累加 | features.py | test_features.py |
| 3 字级声学+语义证据/残差/Tier 敏感 | evidence.py, fixedpoint.py | test_evidence_calibration.py |
| 4 逐轮定点校准/整数衰减/空轮继承 | calibration.py | test_evidence_calibration.py |
| 5 区间汇总/阈值边界/重叠裁决 | aggregation.py | test_aggregation.py |
| 6 指纹缓存键/精确失效/账本审计 | fingerprint.py, ledger.py, pipeline.py | test_incremental.py |
| 7 有界线程池/预算配额/待办优先 | concurrency.py, scheduler.py | test_scheduler.py |
| 8 误差追踪/原子快照/退出与恢复 | error_track.py, snapshot.py, pipeline.py | test_error_tracking.py, test_recovery.py |
