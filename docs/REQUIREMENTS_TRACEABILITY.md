# 需求对照与验证结果

| 业务要求 | 实现位置 | 验证测试 / 演示 |
| --- | --- | --- |
| 1. JSON 强校验，E_SPEC 定位首个违规源与字段 | `spec.py`, `model.py` | `tests/test_spec.py`（10 例） |
| 2. 确定性生成、逐字节摘要、稳定排序 | `prng.py`, `stream.py`, `canonical.py` | `tests/test_stream.py`、`run_demo.py` 摘要断言 |
| 3. 租约签发/乐观续约/单调时钟过期回收 | `lease.py`, `clock.py` | `tests/test_lease.py`（6 例） |
| 4. 全局+每服务双重并发约束的真实线程池调度 | `scheduler.py` | `tests/test_scheduler.py`（峰值不超限且真实并行） |
| 5. 精确/近似双路径与显式阈值自适应、边界值 | `correlation.py` | `tests/test_correlation.py`、三份样本（exact/approximate 均覆盖） |
| 6. E_DIVERGE 一致性、漏/误关联上界、顺序无关 | `correlation.py` | `tests/test_correlation.py`（含顺序无关与误报触发） |
| 7. 关闭告警抑制与根因排序、顺序/批次无关 | `suppression.py`, `rootcause.py` | `tests/test_suppression_rootcause.py` |
| 8. 定点统计、误判归因、账本版本/摘要、整体替换、恢复重放 | `fixedpoint.py`, `ledger.py`, `engine.py` | `tests/test_fixedpoint.py`、`tests/test_ledger.py`、`tests/test_engine.py`、`crash_recovery_demo.py` |

## 实测结论（本机 Python 3.11 + numpy 1.26）
- 中样本 200 告警：跨越双阈值，选择近似路径；精确 1752 对、近似 10 对、
  实测漏关联 1742、误关联 0；全局并发峰值 4、单服务峰值 2，均未超配。
- 大样本 260 告警：近似路径；误差界与阈值共同写入账本。
- 小样本 12 告警：未跨越阈值，默认精确路径，并输出阈值边界余量。
- 定点：scale=10000，向零截断，单值误差界 0.0001、比例比较误差界 0.0002，
  溢出按 ±1,000,000 饱和。
- 账本：摘要/版本损坏触发整体替换；崩溃后 unknown 轮次只重跑不生成退出报告；
  所有确定性字段重放比对零差异。

测试命令：`PYTHONPATH=src python -m unittest discover -s tests`（45 passed）。
