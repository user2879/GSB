# 实际验证结果

验证环境：WSL Ubuntu 22.04，Python 3.11（满足 ≥3.10），虚拟环境内
numpy 2.4.6、PyYAML 6.0.4，全程本地确定性数据，无外部服务。

## 测试集

`PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v`

共 44 个用例，覆盖：

- 定点 round-half-up、量化半步、枚举校验（`test_fixedpoint.py`）
- 修订链覆盖/删除来源、最长匹配与抑制（`test_library.py`）
- 定点打分顺序、阈值/半步边界、平局决断（`test_scoring_dedup.py`）
- 定点 Jaccard 与阈值相等边界（`test_scoring_dedup.py`）
- 文本唯一/字表/环检测/配置枚举/并发数校验（`test_loader_scheduler.py`、
  `test_validation_errors.py`）
- 全局预算、通道配额、待办优先、稳定顺序（`test_loader_scheduler.py`）
- 四类翻转归因及数值证据（`test_attribution.py`）
- 增量重放与空白全量逐字段一致、并发度无关、失效复用、待办排空
  （`test_integration.py`）
- 快照原子提交、版本冲突、仅靠已提交快照续跑一致（`test_snapshot.py`、
  `test_recovery.py`）
- numpy 定点分布与千分比份额（`test_statistics.py`）

## 端到端验收

- `cli run`：产出 `artifacts/report.json`，含逐文本结论、抑制、路径分值、
  去重归属、待办、业务码计数、失效/复用统计与意图分布。
- `cli replay`：对 10 条编辑命令逐条增量重放，产出 `artifacts/replay.json`，
  翻转含参考分、定点分与差值证据。
- `cli consistency`：增量重放最终态与空白全量重建比较，结果
  `"match": true, "diffs": []`（见 `artifacts/consistency.json`）。
- `scripts/demo.py`：现场演示最长匹配抑制、半步边界、定点去重、预算待办、
  阈值归因与引擎复用。

## 确定性保证

- 样本由固定种子 `20261127` 生成，重复运行 YAML 字节一致。
- 决策仅用整数运算；预算裁决在主线程按稳定顺序执行，线程池只做纯计算，
  故 `max_workers=1` 与 `max_workers=8` 的最终产物逐字段相同（集成测试断言）。
