# 验证结果记录

环境：Python 3.11（满足 ≥3.10）、NumPy 1.26.4（满足 ≥1.24）、PyYAML 6。
全程本地运行，不使用容器、不访问网络。

## 1. 自动化单元/集成测试

命令：`python -m unittest discover -s tests -p 'test_*.py'`

结果：**47 个测试全部通过（OK），耗时约 6 秒**。覆盖映射：

- `test_fixedpoint.py`：整数开方、档位改变整数网格、舍入方向、零分母、定点乘除、同票决断、确定性。
- `test_config_loader.py`：权重非有限/全零、定点枚举、预算、缺失/截断/分辨率/邻接/重复 ID/窗口不整除。
- `test_classification.py`：中心区域完整且不重叠、逐像素确定性、残差与舍入记录、档位改变得分、概率和为 1。
- `test_fusion_postprocess.py`：12 个融合对、FV1/FV2 栅格不同、平票确定性、连通域剔除回填。
- `test_cache_snapshot.py`：全量缓存命中、波段补丁精确失效、权重变化全失效、增量=干净重建、快照陈旧码、清单字段。
- `test_scheduler.py`：稳定排序键、三类预算码、融合待办、并发=串行。
- `test_diagnostics_injection.py`：半步边界诊断、6 类注入全部命中业务码且单瓦片故障隔离。

## 2. 可重复演示

命令：`python demo/run_demo.py`

关键结论（写入 `demo_out/demo_summary.json`）：

- 冷启动重算 9 瓦片，热启动 9 瓦片全部命中缓存，冷/热栅格逐像素一致。
- 单瓦片波段补丁仅失效该瓦片及其 4 条融合带；增量结果与干净全量重建逐像素一致、得分摘要一致。
- 权重变化使 9 瓦片全部重算且栅格改变；融合策略 FV1→FV2 改变融合后栅格。
- 版本号升级触发 `E_SNAPSHOT_STALE`；串行与并发栅格一致。
- 预算：分别命中 `E_PIXEL_BUDGET_EXHAUSTED`、`E_TILE_BUDGET_EXHAUSTED`、
  `E_FUSION_BUDGET_PENDING`；融合积压按批次 `[10,8,6,4,2,0]` 确定性收敛到 0。
- 注入 6/6 命中预期业务码。

## 3. 验收产物

- `reports/diagnostic_report.json`：全局类别栅格、逐瓦片得分/概率摘要、12 条融合带的逐像素
  投票明细、436 条半步边界诊断（含 20 条 E_BOUNDARY_FLIP 及归因）、失效/复用清单、失败清单。
- `reports/injection_report.json`：6 个注入项编号、期望/观测业务码、命中证明与其余瓦片仍完成的证据。

## 4. 业务码对照

| 业务码 | 触发条件 |
| --- | --- |
| E_BAND_COUNT_MISMATCH / E_BAND_MISSING / E_BAND_TRUNCATED | 波段数量不符 / 缺失 / 截断损坏 |
| E_RESOLUTION_MISMATCH / E_TILE_ID_DUPLICATE | 分辨率不一致 / 瓦片 ID 重复 |
| E_ADJ_DANGLING / E_ADJ_NOT_SYMMETRIC | 邻接悬空 / 非对称 |
| E_BANDSET_UNKNOWN / E_WINDOW_NOT_DIVISIBLE / E_WINDOW_LARGER_TILE | 波段集合版本 / 窗口整除 / 窗口越界 |
| E_WEIGHT_INVALID / E_FIXED_ENUM / E_BUDGET_INVALID | 权重 / 定点枚举 / 预算非法 |
| E_TILE_BUDGET_EXHAUSTED / E_PIXEL_BUDGET_EXHAUSTED / E_FUSION_BUDGET_PENDING | 三重预算约束 |
| E_SNAPSHOT_STALE / E_BOUNDARY_FLIP / E_TILE_COMPUTE | 快照陈旧 / 边界翻转 / 单瓦片计算故障 |
