# 实际验证结果

环境：Python 3.11.0rc1、NumPy 1.26.4、Pillow 11.0.0（满足 >=3.10 要求）。
样本：5 视图、240x180 PNG，固定种子 20261121，本地秒级完成。

## 1. 单元/集成测试

命令：`python -m unittest discover -s tests -t .`

结果：**Ran 33 tests in 42.7s — OK**（7 个测试模块）。

| 模块 | 用例数 | 覆盖验收点 |
| --- | --- | --- |
| test_loader | 6 | PNG 截断、相机缺失、奇异内参、非法视场角、分辨率不一致、异常隔离 |
| test_determinism | 4 | 特征/匹配多次运行逐项一致、响应+坐标排序、截断上限 |
| test_concurrency | 3 | 串行=并发逐字段一致、并发数封顶、预算挂起与稳定扣减序列 |
| test_incremental | 4 | 全量后全复用、图像编辑部分失效、参数编辑全失效、相机编辑保特征 |
| test_snapshot | 4 | 四阶段提交、续跑判定、`E_SNAPSHOT_MISMATCH` 废弃重建、清单字段 |
| test_precision | 4 | 五阶段误差结构、首超容差唯一标注、体素分歧归因、报告完整性 |
| test_injection | 6 | 五类注入命中业务码、视图隔离、注入编号可追溯 |
| test_shutdown | 2 | 信号优雅退出码 0、仅完整快照可恢复 |

## 2. 逐级误差归因（实测）

float32 主路径相对 float64 参考路径，单视图 120 特征、共 173 匹配、
172 三角点、8 深度图、88 点云代表点：

| 环节 | 最大绝对误差 | 最大相对误差 | 首次超容差 |
| --- | --- | --- | --- |
| feature | 0.000e+00 | 0.000e+00 | 否 |
| match | 4.620e-08 | 1.456e-07 | 否 |
| triangulation | 3.569e-01 | 8.984e-05 | **是** |
| depth | 4.739e-01 | 8.973e-05 | 否 |
| point_cloud | 4.883e-04 | 3.902e-05 | 否 |

误差传播链条清晰：角点整数像素定位零差异 → 描述子浮点微小差异 →
DLT 三角化首次放大并越过 1e-4 容差 → 深度继承 → 体素平均收敛。
体素归属分歧样本被记录并区分为 `triangulation` / `quantization_rounding`。

## 3. 增量与全量一致性

- 首次运行：18 个缓存键全部 `recomputed`，0 复用。
- 同输入二次运行：0 重算，18 键全部 `reused`，快照从 v0001 续跑。
- 替换视图 2 图像后：7 键复用（未触及视图的特征/匹配/深度），
  11 键重算（视图 2 特征、相邻匹配/深度、全局点云）。
- 串行（thread=1）与并发（thread=4）的特征序、匹配序、深度摘要、
  点云点序、预算扣减序列、失败清单经测试逐字段断言相等。

## 4. 预算约束（rebuild_point_budget=40 实测）

扣减序列严格按视图序号 `[1,2,3,4]`，授予额度 `[40,0,0,0]`，
超预算项标记 `E_REBUILD_BUDGET_PENDING`，与线程调度时机无关。

## 5. 快照恢复

- 四阶段（feature/match/depth/point_cloud）均先写临时目录再原子 rename 整体替换。
- 篡改配置哈希后 `verify_resume` 抛 `E_SNAPSHOT_MISMATCH`，
  运行器废弃该版本并从最近有效快照整体重建（非逐文件覆盖）。
- 清单含 version、config_hash、input_hash、precision、completed_views。

## 6. 异常注入隔离（五项全部命中）

| 注入 | 视图/阶段 | 业务码 | 命中 |
| --- | --- | --- | --- |
| 截断 PNG 字节 | v1/load | E_PNG_TRUNCATED | 是 |
| 相机参数缺失 | v3/load | E_CAMERA_MISSING | 是 |
| 描述含非有限数 | v0/feature | E_FEATURE_NONFINITE | 是 |
| 视差越界 | v4/triangulation | E_MIN_DISPARITY | 是 |
| 体素尺寸为零 | point_cloud | E_VOXEL_ZERO | 是 |

单视图异常仅进入失败清单，其余视图产物正常；注入编号 INJ01..INJ05
写入诊断报告并映射到实际观测业务码。

## 7. 可处理退出语义

- 独立进程运行中发送 SIGTERM：完成当前阶段原子替换后以**退出码 0** 结束，
  特征阶段快照完整可读。
- 强制终止（SIGKILL 无法捕获）场景下，仅已完整 rename 的快照可被
  `verify_resume` 识别恢复，临时目录残留不参与恢复；不承诺生成退出报告。

## 8. 复现实验命令

```bash
python -m recon_workbench.cli generate ./sample --config ./sample/rebuild_config.json
python -m recon_workbench.cli run --sample-dir ./sample \
  --config ./sample/rebuild_config.json --work-dir ./work \
  --report ./report.json --viz-dir ./viz
python -m recon_workbench.cli run --sample-dir ./sample \
  --config ./sample/rebuild_config.json --work-dir ./work \
  --injection configs/injection_config.json --injected-sample ./injected \
  --report ./injection_report.json
python demo.py
```
