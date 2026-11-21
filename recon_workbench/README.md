# 多视图三维几何重建可视化与诊断工作台

面向本地确定性样本的多视图三维重建工作台，核心解决重建链路中**数值精度与
误差传播的可观测性**。仅依赖 NumPy 与 Pillow，无网络、无容器、无外部数据集。

## 1. 设计概览

```
样本目录(PNG+相机JSON+配置JSON)
        |
  loader 加载/校验（PNG头、分辨率、视图序号、内参可逆、视场角）
        v
  features  Harris 角点（响应降序+坐标字典序）+ 归一化灰度描述子
        v
  features  对称最近邻匹配（距离门限+最小内点数，边界固定决断）
        v
  geometry  DLT 三角化（最小视差门限）-> 量化深度图 -> 确定性平滑
        v
  geometry  体素级点云合并（同体素稳定排序取代表）
        v
precision_analysis  float32 主路径 vs float64 参考路径逐级误差归因
```

所有随机来源被固定种子取代；角点检测/排序在 float64 参考响应上统一进行，
保证两精度路径角点集合逐字段一致，精度差异由描述子、三角化、深度与点云承担。

## 2. 目录结构

```
recon_workbench/
  config.py              重建配置模型与校验、参数指纹
  errors.py              业务错误码（E_*）与失败记录
  utils.py               规范化序列化、sha256 指纹、原子写、定点/体素量化
  generate_sample.py     固定种子多视图样本生成（透视棋盘格+几何图元）
  loader.py              视图/相机加载与五项校验、异常隔离
  features.py            Harris 角点、描述子、确定性匹配
  geometry.py            三角化、深度图融合、体素点云合并
  precision_analysis.py  四环节误差、首超容差标注、体素分歧归因
  artifact_store.py      内容寻址缓存键 + 阶段快照整体替换
  pipeline.py            纯计算流水线（float32/float64 共用）
  engine.py              缓存调度、线程池、预算约束、快照提交
  runner.py              版本/恢复、双精度调度、信号处理、报告
  injection.py           五类真实故障注入与命中映射
  editing.py             图像/相机/配置编辑命令序列
  visualization.py       特征/匹配/深度/点云投影 PNG
  cli.py                 generate / run / edit 命令行
tests/                   33 个 unittest 用例（7 个模块）
configs/                 重建配置、注入配置、编辑命令序列
demo.py                  端到端验收演示
```

## 3. 快速开始

```bash
python -m pip install -r requirements.txt

# 生成确定性样本
python -m recon_workbench.cli generate ./sample --config ./sample/rebuild_config.json

# 全量重建（float32 + float64），输出诊断报告与可视化
python -m recon_workbench.cli run \
  --sample-dir ./sample --config ./sample/rebuild_config.json \
  --work-dir ./work --report ./report.json --viz-dir ./viz

# 端到端演示（增量、注入、快照恢复）
python demo.py

# 测试
python -m unittest discover -s tests -t .
```

## 4. 八项业务要求的实现映射

| 要求 | 实现位置 | 关键机制 |
| --- | --- | --- |
| 1 加载校验与隔离 | `loader.py` | PNG 魔数/解码、分辨率、序号唯一、det 内参、视场角区间；单视图失败进失败清单，其余继续 |
| 2 确定性特征/匹配 | `features.py` | 响应量化后降序、同响应 `(y,x)` 字典序；对称最近邻 + 距离门限，边界相等保留并标记 |
| 3 三角化/深度/点云 | `geometry.py` | DLT+SVD、视差 `>=` 门限（相等标记边界）、深度档位量化、固定迭代平滑、体素 floor + 稳定排序代表 |
| 4 双精度误差归因 | `precision_analysis.py` | 五阶段（特征/匹配/三角化/深度/点云）最大绝对/相对误差、首超容差环节、体素分歧区分三角化 vs 量化舍入 |
| 5 指纹依赖与增量 | `artifact_store.py` `engine.py` | 图像指纹（特征）→ 匹配键 → 深度键（含相机指纹）→ 点云键逐层派生；复用/重算清单可审计 |
| 6 快照整体替换 | `artifact_store.py` `runner.py` | 临时目录 + 原子 rename；清单含版本/配置哈希/输入哈希/精度/完成视图；不一致 `E_SNAPSHOT_MISMATCH` 废弃重建 |
| 7 并发与预算一致 | `engine.py` | ThreadPoolExecutor 上限取配置；预算在主线程按稳定键串行预扣；超预算 `E_REBUILD_BUDGET_PENDING`；并发结果与串行逐字段一致 |
| 8 注入/退出/报告 | `injection.py` `runner.py` | 截断 PNG、相机缺失、视差越界、非有限描述、零体素；注入编号映射业务码；SIGTERM 原子替换后退出码 0；JSON 诊断报告 |

## 5. 误差归因口径

- 绝对误差：`max(|x32 - x64|)`，逐元素比较同 ID 产物。
- 相对误差：仅在参考值 `|x64| > 1e-6` 的分量上计算 `|x32-x64|/|x64|`，
  避免零坐标被兜底分母放大。
- 首次超容差：按特征→匹配→三角化→深度→点云顺序，首个绝对或相对误差
  超过配置容差的环节被标记 `first_breach=true`。
- 体素分歧归因：float32 漂移量大于参考点到体素边界距离 => `triangulation`；
  否则参考点本就贴边 => `quantization_rounding`。

## 6. 缓存键依赖链

```
feat  = H(图像指纹, 配置指纹, 精度)
match = H(feat_a, feat_b, 配置指纹, 精度)
depth = H(match, 相机指纹, 配置指纹, 精度)
cloud = H(全部 depth 键, 配置指纹, 精度)
```

- 替换图像：仅该视图特征 + 相邻匹配/深度 + 点云失效。
- 改相机参数：特征复用（角点只依赖图像），深度与点云失效。
- 改重建参数：配置指纹变化，全链路失效重算。

## 7. 业务错误码

`E_VIEW_INVALID` `E_PNG_TRUNCATED` `E_CAMERA_MISSING`
`E_RESOLUTION_MISMATCH` `E_VIEW_DUPLICATE` `E_INTRINSIC_SINGULAR`
`E_FOV_ILLEGAL` `E_FEATURE_NONFINITE` `E_MIN_INLIERS` `E_MIN_DISPARITY`
`E_VOXEL_ZERO` `E_CONFIG_INVALID` `E_SNAPSHOT_MISMATCH`
`E_REBUILD_BUDGET_PENDING` `E_INJECTION`

## 8. 验证结果

见 `VALIDATION.md`（记录实际运行的测试与端到端验收输出）。
