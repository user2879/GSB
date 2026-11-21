# 三维关键点确定性求解组件

面向姿态估计后端的本地多视图关键点求解示例。输入是**已完成跨视图关联**的关键点观测，而不是待匹配视图对；组件对每个关键点执行加载校验、多视图三角化、最小视差筛选、粗差剔除、三维点定稿、定点量化误差追踪，并在残差预算与线程池约束下并行推进。

实现只依赖 Python 3.11+、NumPy 与 Pillow，不使用网络、Docker 或虚拟化。所有缓存键、报告列表和快照版本均为内容派生，线程完成顺序不影响结果。

## 快速开始

```powershell
py -3.11 -m pip install -r requirements.txt
py -3.11 -m pose3d generate-sample --out sample
py -3.11 -m pose3d solve --input sample --config sample\solver_config.json --snapshot-dir state\snapshots --output reports\base_fresh.json
```

带独立故障注入运行：

```powershell
py -3.11 -m pose3d solve --input sample --config sample\solver_config.json --injections sample\injections.json --snapshot-dir state\injection-snapshots --output reports\injection_run.json
```

重复运行会读取 `state\snapshots\current\manifest.json` 与阶段缓存。输入版本未变时，所有关键点复用，快照不重写；输入变化时按阶段缓存键精确失效。`--fresh` 强制忽略缓存并全量重建。

完整验收与单元测试：

```powershell
py -3.11 -m unittest discover -v
py -3.11 scripts\verify.py
```

验收结果写入 `reports/verification.json` 与 `reports/verification.md`。

## 交付内容

```text
pose3d/
  loader.py        PNG/相机/JSONL 加载与逐观测校验
  geometry.py      DLT 三角化、重投影、视差筛选
  solver.py        单关键点四阶段求解、剔除轨迹、定点重算
  budget.py        单关键点配额与全局预算账本
  cache.py         内容寻址的阶段缓存
  snapshot.py      临时目录构建、整目录版本切换
  injections.py    独立故障注入配置
  pipeline.py      确定性线程池、预算仲裁、报告组装
  samplegen.py     固定种子样本生成器
  cli.py           generate-sample / solve 命令
tests/             19 项单元与集成测试
scripts/verify.py  验收矩阵与报告生成
sample/            6 视图、16 关键点的固定种子样本
reports/           实际运行 JSON 报告与验收报告
```

## 样本数据

`sample/` 由 `SEED=20261122` 生成：

- 6 张 `800x600` 确定性灰度 PNG，画面含几何体、线段与角点标记。
- 每视图一个相机 JSON，含 `view_index / width / height / K / R / t`。
- `observations.jsonl` 共 16 个关键点，含同名双实例：
  - `rig/corner_00@instance-A` 与 `rig/corner_00@instance-B` 等必须独立求定。
- 正常点 6 视图观测；`rig/corner_02@instance-A` 含 1 个粗差观测。
- `low/parallax@instance-A` 只使用近共视的视图 1/2/3，触发 `E_DISPARITY_INSUFFICIENT`。
- `fragile/tip@instance-A` 只有 3 个观测，剔除粗差后会低于 `min_inliers`，触发 `E_INLIER_SHORTAGE`。
- `bounds/tip@instance-A` 含 1 个越界观测，该观测以 `E_OBS_INVALID` 拒绝，其余观测继续定稿。
- `input_version.json` 记录观测清单、配置、注入配置版本与内容哈希。
- `injections.json` 提供四类真实输入故障，均只作用于指定关键点。

## 输入格式

### 相机

```json
{
  "view_index": 1,
  "image": "views/view_01.png",
  "width": 800,
  "height": 600,
  "K": [[760.0, 0.0, 400.0], [0.0, 760.0, 300.0], [0.0, 0.0, 1.0]],
  "R": [[...], [...], [...]],
  "t": [0.0, 0.0, 0.0]
}
```

`K` 必须为有限、可逆的 `3x3` 矩阵；`R` 必须为有限、正交、行列式为 `+1` 的旋转矩阵；`t` 为世界到相机平移。图像必须是有效 PNG、可解码为灰度、尺寸一致且与相机声明一致。

### 观测清单 JSONL

```json
{"keypoint_id":"rig/corner_00@instance-A","observations":[{"view_index":1,"u":123.25,"v":204.5}]}
```

同一 `keypoint_id` 只允许出现一行；同一行内视图序号不得重复；像素必须满足 `0 <= u < width`、`0 <= v < height`。同名但不同实例必须使用不同的 `@instance-<suffix>`，否则整组按 `E_OBS_INVALID` 拒绝，绝不合并。

### 求解配置

关键字段：

```json
{
  "inputs": {"image_dir": "views", "camera_dir": "cameras", "observations_file": "observations.jsonl", "version_file": "input_version.json"},
  "triangulation": {"min_views": 3, "min_disparity_deg": 20.0, "legal_min_disparity_max_deg": 90.0, "degenerate_relative_tolerance": 1e-10, "degenerate_policy": "fail"},
  "outlier_rejection": {"max_rounds": 2, "threshold_px": 1.5, "legal_threshold_max_px": 1000.0, "min_inliers": 3},
  "quantization": {"levels": [14, 15], "rounding": "round_half_even", "error_tolerance": 0.005},
  "budget": {"unit": "reprojection_evaluation", "global_residual_budget": 20000, "per_keypoint_quota": 2000},
  "scheduler": {"max_workers": 4, "snapshot_commit_interval": 2, "report_precision_digits": 15}
}
```

`quantization.levels` 必须为 `[low_bits, low_bits + 1]`。预算单位固定为“活动观测的一次重投影残差求值”；低/高定点重算均计入配额。单关键点先受 `per_keypoint_quota` 约束，调度器再按关键点字典序扣减全局预算。

## 处理语义

### 加载与校验

- 校验 PNG 容器头、PIL 完整解码、全局固定分辨率、相机视图序号唯一、`K` 可逆、`R` 正交。
- 无效相机产生 `E_CAMERA_INVALID`；引用不可用相机的单个观测产生 `E_OBS_INVALID`。
- 无效观测被移除并进入失败清单，其余观测继续。单关键点失败不会阻断其他关键点。

### 三角化与视差

- 观测固定按视图序号升序。
- 构造齐次线性方程 `uP2-P0`、`vP2-P1`，行归一化后做 float64 SVD；零空间并列、秩不足或数值不可用时输出 `E_TRIANG_DEGENERATE`，并记录参与视图与奇异值。
- 使用初始三维点计算各相机射线夹角。低于 `min_disparity_deg` 的视图对写入 `excluded_view_pairs`；不与任何有效视差对相连的视图写入 `excluded_views`。
- 若保留视图少于 `min_views`，输出 `E_DISPARITY_INSUFFICIENT`，不会静默改用全部视图重三角化。

### 粗差剔除

- 每轮取最大残差观测；残差完全并列时取视图序号最小者。
- 仅当最大残差 `> threshold_px` 时剔除，最多 `max_rounds` 轮。
- 每轮保留剔除前三维点/残差序列、被剔除视图、重新三角化后的状态与残差。
- 若移除会使剩余观测少于 `min_inliers`，输出 `E_INLIER_SHORTAGE`，保留轨迹且不定稿三维点。
- 成功定稿包含最终观测集合指纹。

### 定点量化与误差传播

- 对基础定稿结果按主档位生成定点整数，记录小数位宽、缩放因子与 `round_half_even`。
- 分别以主档位和高一档位完整重跑 DLT、定点化三维点、重投影、残差定点化与剔除链路。
- 逐环节输出三维点 L2 差、残差最大绝对差、逐视图残差差值与首次超过 `error_tolerance` 的环节。
- 若低/高两档末轮剔除观测不同，`removal_flip.origin` 区分：
  - `quantization_truncation`：剔除序列在最后一轮前一致，末轮因定点截断改变最大值。
  - `residual_accumulation_rounding`：更早轮次已分化，源于逐轮定点更新的累积舍入。

### 预算与并发

- `ThreadPoolExecutor(max_workers)` 并发执行关键点。
- 所有结果在汇总后按关键点标识字典序进行预算仲裁，因此线程完成顺序不影响账本、待算队列或报告顺序。
- 单点超出配额：`E_POINT_QUOTA_EXHAUSTED`。
- 全局余额不足：当前及后续的关键点进入 `pending_queue`，并记录 `E_RESIDUAL_BUDGET_EXHAUSTED`。
- 账本每条记录 `before / amount / after / keypoint_id`，满足 `before - amount == after`。

### 缓存与快照

阶段缓存键逐层派生：

```text
load          <- 原始观测内容 + 引用相机指纹 + 加载期配置 + 注入内容
triangulation <- load key + 三角化/视差配置 + 相机指纹
rejection     <- triangulation key + 剔除配置 + 单点配额 + 三角化残差成本
finalization  <- rejection key + 量化配置 + 剔除阶段残差成本
```

- 单观测编辑只改变该关键点及其下游键。
- 仅改剔除阈值时，`load`/`triangulation` 复用，`rejection`/`finalization` 重算。
- 输入未变时全部阶段复用，增量核心报告与空状态全量重建逐字段一致。
- `snapshots/current/` 内含四个阶段文件、`cache.json` 和 `manifest.json`。
- 提交时先写完整临时目录，再整目录切换；提交前比较内容版本，不逐文件覆盖。
- `manifest.json` 记录全局版本、输入版本、配置哈希、相机哈希、已完成关键点、已消耗预算与待算队列。

## 业务码

| 代码 | 含义 |
| --- | --- |
| `E_OBS_INVALID` | 观测 JSONL 行、像素、视图或同名歧义不合法 |
| `E_CAMERA_INVALID` | PNG/相机参数/分辨率/旋转矩阵无效 |
| `E_CFG_INVALID` | 注入后的单点配置越过合法边界（如视差上限、零阈值） |
| `E_TRIANG_DEGENERATE` | 三角化方程退化或零空间并列 |
| `E_DISPARITY_INSUFFICIENT` | 最小视差筛选后视图不足 |
| `E_INLIER_SHORTAGE` | 剔除后剩余观测低于下限 |
| `E_POINT_QUOTA_EXHAUSTED` | 单关键点残差求值配额耗尽 |
| `E_RESIDUAL_BUDGET_EXHAUSTED` | 全局预算不足以继续，关键点进入待算队列 |
| `E_IO` | 单点求解意外异常被隔离 |
| `E_SNAPSHOT` | 快照相关错误码保留 |

## 注入配置

`sample/injections.json` 包含四项，报告 `injections[]` 会给出 `hit`、命中的关键点、实际业务码及注入编号：

| 注入项 | 故障 | 业务码 |
| --- | --- | --- |
| `INJ-001-NONFINITE-PIXEL` | 指定观测 `u=NaN` | `E_OBS_INVALID` |
| `INJ-002-NONORTHOGONAL-ROTATION` | 指定相机旋转加偏斜 | `E_CAMERA_INVALID` |
| `INJ-003-DISPARITY-ABOVE-LIMIT` | 单点最小视差 `180° > 90°` | `E_CFG_INVALID` |
| `INJ-004-ZERO-THRESHOLD` | 单点剔除阈值 `0` | `E_CFG_INVALID` |

注入按关键点隔离：例如 `INJ-001` 只移除目标点的一个观测，该点仍以 5 个观测成功定稿；其他注入目标失败，未注入点继续完成。

## 实际验证结果

`python -m unittest discover -v`：**19/19 通过**。

`python scripts/verify.py`：**17/17 验收检查通过**，主要结果：

- 正常样本：16 点中 14 成功、2 按业务语义失败、0 待算。
- 粗差：`rig/corner_02@instance-A` 剔除视图 4，末轮最大残差 `0.1226 px`。
- 低视差：记录 `(1,2)=9.69°`、`(1,3)=14.68°`、`(2,3)=4.99°` 三个排除对。
- 越界观测：拒绝 1 条，点仍以 6 条有效观测定稿。
- 同名实例：A/B 两个同基名点分别得到不同三维点。
- 增量编辑：仅 `aux/marker@instance-A` 重算，其余 15 点复用，核心报告与全量重建完全一致。
- 配置编辑：`load/triangulation` 16 点全部复用，`rejection/finalization` 按下游失效重算，结果与全量一致。
- 并发：`max_workers=1` 与 `8` 的核心报告逐字段一致。
- 四类注入：4/4 命中且映射到预期业务码。
- 快照：重复运行命中 `exact_snapshot_available`，16 点全部复用，快照 `unchanged`。
- 全局预算为 0：16 点全部按字典序进入待算队列，消耗 0。
- 单点配额收紧为 7：触发 `E_POINT_QUOTA_EXHAUSTED`，账本 `before - amount == after` 全部成立。
- 精度：最终残差最大 `0.1490 px`、RMSE `0.07789 px`；低档三维点最大差 `4.39e-5`，高档 `2.11e-5`。

完整证据见：

- `reports/verification.json`
- `reports/verification.md`
- `reports/base_fresh.json`
- `reports/base_recovered.json`
- `reports/incremental_after_edit.json`
- `reports/fresh_after_edit.json`
- `reports/injection_run.json`
- `reports/budget_exhaustion.json`
- `reports/point_quota.json`

## 关键假设

- 预算是计算预算，单位为活动观测的一次残差求值，而非残差像素值之和。
- 视差“支撑图”中，一个视图必须至少与一个其他视图形成合格视差对；仅有低视差对的视图被排除。
- 当前退化策略固定为 `fail`；遇到退化输出 `E_TRIANG_DEGENERATE`，不尝试丢弃视图后继续。
- 达到 `max_rounds` 但仍高于阈值时停止剔除并记录 `round_limit`；这是配置允许的终止方式。
- JSON 浮点为二进制 float64 最短往返表示，报告不主动截断到 6 位，因此至少保留 6 位有效数字。
