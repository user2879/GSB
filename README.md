# Panorama Mask Workbench

面向本地确定性合成图像的逐图全景分割、像素掩码合并、语义/实例一致性校验与增量重算工作台。项目只依赖 Python 3.10+、NumPy 与 Pillow，不使用 Docker、GPU 或外部服务。

## 快速开始

Windows PowerShell：

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install numpy pillow
```

Linux/macOS：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy pillow
```

生成固定种子的本地演示样本、YAML、命令流与替换图：

```bash
python -m panorama_workbench generate-demo --out .demo
python -m panorama_workbench preflight \
  --samples .demo/samples \
  --config .demo/config.yaml \
  --commands .demo/commands.json
python -m panorama_workbench run \
  --samples .demo/samples \
  --config .demo/config.yaml \
  --state-dir .demo/state
```

应用交互命令流运行增量批次：

```bash
python -m panorama_workbench run \
  --samples .demo/samples \
  --config .demo/config.yaml \
  --commands .demo/commands.json \
  --state-dir .demo/state \
  --report .demo/report.json
```

`--drain` 会在有预算待办时继续下一批，每批使用配置中的 `pixel_budget`；`--stop-after-images N` 用于模拟可处理退出信号，当前批次已启动图像提交后以退出码 `130` 结束。

## 目录

- `panorama_workbench/config.py`：YAML 配置解析、枚举/范围校验、配置指纹。
- `panorama_workbench/validation.py`：PNG 头/解码/尺寸、真值矩形/类别/面积校验。
- `panorama_workbench/inference.py`：确定性滑窗特征、定点投影、量化残差、连通碎片。
- `panorama_workbench/merging.py`：稳定顺序 IoU 合并、面积过滤、实例重编号与截断。
- `panorama_workbench/evaluation.py`：语义准确率、PQ、一对一实例匹配与逐实例诊断。
- `panorama_workbench/engine.py`：线程池调度、预算台账、瓦片/掩码/实例缓存、增量与恢复。
- `panorama_workbench/snapshot.py`：四阶段产物临时目录发布、版本清单与整体切换。
- `panorama_workbench/samples.py`：固定种子 PNG/真值/命令流生成器。
- `tests/`：校验、确定性、合并、质量、预算、并发、增量、快照恢复测试。

## 输入格式

### 真值 JSON

图像与真值按同 stem 配对，例如 `scene_01.png` 对应 `scene_01.json`。

```json
{
  "width": 64,
  "height": 64,
  "instances": [
    {
      "id": 1,
      "category": 1,
      "bbox": [5, 7, 18, 16],
      "area": 288,
      "mask_pixels": [[5, 7], [6, 7]]
    }
  ]
}
```

- `category` 可写类别编码或类别名；编码必须属于配置类别集合。
- `bbox` 为 `[x, y, w, h]` 整数像素矩形，必须完全落在图像内。
- `area` 必须是正整数且不超过矩形面积；`mask_pixels` 可选，若提供则其数量必须等于 `area`。
- 未提供 `mask_pixels` 时，评估层按矩形逐行确定性截取 `area` 个像素构造真值掩码。

### 配置 YAML

```yaml
categories:
  - code: 0
    name: background
    weight: 1.0
  - code: 1
    name: shape_a
    weight: 1.0
window:
  size: 16
  stride: 8
merge:
  iou: 0.08
  min_instance_area: 20
quantization:
  fixed_bits: 10
  levels: [1, 4, 16, 64]
  level: 4
scheduling:
  max_workers: 3
  pixel_budget: 6000000
  max_instances_per_image: 24
  cache_budget: 1024
  snapshot_interval: 2
quality:
  min_semantic_accuracy: 0.45
  min_instance_quality: 0.12
  instance_match_iou: 0.35
```

`fixed_bits` 允许 `4, 8, 10, 12, 16, 20, 24, 32`；当前量化档位必须在 `quant_levels` 中。`cache_budget` 表示驻留瓦片记录上限，超出后按最近使用版本和缓存键确定性淘汰。

### 命令流 JSON

```json
{
  "commands": [
    {"op": "replace_image", "image": "scene_02.png", "source": "replacements/scene_02.png"},
    {"op": "add_category", "category": {"code": 4, "name": "accent", "weight": 0.001}},
    {"op": "remove_category", "code": 4},
    {"op": "set_merge_iou", "value": 0.12},
    {"op": "set_min_instance_area", "value": 28},
    {"op": "set_quant_level", "value": 64},
    {"op": "add_image", "image": "scene_09.png", "source": "incoming/scene_09.png"},
    {"op": "remove_image", "image": "scene_09.png"}
  ]
}
```

相对路径以命令流文件所在目录为基准。命令按数组顺序应用；单条命令失败会生成 `E_CONFIG_INVALID`，该命令不进入后续处理。

## 确定性算法

1. 滑窗位置由窗口尺寸、步长和图像尺寸确定；边界位置显式补齐，行优先固定。
2. 每块计算颜色均值、边缘密度、纹理/梯度等 14 维整数特征；投影矩阵由配置推理指纹与类别编码/名称/权重的 SHA-512 派生。
3. 所有特征、投影权重和得分均使用 `fixed_bits` 对应定点整数。量化按当前档位整数步长执行，平局由类别编码升序决定。
4. 每块记录 `up_count`、`down_count`、`none_count`、残差和、绝对残差和及极值。
5. 瓦片内按类别做四邻域连通分量；碎片按分块行列、碎片顺序稳定合并。IoU 用整数交集/并集和 `Fraction` 比较，避免浮点边界误差；等于阈值时仍合并并记录平局依据。
6. 冲突像素按面积、类别、位置和代表碎片键确定性归属；随后按面积降序、类别编码、最小行/列排序重编号，最后执行单图实例上限截断。

## 缓存与增量失效

缓存分为三层：

- 特征键：瓦片内容指纹 + 推理配置指纹 + 块行列/坐标。
- 掩码键：特征键 + 类别集合指纹 + 量化档位。
- 实例键：全部瓦片掩码内容哈希 + 合并配置指纹 + 图像尺寸。

失效规则：

| 编辑 | 瓦片特征 | 瓦片掩码 | 实例结果 |
|---|---|---|---|
| 图像未改、配置未改 | 命中 | 命中 | 命中 |
| 替换图像局部 | 仅内容变化块失效 | 仅语义变化块失效 | 引用变化块的实例失效 |
| 增删未使用类别 | 命中 | 语义不变则复用 | 不变则命中 |
| 调整 IoU/最小面积 | 全部命中 | 全部命中 | 合并配置变化，重做合并 |
| 切换量化档位 | 相关块失效 | 重算 | 重做合并与评估 |

报告同时给出实际执行审计（本次重算/复用）与当前缓存依赖；全量与增量重建的核心业务字段一致，缓存命中清单用于证明未重新推理。

## 预算与调度

- 图像任务按“上批待办优先、同组图像名升序”排序，分配时使用全局剩余预算作为当前单图配额。
- 配额不足的图像记录 `E_IMAGE_BUDGET_EXHAUSTED`；剩余预算不足以覆盖任何待办时，未处理图像进入 `pending_queue` 并记录 `E_PIXEL_BUDGET_PENDING`。
- 线程池大小严格取 `max_workers`，每批图像任务并发数不会超过配置值。
- 单图执行期间继续累计推理、语义打分和合并掩码像素；超过已分配配额即中止该图，不静默产出半结果。
- 预算记录含 `quota_pixels`、`estimated_pixels`、`reserved_pixels`、`consumed_pixels` 与状态。

## 快照与恢复

四阶段为 `load`、`inference`、`merge`、`evaluation`。阶段产物先在临时目录完成写入与 `fsync`，再整体 rename 到新版本目录；`manifest.json` 原子切换并记录：

- 全局版本号；
- 配置哈希；
- 已完成图像；
- 本批/累计预算消耗；
- 各阶段产物版本与内容哈希。

恢复判定业务码：

- `R_NEW_RUN`：无历史状态。
- `R_STABLE_RERUN`：配置和清单完整，可直接复用。
- `R_RESUME_CHECKPOINT`：同配置存在未完成批次，续跑。
- `R_CONFIG_RECONCILE`：配置变化，按内容缓存做精确重算。
- `R_FULL_REBUILD`：状态/清单校验失败或缺少可信 checkpoint，重建。

退出码：`0` 成功；`2` 加载/配置失败；`3` 预算待办；`4` 质量低于阈值；`5` 运行异常；`130` 收到可处理退出信号并完成当前批次提交。

## 诊断报告

`--report` 或 `.state/reports/latest.json` 包含：

- 配置与各指纹、全局版本、恢复业务码、峰值并发；
- 排序后的失败清单；
- 预算扣减记录与待办队列；
- 瓦片/掩码/实例命中数、重算块清单、淘汰数；
- 每图实例清单、逐像素语义栅格与实例栅格、质量指标；
- 一对一匹配、未匹配预测、漏检真值和原因码；
- 被最小面积过滤、被实例上限截断的清单。

## 验证记录

执行：

```bash
python -m unittest discover -s tests -v
```

实际结果：13 项测试全部通过，约 11.3 秒。覆盖：

- `E_IMAGE_FORMAT`、`E_TRUTH_INVALID`、`E_CONFIG_INVALID` 与失败项隔离；
- 同块重复推理完全一致，量化档位改变特征键、残差与真实评估指标；
- 最小面积丢弃、实例上限截断、实例排序；
- 并发 1 与并发 4 的逐图核心输出完全一致；
- 预算不足时产生 `E_IMAGE_BUDGET_EXHAUSTED` / `E_PIXEL_BUDGET_PENDING`，下一批优先推进待办；
- IoU/最小面积编辑时瓦片与掩码零重算；
- 图像局部替换时仅相关瓦片失效；
- 类别无语义影响编辑时掩码与实例全部复用；
- 多次编辑后的增量输出与最终样本/配置的一次性全量重建逐字段一致；
- 优雅停止后续跑、状态损坏后的 `R_FULL_REBUILD`；
- 稳定重跑阶段产物版本不变。
- 缓存驻留预算淘汰后，稳定重跑与后续合并编辑仍保持可用和一致。

演示规模（64×64、窗口 16、步长 8、4 图）：首次全量生成 196 个瓦片，稳定重跑 `tile_hits=196`、`tile_misses=0`；仅改 IoU/最小面积时同样 196 个瓦片和 196 个掩码全部命中，仅重做合并。

机器可读的本次验证摘要见 `artifacts/verification.json`。

## 说明

- 本地样本生成器让形状颜色来自类别颜色派生函数，因此无需真实训练即可验证完整链路；算法本身不读取真值，真值只进入校验与评估阶段。
- `pixel_budget` 是单批执行预算，不是持久累计上限；累计消耗单独记录在状态与清单中。
- 全量重建与增量重建的核心业务字段一致；实际缓存命中/重算清单应分别作为执行审计读取，二者天然描述不同执行路径。
