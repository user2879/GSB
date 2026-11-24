# GSB：确定性多分辨率高斯场景增量组件

这是一个面向本地确定性样本的三维场景表示优化组件与实验环境。它从零实现了输入校验、逐视图高斯核初始化、多分辨率瓦片渲染、定点量化、层级合成、误差诊断、内容指纹缓存、预算调度、阶段快照和故障注入隔离。

项目不依赖 Docker、数据库或网络服务，只使用 Python 3.11、NumPy、Pillow 和 pytest。

## Windows 快速开始

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt

# 生成确定性样本（64x64、6 视图、8 步编辑、5 个注入项）
python -m gsscene generate --out artifacts/run/sample --width 64 --height 64 --views 6 --force

# 校验样本
python -m gsscene validate --sample artifacts/run/sample

# 运行完整验收：增量 / 全量 / 串行 / 预算 / 注入
python -m gsscene experiment --sample artifacts/run/sample --out artifacts/run/experiment --regenerate --width 64 --height 64 --views 6

# 自动化测试
pytest -q
```

如果在 Linux/WSL 下执行，步骤相同，只需使用 `.venv/bin/activate`。

## 目录

```text
gsscene/
  validation.py     输入扫描、业务码校验、注入样本物化
  state.py          编辑状态、持久化视图、编辑应用
  config.py         配置模型、层级网格、层级参数指纹
  kernels.py        确定性高斯核初始化、逐核缓存键、瓦片引用
  rendering.py      定点采样、量化残差、瓦片与层级合成
  diagnostics.py    高一位宽参考路径、相对误差、排序归因
  cache.py          NPZ/元数据缓存、精确复用、引用清单
  scheduling.py     线程池、核预算、单批点数预算、待办区间
  snapshots.py      四阶段临时目录写入与整体替换、续跑判定
  pipeline.py       端到端编排、失效账本、报告生成
  experiment.py     验收实验驱动
  compare.py        报告字段和缓存数组的逐字段比较
scripts/            样本生成和实验入口
tests/              23 个自动化测试
```

确定性样本默认写入：

```text
artifacts/run/sample/
  images/view_0000.png ... view_0005.png
  depth/view_0000.npy ... view_0005.npy
  cameras.json
  config.json
  edits.jsonl
  injections.json
  assets/replace_0001/
  assets/add_0006/
```

## 输入约定

`cameras.json` 是相机对象数组。每项包含 `view_index`、3×3 `intrinsic`、3×3 正交 `rotation`、长度 3 的 `translation`。视图图与相机通过 `view_index` 一一对应。

`config.json` 的关键字段：

```json
{
  "image_size": [64, 64],
  "base_scale": 1.2,
  "levels": 3,
  "scale_decay": 0.6,
  "level_density": [1.0, 0.75, 0.5],
  "sh_degree": 1,
  "frac_bits": 12,
  "quant_gear": "NEAREST",
  "render_samples": 4,
  "kernel_budgets": [4096, 4096, 4096],
  "point_budget": 20000,
  "concurrency": 4,
  "snapshot_interval": 2,
  "tile_size": 16,
  "tolerance": {"kernel": 0.001, "render": 0.02, "composite": 0.02}
}
```

`edits.jsonl` 每行一个对象，支持：

- `view_replace`：引用 `asset_dir` 与目标 `view_index`。
- `view_add` / `view_remove`：新增或删除视图。
- `camera_tweak`：`rotation_delta` / `translation_delta`，旋转增量经 SVD 重新正交化。
- `level_scale` / `level_density`：修改指定层。
- `sh_modify`：精确指定视图、层、核序号、SH 系数、颜色通道和增量。
- `quant_switch`：切换定点位宽或 `NEAREST`/`FLOOR`/`CEIL`。

`injections.json` 是独立注入清单，支持 `truncate_image`、`missing_camera`、`zero_scale_decay`、`density_over_budget`、`quant_gear_out_of_range`。实验会对每项建立独立输入副本，写入 `injection_hits.json`，并在报告中给出注入编号、命中项和映射业务码。

## 处理流程

1. **加载校验**：扫描全部图像，校验 PNG/JPEG/BMP 容器、固定分辨率、RGB 模式；校验深度形状和有限值；校验内参可逆、旋转正交、视图序号唯一且图像/相机一一对应；校验配置枚举、预算与并发；逐行校验编辑计划。
2. **核初始化**：每层网格由有效尺度比与密度确定。每个核从颜色、深度、相机和层固定投影矩阵计算位置、三轴尺度、不透明度与 SH 系数。所有计算使用 float64，核按 `row, col` 升序排列。
3. **瓦片渲染**：层内核按深度、行列稳定排序；每个像素执行固定数量采样，采样权重、颜色和 alpha 均量化为 int64 定点数。参考路径固定多一位小数位宽。
4. **层级合成**：层 0 保留其定点渲染；更高层按有效尺度衰减因子与下层合成，再量化。每层保存整数像素、浮点值、量化残差和舍入方向。
5. **诊断**：逐核、逐像素保存误差数组；按容差输出首次超限环节；若层级误差排序相对上次运行变化，则归因到核参数、量化或合成顺序。
6. **缓存与失效**：根据内容键复用已持久化 NPZ。只有键不匹配的核、瓦片或合成结果才计算。
7. **快照**：`load`、`kernel`、`render`、`composite` 四阶段写入 `snapshots/staging/`，再整体替换 `snapshots/committed/<stage>/`。`manifest.json` 记录全局版本、配置哈希、相机哈希、量化设置、已完成视图/层和各阶段版本。

## 缓存键与失效边界

缓存键使用规范化 JSON、数组 dtype/shape/原始字节的 SHA-256 派生。

```text
K(v,l,i) = H(view_content_fp, camera_fp, level_fp(l), i, row, col, sh_mods)
G(v,l)   = H(sorted K(v,l,*))
T(v,l,r,c) = H(referenced kernel ids+keys, tile params, render samples, frac_bits, gear)
C(v,l)   = H(current tile keys, lower C key, blend factor, frac_bits, gear)
```

`level_fp` 只包含该层自己的有效尺度、密度、网格和固定投影种子，因此修改第 1 层不会误伤第 0/2 层的核。

实际失效闭包：

| 编辑 | 高斯核 | 渲染瓦片 | 层级合成 |
| --- | --- | --- | --- |
| 视图替换 / 外参微调 | 该视图所有层 | 该视图所有瓦片 | 该视图所有层 |
| 层尺度 / 层密度 | 该视图该层 | 该层瓦片 | 该层到最高层 |
| SH 修改 | 指定核序号 | 引用该核的瓦片 | 该层到最高层 |
| 量化档位/位宽 | 无 | 所有活动瓦片 | 所有活动合成 |
| 视图新增 | 仅新视图 | 仅新视图 | 仅新视图 |
| 视图删除 | 无重算，删除孤儿 | 无重算，删除孤儿 | 无重算，删除孤儿 |

稳定遍历顺序固定为 `(view_index, level, kernel_index)`、`(view_index, level, tile_row, tile_col)`、`(view_index, level)`。报告同时给出去重后的键清单和逐核/逐瓦片动作清单。

说明：`actual_invalidation` 表示本次缓存的真实动作，因此“空白全量构建”必然是全部 recompute，而“已稳定增量运行”可以是全部 reuse；二者不应按动作相同来比较。验收比较的是最终逐核参数、瓦片像素、合成像素、逐核/瓦片/合成缓存键、诊断字段和失败清单。并发与串行从同一初始缓存状态运行时，动作清单也完全相同。

## 预算与并发

- 线程池最大并发数来自配置，执行上限永远不会超过配置；测试可用 `worker_override` 将执行上限降到 1，但不改变配置哈希或产物。
- 核预算按 `(view, level)` 独立核算，按稳定顺序先分配再并发执行。
- 单批点数预算限制一个 `(view, level)` 批次可提交的核数量。
- 超核预算记录 `E_KERNEL_BUDGET_EXHAUSTED`；超单批点数预算记录 `E_POINT_BUDGET_PENDING`。
- 待办使用稳定的半开区间 `[start, end)` 保存，避免极端密度输入产生百万级逐项记录。
- 每次运行报告 `budget_ledger`、`pending`、配置并发、有效并发和观测到的最大在飞任务数。

## 快照恢复与故障隔离

启动时比较快照 schema、配置族哈希和状态。图像尺寸、层数、瓦片大小、SH 阶数等布局变化会触发整体重建；尺度、密度、量化、视图内容等可精确增量处理的编辑会保留兼容缓存。`force_rebuild=True` 会清空状态、缓存与快照。

注入故障在独立副本上物化，故障视图/层不会进入后续诊断统计。例如截断视图不会生成核/瓦片，密度超预算的指定层不会进入诊断。

## 实际验证结果

最近一次完整运行为 64×64、6 视图、3 层、8 步编辑：

- 最终活动视图：`[0, 1, 2, 4, 5, 6]`。
- 最终产物：18 个核组、28,656 个核、288 个瓦片、18 个层级合成。
- 增量最终与从空白全量重建：核/瓦片/合成数组全部 `np.array_equal=True`。
- 并发 4 与串行 1：数组、缓存键、诊断和失效动作相同。
- 增量最终稳定后再运行：`recompute_kernels=0`、`recompute_tiles=0`、`recompute_composites=0`、复用 28,962 项。
- 量化切换：`0` 个核重算，`288` 个瓦片、`18` 个合成重算。
- SH 单核修改：仅 `2` 个核、`2` 个瓦片、`3` 个合成重算。
- 视图删除：`0` 个重算，剩余产物全部复用。
- 预算实验命中 `E_KERNEL_BUDGET_EXHAUSTED` 与 `E_POINT_BUDGET_PENDING`。
- 五个注入项全部命中：`E_IMAGE_TRUNCATED`、`E_CAMERA_MISSING`、`E_CONFIG_SCALE_DECAY`、`E_INJECTION_DENSITY_OVER_BUDGET`、`E_CONFIG_QUANT_GEAR`。
- `pytest -q`：`23 passed`。

完整报告位于 `artifacts/run/experiment/`：

```text
comparison.json          总体验收结论
incremental_steps.json   8 步编辑的逐步报告
incremental_final.json   增量最终状态
full_rebuild.json        空白全量重建
serial.json              串行执行
budget.json              预算扣减账本与待办
injection.json           独立注入命中报告
final_sample/            烘焙后的最终样本
```

## 业务码摘要

输入：`E_IMAGE_HEADER`、`E_IMAGE_TRUNCATED`、`E_IMAGE_RESOLUTION`、`E_IMAGE_MODE`、`E_DEPTH_MISSING`、`E_DEPTH_SHAPE`、`E_CAMERA_MISSING`、`E_CAMERA_INTRINSIC`、`E_CAMERA_ROTATION`、`E_VIEW_DUPLICATE`、`E_VIEW_IMAGE_ORPHAN`、`E_VIEW_CAMERA_ORPHAN`。

配置与编辑：`E_CONFIG_LEVELS`、`E_CONFIG_SCALE_DECAY`、`E_CONFIG_DENSITY`、`E_CONFIG_FRAC_BITS`、`E_CONFIG_QUANT_GEAR`、`E_CONFIG_KERNEL_BUDGET`、`E_CONFIG_CONCURRENCY`、`E_EDIT_TYPE`、`E_EDIT_VIEW_MISSING`、`E_EDIT_LEVEL`、`E_EDIT_SH_COEFF`、`E_ASSET_MISSING`。

运行：`E_INJECTION_DENSITY_OVER_BUDGET`、`E_KERNEL_BUDGET_EXHAUSTED`、`E_POINT_BUDGET_PENDING`、`E_CACHE_CORRUPT`、`E_SNAPSHOT_MANIFEST`。

所有失败项同时记录来源、视图、层、注入编号和稳定排序键，不会被静默跳过。
