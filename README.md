# 确定性多分辨率修复金字塔诊断工作台

本项目在本地实现一个面向图像修复与逐层风格迁移的诊断工作台。它把源图构造成分辨率金字塔，从最粗层开始做确定性扩散修复，再用上一层上采样结果作为本层低频引导，并在每层执行量化查表风格映射，最后按层权重合成。

核心不是单分辨率邻域填充，而是可审计的跨层、跨瓦片失效：编辑某一层后，只重算该层及上层依赖到编辑区域的瓦片；其余瓦片从缓存/快照直接复用。每条命令后的增量状态都可与从相同最终配置出发的全量重建逐像素、逐层指纹和逐瓦片缓存键比较。

## 快速开始

要求 Python 3.10+、`numpy`、`Pillow`、`PyYAML`。不需要网络、容器或外部服务。

```bash
# 方式一：直接使用源码
PYTHONPATH=src python -m inpaint_pyramid --help

# 方式二：可编辑安装后使用 ipw
python -m pip install -e .
ipw --help
```

生成默认 128x128 固定种子样本并运行：

```bash
PYTHONPATH=src python -m inpaint_pyramid generate \
  --output samples --size 128 --levels 4 --tile-size 16 \
  --tile-budget 8 --threads 4

PYTHONPATH=src python -m inpaint_pyramid validate \
  --config samples/config.json \
  --commands samples/commands.json \
  --images samples

PYTHONPATH=src python -m inpaint_pyramid run --reset \
  --config samples/config.json \
  --commands samples/commands.json \
  --images samples \
  --state-dir artifacts/demo-state \
  --output-dir artifacts/demo-output \
  --report artifacts/demo-report.json

PYTHONPATH=src python -m inpaint_pyramid recover \
  --state-dir artifacts/demo-state

PYTHONPATH=src python -m inpaint_pyramid verify \
  --config samples/config.json \
  --commands samples/commands.json \
  --images samples \
  --report artifacts/demo-verify.json
```

`run` 默认是增量模式；`run --mode full` 从最终生效配置一次性全量重建，用于对照。`verify` 会在预算允许时逐命令比较，并在预算造成待办时记录延后项，最终清空待办后做完整对照。

## 关键语义

### 金字塔与定点数

- `pyramid.downsample_factors` 长度为层数，`factor[0]` 必须为 `1`，其余因子单调不减；第 `l` 层尺寸为底图尺寸除以 `product(factor[0..l])`。
- 下采样是块均值，整数定点下使用 half-up 舍入。
- 上采样是边缘钳制的双线性插值，坐标映射为 `src=(dst+0.5)*src_size/dst_size-0.5`，恰好半值时向正方向 half-up，规则写入报告与快照清单。
- 所有像素与中间量均为 `int64`。RGB 初始左移 `fixed_point.fraction_bits` 位，输出时 half-up 回到 8 位。

### 逐层修复

- 最粗层在瓦片内从掩码外已知像素由外向内做 Jacobi 扩散，八邻域权重为 `cardinal=2, diagonal=1`，等价于按 `1/d²` 缩放后的反平方权重。
- 其余层的像素先验为 `(1-g)*本层初始像素 + g*上采样低频引导`，扩散候选与先验按固定 1:1 混合；所有舍入为 half-up。
- 修复是瓦片局部的，不使用未声明的跨瓦片隐藏状态。因此缓存瓦片可安全拼接，增量与全量使用完全相同的局部规则。
- `retention.mask_fill=preserve_source` 时掩码像素直接保留本层输入；层报告与快照清单记录实际掩码集合、保留集合和风格决断集合的计数与指纹。

### 风格映射与合成

- 每层按 `style.quant_levels_by_layer` 构造 256 项、3 通道的查表；输入值先量化到该层档位，再查风格参考图的对应分位值。
- `style.weights_by_layer` 是 RGB 分通道映射强度；`retention.style_inside` / `style_outside` 决定掩码内外取修复结果还是映射结果。
- 合成把所有层双线性上采样到底图尺寸，按 `style.composition_weights` 加权后定点归一化。
- 报告记录每层贡献数组指纹、通道和/均值、权重，以及最终合成指纹；`contributions.npz` 保存完整逐层贡献数组。

## 失效与调度

每个瓦片缓存键由以下内容派生：

1. 本层瓦片输入指纹：层/瓦片坐标、源瓦片内容、掩码瓦片内容、掩码保留策略。
2. 低频引导指纹：当前层对应引导瓦片内容及直接引用的下层瓦片集合。
3. 参数指纹：迭代次数、引导强度、风格逐通道权重、量化档位、保留策略、定点宽度、算法版本等。
4. 显式量化档位与缓存键版本。

依赖关系记录在 `layer_summaries[*].tiles[*].lower_refs`。上采样使用双线性支持域，因此一个上层瓦片可能引用一个或多个下层瓦片。

- 修改某层掩码矩形时，旧矩形和新矩形覆盖的该层瓦片都会失效，再沿上层依赖闭包传播。不会失效任何更低层瓦片。
- 修改某层迭代次数、引导强度、风格权重或量化档位时，该层所有瓦片失效，并向上传播。
- 未失效瓦片保留 `cache_key`、产物版本与像素结果。报告中的 `reused_tiles` 与 `recomputed_tiles` 可直接审计。

每批重算预算来自 `scheduler.recompute_tile_budget`。分配顺序固定为 `(level,row,col)` 行优先，并避开依赖尚未有效的上层瓦片；扣减账本先于并发执行生成，所以与线程数无关。预算不足时：

- 未完成瓦片保持待办；
- 状态为 `E_TILE_BUDGET_PENDING`；
- 后续命令继续优先处理更小层号、行号、列号的待办；
- `budget.ledger` 记录每个瓦片的 `remaining_before/remaining_after`。

线程池只并发执行同一层中已就绪的瓦片，最大并发不超过 `scheduler.threads`。任务结果按稳定键收集，失败也按稳定顺序进入 `failures`。

## 快照与恢复

```text
state/
  CURRENT                       # 原子替换的提交指针
  snapshots/
    v00000015_<statehash>/      # 不可变、完整快照目录
      manifest.json
      state.json
      COMPLETE
      report.json
      layers/*.png
      tiles/l00_r0000_c0000.npy
      composite.png
      contributions.npz
      retention_masks.npz
```

- 新快照完全写入临时目录并 `fsync`，随后整体 rename 为不可变版本目录，最后原子替换 `CURRENT`。
- 相同 `state_hash` 不重复生成目录；版本变化时创建新目录，不做逐文件覆盖。
- `manifest.json` 记录全局版本、配置哈希、命令游标、待办瓦片、每层每瓦片版本、层/合成指纹、线程上限与并发峰值。
- `retention_masks.npz` 保存逐层掩码、源保留像素集合、掩码内外风格映射决断集合的完整布尔数组。
- `recover` 只读取 `CURRENT` 指向且 `COMPLETE`/manifest 哈希全部通过的快照。临时目录、未提交的完整目录或损坏目录均不会被采用。
- `SIGINT`/`SIGTERM` 只设置停止标志；当前命令完成后提交快照并以进程退出码 `0` 退出。

## 配置与命令

配置结构见 `samples/config.json`。主要字段：

```json
{
  "image": {"source": "image_01.png", "style": "style_ref.png"},
  "pyramid": {"levels": 4, "downsample_factors": [1,2,2,2], "tile_size": 16, "tile_order": "row_major"},
  "masks": [{"layer": 0, "rects": [[39,37,32,25]]}],
  "repair": {"iterations_by_layer": [2,3,4,5], "guide_strength_by_layer": [0.12,0.3,0.48,0.66]},
  "retention": {"mask_fill": "diffuse", "style_inside": "mapped", "style_outside": "mapped"},
  "style": {
    "quant_levels_by_layer": [6,10,14,18],
    "weights_by_layer": [[0.08,0.06,0.04],[0.105,0.08,0.055],[0.13,0.1,0.07],[0.155,0.12,0.085]],
    "composition_weights": [1.0,0.5,0.3333333333,0.25]
  },
  "fixed_point": {"fraction_bits": 12},
  "scheduler": {"threads": 4, "recompute_tile_budget": 8},
  "snapshot": {"commit_every_commands": 1}
}
```

配置非法时输出 `E_CONFIG_INVALID`。命令流支持：

| 类型 | 参数 | 语义 |
| --- | --- | --- |
| `set_mask_rect` | `layer`, `rect` | 替换该层掩码矩形 |
| `clear_mask` | `layer` | 清空该层掩码 |
| `set_iterations` | `layer`, `value` | 修改该层迭代次数 |
| `set_guide_strength` | `layer`, `value` | 修改该层低频引导强度 |
| `set_style_weight` / `set_style_weights` | `layer`, `weights` | 修改该层 RGB 风格权重 |
| `set_quant_levels` | `layer`, `levels` | 修改该层量化档位 |
| `noop` | 无 | 不修改参数，可用于优先清空既有预算待办 |

非法命令产生 `E_COMMAND_INVALID`，当前命令跳过，命令游标继续前进，后续命令和待办调度不受影响。

## 验收结果

本次交付实际执行结果：

- `PYTHONPATH=src:. python -m unittest discover -s tests -v`：17 项测试全部通过。
- 默认 128x128、4 层、16px 瓦片、85 个瓦片、预算 8、4 线程，15 条命令（含 1 条故意非法命令）。
- 增量命令 1、2、5、6 出现 `E_TILE_BUDGET_PENDING`；后续命令优先清空待办，最终待办为空。
- 默认样本逐层指纹：
  - L0 `0783290ca04513c1805c6d4f0d8b906dbc5fc02c975ec93840f4b27947b5b5f7`
  - L1 `158a39cbcd1785cfe4760e04a5a3501aa399348d84526d12c41c1b34974ff821`
  - L2 `751b6d88eb0be554f2f3d3d10f6f21f441f99471c803681966a55d14a6ecfc42`
  - L3 `27c12e03e1ad929cb5bce7b9f4dd8f0785efd0d58349c0e93ea069463f5fc010`
- 增量与全量合成指纹均为 `ba86242c5a0b641a8d512159f262ceb1a04678e6f510f63062b7f48489cfc23c`。
- `verify` 中 11 条命令在无待办状态下即时全等，4 条因预算延后，最终全等且差异列表为空。
- 真实 `SIGTERM` 测试：进程退出码 `0`，恢复点版本 `1`，待办为空。

留存物：

- `samples/`：三张同分辨率源图、风格参考图、配置、命令流、样本清单。
- `artifacts/demo-report.json`：增量诊断报告、逐命令失效/复用/重算/待办、预算账本与层耗时。
- `artifacts/demo-verify.json`：增量/全量逐命令与最终对照。
- `artifacts/demo-recover.json`：快照恢复点。
- `artifacts/demo-output/`：各层 PNG、合成 PNG、逐层贡献数组。
- `artifacts/demo-full-output/` 与 `artifacts/demo-full-report.json`：全量重建对照产物。
- `artifacts/demo-state/`：不可变快照、瓦片产物与 `CURRENT` 指针。

## 设计边界

- 初始 bootstrap 是完整构建，不消耗“单批重算”预算；预算只约束命令触发的增量重算批次。
- 修复扩散刻意保持瓦片局部，避免不可见跨瓦片状态破坏精确缓存复用；这优先服务可审计一致性，而不是追求无缝图像质量。
- 当预算不足时，当前命令后的状态只能部分更新，报告明确标记 `E_TILE_BUDGET_PENDING`；待办清空后，增量结果与全量结果必须全等。
- 同一目录中的非风格 PNG 作为同分辨率备选源图校验，`image.source` 选择本次运行的源图。

详细实现可从 `src/inpaint_pyramid/config.py`、`pyramid.py`、`repair.py`、`style.py`、`engine.py`、`snapshot.py` 和 `cli.py` 阅读。
