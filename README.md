# vgcover：矢量填充覆盖率与误差账本

一个可在 Windows 上直接从源码运行、也可安装为 Python 组件的矢量填充领域实现。它把
“场景 -> 子轮廓离散化 -> 浮点覆盖率 -> 8 位量化 -> 预乘 BGRA 合成 -> 误差账本”的整条链路
做成确定性、可逐层哈希、可由第三方独立复算的流程；同时提供精确的增量失效闭包。

组件不依赖外部素材，四类场景均由固定种子在本地生成。核心依赖只有 NumPy，测试使用 pytest。

## 快速开始（Windows）

```bat
py -3 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt

python examples\run_demo.py --out output
python -m pytest
python -m vgcover.cli verify --scene output\scene.json --ledger output\ledger.json
```

在 Linux/macOS 上把 `py -3` 和反斜杠替换为 `python3` 与 `/` 即可。

示例会实际写出并校验：

- `scene.json` / `scene.sha256`：确定性场景与内容哈希。
- `ledger.json` / `ledger.sha256`：逐区域和整图的误差传播账本。
- `composite.bgra`：`uint8`、预乘 BGR + 直通 alpha 的 BGRA 缓冲。
- `regions/region_XXXX/*.f32|*.i32|*.u8|*.bgra`：覆盖率、规则差异、依赖边数、灰度、
  量化/重建误差与单区域图层。
- `manifest.json`：所有产物的 dtype、shape、SHA-256，可再次独立校验。
- `summary.json`：场景、量化、分层和账本摘要。
- `incremental_demo.json`：四次增量修改的闭包、实际重算集合与不变量结果。
- `catalogue/*.json`：四类单类别场景。
- `incremental/`：增量修改后的完整可验证产物。

## 核心 API

```python
from vgcover import (
    IncrementalRenderer,
    export_render,
    generate_scene,
    render_scene,
    verify_ledger,
)

scene = generate_scene(seed=20261119)
render = render_scene(scene)
assert verify_ledger(scene, render.ledger)["ok"]
export_render(render, "output")

engine = IncrementalRenderer(scene)
plan = engine.plan_changes([
    {"region_id": 10, "field": "tolerance", "value": 0.06}
])
receipt = engine.apply_changes(
    [{"region_id": 10, "field": "tolerance", "value": 0.06}],
    expected_closure=plan.closure,
)
assert sorted(set(receipt.recomputed)) == sorted(set(receipt.closure))
assert receipt.outside_layer_hashes_preserved
assert receipt.outside_ledger_entries_preserved
```

支持的区域字段修改：

- `{"region_id": R, "field": "tolerance", "value": T}`
- `{"region_id": R, "field": "fill_rule", "value": "nonzero" | "evenodd"}`
- `{"region_id": R, "field": "alpha", "value": A}`
- `{"region_id": R, "field": "fill_color", "value": [R, G, B]}`
- `{"region_id": R, "field": "control_points", "contour_index": C, "segment_index": S, "value": [...]}`（同时接受 `segment_control_points` 别名）

## 场景与确定性

`generate_scene(seed, width, height)` 固定产出以下四个区域：

| 区域 ID | 类别 | 轮廓 | 默认规则 | 特征 |
|---:|---|---|---|---|
| 10 | `simple_convex` | 单椭圆，4 个三次贝塞尔段 | nonzero | 简单凸形 |
| 20 | `self_intersecting` | 5 点自交星形，直线段 | evenodd | 两种规则结果明显不同 |
| 30 | `multi_nested` | 外环 ccw、孔 cw、岛 ccw | nonzero | 多重嵌套与方向 |
| 40 | `near_degenerate_strip` | 宽度 0.22 px 的旋转窄条 | evenodd | 显式标记、保留并参与计算 |

所有坐标在生成时固定到 6 位小数；规范化 JSON 使用排序键、紧凑分隔符和 UTF-8。相同参数调用两次，
`scene_json_bytes(scene)` 逐字节一致，`scene_hash(scene)` 相同。窄条不会被丢弃，场景中带有：

```json
"degenerate": {
  "flagged": true,
  "reason": "minimum_feature_width_below_threshold",
  "minimum_feature_width_px": 0.22,
  "threshold_px": 0.5,
  "retained": true
}
```

坐标系为左上原点、x 向右、y 向下；`orientation` 按有符号 shoelace 面积标注，正面积为 `ccw`。

## 算法与口径

### 1. 子轮廓离散化

- 直线段直接保留为一条弦，解析误差为 0。
- 三次贝塞尔使用确定性二分 de Casteljau 细分。叶节点接受条件是内部两个控制点到该叶弦段的最大距离
  `<= tolerance`；这是曲线位于控制点凸包内的严格解析上界。
- `measured_chord_height` 在叶节点参数 `0.25/0.5/0.75` 处采样原曲线并计算到弦段的最大距离。
- 同一轮廓/容差两次结果逐点一致；容差收紧时细分树只会继续向下细化，因此顶点数单调不减。
- 非有限或 `tolerance <= 0` 返回 `E_TOLERANCE_RANGE`，并带 `region_id`。

### 2. 浮点覆盖率

使用像素中心 `(x + 0.5, y + 0.5)` 向右水平射线。边采用半开 y 区间 `[min_y, max_y)`，交点严格满足
`x_intersection > sample_x`。一次栅格化同时得到：

- `nonzero`：有向环绕计数非零即覆盖。
- `evenodd`：右向交点计数奇偶决定覆盖。
- `dependency_edge_counts`：该像素实际依赖的右向交点边数。
- `difference = nonzero - evenodd`：逐像素 `float32` 差异。

覆盖率图始终为 `float32`。非法规则返回 `E_FILL_RULE`，不会回落默认值。

### 3. 8 位量化

声明口径：

```text
step        = 1/255
rounding    = nearest_half_up
gray        = floor(coverage * 255 + 0.5)
reconstruct = gray / 255
max error   = 0.5/255
```

组件输出每像素有符号量化误差、重建误差、最大绝对误差、超差像素数、均值/RMS/P95 和 16 桶误差直方图。
相同 `float32` 覆盖率图两次量化得到逐字节相同的 `uint8` 灰度图。

### 4. 预乘 BGRA 合成

区域按 `region_id` 升序覆盖透明背景。每个区域先构造源图层：

```text
a8  = floor(gray * alpha + 0.5)
c_p = floor(fill_color * a8 / 255 + 0.5)
out = src_premultiplied + floor(dst * (255 - src_alpha) / 255 + 0.5)
```

BGRA 中 BGR 预乘、A 为有效 alpha。每层通道舍入误差上界保守声明为 2 个 u8 code value，即
`2/255` 覆盖率单位；账本和合成步骤都记录该上界与累计上界。图层内容哈希覆盖该区域独立的预乘 BGRA
缓冲，`layer_hash` 还覆盖场景输入哈希、几何哈希、覆盖率哈希、灰度哈希、规则、颜色和 alpha。

### 5. 误差传播账本

统一单位是“整图覆盖率率值”（`[0,1]` 内、按整图像素数归一化的平均遮盖/alpha 比例）。每个区域：

```text
delta_px   = 区域最大解析弦误差 + 1e-12 坐标序列化舍入界
perimeter  = 离散折线周长
band_area  = 2 * delta_px * perimeter + pi * (2 * delta_px)^2
E_geometry = min(1, band_area / (width * height))
E_quant    = 最大逐像素绝对量化误差
E_composite= 2 / 255
E_region   = min(1, E_geometry + E_quant + E_composite)

E_image    = min(1, sum(alpha_i * E_region_i))
```

账本保存公式、策略、每层哈希、传播链与总上界；区域条目与图层哈希严格一一对应。任一必需分量缺失时，
该区域与整图的 `total_error_bound` 为 `null`、`complete=false`，缺失项列在
`missing_components`，绝不默认补零。

`verify_ledger(scene, ledger)` 只依赖场景文档和账本内声明的策略，从头重算并逐区域、整图、缓冲与层哈希
比对；篡改或场景不一致返回 `E_LEDGER_MISMATCH`（区域局部差异会带 `region_id`）。

## 增量失效闭包

节点依赖图：

```text
geometry:r                <- 控制点/容差
coverage_nonzero:r        <- geometry:r
coverage_evenodd:r        <- geometry:r
coverage_selected:r       <- 两种覆盖率 + fill_rule
quantize:r                <- coverage_selected:r
layer:r                   <- quantize:r + fill_color + alpha
ledger_region:r           <- layer:r + 几何/量化/合成分量
composite                 <- 所有 layer:r（按 ID 升序）
ledger_image              <- 所有 ledger_region:r + composite
```

闭包规则：

- 控制点或容差：几何、两张覆盖率、选中覆盖率、量化、图层、区域账本、整图合成、整图账本。
- 填充规则：选中覆盖率、量化、图层、区域账本、整图合成、整图账本。
- alpha 或颜色：图层、区域账本、整图合成、整图账本。

`plan_changes()` 给出闭包清单；`apply_changes(..., expected_closure=...)` 若清单不等会先抛
`E_CLOSURE_MISMATCH`。执行后回执同时给出 `recomputed`，并与 `closure` 做集合等值断言。未受影响区域的
`layer_hash`、`layer_content_hash` 和账本条目会被前后比较，若发生闭包外变化立即报错。

## 业务错误码

| 错误码 | 含义 | 定位字段 |
|---|---|---|
| `E_SCENE_INVALID` | 场景/段类型/控制点非法 | region / contour / segment |
| `E_FILL_RULE` | 非法填充规则 | region |
| `E_TOLERANCE_RANGE` | 容差非有限或 `<= 0` | region |
| `E_ALPHA_RANGE` | alpha 非有限或不在 `[0,1]` | region |
| `E_CONTOUR_UNCLOSED` | 首尾或相邻段不连续 | region / contour / segment |
| `E_CLOSURE_MISMATCH` | 闭包或实际重算集合不一致 | 单区域时带 region |
| `E_LEDGER_MISMATCH` | 账本/哈希/产物复算不一致 | 局部差异时带 region |
| `E_COVERAGE_RANGE` | 覆盖率非二维、非有限或越界 | 可用时带 region |

异常对象提供稳定的 `.code`、`.region_id`、`.contour_index`、`.segment_index` 和 `.to_dict()`。

## 代码结构

| 文件 | 职责 |
|---|---|
| `vgcover/scene.py` | 确定性场景、验证、规范化 JSON/哈希 |
| `vgcover/geometry.py` | 贝塞尔/直线离散化、解析与实测误差 |
| `vgcover/raster.py` | 像素中心水平射线、双规则覆盖率与边依赖 |
| `vgcover/quantize.py` | 8 位量化、误差分布、重建 |
| `vgcover/composite.py` | 预乘 BGRA、source-over、分层哈希 |
| `vgcover/ledger.py` | 统一口径、传播链、完整性语义 |
| `vgcover/engine.py` | 全量渲染、独立复算、增量闭包 |
| `vgcover/export.py` | JSON/二进制/清单导出与校验 |
| `vgcover/cli.py` | `demo` / `render` / `verify` 命令 |

更完整的公式、哈希边界与复算步骤见 `docs/technical-notes.md`。

## 实际验证结果

本工作区已执行：

```text
python3 examples/run_demo.py --out /tmp/vgcover-demo-1
  -> 39 个产物文件通过 manifest SHA-256 校验
  -> ledger complete=True
  -> image_total_error_bound=0.02817769079797107
  -> 四次增量变更 actual_matches_closure=True
  -> 每次 outside_layer_hashes_preserved=True / outside_ledger_entries_preserved=True

python3 -m vgcover.cli verify --scene ... --ledger ...
  -> ok=True，4 个区域逐项复算一致

同一配置输出到两个独立目录后执行 diff -rq ... ...
  -> 无差异（所有 JSON、账本与二进制缓冲逐字节一致）

python3 -m pytest -q
  -> 26 passed
```

这里没有使用 Docker、外部素材或网络目标；示例和测试只生成本地确定性样本。
当前可执行环境本身是 Linux，因此以上结果用 `python3` 实测；实现只使用 `pathlib`、显式 UTF-8/LF 和标准
命令行参数，不依赖 shell 或 POSIX 路径，Windows 运行命令见“快速开始”。
