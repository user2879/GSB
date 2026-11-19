# GlyphForge：确定性字形布尔运算与笔画膨胀组件

GlyphForge 是一个纯本地、确定性、仅依赖 Python 3.11 与 NumPy 1.26.4 的字形轮廓处理组件。它覆盖闭合轮廓/开放路径建模、直线与二次/三次贝塞尔扁平化、四类折线布尔运算、开放路径笔画膨胀、8 位灰度覆盖位图，以及容差敏感性、双形态一致性和任务级增量重算审计。

项目不读取真实字体、不访问网络、不使用 GPU、Docker 或显示设备。全部输入由固定种子在本地程序化生成。

## 环境与运行

Windows 10/11 推荐命令：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m glyphforge demo --report reports\demo-report.json
```

也可直接运行兼容包装入口：

```powershell
python run_demo.py --report reports\demo-report.json
```

Linux/macOS：

```bash
python3 -m pip install -r requirements.txt
python3 -m glyphforge demo --report reports/demo-report.json
```

测试：

```bash
python -m unittest discover -s tests -v
```

演示会向标准输出写出一份规范 JSON 报告，并在指定路径写同一份字节流。报告不包含时间戳、随机路径或环境信息，因此同种子两次运行逐字节相同。

## 目录

```text
glyphforge/
  model.py         轮廓节点、段类型、闭合校验、样本序列化
  samples.py       固定种子的五类字形样本
  flatten.py       自适应直线/二次/三次贝塞尔扁平化
  geometry.py      绕数、相交、面积、Hausdorff、像素裁剪
  boolean.py       平面排列法并/交/差/异或
  stroke.py        平头/圆头/方头 + 斜接/圆角/平接
  raster.py        精确像素面积覆盖与双路径一致性核对
  diagnostics.py   对称差面积、Hausdorff、误差区间与漂移判定
  incremental.py   任务依赖图、失效闭包与增量重算
  demo.py          固定端到端序列和确定性报告
tests/             23 个 unittest 测试
reports/           生成后的确定性演示报告
```

## 核心设计

### 数据模型与校验

`Contour` 由一组 `OutlineNode` 组成。首节点是 `move`；后续节点的 `segment_type` 描述到达该节点的段，控制点数量严格匹配：

- `line`: 0 个控制点
- `quadratic`: 1 个控制点
- `cubic`: 2 个控制点

闭合轮廓在末尾重复首点，并在末节点设置 `closed=True`。`GlyphCase` 同时含有独立轮廓源 `source_a`、`source_b` 与开放路径 `open_path`。校验覆盖闭合点一致、坐标有限、首节点类型、段类型及控制点数量。

`generate_suite(seed)` 生成五类样本：简单字形、自相交轮廓、双源重叠/重合边、开放路径、退化段。对相同种子，`canonical_bytes()` 与 `content_hash()` 逐字节一致。

### 曲线扁平化

目标像素尺寸 `s` 推导容差：

```text
tolerance = 0.35 * 64 / s * tolerance_scale
```

直线直接输出端点。二次/三次贝塞尔通过 de Casteljau 递归二分，直到控制点到弦的保守距离上界不超过容差。每一段记录：

- 顶点数
- 实际最大误差估计
- 是否退化
- 退化原因（`zero_length` 或 `coincident_control_points`）

退化段显式标记并跳过，不引入非法顶点。相同输入与容差产生逐点相同折线；容差收紧时，递归叶集只增不减，顶点数单调不减。

### 折线布尔运算

布尔运算不采用顺序 clip-and-retry，而是构建确定性的平面排列：

1. 按声明顺序处理源边。`edge.sort_key()` 使用端点坐标字典序、源标识、源索引、边索引。
2. 对全部边求真交点、端点交点和共线重叠区间，并在交点处切分。
3. 对边端点与交点使用 `1e-9` 坐标量子去重，形成原子边。
4. 对每条原子边左右两侧做非零环绕分类，按并、交、差、异或得到两侧结果状态。
5. 仅保留两侧状态不同的边，并统一调整为“结果内部在左侧”。
6. 按顺时针角序后继重建闭环，合并共线点，并按“外环逆时针、孔洞顺时针”规范化。
7. 环按最小旋转坐标排序，保证重复运算与不同边消解顺序得到相同点序列和哈希。

`BooleanResult` 同时保留：

- `loops`: 规范化、合并共线边后的边界；
- `direct_loops`: 从未合并的定向原子边重建出的直接结构；
- `boundary_edges`: 定向原子边审计信息。

两条结构分别进入统一光栅化路径。边界哈希只包含几何结果，不包含消解顺序元数据，因此顺序不同但几何相同时哈希相同。

### 笔画膨胀

开放路径先按目标尺寸扁平化，再以“段矩形 + 端帽几何 + 拐角几何”的并集构造膨胀轮廓：

- 端帽：`butt`、`round`、`square`
- 拐角：`miter`、`round`、`bevel`

圆头端帽/圆角使用声明容差驱动的正多边形逼近，并记录最大弧高误差。斜接超过 `miter_limit` 时确定性回退到平接。开放中心线两侧各偏移 `width/2`；自交、折返或内部折叠会产生独立孔环，统一由非零环绕布尔并集消解。

宽度非正、端帽/拐角非法分别返回 `E_STROKE_WIDTH`、`E_CAP_JOIN`。

### 统一光栅化

光栅器把字形坐标映射到图像坐标，并对每个像素执行精确多边形裁剪和无符号绕数方向校正。灰度值定义：

```text
gray = round(255 * clamp(signed_clipped_area / pixel_area, 0, 1))
```

因此输出是精确抗锯齿面积覆盖的 8 位量化，而非简单中心点采样。外环逆时针、孔洞顺时针使孔洞面积自然扣除。

双形态核对分别调用：

- `rasterize_loops(result.loops, ...)`
- `rasterize_loops(result.direct_loops, ...)`

若任一像素不同，`check_bitmap_consistency()` 返回首个分歧像素、两路径灰度、边界/直接绕数、像素中心坐标、声明的扁平化容差，以及像素中心水平射线的全部交点。上层可将其升级为 `E_BITMAP_HASH`。

### 敏感性与漂移判定

`region_metrics()` 输出：

- 对称差面积；
- 双向 Hausdorff 距离与最远点/边索引；
- 位图不一致像素数与平均灰度差；
- 声明容差对应的面积、Hausdorff、位图误差上界。

对称差面积优先由精确布尔异或计算。若两条独立扁平化边界形成量子级近重合薄片而使边界链不闭合，则回退到高分辨率 `coverage_field` 精确面积积分，并在报告中显式标出 `method` 与误差界。

只有几何差异超过声明误差区间才判为 `E_CONSISTENCY_DRIFT`。位图存在像素差但几何仍在容差内时，分类为 `WITHIN_TOLERANCE_BITMAP_DIFFERENCE`，不会仅凭像素差报错。

### 增量重算与失效闭包

`IncrementalEngine` 为每个字形、尺寸、源与操作建立任务节点：

```text
flat/{glyph}/{size}/{a|b|open}
bool/{glyph}/{size}/{operation}
stroke/{glyph}/{size}
bitmap/bool/{glyph}/{size}/{operation}
bitmap/stroke/{glyph}/{size}
summary/{glyph}
```

依赖图决定失效闭包。示例：

- 修改轮廓控制点只失效该源的 flat、对应 bool、布尔位图与字形摘要；
- 修改笔画宽度/端帽/拐角只失效 stroke、笔画位图与字形摘要；
- 修改扁平化容差失效该字形全部 flat 及其后代；
- 未进入闭包的字形摘要哈希保持不变。

`recompute(closure)` 逐个任务执行，并断言实际执行集合与闭包逐项一致；不一致抛出 `E_CLOSURE_MISMATCH`。演示同时用全新引擎做全量重建，要求增量摘要与全量摘要完全相同。

## 业务错误码

| 错误码 | 触发场景 |
|---|---|
| `E_OUTLINE_INVALID` | 闭合错误、非法段类型、控制点数量/坐标非法、退化轮廓无法形成边界 |
| `E_BOOLEAN_DEGENERATE` | 边界排列无法唯一重建或出现非可忽略开放链 |
| `E_STROKE_WIDTH` | 宽度为零/负/非有限，或笔画几何为空 |
| `E_CAP_JOIN` | 非法端帽/拐角/容差/斜接限制 |
| `E_CONSISTENCY_DRIFT` | 几何差异超出声明误差区间 |
| `E_CLOSURE_MISMATCH` | 实际重算集合、依赖缺失或全量/增量结果不一致 |
| `E_BITMAP_HASH` | 边界路径与直接结构光栅化并非逐像素一致 |

## 报告内容

`reports/demo-report.json` 包含：

- 样本种子与规范样本 SHA-256；
- 每个字形、5 档尺寸的扁平化顶点/退化统计；
- 四类布尔结果的边界哈希、边界位图哈希、直接位图哈希与一致性计数；
- 笔画边界哈希、位图哈希、弧逼近误差；
- 紧/松容差与另一种消解顺序下的三种差异度量和误差区间；
- 三次增量编辑的根任务、完整闭包、实际重算集合、未受影响字形哈希与全量重建校验；
- 错误码自检结果。

实际测试与报告摘要见 [VERIFICATION.md](VERIFICATION.md)。

## 约束与说明

- 组件以字形单位 `0..64` 为默认视口，12/16/24/32/48 为位图边长。
- 边界坐标保存到 12 位小数；布尔排列内部量化为 `1e-9` 坐标格。
- 光栅一致性要求传入规范化后的结果环；`BooleanResult` 与 `StrokeResult` 已自动完成规范化。
- 所有排序、哈希、样本生成与遍历顺序均为显式确定性；报告不含平台或时间字段。
