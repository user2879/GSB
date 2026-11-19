# 技术说明：独立复算与误差链

本文档说明 `vgcover` 的实现口径，重点是可被第三方重放的确定性规则和误差上界。

## 1. 规范化与哈希边界

### 1.1 JSON

- 场景和账本哈希使用 `sort_keys=True`、紧凑分隔符、UTF-8、禁止 NaN/Infinity 的规范 JSON。
- 场景生成坐标使用 6 位小数；离散化报告中的诊断值存储为 12 位小数。
- 原始浮点数组不进入 JSON；它们以显式小端 dtype 的 C-order 字节写入二进制文件。

### 1.2 数组哈希

数组哈希摘要包含：

```text
"vgcover-array-v1" + dtype + shape + C-order bytes
```

覆盖率为 `<f4`，依赖边数为 `<i4`，灰度/缓冲为 `<u1`。这样 Windows/Linux 上不会因默认字节序不同而产生
不同 hash。

### 1.3 分层哈希

`layer_content_hash` 覆盖单区域独立预乘 BGRA 缓冲、尺寸、dtype 和区域 ID。

`layer_hash` 再覆盖：

- 区域规范化输入哈希；
- 几何哈希（包含容差和所有段报告）；
- nonzero、evenodd、选中覆盖率、规则差异和依赖边图哈希；
- 选中填充规则；
- 量化策略版本、灰度哈希、量化误差哈希；
- 填充色与 alpha；
- 图层内容哈希。

因此控制点、容差、规则、颜色或 alpha 的任一变化都会改变 `layer_hash`；账本中每个区域恰好对应一个
图层，整图账本以 `region_id -> layer_hash/layer_content_hash` 记录该映射。

## 2. 三次贝塞尔离散化

设一段三次贝塞尔曲线控制点为 `P0,P1,P2,P3`。曲线完全位于四个控制点的凸包内。
对弦段 `P0P3`，凸包内任一点到该弦段的距离不超过内部顶点 `P1/P2` 到弦段距离的最大值：

```text
flatness(P0,P1,P2,P3) = max(dist_segment(P1, P0P3), dist_segment(P2, P0P3))
```

当 `flatness > tolerance` 时，在 `t=0.5` 使用 de Casteljau 二分。叶节点记录 `flatness` 作为解析上界；
同时在叶参数 `0.25/0.5/0.75` 采样原曲线，记录到该叶弦段的最大距离作为实测弦高估计。

性质：

- 确定性：所有分割点固定为二分点，无随机数、无自适应浮点阈值。
- 单调性：容差收紧只会继续细分已经保留的叶，不会撤销父节点已产生的采样边界；因此折线顶点数不减。
- 误差：每条原始段的 `analytic_error_bound` 是其所有叶上界的最大值。
- 安全上限：最大细分深度 24；若触顶，报告 `depth_limited_leaf_count`，账本使用实际解析上界而不假定达标。

## 3. 覆盖率栅格化

### 3.1 射线规则

对像素中心 `(px+0.5, py+0.5)`：

1. 只考虑 y 半开区间包含采样 y 的非水平边：`min(y0,y1) <= y < max(y0,y1)`。
2. 计算交点 `x_int`。
3. 只把 `x_int > sample_x` 的交点计入右向依赖边。

半开区间确保顶点恰好落在采样线时不会被重复计数。`dependency_edge_counts` 就是这些右向交点的数量。

### 3.2 两种规则

- `evenodd`：`count % 2 == 1`。
- `nonzero`：每条右向边的有向增量按 `y1 > y0 ? +1 : -1` 求和，和不为零即覆盖。

两种规则共用同一次交点/依赖计算，再按各自判定生成 `float32` 图。差异图直接为
`nonzero - evenodd`，取值 `-1/0/1`。

## 4. 量化

```text
q    = floor(float64(coverage) * 255 + 0.5)
gray = uint8(clip(q, 0, 255))
c_hat= float32(gray) / 255
err  = c_hat - coverage
```

输入越界不裁剪而是 `E_COVERAGE_RANGE`，从而保证“超差像素数”统计诚实。

理论最大绝对误差为 `0.5/255`。由于 float32 表示，代码在超差判定中使用 `+1e-7` 的数值安全余量；
账本仍记录理论界和实际最大误差。直方图范围覆盖实际最大值，确保所有像素被计数。

## 5. 预乘合成

### 5.1 源图层

给定 `uint8 gray`、`uint8 RGB`、`alpha in [0,1]`：

```text
a8  = floor(gray * alpha + 0.5)
cp  = floor(color * a8 / 255 + 0.5)
```

`cp` 为预乘 BGR，`a8` 为 alpha code。灰度为零的像素得到全透明源层。

### 5.2 Source-over

```text
out_alpha = sa + floor(da * (255-sa) / 255 + 0.5)
out_color = sc + floor(dc * (255-sa) / 255 + 0.5)
```

所有 0..255 无符号整数运算；分母为奇数，因此整数分子不会出现恰好半值，`+127 // 255` 即最近舍入。

### 5.3 舍入界

每层保守界为 2 个 u8 code value：

- 有效 alpha 量化：<= 0.5；
- 预乘颜色舍入：<= 0.5（alpha 误差的二次影响被颜色系数 `<1` 吸收）；
- 目的项缩放/alpha 更新舍入：<= 0.5 至 1.0；
- 采用 2.0 的保守上界。

每步记录 `step_rounding_error_bound_u8=2.0` 和累计 `2.0*(step+1)`。整图账本按区域 alpha 加权累计。

## 6. 统一误差口径

几何误差原本是像素距离；覆盖率量化误差是 `[0,1]` 覆盖率误差；合成舍入是 u8 code value。统一为
整图覆盖率率值：

```text
delta      = analytic_error_bound + 1e-12
E_geom_cov = min(1, (2*delta*perimeter + pi*(2*delta)^2) / (W*H))
E_quant_cov= max_abs_quantization_error
E_comp_cov = 2/255
E_region   = min(1, E_geom_cov + E_quant_cov + E_comp_cov)
E_image    = min(1, sum_i(alpha_i * E_region_i))
```

几何项来自边界扰动带的面积：边界移动 `delta` 后，精确与离散化覆盖集合的对称差位于宽度约 `2*delta`
的边界带内；用折线周长近似边界长度，并加入一个 `pi*(2*delta)^2` 的角部余量。所有常数和公式都在
账本 `ledger_policy` 中显式持久化。

## 7. 账本完整性

每个区域必须有：

```text
geometry
quantization
composite
```

每个分量必须含有限的非负 `coverage_equivalent_error`。缺失时：

- `region.complete=false`；
- `region.total_error_bound=null`；
- `region.missing_components=["..."]`；
- `image.complete=false`；
- `image.total_error_bound=null`。

不使用零值补齐。正常渲染路径总是生成三分量；完整性语义主要用于第三方账本审计和损坏检测。

## 8. 独立复算步骤

第三方只读取 `scene.json` 和 `ledger.json`，无需任何本地二进制，按以下顺序即可复算：

1. 校验场景结构、闭合性、规则、alpha、容差和唯一区域 ID。
2. 按区域容差离散化所有段，得到点列、解析界、实测界和周长。
3. 对点列执行像素中心水平射线，生成两张覆盖率图、依赖边计数和差异图。
4. 按区域 `fill_rule` 选择覆盖率，执行 `nearest_half_up` 量化，计算误差统计。
5. 按 ID 升序构造预乘源层并执行 source-over，得到 BGRA 和逐层内容哈希。
6. 根据账本策略计算区域与整图误差上界，重建 `layer_hash`。
7. 与 `ledger.regions[*].layer_hash`、`image.layer_hashes`、`buffer_hash`、`total_error_bound` 和
   `complete` 比对。

代码入口为 `vgcover.verify_ledger(scene, ledger)` 或：

```bat
python -m vgcover.cli verify --scene output\scene.json --ledger output\ledger.json
```

## 9. 增量节点与不变量

节点命名：

```text
geometry:r10
coverage_nonzero:r10
coverage_evenodd:r10
coverage_selected:r10
quantize:r10
layer:r10
ledger_region:r10
composite
ledger_image
```

变更不会调用 `render_scene()` 全量入口。增量引擎从缓存中复用未命中的节点，并执行闭包内节点。
`apply_changes` 完成后比较闭包集合与实际节点集合；若不一致抛 `E_CLOSURE_MISMATCH`。随后统一比较所有
闭包外区域的 `layer_hash` 与账本条目，确保未受影响内容不变。

## 10. 产物校验

`manifest.json` 为每个产物记录相对路径、dtype、shape、字节数和文件 SHA-256。`verify_export_manifest`
重新计算文件哈希。`ledger.sha256` 保存规范化账本哈希，`scene.sha256` 保存规范化场景哈希（不是文件
排版哈希）。因此 JSON 缩进方式改变不会改变业务内容哈希，但文件清单哈希会如实反映字节变化。
