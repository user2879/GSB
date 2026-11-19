# 技术设计

## 1. 组件边界

核心包 `vgsynth` 不依赖 GUI。可单独调用：

- `generate_scene(kind, seed, scene_id)`：确定性场景生成。
- `flatten_scene(scene, tolerance)`：子路径扁平化与逐段误差报告。
- `rasterize_fill(flat_path, width, height)`：非零环绕与奇偶两套覆盖率。
- `rasterize_stroke(flat_path, width, height, stroke_width)`：描边轮廓与覆盖率。
- `composite_scene(scene, flat_scene)`：分层预乘 Alpha 合成。
- `apply_injections(scene, config, params)`：纯参数/数据层异常注入。
- `run_case(...)` / `run_demo(...)`：磁盘产物与固定序列。
- `verify_case_dir(case_dir)`：独立磁盘验证入口。

PySide6 仅在 `vgsynth.viewer` 中使用。

## 2. 场景模型

规范化场景：

```json
{
  "schema": "vgsynth.scene/v1",
  "scene_id": "...",
  "kind": "simple_cross",
  "seed": 123,
  "canvas": {"width": 192, "height": 144},
  "paths": [
    {
      "path_id": "p0",
      "fill_rule": "nonzero",
      "fill_color": {"r": 232, "g": 108, "b": 45, "a": 220},
      "stroke": {"color": {"r": 32, "g": 68, "b": 180, "a": 255}, "width": 2.4},
      "subpaths": [
        {
          "subpath_id": "s0",
          "closed": true,
          "segments": [
            {"type": "line", "p0": [40.0, 50.0], "p1": [100.0, 50.0]},
            {"type": "cubic", "p0": [100.0, 50.0], "c1": [110.0, 50.0], "c2": [110.0, 60.0], "p1": [110.0, 60.0]}
          ]
        }
      ]
    }
  ],
  "seed_audit": {
    "root_seed": 123,
    "derivation": "sha256('vgsynth/v1|<root_seed>|<decision_id>')[0:8]",
    "entry_count": 4,
    "entries": []
  },
  "content_hash": "..."
}
```

所有 JSON 使用 `sort_keys=True`、紧凑分隔符、UTF-8 和单一结尾换行。`content_hash` 对移除自身后的场景计算 SHA-256。

填充时只使用 `closed=true` 的子路径。开放子路径仍参与描边，端帽使用圆帽。

## 3. 确定性种子树

每个随机决策独立派生：

```text
sub_seed = first_8_bytes(SHA256("vgsynth/v1|" + root_seed + "|" + decision_id))
```

随机数使用 SplitMix64，不依赖 Python `random` 实现版本。审计条目记录：

- `decision_id`
- `purpose`
- `sub_seed`
- 算法名
- 参数范围
- 输出值

因此可独立复算每个决策。固定样本包含简单交叉、自相交环、重叠大面积路径和退化路径。

## 4. 扁平化

### 4.1 直线

直线直接输出两端点，误差为 `0`。零长直线跳过并记录 `skip:zero-length-line`。

### 4.2 三次贝塞尔

使用 de Casteljau 二分。平坦度定义为两个内部控制点到端点弦的最大垂距：

```text
flatness = max(distance(c1, chord(p0,p1)), distance(c2, chord(p0,p1)))
```

当 `flatness <= tolerance` 或达到深度上限时输出叶弦。每段记录：

- `error_estimate`：叶节点最大控制点到弦距离上界。
- `error_method`：`control-point-to-chord-bound`。
- `vertex_count`：该段离散顶点数。
- `action`：输出、跳过或降级动作。

同一容差重复调用逐点相同。二分树是确定性的，因此容差收紧时叶分裂只增不减，顶点数单调不减。

### 4.3 退化规则

- 所有控制点同点：`skip:zero-length-cubic`。
- 控制点共线且端点不同：降级为端点直线，记录 `degrade:collinear-cubic-to-line`。
- 控制点共线且端点同点：跳过，记录 `skip:collapsed-collinear-cubic`。
- 未知段类型：`skip:unknown-type`。

每条记录都进入 `flatten.json` 与 `case_report.json` 的逐段摘要。

## 5. 扫描线填充

每个像素行使用 8 条垂直子扫描线。子扫描线位于像素内部半开区间，水平射线与折线边按半开规则求交：

```text
(y0 <= y < y1) or (y1 <= y < y0)
```

同一 X 的交点先分组。

- 非零环绕：按边的 Y 方向累加有向环绕数，环绕数非零的区间为覆盖。
- 奇偶规则：累加交点奇偶，奇数区间为覆盖。

X 方向不采用单点判定，而是将扫描线区间与像素列区间求重叠长度，并对 8 条子扫描线取平均，最后量化为 8 位覆盖率：

```text
coverage8 = floor(clamp(area_coverage, 0, 1) * 255 + 0.5)
```

每条路径始终同时输出 `fill_nonzero.raw` 与 `fill_evenodd.raw`。二者差异记录覆盖像素数与首个行优先差异位置。

## 6. 描边生成

扁平化折线按以下方式构造描边轮廓：

1. 每条边按半宽生成同向矩形四边形。
2. 每个顶点生成 12 边形圆盘，形成圆角连接。
3. 开放路径端点同样由圆盘形成圆帽。
4. 所有多边形统一为正面积方向，再用非零环绕规则光栅化，从而得到重叠部分的并集。

输出描边覆盖率、轮廓多边形数、顶点数和轮廓内容哈希。描边宽度为零时不生成轮廓。

## 7. 预乘 Alpha 合成

像素格式固定为 `BGRA8_PREMULTIPLIED`。给定覆盖率 `c` 和颜色 `(r,g,b,a)`：

```text
src_a  = round(a * c / 255)
src_b  = round(b * src_a / 255)
src_g  = round(g * src_a / 255)
src_r  = round(r * src_a / 255)
```

每层先合成填充，再合成描边：

```text
out = src + dst * (1 - src_a)
```

最终画布从透明开始，按层 ID 升序执行同样的 source-over 合成。

层内容哈希：

```text
SHA256(canonical_layer_header + 0x00 + all_BGRA_bytes)
```

层头包含：

- 层 ID、路径 ID、声明/实际填充规则。
- 非零覆盖哈希、奇偶覆盖哈希、实际选中规则覆盖哈希。
- 描边覆盖哈希、描边轮廓哈希。
- 宽度、高度、像素格式、固定合成顺序标记。

因此任一覆盖位图、规则选择、层头字段或整体像素内容变化都会改变层哈希。

## 8. 异常注入

注入配置是独立可序列化 JSON，不修改源码。支持：

| 类型 | 目标 | 恢复策略 | 业务码 |
|---|---|---|---|
| `coordinate_out_of_bounds` | 段控制点 | 渲染前钳制到画布 | `E_SCENE_INVALID` |
| `degenerate_segment` | 段 | 按退化规则跳过/降级 | `E_PATH_DEGENERATE` |
| `fill_rule_invalid` | 路径 | 回退 `nonzero` | `E_FILL_RULE` |
| `fill_alpha_out_of_range` | 路径填充色 | 钳制 `0..255` | `E_COLOR_RANGE` |
| `stroke_alpha_out_of_range` | 路径描边色 | 钳制 `0..255` | `E_COLOR_RANGE` |
| `tolerance_negative` | 渲染参数 | 取有限绝对值 | `E_TOLERANCE_RANGE` |
| `stroke_width_zero` | 路径描边 | 跳过描边 | `E_PATH_DEGENERATE` |
| `layer_pixel_mutation` | 输出层 | 落盘层缓冲确定性翻转一个通道 | `E_VERIFIER_MISMATCH` |
| `layer_hash_mutation` | 输出层清单 | 用确定性派生值替换清单层哈希 | `E_LAYER_HASH` |

两个操作写入同一字段时，后一个操作不执行并产生 `E_INJECTION_CONFLICT`。目标无法解析、注入 ID 重复、未知类型同样归类为 `E_INJECTION_CONFLICT`，不会导致进程崩溃。

全部恢复动作与原始值写入 `injection.json`、`generation.json`、`issues.json` 和 `case_report.json`。

## 9. 独立验证器

验证入口只接收样本目录。它重新读取：

- `scene.json`
- `injection.json`
- `layers.json`
- `final.bgra`
- 层 BGRA 文件

不接收合成器返回对象，也不复用其内存状态。验证器重新执行注入、场景校验、扁平化、双规则填充、描边和预乘合成，并检查：

1. 场景内容哈希。
2. 注入异常分类与位置是否与层清单声明一致。
3. 每层头部字段及双规则覆盖哈希。
4. 每层内容哈希、像素哈希与落盘像素哈希。
5. 落盘层 BGRA 与复算 BGRA 的逐像素值。
6. 最终 BGRA 与 `canvas_hash`。

任一不一致返回：

```text
code = E_VERIFIER_MISMATCH
first_mismatch = {layer_id, y, x, channel, violation_type, expected, actual}
```

层按 `L0000`、`L0001` 升序检查；层内按行优先、再按 X 升序找首个差异。通道偏移按 BGRA 顺序细分到字节。

层哈希本身不一致时同时记录 `E_LAYER_HASH`。像素变异可同时触发 `E_LAYER_HASH` 与顶层 `E_VERIFIER_MISMATCH`。

## 10. 最小反例

验证失败或出现注入发现时，按以下顺序选择目标：

1. 验证器不一致：使用 `first_mismatch` 的层、坐标、通道和注入 ID。
2. 否则：使用 `issues.json` 中按层 ID、像素行优先顺序得到的首个业务发现。

反例构造：

- 只保留目标路径；目标定位到子路径/段时只保留对应子路径/段。
- 只保留触发目标发现的注入操作，并重映射路径/子路径/段索引。
- 对路径、子路径、段、注入操作逐项执行删除测试。
- 删除后若目标违规消失，则该元素标记为必要。
- 目标签名要求只出现目标违规；删除后若引入 `E_SCENE_INVALID` 等额外错误，不视为同一违规被复现。
- 最终候选会再写入独立 `replay_run`，完整执行一遍生成、合成和验证。

`counterexample.json` 包含 `elements`、`necessary_elements`、每项删除结果、复现报告和反例哈希。`irreducible=true` 表示每个保留元素删除后目标违规都不再触发。

## 11. 错误码

| 错误码 | 含义 |
|---|---|
| `E_SCENE_INVALID` | 场景结构、哈希或坐标边界无效 |
| `E_PATH_DEGENERATE` | 零长/共线退化段或零宽描边 |
| `E_FILL_RULE` | 非 `nonzero` / `evenodd` 规则 |
| `E_COLOR_RANGE` | 颜色分量超出 `0..255` |
| `E_TOLERANCE_RANGE` | 容差非有限正数 |
| `E_LAYER_HASH` | 层哈希、像素哈希或层像素内容不一致 |
| `E_VERIFIER_MISMATCH` | 合成器产物与独立验证复算结论不一致 |
| `E_INJECTION_CONFLICT` | 注入目标冲突、重复或无法解析 |

## 12. 版本化与可复现

- 场景、层、生成清单、报告与哈希清单均有 `schema` 字段。
- 报告不包含时间戳或绝对输出路径。
- `run_id` 仅由样本规格与版本化配置计算。
- 文件名和目录顺序固定。
- PNG 使用固定压缩级别、无时间元数据。
- `hashes.json` 对除自身与 `run_manifest.json` 外的全部产物记录 SHA-256。

这些约束保证相同输入、参数和运行环境下两次运行逐字节一致。
