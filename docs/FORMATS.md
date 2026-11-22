# 交换格式、定点规则与缓存设计

## v1 定宽文本

每行 62 个字符，无分隔符；字段按 UTF-8 字符计宽。换行 `\n` 不计入宽度。

| 字段 | 宽度 | 格式 | 说明 |
|---|---:|---|---|
| image_name | 18 | 左对齐，空格填充 | PNG 文件名 |
| region_index | 3 | 右对齐，`0` 填充 | 1..999 |
| x | 5 | 右对齐，`0` 填充 | 整数像素 |
| y | 5 | 右对齐，`0` 填充 | 整数像素 |
| w | 5 | 右对齐，`0` 填充 | 整数像素 |
| h | 5 | 右对齐，`0` 填充 | 整数像素 |
| style_permille | 5 | 右对齐，`0` 填充 | 0..1000 |
| repair_code | 2 | 整数码 | 见下表 |
| attribute_code | 2 | 整数码 | 见下表 |
| padding | 12 | 全空格 | 缺失位/保留位 |

修补码：`1=mean`, `2=edge_extend`, `3=noise_guard`。

属性码：`1=warm`, `2=cool`, `3=vivid`, `4=mono`, `5=solar`。

业务码：

- 行短于 62 字符：`E_FIELD_TRUNCATED`。
- 行超宽、整数非法、padding 非空格、码表未知、重复键：`E_STRUCTURE_INVALID`。
- 图名/区域序号不在清单、矩形越界：`E_REGION_OUT_OF_RANGE`。

## v2 JSON

```json
{
  "schema": "image-edit-exchange/v2",
  "version": 2,
  "precision": "decimal-source/float-number-serialization",
  "regions": [
    {
      "image": "scene_alpha.png",
      "region_index": 1,
      "rect": {"x": 18.6, "y": 22.4, "w": 68.0, "h": 48.8},
      "style_strength": 0.4237,
      "target_attribute": "warm",
      "repair_method": "mean",
      "precision_label": "v2:decimal-source",
      "rotation_deg": 6.5,
      "feather_radius": 3.25
    }
  ]
}
```

所有字段必填。坐标、强度、旋转和羽化使用十进制字符串进入中间表示；写出 JSON 时使用数值。非有限数、未知命名属性/修补方式、越界或清单不匹配都会进入失败清单。

## 降级旁路

v2→v1 时 `exchange_v1.sidecar.json` 逐区域保存：

- 原始 `source_rect` 与四舍五入后的 `v1_rect`。
- 坐标字段残差 `source - v1_integer`。
- 原始强度、千分比强度和强度残差。
- 原始精度标签、旋转角、羽化半径。
- 命名属性/修补方式与原始来源指纹。

旁路结构与 v1 行逐字段核对；不匹配时计为 miss，并回退到普通 v1→v2 精确整数升级。旁路文件的规范化 JSON SHA-256 写入快照清单。

## 定点渲染

- 坐标/旋转/羽化：整数缩放，掩码坐标 8 位小数，三角常数 20 位定点。
- 风格强度：十进制强度乘以 `2^fixed_fraction_bits` 后 `half_up` 取整。
- 量化档位：`round_half_up(strength * (levels - 1))`，每档有确定性通道增益与偏移 LUT。
- 修补：按半径枚举邻域权重；每轮只填充“至少有一个已知邻居”的边界像素，由外向内推进；所有通道加权和用整数除法 `half_up`。
- 属性替换：warm/cool/vivid/mono/solar 的整数通道变换。
- 合成：8 位 alpha 的 `source-over` 整数混合。
- PNG：Pillow `compress_level=9`, `optimize=False`，不写入时间戳或外部元数据。

产物元数据 `outputs/incremental/metadata.json` 和每阶段 artifact header 记录量化档位、定点小数位宽、舍入方式与确定性标记。

## 缓存键

```text
parse_key     = H(parse, exchange_version, source_format, canonical_region)
mask_key      = H(mask, parse_key, image_fp, image_size, coord_fixed_bits, trig_bits)
repair_key    = H(repair, mask_key, image_fp, repair_radius, fixed_bits, rounding)
attribute_key = H(attribute, repair_key, target_attribute, fixed_bits, rounding)
style_key     = H(style, attribute_key, style_strength, levels, fixed_bits, rounding)
compose_key   = H(compose, style_key, mask_key, exchange_version, alpha_bits, rounding)
```

阶段产物是内容寻址的确定性二进制文件。区域记录保存阶段键、上游键、图像块、摘要、产物路径和内容指纹；文件缺失时自动视为缓存 miss。

## 预算与待办

- 调度开销估算为掩码包围盒面积；旋转时使用包围盒对角线，羽化扩大范围。
- 每批从 `recompute_region_budget` 开始扣减。
- 旧 pending 优先，但各自仍按 `(图名, 区域序号)` 稳定排序。
- 第一个无法满足预算的区域及其后的新区域进入 pending；预算日志逐条记录 `seq`、`required_pixels`、`remaining_before/after`、优先级和业务码。
- `--resume` 从最新快照 `scenario_state` 恢复 pending 与配置。

## 快照

```text
state/incremental/
  CURRENT.json
  snapshots/snap-000006/
    manifest.json
    artifacts/aa/<cache-key>.bin
    ...
```

`manifest.json` 包含 `global_version`、完整配置哈希、交换版本、交换指纹、旁路哈希、源图指纹、已完成区域、pending 队列、区域阶段记录、内容指纹、情景状态和注入/失败信息。写入流程是：`.work` 内逐文件临时写 + 原子替换；写完清单后将 `.work` 整体重命名为快照；最后原子替换 `CURRENT.json`。

