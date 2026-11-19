# NPR 交换格式、迁移与持久化规范

## 1. 项目 JSON（schema v3）

canonical 文档包含：

```json
{
  "schema_version": 3,
  "project_id": "npr-demo-scene",
  "version_stamp": "000000000001",
  "shapes": [
    {
      "id": "tree_crown",
      "layer": "midground",
      "stroke_width": 2.4,
      "style_tags": ["leaf"],
      "contours": [
        {
          "closed": true,
          "segments": [
            {
              "type": "cubic",
              "points": [[150.0, 89.0], [150.0, 102.8], [136.9, 114.0], [119.0, 114.0]]
            }
          ]
        }
      ]
    }
  ],
  "style": {
    "line_color": [18, 18, 20],
    "paper_color": [247, 244, 236],
    "line_opacity": 1.0,
    "width_scale": 1.0,
    "sketch_ink": [72, 70, 68],
    "sketch_hatch_spacing": 7.0,
    "sketch_hatch_angle": 42.0,
    "sketch_shade_strength": 0.24,
    "toon_palette": {"leaf": [92, 160, 96]},
    "toon_default_color": [205, 199, 185],
    "toon_shadow_color": [112, 91, 104],
    "toon_light": [0.45, -0.8],
    "toon_light_strength": 0.22
  },
  "stroke_sample": {
    "curve_steps": 12,
    "line_steps": 1,
    "width_scale": 1.0,
    "min_segment_px": 0.0
  },
  "layer_order": ["background", "midground", "ink"],
  "metadata": {}
}
```

曲线段类型与控制点数：

| type | 控制点 |
| --- | ---: |
| `line` | 2 |
| `quadratic` | 3 |
| `cubic` | 4 |

所有轮廓控制点在载入时量化为 IEEE-754 binary32，再以能精确回读的 JSON 数字输出。JSON canonical 哈希为：

```text
sha256(json.dumps(project_dict, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
```

## 2. schema 迁移表

| 源版本 | 缺失字段 | 固定动作 |
| --- | --- | --- |
| v1 | `style` | 写入 v3 的默认线稿风格 |
| v1 | `stroke_sample` | 写入 `curve_steps=12, line_steps=1, width_scale=1, min_segment_px=0` |
| v1/v2 | `layer_order` | 按图元首次出现顺序派生 |
| v2 | `stroke_sample` | 写入同一 v3 默认值 |
| v3 | 无 | 不改动；重复迁移哈希不变 |

默认值对应旧版固定扁平化器，因此同一场景迁移前后的 lineart 像素与 PNG 哈希一致。若版本不在 `1..3`，或旧文档无法派生必要字段，则抛出 `E_SCHEMA_VERSION`，不会静默降级。

## 3. 二进制描边骨架 `NPBS`

### 3.1 固定头

| 偏移 | 长度 | 字段 | 说明 |
| ---: | ---: | --- | --- |
| 0 | 4 | magic | ASCII `NPBS` |
| 4 | 4 | endian marker | 大端 `01 02 03 04`；小端 `04 03 02 01` |
| 8 | 2 | format version | `1` 旧格式；`2` 新格式 |
| 10 | 1 | index width | `2` 或 `4` 字节 |
| 11 | 1 | width encoding | `0` uint16/256；`1` float32 |
| 12 | 1 | flags | bit0 = 内嵌项目 JSON metadata |
| 13 | 1 | reserved | 必须为 0 |
| 14 | 4 | vertex count | 全局顶点数 |
| 18 | 4 | index count | 全局索引数 |
| 22 | 4 | shape count | 图元数 |
| 26 | 4 | metadata length | UTF-8 JSON 字节数，可为 0 |
| 30 | N | metadata | canonical 项目 JSON |

所有多字节整数、浮点和索引均使用 marker 声明的字节序。未知 marker 抛 `E_FORMAT_ENDIAN`。

线宽编码与格式版本严格对应：

- v1 → encoding 0：`round(width * 256)` 的 uint16。
- v2 → encoding 1：IEEE-754 float32。

### 3.2 图元表

每个图元依次为：

```text
u16 id_len, id UTF-8
u16 layer_len, layer UTF-8
width (u16 或 f32)
u32 vertex_offset, u32 vertex_count
u32 index_offset, u32 index_count
u16 contour_count
contour_count × {
  u32 vertex_offset(相对图元), u32 vertex_count
  u32 index_offset(相对图元), u32 index_count
  u8 closed
}
```

随后是全局顶点缓冲 `float32 x,y` 和全局索引缓冲。索引是相对整个顶点缓冲的全局编号；闭合轮廓额外重复首顶点索引。解析器校验：

- 图元/轮廓 offset 连续且不越界；
- 顶点数、索引数与各层声明完全一致；
- 每个索引位于对应轮廓顶点范围；
- 闭合轮廓末索引等于首索引；
- 无尾随字节；
- float32 坐标有限、索引宽度有效。

结构或计数错误抛 `E_FORMAT_INDEX`。解析函数先构造成员局部对象，全部校验通过后才返回，失败不会向调用方暴露半加载几何。

### 3.3 往返语义

- 有 metadata 时，JSON→NPBS→JSON 保留项目内容哈希与原始曲线控制点。
- 无 metadata 时，骨架恢复为逐段 `line` 轮廓；扁平化后的顶点逐点完全相同，但 JSON 内容哈希会不同，因为曲线表示和风格标签不可逆。
- 两种情况下都输出 JSON 内容哈希与完整二进制哈希。

## 4. 渲染与缓存失效

`GeometryEngine` 以 `(图元几何, stroke_sample)` 签名为缓存键。`StrokeSkeleton` 由同一份扁平化顶点和轮廓范围生成。`lineart` 与 `toon` 都读取该共享几何；卡通填充直接使用参与描边的同一组扁平化顶点。

渲染缓存键包括：

- 图元 ID；
- 渲染模式；
- 骨架几何哈希与几何签名；
- 当前模式实际依赖的风格指纹；
- 有效线宽；
- 图元标签；
- 画布尺寸与超采样倍数。

因此：

- 切换 `lineart/sketch/toon` 不重建几何；
- 只改某图元线宽时，其他图元渲染哈希不变；
- 修改与当前模式无关的风格字段时，lineart 缓存仍命中。

## 5. 快照提交与恢复

目录布局：

```text
workspace/snapshots/
  .staging-<pid>-<uuid>/       # 未发布
  <stamp>-<hash-prefix>/
    project.json
    render.png
    shapes/<sha256(shape_id)>.rgba
    manifest.json
    COMMITTED                  # 内容为 manifest snapshot_hash
```

提交步骤：

1. 计算项目哈希、各图元几何哈希、渲染哈希与缓存键。
2. 在 `.staging-*` 写完所有 payload，逐文件 flush/fsync，并原子写 manifest。
3. 写入 `COMMITTED`。
4. 将 staging 目录原子改名为正式快照目录。

恢复步骤：

1. 删除 `.staging-*` 与缺少 `COMMITTED` 的目录。
2. 对已提交目录校验 COMMITTED、manifest canonical hash、全部文件 SHA-256。
3. 以 `version_stamp` 数字值选择最近提交；同版本戳不同内容抛 `E_SNAPSHOT_STALE`。
4. 恢复渲染缓存时再次校验项目哈希、几何哈希、模式、缓存键和原始 RGBA 哈希。

任何已提交但损坏的快照都会显式报错，不会静默回退或复用旧缓存。

## 6. 版本对比与归因

对比结果包含：

- 两侧 version stamp、项目哈希、几何哈希、lineart 像素/PNG 哈希；
- 逐图元控制点、结构、图层、线宽和标签差异；
- schema 与二进制格式版本差异；
- 迁移动作和格式转换动作；
- 每条差异的 `sources` 与 `suspicious` 标记；
- 并排 PNG 哈希和整体 comparison hash。

来源只有三种可审计值：`edit`、`migration`、`format_conversion`。没有 trace 支撑的变化以空 `sources` 和 `suspicious=true` 报告。对比目录同样采用 staging + `COMMITTED` 原子发布，`ComparisonStore.recompute` 从持久化项目、trace 和渲染配置重新计算并要求结果完全一致。

## 7. 业务错误码

| 错误码 | 触发场景 |
| --- | --- |
| `E_SCHEMA_VERSION` | 非法/越界 schema、无法补齐或非法项目字段 |
| `E_FORMAT_ENDIAN` | 未知二进制字节序 marker |
| `E_FORMAT_INDEX` | 索引越界、顶点/索引计数不符、轮廓范围错误 |
| `E_STROKE_SAMPLE` | 采样步数、缩放或最小段长非法 |
| `E_UNDO_EMPTY` | 空撤销栈 |
| `E_SNAPSHOT_STALE` | 旧版本戳、同戳异哈希、已提交快照损坏或缓存不匹配 |
| `E_RENDER_MISMATCH` | 渲染哈希、PNG 或恢复 RGBA 与声明不一致 |

另有结构辅助码 `E_FORMAT_VERSION`（不支持的二进制版本）和 `E_FORMAT_DATA`（非 UTF-8 metadata 等）。
