# 技术说明：CAD 几何与曲面显示编辑应用

## 总览

纯 Python 实现（Python 3.11+，NumPy + PyOpenGL，版本固定于 `requirements.txt`）。
所有样本由 `cadapp/samples.py` 程序化确定生成（闭式 sin/cos 控制网格），不依赖外部
CAD 文件与网络。演示入口：`python -m cadapp.demo [输出目录]`，测试：`pytest tests`。

## 模块结构

- `cadapp/geometry.py` — NURBS 求值（Piegl A2.1/A2.2 基函数 + 一阶导数）、曲面细分、
  等参线采样。着色网格顶点、顶点法线、等参线采样点全部由同一求值管线产生。
- `cadapp/model.py` — 数据模型、字段校验、规范化 JSON 字节与 SHA-256 内容哈希。
- `cadapp/jsonio.py` — 自有 JSON 曲面描述格式读写（读入时自动迁移旧版本）。
- `cadapp/migrate.py` — schema v1→v2 固定迁移表。
- `cadapp/binio.py` — 二进制网格交换格式（BMESH）读写与版本间双向转换。
- `cadapp/display.py` — 显示模式管理（三模式共享逐面片几何缓存）与确定性软件光栅化器。
- `cadapp/editor.py` — 控制点编辑与快照式撤销栈。
- `cadapp/compare.py` — 版本差异计算与来源归因。
- `cadapp/demo.py` — 固定演示序列，输出哈希与几何偏差报告。

## JSON 曲面描述格式（schema v2）

```json
{"schema_version": 2, "model_id": "...",
 "patches": [{"id": "...", "degree_u": 3, "degree_v": 3,
              "knots_u": [...], "knots_v": [...],
              "control_points": [[[x,y,z],...],...],
              "weights": [[...],...],
              "material_channel": "default", "tangent_space": false}]}
```

规范化序列化（键排序、紧凑分隔符、Python float 最短往返表示）保证同一模型的字节
与哈希确定。控制点全程以 float64 保存，JSON 往返逐值相等（测试断言 `==`）。

## 版本迁移表（v1 → v2）

| 字段 | 规则 | 几何影响 |
|---|---|---|
| `degree_u`/`degree_v` | 由 `len(knots) - len(control_points) - 1` 精确推断 | 无（推断值与原几何严格一致） |
| `material_channel` | 常量默认 `"default"` | 无（仅显示属性） |
| `tangent_space` | 常量默认 `false` | 无（仅显示属性） |
| `schema_version` | 置为 2 | — |

迁移幂等：对当前版本文档再次迁移返回零动作、字节与哈希不变。几何保持性由测试
验证：迁移后模型与直接构造的 v2 参考模型在细分顶点与等参线采样上偏差 ≤ 1e-9
（实际为 0）。

## 二进制网格交换格式（BMESH）

头部 16 字节：magic `"BMSH"`、格式版本（1/2）、字节序标记（0x01 小端 / 0x02 大端）、
索引宽度（16/32 位）、标志位（bit0 法线存在、bit1 法线交错布局）、顶点数、面片数
（后两者按声明字节序）。载荷：顶点 float64×3，法线 float64×3（平面或交错布局），
面片索引 uint16/uint32。可选尾块 `"JSRC"` + 长度 + 源模型规范化 JSON，使二进制 →
JSON 的往返能逐值恢复控制点。

版本差异：v1 固定为小端 / 16 位索引 / 平面布局含法线；v2 允许任意组合。
`convert_mesh` 在版本间双向转换：仅重写容器，float64 载荷不变，转换逐位无损；
顶点数超过 65536 时拒绝写入 16 位索引（`E_INDEX_OVERFLOW`），不做静默有损降级。
解析输出解码摘要（版本、字节序、索引宽度、布局、计数、包围盒、载荷 SHA-256）供比对。

## 显示与编辑

- 三种显示模式（控制网格 / 着色曲面 / 等参线）共享 `DisplayManager` 的逐面片几何
  缓存，模式切换只是视图选择，重建计数不变（测试断言）。
- 控制点编辑仅失效对应面片缓存；未受影响面片的法线数组逐位保持不变，避免整模型
  重算引起的显示抖动。
- 撤销栈保存编辑前控制点快照；撤销后模型哈希与显示几何完全复原；空栈撤销返回
  `E_UNDO_EMPTY`。
- 渲染：为保证两次运行逐字节一致，演示始终走确定性纯软件光栅化器（PPM 输出），
  并记录环境标志 `CADAPP_RENDERER=software`；OpenGL 离屏上下文可用性通过真实创建
  EGL 上下文探测，作为环境事实记录于报告（Windows 上同理验证本地驱动）。

## 版本对比与归因

模型级对比 `compare_models(a, b, op_log)` 逐面片列出控制点差异、字段差异、schema
版本差异；文档级对比 `compare_documents(doc_a, doc_b, op_log)` 递归 diff 原始 JSON
（用于迁移前后对照）。两者均依据操作日志归因：`edit`（编辑，匹配面片/索引/新值）、
`migration`（迁移动作，匹配路径与默认值）、`format_conversion`（转换声明的效果）、
无法归因者显式标记 `suspicious`，绝不丢弃。归因均校验新值与操作记录一致，避免
把无关改动误记到迁移或编辑名下。

## 业务错误码

| 码 | 触发场景 |
|---|---|
| `E_SCHEMA_VERSION` | schema 版本缺失/非法/高于当前支持版本 |
| `E_INDEX_OVERFLOW` | 面片索引 ≥ 顶点数（读/写两侧），或顶点数超出 16 位索引范围 |
| `E_NORMAL_COUNT` | 法线数量与顶点数不一致、法线块截断、v1/交错布局缺法线 |
| `E_KNOT_MONOTONIC` | 节点向量非单调不减（含定位下标） |
| `E_UNDO_EMPTY` | 空撤销栈上执行撤销 |
| `E_FORMAT_ENDIAN` | 字节序标记无法识别 |

另有格式级辅助码 `E_FORMAT_MAGIC` / `E_FORMAT_HEADER` / `E_FORMAT_TRUNCATED`。
所有错误携带定位信息（如 `patches[0](patch_wave).knots_u[2]`、
`mesh.faces[3][1]`），异常链路不产生部分写入。

## 实际验证结果（本机，Linux 容器）

- `pytest tests`：30 项全部通过（JSON/二进制往返、迁移幂等与几何保持、六种错误码、
  编辑撤销复原、模式切换零重建、对比归因、演示确定性）。
- `python -m cadapp.demo artifacts` 两次运行：stdout、`demo_report.json`、三幅 PPM
  渲染图均逐字节相同（`cmp` 验证）。
- 关键指标（见 `artifacts/demo_report.json`）：
  - JSON 往返：控制点逐值相等，哈希一致；
  - 四种二进制容器配置（v2 小端 32 位、v2 大端 32 位、v1 小端 16 位、v2 大端交错）
    往返逐位一致，最大顶点偏差 0.0；
  - 迁移：幂等，迁移后哈希等于 v2 参考模型哈希，几何最大偏差 0.0；
  - 编辑/撤销：编辑仅重建 1 个面片，模式切换重建 0 次，未受影响面片法线逐位不变，
    撤销后哈希复原；等参线与着色网格同参数坐标差 0.0（≤ 容差 1e-9）；
  - 对比归因：edit / migration / suspicious 三类来源均被正确标注；
  - 环境：`CADAPP_RENDERER=software`，EGL 离屏上下文可创建（Windows 目标平台上
    以同样方式探测本地驱动，不可用时自动落回纯软件路径）。
