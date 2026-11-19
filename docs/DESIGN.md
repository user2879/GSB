# 大规模地形遮挡剔除与流式加载组件 — 技术说明

## 1. 范围与运行环境

纯 Python 参考实现，核心数值全部在 IEEE-754 `float64` 下完成，并对“等价 GPU
float32 链路”的舍入误差显式记账。离屏 GL 仅做能力探测（EGL/GLES2 pbuffer），
不参与可复现的数值路径。

固定版本（见 `requirements.txt` 与 `artifacts/environment.json`）：

- Python 3.11（要求 >= 3.10）
- numpy 1.26.4
- PyOpenGL 3.1.7
- pytest 8.3.4
- 离屏上下文：EGL 1.5 / GLES2 pbuffer，探测结果 `available=true`

## 2. 模块结构

| 模块 | 职责 |
| --- | --- |
| `terrain_culling/errors.py` | 7 个业务错误码与类型化异常链 |
| `terrain_culling/config.py` | 固定种子、数值常量、预算与降级配置 |
| `terrain_culling/terrain.py` | 确定性 fBm 高度场、材质分区、顺序无关四叉树清单与落盘 |
| `terrain_culling/camera.py` | float64 相机、固定顺序矩阵、Catmull-Rom 固定轨迹、投影误差界 |
| `terrain_culling/depth.py` | 软件确定性深度射线步进、保守 max-depth Hi-Z、误差表、真值 oracle |
| `terrain_culling/culling.py` | 保守 AABB 外扩、视锥判定、表面点投影足迹、Hi-Z 深度测试、误差合成 |
| `terrain_culling/streaming.py` | 驻留账本、字节预算、先加载后驱逐的流式调度 |
| `terrain_culling/budget.py` | 误差预算汇总、固定优先级降级阶梯 |
| `terrain_culling/engine.py` | 每帧管线编排 |
| `terrain_culling/diagnostics.py` | 规范化哈希、逐帧摘要、首个分歧定位 |
| `terrain_culling/gl_context.py` | 离屏 EGL 上下文探测 |
| `terrain_culling/demo.py` | 一键演示入口 |

## 3. 确定性与逐字节一致

- 随机源全部为固定种子的 `numpy.random.PCG64`（高度 `20261119`、材质
  `30261119`、轨迹 `40261119`），不使用墙钟时间。
- 四叉树块由显式坐标嵌套循环构建，再按 `block_id` 排序；任何记录都不依赖
  dict/set 遍历顺序。
- 数组以裸 little-endian 字节落盘（无 npz/zip 时间戳），清单与报告使用
  `json.dumps(sort_keys=True, separators=(",",":"))` 规范化，非有限值统一转
  `null`；输出文件内只写 basename，哈希与输出目录无关。
- 两次完整运行的 `frames_*.json`、`summary_*.json`、`manifest_*.json`、
  高度/材质二进制、`run_manifest.json` 均 `cmp` 逐字节相同。

## 4. 深度与 Hi-Z（需求 2）

- 深度为相机空间线性正深度（前向轴点积），射线按“不超过一个水平网格 cell”
  自适应定步长，并以固定步数 + 段端点地面跨越检测，避免大步长穿山；交点用
  段内线性插值定位。
- NDC 行轴方向显式取顶部 `+1` 到底部 `-1`，避免射线上下翻转。
- Hi-Z 每级对 2x2 邻域取 **max depth**：存得更远只会少剔除、不会错误剔除。
- 误差表每级记录分辨率、世界像素尺寸、采样足迹、深度误差上界与像素半角，
  字段可由配置独立复算。

## 5. 误差传播与保守判定（需求 3、4）

每个块的判定误差上界由四个显式分量合成：

1. 投影/矩阵 float32 等价误差 `matrix_abs_error + point_abs_error`，按
   `n*eps*(距离+半径)` 一阶模型估计（`camera.projection_error_bounds`）。
2. AABB 保守外扩 `expansion_margin*sqrt(3)`，基线 margin 由
   `F32_REL_EPS * far` 导出。
3. Hi-Z 级别深度误差 `hiz_depth_error`。
4. 像素角走样 `angular_aliasing_error = 最近深度 * tan(像素半角)`，覆盖
   “掠射块与近遮挡物共享同一低分辨率像素”的情形。

遮挡判定以块**最近表面点**对最近遮挡物深度做单门限测试：

- `nearest - hiz <= -total`：确在遮挡物之前 → visible
- `nearest - hiz >= +total`：最近点都在误差带之外的后方 → occluded
- 其余：落入误差带 → uncertain，保守保留并记录原因与分量

视锥判定基于块顶表面采样点（含 AABB 角点），解决“角点全在屏外但表面伸入
视口”的漏判；Hi-Z 只读取表面点实际覆盖像素（一像素保守膨胀），不读整个
AABB 外接矩形，避免纳入无关近山像素。

屏幕空间投影误差超预算时记录 `screen_error_over_budget` 并参与降级触发；
降级决策只依赖可复现的预算利用率与超预算计数。

## 6. 流式调度与账本（需求 5）

- 每块记录 `payload_bytes`（u16 高度 + u8 材质）与 `vram_bytes`（顶点/索引
  预留），账本逐帧可重算并 `verify()`。
- 帧内先加载后驱逐；预算不足时先按 LRU 回收“非期望且非保护”块，再加载，
  可见块始终在保护集合内，绝不因驱逐而缺失。
- 重新加载恢复完全相同的字节占用；专门测试覆盖“卸载→重载”与从未卸载一致。

## 7. 误差预算与降级阶梯（需求 6）

固定优先级、单调非减、同输入可复现：

| step | 动作 |
| --- | --- |
| 0 | 基线 |
| 1 | 降低 Hi-Z 级别（更粗、更保守） |
| 2 | 加倍保守外扩 |
| 3 | 关闭预加载环，仅保留最小可见驻留集 |

触发条件：内存/显存利用率超阈值，或单帧超预算判定数达阈值。任何降级都不
允许丢弃可见块；预算连最小可见集都无法满足时抛 `E_BUDGET_DENIED`，且不产生
部分驻留泄漏。

## 8. 诊断与首个分歧（需求 7）

每帧输出可见/剔除/不确定/视锥外计数、相对误差直方图与预算余量、内存与显存
利用率、加载/保留/驱逐统计、Hi-Z 级别与降级步骤。
`find_first_divergence` 比较的是逐块决策签名（状态+误差+级别+原因），即使
两块数相同也能定位首个不一致帧与块，并回带该帧相机参数与误差分量。

## 9. 业务错误码（需求 8）

| 码 | 触发链路 |
| --- | --- |
| `E_HEIGHT_RANGE` | 非 u16、非方阵、不支持尺寸、材质形状不符 |
| `E_BLOCK_MISSING` | 访问清单中不存在的块 |
| `E_HIZ_LEVEL` | Hi-Z 级别越界 |
| `E_DEPTH_BIAS` | 深度偏置为负或超过 `max_depth_bias` |
| `E_BUDGET_DENIED` | 预算无法容纳必要加载 |
| `E_RESIDENT_LEAK` | 账本重算不符、超预算、驱逐非驻留块 |
| `E_TRACK_DIVERGE` | 两轨迹逐块决策出现首个分歧 |

## 10. 验证结果

- `pytest tests -q`：36 项全部通过（含独立真值 oracle 的“零错误剔除”不变量、
  字节可复现、降级可复现、全部 7 个错误码链路）。
- 演示：`python -m terrain_culling.demo --out <dir> --frames 16` 同时生成
  512 与 2048 两档；两次运行所有产物 `cmp` 逐字节相同，
  `run_manifest.json` 亦一致。
- 保守性：对每一帧每个叶块，独立细步进 oracle 判可见的块，管线状态只可能是
  `visible`/`uncertain`，从不会是 `occluded`。

## 11. 复现步骤

```bash
pip install --user -r requirements.txt
python -m terrain_culling.demo --out artifacts/demo --frames 16
python -m pytest tests -q
# Windows: run_demo.bat ; POSIX: ./run_demo.sh
```
