# 地形 LOD 诊断工作台（terrain-workbench）

面向地形与地理空间渲染的本地确定性诊断工作台。输入为程序化生成的 uint16
高度场（257×257 / 1025×1025 / 4097×4097 三档，固定种子），围绕经典 LOD
取舍展开：四叉树分块、块内网格密度、法线求法与地形着色，并在给定三角形
与显存预算下给出可解释、可复现的降级路径。全部数据本地生成，不依赖外部
地形瓦片或网络服务。

## 环境与运行

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
# 依赖固定版本：numpy==1.26.4 PyOpenGL==3.1.7 PyYAML==6.0.1 pytest==8.3.3

.venv/bin/python -m terrain_workbench.generate --out data   # 生成三档样本+金字塔
.venv/bin/python -m terrain_workbench.demo --size 4097 --out out/run1
.venv/bin/python -m pytest tests -q                          # 41 项测试
```

离屏 GL：优先尝试 EGL surfaceless 上下文；无 GL 驱动时自动切换到软件路径，
纹理上传以相同的字节记账与内容哈希完成，并在每帧报告中以 `E_GL_CONTEXT`
标注（非致命，符合规范要求）。

## 1. 高度场与金字塔

- 噪声模型：PCG64 固定种子值噪声（4→2^n 倍频，双线性上采样，振幅 1/2^k
  衰减）+ 同种子高斯丘。同一 (size, seed) 两次生成逐字节一致。
- 默认起伏按 8×size 高程单位随规模缩放，三档坡度在世界单位下可比。
- 降采样滤波器（显式声明）：**3×3 tent 滤波、步长 2、整数算术 round-half-up、
  边界复制**（`pyramid.FILTER_NAME`）。粗级顶点与细级网格点对齐（2^n+1 布局）。
- 每级记录 `max_dev_vs_prev`（双线性重建到上一级的最大绝对偏差）与
  `cumulative_dev`（对 level0 的累积偏差上界；双线性上采样是凸算子，
  |up(a)−up(b)| ≤ max|a−b|，故累积和是 level0 的严格逐点上界）。
- 序列化：`data/terrain_<size>_L<k>.u16` 原始字节 + `*_meta.json`（滤波器、
  偏差、每级 SHA-256）。两次生成各级内容哈希逐字节一致（已验证）。

## 2. 四叉树分块与 LOD 选择

- 叶块 32×32 单元；深度 d 的块覆盖 `32·2^(max_depth−d)` 单元；块内密度
  g∈[0,5]，顶点数 `2^(5−g)+1`。块 (d,g) 的采样步长为 `2^(span+g)`，
  span = max_depth−d。
- 几何误差上界（高程单位）：
  `geom_bound = min(cum_dev[span+g], block_range/2)`，block_range 为块内
  高程极差（自底向上预计算）。两项均为严格上界。
- 屏幕空间误差上界：`rho = geom_bound · HEIGHT_SCALE · K / dist_min`，
  `K = viewport_h / (2·tan(fov_y/2))`，dist_min 为相机到块 AABB 的最近距离。
- 选择规则：固定顺序自顶向下，取满足 `rho ≤ tau` 的最粗 (d,g)；最细密度
  仍超阈值则分裂。同一相机参数两次运行选择清单逐字节相同（清单含每块
  误差、距离、三角形数与着色参数，见 `manifest.json`）。

## 3. 块间无缝拼接（方案声明）

**边界吸附到较粗邻居（edge snapping to the coarser neighbor）**。每条共享边
的有效步长取两侧较大者 `s_edge = max(sA, sB)`；非角点边界顶点若全局位置不
是 `s_edge` 的整数倍，则吸附到以 `s_edge` 整数倍位置处高度场值构造的分段
线性折线上；角点始终取高度场精确值。两侧用同一全局坐标、同一高度场、同一
整数步长求值，因此共享边界**顶点逐位一致**，对相邻级别差 1~3（及更大）
均无裂缝、无 T 型结点；块内部在 XY 上互不相交，无重叠三角形。
缝隙检测（`mesh.check_cracks`）逐边在两侧顶点位置的并集上采样双方边界
折线并比对（容差 1e-9），违规即抛 `E_CRACK_GAP` 并附首个违规块坐标。
测试含差 1/2/3 级的构造用例与"未吸附必开裂"的对照用例。

## 4. 法线与着色一致性

- 所有顶点法线（含接缝）由**同一 level-0 高度场、同一中心差分管线**求值
  （`normals.normal_at` / `normals_grid`），接缝两侧共享顶点法线逐位一致，
  不存在逐块独立计算导致的光照断裂。
- 着色为高程分段色带 + 坡度覆盖（`shading.ShadingParams`），参数随每帧块
  清单持久化（manifest 的 `shading` 字段），可对历史帧复算。

## 5. 预算决策与降级路径

- 三角形预算：超支时按固定优先级（屏幕误差升序 → 距离降序 → 深度降序 →
  块坐标）逐块逐档调粗（g+1），每轮每块最多一档，直至满足预算。每次降级
  记录块坐标、前后密度、原因（`triangle_budget`）与前后误差上界。
- 降级只改变密度，**从不丢弃可见块**；全部块到最粗仍超支则抛
  `E_BUDGET_DENIED`（附优先级序列首个块坐标与最小可达三角形数）。
- 决策序列完全确定：同一输入两次运行降级记录逐字节相同。

## 6. 显存驻留与纹理搬入

- 每块驻留字节 = 顶点数×(2 字节高程 + 3 字节法线)；`ResidencyManager` 按
  LRU 淘汰，逐事件（upload/evict/hit）记录台账（序号、帧、字节、驻留总额、
  内容哈希），`audit()` 可重放核对。
- 淘汰后再次可见时重新搬入；瓦片内容由 (高度场, 块键) 确定，重传哈希与首次
  一致（测试覆盖）。单块超过显存预算抛 `E_TEXTURE_RESIDENT`。

## 7. 逐帧诊断报告

每帧输出：可见块数、各级别块数直方图（`d<深度>g<密度>`）、最大屏幕误差、
三角形/显存预算使用率、降级次数、淘汰次数、缝隙检测边数与最大缝隙、GL 状
态、清单哈希，并与上一帧比较差值（delta）。报告为规范化 JSONL（键排序、
浮点定精度），逐帧 SHA-256 链式汇总为 `run_hash`；固定相机轨迹下两次运行
报告逐字节一致。

## 8. 业务错误码

| 代码 | 含义 | 触发场景 |
|---|---|---|
| `E_HEIGHT_RANGE` | 非法高程范围/尺寸 | hmin≥hmax、超出 uint16、尺寸非 2^n+1 |
| `E_LOD_LEVEL` | LOD/密度/深度越界 | g∉[0,5]、块坐标越界、叶块求子块 |
| `E_CRACK_GAP` | 检测到边界缝隙 | 逐边采样比对超容差，附首个违规块坐标 |
| `E_BUDGET_DENIED` | 预算不可满足 | 全部最粗仍超三角形预算 |
| `E_TEXTURE_RESIDENT` | 纹理驻留超限 | 单块大于显存预算、台账核对失败、哈希漂移 |
| `E_TRACK_DIVERGE` | 相机轨迹越界 | 关键帧/采样帧越出地形域或非有限值 |
| `E_GL_CONTEXT` | 无 GL 上下文 | 非致命，标注于每帧报告并走软件路径 |

所有异常路径抛 `TerrainError`（携带业务码与结构化上下文），不崩溃、不静默
跳过。

## 验证结果（本机实测）

- `pytest`：41 项全部通过（确定性、金字塔、LOD、接缝 1~3 级差、法线、
  着色、预算、驻留、逐帧报告、六类业务错误码）。
- 数据再生成：三档金字塔全部文件 SHA-256 逐字节一致。
- 固定轨迹双运行（96/96/48 帧）：三档 `report.jsonl` / `summary.json` /
  `manifest.json` / `ledger.json` 全部逐字节一致。

| 档位 | 帧数 | 可见块/帧 | 三角形/帧 | 降级次数 | 淘汰次数 | 检测边数 | 最大缝隙 |
|---|---|---|---|---|---|---|---|
| 257 | 96 | 11..26 | 16,384..39,424 | 0 | 0 | 2,894 | 0.0 |
| 1025 | 96 | 36..69 | 68,608..109,568 | 0 | 0 | 10,050 | 0.0 |
| 4097 | 48 | 64..106 | 106,496..119,808 | 841 | 435 | 8,152 | 0.0 |

（预算样例 `configs/budget.yaml`：三角形 120,000/帧、显存 4 MiB、tau=4px；
257/1025 档在该预算下无需降级，4097 档完整触发降级与 LRU 淘汰路径。）

## 目录结构

```
terrain_workbench/   核心包（heightfield/pyramid/quadtree/selection/mesh/
                     normals/shading/budget/residency/camera/diagnostics/
                     pipeline/gl_backend/errors）
configs/             相机轨迹与预算参数样例（演示入口输入）
tests/               41 项 pytest 测试
data/                三档高度场 + 金字塔序列化输出（generate 生成）
out/                 演示运行输出（report.jsonl/summary/manifest/ledger）
```
