# 点云三维重建编辑应用 — 技术说明

## 1. 目标与约束

面向点云与三维重建显示的领域编辑应用。输入为本地确定性合成样本（固定种子、
120000 个带法线与置信度的点，立方体房间五面 + 地面共六个平面），支持点剔除、
区域框选、法线重估半径调整、重建参数调整四类编辑，视口离屏渲染点云与三角网格，
编辑后仅对受影响曲面片做增量重算，并保证：

- 交互帧在声明帧时预算 `33.3 ms`（30 FPS）内给出可见反馈，超预算按固定阶梯降级；
- 增量重建网格与“编辑后完整点集的全量重建”在声明容差内一致；
- 相同点云、参数、操作序列两次运行产出**逐字节相同**的报告与内容哈希。

不使用容器；不访问任何远程目标；样本与代码均由本仓库生成器首次运行产出。

## 2. 目录结构

```
src/pcrecon/
  config.py          固定版本化配置（种子、房间尺寸、预算、文件名）
  errors.py          8 个业务错误码（含 E_GL_CONTEXT）
  hashing.py         BLAKE2b 内容哈希、规范 JSON、原子写（tmp+fsync+replace）
  binary_format.py   确定性二进制点云/法线格式（magic/版本/校验和）
  generator.py       确定性样本/相机轨迹/编辑序列生成器
  spatial.py         体素栅格空间索引（确定序遍历）
  normals.py         局部法线重估、置信度、邻居数审计
  mesh.py            全局节点高度场 + 曲面片缓存的增量重建
  consistency.py     增量 vs 全量双口径一致性核对
  state.py           不可变版本化快照 + 内容哈希
  service.py         编辑应用、失效闭包、增量/全量重建、乐观并发
  worker.py          工作线程，非阻塞提交，整快照原子替换
  framing.py         帧时预算与三级降级（确定性时钟 / 墙钟两种模式）
  viewport.py        视口控制器（只读快照，不改编辑状态）
  render_gl.py       离屏 Qt/OpenGL FBO 渲染，失败返回 E_GL_CONTEXT
  ledger.py          原子账本、撤销、崩溃恢复、孤儿回收、退出报告
  application.py     端到端编排与报告产出
  cli.py             诊断入口：generate/check-gl/run/crash-demo/hash-verify
scripts/             GL 核对脚本、Windows/Linux 一键验收
tests/               17 个 unittest 用例
docs/                本文件
```

## 3. 确定性设计

1. **随机源**：单一 `numpy.random.default_rng(SEED)`（PCG64），调用顺序固定；
   面积加权用“最大余数法”分配各面点数，最后按 `(面id,x,y,z)` 稳定排序落盘。
2. **二进制格式**：小端 + IEEE-754，头部 `magic/版本/dtype/列数/条数`，
   载荷后附 BLAKE2b-256 校验；读取失败统一映射为 `E_POINT_CLOUD_FORMAT`。
3. **JSON**：排序键、固定分隔符、禁用 NaN、`-0.0` 归一化，路径只存文件名，
   因此与输出目录无关，跨机器逐字节一致。
4. **数值**：法线用 LAPACK `eigh`（确定序），朝向统一指向房间内部；
   节点高度场用 `np.add.at` 在固定全局网格上累加；曲面片按
   `(surface_id, tile_i, tile_j)` 字典序构建与合并。
5. **时钟**：验收路径使用确定性成本模型（仅由点数/面片数/交互类型决定），
   不读取墙钟；实时视口另有 `WallClock` 模式，且其耗时不进入可复现报告哈希。

## 4. 空间索引与法线重估

- `VoxelGrid` 以整数体素坐标建哈希栅格，邻域查询只扫描半径覆盖的相邻体素，
  返回索引恒为升序；非法半径抛 `E_NEIGHBOR_RADIUS`。
- 法线重估：协方差矩阵特征分解，最小特征值对应特征向量为法线，平面度/离散度
  组合得到 `[0,1]` 置信度，邻居少于 4 的退化邻域输出零法线/零置信度。
- 局部性：编辑区域内点为种子，取半径内所有存活点为目标集合，仅重算这些点；
  邻域之外的法线行与重估前**逐点相同**（有单测断言）。审计记录半径、邻居数
  min/max/mean、退化数与目标索引哈希。

## 5. 增量曲面重建与失效闭包

房间六个轴对齐平面各自拥有**全局 2D 节点网格**。每个存活点归到最近节点，
节点值 = 归属点沿平面内法向的平均偏移（空节点为 NaN）。曲面片（tile）是该
全局网格上的固定窗口：

- 一个 tile 的顶点只依赖其窗口及共享边界邻域内节点（支撑半径换算的 halo），
  全量重建与增量重建看到的支撑点集完全相同；
- 点被剔除时，`tiles_touching_points` 计算“受影响 tile + 边缘共享邻接 tile”
  的失效闭包；闭包内 tile 重算，闭包外 tile 直接复用缓存，其顶点哈希逐点不变；
- 法线重估不改几何，因此不使任何 tile 失效；重建参数变更使全部 tile 失效。

**失效闭包与实际重算集合完全一致**在每次编辑的增量报告中以
`closure_equals_rebuilt` 字段给出，并由测试断言。

## 6. 一致性核对（不靠面片数）

`consistency.compare_meshes` 在相同的规范 tile 顺序下逐顶点、逐面片比较：

- 顶点位置：最大欧氏误差、RMS、误差直方图、首个分歧顶点索引；
- 面片法线：叉积归一化后的最大夹角（弧度）、角度直方图、首个分歧面片；
- 同时断言面片索引数组逐元素相等。

声明容差 `POSITION_TOL = NORMAL_TOL = 1e-7`。实测顶点位置误差为 `0.0`；
缓存 tile 与全量重算 tile 的法线仅因全局求和顺序产生 ≤ `3.4e-8 rad` 舍入差，
落在容差内。超出容差抛 `E_MESH_MISMATCH`，上下文携带首个分歧顶点与触发的
`op_id`，不会因面片数相等而误判。

## 7. 帧时预算与降级

`FrameBudgeter` 每帧输出结构化记录，按固定顺序逐级降级直到满足预算：

1. 点云绘制密度 `100% -> 50% -> 25%`；
2. 延迟曲面片刷新（沿用上一版网格快照，成本 ×0.7）；
3. 跳帧更新（仅保留最小帧，成本 2 ms）。

所有档位仍超预算则抛 `E_BUDGET_EXCEEDED`。降级只改变绘制节奏，
**不修改已提交编辑与重建结果**（重建服务与预算器无共享可变状态）。

## 8. 并发与快照

- `Snapshot` 为冻结 dataclass，携带版本戳、点集、法线、参数、tile 字典、
  合并网格、tile 哈希与操作历史；`points_hash/normals_hash/mesh_hash/content_hash`
  均为 BLAKE2b。
- `WorkerReconstruction` 在独立线程基于快照产出新版本，提交为非阻塞队列；
  视口线程只读取当前不可变快照，发布时整体替换引用，不存在部分更新混合态。
- 编辑携带 `base_version`，与 head 不符即 `E_SNAPSHOT_STALE`，不会破坏 head。

## 9. 账本、撤销与崩溃恢复

- `ledger.log` 为追加式规范 JSONL，每条 fsync；`manifest.json` 经
  tmp+fsync+replace 原子更新 head 指针与每版哈希；版本目录写 `committed` 标记。
- 撤销弹出版本栈，复原目标版本的点集/法线/曲面片哈希，并写 undo 记录；
  栈空抛 `E_UNDO_EMPTY`。
- 启动时 `recover()` 回收所有 `cache/tmp-*` 未提交孤儿，只承认最近一次已提交
  manifest；`exit_report.json` 仅在正常退出时写入，强制终止路径不生成该文件，
  由 `crash-demo` 子命令验证“不伪造退出报告”。

## 10. 错误码

| 代码 | 触发位置 |
| --- | --- |
| `E_POINT_CLOUD_FORMAT` | 二进制 magic/版本/长度/校验和错误、点法线条数不符 |
| `E_NEIGHBOR_RADIUS` | 半径/体素尺寸非正或非有限、区域上下界颠倒、未知编辑类型 |
| `E_SNAPSHOT_STALE` | 编辑 base_version 与 head 版本不一致 |
| `E_MESH_MISMATCH` | 增量网格与全量重建双口径超容差，附首个分歧顶点/op_id |
| `E_BUDGET_EXCEEDED` | 所有降级档位后仍无法满足帧预算 |
| `E_UNDO_EMPTY` | 无已提交编辑可撤销 |
| `E_PATCH_DEPENDENCY` | `mesh_grid_step` 不能整除 `voxel_size` 等 tile 依赖错误 |
| `E_GL_CONTEXT` | 离屏 surface/context/FBO 创建失败（不崩溃，结构化返回） |

## 11. Windows 运行

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:QT_QPA_PLATFORM="offscreen"; $env:PYTHONPATH="src"
.\.venv\Scripts\python.exe scripts\check_gl_context.py   # 首次 GL 核对
.\.venv\Scripts\python.exe -m pcrecon generate
.\.venv\Scripts\python.exe -m pcrecon run --undo
.\.venv\Scripts\python.exe -m pcrecon hash-verify
```

或一键：`powershell -ExecutionPolicy Bypass -File scripts/acceptance.ps1`。
本机无显示时 Qt 使用 `offscreen` 平台插件；Windows 本机有驱动时同样适用，
FBO 失败会得到 `E_GL_CONTEXT` 而不是进程崩溃。
