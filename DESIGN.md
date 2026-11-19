# 本地点云离屏重建渲染引擎 — 技术说明

纯 Python 标准库 + NumPy 实现，无 GPU、无渲染库、无外部资源。把带灰度与法线的
稀疏三维点按针孔相机投影为二维灰度图与深度图，支持分块调度、块级持久化、摘要
失效判别、中断恢复与数据驱动的异常注入。

## 1. 目录结构

```
pointcloud_engine/
  errors.py       业务错误码 E_xxx 与 EngineError（结构化 JSON 错误）
  io_utils.py     原子写（临时文件 + os.replace）、规范化 JSON、数组/文件摘要
  camera.py       针孔相机：视图基矩阵、透视投影、屏幕空间点半径
  pointcloud.py   PTS3 文本格式解析/校验、确定性样本生成
  scene.py        场景加载、全字段校验、异常注入、规范化版本标识
  render.py       投影预处理、视锥/近平面剔除、逐采样点遮挡合成、串行基准、分块渲染、降采样
  state.py        块持久化、块内容摘要、版本/像素数校验、旧版本归档
  scheduler.py    有界线程池、索引序准入、内存上限、块边界中断、异常取消链路
  image.py        行序装配、PGM/PFM/npy 原子输出、同版本保护
  engine.py       顶层编排：恢复 -> 调度 -> 全块校验回退 -> 装配 -> 报告
  cli.py          命令行：gen-sample / render
tests/            unittest 套件（31 个用例，分四组）
acceptance.py     一键端到端验收脚本（8 类场景）
```

## 2. 相机与投影（业务要求 2）

* 右手世界坐标；相机局部采用 OpenGL 约定，视线沿局部 **-Z**，可见点相机空间 z<0。
  基矩阵三行为 `[right, up_cam, -forward]`，深度取 `depth = -z_cam`（世界单位）。
* `up` 与视线平行（或反平行）时点积绝对值接近 1，抛 `E_CAMERA`。
* 透视 NDC：`ndc_x = x_cam / (tan(hfov/2) * depth)`，`ndc_y` 同理；再线性映射到
  超采样网格坐标。垂直视场角必须在开区间 (0,180)，否则 `E_FOV`。
* 点半径按各点深度独立投影：`r_px = atan(r_world / depth) / 每半格角`，
  形成各向同性圆盘；枚举圆盘外接矩形内、满足 `dx²+dy² <= r_px²` 的采样点。
* 视锥外（圆盘与网格无交集）或近平面后（depth <= near）的点剔除，并在报告中
  分别计数 `points_culled_frustum` 与 `points_behind_near`。

## 3. 深度遮挡与衰减（业务要求 3）

* 每个超采样点保存最近深度与对应灰度。点按点云文件顺序访问，仅当新点深度
  **严格更小**才覆盖；等深保留先到点，保证与串行逐点结果一致。
* 深度与衰减统一在世界坐标单位下计算。衰减使用**归一化深度**
  `d / reference_distance`（reference_distance = 相机到目标点距离），因此对世界
  做均匀缩放时：遮挡顺序不变、各点衰减灰度的相对排序不翻转（有单测固化）。
* 支持 `exp` 与 `linear` 两种衰减，输出灰度裁剪到 [0,1]。

## 4. 确定性分块与像素等价（业务要求 4）

* 超采样网格按 `tile_width/tile_height` 等分，必须整除，否则 `E_TILE`。
* 每块只读共享的点云/相机/参数，写块私有缓冲，块间无共享可写状态。
* 串行基准与并行调度调用同一个 `_splat_into_tile`，区别仅是作用域（整网格 vs
  单块）与候选点集合，因此降采样结果逐像素相同，与线程完成顺序无关
  （`test_parallel_matches_serial_pixel_for_pixel`、并发 1/2/8 对比）。
* 降采样对每个输出像素的 `s×s` 超采样做盒平均。

## 5. 持久化、版本与恢复（业务要求 5、6）

两层摘要，职责分离：

* **版本标识 version_id**：覆盖相机、尺度/偏移、分辨率/超采样、半径、衰减、分块
  几何，以及点云结构骨架（相对源名、点数、数组形状）。配置或结构变化 → 版本变化
  → 旧块整体移动到 `state/archive/<前缀>/` 后从零重渲，绝不原地覆盖。
* **块内容摘要 input_digest + block_digest**：
  - input_digest 覆盖该块足迹覆盖到的点（按文件序）的世界坐标、灰度、投影中心、
    屏幕半径、深度与块几何。局部改一个点只会改变覆盖其圆盘的块，其余块复用。
  - block_digest 覆盖落盘的灰度/深度缓冲、声明命中像素数与 input_digest。
* 落盘统一 `np.savez -> .tmp.npz -> os.replace`，崩溃只可能留下完整旧文件或完整
  新文件；启动时清理两类临时文件，临时文件不会被当作完成块。
* 恢复时逐块 `verify_record`：版本、input_digest、形状、**声明像素数（直接对深度
  缓冲计数，独立于 block_digest）**、block_digest 全部通过才复用；否则归档补渲。
  调度完成后再做一次全块校验，仍失败则回退重渲，重渲后还不过则 `E_DIGEST`。
* 报告给出 `tiles_invalidated / tiles_rerendered / tiles_reused / archived_files`。

## 6. 中断、强杀与原子写出（业务要求 7）

* 安装 SIGINT/SIGTERM 处理器，仅在分块边界生效：停止准入新块，在途块通过内部
  `InterruptCancelled` 协作结束，写 `state/interrupted.json`，退出码 **130**。
* 未完成块不会产生可被误认的结果（只有完整 `os.replace` 后的 `.npz` 才算数）。
* SIGKILL 后无中断标记、无退出报告承诺；重启仅依赖已持久化且摘要有效的块恢复。
* 全部块通过后按行序拼装，灰度写 16 位 PGM、深度写 32 位小端 PFM，并写精确
  float32 的 npy 与 `outputs.json`（尺寸 + 摘要 + 版本）。均先写同目录临时文件再
  整体替换；同版本不重复替换，灰度与深度始终同版本。

## 7. 并发资源与异常链路（业务要求 8）

* 内存上限按 `每块字节 = tile_w * tile_h * 8`（float32 灰度 + float32 深度）折算
  可同时在途块数 `min(concurrency, budget // per_tile)`。
* 准入严格按分块索引序：预算不足时后续块等待，绝不跳块、绝不降低超采样倍数。
* 任一块抛 `EngineError`：取消其余待处理/在途块，返回首个错误，报告记录
  `failed_tile` 与 `success_count_at_failure`，且不产生输出图像。
* 异常全部由场景 `fault_injections` 数据驱动，不改源码：
  `zero_radius→E_RADIUS`、`non_finite_normal→E_FINITE`、
  `out_of_bounds_tile→E_TILE`、`tile_runtime_error→E_TILE_RUNTIME`、
  未知类型→`E_INJECT`。

## 8. 错误码一览

| 码 | 触发条件 |
|----|----------|
| E_SCENE / E_SCHEMA | 场景文件缺失/非法 JSON / 字段结构错误 |
| E_POINTCLOUD | 点云文件缺失、格式或行数不符 |
| E_FINITE / E_GRAY / E_NORMAL | 非有限数 / 灰度越界 / 零法线 |
| E_SCALE / E_FOV | 尺度非正有限 / 视场角不在 (0,180) |
| E_CAMERA | 上下方向与视线平行或相机退化 |
| E_DIM / E_TILE | 分辨率非正整数 / 分块未完整覆盖 |
| E_RADIUS / E_DECAY | 半径非法 / 衰减参数非法 |
| E_MEMORY | 预算装不下哪怕一个块缓冲 |
| E_INJECT | 未知或畸形异常注入 |
| E_DIGEST | 块摘要/声明像素数校验失败且无法修复 |
| E_TILE_RUNTIME | 注入的块级运行时失败 |
| E_IO | 原子写出失败 |

校验失败在任何输出文件创建之前抛出，因此非法输入不会产生输出文件。

## 9. 运行方式

```bash
# 生成确定性样本与场景
python3 -m pointcloud_engine.cli gen-sample workspace/demo
# 渲染（可重复执行验证复用；SIGTERM 后再执行即恢复）
python3 -m pointcloud_engine.cli render workspace/demo/scene.json
# 一键端到端验收
python3 acceptance.py
# 单元测试
python3 -m unittest discover -s tests -v
```

Windows 终端同样适用；中断可在任务管理器/Ctrl+C（SIGINT）或
`taskkill`（可处理终止）验证，强制结束对应 SIGKILL 语义。

## 10. 验证结果

`python3 -m unittest discover -s tests`：31 个用例全部通过，覆盖校验错误码、
投影/剔除、遮挡、尺度不变性、串并行像素等价、版本归档、局部失效判别、摘要/像素
数防篡改、SIGTERM 中断恢复、SIGKILL 恢复、内存上限与异常注入。
`python3 acceptance.py`：8 类端到端场景全部通过（实际输出见验收运行记录）。
