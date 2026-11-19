# VolRender Cache

本地、离线、确定性的体数据离屏渲染与分块缓存失效验证工具。实现只依赖 Python 3.11+ 与 NumPy，不访问网络、不使用容器或外部数据集。

## 快速开始

```bash
python -m pip install -r requirements.txt
python -m volrender generate-samples --samples samples --seed 20261119
python -m volrender render samples/tasks/00_initial.json
python -m volrender render samples/tasks/01_region_edit.json
python -m volrender verify-samples --samples samples
python -m unittest -v tests.test_volrender
```

也可以安装为命令行程序：

```bash
python -m pip install .
volrender verify-samples --samples samples
```

运行成功时命令输出 JSON 摘要；业务错误输出带 `code`、`message`、`path` 的 JSON。退出码约定：`0` 成功，`2` 输入/任务校验失败，`1` 运行时/状态/摘要失败，`130` 在分块边界收到 SIGINT/SIGTERM。

## 交付内容

- `volrender/`：渲染、几何失效判定、缓存、调度、样本生成与 CLI。
- `tests/test_volrender.py`：10 组本地测试，覆盖校验、编辑顺序、失效规则、串并等价、缓存归档、摘要回退、中断恢复与错误取消。
- `samples/`：固定种子生成的体数据、5 个增量任务与一份 `verification.json`。
- `VERIFICATION.md`：本机实际验证结果。

## 数据与坐标约定

- `volume.dims` 为 `[nx, ny, nz]`；`.npy` 数组形状必须为 `[nz, ny, nx]`。
- 第 `i` 个体素中心位于世界坐标 `(i + 0.5) * spacing_mm`。体积包围盒为 `[0, nx*sx] × [0, ny*sy] × [0, nz*sz]`，单位为毫米。
- 相机使用右手视角基：`forward = normalize(target-position)`、`right = normalize(cross(forward, up))`、`true_up = cross(right, forward)`。
- 灰度图与深度图均为 `float64` 的 `[height, width]` NumPy 数组，严格同尺寸。
- 标量 `0` 定义为空体素，不参与传递函数合成；负值或正值均可由传递函数映射。
- 深度记录光线首个有效采样点（标量非零且传递函数不透明度大于 0）的相机到点距离。没有有效采样时使用确定值 `background_depth_mm`，即相机到体积包围盒最远角的距离。

## 任务 JSON

```json
{
  "schema_version": 1,
  "volume": {
    "path": "../volume.npy",
    "dims": [40, 36, 32],
    "spacing_mm": [0.9, 0.9, 1.1],
    "scalar_domain": {"min": 0.0, "max": 1.0}
  },
  "transfer_function": {
    "points": [
      {"scalar": 0.0, "opacity": 0.0, "gray": 0.0},
      {"scalar": 0.55, "opacity": 0.06, "gray": 0.62},
      {"scalar": 1.0, "opacity": 0.16, "gray": 1.0}
    ]
  },
  "camera": {
    "position": [55.0, 47.0, 53.0],
    "target": [18.0, 16.0, 15.5],
    "up": [0.0, 1.0, 0.0],
    "fov_y_deg": 58.0
  },
  "output": {"width": 48, "height": 48},
  "blocks": {"width": 8, "height": 8},
  "concurrency": 4,
  "memory_budget_bytes": 16777216,
  "sampling_step_mm": 0.45,
  "background_gray": 0.0,
  "edits": [
    {"center": [24, 18, 18], "size": [7, 5, 7], "value": 0.95}
  ],
  "state_dir": "../state",
  "outputs": {
    "gray": "../out/gray.npy",
    "depth": "../out/depth.npy",
    "report": "../out/report.json"
  }
}
```

编辑区域索引范围为半开区间：

```text
start = center - floor((size - 1) / 2)
stop  = start + size
```

所有编辑都从同一个基础体数据开始，按清单顺序应用；重叠区域由后声明者覆盖。相机 `aspect` 可省略，默认使用 `output.width / output.height`。传递函数点可携带额外字段（例如 `bandwidth`）；这些字段参与规范化摘要，因此任一字段变化都会触发传递函数失效规则。

## 失效判定

一次运行先读取上一个完整状态中的任务、体积和摘要，再重建上一版编辑后体数据，与当前编辑后体数据逐体素比较。

- 体积编辑：得到真实变化的体素掩码。每个分块先根据像素中心光线与体积包围盒的交段求体素索引包围范围；随后以精确的“光线段 vs 三线性插值支撑盒”测试判断光线是否真正依赖任一变化体素。只有命中的块标记 `volume_edit`。仅包围范围相交、但所有光线都未命中的块保持复用。
- 传递函数：任一点标量、opacity、gray 或附加字段变化时，所有在当前采样中遇到非零标量的块标记 `transfer_function`；仅含零标量/空光线的块复用。
- 相机：位置、目标、上方向、FOV 或有效宽高比任一变化时，全部分块标记 `camera`。
- 输出网格、分块尺寸、采样步长、背景值或体数据几何变化时，分别使用 `layout`、`render_settings`、`volume_base_changed`，并全部失效。
- 同一轮的多类变化按并集写入每个块的 `invalid_categories`，不会压成单一标记。

报告中的每个块包含像素范围、光线穿行体素包围范围、包围范围内变化体素数、扩展插值支撑范围变化体素数、命中变化体素的光线数、是否非空、失效类别、缓存状态、旧/新版本与摘要。

## 渲染与确定性

- 按像素中心生成单位方向光线，使用毫米单位和固定 `sampling_step_mm` 做前向合成。
- 体数据采样为三线性插值；传递函数线性插值，端点外取端点值。
- 合成公式固定为：`color += T * opacity * gray; T *= (1 - opacity)`。
- 每个线程只读取不可变的体数据、传递函数、相机与光线数组；按行优先块序拼装。并发度 1 与并发度 4 的结果逐像素、逐深度值完全一致。
- 内存预算按每个块的确定性采样缓冲大小做准入控制。块按索引顺序提交；预算不足时后续块等待，不降低采样步长或插值精度。

## 缓存、原子性与恢复

- 每个块写入 `state_dir/blocks/block_rRRRR_cCCCC.npz`，包含灰度、深度、规范化元数据和 payload SHA-256。
- 块文件先写同目录临时文件、`fsync`，再整体替换；旧版本先移动到 `state_dir/archive/...`，不会直接覆盖。
- 每个块记录当前全局 `version_id`（体积内容、传递函数、相机与渲染设置的规范化摘要）以及该块依赖的编辑清单摘要。
- 同一任务重启时，版本一致的块直接复用；增量任务中不受变化影响但全局版本较旧的块也可按逐块判据复用。
- 所有块完成后重新读取并校验数组形状、声明像素数和摘要。失败块记录 `E_DIGEST` 并自动补渲一次；再次失败则返回 `E_DIGEST`，不替换输出。
- 灰度、深度、报告与状态清单全部先写临时文件，再通过备份/替换事务发布；异常时回滚到旧输出。
- SIGINT/SIGTERM 只设置中断标志。调度器只在块边界停止提交，等待已运行块完成并持久化，写入 `state_dir/INTERRUPTED.json`，以退出码 `130` 退出，不生成灰度图、深度图或报告。重启后仅复用已持久化且摘要有效的块。

## 错误码

常见业务码：`E_JSON`、`E_TYPE`、`E_SCHEMA`、`E_DIMENSIONS`、`E_SPACING`、`E_VOLUME`、`E_TRANSFER`、`E_CAMERA`、`E_OUTPUT`、`E_BLOCK`、`E_CONCURRENCY`、`E_MEMORY`、`E_EDIT`、`E_DIGEST`、`E_STATE`、`E_RENDER`、`E_INTERRUPTED`、`E_IO`。所有错误 JSON 均包含字段路径 `path`。

## 实际结果

完整测试与样本验证结果见 `VERIFICATION.md` 和 `samples/verification.json`。
