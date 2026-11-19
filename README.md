# Volumetric Media Editor

面向透明介质与体积光的本地确定性离线渲染与增量编辑应用。场景完全由固定随机种子程序化生成，不读取外部素材，也不访问网络。

## 运行环境

- Windows 10/11 64 位，Python 3.11 64 位，`python` 与 `pip` 在 `PATH` 中。
- 工作路径建议仅包含 ASCII 字符且不含空格。
- 安装依赖：

```bat
python -m pip install -r requirements.txt
```

`numpy` 是渲染与积分内核的必需依赖；`PyOpenGL` 仅用于离屏上下文能力探测。驱动不支持 WGL/EGL 离屏上下文时，程序自动使用 NumPy 软件离屏帧缓冲，并把原因写入 `work/backend.log`。

## 一条命令演示

```bat
python -m volumetric_editor demo --root .
```

也可直接双击或在命令提示符执行：

```bat
run_demo.bat
```

演示会依次完成：

1. 生成并序列化场景，验证两次生成逐字节一致。
2. 在 `(线程数, 瓦片粒度) = (1,16), (2,16), (4,16), (2,24), (3,24)` 下渲染，验证图像与体积轨迹逐字节一致。
3. 以队列内存和计算预算触发固定优先级的并行度降级，确认所有瓦片仍恰好产出一次。
4. 注入零预算与任务准入不足，验证 `E_BUDGET_DENIED` 和可追溯的拒绝清单。
5. 增量修改折射率、光锥散射系数和瓦片粒度，验证闭包、复用与规范全帧哈希。
6. 注入全部业务错误，输出固定的错误码序列、逐瓦片哈希与体积积分段报告。

生成的目录：

- `data/scene.json`：可序列化场景、透明面法线/正反面材质与折射率。
- `data/volume_trace.npy`：最终规范体积轨迹，形状为 `(H, W, steps, 6)`。
- `work/demo_report.json`：完整演示报告、逐瓦片哈希、降级统计、闭包/复用清单和校验结果。
- `work/scatter_closure.json`：散射编辑的精确体积段闭包和复用清单。
- `work/volume_segments.csv`：锥内有效积分的逐段可读报告。
- `work/frames/*.rgba`：线性空间 `float32` RGBA 原始帧。
- `work/backend.log`：离屏上下文探测与软件回退记录。

## 测试

```bat
python -m unittest discover -s tests -v
```

覆盖场景序列化、跨线程/跨瓦片逐字节一致、预算降级与串行回退、拒绝追踪、折射率/散射/重排增量闭包，以及全部业务错误码。

## 核心实现

### 确定性场景

`volumetric_editor/scene.py` 使用 `random.Random(seed)` 生成两片曲面透镜、两片玻璃层、三处地面遮挡物和一个体积光锥。所有浮点参数量化到 6 位小数，JSON 使用 `sort_keys=True` 与紧凑分隔符写出 UTF-8 无 BOM 字节。每个透明面同时记录法线、折射率、正反面信息与稳定排序索引。

### 透明介质合成

`volumetric_editor/rendering.py` 对每个像素收集透明面命中，按以下稳定键从远到近合成：

```text
(-depth, -camera_distance, surface_sort_index)
```

每层使用 Snell 折射后的光线重新采样基础场景，并按透射率与确定性的表面反射项做部分透过合成。所有运算逐像素独立，不依赖对象集合遍历顺序。

### 体积光积分

每条视线使用固定步长与固定步数。每个分段记录：

```text
segment_start
segment_step
cumulative_transmittance
density
shadow_visibility
contribution_luminance
```

阴影采样数固定。抖动值由“全局种子 + 全局像素 x/y + 分段索引 + 阴影样本索引”的整数哈希产生，不使用 `Python hash()`，也不依赖瓦片局部坐标。因此改变线程数或瓦片划分不会改变抖动序列。

### 预算调度

`volumetric_editor/scheduling.py` 在提交任务前估算每个瓦片的队列内存与计算代价，并按固定优先级降级：

1. `QUEUE_MEMORY_LIMIT`：队列内存只能容纳更少的并发任务。
2. `COMPUTE_BUDGET_LIMIT`：计算准入窗口只能容纳更少的并发任务。
3. `SERIAL_FALLBACK`：并行度降至 1，但全部瓦片仍执行。

每次降级记录 `before` 与 `after` 中的工作线程数、在途队列内存和计算占用。结果通过 `as_completed` 收集，但只按瓦片键归并；完成顺序不参与图像组装。预算无法准入单个任务时返回 `E_BUDGET_DENIED`，拒绝清单逐任务列出瓦片键、原因和估算资源。

### 增量失效闭包

- 折射率编辑：每个瓦片缓存透明面命中位图。只重算命中该面的瓦片；体积轨迹原样复用。
- 光锥散射编辑：闭包为“每个有效视线的首个锥内段到末段”，因为累计透射率会沿该依赖链变化。闭包外瓦片及闭包内其它段都从旧缓存复制。
- 瓦片粒度编辑：不重新渲染，直接从规范全帧图像、可见性位图和体积轨迹裁剪出新瓦片，记录 `REUSED_RETILE`。

三种编辑都会对比编辑前后的复用瓦片内容哈希；任何闭包外变化都会触发运行期一致性错误，而不是静默接受。

## 错误码

| 场景 | 错误码 |
| --- | --- |
| 非法瓦片粒度 | `E_TILE_SIZE` |
| 线程数越界 | `E_THREAD_BOUNDS` |
| 折射率超出 `[1.0, 3.0]` | `E_IOR_RANGE` |
| 光锥散射系数为负或非有限值 | `E_MEDIUM_COEFF` |
| 队列内存预算为零或任务无法准入 | `E_BUDGET_DENIED` |

## 本次实测结果

验证环境：Python 3.11.0rc1，NumPy 2.4.6，当前 Linux/WSL 验证机。该机 EGL 无法创建 OpenGL 上下文（`EGL_BAD_ALLOC`），程序按设计回退到 NumPy 软件离屏，并记录了回退原因。Windows 的隐藏窗口 WGL 探测分支已实现，但本轮无法在真实 Windows 显卡驱动上复验。

- 自动化测试：`11` 项全部通过，耗时约 `3.9 s`。
- 演示固定场景：`96 x 64`，种子 `20261119`。
- 场景 JSON SHA-256：`4208dcce5176f5836a4440b46d74628ad546630387faf139e7dcfacd27581818`。
- 不同线程/瓦片组合的规范图像 SHA-256：`472d3ece9fe3607e8b09e3ebda11fe61e14721f20ee011fee84a295dfb4f7bdb`。
- 不同线程/瓦片组合的规范体积轨迹 SHA-256：`7a9aebcb6768045d8a27d1a29e3a9b3ead9d64780abb5a8f3d1b32e5802ef7ea`。
- 演示报告 SHA-256：`b56ca418a2079a99a0264f9e5f8aa10de2edd8d6da12cb1734a9c52c70995e06`。
- 同一命令连续运行两次，`data/` 与 `work/` 下全部 `14` 个输出文件逐字节一致。

复验可执行：

```bat
python -m volumetric_editor demo --root .
python -m volumetric_editor demo --root .
certutil -hashfile work\demo_report.json SHA256
```

两次输出的 `demo_report.json` 哈希应保持相同；报告中的 `checks.all_pass` 应为 `true`。

## 代码结构

```text
volumetric_editor/
  scene.py       确定性场景生成与序列化
  rendering.py   透明排序合成、体积步进、阴影采样、瓦片渲染
  scheduling.py  预算感知线程池、波次分配、降级与拒绝追踪
  session.py     规范帧缓存、增量闭包、复用与重排
  offscreen.py   WGL/EGL 离屏上下文探测与软件回退
  validation.py  业务校验与稳定错误码
  demo.py        可重复演示、报告与注入错误序列
  __main__.py    命令行入口
tests/           确定性、预算、增量与错误码测试
```
