# 竖排版面关键点诊断

这是一个本地、确定性、可增量重放的竖排版面关键点诊断实现。仓库包含固定种子样本生成器、加载校验、滑窗粗定位、连通域细定位、定点关键点回归、阅读顺序冲突消解、规模档位选择、精确缓存失效、预算调度、原子快照、恢复判定和诊断报告。

核心实现不依赖 GPU、网络服务或外部字体。运行依赖仅为 Python 3.10+、NumPy 和 Pillow；Tk 界面为可选入口。

## 快速开始

```bash
cd /home/zsl/feedback/uidfc5909d3-default-302

# 当前镜像的 base Python 缺少 ensurepip；本仓库已用 virtualenv 兼容方式创建 .venv。
python3 -m virtualenv --system-site-packages .venv
source .venv/bin/activate
python -m pip install numpy pillow

# 生成 640x960 灰度 PNG、真值、v1/v2 标注版本和默认配置
python -m layout_diag generate --out samples --config-out config/default.json

# 校验并运行
python -m layout_diag validate --dataset samples --config config/default.json
python -m layout_diag run --dataset samples --config config/default.json --out diagnostics/demo
python -m layout_diag report --out diagnostics/demo
```

默认样本包含 3 张图、74 个字符块：

| 图像 | 栏数 | 每栏字符数 | 点数 |
|---|---:|---:|---:|
| `small_layout_01.png` | 2 | 3 | 6 |
| `medium_layout_01.png` | 3 | 6 | 18 |
| `large_layout_01.png` | 5 | 10 | 50 |

所有图像固定为 `640x960`、8-bit 灰度 PNG。字模由内置 5x7 点阵放大绘制，并施加确定性的轻微旋转、笔画抖动、遮挡条和背景噪点。

## 命令

```bash
# 强制忽略旧缓存并全量重建
python -m layout_diag run ... --rebuild

# 强制单 worker，用于并发/串行一致性核对
python -m layout_diag run ... --serial

# 应用可重放的编辑命令流
python -m layout_diag edit --dataset samples --config config/default.json --commands edits.json

# 比较两次运行的确定性产物；默认不比较实际 cache hit/miss 状态
python -m layout_diag compare --left diagnostics/a --right diagnostics/b

# 可选 Tk 检查器；依赖系统 libtk
python -m layout_diag gui
```

编辑命令支持：

- 图像：`replace_image`、`add_image`、`remove_image`
- 标注：`replace_keypoints`、`set_keypoint_version`
- 滑窗：`set_window`、`set_step`
- 档位：`set_tier`
- 连通域：`set_merge_threshold`、`set_cc`
- 定点：`set_quantization`
- 调度：`set_point_budget`、`set_max_workers`

## 诊断链路

1. **加载与校验**：逐 PNG 检查容器签名、IHDR、固定分辨率、8-bit 灰度模式；校验真值和所有关键点版本，包括整数坐标、图内边界、未知栏、栏区间重叠、连续阅读序号、声明字符数与实际点数。
2. **滑窗粗定位**：按窗口和步长切窗，计算墨迹密度、竖直投影谷、纹理梯度，并叠加由图像内容指纹和窗口几何派生的固定投影偏置；定点化后做稳定排序和 NMS。
3. **连通域细定位**：8 邻域连通域标号、投影谷分裂、Chebyshev 边界间隔合并、面积/长宽比/候选区域过滤。所有分裂与合并逐条记录依据。
4. **关键点回归**：以墨迹重心和扩展后的块边界联合估计中心，在 `2^fixed_bits` 定点域内量化，输出坐标、量化残差、滑窗 ID 和连通域 ID 引用。
5. **阅读顺序**：按栏右到左、栏内上到下串接，检测 `cross_column`、`wrong_order`、`missing_number`，冲突按关键点序号升序输出，不静默丢弃。
6. **档位选择**：按图数、总点数、最大栏数和每栏最大字符数，从预置策略表选择最小可容纳档位，并把单图估计代价和依据写入报告。
7. **精确失效**：滑窗、连通域、逐栏关键点、阅读顺序分别使用内容/参数/量化指纹缓存键；替换单图只失效该图的引用链。
8. **预算与快照**：按文件名、栏序号稳定排序消费预算，超预算栏进入待办并输出 `E_POINT_BUDGET_PENDING`；产物写入临时目录后整目录替换，支持中断后的 `.previous` 恢复。

## 档位策略

| 档位 | 窗口/步长 | 合并阈值 | 定点位宽 | 量化步长 | 容量 |
|---|---|---:|---|---:|---|
| `small` | 32 / 16 | 4 | 10 | 1 | ≤48 点、≤3 栏、≤5 字/栏 |
| `medium` | 64 / 32 | 3 | 8 | 2 | ≤180 点、≤4 栏、≤8 字/栏 |
| `large` | 80 / 40 | 5 | 6 | 4 | 大版面兜底 |

档位策略来自 `config/default.json` 的 `strategies`，也可用 `tier: "custom"` 直接使用顶层 `window`、`cc`、`precision`。

## 输出

`diagnostics/<run>/current/` 是一个原子提交快照：

- `manifest.json`：稳定全局版本、配置/数据集/有效参数哈希、完成/待办/失败图像、完成点数、待办点数。
- `validation.json`：加载期失败清单与档位依据。
- `schedule.json`：预算计划、稳定排序任务、完整/部分/待办栏和待办点数。
- `coarse.json`：每图窗口数、候选窗口、密度/投影谷/梯度/指纹偏置/定点得分。
- `components.json`：连通域、过滤记录、分裂记录、逐条合并依据、滑窗引用。
- `keypoints.json`：定点坐标、量化残差、栏、窗口、连通域引用和全局关键点序号。
- `reading_order.json`：逐图序列、逐点阅读序号、全局冲突清单。
- `failures.json`：最终业务码失败清单。
- `reuse.json`：逐阶段、逐栏、逐关键点复用命中或失效记录。
- `report.json`：档位依据、关键点统计、标注版本哈希、冲突计数、业务码计数、缓存统计、调度键。
- `cache.json`：下一次增量运行使用的内容寻址阶段缓存。

## 业务码

加载与运行会输出包括但不限于：

`E_KEYPOINT_OOB`、`E_COLUMN_OVERLAP`、`E_WINDOW_ALIGN`、`E_IMAGE_HEADER`、`E_IMAGE_SIZE`、`E_IMAGE_MODE`、`E_READING_SEQUENCE`、`E_QUANT_ENUM`、`E_POINT_BUDGET_PENDING`。

未知栏和越界坐标都归入 `E_KEYPOINT_OOB`，通过 `details.reason` 区分 `unknown_column` 与 `outside_image`。

## 验证

```bash
python -m unittest discover -v
python tools/verify.py --json-out verification/results.json
```

最近一次实际验证结果见 [`verification/results.json`](verification/results.json) 和 [`docs/VERIFICATION.md`](docs/VERIFICATION.md)。摘要：

- 26 个测试全部通过。
- 默认样本 74/74 关键点检出，74 个最终连通域，295 个粗候选，0 失败、0 阅读冲突。
- `small/medium/large` 候选数分别为 1334/454/295，首个关键点定点坐标均不同。
- 预算 20 时处理 20 点、待办 54 点、8 个待办栏；同预算续跑结果一致。
- 替换单图后，150 条产物复用命中、17 条失效；未替换图像全部命中。
- 线程池和串行执行输出一致。
- 增量运行与强制全量重建的确定性诊断载荷一致；实际 hit/miss 状态因缓存是否存在而天然不同，但复用清单的稳定键和顺序一致。

## 环境说明

当前执行镜像的 `python3 -m venv` 因缺少 `ensurepip` 失败，已改用等价的 `python3 -m virtualenv --system-site-packages .venv`。NumPy/Pillow 可在该虚拟环境中使用。

当前镜像同时缺少 `libtk8.6.so` 且无免密 `sudo`，因此可选 Tk 入口已实现但无法在本镜像内实际打开窗口；核心诊断、CLI、测试和验证均在无 GUI 模式下完成。

更详细的算法与不变量见 [`docs/DESIGN.md`](docs/DESIGN.md)。
