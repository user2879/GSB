# 确定性合成图表：数据点还原与诊断工作台

这是一个完全本地运行的 Python 3.11+ 工作台。它从 PNG 图表页本身恢复坐标轴刻度语义，再把柱顶或折线拐点反解为业务数值；图表真值 JSON 只用于输入合法性检查和最终诊断，不参与标定计算。

工作台包含固定种子样本生成器、线性/对数/断轴轴映射、标签遮挡处理、阶段级内容指纹缓存、确定性线程池预算调度、原子快照恢复，以及串行/并行逐字段一致性自检。

## 快速开始

```bash
python -m pip install -r requirements.txt

# 可重复生成 PNG、真值 JSON、配置与编辑命令流
python -m chart_workbench generate

# 可复现串行基线
python -m chart_workbench run --serial --config config/calibration.json --output out/serial

# 线程池运行
python -m chart_workbench run --config config/calibration.json --output out/parallel

# 严格比较两份完整报告（包括缓存清单、预算序列、失败清单和快照摘要）
python -m chart_workbench compare \
  --left out/serial/report.json \
  --right out/parallel/report.json \
  --strict

# 一条命令执行串行/并行一致性验收并写入 out/selfcheck/selfcheck.json
python -m chart_workbench selfcheck

# 运行自动化测试
python -m unittest discover -s tests -v
```

Windows PowerShell 同样使用上述命令。代码只使用本地文件、NumPy、Pillow 和标准库线程池，不依赖 GPU、容器、外部站点或账号。

仓库根目录另外提供 `run_serial.py` 和 `run_parallel.py`，它们是上述 CLI 的薄入口：

```bash
python run_serial.py --config config/calibration.json --output out/serial
python run_parallel.py --config config/calibration.json --output out/parallel
```

## 样本与配置

`samples/` 中有 7 张固定 960×640 PNG 页及同名真值 JSON：

| 页面 | 图形 | 轴特征 |
|---|---|---|
| `page_01_linear_bars` | 柱状 | 线性轴，6 个柱 |
| `page_02_log_line` | 折线 | RGB PNG，纵轴 `log10` |
| `page_03_broken_line` | 折线 | 纵轴分两段 `broken_linear` |
| `page_04_linear_line_occluded` | 折线 | 纵轴 1 个标签被遮挡 |
| `page_05_dense_bars` | 柱状 | 横轴 11 个高密度刻度与柱 |
| `page_06_linear_bar_occluded` | 柱状 | 横轴 1 个标签被遮挡 |
| `page_07_log_bars` | 柱状 | 对数轴柱状图 |

固定种子、确定性像素坐标抖动和内置 3×5 位图字体使 PNG、真值、OCR 与测试结果可重复。`variants/` 提供单页替换和新增页面样本，供编辑命令流演示。

`config/calibration.json` 的核心参数：

| 参数 | 当前值 | 含义 |
|---|---:|---|
| `tick_min_length` | 6 | 最短刻度线段像素长度 |
| `line_width_tolerance` | 2 | 线宽/相邻行合并容差 |
| `axis_detection_threshold` | 0.45 | 横/纵轴线相对于画布边长的最小占比 |
| `broken_axis_detection_threshold` | 0.35 | 断轴判定阈值 |
| `log_axis_detection_threshold` | 0.01 | 对数拟合归一化残差上限 |
| `label_occlusion_tolerance` | 0.34 | 每轴允许缺失标签的比例 |
| `inverse_mode` | `ols` | `ols` 或 `endpoints` 坐标反解 |
| `fixed_decimal_places` | 3 | 定点输出小数位宽 |
| `quantization_step` | 0.1 | 业务值量化档位 |
| `thread_pool_concurrency` | 4 | 最大工作线程数 |
| `tick_comparison_budget` | 24 | 单批刻度比对点预算 |
| `page_data_point_quota` | 12 | 单页数据点配额 |
| `snapshot_commit_interval_pages` | 2 | 周期快照提交页数间隔 |

`thread_pool_concurrency` 和 `snapshot_commit_interval_pages` 是纯执行参数，不会进入结果配置哈希，因此改变并发度不会改变结果字段。预算、配额、反解方式、量化档位、阈值和样本路径都会进入相应缓存键或结果哈希。

## 输出

每次运行写入：

- `out/report.json`：主诊断报告。
- `out/reports/latest.json`：同一报告的最新副本。
- `out/reports/run-<配置哈希前缀>.json`：按配置哈希归档的报告。
- `out/state/versions/vNNNNNNNN/state.json`：完整快照清单。
- `out/state/versions/vNNNNNNNN/artifacts/<page>/<stage>.json`：阶段产物。
- `out/state/CURRENT`：原子切换的快照指针。

报告包含：

- `summary`：页数、数据点、刻度比对点、预算、失败与真值验证统计。
- `pages`：按文件名稳定排序；每页含刻度摘要、合并来源、轴映射类型、残差、分段、参与/排除刻度标识、数据点像素、还原值、定点文本、舍入方向、引用映射和真值验证指标。
- `failures`：稳定的失败清单，含 `E_CHART_INVALID`、`E_TICKS_NOT_FOUND`、`E_AXIS_MAPPING_FAILED`、`E_PAGE_QUOTA_EXCEEDED`、`E_CALIB_BUDGET_PENDING` 等业务码。
- `cache`：`reused`、`recomputed`、`recompute`、`skipped`、增删页、配置变化页及阶段统计。
- `budget`：逐批扣减序列、余额、待办事件及超单批预算页面。
- `snapshot`：版本号、配置哈希、完成/待办页、已消耗预算和产物版本。
- `result_digest`：排除缓存/快照执行痕迹后的确定性结果摘要。

## 实现结构

| 模块 | 职责 |
|---|---|
| `loader.py` | PNG 容器/尺寸/模式检查、灰度解码、真值结构与单调性校验 |
| `vision.py` | 轴线检测、刻度候选合并、标签连通域/OCR/关联、柱与折线标记检测 |
| `calibration.py` | 线性/对数/断轴拟合、坐标反解、定点量化、数据点还原 |
| `engine.py` | 阶段缓存、稳定预算批次、线程池、报告、恢复编排 |
| `snapshot.py` | 临时目录写入、版本目录提升、`CURRENT` 原子替换 |
| `generate.py` | 固定种子样本与真值生成 |
| `edits.py` | 替换/增删页面与配置修改命令流 |

像素阶段不读取真值。真值仅在加载校验和 `verification` 诊断中比较图形区域、刻度位置/值、数据像素与业务值。

## 标定算法

1. 用暗像素投影与最长连续游程检测长横线/长竖线，按轴阈值保留横轴与纵轴，得到左下绘图框。
2. 在轴外侧窄带提取逐行/逐列刻度游程；按线宽容差与相邻位置合并，保留每个合并候选的原始来源序号，并按“像素位置升序、再原候选序号升序”固定顺序。
3. 对轴外侧连通域按行邻接与字间距聚类为标签块，用内置位图模板做确定性 OCR，再按最近刻度关联；距离并列时按标签像素坐标升序决断。
4. 对关联刻度做线性拟合；正值刻度同时做 `log10` 拟合。仅当对数归一化残差达到阈值且优于线性时选 `log10`。
5. 对非对数轴计算相邻刻度值差与局部像素/值斜率；只有值跳变与斜率跳变同时超过断轴阈值才分段，避免“标签缺失”被误判为断轴。
6. 每轴记录参与刻度 ID、被遮挡刻度 ID、拟合残差和分段范围。标签缺失比例超过容忍度时输出轴映射错误。
7. 柱形按中灰填充连通域、底边贴合与矩形面积比识别；折线按高亮标记连通域识别。候选按像素横坐标稳定排序，容差内并列保留先出现的序列位置。
8. 纵轴映射反解业务值，再按 `quantization_step` 与 `fixed_decimal_places` 量化；输出舍入方向、原始值、定点文本与引用映射/分段。

## 缓存与精确失效

缓存键是完整 SHA-256，不依赖 Python 进程随机哈希：

```text
tick_key    = H(stage=ticks,   image_sha256, tick_parameters)
mapping_key = H(stage=mapping, tick_key,    mapping_parameters)
data_key    = H(stage=data,    mapping_key, data_parameters)
```

因此：

- 替换页面图像：该页三段全部重算，其他页复用。
- 增删页面：只记录新增页重算与删除页，其他页复用。
- 改 `quantization_step`：刻度、映射复用，仅数据阶段重算。
- 改对数/断轴阈值：刻度复用，映射与数据重算。
- 改最短刻度长度、线宽容差或轴阈值：刻度及下游重算。
- 改并发度：所有产物复用，且报告逐字段不变。

缓存命中时对应函数不会被调用。`cache.recompute` 是按页面名、阶段顺序排列的精确重算清单。

## 预算与恢复

刻度阶段按文件名顺序预计算每页比对点成本。调度器以固定顺序填充每批预算，装不下的页保留到下一批，并写入 `E_CALIB_BUDGET_PENDING` 事件；如果单页成本本身超过单批预算，会在最终失败清单保留该页。最终页面顺序、批次、扣减序列和失败清单都不受线程完成顺序影响。

快照先写入 `state/.staging-*`，随后以目录重命名提升为不可变版本目录，再用原子 `CURRENT` 文件切换指针。启动时读取指针版本，比对配置哈希和每阶段缓存键；只有键不匹配的产物才重算。收到 SIGINT/SIGTERM 时停止提交新批次，等待当前批次完成并提交快照后输出恢复所需状态。

## 编辑命令流

`config/edit_commands.json` 演示以下命令：

```json
{"op": "replace_page", "page": "page_01_linear_bars.png", "source_png": "...", "source_truth": "..."}
{"op": "add_page", "page": "page_08_extra_line.png", "source_png": "...", "source_truth": "..."}
{"op": "remove_page", "page": "page_06_linear_bar_occluded.png"}
{"op": "set_config", "path": "log_axis_detection_threshold", "value": 0.000001}
{"op": "set_config", "path": "quantization_step", "value": 0.5}
```

应用前建议先备份样本目录；命令是显式文件编辑：

```bash
python -m chart_workbench apply-edits \
  --config config/calibration.json \
  --commands config/edit_commands.json
```

随后再次运行工作台即可看到精确失效清单。仓库交付的 `samples/` 保持基线状态，示例命令不会自动执行。

## 验证

```bash
python -m unittest discover -s tests -v
python -m chart_workbench selfcheck
python -m chart_workbench validate-report --report out/selfcheck/parallel/report.json
```

实际验证结果见 `VALIDATION.md`。

## 边界与假设

- 合成页采用左下直角坐标轴、向外刻度、3×5 位图数值标签；OCR 面向该确定性字体，而非任意字体。
- 柱状图为由底边向上的竖柱；折线图在每个数据点绘制中灰实心标记，以便与黑线分离。
- 断轴页的点应位于可用分段内；落入分段空隙时会选择最近分段并在数据点记录 `selected_from_break_gap`。
- 单页刻度成本大于单批预算时无法在固定批量上限内完成，会保留为待办失败；可提高预算后恢复。
