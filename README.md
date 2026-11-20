# 确定性版面阅读顺序工作台

这是一个完全本地、CPU 上运行的合成版面分析工作台。它从固定种子的 PNG 页面中检测块、判定角色、推断栏组、生成阅读顺序，并通过内容指纹缓存、原子快照、并发确定性和真实故障注入来验证结果的可复现性。

项目不依赖网络、账号、GPU、Docker 或外部数据集。运行依赖仅为 Python 3.10+、NumPy 与 Pillow。

## 快速开始

```bash
python3 -m pip install numpy pillow

# 可选：重新生成固定种子样本（当前仓库已包含 samples/）
python3 scripts/generate_samples.py --out samples --seed 20261121

# 全量运行，自动读取 samples/layout_truth.json
python3 -m layout_reader analyze \
  --input samples \
  --config configs/default.json \
  --state runtime \
  --report out/report.json

# 串行运行（分析结果应与并发运行逐字段一致）
python3 -m layout_reader analyze \
  --input samples --config configs/default.json \
  --state runtime_serial --report out/report_serial.json --serial

# 故障注入运行
python3 -m layout_reader analyze \
  --input samples --config configs/default.json \
  --faults faults/sample_injections.json \
  --state runtime_faults --report out/report_faults.json

# 查看最近一次完整提交的快照与恢复点
python3 -m layout_reader status --state runtime

# 自动化验收
pytest -q
```

CLI 正常完成或收到可处理信号时返回整数 `0`。配置错误或不可恢复的内部错误返回 `2`；预期中的页面级业务故障仍在报告和失败清单中体现，但不会中断其他页面。

## 样本与配置

`samples/` 包含 6 张 720×960 PNG：

- `page_001.png`、`page_005.png`：单栏；
- `page_002.png`、`page_006.png`：双栏；
- `page_003.png`：三栏；
- `page_004.png`：双栏混合通栏图片；
- 覆盖标题、正文、图片、页眉页脚四种角色，正文/标题/图片/页眉页脚均为整数像素矩形；
- `page_006.png` 使用 RGB 模式，其余页面使用 L 模式；
- `samples/layout_truth.json` 记录每个块的 bbox、角色、行列分组、通栏标记、`column_first` 阅读序号和 `region_first` 阅读序号。

`configs/default.json` 包含二值化阈值、最小块尺寸、投影窗口、栏间距、标题/正文高度比、图片密度、角色优先级、阅读顺序规则、并发数、批大小与快照间隔。

`faults/sample_injections.json` 包含四条真实输入故障：

| 编号 | 页面/阶段 | 故障 | 业务码 |
| --- | --- | --- | --- |
| F001 | page_002 / image | 截断 PNG 字节 | E_IMAGE_INVALID |
| F002 | page_003 / detect | 块坐标越界 | E_BLOCK_BOUNDS |
| F003 | page_004 / columns | 分栏间距置零 | E_COLUMN_GAP_INVALID |
| F004 | page_005 / order | 阅读序号重复 | E_ORDER_DUPLICATE |

## 流水线与确定性设计

每个页面依次计算五个阶段：

1. `image`：校验 PNG 容器签名、Pillow 可解码性、固定尺寸和允许的像素模式，解码为灰度并持久化压缩灰度产物；
2. `detect`：按二值化阈值得到墨迹掩码，使用游程连通域和水平/垂直投影窗口聚类，过滤小噪声块，输出整数 bbox、墨迹密度与投影周期性；
3. `role`：由块高/正文中位高度比、密度、水平投影周期性、页眉页脚位置得到所有合格角色，再按配置的固定角色优先级决断；记录全部指标、合格角色、命中规则和 tie-break 情况；
4. `columns`：从非通栏候选块的水平区间推栏目带，按水平重叠度分配块并单独标记通栏块；边界冲突按“更大栏宽稳定性 → 更多栏内块 → 最左栏”记录决断；
5. `order`：`column_first` 按栏组、纵坐标、横坐标排序；`region_first` 按纵向带、带内横坐标排序；通栏块作为纵向分隔符插入，并记录其覆盖栏组和前后块数量。

所有输出都显式排序：页面按文件名、块按 `(y0, x0, x1, y1)`、失败按 `(page, stage)`、注入项按编号。集合只用于成员判断，不用于决定输出顺序。每条阶段任务还携带稳定的 `filename\0stage` 排序键。

阅读顺序规则示例：

- 分栏优先：先输出通栏分隔符之上的栏内块，再插入分隔符，然后继续后续栏内块；
- 区域优先：先按纵向间隙形成水平带，带内按横坐标，再在纵向位置插入通栏块。

## 内容指纹与增量失效

缓存键由阶段实现版本、上游内容指纹、该阶段参数投影和该页面/阶段的注入项共同派生：

```text
image_key   = H(image_version, raw_png_sha256, image_validation, image_faults)
detect_key  = H(detect_version, image_key, gray_sha256, threshold/min/projection, detect_faults)
role_key    = H(role_version, detect_key, role_thresholds, role_faults)
columns_key = H(columns_version, role_key, column_parameters, column_faults)
order_key   = H(order_version, columns_key, ordering_parameters, order_faults)
```

因此：

- 修改二值化阈值只重算 `detect` 及下游，`image` 直接复用持久化灰度；
- 修改分栏间距只重算 `columns` 与 `order`；
- 切换阅读顺序规则只重算 `order`；
- 替换单页 PNG 只使该页五个阶段失效；
- 增加/删除页面不会重算未变化页面；
- 执行参数（并发数、批大小、快照间隔）不进入分析阶段参数哈希。

报告中的 `page_cache` 逐页列出每个阶段是 `reused`、`recomputed` 还是 `failed`，并给出缓存键和对象 ID。`result_digest` 只覆盖可复现的分析内容、失败、注入命中与真值校验，不覆盖执行器或缓存命中元数据，因此增量结果可与空状态全量重建逐字段比较。

## 原子快照与恢复

`state/objects/<sha256>.json` 是不可变的内容寻址阶段产物。写盘流程为同目录临时文件、`fsync`、`os.replace`。每次提交还会写一个完整不可变快照 `state/snapshots/vNNNNNN.json`，最后才原子替换根 `state/manifest.json`。

清单包含：

- 全局版本号、配置哈希、已完成页、失败页；
- 每个页面每个阶段的缓存键、实现版本和对象 ID；
- 最后处理页、最后完成页、是否处于可处理中断状态。

提交前会比较规范化状态投影；没有状态变化时不会创建新版本。强制终止只会丢失尚未提交的页，重启时只读取根清单指向的完整快照。报告的 `recovery` 字段输出恢复版本、恢复页和恢复点页面。

SIGINT/SIGTERM 由主线程捕获：不再提交新批次，等待当前批次页面完成阶段写入，提交进度快照，写诊断报告，并以 `0` 退出。SIGKILL 无法捕获，因此不承诺生成退出报告，但已提交快照和进度仍可用于下次恢复。

## 报告结构

诊断报告由 `--report` 原子写出，主要字段如下：

```text
schema_version / status / config_hash / result_digest
execution                 并发模式、worker 数、批大小、快照间隔、稳定排序键
snapshot                  基准版本、提交版本、本次提交版本序列
recovery                  是否恢复、恢复点、恢复页
summary                   页数、成功/失败、阶段复用/重算、注入命中、真值通过
pages[]                   每页图像摘要、块清单、角色依据、栏组、阅读序号、诊断摘要
page_cache[]              每页五阶段的缓存键/对象 ID/复用状态/稳定任务键
failures[]                页面级失败，含业务码、阶段、注入编号与详情
injections[]              注入命中证据及故障到业务码的映射
truth_validation          块 bbox、角色、栏组、阅读顺序与真值逐项比对
pending_pages[]           可处理中断时尚未处理的页面
```

页面块输出包括 `bbox`、`ink_density`、`projection_periodicity`、`role`、`role_evidence`、`column_id`、`is_spanning`、`reading_order` 和 `reading_order_rule`。鉴权类敏感信息不会被读取或写入。

## 实际验证记录

在交付环境中执行：

```text
$ pytest -q
........                                                                 [100%]
8 passed
```

自动化覆盖：

- 6 页全量检测、角色、栏组、两种阅读顺序和真值比对；
- 串行与 4 线程并发的 `pages`、`failures`、`injections`、`page_cache`、`result_digest` 完全一致；
- 阈值调整、排序切换、单页替换、页面增加与删除后的增量结果与空状态全量重建逐字段一致；
- 阈值运行复用 6 个 image 阶段、重算 24 个下游阶段；排序切换复用 24 个上游阶段、仅重算 6 个 order 阶段；替换单页仅使该页 5 个阶段失效；增删页面后未变化页面全部复用；
- 四条注入全部命中并分别映射到四个业务码，未命中的页面与干净运行逐字段一致；
- 角色高度/密度边界为闭区间，多角色合格时按配置优先级决断并记录 tie-break；
- 栏组边界冲突按栏宽稳定性、栏内块数、最左栏顺序记录；
- 可处理中断写入快照和恢复点，恢复运行与全量运行 `result_digest` 一致；
- CLI 收到 SIGTERM 后返回 `0` 并生成 `interrupted` 报告。

干净全量运行的样本统计：

```text
page_001: 17 blocks, 1 column
page_002: 28 blocks, 2 columns
page_003: 41 blocks, 3 columns
page_004: 27 blocks, 2 columns + spanning separators
page_005: 17 blocks, 1 column
page_006: 30 blocks, 2 columns
total:    160 blocks; truth 6/6 pages passed
```

故障注入运行的结果为：2 页成功、4 页按预期失败、4/4 注入命中；`page_001` 与 `page_006` 的分析结果与无注入运行完全一致。

## 目录

```text
layout_reader/          分析、缓存、快照、报告与 CLI
scripts/generate_samples.py
samples/                固定种子 PNG 与真值 JSON
configs/default.json
faults/                 示例注入清单与空清单
tests/                  验收测试
pyproject.toml
requirements.txt        运行依赖
requirements-dev.txt    测试依赖
```

## 已知边界

样本是程序生成的矩形版面，检测器利用墨迹游程与投影窗口，不包含 OCR、旋转文本、抗锯齿字体或真实扫描噪声；这些不属于本任务的可复现要求。输入目录当前扫描顶层 PNG 文件。故障注入是输入级故障模拟，不会破坏原始样本文件。
