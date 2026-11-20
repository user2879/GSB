# 确定性合成版面表格与图表结构化解析器

本项目从零实现一个离线、可重复的本地版面解析应用。输入是固定种子生成的
PNG 页面、逐页真值 JSON 和解析配置 JSON；输出是逐页表格单元格网格、合并
跨度、行列线引用、图表数据点、缓存失效记录、预算账本、失败/待算队列和
快照恢复信息。

实现只依赖 Python 标准库、NumPy 和 pytest。不下载模型、图像或外部数据，
不使用 Docker 或其他容器环境。

## 环境

- Python `3.11.0rc1`（当前环境中的 `python3.11` / `python3`）
- NumPy `1.26.4`
- PyYAML `6.0.2`
- pytest `8.3.3`（仅测试）

依赖版本记录在 `requirements.txt` 和 `requirements-dev.txt`。这些包均来自
当前环境的本地缓存/预装源；实现本身不访问网络。

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -r requirements-dev.txt
```

## 快速运行

仓库已包含固定种子生成的 `samples/` 样本与 `config/parse_config.json`：

```bash
python3 -m parseapp run \
  --samples samples \
  --config config/parse_config.json \
  --state-dir .parse_state/demo
```

默认诊断报告写入 `.parse_state/demo/report.json`。CLI 退出码：

- `0`：运行完成，无失败、无待算页
- `2`：预算未完成，存在 `E_POINT_BUDGET_PENDING`
- `3`：运行完成但存在 `E_PAGE_INVALID` 失败页
- `4`：配置、输入目录或快照错误
- `75`：测试用强制终止点触发，未写最终报告

串行对照模式只改变执行器并发度，不改变报告业务字段：

```bash
python3 -m parseapp run --samples samples \
  --config config/parse_config.json \
  --state-dir .parse_state/serial --serial
```

重新生成确定性样本：

```bash
python3 scripts/generate_samples.py --out samples --seed 20261120
```

## 目录结构

```text
parseapp/
  png.py          PNG 容器校验、解码和确定性编码
  validation.py   真值 schema、边界、线序、单元格覆盖与图表点校验
  loader.py       文件名稳定扫描、页面加载与内容指纹
  detection.py    线段候选、短候选丢弃、线宽合并、区域角色评分
  grid.py         交点切格、碎片过滤、缺线合并、单元格缓存键
  charts.py       柱/折线识别、轮廓/标记测量、归一化、配额和点缓存键
  scheduler.py    与完成顺序无关的预算准入和待算队列
  snapshot.py     整代快照、哈希清单、原子 CURRENT 指针
  pipeline.py     并行调度、缓存复用、恢复、诊断报告
  cli.py          命令行入口
scripts/
  generate_samples.py    固定种子样本生成
  check_windows_atomic.py Windows 临时目录/原子替换检查
tests/
  test_parseapp.py       端到端和单元验收测试
config/
  parse_config.json      默认配置
  parse_config.budget_demo.json  低预算演示配置
samples/                 4 页 PNG 与 4 份真值 JSON
```

## 页面真值 JSON

区域和单元格边界使用包含端点的整数像素坐标 `[left, top, right, bottom]`；
行列线坐标是合并后线段中心的整数坐标。页面真值示例字段：

```json
{
  "schema_version": "1.0",
  "page_id": "page_01",
  "image": "page_01.png",
  "image_size": [960, 600],
  "pixel_mode": "RGB8",
  "tables": [{
    "table_id": "T1",
    "region": [40, 80, 540, 400],
    "row_lines": [80, 160, 240, 320, 400],
    "col_lines": [40, 150, 260, 370, 480, 540],
    "cells": [{
      "cell_id": "T1_R0C0",
      "row": 0, "col": 0, "row_span": 1, "col_span": 1,
      "bbox": [40, 80, 150, 160],
      "text": "T1:R0C0",
      "content_bbox": [52, 92, 138, 148]
    }]
  }],
  "charts": [{
    "chart_id": "G1",
    "kind": "bar",
    "region": [560, 65, 930, 465],
    "plot_area": [635, 105, 905, 430],
    "axis": {
      "x": {"pixel_min": 650, "pixel_max": 885, "value_min": 0, "value_max": 5},
      "y": {"pixel_min": 420, "pixel_max": 120, "value_min": 0, "value_max": 100}
    },
    "points": [{"index": 0, "x": 0, "y": 35, "series": "s0", "pixel": [650, 315]}]
  }]
}
```

校验内容包括：

- PNG 头、IHDR、CRC、尺寸、8 位像素模式、非交错解码；输出为 `uint8`
  灰度数组。
- 表和图表区域为整数且落在图像内；行线严格递增，列线严格递增，重复/
  乱序线被拒绝。
- 单元格的 `row_span` / `col_span` 与 `bbox` 必须和行列线坐标一致；
  单元格完整且不重叠地覆盖基础网格。
- 文本字段必须存在；图表点必须含唯一 `index`、数值 `x/y`、序列名，
  可选 `pixel` 必须在图表区域内。
- 任一页面非法时输出业务码 `E_PAGE_INVALID` 和细粒度 `detail_code`，
  放入失败清单，其他页面继续运行。

文本内容在无 OCR 模型约束下作为页面真值标注回填；图像负责线、单元格和
图表几何。柱/折线点的像素位置从图像测量，数值通过真值中提供的像素-数值
坐标轴标定映射。真值中的期望点值用于验收，不参与点的像素定位。

## 解析配置

默认配置见 `config/parse_config.json`。

| 字段 | 含义 |
| --- | --- |
| `binary_threshold` | 像素 `< threshold` 视为前景，范围 0..255 |
| `min_line_length` | 合并后线段最短长度 |
| `line_merge_tolerance` | 相邻同向线段中心坐标合并容差；最大合并宽度为 `2*tolerance+1` |
| `cell_min_area` | 单元格最小面积，碎片记录 `E_CELL_AREA_TOO_SMALL` |
| `chart_decision_threshold` | 区域表网格分数阈值；分数 `>=` 阈值判表，否则判图表 |
| `coordinate_normalization` | `minmax` 或 `pixel_relative` |
| `thread_pool_concurrency` | 页面加载和解析线程池最大并发数 |
| `batch_detection_point_budget` | 单批线检测候选点数预算 |
| `page_chart_point_quota` | 每个页面图表点配额 |
| `snapshot_commit_interval` | 每完成多少个页面提交一次活动快照 |
| `expected_image_width/height/pixel_mode` | 可选的固定页面尺寸/模式约束 |

配置别名（例如 `thread_pool_size`、`batch_detection_points`、
`chart_point_quota`、`snapshot_interval`）会被归一化为上表字段。

### 检测点数语义

每段长度至少 8 像素的前景连续行程算作一个候选检测点，包括长度在
`8..min_line_length-1` 的短候选。短候选记录 `E_LINE_TOO_SHORT`；合并后
厚度超过容差的候选记录 `E_LINE_WIDTH_EXCEEDS_TOLERANCE`。每个候选消耗
1 个预算点。图表数据点不占线检测预算，受独立的单页图表配额限制。

### 图文区域评分

对区域统计长水平/垂直线的数量与覆盖率：

```text
table_grid_score = min(1, h_lines/2) * min(1, v_lines/2) * min(h_coverage, v_coverage)
```

分数低于阈值判为图表，否则判为表格。图表区域与表格区域重叠时固定采用
表格优先；图表分类记录 `TABLE_PRIORITY_ON_OVERLAP`、交叠框和被排除框。
当区域评分与真值声明的角色冲突时，页面以 `E_REGION_ROLE_MISMATCH` 进入
`E_PAGE_INVALID` 失败清单。

## 确定性调度与预算

- 页面按 PNG 文件名字典序扫描；每个页面任务带有 `<file>|line`、
  `<file>|grid`、`<file>|chart` 稳定排序键。
- 预算准入在主线程按文件名顺序一次性规划，再提交线程池执行，因此完成
  顺序不会改变扣减序列。
- 一旦某个正成本页面无法准入，后续正成本页面不再越过队列；这些页面输出
  `E_POINT_BUDGET_PENDING`，队列保持文件名顺序。
- 新一次调用是新的预算批次，但优先复用快照中已完成的页面和产物。待算
  页面在下一批继续。强制终止后的同一活动批次会恢复原先冻结的预算计划，
  不会重复扣减。
- 单页图表配额按文件名顺序分配，报告中记录 `chart_quota.assignment_order`；
  超出配额的数据点截断并记录 `E_CHART_POINT_QUOTA_TRUNCATED` 警告。

## 缓存与失效

缓存键均以 canonical JSON 和 SHA-256 派生：

- 线检测：页面灰度内容指纹 + 二值化/最短线长/合并容差。
- 网格：线检测键 + 单元格面积参数 + 真值表格布局 + 线 ID + 表格角色。
- 图表：页面内容指纹 + 图表参数 + 真值图表布局/轴标定 + 图表角色/重叠。
- 单元格：单元格裁切像素哈希 + 行列跨度 + 边界与线 ID + 文本标注。
- 数据点：图表区域裁切哈希 + 图表参数 + 像素位置 + 轴标定 + 轴线段引用。

修改图表阈值或归一化方式只重算图表；修改 `cell_min_area` 只重算网格，且
仍满足条件的单元格会复用；替换单页图像只影响该页。页面图像的无关区域
变化可以重跑该页线/图阶段，但未变化的单元格和图表数据点仍按裁切内容键
复用。每次报告列出每个线、网格、单元格、图表点单元的
`reused` / `recomputed` 状态与缓存键。

## 快照与恢复

状态目录布局：

```text
<state-dir>/
  snapshots/
    CURRENT.json                 # 原子替换的当前版本指针
    generations/v000001/          # 不可变、完整的一次提交
      manifest.json
      pages/page_01.json
      ...
  report.json                    # 完整调用结束后原子写入
```

每次提交先在同一文件系统的临时目录写完整代清单和全部页面产物，再通过
原子 `os.replace` 发布代目录；最后原子替换 `CURRENT.json`。`CURRENT.json`
记录代名、全局版本号和 `manifest.json` 的 SHA-256。读取时验证清单哈希和
每个产物文件哈希；只有完整代才能恢复。临时目录和未发布代不会被恢复。

模拟强制终止：

```bash
python3 -m parseapp run --samples samples \
  --config config/parse_config.json \
  --state-dir .parse_state/crash \
  --fail-after-commits 1
# 退出码 75，且没有 report.json

python3 -m parseapp run --samples samples \
  --config config/parse_config.json \
  --state-dir .parse_state/crash
```

恢复报告包含 `recovery.snapshot_version`、`recovery_point_pages`、
恢复的已消耗预算和待算队列。强制终止路径不承诺生成最终退出报告；只有
完整结束的调用才写顶层 `report.json`。

### Windows 原子替换

提交实现把临时目录和 `CURRENT.json` 临时文件都建立在同一目录/卷中，再
使用 `os.replace` 发布；不会覆盖已有代目录。该方式在 Windows 上应使用
同一卷文件系统。需要在目标 Windows 主机上实际确认时运行：

```powershell
python scripts\check_windows_atomic.py
```

输出中的 `temp_dir_created` 和 `atomic_replace_supported` 应为 `true`。
本次实际验证在 Linux/WSL2 上完成；Windows 分支未在本机实际执行，只完成了
同路径、同卷、原子文件替换的兼容实现和检查脚本。

## 诊断报告

`report.json` 主要字段：

- `page_order`：稳定页面顺序
- `pages[]`：状态、业务码、线检测合并/丢弃记录、表格网格、图表点、
  单元格/点缓存键与复用状态
- `budget_ledger`：批次、上限、扣减序列、准入页、待算队列
- `pending_queue` / `failures`：稳定排序的待算与失败清单
- `invalidation`：逐页逐单元的失效/复用清单和汇总
- `chart_quota`：按文件名的单页图表配额分配顺序
- `recovery`：恢复点页面、版本、恢复预算和快照警告
- `snapshot`：最终全局版本、代目录、配置哈希和产物版本

## 已验证结果

详细运行记录见 `VALIDATION.md`。核心验收：

- 4 页样本的 60 个单元格、25 个图表数据点与真值逐字段一致。
- 全新并行运行与 `--serial` 运行的完整 canonical JSON 报告 SHA-256 相同。
- 预算 1500 时第 1/2 页完成，第 3/4 页进入稳定待算队列；下一次调用继续并
  完成，待算队列清空。
- 在第一次快照提交后强制终止：无最终报告；重启从版本 1 恢复第 1/2 页，
  冻结的四条预算扣减序列与无中断运行一致。
- 单页图像无关区域修改只重算该页三阶段，但其 14 个单元格与 7 个数据点
  全部复用；其他三页全部复用。
- 图表阈值调整只重算图表；网格参数调整只重算网格；归一化方式调整只重算
  图表。
- `pytest -q`：13 passed。
