# MocapClean Workbench

面向运动捕捉数据清洗的 Windows 桌面工作台。它从固定种子程序化生成单套 19 关节骨架的 60 fps 动捕记录，注入四类废帧，完成可配置检测、稀疏关键姿态修复、真值参照评估、增量失效闭包、后台候选预览与原子账本恢复。

**本地、确定性、无外部动捕资产。** 记录、废帧清单和真值均由代码生成；真值只用于检测退化和修复结果核对，不参与修复算术。

## 快速开始（Windows）

```powershell
cd <工作区>
powershell -ExecutionPolicy Bypass -File setup_windows.ps1
.\run_workbench.bat 1200
```

手动安装：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m mocap_workbench demo --workspace . --frames 1200
.\.venv\Scripts\python.exe -m mocap_workbench app --workspace . --frames 1200
```

`numpy` 与 `PySide6` 是唯一运行依赖。无 GPU、无网络服务；依赖安装后可离线运行。三档数据、审计输出、日志、状态账本全部落在工作区相对目录。

## 工作台操作

- 顶部选择 `1200 / 6000 / 12000` 帧档并点击“载入/生成”。首次进入时生成数据、计算证据并建立基线提交；再次进入时从最近提交恢复。
- 时间轴上半部显示缺陷段颜色、根 Y 轨迹、修复最大误差热度和关键帧菱形。白线是当前帧，青色阴影是增量闭包帧。
- 单击移动游标，双击干净帧添加关键帧；拖动菱形关键帧可移动自动边界或用户锚点。自动边界移动会抑制原边界并在目标帧建立用户锚点。
- 修改任一侧阈值后自动生成候选预览；按钮 `Ctrl+P` 可立即重算。
- 候选预览整体替换画布视图，但不会写入已提交结果。只有“确认提交候选版本”或 `Ctrl+Enter` 才写账本并生成版本。
- “逐帧评估明细”包含每个修复帧/关节的误差、容差和可信标记；完整明细同时导出到审计文件。超过 `2.0°` 或 `0.02 m` 的项显示为红色并标为不可信。
- 右上版本徽标明确区分 `COMMITTED` 与 `CANDIDATE`。超交互预算、快照过期或全量一致性失败时不会静默提交。

## 命令

```powershell
# 生成一档数据
.\.venv\Scripts\python.exe -m mocap_workbench generate --frames 1200 --seed 20261120 --out data/my_sample

# 检测（可重复 --threshold key=value）
.\.venv\Scripts\python.exe -m mocap_workbench detect --bundle data/my_sample --out outputs/detection.json

# 全量修复并导出 detection/repair/evaluation/summary
.\.venv\Scripts\python.exe -m mocap_workbench repair --bundle data/my_sample --keyframe 100 240 --out outputs/repair

# 三档完整验收；报告写到 artifacts/verification_report.json
.\.venv\Scripts\python.exe -m mocap_workbench verify --workspace . --frames all

# 自动化测试
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## 数据与确定性

固定 60 fps、19 关节、四元数 `wxyz`、根位移单位米。记录文件含 `frames[]`，每个帧有 19 个关节旋转与 `root`；truth 使用独立 schema 与独立文件；缺陷清单含 `frame / joint / type / segment_id`。

规范化 JSON 使用排序键、紧凑分隔与 9 位小数；每个文件含内容 SHA-256。相同种子和帧数两次生成、相同记录与阈值两次检测、相同关键帧集合与缺陷清单两次修复，均按字节比较通过。检测证据与阈值解耦：阈值编辑只重新分类，不重算全部四元数测量。

详细格式、公式和算法见 `docs/TECHNICAL.md`。

## 核心机制

- **检测**：无效范数直接判 `missing_marker`；局部 SLERP 中点残差、相邻角差与相对旋转二阶差分识别跳变/抖震；根轨迹中点残差识别滑移。段级峰值归因消除单帧跳变与有限抖震块两侧的过渡误报。
- **修复**：缺陷区间两端自动锚点 + 用户关键帧；旋转用最短路径 SLERP，根用分段线性。所有干净帧原样复制，锚点帧不改写。
- **评估**：逐坐标夹角误差/根误差，区间最大、RMS、原始/修复/真值角加速度峰值与改善比例；超容差逐项标记不可信。
- **闭包**：阈值或关键帧键变化后，按关节比较缺陷状态与最近锚点对，仅重算变化坐标；实际重算集合必须与闭包完全相等，否则 `E_CLOSURE_MISMATCH`。
- **一致性**：每次增量候选都做全量重算，逐帧逐关节比较；不一致时定位首个帧/关节并输出阶段、锚点端点与参数版本。
- **并发**：后台线程基于不可变快照生成候选，单指针整体替换；基线版本已变时返回 `E_SNAPSHOT_STALE`。
- **持久化**：内容寻址 blob 先写，`current.json` 最后原子替换作为唯一提交点；正常退出写关闭标记，强制终止后记录不干净退出并回收未引用孤儿。

## 验收结果

本仓库已在 Linux/WSL 的 Python 3.11、numpy 2.2.6、PySide6 6.9.3 环境执行三档完整验收。详细机器可读结果见 `artifacts/verification_report.json`。

| 档位 | 缺陷真值/检出 | 漏检/误检 | 最大角误差 / RMS | 最大根误差 | 平滑改善 | 阈值增量脏坐标 | 关键帧增量脏坐标 | 全量一致（分量/根） | 恢复 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1200 | 57 / 57 | 0 / 0 | 0.3734° / 0.1617° | 0.0000338 m | 99.883% | 10 | 1 | 0 / 0 | 通过 |
| 6000 | 190 / 190 | 0 / 0 | 0.2599° / 0.0785° | 0.0000336 m | 99.942% | 72 | 13 | 0 / 0 | 通过 |
| 12000 | 332 / 332 | 0 / 0 | 0.5211° / 0.1495° | 0.0000365 m | 99.883% | 69 | 1 | 0 / 0 | 通过 |

三档总验收耗时约 `57.3 s`；单档约 `3.4 s / 17.5 s / 36.3 s`。阈值/关键帧候选预览实测为 `51.9/61.2 ms`、`354.0/305.0 ms`、`868.6/837.7 ms`，均低于对应预算 `900/1800/3200 ms`。1200/6000/12000 的默认修复最大角误差均低于 `2.0°`，最大根误差均低于 `0.02 m`，总体状态为 `TRUSTED`。阈值压力变更后的检测不再是“真值精确匹配”，但增量与全量姿态仍逐帧零分量偏差；该状态会正确显示检测误差并保持闭包外哈希不变。

测试覆盖：生成/检测/修复逐字节复现、四类检测零漏检零误检、干净帧不改写、可信容差与不可信标记、阈值增量、自动锚点移动增量、闭包外姿态/指标哈希不变、增量与全量逐帧一致、正常与强制终止恢复、孤儿回收、后台过期快照拒绝、交互预算错误码。GUI 另以 `QT_QPA_PLATFORM=offscreen` 完成启动冒烟。

## 业务错误码

`E_FRAME_MISSING`、`E_THRESHOLD_RANGE`、`E_KEYFRAME_INVALID`、`E_QUAT_INVALID`、`E_CLOSURE_MISMATCH`、`E_SNAPSHOT_STALE`、`E_BUDGET_EXCEEDED`。命令行以 JSON 输出错误和上下文；GUI 用业务错误对话框显示。

## 目录

```text
mocap_workbench/     业务、CLI、PySide6 工作台
tests/               核心与恢复自动化测试
docs/TECHNICAL.md    指标口径、闭包、并发与事务设计
data/                生成记录（运行后）
runtime/             状态账本、日志、验收中间数据（运行后）
outputs/             人工导出快照（运行后）
artifacts/           验收报告
```

## 资源建议

1200 帧用于快速全流程；6000/12000 帧会生成更大的 JSON 与账本。验收环境峰值常驻内存约 400 MB，工作区磁盘占用由三档原始记录、truth、检测明细、修复快照和账本共同决定，建议预留至少 2 GB。所有耗时均记录在验收报告与 `runtime/logs/interaction/`。
