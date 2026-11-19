# 字幕排版与无障碍工作台 — 技术说明

## 1. 环境与运行

- 运行时：Python 3.11+，依赖仅 `numpy`、`PySide6`（见 `requirements.txt`），无外部服务或音视频素材。
- 虚拟环境：本机 Debian 镜像缺 `ensurepip`，使用 `get-pip.py` 在 venv 内引导 pip；
  Windows 上直接 `py -3.11 -m venv .venv` 即可，脚本见 `run_demo_windows.ps1`、`run_gui_windows.ps1`。
- 离屏演示：`QT_QPA_PLATFORM=offscreen`，无需显示器；桌面交互运行 `python -m subtitle_workbench.app`。

```powershell
.\run_demo_windows.ps1                 # 建 venv、装依赖、跑演示与全部测试
.\run_gui_windows.ps1                  # 打开交互窗口（真实桌面）
```

## 2. 模块结构

| 模块 | 职责 |
| --- | --- |
| `subtitle_workbench/models.py` | 不可变值对象、确定性 JSON（排序键、固定分隔符）、SHA-256 内容哈希 |
| `subtitle_workbench/sample.py` | 固定种子（默认 20261119）合成 90s/30fps 时间轴、9 个镜头、24 条字幕与逐条真值清单 |
| `subtitle_workbench/wrapping.py` | 确定性字宽表、词边界优先折行、字号档位降档、`E_WRAP_IMPOSSIBLE` |
| `subtitle_workbench/layout.py` | 全序避让求解、AD/对白声明轨道带、让位裁决、安全区几何、拖动校验三错误码 |
| `subtitle_workbench/snapshot.py` | 不可变快照、tmp+`os.replace` 原子提交、版本指针、哈希校验、重启恢复、撤销 |
| `subtitle_workbench/controller.py` | 拖动会话与候选预览（不污染已提交态）、提交/取消/撤销 |
| `subtitle_workbench/gui.py` | PySide6 时间轴轨道视图、即时预览、红色冲突提示、Enter 提交/Esc 取消 |
| `subtitle_workbench/ass_export.py` | ASS 导出（标准段 + `[SWB_META]` 精确帧 JSON）与只读独立核对器 |
| `demo.py` | 端到端演示：生成→排版→合法/非法拖动→撤销→导出→核对→重启恢复 |
| `tests/` | 30 个 unittest，覆盖全部 8 项业务要求、安全区几何与离屏 GUI 交互 |

## 3. 关键设计与确定性保证

### 3.1 逐字节一致
- 唯一随机源是 `numpy.random.default_rng(seed)`，决定镜头抖动、帧抖动与文案选择。
- 所有落盘数据走 `canonical_json`：`sort_keys=True`、固定分隔符 `(",",":")`、
  `ensure_ascii=False`，不写入时间戳、进程 ID 或主机名。
- 快照文件名仅含单调版本号；同版本内容在任意机器重算哈希相同。
- 验证：两次独立进程 `diff -r run_a run_b` 无差异。

### 3.2 轨道避让（要求 2）
- 分配顺序为全序键 `(priority, start_frame, end_frame, item_id)`，无并列、可复算。
- 轨道几何：AD 声明带为顶部 2 条（lane 0-1），对白声明带为 lane 2-5，
  溢出带 lane 6-7；同帧两条字幕绝不会落在同一 lane（测试逐对断言）。
- 槽位间距固定为单行 `lane_height`（与渲染行数解耦，保证 8 条轨道全部落入
  1080 安全区），多行 cue 在槽位内底边对齐并经 `cue_y` 夹紧；每条结果给出
  `lane_index/x/y/width/side_margin`，测试断言所有 cue 均在安全区内。

### 3.3 折行与字号（要求 3）
- 字宽来自确定性字宽表（CJK=100、拉丁=56、数字=55、空格=28，按字号线性缩放），
  不依赖宿主字体，headless 结果与桌面一致。
- 贪心装箱优先词边界；档位 `(48,42,36,32,28)` 从大到小，首个满足行数的档位胜出；
  最小档仍超行数时允许硬折行但保证不超宽，连最小档都超宽则返回 `E_WRAP_IMPOSSIBLE`。

### 3.4 无障碍差异化与让位裁决（要求 4）
- AD 使用独立顶部轨道带、金黄样式/加粗/顶对齐（ASS `Alignment=8`），
  最短显示 90 帧（对白 45 帧），样本生成器强制满足该时长。
- 声明带满时按固定规则裁决：AD 永远向对白让位，同类型按全序键后者让位，
  胜者、原因（`ad_yields_to_dialogue` / `declared_band_full_overflow`）、
  声明/实际 lane 全部写入 `arbitrations`；任何条目都不会被静默丢弃。

### 3.5 拖动、拒绝与快照（要求 5-7）
- 拖动只产生 `DragPreview` 候选；已提交 items 在会话期间不可变。
- 三类确定性拒绝：越界 `E_DRAG_BOUNDS`、`start>=end` `E_TIME_INVALID`、
  与高优先级不可让位条目同轨 `E_TRACK_CONFLICT`，均带 `conflict_item_id`，
  拒绝后版本号与内容保持不变（测试断言）。
- 提交走临时文件 + `os.fsync` + `os.replace`；`current.json` 指针与快照同样原子替换。
- 缓存以 `layout_hash` 为键，旧版本哈希不匹配即 `E_STALE_VERSION`，杜绝旧缓存复用。
- 撤销只改指针、不改写不可变快照；模拟强杀后新进程仅依据指针+哈希恢复，
  指针悬空或哈希被改时拒绝恢复，绝不伪造状态。

### 3.6 ASS 导出与独立核对（要求 8）
- ASS 标准段：每 cue 一个 `Dialogue`，`Layer`=lane、`MarginV`=精确 Y、
  每 lane/类型一个命名 `Style`（字号、颜色、边距、对齐）。
- `[SWB_META]` 段：每 cue 一行规范化 JSON，保存精确帧区间、几何与折行结果，
  规避厘秒取整导致的帧误差。
- 核对器只读导出文件与快照文件，逐字段比对 15 项（含事件段与 META 段），
  定位首个不一致条目与字段；条数相等不判通过，交换两条 lane 也会被检出。

## 4. 实际验证结果

- `python -m unittest discover -s tests`：**30/30 通过**（含 4 个 offscreen GUI 用例）。
- `demo.py` 端到端：24 条字幕、9 镜头、4 路重叠簇、跨镜头、7 条 AD、3 处长文本降档；
  三类错误码全部按预期拒绝；撤销后版本回到 1；导出核对 24/24 匹配；重启恢复哈希一致。
- 两次独立运行输出目录 `diff -r` 为空（逐字节相同）。
- 离屏截图：`artifacts/workbench_offscreen.png`。

## 5. 已知边界

- 字宽为确定性模型而非字体实测，保证可复算；接入生产字体时可在 `wrapping.py`
  单点替换 `glyph_width`，接口与契约不变。
- 安全区物理轨道耗尽（超过 8 条同帧且均不可让位）时显式抛 `E_TRACK_CONFLICT`，
  这是声明的失败路径而非静默丢条。
