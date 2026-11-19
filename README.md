# 本地字幕排版与无障碍呈现引擎

这是一个从零实现、无外部媒体依赖的确定性字幕排版引擎。输入是 JSON 字幕描述，输出是逐帧灰度字幕画面、无障碍呈现报告、可恢复的块级状态目录，以及带原子提交指针的导出结果。

实现覆盖：

- 字幕事件与样式规则严格校验，失败返回稳定的 `E_xxx` 业务码。
- 词边界自动换行、最大行数检查、阅读速率下限与速率超限报告。
- 锚点/内边距/安全区包围盒、灰度背景遮罩、对比度增强、关键音效视觉标记。
- 样式/安全区依赖图、逐帧可见集合、编辑步骤级失效块与失效区域。
- 固定毫秒块持久化、版本摘要、逐块摘要、原子替换、断点恢复与 `E_DIGEST` 修复。
- 有界线程池、内存预算准入、按块索引等待、首错取消与确定性并发输出。
- 版本化 generation + 原子 `commit.json` 的帧/报告成对发布。

## 环境

- Python 3.10+
- NumPy（`requirements.txt`）
- 不需要网络、GPU、Docker 或媒体工具

通用安装方式：

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

当前工作区已提供可运行的 `.venv`。入口为：

```bash
./subtitle-engine --help
# 等价于
.venv/bin/python -m subtitle_engine --help
```

Windows 下可直接使用 Python 命令，核心文件写入使用同目录临时文件加 `os.replace`，不依赖 POSIX shell 行为：

```powershell
py -3.10 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m subtitle_engine --input samples\generated\main.json `
  --state-dir samples\generated\state\main `
  --output-dir samples\generated\output\main
```

## 快速运行本地样本

```bash
python samples/generate_samples.py

python -m subtitle_engine run \
  --input samples/generated/main.json \
  --state-dir samples/generated/state/main \
  --output-dir samples/generated/output/main \
  --clean-state
```

`--clean-state` 要求显式给出 `--state-dir`，只清理该目录。`state_dir` 与 `output_dir` 必须不同；样本中二者是兄弟目录。再次运行同一命令时不要加 `--clean-state`，即可验证版本一致时的块复用。

主要输出：

- `output/frames.npy`：形状为 `[frame_count, height, width]` 的 `uint8` 灰度帧序列。
- `output/report.json`：无障碍呈现报告。
- `output/commit.json`：当前帧/报告 generation 的原子提交指针、版本、帧数和摘要。
- `output/generations/<version>-<run-id>/`：不可变发布内容；不同版本不会互相覆盖。
- `state/<version>/manifest.json`：版本、块清单和完成状态。
- `state/<version>/blocks/000000.npz`：块帧、元数据、版本和块摘要。
- `state/<version>/interrupted.json`：可处理终止信号在块边界生效后的恢复标记。

## 输入 JSON

完整示例见 `samples/generate_samples.py`。核心结构如下：

```json
{
  "schema_version": "subtitle-engine/v1",
  "duration_ms": 1600,
  "frame": {
    "width": 320,
    "height": 180,
    "frame_ms": 20,
    "background": 18
  },
  "safe_area": {"x": 16, "y": 16, "width": 288, "height": 144},
  "style": {
    "font_size": 12,
    "line_height": 16,
    "max_lines": 5,
    "padding": 6,
    "contrast_enhance": true,
    "mask_opacity": 0.62
  },
  "events": [
    {
      "id": "e1",
      "start_ms": 0,
      "end_ms": 420,
      "text": "System alert: hull pressure rising.",
      "speaker": "COMPUTER",
      "language": "en",
      "max_chars_per_sec": 16,
      "critical_sound": true,
      "anchor": {"x": 160, "y": 160, "horizontal": "center", "vertical": "bottom"}
    }
  ],
  "edits": [
    {"op": "set_text", "event_id": "e1", "text": "New text"},
    {"op": "set_style", "field": "font_size", "value": 14},
    {"op": "set_time", "event_id": "e1", "start_ms": 80, "end_ms": 500}
  ],
  "runtime": {
    "concurrency": 3,
    "memory_budget_bytes": 4194304,
    "block_ms": 100,
    "state_dir": "state/main",
    "output_dir": "output/main"
  }
}
```

字段规则：

- `start_ms`、`end_ms` 必须为非负整数，且 `end_ms > start_ms`。
- `text` 必须为非空字符串；不会被静默截断。
- `max_chars_per_sec` 必须为正整数。
- `critical_sound` 必须为布尔值。
- `font_size`、`line_height` 必须为有限正数。`line_height` 单位是像素。
- `max_lines` 必须为正整数。
- `contrast_enhance` 必须为布尔值，`mask_opacity` 必须位于 `[0, 1]`。
- 锚点可写为 `top_left`、`bottom_center`、`center` 等命名值，或 `{x, y, horizontal, vertical}`。锚点必须位于安全区内。
- `block_ms` 必须是 `frame_ms` 的整数倍；所有块按固定毫秒边界切分。
- 编辑脚本按数组顺序应用。支持 `set_text`、`set_style`、`set_time`。

## 排版与渲染语义

### 换行与阅读速率

- 使用纯代码内置的确定性字符宽度。ASCII、宽字符和空格有固定宽度；同一输入在不同平台得到相同结果。
- 优先在词边界换行；单个超长 token 才按字符边界切分。
- 若完整文本需要的行数超过 `max_lines`，引擎返回 `E_LINE_OVERFLOW`，不输出帧，也不会截断文本。
- 最小显示时长按 `ceil(len(text) * 1000 / max_chars_per_sec)` 计算。
- 当事件时间窗短于最小显示时长时，保留原时间窗，并在报告中标记 `speed_limited=true`、`min_display_ms` 和 `E_SPEED_LIMIT`。

### 帧与时间窗

- 第 `n` 帧的时间戳为 `n * frame_ms`。
- 字幕在 `start_ms <= timestamp < end_ms` 时可见。
- 报告中的 `bbox_start_frame` 取 `ceil(start_ms / frame_ms)`，`bbox_end_frame` 取 `ceil(end_ms / frame_ms)`（结束帧不包含）。
- 可见集合只含时间窗内字幕；空帧严格等于 `frame.background`，不会残留上一帧像素。

### 灰度合成

- 包围盒由锚点、对齐方式、内边距、行数和行宽计算，并校验完全落在安全区内。
- 遮罩值由背景值和 `mask_opacity` 确定；`contrast_enhance=true` 会增强遮罩、文字对比并增加轮廓/边框。
- 关键音效字幕在包围盒左上角绘制固定棋盘标记，并记录在块元数据和报告中。
- 内置 5x7 位图字体；不支持的字形使用稳定码点图案，确保无外部字体依赖且可复现。

## 依赖图与失效判定

每条最终字幕在报告 `subtitles[].dependencies` 中记录字号、行高、最大行数、内边距、对比度、遮罩、安全区和锚点依赖。报告 `dependency_graph` 给出字段到字幕集合的映射。

编辑脚本逐步执行：

- `set_text`：只重算该字幕布局，失效块为旧/新时间窗内该字幕可见块。
- `set_time`：只重算该字幕，失效块为旧窗口与新窗口可见块的并集。
- `set_style`：所有字幕依赖样式布局/合成字段，因此所有字幕受影响；失效块为其可见块并集。
- 每一步的 `affected_event_ids`、`dirty_blocks`、`dirty_regions`、`warning_codes` 都写入 `report.json` 的 `edit_steps`。

块元数据保存 `frame_visible_event_ids`，即每个帧区间的可见字幕集合。未失效块在相同版本恢复时直接复用。

## 版本、状态与恢复

版本标识是规范化 SHA-256，覆盖：

- 引擎格式版本；
- 帧尺寸、帧间隔、背景值；
- 安全区与初始样式；
- 字幕事件清单；
- 有序编辑脚本；
- 最终应用编辑后的状态与最终时长；
- 块大小。

状态目录按完整版本分目录，因此新版本不会覆盖旧版本：

```text
state/
  <version-a>/manifest.json
  <version-a>/blocks/000000.npz
  <version-b>/manifest.json
  <version-b>/blocks/000000.npz
```

每个块 `.npz` 包含帧数组、元数据、`version` 和块摘要。写入流程为：

1. 在块目录写 `.tmp.npz`；
2. `fsync`；
3. `os.replace` 到正式块文件；
4. 原子更新 manifest。

未完成临时文件不会被块读取器识别。重启时逐块验证版本、形状、声明帧数、元数据和摘要；不匹配块被移到 `.invalid` 并补算。

### 可处理信号

- `SIGINT` / `SIGTERM` 处理函数只设置停止标记。
- 停止只在块边界生效；已在执行的块会完成并原子持久化，不再提交后续块。
- 写入 `interrupted.json`，统一返回退出码 `130`。
- 重启只使用已完成且摘要有效的块，不承诺在中断运行中生成报告。
- `SIGKILL` 不写中断标记；重启仍只依赖已持久化且摘要有效的块。

### `E_DIGEST` 修复

所有块完成后逐块回读校验。摘要不匹配或声明帧数不符时：

- 将该块隔离；
- 加 `E_DIGEST` 业务码；
- 回退补算该块并再次校验；
- 修复成功则报告保留 `E_DIGEST`，继续输出；
- 两轮仍不能收敛则返回 `E_DIGEST` 错误，不发布新输出。

## 并发与内存预算

- 使用 `ThreadPoolExecutor`，工作线程数等于 `runtime.concurrency`。
- 主调度器按块索引顺序尝试提交。
- 每块进入计算前按“最终块帧缓冲 + 工作帧 + 元数据/布局开销”估算内存。
- 若加入下一块会超过 `memory_budget_bytes`，后续块等待已有块完成；不会降低换行精度或安全区尺寸。
- 单块自身超过预算时返回 `E_MEMORY_BUDGET`。
- 任一块异常会设置取消标记、取消待执行 future，并按块索引返回首个实际错误。
- 已完成块立即写盘并从内存工作集移除；最终拼装使用 `npy` memmap 按块流式写入，避免把所有块帧同时留在内存。
- 渲染顺序按 `(start_ms, event_id)` 排序，因此并发度 1 和 N 的帧逐像素相同。

## 原子导出

导出协议：

1. 在 `generations/<version>-<run-id>/` 写入 `frames.npy` 和 `report.json`；该目录在提交前不被引用。
2. 计算两个文件摘要。
3. 最后原子替换 `commit.json`，其中包含版本、帧数、两个摘要和 generation 路径。
4. 更新 `frames.npy` / `report.json` 稳定镜像。
5. 若进程在稳定镜像更新中退出，下一次启动根据 `commit.json` 自动修复镜像。

因此帧序列与报告的正式一致性由 `commit.json` 指向的 generation 保证。相同版本再次运行会刷新运行统计报告；不同版本会生成新 generation，旧 generation 保留。

## 业务码与退出码

错误 JSON 形状：

```json
{
  "ok": false,
  "error": {
    "code": "E_ANCHOR",
    "message": "Screen anchor must lie inside the safe area",
    "details": {"path": "events[0].anchor"}
  }
}
```

| 业务码 | 含义 | 是否阻止输出 |
| --- | --- | --- |
| `E_SCHEMA` | 顶层结构/类型错误 | 是 |
| `E_TIME` | 时间窗非法 | 是 |
| `E_TEXT` | 文本为空或类型错误 | 是 |
| `E_RATE` | 阅读速率上限非法 | 是 |
| `E_ANCHOR` | 锚点越出安全区 | 是 |
| `E_EVENT_ID` | 字幕 id 重复 | 是 |
| `E_STYLE_*` | 样式字段非法 | 是 |
| `E_SAFE_AREA` / `E_FRAME` | 安全区或帧参数非法 | 是 |
| `E_CRITICAL` | 关键音效开关不是布尔值 | 是 |
| `E_EDIT` | 编辑操作非法 | 是 |
| `E_LAYOUT` | 包围盒无法落入安全区 | 是 |
| `E_LINE_OVERFLOW` | 文本无法在 `max_lines` 内完整排下 | 是 |
| `E_MEMORY_BUDGET` | 单块超过内存预算 | 是 |
| `E_DIGEST` | 块摘要/声明帧数不匹配 | 可修复则否，不可修复则是 |
| `E_SPEED_LIMIT` | 阅读速率超限 | 否，报告警告 |
| `E_INPUT` / `E_PATH` / `E_RUNTIME` | 输入、路径或运行时参数错误 | 是 |
| `E_INTERNAL` | 非预期内部错误 | 是 |

CLI 退出码：

- `0`：完成并导出；
- `2`：业务/内部错误；
- `130`：可处理中断在块边界生效，状态可恢复。

## 无障碍报告

`report.json` 包括：

- 版本、帧数、帧间隔、帧尺寸、安全区、块大小和业务码；
- 每条字幕的说话人、语言、最终文本、逐行文本、行宽、最小显示时长、实际时长、速率超限标记；
- 包围盒、起止帧、可见块、依赖和失效/重算块数；
- 每个编辑步骤的受影响字幕、失效块和失效区域；
- 每个块的帧范围、可见字幕集合、关键音效集合、逐帧可见集合和块摘要；
- 复用块数、实际重算块数、失效块数、验证业务码；
- 导出模式与前一版本标识。

## 测试与实际验证

运行：

```bash
.venv/bin/python -m unittest discover -s tests -t . -v
.venv/bin/python scripts/interruption_test.py
```

实际结果（2026-11-19，本机 Python 3.11 / NumPy 2.4.6）：

- 13 个单元/集成测试全部通过。
- 主样本 80 帧、16 块：首次全部计算；同版本第二次运行 `reused=16, recomputed=0`。
- 并发度 1 与 3 的 `frames.npy` 经 `numpy.array_equal` 验证逐像素一致。
- 编辑轨迹中文本、样式、时间窗、样式四步分别失效 `7/16/6/16` 块，证明样式与单字幕编辑粒度不同。
- 人为破坏一个块后重跑：报告包含 `E_DIGEST`，`reused=15, recomputed=1`。
- SIGTERM 测试：退出码 130、无输出帧、重启复用 1 个已完成块并补算 23 块。
- SIGKILL 测试：退出码 `-9`、重启复用 1 个有效持久化块并补算 23 块。

样本生成器 `samples/generate_samples.py` 连续两次运行生成的 JSON 摘要一致。

## 目录

```text
subtitle_engine/       引擎、布局、渲染、状态、导出、CLI
samples/generate_samples.py  确定性本地样本生成器
samples/generated/     生成样本及运行产物
tests/                 校验、布局、失效、摘要、并发、导出、中断测试
scripts/interruption_test.py 可直接运行的 SIGTERM/SIGKILL 恢复测试
```

## 明确的设计边界

- 字体为内置 5x7 位图字体，不尝试复刻平台字体度量；这保证无媒体依赖和跨平台确定性。
- `line_height` 是像素值，不是倍数。
- 阅读字符数使用 Python 字符长度 `len(text)`，包含空格和显式换行。
- `max_lines` 是硬限制；无法完整排版时返回业务错误而不是截断文本。
- 内存预算约束并发计算工作集；最终 `frames.npy` 通过 memmap 流式生成，不把全部帧读入内存。
- 中断退出码统一为 `130`，实际信号编号记录在 `interrupted.json` 和 CLI 摘要中。
