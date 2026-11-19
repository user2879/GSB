# 本地确定性版面文字检测识别工作台

本项目从零实现一个完全离线的确定性合成版面 OCR 工作台。输入是固定分辨率的 PNG 版面图、同名真值 JSON、识别配置 JSON 和在线编辑命令流 JSON；输出逐命令的增量识别结果、依赖级缓存复用/重算清单、点预算队列、原子快照和一致性诊断报告。

实现不依赖 Docker、云服务或网络资源。运行期只读取本地文件，使用 `numpy`、`Pillow`、`pytest` 和 Python 标准库。

## 快速开始

### Linux / macOS

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'

layout-workbench generate-samples --data-dir data
layout-workbench run \
  --data-dir data \
  --config config/recognition.json \
  --commands config/commands.json \
  --work-dir artifacts/run \
  --base-dir .
layout-workbench validate \
  --data-dir data \
  --config config/recognition.json \
  --commands config/commands.json \
  --work-dir artifacts/validation \
  --base-dir .
pytest -q
```

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"

layout-workbench generate-samples --data-dir data
layout-workbench run --data-dir data --config config/recognition.json --commands config/commands.json --work-dir artifacts/run --base-dir .
layout-workbench validate --data-dir data --config config/recognition.json --commands config/commands.json --work-dir artifacts/validation --base-dir .
pytest -q
```

仓库中的 `data/` 已经包含由固定种子生成的样例；重复执行 `generate-samples` 会得到逐字节一致的 PNG 和真值 JSON。

## 目录

```text
config/
  recognition.json          默认识别配置
  commands.json             8 条编辑命令
  recognition_budget.json   小预算配置
  commands_budget.json      预算演进命令
data/
  sample_a.png / .truth.json
  sample_b.png / .truth.json
  sample_c.png / .truth.json
  replacements/             替换图像及其真值
src/layout_workbench/
  loading.py                PNG/真值校验与灰度解码
  font.py                   内置点阵字模与版本号
  detection.py              滑窗检测与行聚合
  segmentation.py           字符切分、窄片段合并、宽片段二次切分
  matching.py               四种匹配度量、并列决断、候选溯源
  cache.py                  内容寻址阶段缓存
  pipeline.py               四阶段调度、线程池、预算、报告
  commands.py               命令流与图像替换/删除
  snapshot.py               单文档原子快照与缓存审计
  compare.py                增量/全量逐字段比较和首次分歧定位
  validation.py             命令序列一致性验证
  runner.py                 增量命令执行器
  cli.py                    命令行入口
tests/                      21 个 pytest 用例
```

## 输入与配置

每张 PNG 必须有 `<stem>.truth.json`。真值结构如下：

```json
{
  "image_size": [360, 220],
  "lines": [
    {
      "bbox": [35, 42, 134, 63],
      "chars": [
        {"bbox": [35, 42, 50, 63], "text": "H"}
      ]
    }
  ]
}
```

加载层会校验 PNG 签名、IHDR、解码尺寸、像素模式、固定分辨率，并把图像转换为 `uint8` 灰度数组。真值校验要求所有边界框为整数、位于图像和所属行内、同行字符按 `x0` 严格递增且水平不重叠。失败时写入失败清单，业务码为 `E_TRUTH_INVALID`；PNG 错误为 `E_IMAGE_INVALID`。

主要配置字段：

| 字段 | 作用 |
|---|---|
| `window_scales`, `window_step` | 滑窗尺度与步长 |
| `binarization_threshold` | 灰度二值化阈值 |
| `stroke_density_threshold`, `edge_strength_threshold` | 窗内笔画密度/边缘强度门槛 |
| `line_merge_projection_gap` | 行候选 y 投影聚合及 x 投影间距 |
| `line_connectivity_min_pixels` | 行列连通/去孤立噪声条件 |
| `candidate_score_threshold` | 行候选得分门槛 |
| `min_char_height`, `char_min_width`, `char_max_width` | 行与字符尺寸约束 |
| `fragment_merge_gap`, `fragment_merge_preference` | 窄片段左右邻合并规则 |
| `match_metric` | `l1`、`xor`、`cosine`、`ink_balance` |
| `top_k_candidates` | 每字符保留的候选模板数 |
| `thread_pool_workers` | 线程池上限 |
| `char_match_point_budget` | 单批字符匹配点预算 |
| `snapshot_commit_interval_commands` | 快照提交间隔命令数 |

命令流支持：

- `set_threshold`
- `set_window`
- `set_line_merge`
- `set_char_width_constraints`
- `set_match_metric`
- `set_point_budget`
- `set_snapshot_interval`
- `replace_image`
- `delete_image`

## 检测、切分与识别

1. **检测**：对每个尺度在积分图上扫描窗口，统计二值笔画密度、边缘密度和暗度，写入候选窗口块。窗口块按稳定键排序，不依赖扫描线程顺序。
2. **行聚合**：先按 y 投影间距聚合窗口块，再按 x 投影间距形成行候选。边界框用真实二值像素收边，行列计数低于连通门槛的孤立列/行会被去除。低于得分门槛或低于最小字符高度的行进入 `suppressed_lines`，记录得分、尺度与原因，不再进入后续阶段。
3. **字符切分**：沿行内水平投影形成初始片段。窄于 `char_min_width` 的片段按最近左/右邻合并，记录规则、方向、间隙、两侧原始宽度区间和合并结果。宽于 `char_max_width` 的片段在最低投影谷处二次切分，记录切点、谷值比例和两个结果区间。`classification` 区分 `normal_intra_glyph_gap`、`forced_swallow`、`normal_projection_valley`、`forced_width_split`。
4. **模板识别**：字模版本为 `dotmatrix-5x7-v1`。字符块做保持长宽比的最近邻规范化后，与 39 个内置模板逐一比较。每个模板比较消耗 1 个匹配点。得分并列时按 Unicode 编码升序选择；每个字符保留前 `top_k_candidates` 名、得分和 `selected_highest_score`、`tie_lost_lower_codepoint`、`lower_score` 等淘汰原因。

样例中的关键修正记录：

- `sample_a` 的两处双引号被投影切成两个 3px 片段，按规则合并为 9px，记录为 `normal_intra_glyph_gap`。
- `sample_b` 的 42px 宽桥接片段在投影谷处切回 W/O 两个字符，记录为 `normal_projection_valley`。
- `sample_c` 的 1px 噪点被并入邻字符，记录为 `forced_swallow`，可与正常修正区分。
- `l1` 切换到 `ink_balance` 后，同一 `[56,42,71,63]` 字符块从 `E/0.998264` 变为 `D/0.998646`；D/E 并列按编码选择 D，E 被标记为 `tie_lost_lower_codepoint`。

## 增量缓存与失效

缓存对象位于 `cache/objects/<前缀>/<key>.json`，写入采用临时文件加 `os.replace`。每个对象包含缓存键、阶段、图像内容指纹、依赖、字模版本、产物摘要和完整阶段产物。

除业务参数外，缓存键还包含 `PIPELINE_CACHE_VERSION`。算法实现升级时必须提升该版本，旧阶段产物会自动失配，避免把旧坐标或旧字段语义误当作可复用结果。

缓存键的依赖关系：

```text
detection        = image_fingerprint + detection_params + font_version
line_aggregation = detection_key + image_fingerprint + line_params + font_version
segmentation     = line_key + line_product_digest + segmentation_params + font_version
recognition      = segmentation_key + segment_bbox + recognition_params + font_version
```

因此参数变化时，即使本次输出恰好相同，受影响阶段也会按规定失效，而不是仅比较产物内容后误复用。

| 变化 | detection | line_aggregation | segmentation | recognition |
|---|---|---|---|---|
| 阈值/滑窗 | 重算 | 重算 | 重算 | 重算 |
| 行合并间距/连通/候选门槛 | 复用 | 重算 | 重算 | 重算 |
| 字符宽度约束 | 复用 | 复用 | 重算 | 重算 |
| 匹配度量/候选数 | 复用 | 复用 | 复用 | 重算 |
| 替换某图 | 该图重算 | 该图重算 | 该图重算 | 该图重算 |
| 删除某图 | 删除该图 | 删除该图 | 删除该图 | 删除该图 |
| 点预算 | 复用 | 复用 | 复用 | 只调整调度队列 |

报告中的 `cache.per_image[*].stages` 逐图逐阶段给出 `reused`/`recomputed`/`pending` 状态，`cache.reused` 与 `cache.recomputed` 是稳定排序后的清单。验证器会按命令类型断言上述依赖计划，确认未失效阶段没有重算。

## 点预算、快照与并发

匹配调度按 `(image_name, line_index, segment_index)` 稳定顺序消耗预算。一次字符匹配所有 39 个模板需要 39 点。预算不足后，后续字符全部进入稳定顺序的待匹配队列，并在字符、图像和全局队列中标记 `E_POINT_BUDGET_PENDING`。预算按本次批次的逻辑调度额度计算：缓存命中不重算，但仍在稳定调度包络中占位。这样冷缓存全量执行和热缓存增量执行会得到相同的前缀、待匹配队列与剩余额度；报告同时给出实际重算点数和复用节省点数。

快照是一个完整的 `current.json` 文档，包含全局版本、前一版本、配置哈希、字模版本、命令编号，以及四阶段的阶段名、配置哈希、字模版本、已完成图像、已消耗点数、产物摘要和阶段全局版本。版本变化时构建整个新文档并通过临时文件原子替换，不做逐阶段文件覆盖。验证器还会按快照中的缓存键审计缓存对象是否齐全。

线程池只并行检测、行聚合、切分阶段任务和逐字符匹配任务。每个任务携带稳定排序键，所有结果在主线程按键回收并排序。报告记录配置并发数、观测到的最大并发数、是否超过上限，以及单线程与多线程的语义一致性结论。

## 一致性验证

`layout-workbench validate` 会对初始状态和每个命令后状态分别执行：

1. 使用持久缓存的增量运行；
2. 使用全新临时缓存的冷全量运行；
3. 逐字段比较两份语义投影；
4. 若不同，定位首个分歧的图像、行号、阶段和字段；
5. 独立断言缓存依赖失效计划；
6. 验证最终快照和新进程式快照加载；
7. 用同一缓存恢复运行并比较；
8. 用 `thread_pool_workers=1` 冷跑最终状态并比较。

比较字段包括图像/行/字符顺序、失败清单、抑制行、行框与分数、切分合并记录、逐行文本、逐字符框/文本/分数/候选排名、待匹配队列、预算调度额度和剩余量。计时、缓存命中元数据和快照版本属于运行局部状态，不参与语义字段相等判断；缓存复用/重算本身由上述依赖计划断言单独验证。冷全量参考从空缓存开始，按定义会重算增量运行所复用的阶段，因此不能把两者的实际命中清单当作同一运行局部量比较。

## 实际验证结果

在本工作区的 Python 3.11 虚拟环境中执行结果如下。

### pytest

```text
21 passed
```

覆盖 PNG/真值校验、失败清单、滑窗检测、抑制行、窄片段合并、宽片段切分、错误吞并、四阶段失效、图像替换、度量切换、并列决断、点预算、待匹配队列、线程数一致性、快照整文档替换、命令序列增量/全量一致性和故障首次分歧。

### 主编辑序列

`config/commands.json` 依次执行阈值、滑窗、行合并、宽度约束、度量切换、替换 `sample_b`、再次调整宽度和删除 `sample_c` 共 8 条命令。

```text
validation.passed = true
semantic_equivalence.all_steps_equal = true
invalidation.all_steps_match_dependency_plan = true
first_overall_divergence = null
snapshot_verification.valid = true
final_snapshot_version = 5
snapshot_verification.stage_count = 4
snapshot_restore.semantic_equal = true
concurrency_equivalence.semantic_equal = true
```

### 点预算序列

`config/recognition_budget.json` 使用 195 点启动，随后变为 234 点并切换度量：

```text
step 0: consumed=195, pending=20
step 1: consumed=234, pending=19, reused_points=195, recomputed_points=39
step 2: consumed=234, pending=19
validation.passed = true
final_snapshot_version = 3
```

两种预算场景下，增量与冷全量语义一致，所有待匹配项均带有 `E_POINT_BUDGET_PENDING`，快照缓存审计通过。

可重复运行：

```bash
layout-workbench validate --data-dir data --config config/recognition_budget.json \
  --commands config/commands_budget.json --work-dir artifacts/validation_budget --base-dir .
layout-workbench snapshot-check --snapshot-dir artifacts/validation/snapshots \
  --config config/recognition.json --cache-dir artifacts/validation/cache
```

## 报告结构

每次运行输出一个 JSON 报告，核心字段包括：

- `images[*].lines[*].text` 与逐字符结果；
- `segmentation_records` 与 `initial_fragments`；
- `suppressed_lines`；
- `pending_match_queue` 与 `budget`；
- `cache.reused`、`cache.recomputed`、逐图逐阶段状态；
- `snapshot_version`；
- `fault_hits`；
- `timings_ms`；
- `concurrency.conclusion`；
- `failures`；
- `diagnostics`。

故障注入通过配置项 `fault_injections` 提供，支持 `drop_last_line`、`reverse_character_order`、`force_l1_metric`。每条命中记录包含稳定编号、故障类型、阶段和图像；验证器遇到注入故障时会报告首个分歧位置。
