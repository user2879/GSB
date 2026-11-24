# mmwin：确定性流式多模态事件窗口聚合

这是一个完全本地、无外部服务的 Python 3.11+ 示例组件。图像帧、结构化文本断言和传感器读数按**注入计划中的到达序号**交错投递；组件维护事件时间水位线、滑动窗口、迟到再结算、定点融合摘要、内容缓存和原子检查点。

运行时唯一的第三方依赖是 `numpy`。PNG 的读写由项目内的小型标准库实现完成，不依赖 Pillow、数据库、消息队列或容器。

## 快速开始（Windows PowerShell）

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install numpy

python tools\generate_samples.py --samples samples --seed 20261124
python -m mmwin run `
  --samples samples `
  --config samples\window_config.json `
  --plan samples\injection_plan.json `
  --checkpoint .mmwin-checkpoint `
  --output out\report.json `
  --reset
```

也可以安装本仓库的 CLI：`pip install -e .` 后使用同一组参数的 `mmwin run ...`。
不安装时 `python -m mmwin` 与 `python tools\generate_samples.py` 均可直接运行。

Linux/macOS 下把续行符改为 `\` 即可：

```bash
python3 tools/generate_samples.py --samples samples --seed 20261124
python3 -m mmwin run --samples samples \
  --config samples/window_config.json \
  --plan samples/injection_plan.json \
  --checkpoint .mmwin-checkpoint \
  --output out/report.json --reset
```

不带 `--reset` 再次执行就是增量续跑。检查点的计划哈希、来源哈希、配置哈希均匹配时从已提交的到达位置续跑；配置或状态版本不匹配时输出 `E_CHECKPOINT_VERSION_MISMATCH` 并从到达序号 0 全量重放。

## 目录

```text
mmwin/                      核心组件和 CLI
  config.py                 配置加载与校验
  events.py                 图像/文本/传感器事件加载与校验
  plan.py                   乱序注入计划
  png.py                    无 Pillow 的 PNG 编解码
  features.py               视觉摘要、证据向量、定点融合
  engine.py                 回放、水位线、窗口、缓存、调度
  checkpoint.py             分阶段检查点与原子代际提交
  generate.py               确定性样本生成
  cli.py                    python -m mmwin
tools/generate_samples.py   样本生成入口
samples/                    已生成的本地样本
tests/test_mmwin.py         完整验收测试
```

## 输入

### 图像帧

`samples/frames.jsonl` 是 PNG 帧元数据 sidecar，`path` 相对于样本目录：

```json
{"event_id":"img-0000","timestamp_ms":60,"frame_index":0,"path":"frames/frame_0000_t00000060.png"}
```

事件时间戳来自 sidecar，不从 PNG 文件时间推断。所有帧必须是同一固定分辨率；默认样本是 128x96 RGB。没有 `frames.jsonl` 时，也会发现符合 `frame_<序号>_<毫秒>.png` 的文件。

### 文本断言

```json
{"event_id":"txt-0000","timestamp_ms":90,"entity":{"id":"vehicle-00","type":"vehicle"},"attribute":"lane_state","value":"clear","confidence":0.92,"polarity":0.31}
```

`entity.id`、`entity.type`、`attribute` 必须完整；`value` 是有限标量；`confidence` 和 `polarity` 可选且必须有限。

### 传感器读数

```json
{"event_id":"sen-0000","timestamp_ms":70,"channel":"temperature","value":20.4}
```

`channel` 必须属于配置中的 `sensor_channels`，`value` 必须是有限数。

### 配置

完整样例为 `samples/window_config.json`。核心字段：

| 字段 | 含义 |
| --- | --- |
| `window_length_ms` / `window_step_ms` | 半开窗口 `[start, end)` 的长度与步长；长度必须被步长整除 |
| `allowed_lateness_ms` | 水位线回退量，必须是非负整数 |
| `watermark_rule` | 固定规则 `max_seen_minus_lateness` |
| `late_policy` | `resettle` 或 `drop` |
| `visual_feature_dim` / `text_feature_dim` | 视觉投影维度与文本/传感器证据维度 |
| `visual_block_rows` / `visual_block_cols` | 每帧分块网格 |
| `visual_frames_per_window` | 每窗口取离窗口中心最近的帧数 |
| `quantization_level` / `fractional_bits` | 当前量化档位与定点小数位宽；档位值必须等于位宽 |
| `fusion_temperature` | 定点 softmax 融合温度，必须大于 0 |
| `thread_pool_size` / `concurrency_limit` | 实际线程数和硬上限 |
| `max_events_per_window` | 窗口事件数上限 |
| `checkpoint_commit_interval_windows` | 每累计闭窗多少提交一次中间检查点 |
| `batch_event_budget` | 默认单批事件预算 |
| `sensor_channels` | 通道枚举及其归一化 `min`/`max` |

完整配置指纹用于检查点兼容判定；**语义配置指纹**排除了线程数、并发上限、检查点间隔和批预算，因此把 `thread_pool_size` 从 1 改成 6 不会改变视觉整数、融合或窗口结论。

### 乱序注入计划

```json
{"schema_version":1,"seed":20261124,"arrivals":[{"arrival_seq":0,"event_id":"sen-0011","delay_ms":911}]}
```

到达按 `arrival_seq` 排序后逐条投递，`delay_ms` 只记录注入的到达延迟。回放绝不按事件时间戳重排。

额外样本：

- `samples/injection_plan_duplicate.json`：末尾重复投递一个事件，用于触发 `E_EVENT_INVALID`。
- `samples/window_config_drop.json`：迟到事件丢弃策略。
- `samples/window_config_q12.json`：1024 定点档位。

## 固定语义规则

1. 窗口起点是 0 的整数倍步长；窗口是半开区间 `[start, end)`。
2. 水位线为已接受投递事件的最大时间戳减去允许迟到毫秒。没有投递事件时为 `null`，可以小于 0。
3. 当 `window_end_ms <= watermark_ms` 时窗口关闭。
4. 当前事件迟到量为 `max_seen_before - event_timestamp_ms`。**仅当迟到量严格大于允许迟到时判为迟到；恰好等于边界时按准时处理。**
5. 窗口内事件引用按 `(timestamp_ms, event_id)` 排序；这不会改变实际投递顺序。
6. 迟到 `resettle` 会加入事件并生成新的窗口快照；`drop` 不改变成员，但会记录旧/新结论相同的审计项。
7. 每窗口事件数超过上限时，窗口状态为 `overflow`，不输出置信度，并产生 `E_WINDOW_OVERFLOW`。
8. 批预算按到达逐条消耗。预算耗尽后剩余到达留在待办队列，每个待办事件产生 `E_EVENT_BUDGET_PENDING`；下一次运行从该前缀继续，待办优先。

## 视觉与定点融合

每个窗口：

1. 在窗口覆盖的事件中选离窗口中心最近的 `visual_frames_per_window` 帧，再按帧序号升序处理。
2. 每帧按块计算 RGB 均值、梯度强度均值、边缘密度，形成 5 个值/块。
3. 将原始特征定点化，用由语义配置指纹和量化档位派生的固定整数投影矩阵映射到 `visual_feature_dim`。
4. 文本与传感器事件生成稀疏有符号证据向量；视觉向量与其叠加。
5. 融合 logits 先减去最大值，再做稳定 softmax；置信度、logits、文本/传感器/视觉贡献均为定点整数。

量化档位改变会改变缩放、投影矩阵种子和中间整数，不只是格式化输出。空窗输出 `empty_visual=true`，视觉向量为全零。

## 输出报告

`--output out/report.json` 生成完整报告，主要字段：

- `delivery_order`：严格等于注入计划的逐条投递轨迹。
- `watermark`：最终水位线、最大事件时间及边界规则。
- `windows`：按窗口起止排序的结论、事件引用、视觉摘要、置信度、贡献分解和各阶段缓存键。
- `resettlements`：迟到窗口、到达序号、迟到量、策略、旧/新快照和旧/新结论。
- `empty_windows`：无帧事件的窗口编号。
- `failures`：按窗口起止、事件 ID、到达序号、错误码稳定排序的失败清单。
- `invalidation_manifest`：仅含窗口、事件引用和阶段内容键，是跨增量/全量可比较的精确失效清单。
- `cache.stats` / `cache.window_manifest`：命中/计算遥测；命中状态属于运行遥测，不参与业务等价比较。
- `checkpoint`：全局版本、配置/计划/来源哈希、已处理到达、待办序号和阶段版本。
- `diagnostics`：如 `E_CHECKPOINT_VERSION_MISMATCH`、`E_CHECKPOINT_PLAN_CHANGED`。

## 增量缓存与检查点

缓存分阶段内容寻址：

- 窗口归属键：窗口起止、全部引用事件指纹、语义配置。
- 视觉键：窗口起止、帧事件指纹、选中帧集合、视觉配置和量化档位。
- 融合键：全部事件引用指纹、视觉键、融合配置和量化档位。
- 再结算键：窗口快照语义链和迟到决策链，不依赖全局快照序号。

因此替换一个文本事件时，未引用该事件的窗口直接复用归属、视觉、融合和再结算产物；引用它的窗口只重算变化的下游阶段。删除事件不会因为快照序号移动而误伤无关窗口。

检查点目录：

```text
.mmwin-checkpoint/
  MANIFEST.json
  generations/
    g00000002-<id>/
      replay.json membership.json visual.json fusion.json
      resettle.json pending.json runtime.json cache.json
```

一次提交先把全部阶段、水位线、待办和缓存写入同盘临时目录，再原子改名成新代际目录，最后原子替换 `MANIFEST.json` 指针。清单记录全局版本、配置哈希、计划哈希、来源哈希、已处理到达序号、待办序号及每个阶段文件的内容哈希。

窗口视觉/融合任务在线程池中执行，每个任务携带 `(window_start_ms, window_end_ms, phase, snapshot_id)` 稳定排序键；提交顺序和最终合并顺序固定。失败清单、预算轨迹和窗口顺序与串行结果一致。

## 测试

```powershell
python -m unittest discover -s tests -v
```

测试覆盖：

- 配置、三类事件、重复 ID、非法时间戳、实体、通道、有限数和重复投递校验。
- 乱序投递顺序、水位线公式、边界恰好等于允许迟到、迟到再结算。
- 空窗、量化档位改变中间整数、窗口溢出跳过结论。
- 批预算待办、检查点续跑和阶段清单。
- 乱序计划替换、单事件替换/删除、窗口/迟到/量化编辑序列的增量与空白全量逐字段一致。
- 内容键精确失效、未受影响窗口零重算、线程池串并行一致、配置版本不匹配全量重放。

实际验证摘要见 `VERIFICATION.md`。
