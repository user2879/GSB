# StemScope · 分轨渲染诊断工作台

StemScope 是一个纯本地、确定性的音乐制作诊断工作台。它解析三种声源分离结果包，把分轨统一为带内容指纹的 PCM 中间表示，在有限交互预算内执行可取消等待的重混预览，并保证任意编辑序列后的增量结果与空白全量重渲染逐帧一致。

仅使用 Python 标准库与 NumPy。所有样本按固定种子在本地合成，不访问网络、不使用真实账号或外部音频资源。

## 快速开始

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt

python scripts/generate_samples.py --out sample_data
python -m stemscope run --data sample_data --out artifacts
python -m unittest discover -v
```

`run` 会生成：

- `artifacts/report.json`：完整、可重复的诊断报告。
- `artifacts/dashboard.html`：无外部资源的领域可视化面板。
- `artifacts/mix_final.wav`：最终 16 kHz 单声道 16-bit PCM 混音。
- `artifacts/frame_details.jsonl`：逐帧分轨加权项与最终帧值。
- `artifacts/conversion_records.json`：格式转换与丢失字段账本。

## 样本布局

`scripts/generate_samples.py` 使用固定种子 `20261122` 生成 4 个 750 ms 片段：

```text
sample_data/
  generator_manifest.json
  remix_config.json
  interactions.json
  injection_manifest.json
  segments/seg_000/
    mix.wav
    source/{bass,drums,vocals,alt_vocals}.wav
    packages/v1/stems.wav
    packages/v2/manifest.json
    packages/v2/{bass,drums,vocals}.wav
    packages/v3/stems.v3
```

样本同时覆盖 16 kHz、单声道、16-bit PCM、三条合成分轨、音量包络、替换分轨与三种分离包。

## 三种容器

- **v1**：多声道交错 WAV。默认声道顺序为 `bass, drums, vocals`；可选同名 `.sidecar.json` 保存 `channel_names` 与旁路信息。
- **v2**：`manifest.json` 记录 `sample_rate`、`frame_count`、逐轨 `name/channel/file/frame_count`；独立分轨必须是单声道 PCM16，并逐轨校验采样率、声明帧数与实际帧数。
- **v3**：32 字节定宽头部 `<4sHHIIHHQI>`，随后为每轨 32 字节定宽记录 `<16sHHIII>`。记录包含 UTF-8 分轨名、声道号、绝对偏移、帧数与 CRC32；解析时校验 magic、版本、头部宽度、数据偏移、记录边界、块重叠、帧数与块校验和。

三种解析器统一返回 `ParsedPackage` / `Track`：

- `source_format_version`
- 分轨名与逐帧 `int16` PCM
- 帧数、采样率、最大绝对幅值
- 包内容指纹、来源内容指纹与解析缓存键
- 旁路元数据字节及其指纹

任何解析失败都转为 `E_TRACK_PARSE_FAILED` 并进入稳定排序的失败清单。相同输入重复解析得到相同包指纹与逐帧 PCM。

## 双向转换与信息保留

转换器支持 `v1/v2/v3` 任意源到任意目标。转换后会重新解析目标包并比较逐帧 PCM。

目标格式不能表达的字段不会静默丢失：

- 丢失项写入目标包旁的 `.sidecar.json`。
- 转换记录逐条列出 `field`、`source_track`、来源格式、目标格式和原值。
- 后续包指纹包含旁路元数据字节，因此改动旁路信息会改变下游缓存键。

例如 `v3 -> v2` 会把三轨 `block_checksum_crc32` 写入旁路元数据，并在转换记录中逐轨报告。

## 确定性重混

配置示例：

```json
{
  "sample_rate": 16000,
  "gain_quantization_steps": 1000,
  "render_frames": 2048,
  "workers": 4,
  "interaction_budget_ms": 2000,
  "snapshot_interval_segments": 2,
  "format_version": "v1",
  "tracks": {
    "bass":   {"gain": 0.9073, "muted": false, "start_ms": 0, "end_ms": 750},
    "drums":  {"gain": 0.7531, "muted": false, "start_ms": 0, "end_ms": 750},
    "vocals": {"gain": 1.0137, "muted": false, "start_ms": 0, "end_ms": 750}
  }
}
```

数值规则固定且可复现：

1. 裁剪毫秒用 `Decimal` 半入到帧边界，并夹取到片段范围。
2. 增益用 `Decimal` 按量化档位 `ROUND_HALF_UP` 转为整数：`gain_int = round(gain × steps)`。
3. 每帧每轨加权项是 `int64 PCM × gain_int`；静音与裁剪外帧为零。
4. 求和顺序严格按分轨名字典序。
5. 混音除以量化档位时使用“半远离零”整数舍入。
6. 结果饱和到 `[-32768, 32767]` 后写成 little-endian `int16`。

样本文档使用非量化网格增益，因此切换 1000 与 2000 档时定点整数和输出 PCM 都会真实改变，而不是只改显示值。

## 内容指纹与增量失效

缓存键均由稳定 JSON 规范化后 SHA-256 派生：

- 解析键：格式版本 + 包目录内容哈希。
- 转换键：来源包指纹 + 目标格式 + 来源旁路元数据哈希。
- 增益键：格式版本 + 量化档位 + 分轨参数指纹 + 片段位置 + 分轨 PCM 内容指纹。
- 混音键：格式版本 + 片段位置 + 排序后的各轨增益键。

每个增益与混音条目记录引用分轨集合和稳定排序键 `stage|segment|track`。单轨增益、静音或替换 PCM 只改变该轨上游键；裁剪变化会计算新旧区间的对称影响，只让与受影响帧区间重叠的片段失效；混音片段随其任一上游键变化而下失效。未变化条目直接复用。

## 交互预算语义

命令进入后台串行队列，计算任务仍在配置的线程池内执行。调用线程只等待到预算边界，并为 Python 调度抖动保留小比例安全余量；不会用额外延迟等待来“假装”命中预算。

- 最新参数在预算内完成：返回 `status=completed`、`is_latest=true`。
- 未在预算内完成：返回最近一次已完成摘要，`status=budget_exceeded`、`code=E_INTERACT_BUDGET_EXCEEDED`、`is_latest=false`。
- 旧摘要携带自己的 `param_version`；同时记录 `requested_param_version`，不会把旧结果标成新参数结果。
- 后台任务继续完成；`wait_for_idle()` 后可取得最新参数结果。

## 并发一致性

解析、增益与混音阶段都使用大小取自配置的 `ThreadPoolExecutor`，池大小不得超过配置。所有任务有稳定排序键；完成顺序只影响收集时机，不影响输出顺序。失败清单、片段表、复用与重算清单都按稳定键排序。

## 阶段快照

阶段为 `parse / convert / gain / mix`。每次提交：

1. 在阶段目录内建立临时 generation。
2. 写入条目清单与二进制 blob，并计算阶段产物版本。
3. 通过目录 rename 整体落位，再原子替换 `CURRENT` 指针。
4. 顶层 `manifest.json` 原子写入全局版本、配置哈希、格式版本集合、已完成片段与各阶段版本。

若新阶段产物版本和上下文未变，直接识别为无需替换。启动时校验全局版本与配置哈希；不一致返回 `E_SNAPSHOT_STALE` 并拒绝复用旧缓存。报告同时检查临时目录无残留。

## 增量/全量一致的判定

每次编辑后报告比较：

- 混音 PCM SHA-256 与逐字节 PCM。
- 逐片段峰值、片段 PCM 指纹、引用分轨集合。
- 逐帧加权明细 SHA-256。
- 稳定排序的失败清单。
- 任务覆盖集：`复用任务键 ∪ 重算任务键` 是否等于空白全量任务集。

复用/重算的**动作标签**依赖运行前缓存状态：空白全量运行必然把所有任务标为 `recomputed`，增量运行可能把同样任务标为 `reused`。因此内容等价判定使用“任务覆盖集 + 内容结果”，同时报告保留两套操作清单供诊断。这避免把缓存动作状态误当成渲染内容。

## 故障注入

```bash
python scripts/apply_injection.py --case all --out /tmp/stemscope-injected --parse
python -m stemscope inject --case v3_offset_oob --out /tmp/stemscope-v3-oob --parse
```

样例覆盖：

- v1 WAV chunk 长度字节损坏。
- v3 偏移表记录越界。
- v2 清单声明帧数与独立分轨实际帧数不一致。

注入只修改样本副本或参数，不修改程序源码。

## 代码地图

- `stemscope/package.py`：格式探测、v1/v2 解析与统一包构造。
- `stemscope/v3codec.py`：v3 固定头、偏移表、CRC32 编解码。
- `stemscope/conversion.py`：任意双向转换、旁路账本。
- `stemscope/render.py`：定点混音、逐帧明细、线程池与稳定排序。
- `stemscope/cache.py`：内容寻址缓存条目与包序列化。
- `stemscope/snapshot.py`：阶段 generation、原子指针与陈旧判定。
- `stemscope/workbench.py`：异步交互队列、预算响应、格式切换与恢复。
- `stemscope/report.py`：完整验收、全量对照与报告。
- `stemscope/dashboard.py`：自包含 HTML 诊断面板。

## 验证结果

运行 `python -m stemscope run` 后，`artifacts/report.json` 的 `acceptance.checks` 会给出机器可读结论；`artifacts/dashboard.html` 给出同源可视化。

本次本地验证记录：

- 最终报告 ID：`stemscope-9546386022c175d4`。
- 报告指纹：`9546386022c175d4a9a632e146a2ff8c71c2700fff682e982616fbde3597cc79`。
- 自动测试：12/12 通过（`unittest`，含一次完整验收）。
- 验收检查：15/15 通过。
- 三格式解析逐帧一致、重复解析指纹一致。
- 6 个转换方向均逐帧一致，`v3 -> v2` 丢失校验和逐轨进入旁路账本。
- 1000/2000 量化档把 Bass 定点整数从 `907` 改为 `1815`，产生 2,483 个不同帧。
- 10 条交互命令中 9 条预算内完成；1 条 3 秒模拟计算约在 1.79 s 返回旧摘要并标 `E_INTERACT_BUDGET_EXCEEDED`，旧摘要版本为 9、请求版本为 10。
- 每次编辑的增量与空白全量 PCM、峰值、逐帧明细、失败顺序和任务覆盖集一致。
- 最终混音 SHA-256：`7b2d3bb5663c4d3f68948fa14005fb87ea57b935fc5022433d5e2fec17d36de3`；峰值 `32768`；48,000 帧、24 个输出片段。
- 最终增量复用 48/96 个任务，重算 48/96；全量基准重算 96/96；两者任务覆盖集和内容结果一致。
- 4 线程与 1 线程逐帧一致，增益/混音阶段观测并发为 4/1，解析阶段观测并发也不超过配置上限。
- 快照续跑命中 4 个解析、72 个增益、24 个混音缓存条目；版本或配置哈希不一致均被 `E_SNAPSHOT_STALE` 拒绝。
- 交互运行按每 2 个片段提交阶段检查点，共记录 120 次检查点提交、0 个临时目录残留。
- 三类注入故障全部进入 `E_TRACK_PARSE_FAILED`。
- 报告自身不嵌入时间戳；实测耗时单独放在 `stage_timings_ms`，不参与可重复性指纹。

最终报告、面板与测试日志：

- `artifacts/report.json`
- `artifacts/dashboard.html`
- `artifacts/mix_final.wav`
- `artifacts/frame_details.jsonl`
- `artifacts/test_output.txt`
