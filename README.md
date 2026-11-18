# 声源分离分轨重混工作台

这是一个完全本地、确定性的声源分离分轨管理与重混工作台。项目支持 v1/v2/v3 三种分离结果交换格式、双向格式转换、基于内容指纹的片段级增量渲染、线程池并发、原子阶段快照、可处理退出信号、强制终止恢复，以及带业务码的异常注入隔离与 JSON 报告。

运行环境实测为 Python 3.11、NumPy 2.4.6、pytest 9.2.0。项目不访问网络、不依赖模型权重、外部账号或容器环境。

## 快速开始

```bash
# 当前工作区已经生成好 .venv、samples、configs 与验收产物。
.venv/bin/python -m pytest -q

# 正常三格式增量渲染，并执行 4 线程 vs 1 线程对照。
.venv/bin/python -m remix_workbench run \
  --samples-root samples \
  --config configs/remix.json \
  --workdir artifacts/release_run \
  --report artifacts/report.json

# 七步编辑序列；每次编辑都与空白工作区最终全量渲染逐帧比较。
.venv/bin/python -m remix_workbench run-sequence \
  --samples-root samples \
  --config configs/remix.json \
  --edits configs/edit_sequence.json \
  --workdir artifacts/sequence_final \
  --report artifacts/report_sequence.json

# 五类真实输入故障注入。
.venv/bin/python -m remix_workbench run \
  --samples-root samples \
  --config configs/remix.json \
  --injections configs/fault_injections.json \
  --workdir artifacts/injected_final \
  --report artifacts/report_injected.json
```

若需要从零重建确定性样本与配置：

```bash
.venv/bin/python -m remix_workbench generate-samples --root . --seed 20261118
```

## 目录

- `remix_workbench/formats.py`：三格式解析与统一 `TrackPlane`。
- `remix_workbench/conversion.py`：v1/v2/v3 写出、声道重排、量化残差与旁路损失元数据。
- `remix_workbench/mixing.py`：Q23 定点增益、词典序求和、饱和输出与片段分量摘要。
- `remix_workbench/engine.py`：四阶段缓存、增量失效、线程池稳定归并、报告。
- `remix_workbench/snapshots.py`：版本目录、原子清单指针、恢复判定。
- `remix_workbench/injections.py`：真实输入文件故障构造。
- `remix_workbench/sequence.py`：编辑序列及增量/全量比较。
- `samples/`：固定种子生成的 16 kHz 单声道 16-bit WAV、三格式包、替换 PCM、包络与样本清单。
- `configs/`：重混配置、编辑序列与异常注入清单。
- `tests/`：19 项回归测试。
- `artifacts/`：实际运行报告、混音结果、分量数组与测试日志。

## 确定性样本

`samples/manifest.json` 记录固定种子 `20261118`、每片段种子、帧数、包路径、来源帧标注与分轨增益元数据。三个混音片段分别为 12,800、14,080、15,360 帧，包含鼓、贝斯、人声三条程序合成信号及可控音量包络：

```text
samples/clip_000/mix.wav                     # 合成单声道混音
samples/clip_000/envelopes.json              # 100 ms 采样的可控包络点
samples/clip_000/packages/v1/mix.wav         # 三声道交错 16-bit
samples/clip_000/packages/v2/manifest.json   # JSON 清单
samples/clip_000/packages/v2/tracks/*.wav    # 逐分轨 24-bit WAV
samples/clip_000/packages/v3/tracks.r3p      # 二进制偏移表包
samples/replacements/clip_001_drums.wav      # 编辑序列使用的 PCM 替换
```

## 交换格式

### v1

单个 `mix.wav`，16 kHz、16-bit、三声道交错。规范声道顺序固定为 `bass, drums, vocal`。`package.json` 仅用于诊断，不参与权威音频读取。v1 本身没有逐轨增益或来源帧数标注通道。

### v2

`manifest.json` 加逐分轨独立 WAV。清单记录格式版本、采样率、位深、总帧数、`track -> channel` 映射、每轨文件、帧数、位深、增益元数据和来源帧标注。解析器逐轨校验 WAV 实际帧数与清单帧数；单轨损坏会生成 `E_TRACK_FORMAT` 失败，并以确定性静音补位后继续处理其他分轨。生成样例使用 24-bit 分轨，用于验证位深差异量化残差。

### v3

`tracks.r3p` 二进制包。头部为：

```text
8 bytes   magic = RMX3PK01
uint16    header version = 3
uint16    track count
uint32    frame count
uint32    sample rate
uint16    bit depth = 16
uint16    reserved
repeat track_count times:
  uint16  UTF-8 track name length
  bytes   track name
  uint32  absolute PCM block offset
  uint32  PCM byte length
payload: int16 little-endian PCM blocks
```

解析器验证 magic、表长度、名称、每块字节数、绝对偏移边界和块重叠。偏移越界与负载截断产生 `E_TRACK_FORMAT`。

## 统一表示

`TrackPlane` 携带：

- 来源格式版本与来源元数据；
- 采样率、位深、总帧数与每轨帧数；
- `track -> channel` 映射；
- 每轨 PCM；
- 缺失轨、解析问题与包路径。

内部 PCM 使用 Q23 有符号 `int32`：16-bit 样本左移 8 位，24-bit 样本原样保存。这样 16-bit 可精确往返，24-bit 低 8 位不会在统一表示处静默丢失。

## 格式转换与信息保留

任意两种格式都通过统一表示转换。写出规则：

- v1：按 `bass, drums, vocal` 固定声道顺序重新交错并量化到 16-bit。
- v2：拆分独立 24-bit WAV，重写 JSON 声道映射、逐轨帧数与可表达元数据。
- v3：按稳定分轨顺序生成头部、绝对偏移表与 int16 LE PCM 块。

转换目录总包含 `conversion_report.json` 和 `loss_metadata.json`。无法在目标格式表达的字段逐项记录，包括：

- `track_gain_metadata.*`；
- `source_frame_annotations.*`；
- `source_metadata.*`；
- 源格式版本、非规范声道映射；
- 16-bit 目标产生的逐轨量化残差统计与残差指纹。

v2 -> v1 的样例同时产生增益元数据、来源帧标注和 24-bit -> 16-bit 残差损失；v3 -> v2 目标可表达源格式、声道映射和帧数，因此无降级损失。

## 定点增益与混音

配置中的每轨：

- `gain`：非负有限线性增益；
- `mute`：布尔静音；
- `start_ms`, `end_ms`：该分轨有效区间。

增益先按 `gain_quantization` 量化为定点整数：

```text
gain_q = floor(gain * gain_quantization + 0.5)
contribution[frame] = pcm_q23[frame] * gain_q
```

片段内按分轨名字典序逐项整数相加，不使用浮点累加。输出使用确定的 half-away-from-zero 整数舍入并饱和到 `[-32768, 32767]`：

```text
mixed_i16 = clamp(round_half_away(sum(contribution) / (gain_quantization * 256)))
```

任一分轨增益为负或非有限数、裁剪越界、全部静音时映射 `E_MIX_CONFIG`。配置无效的片段不写出混音；同一配置重复运行逐帧一致。

## 增量缓存与失效范围

四个阶段均为内容寻址缓存：

1. `parse`：包文件内容摘要 + 格式版本 + 替换 PCM 摘要 + 解析器版本。
2. `convert`：解析内容指纹 + 源/目标格式 + 声道映射 + 转换器版本。
3. `gain`：单轨 PCM 指纹 + 量化增益 + 静音 + 格式版本 + 轨道声道 + 增益算法版本。
4. `mix`：片段引用的增益产物摘要 + 每轨裁剪 + 片段绝对/输出范围 + 量化档位 + 格式版本 + 混音算法版本。

每个片段记录 `tracks` 引用集合。改变某轨增益或静音只改变该轨增益键及其引用片段；替换单轨 PCM 不会使其他片段或分轨失效；切换格式会改变该片段解析/增益键并精确传播到下游；裁剪只改变相交片段键。报告分别输出 `cache_reuse`、`cache_recompute` 与稳定顺序 `cache_event_order`。

## 并发确定性

配置的 `thread_pool_size` 控制线程池。每个解析、转换、增益和片段任务都有稳定键；任务结果回到主线程后按阶段、片段序号、分轨名和键排序。最终 WAV 帧序、分轨顺序、复用/重算清单和失败清单与单线程执行一致。正常运行会额外在空白临时工作区以单线程全量渲染，并输出：

```json
{
  "concurrency_comparison": {
    "matches": true,
    "parallel_workers": 4,
    "serial_workers": 1,
    "cache_event_order_matches": true,
    "failure_order_matches": true
  }
}
```

## 快照、退出与恢复

解析、转换、增益、渲染均为阶段；渲染每 `snapshot_interval_fragments` 个片段提交一次进度（兼容旧字段 `snapshot_interval_clips`）。每次提交：

1. 缓存内容先写临时文件并原子替换；
2. 完整清单写入临时目录；
3. 整个目录原子改名为 `snapshots/vNNNNNNNN`；
4. 原子替换 `snapshots/CURRENT.json`。

只有被完整清单引用的缓存键才可信。强制终止后，半写临时目录和孤立缓存文件不会被恢复。

SIGINT/SIGTERM 可处理退出：引擎完成当前阶段、提交快照和 `progress.json` 后以整数退出码 `0` 退出，不承诺生成最终报告。恢复时报告包含快照版本、各阶段产物版本、已完成片段与已恢复产物数。

为可重复测试提供以下环境钩子：

```text
RMXW_GRACEFUL_STOP_AFTER_STAGE=parse|convert|gain|render
RMXW_FORCE_KILL_AFTER_STAGE=parse|convert|gain|render   # 提交该阶段后 os._exit(137)
RMXW_STAGE_DELAY_SECONDS=0.4                           # 测试 SIGTERM 投递时机
```

## 异常注入

`configs/fault_injections.json` 在真实输入副本上执行故障，而不是仅修改报告：

| 编号 | 阶段 | 真实故障 | 业务码 |
| --- | --- | --- | --- |
| INJ001 | parse | 截断 v2 bass WAV 字节 | E_TRACK_FORMAT |
| INJ002 | parse | 把 v3 vocal 偏移写到文件边界外 | E_TRACK_FORMAT |
| INJ003 | parse | 将 v2 drums 声明帧数加一 | E_TRACK_FORMAT |
| INJ004 | gain | 非有限增益 | E_MIX_CONFIG |
| INJ005 | config | 量化档位为 0 | E_MIX_CONFIG |

每个命中项将注入编号、片段、分轨、阶段与业务码写入 `failures` 和 `injection_summary`。解析单轨故障继续其他分轨；配置/增益故障阻止受影响片段写出，但继续处理其他片段与分轨。

## 报告

每次完整运行输出 JSON，主要字段：

- `clips[].fragments[]`：片段范围、引用分轨、峰值、PCM 指纹、逐帧加权分量哈希、首末帧分量摘要与分量和；
- `clips[].weighted_components`：完整 `frames x tracks` int64 加权矩阵和 mixed Q23 的 NPZ 路径与 SHA-256；
- `cache_reuse` / `cache_recompute` / `cache_event_order`；
- `conversion_degradation_losses`；
- `concurrency_comparison`；
- `recovery` / `snapshots`；
- `failures` / `failure_order` / `injection_summary`。

## 实际验证

本次实现已执行：

- 三格式解析；v1 -> v3、v3 -> v2、v2 -> v1 转换；v2 -> v1 损失清单；
- 同配置首次运行 100 个阶段产物，重跑 100/100 全部复用，渲染签名相同；
- 4 线程与 1 线程空白工作区渲染逐帧、片段峰值、分量摘要和失败顺序一致；
- 七步编辑序列（增益、静音、裁剪、PCM 替换、三次格式切换）每步均与单线程空白全量渲染一致；
- 五个注入项全部命中并映射到预期业务码；
- SIGTERM 收尾退出码 0、阶段快照已提交且未写最终报告；强杀退出码 137，重启只依赖完整快照恢复；
- `pytest -q`：19 passed。

具体报告见 `artifacts/report.json`、`artifacts/report_sequence.json`、`artifacts/report_injected.json`，测试输出见 `artifacts/pytest.txt`，机器可读验收摘要见 `artifacts/verification.json`。
