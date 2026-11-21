# 本地确定性语音转换工作台

这是一个从零实现的本地 Voice Conversion 会话工作台。它把音频分析、目标音色统计、逐段映射合成和总装拆成四个可快照阶段，重点验证以下不变量：

- 同一配置、同一音频、同一命令前缀必然得到相同的定点特征、段指纹和 PCM。
- 每条命令执行后，增量恢复得到的快照与单进程全量重放得到的快照逐工件一致。
- 参数、片段替换或追加只使受影响段失效；未失效段直接复用持久化 PCM。
- 预算耗尽时已计费段不回滚、不重复计费，未完成段和游标保留在账本。
- `os._exit(137)` 模拟强制终止后，只通过已完整提交的 `CURRENT` 快照恢复。
- 分析、映射、合成中的单段异常被隔离，失败段不会进入总装。

实现仅依赖 Python 标准库、`numpy` 和 `pytest`；WAV 读写使用标准库 `wave`，PCM 分片压缩使用标准库 `zlib`，没有使用 Docker、外部音频库或网络服务。

## 快速开始

### Windows PowerShell

```powershell
py -3.11 -m pip install numpy pytest

py -3.11 -m vc_workbench generate --dir samples --force
py -3.11 -m vc_workbench run `
  --samples samples `
  --config samples/config.json `
  --commands samples/commands.json `
  --state workbench_state `
  --output converted.wav `
  --report run_report.json `
  --fresh

py -3.11 -m vc_workbench inspect --state workbench_state
py -3.11 -m pytest -q
```

也可执行完整 Windows 验收入口：

```powershell
powershell -ExecutionPolicy Bypass -File tools/run_windows.ps1
```

### CMD

```bat
py -3.11 -m vc_workbench run --samples samples --config samples/config.json --commands samples/commands.json --state workbench_state --output converted.wav --report run_report.json --fresh
```

不安装包也可以从仓库根目录直接以 `python -m vc_workbench` 运行。若要安装命令行入口，可执行 `py -3.11 -m pip install -e .`，之后使用 `vc-workbench`。

## 四阶段流水线

1. **分析**：扫描 `samples/references/**/*.wav` 和 `samples/sources/**/*.wav`，严格校验 RIFF chunk、PCM format、16000 Hz、单声道、16 bit、data chunk 长度与样本数。帧长/帧移来自配置，末帧补零并标记。每帧计算梅尔频带能量、倒谱系数和自相关基频候选；帧按索引升序、频带按编号升序累加。
2. **音色统计**：按能量门限剔除静音帧，对每个目标说话人汇聚 F0、能量、倒谱与 log-mel 的均值、方差和分位数。分位数使用精确有理数位置和配置指定插值规则。段级源统计同样单独持久化。
3. **逐段映射**：按源片段边界处理；每段只使用自身源帧和自身源统计。目标基频按半音偏移确定性平移；目标/源长期平均倒谱距离派生弯曲量；音色强度插值谱包络；共振峰弯曲和频带平滑均采用确定性定点运算。段 PCM、逐帧摘要、源帧集合和段指纹一起持久化。
4. **总装**：仅按段序拼接状态为 `mapped` 的段。预算未完成、分析失败、映射失败或合成失败的段被列入 `skipped`，不会借用其他段统计或 PCM。

## 配置

`samples/config.json` 是可直接运行的示例。核心字段如下：

| 字段 | 含义 |
| --- | --- |
| `frame_length_ms`, `frame_shift_ms` | 分析帧长与帧移 |
| `mel_bands`, `mel_max_hz` | 梅尔频带数与上限频率 |
| `cepstrum_order` | 倒谱阶数 |
| `f0_quantization_levels`, `f0_min_hz`, `f0_max_hz` | F0 候选范围与量化档位 |
| `thread_pool_workers` | 分析/映射线程池最大并发 |
| `speech_frame_budget` | 会话累计语音帧预算 |
| `snapshot_interval_segments` | 命令内每处理多少个稳定段提交检查点 |
| `max_frames_per_segment` | 单段帧数上限 |
| `fixed_point_decimals` | 定点小数位宽；示例为 6 |
| `energy_threshold_db` | 静音帧门限 |
| `quantiles`, `quantile_interpolation` | 统计分位数和插值规则 |

配置哈希覆盖全部数值语义。分析、统计、映射和总装还分别有独立配置指纹；缓存键由上游内容指纹、配置指纹和量化档位派生。

## 命令流

`samples/commands.json` 的顶层包含默认参数、初始源片段和命令数组。命令序号必须稳定；非法命令记录 `E_COMMAND_INVALID` 并继续执行后续条目。

支持的命令：

```json
{"seq": 1, "op": "set_target", "target_speaker": "target_b"}
{"seq": 2, "op": "set_pitch_offset", "value": 1.5}
{"seq": 3, "op": "set_timbre_strength", "value": 0.72}
{"seq": 4, "op": "set_formant_warp", "value": 1.08}
{"seq": 5, "op": "set_smoothing", "value": 0.35}
{"seq": 6, "op": "override_segment", "segment_id": "seg_001", "timbre_strength": 0.55}
{"seq": 7, "op": "append_segment", "segment_id": "seg_004", "path": "sources/append.wav"}
{"seq": 8, "op": "replace_segment", "segment_id": "seg_002", "path": "sources/replacement.wav"}
```

`set_params` 可一次设置多个参数。参数也接受定点整数字段，例如 `pitch_offset_semitones_q`、`timbre_strength_q`。兼容 `append`/`replace`、`set_target_speaker`、`set_formant_curvature` 等别名。所有路径必须位于样本目录内，禁止绝对路径和 `..`。

## 增量失效与复用账本

每个映射段保存：

- `mapping_key`：源分析键、段统计指纹、目标统计指纹、有效参数、映射配置指纹的哈希。
- `local_stats_key`, `target_stats_key`, `stats_fingerprint`。
- `source_frame_set`：本段实际引用的稳定帧键。
- `pcm_sha256`, `segment_fingerprint` 和 `piece_file`。

命令执行时逐段比较期望 `mapping_key` 与已持久化映射。命中且 PCM 校验通过则进入 `reuse`，否则进入 `recompute`。替换片段会改变源内容指纹和分析键；段级参数只影响该段；全局参数影响所有段。`reuse` 和 `recompute` 按段序号稳定排序。

预算账本只对本次实际调度并处理的重算段计费。命中复用不扣预算。计费记录包含命令序号、稳定序号、段 ID、帧数和映射身份；报告中的 `replay_consumed` 从账本重放求和。待办队列和 `todo_cursor` 同样可从映射状态重放验证。

## 快照与恢复

状态目录结构：

```text
state/
  CURRENT                         # {"snapshot_id": ..., "manifest_sha256": ...}
  snapshots/<snapshot_id>/
    manifest.json
    analysis.json
    stats.json
    mappings.json
    assembly.json
    ledger.json
    pieces/<segment>.pcm.z
    assembly.pcm.z
  tmp/                            # 未提交临时目录，永不作为恢复依据
```

提交过程：

1. 在 `tmp/` 下创建完整自包含快照，写入所有阶段工件和 zlib PCM。
2. 对文件执行 `fsync`（目录 fsync 在不支持的平台上尽力而为），最后写 `manifest.json`。
3. 完整目录以一次 rename 暴露为 `snapshots/<id>`。
4. 最后用原子 `os.replace` 切换 `CURRENT`。在此之前的任意强杀都只会看到旧快照。
5. 加载时校验 `CURRENT` 中的 manifest 哈希、每个阶段文件哈希、每个 PCM 分片哈希和总装 PCM 哈希。

若快照 ID 对应目录已经损坏，新完整快照会使用 `-rN` 修订 ID；旧 `CURRENT` 在新快照提交完成前保持可见。

启动兼容性检查覆盖格式版本、引擎版本、配置哈希、当前段引用集的音频指纹、命令前缀哈希和全局版本/游标一致性。失败输出 `E_SNAPSHOT_STALE` 并整阶段重建。命令流**尾部追加**不触发失效；新增尚未被当前段引用的 WAV 也不会改变当前音频指纹。若前缀命令、参考音频或被引用源音频变化，则重建。

### 恢复与退出码

| 退出码 | 语义 |
| --- | --- |
| `0` | 完成、正常检查点停止，或收到可处理退出信号后完成当前命令并提交 |
| `2` | CLI、配置或命令流无法加载 |
| `3` | 内部错误或快照无法恢复 |
| `4` | 预期业务失败：音频格式、段超限、预算、非法命令、段异常等 |
| `137` | 测试钩子 `--crash-after-command` / `--crash-on-checkpoint` 模拟强杀，不承诺报告 |

`SIGINT`、`SIGTERM`、Windows `SIGBREAK` 只设置停止标志，当前命令会继续完成并提交快照，随后以 0 退出。

## 并发与异常隔离

- 帧任务稳定键由类别、片段序号/参考路径和帧序号组成；映射任务稳定键由段序号组成。
- 线程池规模严格不超过 `thread_pool_workers`。
- 即使线程完成顺序不同，结果都按稳定键收集、排序和提交。
- 映射按稳定段序分批调度；预算按同一顺序分配，因此并发度与预算耗尽时机不改变最终顺序。
- 单段分析、映射或合成异常记录 `E_SEGMENT_FAILURE`，包含段 ID、阶段、异常类型和详情；其他段继续。总装仅引用成功段。

## 样本与异常夹具

`tools/generate_samples.py` 使用固定种子 `20261121` 生成：

- `samples/references/target_a/*.wav`、`target_b/*.wav`：不同 F0 和共振峰统计的参考音色。
- `samples/sources/session.wav`：含三个边界的多段源音频。
- `samples/sources/replacement.wav`、`append.wav`：替换/追加夹具。
- `samples/errors/truncated.wav`：人为截断 RIFF。
- `samples/errors/audio_format/`：可执行 CLI 的截断音频样本集。
- `samples/errors/overflow_config.json`、`overflow_commands.json`：段帧数超限。
- `samples/errors/invalid_commands.json`：乱序序号和未知操作。

重新生成：

```powershell
py -3.11 -m vc_workbench generate --dir samples --force
```

## 验证结果

仓库中已附本次实际运行结果：

- `artifacts/verification_summary.json`：10 项验收检查，全部通过。
- `artifacts/sample_run_report.json`：完整 8 命令样本报告。
- `artifacts/sample_converted.wav`：完整转换输出，`28800` 个采样，`57644` 字节。
- `artifacts/verification/`：预算、恢复、错误、串并行、故障隔离等报告。

本次执行结果摘要：

```text
all_passed: true
elapsed_seconds: 77.264
final assembly sha256: cc209855cea0d12d4a8ee9abdeaa984af9aab98e07b0255b96e604f08a2e78e9
budget consumed: 765 / 1000
incremental prefix snapshots 1..8: all equal to full replay
budget case: 49 consumed, 21 remaining, todo [seg_002, seg_003], todo_cursor 1
forced kill after command 3: recovered to same final PCM
intra-command kill at checkpoint 2: active_command_seq=1, budget=91 recovered
serial workers=1 and parallel workers=4: identical PCM and stable lists
```

运行验收矩阵：

```powershell
py -3.11 tools/run_verification.py
```

单元/集成测试共 16 项：

```powershell
py -3.11 -m pytest -q
```

当前交付环境实际为 WSL/Linux、Python 3.11.0rc1、numpy 1.26.4、pytest 8.3.3。代码路径、文件替换、WAV 读写、PowerShell 验收入口均按跨平台和 Windows 本地运行方式编写；`tools/run_windows.ps1` 提供原生 Windows 复核入口。本次无法声称已在原生 Windows 主机上执行，报告中保留了实际平台信息以避免误报。
