# 技术设计

## 1. 数据流

```text
固定种子合成器
  ├─ WAV 44.1k / 16-bit / mono
  ├─ WAV 48k / 24-bit / stereo
  ├─ OGG Vorbis 48k / stereo（本地 ffmpeg）
  ├─ 移调/变速变体
  ├─ 重复片段查询
  └─ 损坏样本
        ↓
严格解码与来源参数记录
        ↓
单声道混合 + 11025 Hz 重采样
        ↓
STFT → 对数频带局部峰 → 时频邻域约束 → 峰对哈希
        ↓
SQLite 倒排索引（哈希 → 曲目/来源/锚点）
        ↓
时间差投票直方图 → 候选排序 + 对齐时间
        ↓
格式归并 / 迁移 / 增量更新 / 删除 / 崩溃恢复
```

## 2. 音频格式契约

`audiofp.audio.decode_audio` 不接受“猜测参数”的降级路径：

- WAV 先解析 RIFF chunk，校验 RIFF 大小、fmt、block align、byte rate、data 帧对齐及截断。
- soundfile 返回的实际帧数必须等于容器声明帧数；WAV 还必须等于严格头部声明值。
- 有损容器记录 `bit_depth=None`，容器与 codec 分开记录。
- 调用方可通过 `expected` 声明采样率、位深、声道、容器等；不一致返回 `E_AUDIO_FORMAT`。
- 解码后统一为 `float64` 单声道，并用 SciPy `resample_poly` 重采样到配置分析采样率。

## 3. 指纹语义

### 3.1 STFT 与峰值

- 不做 center padding，帧网格只由 `n_fft` 与 `hop_length` 决定。
- Hann 周期窗，`rfft` 幅度谱。
- 频率轴按几何级数分成固定数量的对数频带，每个频带取最大幅度。
- 使用 `maximum_filter` 做时间/频带邻域局部极大值检测，排除边界与低能量候选。
- 候选按峰值强度截断到 `duration * peak_density`，最终按 `(frame, band, score)` 稳定排序。

### 3.2 峰对与哈希

每个锚点峰在 30–140 帧目标区内最多配对 5 个强目标峰。哈希原像只含编码空间、两个频带和量化 Δt：

```text
logband-pair-dt-v2|17|42|0007
```

因此任何第三方都可以根据索引中保存的 `band1/band2/delta_frames` 重算哈希。最终指纹序列按锚帧、目标帧、频带和哈希排序。

### 3.3 版本与配置摘要

`FingerprintConfig.extraction_dict()` 包含 STFT、窗函数、峰值密度、频带边界、邻域、目标区、配对上限、Δt 量化与编码空间。其规范化 JSON 的 SHA-256 就是 `config_digest`。索引同时保存 `fingerprint_version`、`config_digest`、`encoding_space` 与完整 `config_json`。

`fp-v1` 与 `fp-v2` 使用完全相同的提取峰与目标区，只有 Δt 量化和编码空间不同，因此可迁移。若 `n_fft`、频带、峰值密度等任一提取参数变化，则 `migration_compatible_from` 返回假，必须全量重提取。

## 4. 跨格式一致性

对两段指纹：

1. 按哈希建立右序列的锚点位置表；
2. 每个左指纹只消费一个右指纹，锚点差不超过配置容差；
3. 计算 `matched / min(count_left, count_right)` 与 Dice；
4. 未匹配指纹的频带中位数差用于识别 `frequency_band_shift`；
5. 结果附带可能的 `resampling_or_codec_frequency_shift`、`decoder_time_base_offset`、`lossy_codec_peak_perturbation` 等原因。

阈值是提取阶段的独立验收条件，检索阶段不能改变它。

## 5. 检索与对齐

查询指纹在倒排表中命中后，对每个 `(source_id, db_anchor - query_anchor)` 计票。投票直方图以 ±1 帧平滑寻找主导时差：

```text
score = smoothed_dominant_votes / query_fingerprint_count
alignment_start = dominant_offset_frames / frames_per_second
```

结果按 `(-score, -votes, group_id)` 稳定排序，同一逻辑组只保留最佳来源，并携带全部格式来源。查询上分割出的重复片段可能在曲目中出现多次；返回的是投票最强的一个对齐位置，直方图保留其他峰供复核。

当精确匹配最高分低于 0.12 时，进入受限变换探查：

- 频带平移 `-6..+6`；
- 时间尺度 `0.88..1.12`、步长 `0.02`；
- 每个假设都重新构造指纹哈希并独立投票；
- 结果中记录 `pitch_shift_bands`、`time_scale` 与变换标签。

该路径只影响弱匹配查询；精确同内容查询不会触发变换扩大召回。

## 6. 同曲归并与误并保护

导入新来源时，对已有每个来源做时间差投票，分数达到 `dedup_score_threshold`（默认 0.42）才归入该组。判定只看指纹，不使用文件名、路径、标题或时长。

验收时使用真值映射检查每个归并组是否包含不同真值 ID；若是，构造 `E_DEDUP_MERGE`，详情包含组、真值标签、来源列表、每个来源对的投票数与偏移。

## 7. 索引、迁移与原子性

SQLite 表：

- `meta`：schema、指纹版本、配置摘要、编码空间、修订号、索引内容哈希。
- `work_groups`：逻辑曲目。
- `sources`：格式来源、解码参数、原始文件 SHA-256、指纹内容哈希。
- `fingerprints`：倒排项 `(hash, source_id, group_id, anchor, target, band1, band2, delta)`。

写入使用 `BEGIN IMMEDIATE` 与 `synchronous=FULL`，进程中断时 SQLite 回滚 hot journal。删除组通过外键级联清理来源和指纹。索引内容哈希对活动组、来源、指纹的稳定排序结果计算，因此不依赖 SQLite 页布局。

迁移流程：

1. 读取旧配置并检查提取参数是否等价；
2. 在事务中根据保存的 `band1/band2/delta_frames` 重算新哈希；
3. 更新 meta，修订号加一；
4. 再次打开时配置摘要相同，不再修改。

不可迁移时在返回任何部分结果前抛出 `E_INDEX_VERSION`。

## 8. 确定性策略

- 合成器仅用固定种子生成，不读取 `time`、环境随机源或网络。
- JSON 全部使用 `sort_keys=True` 和固定分隔符。
- 报告不含时间戳、绝对路径和运行目录。
- ffmpeg 错误文本中的绝对路径归一为文件名。
- Ogg muxer 的随机 bitstream serial 被固定，并重新计算每页 CRC；Vorbis 数据包与解码结果不变。
- SQLite 插入顺序、查询顺序和索引构建顺序固定；同一操作序列产生逐字节一致的数据库文件。
- 完整演示默认执行两次，比较所有文件 SHA-256。

## 9. GUI 与便携模式

PySide6 窗口包含查询文件选择、Top K、候选表、时间轴和得分/投票解释。无 `DISPLAY`/`WAYLAND_DISPLAY`、PySide6 导入失败或 Qt 初始化失败时，明确打印 `[PORTABLE]` 原因并进入终端命令模式。`--smoke` 强制使用 Qt offscreen 平台构造窗口，用于 CI 验证。

## 10. 已知边界

- 移调/变速匹配使用受控离散假设，适合素材库演示，不等价于连续音高/速度不变性算法。
- 大规模曲库应把 SQLite 倒排查询进一步批量化，并将投票直方图改为按候选分片。
- 当前实际运行验证环境是 WSL2 Linux；Windows 入口、PowerShell 脚本、路径处理和 PySide6 平台选择均已提供，但本机没有独立 Windows 桌面可执行实机验证。
