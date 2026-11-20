# 技术说明

## 1. 设计目标与边界

工作台只处理本地确定性夹具，不访问网络、真实编解码服务或外部媒体。视频使用 YUV4MPEG2 的 `420mpeg2` 子采样，音频使用标准 PCM16 WAVE 与 interleaved float32 参考波形。所有容器、解析器、指标和账本格式均在本仓库实现。

完整夹具为 300 帧、640x360、30fps、10 秒；音频为 48kHz 双声道、每声道 480000 样本、总计 960000 样本。源视频与播放视频各约 99 MiB，完整工作区约 210 MiB。

## 2. 模块划分

| 模块 | 职责 |
| --- | --- |
| `generator.py` | 像素字体、渐变、色块、硬切、YUV420 转换、有损播放形态、WAV/参考波形、真值清单 |
| `y4m.py` | Y4M 流式写入、头部校验、逐帧读取、同偏移有限重试 |
| `formats.py` | JSON v2、旧版 JSON、EMF1 二进制写入/解析、迁移表、统一 `CarrierModel` |
| `quality.py` | PSNR、SSIM、逐帧记录、分段聚合、内容哈希 |
| `audio.py` | WAV/参考波形解码、固定互相关、增益/相似度、统一时基 |
| `video_eval.py` | 选择帧计算、丢帧/时间戳故障隔离、回放预览 |
| `closure.py` | 依赖签名、帧闭包、段闭包、实际重算集合核对 |
| `equivalence.py` | 三承载逐字段比较、首个不一致字段、迁移假设分类 |
| `ledger.py` | 原子分片、manifest 提交、CURRENT、关闭标记、恢复与孤儿回收 |
| `pipeline.py` | 生成后构建承载、评估编排、注入/变更应用、账本提交 |

## 3. 确定性素材

### 3.1 视频

固定种子 `20261120`。画面由 NumPy 直接渲染，不依赖系统字体或图像库字体渲染差异。滚动文字使用代码内置 5x7 像素字体；每个镜头有独立渐变、色块轨迹、颜色和文字。硬切边界固定为：

```text
shot 0: frames   0..89
shot 1: frames  90..179
shot 2: frames 180..239
shot 3: frames 240..299
```

生成流程：

1. 渲染 RGB 帧。
2. 按 BT.601 系数转换为 YUV420p，U/V 由 2x2 区域平均后取整。
3. 写出 `source.y4m`。
4. 从无损 YUV 派生确定性有损形态：Y 以 `qstep=6` 量化并混入平滑分量与固定图案；U/V 以 `qstep=10` 量化并平滑。
5. 写出 `encoded.y4m`。PSNR/SSIM 使用源 Y 与播放 Y 比较。

整个过程没有时间戳、随机数生成器状态、临时文件名或绝对路径参与媒体内容。

### 3.2 音频

四个镜头使用不同左右频率与包络，镜头边界处形成确定性硬切。PCM 由浮点波形乘以 `32767`、四舍五入并裁剪到 `[-32768, 32767]`。参考波形为同一 PCM 除以 `32768` 后的 interleaved float32：

```text
audio.wav              PCM16LE interleaved, 48kHz, stereo
audio_reference.f32le float32LE interleaved, 48kHz, stereo
```

真值清单记录每个镜头的首个音频采样、64 个等距参考采样、每帧首采样对应的左右参考值，以及 `sample_count_per_channel`、`total_samples`。

### 3.3 真值与内容哈希

`assets/truth.json` 包含：

- 帧数、镜头数、镜头边界、每帧镜头号和时间戳。
- 每帧色块 `x/y/w/h/color`、文字滚动偏移、是否硬切。
- 音频采样率、声道数、每声道样本数、总样本数、参考波形文件。
- `source.y4m`、`encoded.y4m`、`audio.wav`、`audio_reference.f32le` 的 SHA-256。
- `content_hash`：由媒体哈希和去掉自身哈希字段后的真值核心内容共同计算。

同一配置两次生成后，媒体文件、参考波形与真值 JSON 均逐字节一致。

## 4. 三种承载

### 4.1 评估工程 JSON v2

`carriers/project_v2.json` 必需字段：

```json
{
  "schema_version": "eval-project/2.0",
  "content": {},
  "encoding_params": {},
  "shots": [],
  "metric_policy": {},
  "frames": []
}
```

每个帧记录至少包含 `frame_index`、`psnr_db`、`ssim`，工作台样本还带 `shot_id`、`timestamp_ms`、`chroma_psnr_db`、`flags`。解析器不接受缺失必需字段或未知 schema 版本。

### 4.2 旧版 JSON v1

`carriers/project_legacy_v1.json` 使用 `eval-project/1.0`，只包含 `content`、`encoding_params`、`frames`。缺少 `shots`、`metric_policy`、帧 `shot_id` 和 `flags` 等字段。

固定迁移表：

| 缺失字段 | 补齐来源 | 语义分类 |
| --- | --- | --- |
| `shots` | 内容真值的 `shots` | 不确定假设 |
| `metric_policy` | 规范口径 `quality/2.0` | 不确定假设 |
| `frame.shot_id` | 真值逐帧镜头映射 | 不确定假设 |
| `frame.flags` | `0` | 确定性补齐 |
| `frame.chroma_psnr_db` | `null` | 确定性补齐 |

等价性报告不会把旧版直接标成严格等价，而是标为 `equivalent_with_assumptions` 并列出所有迁移来源。

### 4.3 二进制指标帧 EMF1

头部固定 29 字节，所有多字节数值按当前声明的字节序解释：

| 偏移 | 长度 | 字段 |
| --- | ---: | --- |
| 0 | 4 | magic `EMF1` |
| 4 | 2 | BOM：`FF FE` 小端，`FE FF` 大端 |
| 6 | 1 | `index_width`，支持 1/2/4 |
| 7 | 2 | `metric_encoding_version` |
| 9 | 4 | `field_bitmap` |
| 13 | 4 | `frame_count` |
| 17 | 4 | `segment_count` |
| 21 | 4 | `meta_len` |
| 25 | 4 | `payload_crc32` |

之后是规范 JSON 元数据与连续指标记录。元数据包含 `content`、`encoding_params`、`shots`、`metric_policy` 和 `binary_contract`。`binary_contract` 重复声明字节序、索引宽度、版本和字段位图，解析器必须与头部完全一致。

字段位图：

| 位 | 字段 | 编码 |
| ---: | --- | --- |
| 0 | `frame_index` | `index_width` 字节无符号 |
| 1 | `shot_id` | `u16` |
| 2 | `psnr_db` | `f64` |
| 3 | `ssim` | `f64` |
| 4 | `timestamp_ms` | `f64`，保留 30fps 的 33.333ms 时基 |
| 5 | `flags` | `u16` |
| 6 | `chroma_psnr_db` | `f64`，v2 新增 |

v1 必需位图为 `0x3F`，v2 为 `0x7F`。解析器拒绝未知位、缺失必需位、v1 声明 v2 专属位、头部与元数据位图不一致、记录长度不一致、CRC 错误、索引溢出及不支持的版本。

迁移 `1 -> 2` 只补 `chroma_psnr_db=null` 与规范化版本号，不修改核心帧指标。迁移函数对已是 v2 的记录幂等。

## 5. 统一内部模型与等价性

三种承载都归一为 `CarrierModel`：

```text
carrier_type, schema_version, content, encoding_params, shots,
metric_policy, declared_frames, segments, assumptions, provenance
```

分段结论由统一函数从声明帧、镜头和口径重新计算，而不是信任文件中可能存在的 `segments` 数组。

逐对比较顺序固定：

```text
project_json <-> metric_binary
project_json <-> legacy_json
metric_binary <-> legacy_json
```

比较内容、编码快照、镜头、指标口径、每帧字段和统一分段字段。浮点容差为 `1e-6`。报告记录每对的状态：

- `equivalent`
- `equivalent_with_assumptions`
- `mismatch`，并记录首个不一致字段路径
- `invalid_carrier`，并记录业务码

`report_digest` 只依赖规范内容哈希、比较结果、迁移假设与格式条件，不依赖工作区绝对路径。因此同一内容位于不同目录时摘要不变。

## 6. 指标口径

当前规范口径：

```json
{
  "policy_version": "quality/2.0",
  "pixel_range": [0, 255],
  "pixel_format": "yuv420p8",
  "channel_weights": {"y": 1.0, "u": 0.0, "v": 0.0},
  "downsample": "none",
  "ssim_window": 8,
  "ssim_c1": 6.5025,
  "ssim_c2": 58.5225,
  "psnr_ceiling_db": 99.0,
  "aggregation": {
    "psnr": "mean_frame_db",
    "ssim": "mean_frame"
  }
}
```

PSNR 作用于 Y 通道：

```text
MSE = mean((Y_source - Y_decoded)^2)
PSNR = min(99.0, 10 * log10(255^2 / MSE))
```

SSIM 使用不重叠 8x8 窗口，逐窗口计算均值、方差和协方差，再对窗口 SSIM 求平均。窗口必须整除 640x360。U/V 的 PSNR 作为补充字段写入账本，但当前通道权重为零，不参与主 PSNR 和分段结论。

分段 PSNR 是有效帧 PSNR 的算术平均，SSIM 是有效帧 SSIM 的算术平均。缺失帧被排除但会记录 `missing_frames`、`complete=false` 与覆盖率相关字段，不会被静默计为通过。所有浮点先按固定小数位舍入再进入规范 JSON。

## 7. 音频对齐与联合时基

统一时基：

```text
master  = global_content_start
video   = frame_index / 30
audio   = sample_index / 48000
units   = milliseconds
```

每段使用固定 FFT 互相关，在 `±50ms` 范围内搜索样本偏移。最佳偏移处计算最小二乘增益：

```text
gain = dot(decoded_aligned, reference) / dot(reference, reference)
similarity = dot(decoded_aligned, reference) / (||decoded_aligned|| * ||reference||)
```

每段输出 `offset_samples`、`offset_ms`、`gain_ratio`、`waveform_similarity`、`video_offset_ms` 与 `joint_av_skew_ms`。联合偏差定义为：

```text
joint_av_skew_ms = audio_offset_ms - video_offset_ms
```

因此音视频偏差引用同一全局零点，而不是各自独立标定。若任一段 `abs(audio_offset_ms) > tolerance_ms`，总体状态为 `drift`，业务码为 `E_AUDIO_DRIFT`，并记录首个越界段。声明采样率与 WAV 头不一致时返回 `E_AUDIO_SAMPLE_RATE`，该承载的音频分析缺失，但帧和承载报告继续产出。

## 8. 依赖闭包与增量重算

逐帧依赖签名：

```text
hash(
  content_hash,
  encoding_params_hash,
  metric_policy_hash,
  frame -> shot_id,
  {dropped, timestamp_jump_ms, timestamp_tolerance_ms},
  metric_encoding_version
)
```

分段依赖签名：

```text
hash(
  content_hash,
  encoding_params_hash,
  metric_policy_hash,
  shot 完整声明,
  成员帧的 record_hash 与 deps_hash
)
```

失效闭包计算：

- 无已提交账本：全部重建。
- 编码快照、指标口径或内容变化：所有帧和所有段失效。
- 镜头边界/标签变化：只有帧到镜头映射变化的帧失效；分段由新的帧依赖和镜头声明重新比较。
- 丢帧或时间戳跳变：只有对应帧依赖变化。
- 二进制故障或音频故障：视频帧闭包为空，仅对应载体/音频部分变化。

执行器只对闭包内帧调用指标计算，只对闭包内段调用聚合。未重算的帧使用上一提交账本中的原始 JSON 记录，因此其 `record_hash` 和 `deps_hash` 保持不变。完成帧重算后再次计算所有新段依赖，得到段闭包，然后严格比较：

```text
actual_recomputed_frames   == closure.frames
actual_recomputed_segments == closure.segments
```

不相等时返回 `E_CLOSURE_MISMATCH` 并将实际集合写入账本。测试注入 `force_closure_mismatch` 会故意加入不存在的帧 `-1` 触发该路径。

## 9. 注入与单元隔离

注入配置 schema 为 `injection/1.0`，支持：

| 字段 | 作用 |
| --- | --- |
| `dropped_frames` | 标记指定帧缺失，仍消费视频流保持后续同步 |
| `timestamp_jumps` | 给指定帧时间戳注入毫秒跳变 |
| `binary_fault` | `endian` / `index_width` / `metric_version` / `field_bitmap` |
| `audio_declared_sample_rate` | 覆盖声明采样率并与 WAV 头核对 |
| `audio_sample_offset` | 对解码音频施加样本偏移 |
| `audio_gain` | 对解码音频施加增益 |
| `audio_tolerance_ms` | 音频漂移容差 |
| `timestamp_tolerance_ms` | 视频时间戳容差 |
| `force_closure_mismatch` | 仅测试闭包不一致业务码 |

二进制故障在内存中改写头部后交给正式解析器，不修改承载文件。承载解析按 JSON v2、二进制、旧版 JSON 独立执行；一个载体失败不会阻止其他载体和实际解码继续。帧读取使用同文件偏移最多重试 2 次，失败后记录 `E_DECODE` 和重试次数，并将剩余无法安全定位的帧标为缺失。

## 10. 原子账本与恢复

每次运行创建 `ledger/snapshots/<run-id>/`。所有分片通过：

1. 同目录 `mkstemp`。
2. 写入、flush、`os.fsync`。
3. `os.replace` 原子改名。
4. POSIX 下对目录 `fsync`；Windows 下跳过目录 fsync。

分片写入完成后最后写 `manifest.json`，manifest 包含版本戳、父版本、内容哈希、每个分片的 SHA-256 和指标账本摘要。manifest 是提交点。随后原子更新 `CURRENT.json`。只有完整正常退出才写 `closed.json`。

新运行开始时删除旧的 `closed.json`，避免把新会话错误标记为已正常退出。强制终止后：

- `recover` 扫描快照目录，按版本号、内容哈希和 manifest 哈希选出最新有效提交。
- 校验 manifest 自身哈希、所有分片 SHA-256，以及关键分片的 `content_hash`。
- 无 manifest 的目录、损坏的新快照和同目录临时文件被登记并回收。
- 只写 `recovery.json`，不创建 `closed.json`，因此不会伪造退出报告。
- 若崩溃发生在 manifest 之后，新快照已经是有效提交，可直接作为恢复目标；若发生在 manifest 之前，则恢复上一提交。

`E_SNAPSHOT_STALE` 用于乐观并发检查：调用方声明期望版本时，`begin` 与 `commit` 都会再次核对最新提交版本。

## 11. 实际验证

执行环境：Linux/WSL2 上的 Python 3.11.0rc1、NumPy 2.4.6、Pillow 13.0.0。代码使用跨平台标准库原子改名接口，目标 Windows Python 3.10+ 可直接运行。

```bash
python -m unittest -v tests.test_workbench
python scripts/validate.py --workspace /tmp/evalworkbench-validation-2 \
  --report artifacts/validation_report.json
```

结果：

- 单元/集成测试：10 项全部通过，约 30.9 秒。
- 验收脚本：29 项全部通过，约 45.5 秒。
- 素材内容哈希：`6c2b2bd21c6a04103cf4982d7c418be58654ec27ccdb45b7ffe061aa2a2d94c4`
- 指标账本摘要：`cac388b6a43161c21402427051fb93796ce805ca4f44ad2afb0307ab1a418695`
- 等价性摘要：`26b252542a7b83156d99fd1bf47d94ecde872c21b7d93211dca808683390ffe2`
- 两次生成的所有素材/真值文件 SHA-256 相同。
- 重复解码后 `actual_frames.json`、`actual_segments.json`、`audio.json`、`equivalence.json` 分片 SHA-256 相同。
- 编码快照或指标口径变化时，依赖闭包为 300 帧和全部 4 个段。
- 镜头合并闭包：帧 `180..299`、段 `1,2,3`，实际重算集合与闭包完全一致，闭包外 180 帧记录哈希不变。
- 音频偏移注入：`96` 样本，检出 `2.0 ms`，首越界段 `0`，增益比约 `1.05`，波形相似度约 `0.9999996`。
- 强制终止前提交点：`before_manifest` 返回码 92，恢复上一版本并回收 11 个未提交文件。
- 强制终止后提交点：`after_manifest` 返回码 93，恢复新增已提交版本，孤儿数为 0，关闭标记未被伪造。

完整结构化结果见 `artifacts/validation_report.json`。
