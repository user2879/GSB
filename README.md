# AED 分片标定与漂移诊断工作台

这是一个完全离线的音频事件检测（AED）分片概率序列工作台。它把旧检测器的未归一化整数对数几率、新检测器的归一化概率、不同类别码表和不同帧网格统一到同一类别语义、同一时间轴和同一标定尺度，并输出判决、边界样本、置信度漂移、来源归因、失败清单、增量复用账本和阶段快照。

工作台不是简单区间拼接：每个分片先按码表条目名映射类别，再映射到全局帧轴，随后使用由基准分片唯一推导的共享标定参数。重叠分片会在全局帧级做确定性后验融合，漂移诊断则逐帧对照基准分片的 float64 完整标定后验。

## 快速开始

要求 Python 3.11+、numpy、pytest。所有样本和测试均为本地确定性生成，不需要网络、账号、外部服务或容器。

```bash
python3 --version
python3 -c "import numpy, pytest; print(numpy.__version__, pytest.__version__)"

# 常规样本：3 个 16 kHz 单声道 16-bit PCM WAV，21 个交错分片
python3 -m aedwb generate --out samples/base

# 故障注入样本：在相同基础样本上增加 5 个探针分片和 injections.json
python3 -m aedwb generate --out samples/injected --injected

# 全量运行并摘要输出
python3 -m aedwb run --sample-dir samples/base --state-dir state/base --summary
python3 -m aedwb run --sample-dir samples/injected --state-dir state/injected --summary

# 完整诊断报告写到 state/<name>/report.json；再次运行同一命令验证复用
python3 -m aedwb run --sample-dir samples/base --state-dir state/base --summary

# 离线测试
python3 -m pytest -q
```

`generate` 默认清理目标目录；加 `--no-clean` 可保留。`run` 的 `--state-dir` 默认是 `<sample>/.workbench`。CLI 另有 `--report PATH` 指定完整报告位置；`--summary` 只打印关键计数。

## 目录

```text
aedwb/
  loader.py       分片发现、注入、严格原始校验
  mapping.py      码表条目名映射与显式未映射记录
  alignment.py    全局帧轴线性重采样、原帧/全局帧溯源与残差
  calibration.py  基准分位标定、自然指数换算、概率和校验
  drift.py        float64 基准后验、绝对/KL 型漂移与来源归因
  decision.py     量化、阈值判决、固定优先级边界决断、重叠融合
  cache.py        内容寻址阶段对象缓存
  snapshot.py     分块阶段快照、临时目录整体替换、恢复判定
  pipeline.py     并发流水线、账本、失败隔离、诊断报告
  samplegen.py    固定种子 WAV/分片/配置/真值/注入生成器
tests/            10 项离线测试
scripts/generate_samples.py  generate 子命令的等价包装
reports/verification.json    本次实际验收摘要
```

## 输入契约

样本目录布局：

```text
sample/
  audio/*.wav
  packets/*.json
  codebooks.json
  config.json
  baseline_packets.json
  truth.json
  injections.json          # 可为空注入列表
```

### `config.json`

关键字段如下（生成器提供完整可运行版本）：

```json
{
  "schema_version": "aed-global-config/1",
  "global_version": "g-2026-11-21.1",
  "frame_shift_ms": 20,
  "classes": ["speech", "music", "noise", "alarm"],
  "class_codebook_version": "global-v2",
  "class_codebook": [
    {"code": 0, "name": "speech"},
    {"code": 1, "name": "music"},
    {"code": 2, "name": "noise"},
    {"code": 3, "name": "alarm"}
  ],
  "class_priority": ["alarm", "speech", "music", "noise"],
  "calibration_quantiles": [0.10, 0.90],
  "calibration_tolerance": 1e-9,
  "drift_alert_threshold": 0.05,
  "decision_threshold": 0.40,
  "probability_quantum": 0.05,
  "thread_pool_workers": 4,
  "snapshot_commit_interval": 2,
  "interaction_budget_ms": 3000
}
```

### 分片结果包

每个 `packets/*.json` 是一个结果包：

```json
{
  "schema_version": "aed-packet/1",
  "packet_id": "a0_base_legacy",
  "audio_file": "clip_000.wav",
  "start_ms": 2000,
  "end_ms": 5000,
  "frame_shift_ms": 20,
  "score_type": "logit",
  "class_codebook_version": "legacy-v1",
  "frame_grid_version": "fg20-v1",
  "detector_version": "legacy-logit-1.4",
  "frames": [[-11, -7, -3, 18]]
}
```

`score_type` 只能是 `logit` 或 `probability`；旧 `logit` 值必须为整数。`frames` 的列顺序由该分片的码表条目 `code` 升序确定。帧数必须严格等于 `(end_ms-start_ms)/frame_shift_ms`，区间必须位于对应 WAV 时长内。全局时间轴是“同一 `audio_file` 内的时间轴”；不同 WAV 的帧分别归属其文件。

`codebooks.json` 保存历史码表。生成样本中的旧码表是 `music, noise, alarm, speech`，与全局码位顺序不同，因此只按码位拼接会得到错误结果；实现始终按 `name` 映射。

`baseline_packets.json` 的 `baseline_packet_ids` 是标定与漂移参考的唯一来源。`truth.json` 只用于运行后的合成事件评估，不会参与标定参数推导。

## 处理算法

### 1. 解析与失败隔离

- 校验 `packet_id` 唯一、WAV 存在且为 16 kHz/mono/16-bit PCM、区间合法、帧数严格一致、类型标记存在且合法、四列数值均有限、旧 logit 为整数、概率位于 `[0,1]`。
- 任一失败产生 `E_PACKET_INVALID`，保留错误明细并跳过该分片的下游阶段；其余分片继续。
- 有效分片在报告和阶段产物中保留 `raw_scores`、`raw_score_type` 与 `raw_score_provenance`。

### 2. 语义码表映射

- 分片码表版本等于全局版本时使用全局码表；旧版本从 `codebooks.json` 读取。
- 逐条目按 `name` 查找全局类别索引，绝不按位置推断。
- 任一编码无法映射时，该帧四类后验置为零向量，记录原码位、原始分数、码表版本，并产生 `E_CLASS_UNMAPPED`。零向量造成的概率和偏差会进一步产生 `E_CALIB_RESIDUAL`，两码同时可审计。

### 3. 帧轴重对齐

全局目标帧为 `k * global_frame_shift_ms`，仅取落在 `[start_ms,end_ms)` 的全局时间戳。固定规则：

- 在局部帧时间戳之间做线性插值；
- 源支撑边缘采用端点保持；
- 每行保留 `original_frame_indices`、`global_frame_index`、`global_time_ms`、插值权重；
- 每行以概率空间绝对差记录 `alignment_residual`；
- 输出 `edge_hold_frame_count`、`uncovered_intervals_ms` 和重采样前后帧数。因此不会静默截断、补零或丢弃边缘覆盖信息。

### 4. 统一标定

只由基准分片与配置推导一次参数，所有分片共用：

1. 旧 `logit` 使用自然指数 softmax；概率值先按行 L1 归一化。
2. 从基准分片映射后帧收集每类分数，按配置分位点计算 `low/high`。
3. 所有分片逐类裁剪到 `[low,high]`、仿射缩放到 `[0,1]`，再逐行归一化。
4. 参数与基准分片上游指纹共同生成 `calibration_fingerprint`，写入每个分片、报告和缓存键。
5. 非未映射帧的标定概率和必须在 `calibration_tolerance` 内；越界记录 `E_CALIB_RESIDUAL`。

### 5. 漂移量化与来源归因

- 基准参考是基准分片标定后验的 float64 等权逐帧均值，形成 `drift-reference` 指纹。
- 每个分片每类输出绝对误差分布、KL 贡献分布、均值方向、最大误差帧及其正负方向。
- 实际机械残差分量分别计算为：
  - `logit_conversion`：logit 自然指数与最终标定之间的平均转换残差；
  - `frame_resampling`：对齐阶段逐帧概率空间插值残差；
  - `codebook_mapping`：按名映射前后概率空间差异；
  - `confidence_shift`：分片后验相对基准后验的平均绝对差，作为独立诊断量同时输出。
- 告警来源取三个机械残差中实际数值最大的一个；三者数值均为零时才标为 `confidence_shift`。来源依据是残差向量，而不是“旧版本/新版本”标记。

### 6. 判决与边界样本

- 标定概率按 `probability_quantum` 以 half-up 规则量化，量化后分数用于判决。
- 最高分低于 `decision_threshold` 时判决为 `none`。
- 若第一名与第二名差不超过半个量化档位，或最高分恰等于阈值，则记为边界样本。
- 边界时在 `[top - quantum/2, top]` 内按固定 `class_priority` 选类，保证结果不依赖数组顺序或线程完成顺序。
- 全局帧融合：同一 `audio_file + global_frame_index` 的所有有效分片标定后验按稳定分片顺序等权平均，再执行同一量化/阈值/优先级规则。

## 增量缓存、并发与快照

缓存键由上游阶段指纹、内容哈希、标定指纹、算法版本和 `global_version` 派生：

| 变更 | 失效范围 |
|---|---|
| 替换某个非基准分片 | 该分片 parse/map/align/calibrate/decide/drift；其他分片阶段复用 |
| 替换基准分片 | 标定参数变化，所有分片 calibrate/decide/drift 重算；parse/map/align 可复用 |
| 修改标定分位上下限或容差 | 标定指纹变化，calibrate/decide/drift 重算 |
| 修改判定阈值或量化档位 | 仅 decide 重算 |
| 修改漂移告警阈值 | 仅 drift 重算 |
| 修改 `global_version` | 所有缓存键变化并全量重算 |

每阶段按配置的 `snapshot_commit_interval` 分块写入临时目录，再用目录整体替换提交到 `state/snapshots/<stage>/`。快照清单包含全局版本、配置哈希、标定参数指纹、分片头版本集合、已完成分片标识、记录数和缓存键。最终 `state/manifest.json` 还包含失败分片与一致性签名。

恢复时比较已有清单的 `global_version` 与完整配置哈希；任一不一致会在报告的 `run.snapshot_status` 中输出 `E_SNAPSHOT_STALE`。若最终清单缺失但阶段清单完好，状态为 `partial_resume`，仍通过精确缓存键恢复。内容寻址缓存会复用键未变化的上游阶段，因此“快照陈旧告警”和“局部增量复用”可以同时成立。

线程池最大并发严格取自配置。所有任务携带 `[packet_index]` 稳定键，帧携带 `[packet_index, global_frame_index]` 键；任务完成顺序只影响等待顺序，输出始终先收集再排序。串行与并发运行的一致性签名、逐帧结果、漂移和失败顺序相同。

## 故障注入

`samples/injected/injections.json` 包含 5 种独立注入，命中编号会写入报告的 `injection_hits`：

- `truncated_packet`：真实截断 JSON，期望 `E_PACKET_INVALID`；
- `frame_count_mismatch`：删除一帧，期望 `E_PACKET_INVALID`；
- `missing_score_type`：删除类型标记，期望 `E_PACKET_INVALID`；
- `nonfinite_logit`：旧 logit 写入非有限值，期望 `E_PACKET_INVALID`；
- `unmappable_codebook`：替换为未知码表版本，期望 `E_CLASS_UNMAPPED`，并因零向量额外记录 `E_CALIB_RESIDUAL`。

注入只作用于目标探针分片，其他分片、全局帧和判决继续产出。

## 诊断报告

完整报告默认位于 `state/<name>/report.json`，主要字段：

- `run`：版本、配置哈希、标定指纹、快照恢复状态、实际并发数、交互预算与实际耗时；
- `calibration`：共享参数、基准分片清单、参考指纹；
- `packets[]`：原始分数/类型来源、码表映射条目、重采样元数据、逐帧标定概率、逐帧判决、逐分片漂移；
- `global_frames[]`：融合后验、来源分片、量化分数、判决、边界原因；
- `boundary_samples[]`、`decision_counts`、`drift_alerts[]`；
- `failures[]`：稳定排序的业务码、阶段、分片、错误明细和注入编号；
- `injection_hits[]`、`truth_evaluation`；
- `ledger.entries[]`：每个分片每阶段缓存键、`reused/computed/failed/skipped` 状态、码表条目、基准引用；
- `snapshots` 与 `consistency_signature`。

## 实际验证结果

本次在 2026-11-21 执行，摘要也写入 `reports/verification.json`。

- 环境：Python 3.11.0rc1、numpy 1.26.4、pytest 8.3.3；未联网、未使用 Docker。
- `python3 -m pytest -q`：`10 passed in 28.65s`。
- 基础样本：21/21 分片有效，750 个全局帧，0 个业务失败，84 条漂移告警，74 个边界样本。漂移来源计数为 `codebook_mapping=24`、`confidence_shift=24`、`frame_resampling=24`、`logit_conversion=12`。
- 基础样本一致性签名：`c3ccc0da0389af914b2fa3cb186a5a1c92b6e324f492017a4a3550ef8bb81d00`。
- 同状态再次运行：0 个阶段重算、126 个阶段复用，快照状态 `resumed`，一致性签名不变。
- 注入样本：26 个分片、22 个有效、750 个全局帧；6 条失败记录为 `E_PACKET_INVALID=4`、`E_CLASS_UNMAPPED=1`、`E_CALIB_RESIDUAL=1`；5 个注入全部命中且期望码均观察到。
- 测试还验证了：替换非基准分片只重算该分片六阶段、其他分片全部复用；增量结果与空白全量运行的逐帧结果/漂移/边界/失败顺序和一致性签名相同；阈值/量化档位只重算 decision；`global_version` 变化全量重算；串行 `workers=1` 与并发 `workers=8` 的业务签名和账本缓存键一致。

## 环境说明与解释边界

- 任务文字要求确认 Windows；本执行容器实际是 WSL2 Ubuntu-22.04。实现刻意避免 POSIX-only 代码，使用 `pathlib`、`os.replace` 和 Python 标准库，目录交换也兼容 Windows 的“先移旧目录再移新目录”流程；但本次实际测试是在 WSL2 完成的，未在原生 Windows 进程上复测。
- 码表列顺序的契约是“按条目 `code` 升序”；若真实上游使用其他物理列顺序，应在解析适配层明确给出列到 code 的映射。
- 未映射帧按任务要求保持全零，因此其概率和不为 1；这既产生 `E_CLASS_UNMAPPED`，也按同一规则产生 `E_CALIB_RESIDUAL`，不会静默归一化。
- `consistency_signature` 覆盖标定参数、逐分片/逐帧业务产物、判决、漂移、边界、失败和注入命中；不覆盖运行耗时、缓存复用状态和快照历史，因为这些会随执行历史变化。
- 缓存状态是运行态审计信息；空白全量与增量运行的业务字段和账本顺序一致，但 `reused/computed` 状态必然按本次执行历史变化。
