# timbre-lab：可编辑说话人音色分析

这是一个纯本地、确定性的说话人音色分析示例，面向音色克隆与发音建模的前处理环节。它扫描 16 kHz / 单声道 / 16-bit PCM WAV，切分发音段，计算段级声学特征和自适应量化音色描述向量，校验相邻段归属，按有效帧数聚合说话人向量，并支持交互编辑后的精确增量重算与快照恢复。

运行环境仅需要 Python 3.10+ 和 NumPy 1.24+，不使用 GPU、网络、账号或容器。

## 快速开始

```bash
python3 -c "import numpy; print(numpy.__version__)"   # 仅需 NumPy 1.24+
python3 -m timbre_lab synth --out examples/demo --seed 20261121
python3 -m timbre_lab demo --out artifacts/demo
python3 -m pytest -q
```

`demo` 会在 `artifacts/demo/input` 生成固定种子样本、配置、命令流和异常清单，并在 `artifacts/demo/run` 执行完整流程。也可以分别执行：

```bash
python3 -m timbre_lab synth --out examples/demo --seed 20261121
python3 -m timbre_lab run \
  --samples examples/demo/samples \
  --config examples/demo/config.json \
  --commands examples/demo/commands.json \
  --output artifacts/demo-run
```

再次运行同一条 `run` 命令会读取 `CURRENT` 指针并续跑；配置哈希或命令流不一致时返回业务码 `E_SNAPSHOT_STALE` 和进程退出码 2。需要强制从输入音频重建时加 `--fresh`。

## 工作区

```text
timbre_lab/
  audio.py          RIFF 严格校验与 WAV 读写
  segmentation.py   双门限迟滞切分及最小时长清理
  features.py       F0、LPC 共振峰、语速、分位数与内容指纹
  quantization.py   逐维自适应量化与双档位参考
  assignment.py     相邻段相似度、标注冲突、边界决断
  aggregation.py    定点整数加权说话人聚合
  pipeline.py       四阶段缓存、线程池和全量比对
  commands.py       切分、合并、改标注、替换区间、改档位
  engine.py         预算内预览、后台命令会话、快照提交
  snapshot.py       不可变快照目录、原子 CURRENT 指针和恢复校验
  report.py         诊断报告
  synth.py          固定种子合成器和异常样例生成器
tests/              16 个单元/集成/CLI/信号测试
examples/demo/      已生成的样本、配置、命令和异常注入清单
```

## 输入协议

配置的关键字段如下。完整样例见 `examples/demo/config.json`。

```json
{
  "sample_rate": 16000,
  "channels": 1,
  "sample_width_bits": 16,
  "frame_length_ms": 25.0,
  "frame_shift_ms": 10.0,
  "energy": {"enter_db": -44.0, "exit_db": -54.0},
  "min_segment_ms": 90.0,
  "min_gap_ms": 60.0,
  "quantile_interpolation": "linear",
  "f0": {"min_hz": 60.0, "max_hz": 500.0, "voiced_autocorr": 0.24},
  "timbre_dimensions": [
    {
      "name": "f0_mean",
      "metric": "f0_mean_hz",
      "dynamic_range": [70.0, 360.0],
      "target_step": 3.0,
      "bit_widths": [4, 6, 8, 10]
    }
  ],
  "similarity": {"threshold": 0.82, "weights": {"f0_mean": 2.4}},
  "preview_budget_ms": 200.0,
  "max_workers": 4,
  "snapshot_commit_every": 1,
  "verify_incremental_vs_full": true,
  "initial_annotations": [
    {"file": "voice_dialogue.wav", "contains_ms": 430.0, "speaker": "spk_amber", "style_hint": "slow"}
  ]
}
```

每维允许的档位由 `bit_widths` 表示。未强制覆盖时，量化器在允许位宽中选择满足 `dynamic_range / (2^bits - 1) <= target_step` 的最小位宽；该规则只依赖该维配置，因此改变其他段的内容不会让未失效段重新量化。实际使用位宽可以随维配置或编辑命令改变。

编辑命令文件是命令对象的数组，也接受 `{"commands": [...]}`。选择器支持 `segment_id`、`index`、`contains_ms`、`start_ms`。示例：

```json
[
  {"id": "s1", "op": "split", "target": {"file": "voice_dialogue.wav", "contains_ms": 430}, "at_ms": 550},
  {"id": "m1", "op": "merge", "target": {"file": "voice_dialogue.wav", "contains_ms": 300}, "direction": "next"},
  {"id": "a1", "op": "set_speaker", "target": {"file": "voice_dialogue.wav", "contains_ms": 2200}, "speaker": "spk_noah"},
  {
    "id": "r1",
    "op": "replace",
    "target": {"file": "voice_dialogue.wav", "contains_ms": 3000},
    "interval": {"start_ms": 100, "end_ms": 350},
    "source": "assets/replacement_noah.wav"
  },
  {"id": "q1", "op": "set_quantization", "dimension": "f0_mean", "bit_width": 4}
]
```

替换源必须是相同格式 WAV，且采样数必须精确等于目标区间；未知维度、越界/不允许位宽、非法切分点、替换长度不匹配等均返回业务码并原子回滚当前命令。

## 处理语义

**音频校验与切分**：严格校验 RIFF 长度、`fmt ` / `data` 块、PCM=1、16 kHz、单声道、16-bit、块对齐、字节率与偶数数据长度。坏文件记入失败清单但不会阻断其他文件。切分使用进入/退出双门限迟滞，再按最短段和最短间隙反复合并/丢弃。每段输出半开区间 `[起始帧, 结束帧)`、毫秒边界和有效帧数；结束帧与段内实际完整帧数严格一致。

**段级特征**：只遍历该段自己的帧，帧序严格升序。F0 使用归一化自相关与抛物线细化；共振峰使用 LPC 根轨迹并带频谱峰值回退；语速由段内能量包络峰率得到。支持 `linear/lower/higher/midpoint/nearest` 分位数插值。内容指纹只对该段原始采样区间做 SHA-256，段外数据不会进入特征、指纹或缓存键。

**量化**：每维输出选中位宽带、档位数、量化步长、整数、归一化值、重建值和残差。相同内容、配置和档位重复运行时逐字段一致。

**归属判定**：逐文件比较相邻段描述向量的加权归一化 L1 相似度。`similarity >= threshold` 判为同说话人；恰好相等时输出 inclusive/exclusive 双参考，并按段起始帧升序记录决断顺序。显式说话人标注拥有固定优先级：标注与向量冲突时保留标注、输出 `E_ANNOTATION_VECTOR_CONFLICT`，不会静默覆盖。边界或冲突会额外输出当前档位及相邻参考档位的 `dual_level_reference`。

**说话人聚合**：显式标注全局聚合；无标注段可用向量回退连接，禁止无标注段桥接两个不同的显式说话人。每维按有效帧数做定点整数加权，使用 nearest-half-up 整数舍入。输出参与段、权重、总权重、每维整数和归一化结果；单段说话人带 `degenerate_single_segment: true`。

## 增量与并发

四类缓存键分别由以下稳定输入派生：

| 阶段 | 缓存键输入 |
| --- | --- |
| 段特征 | 段内容指纹、段绑定（文件/起止采样/起始帧）、帧/能量/特征配置指纹 |
| 段描述 | 特征缓存键、维度配置、所选位宽与命令覆盖 |
| 归属判定 | 左右描述键、阈值、权重、两侧标注、段起始帧 |
| 说话人聚合 | 说话人、段清单、描述键、有效帧权重、量化配置 |

切分或合并只让新旧段特征失效；替换只让目标段特征失效；改标注复用特征和描述，只重算相关归属和聚合；改量化档位复用全部特征，重算描述、归属和聚合。命令结果记录 `cache_audit.reused/recomputed`，未失效产物直接取缓存。

特征、描述和归属任务提交到 `max_workers` 指定的线程池。每个任务带 `filename|start_frame|end_frame|phase` 稳定排序键，结果按该键收集，而不是按完成时刻；因此并发与单线程稳定键执行逐字段一致。最终还会调用无缓存全量管线比较分段、统计量、量化、归属、聚合、冲突和失败清单。

每条命令通过后台命令线程执行。调用方最多等待 `preview_budget_ms`：完成则返回最新完整结果，否则立即返回 `E_PREVIEW_BUDGET_PENDING`、最近一次已完成摘要、受影响段、当前阶段和百分比进度；后台继续完成并提交快照。CLI 批处理会在后续命令前等待当前命令，但仍返回最初预算响应，便于检查语义。

## 快照与恢复

每次提交先在工作目录下写完整 staging 目录，fsync 后将整个目录重命名到不可变版本目录，最后以临时文件 + `os.replace` 原子替换 `snapshots/CURRENT`。任何时刻只有 `CURRENT` 指向的完整目录有效；强制终止遗留的 staging 或未指向目录不会被恢复。

`manifest.json` 记录 schema/引擎版本、全局版本、配置哈希、命令游标、完整命令流哈希、已完成段清单和各阶段产物哈希。启动时校验 schema 版本、配置哈希、命令流哈希和产物哈希。不一致分别返回 `E_SNAPSHOT_STALE` 或 `E_SNAPSHOT_CORRUPT` 并拒绝复用。

已安装的 `SIGINT` / `SIGTERM` 处理器会让主循环停止提交后续命令，等待当前命令完成、提交快照并写出报告，随后以整数码 0 退出。无法捕获的强制终止只依赖最后一次完整 `CURRENT` 快照恢复。

## 输出

完整运行写出：

- `OUTPUT/report.json`：逐段统计、说话人归属、发音风格、冲突/边界、说话人向量、量化档位与残差、缓存失效复用、命令预算响应、快照和失败码。
- `OUTPUT/snapshots/<version>/`：state、四阶段缓存、当前编辑后音频、manifest。
- `OUTPUT/snapshots/CURRENT`：唯一有效快照指针。
- `artifacts/verification.json`：本次交付环境的测试与 demo 实测摘要。

失败码包括 `E_AUDIO_FORMAT`、`E_AUDIO_SAMPLE`、`E_CONFIG_INVALID`、`E_QUANTIZATION_LEVEL`、`E_COMMAND_INVALID`、`E_PREVIEW_BUDGET_PENDING`、`E_SNAPSHOT_STALE`、`E_SNAPSHOT_CORRUPT`、`E_INTERNAL`。单个音频或单条命令失败会隔离并继续；配置/快照启动失败返回非零退出码。

## 异常注入

`examples/demo/anomalies.json` 与同目录夹具覆盖：

- `bad_riff.wav`、`truncated_data.wav`：`E_AUDIO_FORMAT`，且不影响有效 WAV。
- `nonfinite_samples.json` / 测试中的 `NaN`、`Inf`：`E_AUDIO_SAMPLE`。
- `quantization_out_of_range.json`：`E_QUANTIZATION_LEVEL`，命令回滚。
- 相似度恰等于阈值：测试把实测相邻相似度设为阈值，断言双参考值和起始帧升序。

## 固定种子样本

合成器用 NumPy `default_rng(seed)` 生成平稳底噪、多谐波发音、共振峰包络、音节幅度调制和短渐变。两个说话人主要差异为 F0 与 F1/F2，风格差异为语速与 F0 起伏。默认样本包含跨说话人相邻段、同说话人跨风格相邻段和独立底噪间隙。

## 实际验证

本工作区已执行：

```text
python3 -m pytest -q
16 passed

python3 -m timbre_lab demo --out artifacts/demo --seed 20261121
status=complete, command_cursor=7, segments=8, speakers=2,
vector_dimensions=7, incremental_full_equal=true, failure_codes=[]
```

最终报告显示：量化位宽确随命令改变；改标注产生 2 条 `E_ANNOTATION_VECTOR_CONFLICT`；最后一个量化命令复用 8 个段特征、重算 8 个描述、6 个归属判定和 2 个说话人聚合。快照测试验证了完整恢复、配置哈希不一致时 `E_SNAPSHOT_STALE`，SIGTERM 场景下退出码 0 且存在完整快照。
