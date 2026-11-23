# 说话人音色分析诊断工作台

本项目面向音色克隆与发音建模场景，使用纯 Python + NumPy 实现一个本地、确定性、命令流驱动的说话人音色分析与诊断工作台。样本音频由固定种子脚本合成，不依赖网络、账号、真实录音、Docker 或其他容器环境。

核心交付包括：

- 严格 WAV 校验、双门限迟滞发音段切分、段内容指纹。
- 基频统计、LPC 共振峰、语速节奏与有效帧数汇聚。
- 逐维自适应量化档位，物理改变整数码、量化解码值与描述向量。
- 相邻段归属、标注一致性冲突记录、字典序阈值相等决断。
- 有效帧加权定点说话人向量聚合与单段退化标记。
- 四阶段缓存键、上游依赖、精确增量复用与失效清单。
- 首响预算 pending 摘要、受限线程池和稳定排序键。
- 增量/全量语义一致性比较、原子阶段快照、恢复与 stale 判定。
- 结构化 JSON 报告与零外部依赖的静态 HTML 可视化工作台。

## 环境与安装

Windows 10/11，Python 3.11 或 3.12：

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

也可以直接安装本包：

```powershell
python -m pip install -e .
timbre-workbench --help
```

## 快速开始

PowerShell：

```powershell
.\scripts\run_demo.ps1
```

手动执行：

```powershell
python -m timbre_workbench demo --out examples/demo
python -m timbre_workbench run `
  --input examples/demo/audio `
  --config examples/demo/config.json `
  --labels examples/demo/labels.json `
  --commands examples/demo/commands.json `
  --work examples/demo/.workbench `
  --report examples/demo/report.json `
  --html examples/demo/report.html `
  --responses examples/demo/responses.json
python -m timbre_workbench compare `
  --input examples/demo/audio `
  --config examples/demo/config.json `
  --labels examples/demo/labels.json `
  --commands examples/demo/commands.json `
  --output examples/demo/compare.json
```

续跑并验证快照：

```powershell
python -m timbre_workbench run ... --resume --expected-version 3
```

版本号、基础配置哈希、标签哈希、命令前缀或媒体哈希不一致时，报告 `diagnostics.snapshot_warnings` 输出 `E_SNAPSHOT_STALE`，随后忽略旧缓存并从空白重算。

## 目录结构

```text
timbre_workbench/
  wavio.py       严格 RIFF/WAVE 校验、扫描、确定性写入
  segment.py     帧切分、双门限迟滞、最短段/间隙、段内容指纹
  features.py    F0 自相关、LPC 共振峰、语速、确定性分位数
  quantize.py    逐维动态范围、自动/强制档位、单元中心量化
  assign.py      相邻相似度、阈值相等字典序、冲突与候选
  aggregate.py   定点整数加权聚合、单段退化标记
  pipeline.py    四阶段并行计算、缓存键、依赖、稳定归并
  snapshot.py    临时目录写入、目录提交、原子 CURRENT 指针
  engine.py      命令状态机、首响 pending、增量/全量、报告
  render.py      静态 HTML 诊断仪表盘
  synth.py       固定种子 WAV/标签/命令流生成器
  cli.py         demo / run / compare 命令
tests/           13 项单元与集成测试
examples/demo/   可直接运行的确定性样本、报告和对比证据
docs/            架构与验证记录
scripts/         Windows PowerShell / Bash 一键验收脚本
```

## 输入与配置

输入目录递归扫描全部 `.wav`。每个有效文件必须是 RIFF/WAVE、16 kHz、单声道、16-bit PCM，RIFF 长度、chunk 边界、块对齐、字节率与 data 长度必须一致。坏文件记录 `E_AUDIO_FORMAT` 后继续处理其余文件。

标签文件将说话人标注绑定到采样区间，切分/合并后以区间覆盖量最多者继承标注，平票按说话人标识字典序：

```json
{
  "files": {
    "voice_alpha.wav": [
      {"start_sample": 1920, "end_sample": 10240, "speaker_id": "spk_alpha"}
    ]
  }
}
```

配置由 `timbre_workbench/config.py` 提供完整默认值；关键段落包括：

- `segmentation`：帧长/帧移、起始/延续能量门限、最短段、最短间隙。
- `features`：F0 范围、自相关浊音阈值、LPC 阶数、分位数与固定插值规则。
- `quantization.dimensions`：维度来源、候选位宽档位与允许步长。
- `assignment`：`rbf`/`cosine`/`normalized_l1` 相似度、阈值、字典序 tie-break。
- `aggregation`：定点缩放与 `half_away_from_zero` 舍入。
- `runtime`：首响预算、操作系统调度容差、线程上限、快照策略。

## 命令流

`commands.json` 形如：

```json
{
  "schema_version": 1,
  "commands": [
    {"op": "split_segment", "file": "voice_alpha.wav", "ordinal": 0, "at_sample": 6200},
    {"op": "replace_segment_audio", "file": "voice_alpha.wav", "ordinal": 1,
     "source": "variants/alpha_split_replacement.wav"},
    {"op": "set_quantization_tier", "dimension": "f0_mean", "tier": 4},
    {"op": "set_similarity_threshold", "value": 0.95},
    {"op": "merge_segments", "file": "voice_alpha.wav", "ordinals": [0, 1]},
    {"op": "set_segment_boundary", "file": "voice_beta.wav", "ordinal": 1,
     "edge": "end", "sample": 16960},
    {"op": "noop"}
  ]
}
```

支持：`split_segment`、`merge_segments`、`set_segment_boundary`、`replace_segment_audio`、`replace_audio`、`set_quantization_tier`、`set_similarity_threshold` / `set_threshold`、`set_assignment_metric`、`noop`。命令失败记录 `E_COMMAND_INVALID`，游标仍前进，保证相同命令序列可重放。

## 确定性与一致性规则

- 全部集合按 `file.casefold()`, `file`, 采样位置或说话人标识排序。
- 段特征只累加完全落在本段内的帧，帧序号升序；无完整帧时使用单个裁剪帧。
- 分位数固定排序并通过配置选择 `linear`、`lower`、`higher` 或 `midpoint` 插值。
- 自适应量化以当前全部段在该维的最小/最大值为动态范围；候选档位中选择满足 `max_step` 的最小位宽，也可由命令强制档位。
- 描述值取量化单元中心 `(code + 0.5) / 2^bits`。因此切换位宽时，即使值处于动态范围端点，整数码、解码值和描述向量仍真实变化。
- 相似度恰等于阈值时，比较前一段说话人标识与当前标注标识的字典序；结果与冲突均写入报告，标注不会静默覆盖预测。
- 说话人向量使用 `int(round_half_away(value * fixed_scale)) * valid_frame_count` 累加，再以整数除法确定性舍入。
- 每个产物键由内容指纹、相关配置指纹、上游键、量化档位/步长与阶段名共同派生。
- `dependencies` 持久记录每段与每说话人的上游产物；`invalidation_reuse_manifest` 是路径无关的最终键清单，参与增量/全量语义比较。
- 实际执行期的 `diagnostics.cache_events` 单独记录本次命中、失效和重算键；它是执行轨迹，不参与语义等价断言。

## 首响与并发

每条命令先应用编辑、递增版本与游标，再由协调线程启动计算。主线程最多等待配置预算；未完成时立即产出最近一次稳定摘要，并附当前受影响段与实时进度，错误码为 `E_FIRST_RESPONSE_PENDING`。批处理 CLI 随后等待稳定状态以便落盘和续跑，但响应对象记录的是截止时刻的首响语义。

特征、描述、归属与聚合四个阶段均使用线程池。任务携带稳定排序键，完成顺序不会影响输出；`max_workers` 是硬上限，报告记录实测峰值。最终段、说话人、冲突和失败清单均按稳定键排序。

## 快照与恢复

每次稳定后：

1. 在 `work/tmp/snapshot-*` 写入该阶段全部媒体与 `snapshot.json`。
2. 将完整临时目录重命名到 `work/snapshots/snap-*`。
3. 原子替换 `work/CURRENT.json` 指针。

快照清单记录全局版本、基础/有效配置哈希、标签哈希、量化计划与逐维档位、命令游标、命令前缀哈希、媒体 SHA 与各产物版本。只有版本、哈希和媒体均匹配才续跑，否则 `E_SNAPSHOT_STALE` 并从头重算。保留最近 `runtime.snapshot_keep` 个快照。

## 报告

`report.json` 的主要字段：

- `segments`：采样区间、内容指纹、特征键、统计量、描述键、整数码/档位/步长/残差、归属键、相似度、抑制候选、上游依赖。
- `speakers`：加权参数、定点分子、权重分母、向量、聚合键、退化标记、重算依据。
- `conflicts`、`failures`：稳定排序。
- `quantization`：逐维动态范围、位宽、档位、步长与选择模式。
- `dependencies`、`cache_manifest`、`invalidation_reuse_manifest`、`product_versions`。
- `diagnostics`：首响、线程、任务顺序、快照警告、执行期缓存轨迹。
- `canonical_digest`：增量/全量必须一致的语义投影摘要。

`report.html` 提供 Speaker 向量条形图、逐维量化、段级明细过滤、冲突/失败清单、缓存与线程诊断。`responses.json` 的每条稳定命令响应还带 `stable_state.speaker_vectors`、当前档位/阈值与规范摘要，可直接观察参数编辑过程中的说话人向量演进。

## 错误码

| 错误码 | 含义 |
|---|---|
| `E_AUDIO_FORMAT` | WAV 容器、格式、采样率、声道、位深或长度不合规 |
| `E_COMMAND_INVALID` | 命令选择器、边界、档位或参数无效 |
| `E_FIRST_RESPONSE_PENDING` | 首响预算内未完成最新参数下的稳定结果 |
| `E_SNAPSHOT_STALE` | 快照版本/配置/标签/命令前缀/媒体不一致，已从头重算 |
| `E_SNAPSHOT_WRITE` | 原子快照写入失败 |
| `E_CONFIG_INVALID` | 配置不合法 |
| `E_ASSIGNMENT_CONFLICT` | 向量预测归属与标注一致性冲突记录 |

## 验证

执行：

```powershell
python -m unittest discover -s tests -v
python -m timbre_workbench compare --input examples/demo/audio --config examples/demo/config.json --labels examples/demo/labels.json --commands examples/demo/commands.json --output examples/demo/compare.json
```

测试覆盖严格 WAV 失败隔离、双门限合并/丢弃、F0/共振峰、确定性、档位真实变化、阈值相等字典序、冲突记录、定点聚合、退化说话人、定向复用、首响 pending、线程上限、快照续跑、版本/配置 stale 和增量/全量逐字段语义一致。实际结果见 [docs/VALIDATION.md](docs/VALIDATION.md)。
