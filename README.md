# VoiceCal：确定性声学证据链与逐轮校准组件

这是一个不依赖 GPU、真实账号、外部语音数据或容器环境的完整本地示例。组件读取固定种子合成的 16 kHz 单声道 16 bit PCM WAV、对话脚本、校准配置和编辑命令流，完成：

1. RIFF/WAVE 校验、逐帧声学特征提取；
2. 固定点初始情感证据构建；
3. 按会话与轮次序号推进的证据链校准；
4. 基于校准证据的说话人归属与轨迹重建；
5. 精确失效、缓存复用和预算受限的增量推进；
6. 原子阶段快照、恢复判定、故障注入隔离与审计报告。

规范产物中的每个强度都是定点整数。校准链逐情感、逐轮显式运算，不依赖数组归约的默认顺序。相同最终输入和配置下，增量运行与全量运行的五类规范 JSON 产物逐字节一致。

## 目录

```text
voicecal/                  组件实现（无第三方语音库，仅 NumPy）
scripts/generate_samples.py 固定种子 WAV 与输入生成器
scripts/run_demo.py         增量/全量/并发/注入完整演示
scripts/bootstrap_env.sh    本地虚拟环境安装脚本
scripts/verify.sh           样本、测试、演示一键验证
samples/                    已生成样本、配置、编辑流、注入配置和预期业务码
tests/                      13 项单元与集成测试
docs/verification.json      最近一次完整验证的实际结果与哈希
```

运行期目录（不会预先提交）：`snapshots/`、`outputs/`、`demo_runs/`。快照目录由组件创建。

## 环境与运行

要求 Python 3.10+ 与 NumPy，使用本地虚拟环境：

```bash
bash scripts/bootstrap_env.sh
source .venv/bin/activate
```

本工作区实际验证使用 `.venv-local`（系统缺少 `ensurepip`，安装脚本会自动回退到 PyPA `get-pip.py`）。也可直接使用现有环境：

```bash
.venv-local/bin/python -m voicecal run \
  --fresh \
  --config samples/calibration.json \
  --dialogue samples/dialogue.json \
  --edits samples/edits.json \
  --snapshot-dir demo_runs/manual/snapshot \
  --output-dir demo_runs/manual/output
```

`--fresh` 忽略现有快照，直接从空白状态应用全部编辑批次。默认运行会复用同一快照。若预算耗尽，退出码为 `3` 且报告含 `E_CALIB_BUDGET_PENDING`；再次执行同一命令即可恢复续跑。

完整验收：

```bash
make verify
# 等价于：
# .venv-local/bin/python scripts/generate_samples.py
# .venv-local/bin/python -m unittest discover -s tests -v
# .venv-local/bin/python scripts/run_demo.py
```

## 输入契约

### 对话脚本

`samples/dialogue.json` 的每轮包含稳定唯一的 `turn_id`、全局递增 `turn_index`、`conversation_id`、相对音频路径、脚本说话人、候选说话人、前一轮衔接和轮间静音：

```json
{
  "turn_id": "t0003",
  "turn_index": 3,
  "conversation_id": "conv-a",
  "audio": "audio/t0003.wav",
  "speaker_id": "spk_a",
  "speaker_candidates": ["spk_a", "spk_b"],
  "previous_turn_id": "t0002",
  "gap_ms": 80.0,
  "link_type": "response"
}
```

每个会话内的 `previous_turn_id` 必须等于该会话前一轮。删除中间轮次时，组件会同步修复后继轮次的前向链接和会话位置。

### 校准配置

`samples/calibration.json` 的关键字段：

| 字段 | 含义 |
|---|---|
| `frame.length_ms` / `shift_ms` | 帧长与帧移，样本数按 16 kHz 换算 |
| `frame.pitch_min_hz` / `pitch_max_hz` | 自相关基频搜索范围 |
| `frame.voiced_correlation` | 清浊音判定阈值 |
| `frame.silence_floor_dbfs` / `onset_margin_db` | 静音与节奏起音阈值 |
| `evidence.initial_weights` | 情感 × 特征的非负权重 |
| `evidence.feature_ranges` | 特征归一化区间 |
| `calibration.gain` / `decay` | 上轮证据的增益与衰减 |
| `calibration.same_speaker_factor` / `different_speaker_factor` | 说话人连续/切换系数 |
| `calibration.speaker_switch_penalty` | 仅说话人标识变化时扣除的固定量 |
| `attribution.min_segment_ms` | 可归属片段的最短时长 |
| `attribution.min_silence_ms` | 同轨迹允许的最大直接静音 |
| `attribution.max_gap_turns` | 同轨迹允许跨过的空档轮数 |
| `attribution.speaker_profiles` | 音高中心、容差与情感亲和度 |
| `fixed_point.decimal_places` / `bit_width` / `quantization_levels` | 定点比例、位宽与量化档位 |
| `concurrency` | 线程池最大值，运行中不会超过 |
| `snapshot_commit_interval` | 每处理多少轮提交一次阶段快照 |
| `batch_budget` | 单批最多重算多少个失效轮次 |

配置会严格校验字段类型、范围、权重特征集合、情感集合和说话人配置；错误业务码为 `E_CONFIG_INVALID`。

### 编辑命令流

`samples/edits.json` 支持批次和五类命令：

```json
{"schema_version": 1, "batches": [
  {"batch_id": "round-1", "commands": [
    {"op": "replace_audio", "turn_id": "t0003", "audio": "replacements/t0003_v2.wav"},
    {"op": "adjust_gain", "turn_id": "t0005", "gain": 0.52}
  ]},
  {"batch_id": "round-2", "commands": [
    {"op": "modify_speaker", "turn_id": "t0006", "speaker_id": "spk_b"},
    {"op": "delete_turn", "turn_id": "t0007"},
    {"op": "append_turn", "turn": {"turn_id": "t0010", "turn_index": 10}}
  ]}
]}
```

`adjust_gain` 省略 `turn_id` 时修改全局增益，指定轮次时只覆盖该轮。所有命令在批次内应用完后重新做对话结构校验。

### 注入配置

`samples/injections_validation.json` 是独立输入。`corrupt_format` 在解析前真实截断 RIFF 字节；`force_exception` 在指定阶段抛出确定性业务故障。注入编号、阶段、轮次、结果和实际业务码写入报告。

## 处理语义

### 1. 音频与逐帧证据

WAV 解析器直接检查 RIFF 大小、WAVE 标识、`fmt ` 与 `data` 块边界、PCM 格式、单声道、16000 Hz、16 bit、字节率、块对齐、数据块长度和采样数。失败为 `E_AUDIO_FORMAT`，文件缺失为 `E_AUDIO_MISSING`。

每帧先补零后使用 Hann 窗。逐帧记录：

- 归一化自相关的前三个局部峰值基频候选、最佳候选与自相关值；
- RMS dBFS、线性能量、过零率、清浊音标记；
- 帧级 `speech_rhythm`：能量差分、活动标记、起音标记与阈值；
- `padded`、原始有效样本数、稳定任务键 `文件名|帧序号|extract.frame`。

片段聚合显式按帧序号累加，得到基频均值/标准差、能量均值/动态范围、发声比例、能量起音率（语速代理）与起音间隔变异系数（节奏代理）。算法与参数写入 `extraction.json`。

### 2. 初始情感证据

每类情感按字典序情感名、固定特征顺序显式计算加权项：

```text
raw_score[e] = Σ round(weight[e,f] * normalized_feature[f])
```

随后执行稳定 softmax：先减去最大 `raw_score`，再计算指数与归一化。结果按 `quantization_levels` 量化，并保存为 `10^decimal_places` 比例的整数。每段记录参与帧数、聚合特征、归一化固定点特征、原始分数、逐项和与最终定点强度。

### 3. 逐轮校准

同一会话内按 `turn_index` 前向推进。设 `S=10^decimal_places`、`I_t` 为初始证据、`C_{t-1}` 为上一成功校准结果、`G/D` 为增益/衰减，则对每个情感 `e`：

```text
base = I_t[e] + round(C_{t-1}[e] * G * D)
F    = same_speaker_factor  若说话人未变化
       different_speaker_factor 若说话人变化
value = round(base * F) - (speaker_switch_penalty 若说话人变化，否则 0)
C_t[e] = quantize(clamp(value, 0, S))
```

乘法每一步立即定点取整，全部按情感字典序执行。说话人切换标记、实际系数、惩罚与各中间项写入账本。最终情感取校准向量最大项，同分按情感标签字典序。

### 4. 归属与轨迹

候选说话人得分是以下固定点加权和：

- 校准情感向量与说话人情感亲和度的点积；
- 片段基频均值与说话人音高中心的容差匹配；
- 脚本说话人先验。

同分按说话人标识字典序。片段短于 `min_segment_ms` 时产生 `E_SEGMENT_TOO_SHORT`，不进入轨迹。

同会话轨迹连接优先级为：说话人变化断开 → 跨过空档轮数超过 `max_gap_turns` 断开 → 静音超过 `min_silence_ms` 断开 → 继续。首轮原因记为 `conversation_start`。轨迹输出标识、轮次序列、说话人、每轮连接结果和断开原因。

### 5. 缓存键与精确失效

四阶段缓存键严格串联：

```text
extract     = H(文件相对路径, 音频SHA-256, 帧参数, 算法版本, 定点位宽, 提取阶段注入)
initial     = H(extract键, 证据权重/区间/归一化, 定点比例, 量化档位, 初始阶段注入)
calibration = H(初始键, 上一校准键, 轮次/会话/说话人, 增益/衰减/切换系数, 定点参数, 校准阶段注入)
attribution = H(校准键, extract键, 候选说话人, 归属配置/轨迹前键, 归属阶段注入)
```

修改某轮音频只失效该轮 extract/initial，以及该会话该轮起的 calibration/attribution；修改增益或说话人只失效该会话该轮起的 calibration/attribution；删除轮次再失效结构性后继。其他阶段直接命中缓存，不调用计算函数。审计事件按 `turn_index`、阶段固定顺序排序，报告分别给出命中、重算/失效计数。

### 6. 预算与增量等价

每批按轮次序号扫描，只有至少一个阶段缓存未命中的轮次消耗一个预算单位。预算耗尽时：

- 当前及后续未完成轮次保留为 pending；
- 报告加入 `E_CALIB_BUDGET_PENDING`；
- 已处理结果与目标有效输入写入快照；
- 下次运行恢复同一目标批次和缓存池。

预算只影响推进次数，不影响最终规范产物。测试会将多次预算受限运行与空白快照全量运行的 `initial_evidence.json`、`calibration.json`、`attribution.json`、`calibration_ledger.json`、`failures.json` 做 SHA-256 全等比较。

### 7. 快照与恢复

快照采用 generation 目录：先在临时目录写完四个阶段 payload 和状态，再重命名 generation，最后原子替换 `manifest.json` 指针。manifest 的每个阶段记录：

```text
stage, config_hash, injection_hash, turn_cursor,
fixed_point_bits, artifact_version, payload, payload_hash
```

恢复时校验快照产物版本、产品版本和基础配置哈希；任一不符返回 `E_CONFIG_MISMATCH` 并拒绝续跑。编辑命令引起的变化属于目标状态变化，不通过修改基础配置绕过恢复校验。`--fresh` 是显式重建路径。

### 8. 并发与稳定排序

帧任务进入线程池，任务键为 `文件名|12位帧序号|extract.frame`。全量批次且预算覆盖整批时，片段提取和初始证据任务也通过线程池并行预取，共享同一并发上限；线程完成顺序不影响结果，所有输出按任务键或轮次序号重新排序。单轮异常记录文件名、轮次、阶段、业务码与异常类型，后续轮次继续，失败轮次不会进入归属轨迹。

## 输出

运行后 `--output-dir` 包含：

| 文件 | 内容 |
|---|---|
| `extraction.json` | 逐段校验、逐帧候选基频、能量、节奏代理与聚合特征 |
| `initial_evidence.json` | 初始定点证据强度及计算中间量 |
| `calibration.json` | 逐轮校准定点向量、情感判定、链摘要与系数 |
| `attribution.json` | 说话人分数、归属、轨迹、断开原因 |
| `calibration_ledger.json` | 每轮上游引用、缓存键、初始/校准证据、归属依据；规范可回溯账本 |
| `failures.json` | 稳定排序的失败清单 |
| `run_report.json` | 本次运行诊断、业务码、命中/失效计数、注入命中、并发与 pending |
| `effective_dialogue.json` / `effective_state.json` | 编辑应用后的最终输入快照 |

规范账本不记录“本次命中/重算”等运行差异；这些放在诊断报告中。因此增量最终账本可与全量直接比较。

## 业务码

| 业务码 | 场景 |
|---|---|
| `E_AUDIO_FORMAT` | RIFF/WAVE 或 PCM 参数不合规 |
| `E_AUDIO_MISSING` | 引用音频缺失 |
| `E_CONFIG_INVALID` | 配置或注入配置非法 |
| `E_CONFIG_MISMATCH` | 快照版本或基础配置哈希不符 |
| `E_CALIB_BUDGET_PENDING` | 预算耗尽，仍有待处理轮次 |
| `E_INJECTED_FAULT` | 注入故障命中 |
| `E_SEGMENT_TOO_SHORT` | 低于最短片段约束 |
| `E_DIALOGUE_INVALID` | 对话结构非法 |
| `E_EDIT_INVALID` | 编辑命令非法 |
| `E_SNAPSHOT_INVALID` | 快照缺失或不可读 |
| `E_INTERNAL` | 未预期单轮异常被隔离 |

## 实际验证

执行 `.venv-local/bin/python -m unittest discover -s tests -v`：13 项测试全部通过；覆盖逐帧确定性、末帧补零、RIFF 错误、定点重复性、切换惩罚、同分字典序、最短片段、预算 pending/恢复、增量全量等价、快照版本与配置哈希拒绝、注入命中排除、精确失效范围、并发上限。

执行 `.venv-local/bin/python scripts/run_demo.py` 的实际结果保存在 `docs/verification.json`，最近一次包括：

- 增量运行 4 次收敛，曾观察到 `E_CALIB_BUDGET_PENDING`；
- 增量与全量五类规范产物全部 SHA-256 一致；
- 4 线程与 1 线程五类规范产物全部 SHA-256 一致；
- 两个注入项分别命中 `E_AUDIO_FORMAT` 与 `E_INJECTED_FAULT`，故障轮次未进入轨迹；
- 最终 9 轮、9 条轨迹、0 个非注入失败（注入专项运行为独立快照与输出）。

这些数字以 `docs/verification.json` 为准；重新运行会刷新哈希。
