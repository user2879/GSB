# voice_diagnostic

可独立复核的本地语音交互诊断组件。它不访问网络、不依赖真实录音或容器，使用固定种子生成中文多轮命令对话，并在以下条件下给出一致、可审计的帧级端点、交叠归属、命令结论和双版本回执：

- 不同说话人的区间交叠；
- 16 kHz / 单声道 / 16 bit PCM 音频片段按到达顺序滚动可用；
- 窗跨越说话人交叠区或多个音频片段；
- 单批预算、单区间配额和有限并发；
- 进程在任意阶段提交后被强制终止。

## 快速开始

```bash
cd /home/zsl/feedback/uidfc5909d3-default-402
source .venv/bin/activate

# 生成固定种子样本
voice-diag generate --out demo/sample

# 运行并自动按预算分批
voice-diag run \
  --config demo/sample/config.json \
  --manifest demo/sample/manifest.json \
  --out verification/run

# 一条命令生成并运行内置样本
voice-diag demo --out verification/demo

# 打开只读人工复核视图（需要解释器可导入 Tkinter）
voice-diag view --out verification/run

# 运行全部自动化验收
pytest -q

# 运行端到端验收矩阵并输出机器可读结果
python scripts/verify_acceptance.py --out-json verification/acceptance.json
```

若虚拟环境没有继承系统 Tkinter，可用 `PYTHONPATH=src python3 -m voice_diag view --out verification/run`；图形视图只读取已提交产物，不影响命令行、测试和恢复路径。

`run` 在预算耗尽时返回 `E_POINT_BUDGET_PENDING`，用同一命令再次运行即可从持久化待办队列继续。`--arrival-limit N` 可模拟只到达前 N 个音频片段；所有片段到齐前状态为 `waiting_arrival`，不会伪装成最终完成。

## 目录

- `src/voice_diag/config.py`：配置枚举与参数校验。
- `src/voice_diag/schema.py`：manifest、说话人、命令表、逐字偏移、命令流、回执引用校验。
- `src/voice_diag/audio.py`：严格 RIFF/WAVE/PCM16 校验与音频混合。
- `src/voice_diag/receipts.py`：a1 定宽文本、a2 JSON、统一内部表示与旁路元数据。
- `src/voice_diag/endpoint.py`：帧能量/谱平坦度、双门限迟滞、最短语音/静音后处理、重叠分析窗。
- `src/voice_diag/features.py`：梅尔能量、过零率、谱质心、定点整数、固定投影与命令后验。
- `src/voice_diag/adjudicate.py`：帧级还原、交叠加权、并列裁决、命令段合并与仲裁证据。
- `src/voice_diag/cache.py`：窗级/区间级内容指纹、缓存键和失效依据。
- `src/voice_diag/scheduler.py`：预算、配额、稳定排序键与线程池。
- `src/voice_diag/snapshot.py`：版本化目录快照、临时目录写入和原子 `CURRENT.json` 切换。
- `src/voice_diag/pipeline.py`：阶段化流水线、增量复用、恢复、回执和诊断报告。
- `src/voice_diag/synthesize.py`：固定种子中文多轮样本生成。
- `src/voice_diag/viewer.py`：读取已提交产物的只读 Tkinter 复核视图。
- `tests/`：43 项单元与端到端测试。
- `docs/DESIGN.md`：算法、格式和不变量说明。
- `verification/acceptance.json`：本机实际验收结果。

## 输入布局

`config.json` 声明采样格式、端点参数、分析窗、量化档位枚举、定点小数位宽、裁决规则、并发/预算/配额和 a1 字段宽度。`manifest.json` 声明：

- `speakers`、受控 `commands` 与命令字符表；
- 每个 `fragments` 的 `audio_file`、`arrival_index`、采集时刻与到达时刻；
- 每个 `intervals` 的全局采样范围、说话人、命令真值和逐字转写；
- `command_stream` 对区间与字项索引的引用；
- `receipts.a1` 与 `receipts.a2` 两个逐区间证据文件。

字级偏移按区间内相对采样偏移表示；必须满足逐字单调，并落在区间采样范围内。音频片段的 PCM 长度必须等于其区间长度。格式不合规的片段记录 `E_AUDIO_FORMAT` 后被跳过；结构或引用不合规则在任何快照写入前以 `E_SCHEMA` 终止。

## 双版本回执

a1 是固定字段宽度的文本行，默认布局：

```text
I001        |S01     |000969|00001
```

字段依次为区间序号、说话人序号、千分比整数端点置信度和整数命令码。每行总宽度固定；字段溢出、非法字符或行宽不足抛出 `E_FIELD_TRUNCATED`，不会静默截断。

a2 是命名 JSON：

```json
{
  "version": "a2",
  "records": [
    {
      "interval_id": "I001",
      "speaker_id": "S01",
      "endpoint_confidence": 0.968912,
      "command": "开灯",
      "quant_scheme": "permille"
    }
  ]
}
```

两个版本归一为 `ReceiptRecord`，保留来源版本、缺失字段和物理存储字段。a2 降级到 a1 时，旁路元数据写入：

- 原浮点置信度与被覆盖字段；
- `float_to_permille_round_half_away` 精度标签；
- 截断残差；
- 命令名与量化方案；
- 原始 a1/a2 渲染值，用于校验 sidecar 是否仍对应当前文本。

携带 sidecar 将 a1 升回 a2 时，可精确恢复原先被覆盖的浮点置信度、命令名和量化方案。测试覆盖了这条往返路径。

## 处理与裁决

1. 全局混合时间线按 `arrival_index` 稳定叠加。每个分析窗记录采样覆盖、帧索引、引用片段、所跨区间、说话人、重叠帧数和来源可信度。
2. 每个窗分别抽取全局及逐区间定点特征。固定点整数按“帧序号升序、特征维升序”累加；命令投影按命令码升序归约。
3. 来源可信度由说话人声明可信度、窗口内字项覆盖率和区间覆盖比例构成。
4. 帧级交叠权重严格为：`后验余量 × 来源可信度 × 重叠帧数`。
5. 归属余量落在量化半步内时，按区间起点升序、再按说话人标识字典序裁决，并标记边界帧。命令分歧始终写入 `contested`、次优区间/命令和仲裁原因，不会被静默丢弃。
6. 命令结论按最短命令段合并；过短段进入 `suppressed` 列表并保留原因。

## 缓存与恢复

缓存键分别覆盖阶段：

- `features`：全局版本、配置哈希、窗口几何、引用片段内容哈希、字项偏移指纹、参数和量化档位；
- `posterior`：上游特征键、投影指纹、命令码、量化方案；
- `attribution`：上游后验键、区间重叠、候选帧掩码和裁决参数；
- `interval`：引用该区间的窗后验键和回执参数；
- `adjudication`：全部窗归属键、裁决/回执参数和量化方案。

替换一个区间音频或修正其字级偏移时，只有引用它的窗及其区间、最终结论失效；其他窗直接从持久化账本复用。每条命中或失效都带 cache id、阶段、键、引用集合、稳定键和失效原因。

每个阶段产物、待办队列、已完成片段游标和累计预算都写入新的版本化目录。提交顺序为：

1. 临时目录内完整写入并 `fsync`；
2. 目录整体 `rename` 到不可变快照名；
3. 原子替换 `CURRENT.json` 指针。

因此强杀不会留下半写快照。恢复时从 `CURRENT.json` 读取账本，重算不足阶段；最终摘要与一次全量运行逐字段一致。

## 输出

完成运行后输出：

- `diagnostic-report.json` / `diagnostic-report.md`；
- `final.json`；
- `receipts/evidence.a1`、`receipts/evidence.a2.json` 及 sidecar；
- `CURRENT.json` 和不可变 `snapshots/.../state.json`。

报告包含双版本回执计数、交叠帧数、边界帧清单、多来源窗数量、区间真值匹配数、失效/复用条目、业务码计数、待办数量和唯一复核摘要 `final_digest`。

## 业务码

| 代码 | 含义 |
| --- | --- |
| `E_AUDIO_FORMAT` | WAV 容器、格式、位深、采样率、声道、数据长度或采样数不合规；跳过该片段 |
| `E_SCHEMA` | 配置、清单、偏移、引用或回执不一致；不产生部分产物 |
| `E_FIELD_TRUNCATED` | a1 字段不足、行宽不足或出现非法字符 |
| `E_QUOTA_EXCEEDED` | 单区间配额超限；整批中止且不提交窗产物 |
| `E_POINT_BUDGET_PENDING` | 当前批预算不足；任务保留到待办队列 |

## 实际验证

本机 `Python 3.11.0rc1`。环境的 `ensurepip` 不可用，因此用等价的 `virtualenv -p python3 .venv` 建立隔离环境，并安装：

- `numpy 2.4.6`
- `scipy 1.17.1`
- `pytest 9.2.0`

实际执行：

```text
$ .venv/bin/python -m pytest -q
...........................................                      [100%]
43 passed
```

端到端验收矩阵结果见 `verification/acceptance.json`，关键值：

- 完整运行：`final_digest=87a8b5d6bd6d62467dfe5d3531374f8b7171eba3fd383d3f8fc42d35286bc8df`；
- 4/4 双版本输入回执归一，4/4 输出 a1/a2 回执生成并通过旁路恢复；
- 84 个交叠帧、17 个多来源窗、0 个并列边界帧；
- 4/4 区间命令与真值匹配；
- 静默重跑复用 123 个缓存条目，失效 0 条；
- 替换 `F002` 音频后，增量与独立全量摘要同为 `fbb0188a9d6bd2320f30fd56b784444235f17983fbfb7ab12f2e6b37399e40f4`，复用 76 条、精确失效 51 条；
- 子进程在首批提交后被 `_exit(137)` 终止，恢复后摘要与全量基线相同；
- 切到 Q8 后中间特征和帧摘要真实变化；
- 滚动到达 `arrival_limit=2` 时状态为 `waiting_arrival`，到齐后恢复为完整基线摘要。

本实现生成的是诊断用受控声学信号，不把合成音色当作语音识别准确率基准；验收目标是确定性、可复核性、精确失效和恢复一致性。
