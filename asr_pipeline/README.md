# ASR 流式并行解码调度与结果一致性验证管线

一个完全本地、确定性、无外部服务依赖的语音识别流式转写调度管线。输入为固定种子合成的
16kHz/单声道/16bit PCM WAV 与三份配置（打分参数、调度配置、音素映射），输出可逐字段比对的
转写结果与 JSON 诊断报告。核心目标：**在任意并发度、任意预算耗尽时机、任意批次切分下，
并行结果与单线程按分片序号串行执行逐字段一致**。

## 环境

- Python ≥ 3.10（开发验证使用 3.11），仅依赖 NumPy，WAV 使用标准库 `wave` 读取。
- 无网络、无 Docker、无外部账号。

```bash
pip install -r requirements.txt
```

## 目录结构

```
asr_pipeline/
├── asr_pipeline/
│   ├── errors.py        # 业务码 E_* 与 PipelineError、失败记录
│   ├── config.py        # 三份配置 + 注入清单的严格校验与规范化指纹
│   ├── fingerprints.py  # 内容指纹、分阶段缓存键、原子 JSON 写入
│   ├── storage.py       # 缓存/快照目录布局与整体替换提交
│   ├── audio.py         # RIFF/PCM 校验、帧移对齐的分片规划
│   ├── features.py      # 切帧、梅尔能量、对数能量、发射打分、定点量化、有效位
│   ├── decoder.py       # 对数域维特比、帧级预算 guard、检查点续跑
│   ├── stitching.py     # 重叠区时间对齐、后到分片优先、冲突记录
│   ├── pipeline.py      # 线程池调度、全局预算账本、缓存复用、快照、报告
│   └── cli.py           # 运行入口、退出码、自动续跑
├── sample_gen/generate_samples.py  # 固定种子样本与配置生成器
├── tools/verify_consistency.py     # 跨运行逐字段一致性校验
├── tests/               # unittest 套件（17 个用例）
├── run_demo.sh          # 一键端到端验收
└── requirements.txt / pyproject.toml
```

## 快速开始

```bash
# 1. 生成确定性样本（首次已随仓库提供，可重复生成，字节一致）
python3 sample_gen/generate_samples.py --out samples

# 2. 并行全量运行
python3 -m asr_pipeline.cli \
  --sample-dir samples/wav \
  --score-config samples/score_params.json \
  --schedule-config samples/schedule.json \
  --phoneme-map samples/phoneme_map.json \
  --workdir .asr_work --report report.json

# 3. 一键验收（并行/串行/紧预算续跑/注入/一致性）
./run_demo.sh
python3 -m unittest discover -s tests
```

### CLI 参数与退出码

| 参数 | 含义 |
| --- | --- |
| `--sample-dir` | WAV 所在目录（扫描全部 `*.wav`） |
| `--score-config` / `--schedule-config` / `--phoneme-map` | 三份配置路径 |
| `--injections` | 可选异常注入清单 |
| `--workdir` | 缓存与快照目录（默认 `<sample-dir>/.asr_work`） |
| `--report` | JSON 诊断报告输出路径 |
| `--serial` | 单线程按分片序号执行（一致性基准） |
| `--jitter-ms` | 由排序键派生的确定性线程抖动，用于证伪线程顺序依赖 |
| `--max-runs` | 存在待办分片时的自动续跑次数（每轮刷新全局预算） |

退出码：`0` 全部完成；`2` 全局预算耗尽仍有 `E_POINT_BUDGET_PENDING` 待办（可下轮续跑，
不计为失败）；`3` 存在永久性失败分片（含注入命中，如 `E_AUDIO_FORMAT`、
`E_CONFIG_INVALID`、注入型 `E_SHARD_BUDGET_EXHAUSTED`）；`1` 其它运行期错误。

普通的分片配额中止与全局预算耗尽都属于**可恢复资源状态**：分片保留完整维特比检查点进入
待办队列，下一批/下一运行在刷新预算后优先续跑，最终结果与一次性运行逐字段一致；只有真正
的处理异常或注入强制的不可恢复配额才进入失败清单。

## 关键设计与确定性不变量

### 1. 加载校验（业务码隔离）
- 音频校验 RIFF/WAVE 头、单声道、16bit、16000Hz、数据块长度，异常映射 `E_AUDIO_FORMAT`。
- 配置校验梅尔频带正整数、二元概率/回退权重有限、并发上限、预算/配额正整数、
  量化档位与小数位宽枚举，异常映射 `E_CONFIG_INVALID`。
- 失败条目进入失败清单，合法条目继续后续阶段，单条故障不阻断整体。

### 2. 特征与发射打分
- 帧长 25ms、帧移 10ms 切帧，末帧不足补零并以 `last_padded` 标记。
- 确定性三角梅尔滤波器组（向量化）、梅尔能量与对数能量，按主导频带查发射表，
  缺失候选用 `log_smooth` 补齐；全程 float64，先定点量化再按有效位保留。
- 同输入同配置多次运行，发射矩阵逐项 `array_equal`（测试覆盖）。

### 3. 维特比解码与重叠对齐
- 对数域维特比，转移概率取二元 LM，缺失走 unigram + 回退权重。
- 分片按帧移整数倍对齐，相邻分片在重叠帧区间以**后到分片优先（last-writer-wins）**
  保留唯一权威字，逐条记录被覆盖字、帧与来源分片。
- 检查点持久化**完整 trellis 与 backptr**，跨运行回溯与一次性运行字节一致。

### 4. 预算与并行调度（核心）
- 调度器在**规划阶段**按分片稳定排序键 `(audio, shard_index)` 顺序预扣全局预算，
  工作线程只消费已规划额度；结果回放仍按稳定顺序记账，与线程完成先后无关。
- 帧级硬 guard：`consumed + frame_cost > budget_cap` 即停止，全局剩余永不为负。
- 全局预算绑定 → `E_POINT_BUDGET_PENDING`（可续跑）；分片配额绑定且无法完成 →
  `E_SHARD_BUDGET_EXHAUSTED`；待办队列下一批优先，检查点支持真续跑。
- `allowance` 为单次调用增量预算，`budget_cap = 已消耗 + 增量`，首跑/恢复同语义。

### 5. 内容指纹与精确失效
缓存键分阶段派生，音频指纹使用完整 SHA-256（不截断）：
- `audio`：原始 WAV 字节指纹。
- `emission`：音频指纹 + 声学指纹（帧参数/发射表/平滑/有效位）+ 量化档位与位宽 + 帧区间。
- `decode`：emission 键 + 语言模型指纹（二元/unigram/回退）+ 量化参数。
- `stitch`：全部分片 decode 键 + 配置指纹 + 重叠时长。

失效范围（测试覆盖）：
- 替换某音频字节：仅该音频的 audio/emission/decode/stitch 重算，其余全复用。
- 改回退权重：emission 全复用，decode/stitch 重算。
- 切量化档位/改重叠时长：emission 级联失效重算，audio 复用。
- 启动恢复时按当前指纹重算缓存键，陈旧的 completed 条目被剔除强制重算；
  版本不匹配记 `E_SNAPSHOT_STALE`，配置变化保留内容寻址缓存做键级失效。

### 6. 快照整体替换
各阶段产物先写临时文件再 `os.replace` 原子提交；manifest 记录版本号、配置哈希、
音频指纹、已完成分片、已消耗/剩余预算、待办队列与检查点。

### 7. 异常注入隔离
四类真实故障（截断 WAV、非有限采样、发射表维度不符、配额低于单帧最小点数）
按分片施加，命中注入 ID 与业务码写入报告，失败分片不进入转写与拼接，其余分片照常完成。

## 诊断报告字段

`transcripts`（逐字起止帧、发射/LM/总分、来源分片、对齐冲突）、`budget_ledger`
（每分片分配/消耗/剩余/状态，稳定有序）、`pending_queue`、`failures`、
`reuse_recompute_log`（每阶段 computed/reused）、`shard_timings_ms`、
`snapshot_events`、`injections_hit` 及汇总计数。

## 验证结果（本地实测）

- `python3 -m unittest discover -s tests`：**17/17 通过**，无 ResourceWarning。
- `./run_demo.sh`：一致性工具输出 **ALL CONSISTENCY CHECKS PASSED**。
  - 并行（含确定性抖动）vs 串行：转写、逐字帧/分数、冲突、预算账本、待办、失败、
    复用清单逐字段一致。
  - 紧预算（2500 点）首轮：8 COMPLETED / 7 `E_POINT_BUDGET_PENDING` /
    0 永久失败，账本剩余恒 ≥ 0 且退出码为 2；多轮续跑后 25 分片全部完成（退出码 0），
    转写、逐字帧与分数与一次性串行完全相同。
  - 四类注入全部命中并映射预期业务码，失败音频被排除出最终转写。
