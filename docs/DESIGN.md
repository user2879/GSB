# 流式 ASR 数据编辑与操作应用 — 技术说明

## 1. 目标与核心矛盾

应用消费确定性合成的 16kHz/单声道/16bit PCM 流，按分片做帧级声学特征、字级维特比
解码与重叠区时间对齐，并在交互编辑（重叠时长、稳定前缀帧数、LM 回退权重、分片音频
替换）下于重解码预算内返回可用转写。核心约束是：**交互延迟与最终一致性不得互相妥协**。

解决思路是把整条流水线实现为「纯函数 + 显式缓存失效 + 确定性预算记账」：

- 所有数值计算 float64、无 RNG、固定滤波器几何，同输入必产生同输出；
- 编辑只使受影响分片的缓存失效，未触及区间直接复用；
- 预算判定在任何线程启动前基于稳定排序键完成，并发度只影响墙钟时间，不影响结果与
  预算消耗汇总；
- 增量引擎最终态与独立的一次性全量参考解码器（`full_decode.py`，不复用引擎）逐字段
  比较，作为一致性验收口径。

## 2. 模块结构

| 模块 | 职责 |
| --- | --- |
| `errors.py` | 业务码（`E_AUDIO_FORMAT`/`E_CONFIG_INVALID`/`E_FEATURE_FAILED`/`E_DECODE_FAILED`/`E_REDECODE_BUDGET_PENDING` 等）与失败记录 |
| `config.py` | 三类资源加载、严格校验、有效位舍入、规范化配置哈希 |
| `audio.py` | RIFF/WAVE 严格校验、PCM 归一化、帧规划、滑窗分片规划、归属区替换 |
| `features.py` | 切帧补零、确定性三角 mel 滤波器组、对数能量、最近质心音素分类、发射打分查表与平滑补齐 |
| `decoder.py` | 二元 LM + 回退的转移矩阵、对数域维特比、连续同字合并与每字分数 |
| `streaming.py` | 单分片解码、全局帧时间戳对齐、最早分片优先的合并与冲突记录、分数重算 |
| `stability.py` | 以最小连续一致帧数为判据的稳定前缀/可变尾部切分 |
| `snapshot.py` | 临时目录构建 + 目录级原子替换、版本化清单、恢复点查询 |
| `engine.py` | 编辑处理、缓存失效、预算内任务前缀、线程池并发、快照提交、信号/崩溃注入 |
| `full_decode.py` | 独立一次性全量参考解码器（一致性 oracle） |
| `runner.py` | 目录扫描、命令流编排、信号门、JSON 诊断报告 |
| `cli.py` / `__main__.py` | `run` 与 `full` 两个命令行入口 |
| `samplegen/generate_samples.py` | 固定种子合成音频与全部配置（唯一允许使用 RNG 的组件） |

## 3. 关键设计与不变量

### 3.1 确定性
- mel 滤波器组由频带数与 FFT 大小几何确定，窗口、相位（生成器侧）均固定；
- 发射分缺失候选统一填 `log_smooth`；
- 浮点仅在写入产物时按精度档位（low=4 / medium=6 / high=10 位有效数字）舍入，
  内部全程 float64；
- 跨进程重复运行对全量结果做 SHA-256 比对，哈希完全一致。

### 3.2 分片、归属区与替换影响面
- 分片为滑窗：`hop = chunk_ms - overlap_ms`，末片截断到流末尾；
- 每个分片有排他「归属区」`[own_start, own_end)`，首片从 0 开始、末片到流末尾，
  归属区互不重叠且覆盖整条流；
- 音频替换只改写目标分片归属区，且替换样本长度必须等于归属区，保证帧/时间几何不变，
  否则抛 `E_AUDIO_FORMAT`；受影响分片 = 与该归属区时间相交的分片集合，其余分片复用。

### 3.3 重叠对齐与冲突
- 每个分片独立解码（LM 状态在片首重置），帧标签映射到全局帧号；
- 合并时同一全局帧只保留一个字：最早分片为属主，后续分片标签不同则记录
  `alignment_conflicts`（含帧号、毫秒、属主/冲突分片、保留字、丢弃字），绝不静默覆盖。

### 3.4 稳定前缀与可变尾部
- 逐分片边界比较两侧重叠区标签，取最长连续一致 run；run 长度达到
  `min_consistent_frames` 才判该边界稳定；
- 第一个未通过边界的帧号即稳定前缀截止位置；全部通过则整段稳定。

### 3.5 编辑与缓存失效
| 编辑 | 失效范围 | 复用范围 |
| --- | --- | --- |
| `set_overlap` | 分片几何变化，特征/解码缓存全失效 | 原始 PCM 样本 |
| `set_stable_frames` | 无声学/解码失效，仅重算前缀切分 | 全部特征与解码 |
| `set_backoff` | 全部分片解码（转移矩阵变化） | 全部帧级特征 |
| `replace_chunk` | 与归属区相交分片的特征+解码 | 其余分片特征/解码 |

每条响应都给出 `reused_chunks`/`redecoded_chunks` 及对应的样本区间清单。

### 3.6 预算语义
- 每个任务有确定性模拟成本 `simulated_task_cost_ms`；
- 预算判定先按 `stable_sort_key = chunk_index*10 + stage_rank` 排序，取累计成本不超过
  预算的任务前缀，随后才交给线程池；因此并发度与调度无法改变「哪些任务运行」；
- 预算耗尽返回 `E_REDECODE_BUDGET_PENDING` 的 pending 中间态，未完成任务进入
  `_pending`，下一条命令（或 `resume`）继续推进，必然收敛到最终态；
- 响应含稳定前缀文本、可变尾部帧数、受影响分片、进度百分比、预算消耗。

### 3.7 并发一致性
- 线程池 `max_workers = min(concurrency, concurrency_limit)`，不允许超过上限；
- 失败记录、结果汇总均在汇合后按分片号/阶段稳定排序，单线程与多线程的转写、时间戳、
  冲突、失败顺序、预算汇总逐字段一致（由 `test_concurrency.py` 守护）。

### 3.8 快照、恢复与退出
- 四个阶段（features/decode/merge/prefix）产物先写入 `snapshots/tmp` 下全新版本目录，
  再用 `os.rename` 整体替换 `active`；版本变化永远是目录替换而非逐文件覆盖；
- 清单 `manifest.json` 记录全局版本号、配置哈希、原始音频指纹、当前样本指纹与各阶段
  产物版本及文件 SHA-256；
- 按 `snapshot_commit_interval_chunks` 间隔增量提交，强制终止（`os._exit(9)`）后重启
  只可能读到上一个完整替换的快照，并报告恢复点分片序号；
- `SIGTERM/SIGINT` 经信号门在当前分片产物提交后抛 `SystemExit(0)`，退出码为 0；
- 单分片特征/解码异常被捕获为失败记录（含分片号、阶段、业务码、异常类型），其余分片
  继续，失败分片不阻塞整体收敛。

## 4. 需求到验证的映射

| 业务要求 | 验证位置 |
| --- | --- |
| 1 资源加载与校验 | `tests/test_validation.py`（坏 RIFF、采样率、概率越界、overlap 规则、精度枚举、并发上限） |
| 2 特征/发射确定性、补零标记、平滑补齐 | `tests/test_features.py` |
| 3 维特比 + 重叠对齐 + 冲突记录 | `tests/test_alignment.py` |
| 4 稳定前缀 + 尾部重解码 + 复用区间 | `tests/test_equality.py` |
| 5 预算内响应与 pending 收敛 | `tests/test_budget.py` |
| 6 增量 vs 全量逐字逐字段一致 | `tests/test_equality.py`、`verify_equality.py` |
| 7 并发串行一致性 | `tests/test_concurrency.py` |
| 8 快照/恢复/信号/失败继续/诊断报告 | `tests/test_snapshot.py` |

## 5. 本地运行

```bash
pip install -r requirements.txt
python -m samplegen.generate_samples --output samples   # 已随仓库提供生成结果
python -m pytest tests/ -q
python verify_equality.py samples artifacts/verify
python -m asr_stream_editor run  --input samples --output artifacts/demo \
    --commands samples/command_stream.json
python -m asr_stream_editor full --input samples --output artifacts/ref
```

诊断报告 `run_report.json` 含：整段转写、各分片稳定前缀、预算内/超预算响应次数、
对齐冲突清单、失败清单、各阶段耗时（调用次数/总耗时/均值）、全局版本与恢复点。

## 6. 已知边界

- 声学模型是面向确定性数据的最近质心分类器 + 查表发射分，不是真实神经网络 ASR；
  其价值在于可复现地驱动整套流式/增量/一致性机制。
- 预算以确定性模拟任务成本计量（保证并发与串行预算汇总可逐字段比较），真实墙钟耗时
  另行记录在 `stage_timings_ms`，不参与一致性比较。
