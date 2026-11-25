# 流式语音命令识别链路 — 技术说明

## 1. 工程结构

```
voicecmd/
  errors.py        业务错误码（E_AUDIO_FORMAT/E_FRAGMENT_INVALID/...）
  fingerprint.py   blake2b 内容指纹、分层缓存键、窗稳定排序键
  config.py        pydantic 配置模型（含量化档位枚举、词表、配置/词表指纹）
  audio.py         标准库 wave + 手工 RIFF 双通道校验、清单/命令流加载
  samplegen.py     确定性样本生成（固定种子，语音区刻意跨片段边界）
  vad.py           帧能量/谱平坦度、双门限迟滞、最短语音/静音后处理
  windowing.py     全局坐标重叠分析窗、跨片段引用集合
  features.py      梅尔能量/过零率/谱质心、定点量化、固定顺序累加
  classifier.py    确定性整数投影、稳定 softmax、平局按命令编码决断
  commands.py      命令结论汇总与引用窗集合
  injection.py     5 类异常注入引擎与命中证据/诊断报告
  snapshot.py      临时目录 + 原子整体替换快照与恢复判定
  pipeline.py      调度器：精确失效、预算待办、线程池并发、账本
  cli.py           generate / run / resume-check
configs/           pipeline.json、injection.json、injection_full.json
tests/             27 个 pytest 用例
scripts/           generate_samples.py、run_batches.py、run_pipeline.sh
```

## 2. 关键设计

### 2.1 加载校验（要求 1）
- `audio.parse_riff` 手工校验 RIFF/WAVE 头、chunk 尺寸与截断；`wave` 模块二次校验
  采样率 16000、单声道、16bit、data 块长度与帧数一致。
- 清单校验片段序号唯一、`arrival_ms/capture_ms` 为非负整数且允许不同、
  命令词落在受控词表；失败分别归类 `E_AUDIO_FORMAT` / `E_FRAGMENT_INVALID`，
  进入失败清单，不截断、不静默跳过。
- 样本生成器将到达顺序用固定种子打乱，制造乱序到达与更早采集时刻。

### 2.2 VAD 与分析窗（要求 2）
- 帧长 400 / 帧移 160，计算帧能量与谱平坦度；高门限进入语音、
  低于低门限退出（迟滞），再按最短语音帧/最短静音帧合并与过滤。
- 分析窗长 1600 / 窗移 400，全部使用全局采样坐标；每个窗显式记录
  `[start,end)` 与引用片段序号集合，跨边界窗包含全部相关片段。
- 纯函数实现，同音频同配置重复运行逐字段一致（测试 `test_window_set_byte_identical_across_runs`）。

### 2.3 定点特征与后验（要求 3）
- 每帧特征：6 维梅尔对数能量 + 过零率 + 谱质心（共 8 维）。
- 量化 LSB 随档位变化：`fractional_bits = max(level-8, 1)`，
  再按各维满量程缩放并按有符号位宽裁剪；切换 8/16/24/32 位宽会真实改变
  中间整数、投影得分与缓存键（测试 `test_quant_level_changes_integers_and_scores`）。
- 累加严格按帧序号升序、特征维序号升序的嵌套循环写入。
- 投影矩阵由 `weight_seed` 确定性生成，整数点积给出精确得分，
  稳定 softmax 产生后验；得分/后验相等时按命令编码升序决断，
  平局或领先优势小于 `boundary_margin` 标记边界窗。

### 2.4 依赖指纹与精确失效（要求 4）
- 四层缓存键：片段特征键 -> 窗特征键 -> 窗后验键 -> 命令结论键，
  均由上游内容指纹 + 参数指纹 + 位宽/档位派生（`fingerprint.cache_key`）。
- 替换片段时仅失效引用该片段的窗；量化档位变更失效全部窗。
- 未受影响窗直接复用，测试断言其缓存键与产物逐字段不变
  （`test_fragment_replacement_precise_invalidation`）。

### 2.5 预算与待办（要求 5）
- 每批维护剩余窗点数，按稳定排序键（片段序号集合 + 窗序号）分配；
  预算不足的窗进入待办并标记 `E_WINDOW_BUDGET_PENDING`，下一批优先消费待办。
- 失败窗隔离进 `failed_windows`，不会被误记为完成，也不会饿死待办队列。

### 2.6 并发与一致性（要求 6）
- `ThreadPoolExecutor` 并发执行窗特征+后验，`max_workers` 取自配置且有测试
  断言实际线程池参数永不超限。
- 所有结果在收集后按窗序号排序落库；并发度 1/2/4/8 的最终窗特征、后验、
  命令结论与串行执行逐字段一致（`test_concurrency_matches_serial_signature`）。

### 2.7 阶段化快照（要求 7）
- 阶段：VAD、窗特征、后验、命令汇总；每批将状态写入同级临时目录后
  `os.replace` 整体替换，清单含全局版本号、配置哈希、词表指纹、各阶段版本。
- 启动比对配置/词表指纹决定 `resume` 或 `rebuild`；恢复仅依赖已持久化快照，
  并按快照片段清单重新装载 WAV 重建内存音频。强制终止不承诺退出报告。

### 2.8 异常注入隔离（要求 8）
- 独立注入配置支持：截断 WAV、非有限采样值、采样区间越界、
  非法量化档位、后验维度不符。
- 单片段/单窗失败记录片段序号、窗序号、阶段名与业务码，其余单元继续处理；
  报告含注入项编号、命中证据与失败单元清单（`artifacts/injection_report.json`）。

## 3. 本地运行

```bash
python3.11 scripts/generate_samples.py
python3.11 -m voicecmd.cli run --report diagnostic_report.json
python3.11 -m voicecmd.cli run --injection configs/injection_full.json --report injection_report.json
python3.11 -m voicecmd.cli resume-check
python3.11 -m pytest tests/ -q
```

全程仅使用本地文件与标准库 `wave`，无网络访问、无容器、无音频编解码第三方库。

## 4. 验证结果

- `pytest tests/ -q`：27 passed。
- 确定性样本：8 片段 × 8000 采样，67 个分析窗，其中 28 个跨片段边界窗。
- 增量：替换片段 2 仅重算 13 个引用窗，其余 54 个窗直接复用且逐字段不变。
- 预算：budget=5 时首批 5 个重算、57 个带 `E_WINDOW_BUDGET_PENDING` 待办，
  后续批次优先消化待办。
- 快照：两批后强制中断，恢复判定 `resume`，续跑结果与空白全量重建一致；
  改档位后判定 `rebuild`。
- 注入：5 类注入全部命中，业务码分别为 E_AUDIO_FORMAT、E_SAMPLE_NON_FINITE、
  E_SPAN_OUT_OF_RANGE、E_QUANT_LEVEL_INVALID、E_DIMENSION_MISMATCH，未注入单元不受影响。
