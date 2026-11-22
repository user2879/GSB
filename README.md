# 说话人日志分窗聚类组件（speaker diarization）

一个可集成的、完全确定性的说话人日志组件：长音频按固定窗长/窗移切分为重叠
分析窗，逐窗抽取声纹嵌入并定点化，在窗间以相似度阈值做增量聚类，经合并门
迟滞生成说话人分段；同时在配置定点位宽与高一档位宽两条路径下完整重算，
输出逐级误差归因、聚类边界翻转诊断与说话人数目差异。

仅依赖 Python 3.10+ 标准库（`wave`/`json`/`logging`/`threading`/`signal`）
与 NumPy（仅用于特征向量化计算）。无 GPU、无外部服务、无容器。

## 快速开始

```bash
python3 generate_samples.py     # 固定种子合成样本到 samples/
python3 main.py                 # 并发模式，报告写入 output/report.json
python3 main.py --serial        # 串行模式（结果与并发逐字段一致）
python3 verify.py               # 验收：21 项检查
python3 tests/test_units.py     # 单元测试
```

进程退出码为整数：`0` 成功（含收到 SIGTERM/SIGINT 后的优雅退出），
`2` 配置错误（stderr 输出 `E_CONFIG`）。业务错误码写入报告 `failures`
清单与 stderr 日志。

## 目录结构

- `generate_samples.py` — 固定种子（20261123）合成 16kHz 单声道 16bit PCM
  WAV：2~3 名说话人交替、部分重叠、段间静音与平稳底噪；说话人靠基频均值
  （A=120Hz / B=210Hz / C=90Hz）与共振峰位置区分；另生成一个违规文件
  `bad_format.wav`（立体声 8kHz）用于触发 `E_AUDIO_FORMAT`。
- `config.json` — 日志配置：窗长/窗移、帧长/帧移、嵌入维度、相似度阈值、
  合并门迟滞、最短分段/最短静音、噪声底估计窗帧数、定点小数位宽与量化
  档位、容差、线程池并发数、快照提交间隔窗数、末窗策略与边界裁决规则。
- `injections.json` — 观测注入清单：对指定音频的指定帧区间注入非有限采样、
  连续帧削波、全静音区间与数据块截断。
- `diarization/` — 组件包：
  - `audio_io.py` — RIFF 容器校验（采样率/声道/位深/数据块长度）与重叠分窗；
  - `embedding.py` — 帧级能量/基频（自相关）/MFCC 倒谱特征，固定投影映射，
    固定累加顺序聚合，float64 全程；定点量化（步长 2^-frac_bits，逐维残差）；
  - `clustering.py` — 维度升序累加的余弦相似度、字典序首命中增量聚类、
    迟滞带未归属、合并门；
  - `segmentation.py` — 窗标签到帧级分段的还原与后处理；
  - `tracking.py` — 双位宽四环节误差传播追踪与翻转归因；
  - `injection.py` — 观测注入（真实改变输入采样流）；
  - `snapshot.py` — 阶段快照的临时目录构建与整体原子替换；
  - `pipeline.py` — 阶段编排、线程池、信号处理、报告输出。
- `main.py` — CLI 入口；`verify.py` — 验收脚本；`tests/test_units.py` — 单元测试。

## 关键设计

### 分窗（阶段 1）
窗序号从零严格递增，记录每窗起止采样偏移；末窗不足按
`last_window_policy` 补齐（pad）或裁剪（crop）并标记 `is_last`。
容器不合规输出 `E_AUDIO_FORMAT` 进入失败清单；注入的非有限采样在分窗
阶段识别，输出 `E_NONFINITE_INPUT`（报告含首个非有限采样所在窗序号）。

### 嵌入与定点化（阶段 2）
每帧计算能量、基频估计（归一化自相关，60–400Hz）与 MFCC 倒谱系数
（26 通道梅尔滤波器组 + DCT，去 c0）。噪声底取全音频最低
`noise_floor_frames` 个正能量帧的均值（零能量静音帧不参与估计——数字
静音不是噪声底）。窗特征为有声帧（能量超过 4 倍噪声底）均值，经固定种子
投影矩阵映射到嵌入维度，聚合与归一化均按固定累加顺序，同一窗多次运行
逐位一致。量化记录每窗统一步长与逐维残差、逐维定点整数。

### 相似度与增量聚类（阶段 3）
相似度为余弦，点积与模长按维度序号升序累加。新窗按簇标识字典序探测既有
簇，取首个满足阈值的簇；相似度恰等于阈值按同说话人决断。最佳相似度落入
迟滞带 `[threshold - hysteresis, threshold)` 的迟到窗保持未归属（不被静默
吸收），完整输出于 `unassigned_windows_*` 与 `hysteresis_events_*`。
合并门对簇代表嵌入做全对比较：≥阈值合并（并查集，记录事件），落入迟滞带
记录为合并门边界事件。簇代表为成员窗按窗序号升序的均值。

### 分段生成（阶段 4）
帧标签取覆盖该帧的窗标签多数票（平票取字典序最小簇）；短于最短分段的
语音段丢弃；间隔短于最短静音的同说话人段合并；不同说话人之间的短静音
间隙按 `boundary_rule`（midpoint/left/right）裁决吸收。输出说话人标识、
起止毫秒、段内平均能量，按起始毫秒升序、同起始按说话人标识字典序。
无语音时输出空列表。

### 逐级误差传播追踪
每个音频在 `frac_bits`（6）与 `frac_bits_high`（10）两条路径下完整重算
嵌入量化 → 相似度比较 → 聚类归属 → 分段合并四个环节，输出逐窗差值、
每环节最大差值与首个超容差（`tolerance`）环节标注。说话人数目或分段
边界不一致时按链路顺序归因：`embedding_quantization_residual` /
`similarity_rounding` / `threshold_neighborhood_decision` /
`merge_gate_hysteresis_boundary`，并记录两档位的簇数量与边界帧。

### 并发确定性
窗嵌入与窗间相似度比较用 `ThreadPoolExecutor` 并发（最大并发数取自配置，
不得超过），经有序 `map` 收集；音频顺序、逐窗记录、未归属清单、分段、
误差归因与失败清单的顺序与单线程按文件名+窗序号执行完全一致
（`verify.py` 中 `serial_parallel_identical` 逐字段对比通过）。

### 快照与恢复
分窗/嵌入/聚类/分段合并四阶段产物先写入 `snapshots.new/` 临时目录，再
整体替换提交（旧目录改名备份→新目录改名上线→删除备份，绝不逐文件覆盖）。
清单记录全局版本号、上一版本号（替换前识别版本变化）、配置哈希、注入
清单哈希、已完成音频游标、定点位宽与各阶段产物版本。嵌入阶段每
`snapshot_commit_interval_windows` 窗提交一次中间快照。收到
SIGTERM/SIGINT 时完成当前音频产物与快照原子替换后以退出码 0 退出；
强制终止后仅依据已完整替换的快照目录做恢复判定（启动时读取
`snapshots/manifest.json`，记入报告 `recovered_snapshot`）。

## 业务错误码

- `E_CONFIG` — 配置/清单文件不可读或非法（stderr，退出码 2）
- `E_AUDIO_FORMAT` — RIFF 容器、采样率、声道、位深或数据块长度不合规
- `E_NONFINITE_INPUT` — 分窗阶段检测到非有限采样值
- `E_EMBEDDING` / `E_CLUSTERING` / `E_SEGMENTATION` — 对应阶段异常
- `E_INTERNAL` — 未分类异常

单音频失败记录文件名、阶段名、窗序号与业务码后继续处理其余音频，
该音频不再产出分段。

## 验证结果（本机实测）

`python3 verify.py` 全部 21 项 PASS：

- 说话人数目：`two_speakers`=2、`three_speakers`=3、`quiet_single`=1、
  `overlap_pair`=2，均与合成真值一致；
- 注入链路：`inj-001` 非有限采样 → `E_NONFINITE_INPUT`（分窗阶段，窗 6）；
  `bad_format.wav` → `E_AUDIO_FORMAT`；`inj-002/003/004`（削波/静音/截断）
  命中编号完整记录，音频继续处理（截断后窗数减少、静音区间被 VAD 排除、
  削波区间形成一个额外的伪簇——扰动真实改变了输入数据）；
- 双位宽归因：所有音频首超容差环节为 `embedding_quantization`；
  `overlap_pair.wav` 在 6/10 位宽下窗 11 归属翻转（高挡位落入迟滞带未
  归属），分段边界帧 238→239，归因 `threshold_neighborhood_decision`；
- 串行与并发报告逐字段一致；重复运行逐字段一致；
- SIGTERM 注入 1 秒后退出码 0，报告与快照完整落盘；
- 快照清单版本、配置哈希、注入哈希、游标与阶段版本齐全，二次运行识别
  到已提交快照（`recovered_snapshot`）。
