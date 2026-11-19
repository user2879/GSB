# 说话人识别与分离 · 确定性数据编辑与操作应用

面向本地确定性合成 16kHz 单声道 16bit PCM WAV 样本的说话人识别 / 分离数据编辑应用。
仅依赖 Python 3.10+ 标准库 `wave` 与 NumPy，不下载任何数据或模型权重，不使用容器。

## 1. 环境与安装

```bash
python3 --version            # 需要 3.10+
python3 -m pip install numpy # 仅有的第三方依赖
```

## 2. 生成确定性样本

```bash
python3 -m scripts.generate_samples --out samples/dataset
```

固定种子（`SEED=20260517`）生成 3 个 WAV、4 份 JSON，重复运行逐字节一致：

- `sample_001..003.wav`：3 个说话人（不同基频/共振峰），含交替片段、重叠片段、段间静音与平稳底噪。
- `config.json`：帧长/帧移、特征维度、能量/谱平坦度/重叠/聚类门限、最短语音与静音、线程数、快照间隔。
- `commands.json`：裁剪、替换、合并簇、拆分簇、改阈值，以及一条带 `INJ-004` 的越界替换命令。
- `injections.json`：4 项独立异常注入配置。

## 3. 模块结构

| 模块 | 职责 |
| --- | --- |
| `speaker_pipeline/audioio.py` | WAV 容器头/采样率/声道/位深校验，PCM 读写（`wave`） |
| `speaker_pipeline/features.py` | 切帧（末帧补零并标记）、短时能量、谱平坦度、固定维 float64 声纹特征 |
| `speaker_pipeline/segmentation.py` | 双门限迟滞 VAD、最短语音丢弃、最短静音合并、相邻区间重叠标注、排序 |
| `speaker_pipeline/clustering.py` | 确定性凝聚聚类、合并/拆分覆盖、最近簇归属与同距字典序决断 |
| `speaker_pipeline/concurrency.py` | 有界线程池，稳定排序键保证并发结果与串行逐字段一致 |
| `speaker_pipeline/store.py` | 内容寻址缓存、临时目录 + `os.replace` 原子快照、完整清单 |
| `speaker_pipeline/commands.py` | 编辑命令模型、音频裁剪/替换物料化、簇覆盖账本 |
| `speaker_pipeline/injections.py` | 损坏 WAV、非有限值、阈值为零、替换越界四类真实故障注入 |
| `speaker_pipeline/engine.py` | 增量失效编排、阶段缓存键、命令重放、快照提交与恢复 |
| `speaker_pipeline/report.py` | 规范结果投影与 JSON 诊断报告 |
| `speaker_pipeline/cli.py` | `full` / `incremental` / `rebuild-final` / `verify` 命令 |

## 4. 关键确定性规则

- **特征**：全程 `numpy.float64`，所有输出数值保留 6 位小数；同一音频多次运行帧记录与摘要逐字段一致。
- **切分**：进入门限用配置值，退出门限放宽 20% 形成迟滞；区间按 `(start_ms, end_frame)` 升序。
- **聚类**：质心欧氏距离凝聚聚类，距离同分按簇内最小 `(文件序号, 帧序号)` 决断；阈值为零抛 `E_INVALID_THRESHOLD`。
- **归属**：归属最近簇，距离相同按簇 ID 字典序；同时保留命令强制簇与最近簇两个字段。
- **簇标识**：合并/拆分写入覆盖账本，最终一律按成员最小全局帧序重排为 `SPK-000…`；增量与全量共用同一函数 `apply_overrides`，因此标识天然一致。
- **缓存键**：每阶段键 = `阶段名 + SHA256(上游键 + 参数指纹)`；区间/簇记录的 `_references` 与清单的 `stage_products` 可审计。
- **失效粒度**：替换/裁剪只令该文件特征及其下游重算，其他文件特征直接复用；改阈值只重算聚类与归属。

## 5. 增量与全量一致性

```bash
WS=.tmp/ws
python3 -m speaker_pipeline.cli incremental --dataset samples/dataset \
  --config samples/dataset/config.json --workspace $WS --commands .tmp/commands_clean.json
python3 -m speaker_pipeline.cli rebuild-final --dataset samples/dataset \
  --config samples/dataset/config.json --workspace $WS
python3 -m speaker_pipeline.cli verify \
  --full $WS/canonical_full.json --incremental $WS/canonical_incremental.json
# => {"parity": true}，退出码 0
```

`rebuild-final` 在**同一份最终音频**上从空白缓存全量重建，并复用增量快照中的阈值与覆盖账本，
随后对簇划分、区间归属、成员顺序与归属距离做逐字段比对。

## 6. 快照与恢复

- 产物先写 `tmp/` 临时文件，再 `os.replace` 原子落位；清单 `manifest.json` 始终最后替换，
  带 `complete=true`，因此进程被强制终止后只会留下陈旧临时目录，不会发布半截快照。
- 清单记录：全局版本号、配置哈希、命令游标、阈值、覆盖账本、已完成区间、全部阶段产物版本。
- 断点续跑：`incremental --resume` 仅依据已完整替换的清单恢复，并打印 `recovered_version`。
- 可处理退出信号（POSIX `SIGTERM/SIGINT`，Windows 下该信号测试自动跳过）：完成当前命令快照后以退出码 0 退出；
  强制终止路径不承诺退出报告（由 `crash_before_commit` 测试覆盖）。

## 7. 异常注入与业务码

| 注入 | 真实故障 | 业务码 | 阶段 |
| --- | --- | --- | --- |
| `INJ-001` | 破坏 WAV 头字节 | `E_AUDIO_FORMAT` | io |
| `INJ-002` | 采样值注入 NaN | `E_NON_FINITE` | feature |
| `INJ-003` | 聚类阈值置零 | `E_INVALID_THRESHOLD` | clustering |
| `INJ-004` | 替换区间超出音频范围 | `E_OUT_OF_RANGE` | command |

单文件/单区间故障被隔离并记录 `file / interval_index / stage / code / error_type / injection_id`，
失败区间不进入聚类与归属汇总，其余区间继续处理。诊断报告含各阶段耗时、快照版本、复用/重算清单与注入命中。

## 8. 并发一致性

线程池并发度取自 `worker_threads` 且不超过；每个任务带由文件名/帧序号/区间序号派生的稳定排序键，
`map_ordered` 在收集后按键重排。测试在随机 sleep 调度下验证并发与单线程结果逐字段一致，
并断言并发高水位不超过配置值。

## 9. 一键验证

```bash
./run_demo.sh            # 生成 + 增量 + 全量对照 + 注入隔离 + 全部单测
python3 -m unittest tests.test_pipeline -v
```

## 10. Windows 说明

代码只使用跨平台 API（`os.replace`、`tempfile`、`wave`、`concurrent.futures`），可在 Windows 原生运行；
POSIX 信号语义测试在 Windows 自动跳过，其余 16 项测试同样适用。
