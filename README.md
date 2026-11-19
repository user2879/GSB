# 确定性、规模自适应 VAD 示例组件

本项目在空白工作区中实现了一个可集成的语音端点检测（VAD）领域组件。它只依赖 Python 3.10+ 与 NumPy，使用固定种子在本地生成 16 kHz、单声道、16 bit PCM WAV，并完成分帧特征、四态双门限迟滞判决、端点合并、规模策略选择、阶段缓存、并发执行和可恢复快照。

## 快速开始（Windows / PowerShell）

```powershell
py -3.10 -m pip install numpy
python -m vad.cli generate-demo --data-dir demo/input --config demo/vad_config.json --manifest demo/sample_manifest.json --seed 20261120
python -m vad.cli run --input demo/input --config demo/vad_config.json --manifest demo/sample_manifest.json --state-dir .vad_state --output reports/demo_report.json
python -m unittest discover -s tests -v
```

Linux/macOS 将 `py -3.10` 换成 `python3.10` 即可。项目不使用 Docker，也不需要网络、账号或外部音频资源。`python -m vad ...` 与 `python -m vad.cli ...` 等价。

## 目录

- `vad/audio.py`：严格校验 RIFF/WAVE、PCM、16 kHz、单声道、16 bit、块对齐、字节率和 data 块长度。
- `vad/features.py`：块式分帧，末帧补零；计算自然对数 RMS、过零率和 Hann 窗幅度谱平坦度，float64 计算后按策略精度量化。
- `vad/state_machine.py`：噪声底估计与 `SILENCE / SUSPECT / SPEECH / TRAIL` 四态转移，逐帧记录状态和判据。
- `vad/endpoints.py`：短语音段丢弃、短静音间隔合并、端点半开区间输出及期望数量核对。
- `vad/policy.py`：依据样本数、总帧数、活跃帧密度和档位目标计算候选评分，选择精度、噪声窗与门限估计方式。
- `vad/cache.py`：内容寻址的四阶段缓存对象，临时文件写盘后 `os.replace` 整体提交。
- `vad/snapshot.py`：运行清单、样本快照、完整前缀恢复校验和输入内容变化截断。
- `vad/pipeline.py`：有界线程池、稳定排序、增量失效审计、报告与确定性摘要。
- `vad/demo.py`：固定种子谐波语音/底噪/单帧突变生成器和注入清单生成器。
- `demo/`：可直接运行的配置、清单与 6 个确定性 WAV 样本。
- `tests/test_vad.py`：标准库 `unittest` 端到端测试。

## 数据与配置约定

样本幅度使用原始 int16 单位，因此 RMS 对数为 `ln(max(sqrt(mean(x²)), energy_floor))`。频谱平坦度使用去除直流分量后的幅度谱：`exp(mean(log(mag + eps))) / (mean(mag) + eps)`。过零率只统计严格正负号之间的变化。特征先以 float64 计算，再按当前规模档的有效位数四舍五入。

帧起点为 `frame_index * hop_ms`，帧序号从 0 开始。只有最后一个起点之后不足 `frame_len` 时补零，并记录 `is_last_frame=true` 与实际 `valid_samples`。端点使用半开区间：`start_frame` 包含、`end_frame` 不包含；`start_ms/end_ms` 同样按半开时间轴输出。

状态门限规则：

- 低门限来自前 `noise_window_frames` 帧。`quantile` 模式取指定低分位；`fixed_ratio` 模式取窗口内能量较低一半的均值，再加 `low_margin`。
- 有效高门限为 `max(high_threshold, low_threshold + hysteresis)`。
- 进入语音要求严格高于高门限、严格低于过零率门限、严格低于平坦度门限，并连续满足 `confirm_frames` 帧；因此单个幅度突变最多进入 `SUSPECT`。
- 保持语音使用低门限形成迟滞；释放连续 `release_frames` 帧不满足后回到静音。
- 所有门限比较在相等时按静音优先，即 `>` 和 `<`，不使用 `>=` 或 `<=`。

## 规模策略

`scale_profiles` 中每个档位包含目标帧数、目标语音密度、目标样本数、特征有效位数、噪声窗倍数和门限估计方式。选择器对每个候选计算：

```text
0.60 * |log2((frames+1)/(target_frames+1))|
+ 0.25 * |log2((density+0.02)/(target_density+0.02))|
+ 0.15 * |log2((samples+1)/(target_samples+1))|
score = 1 / (1 + weighted_distance)
```

评分、未加权输入指标、有效精度、有效噪声窗和最终理由全部写入每样本的 `scale.decision`。切换档位会实际改变特征量化精度、噪声窗口和门限估计路径，而不是只修改记录。`scale_profile_override` 仅用于诊断或验收时强制切换档位，正常配置保持 `null`。

## 增量缓存与失效

阶段缓存键至少包含样本稳定文件名、音频内容 SHA-256、该阶段配置指纹、所选策略档和上游阶段键。策略阶段还包含规模指标；特征阶段包含有效精度；状态阶段包含有效噪声窗和门限估计方式；端点阶段包含期望段数区间。

因此：

- 替换一个 WAV：只重算该样本的策略、特征、状态和端点。
- 修改检测门限：特征缓存仍可复用；策略因密度指标变化而重算，状态和端点重算。
- 修改最短语音段/最短静音段：仅端点阶段失效。
- 修改并发数或快照间隔：四个计算阶段均可复用。
- 切换规模档：只有有效输入发生变化的阶段重算；例如已处于 `large` 档的样本可继续复用特征与状态。

`cache_audit.reuse`、`cache_audit.recompute` 按样本文件名和固定阶段顺序列出缓存键、动作与原因，并给出总数和分阶段计数。

## 快照与恢复

状态目录结构：

```text
.vad_state/
  cache/{policy,features,state,endpoint}/<cache-key>.json
  snapshot/manifest.json
  snapshot/samples/<index>-<hash>.json
```

每个缓存对象和样本快照都先写入同目录临时文件、`fsync`，再用 `os.replace` 整体替换。运行清单记录全局版本号、配置哈希、注入清单哈希、样本游标、产物版本和快照引用，并按 `snapshot_interval` 提交。

重启时只接受清单中游标之前的连续完整快照，同时校验当前文件名顺序、音频内容哈希、配置哈希、清单哈希、索引和产物版本。临时文件、损坏 JSON、晚于游标的样本文件都不会被信任。若前缀在中间失效，恢复游标退到最后一个有效前缀，后续样本重新处理。`SIGINT/SIGTERM/SIGBREAK` 只设置停止事件：不再提交新样本，等待在途样本完成快照后提交，CLI 以退出码 0 返回。

## 报告字段

`reports/demo_report.json` 包含：

- 顶层：状态、全局版本、配置/清单哈希、算法取舍说明、执行并发、恢复点、确定性摘要。
- `samples[]`：语音段、完整逐帧特征/状态/判据、状态计数、噪声底与有效门限、规模策略候选评分、缓存事件、噪声容差核对。
- `failures[]`：不合规音频按文件名稳定排序，业务码为 `E_AUDIO_FORMAT`；处理器内部异常隔离为 `E_INTERNAL`。
- `diagnostics[]`：期望段数不符、规模档提示不符、噪声容差失败、清单文件缺失等诊断。
- `cache_audit`：复用/重算明细及计数。

并发模式最终仍按文件名升序整理样本、帧、段、缓存审计、失败和诊断。串行与线程池的 `samples`、`cache_audit`、`failures`、`deterministic_digest` 逐字段一致；仅 `execution` 中的执行模式/观察并发数可以不同。

## 实际验证结果

在 Python 3.10.12、NumPy 2.2.6 下执行 `python3.10 -m unittest discover -s tests -v`，10 项测试全部通过。固定种子 `20261120` 的 6 个样本共 23,401 帧，最长的单片为 20,000 帧；共检测到 28 个语音段，与清单期望 28 段完全一致；稳定底噪样本的 RMS、ZCR、平坦度统计全部落在配置容差内。

已验证的行为包括：

- 全量运行 24 个阶段产物全部重算；同配置再次运行 24 个全部复用，端点结果摘要不变。
- 将高门限从 5.0 改为 5.1 后，6 个特征产物全部复用，策略/状态/端点按阶段重算；增量结果与同等配置全新状态目录的全量结果摘要一致。
- 强制切换到 `large` 档后，已属于 `large` 的样本继续复用可复用阶段，其余阶段真实重算；增量与全量结果摘要一致。
- 串行与 4 线程结果的所有确定性字段摘要一致，线程池观察并发数未超过配置上限。
- 异常 RIFF 文件返回 `E_AUDIO_FORMAT` 且其余样本继续成功；损坏快照前缀被拒绝并从最后完整前缀恢复；CLI 收到 `SIGINT` 后退出码为 0。

测试会使用临时目录，不修改仓库内样本。要保留一次人工运行报告，执行快速开始中的 `run` 命令即可。
