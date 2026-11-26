# 音频指纹检索诊断实现说明

## 1. 运行方式

```bash
python3 -m venv --without-pip --system-site-packages .venv
.venv/bin/python -m pip install --ignore-installed numpy==1.26.4

PYTHONPATH=. .venv/bin/python scripts/generate_samples.py
PYTHONPATH=. .venv/bin/python -m audio_fingerprint.cli validate --workspace sample_workspace
PYTHONPATH=. .venv/bin/python -m audio_fingerprint.cli rebuild --workspace sample_workspace --state state/cli
PYTHONPATH=. .venv/bin/python -m audio_fingerprint.acceptance
PYTHONPATH=. .venv/bin/python -m unittest discover -s tests -v
```

验收报告写入 `artifacts/verification_report.json`，简表写入 `artifacts/verification_summary.txt`。

## 2. 目录与数据流

- `audio_fingerprint/synth.py`：固定整数种子和本地确定性 RNG 合成旋律、和声与打击瞬态。
- `audio_fingerprint/wavio.py`：仅用标准库 `wave` 读写 8/16/24/32-bit WAV，并手工解析 RIFF/`fmt `/`data` 块。
- `audio_fingerprint/manifest.py`：曲目清单、片段声明、配置和编辑命令流校验。
- `audio_fingerprint/normalize.py`：采样率整数倍块平均/确定性线性重采样和位深归一。
- `audio_fingerprint/spectrogram.py`：numpy FFT、频带幅度、能量、Q 格式定点量化和逐帧残差。
- `audio_fingerprint/peaks.py`：阈值、频带范围和邻域局部最大峰值。
- `audio_fingerprint/fingerprints.py`：频率对 + 时间差分指纹和时间档位取整方向。
- `audio_fingerprint/buckets.py`：内容寻址倒排桶、容量淘汰和被挤出条目账本。
- `audio_fingerprint/retrieval.py`：候选计数、边界差值、定点偏移投票、版本裁决。
- `audio_fingerprint/scheduler.py`：稳定预算分配、片段配额、待办和串行/线程池映射。
- `audio_fingerprint/store.py`：JSON 内容缓存、临时文件 + `os.replace` 原子快照。
- `audio_fingerprint/engine.py`：依赖键、复用账本、批次调度、快照提交和 canonical 导出。
- `audio_fingerprint/acceptance.py`：本地端到端验收。
- `tests/test_pipeline.py`：规则单测、预算/并发/fuzz 一致性测试。

主链路：

```text
manifest/config/commands
  -> WAV 容器与声明校验/失败待办
  -> SR/bit-depth 归一与原始格式指纹
  -> 定点谱图帧 + 量化残差
  -> 峰值集合
  -> 峰值对/时间量化指纹
  -> 容量倒排桶
  -> 查询候选/边界差值
  -> 偏移投票/版本权威裁决
  -> 内容寻址缓存与复用账本
  -> 全局版本快照原子提交
```

## 3. 确定性与格式归一

所有哈希输入均为 `sort_keys=True, separators=(",",":")` 的 canonical JSON；没有使用系统时间、全局 `random` 或网络音频。随机源由固定主种子 `0xAF9408`、片段整数种子和确定性 `SeedSequence` 派生。

合成器先产生精确的 int8 网格振幅 `k`，再编码：

- 8-bit：`k + 128`
- 16-bit：`k * 256`
- 24-bit：`k * 65536`
- 32-bit：`k * 16777216`

因此不同位深解码后均为精确的 `k / 128.0`。样例源采样率是内部 11025 Hz 的整数倍（11025/22050/44100），归一使用重复样本的整数块平均，所以同一内容的 8-bit/44.1k 与 16-bit/22.05k 版本产生逐帧相同的 `energy_q`、`bands_q`。原始 `format_fingerprint` 仍不同，并保留原始 SR/位深/重采样比率。

## 4. 业务码

| 业务码 | 含义 |
| --- | --- |
| `E1001` | WAV 文件缺失 |
| `E1002` | RIFF/WAVE/fmt 等容器头非法 |
| `E1003` | 实际采样率不在声明集合 |
| `E1004` | 实际位深不在声明集合 |
| `E1005` | data 块长度/采样数不一致 |
| `E2002` | `(track_id, version)` 冲突 |
| `E2003` | 命令结构或目标非法 |
| `E3001` | 峰值阈值不是正整数 |
| `E3002` | 桶容量不是正整数 |
| `E3003` | 时间量化档位越界 |
| `E3004` | 预算/配额不是正整数 |
| `E3005` | 其他配置非法 |
| `B4001` | 单片段累计采样点超过配额，本批中止该片段 |
| `B4002` | 单批采样点预算不足以分配下一稳定单元 |
| `B4003` | 帧计算失败 |
| `B4004` | 收到可处理退出信号，当前批次后落盘 |

校验层收集全部错误并写待办，不会因单个坏文件或坏命令整体退出。

## 5. 精确失效与缓存键

缓存键均带类型前缀和 SHA-256：

- 帧键：内容指纹、归一内容指纹、`format_fingerprint`、谱图配置指纹、频谱量化档位、帧序号和起点。
- 峰值集合键：自身及邻域帧键、该帧生效阈值的 scoped peak-rule 指纹。
- 指纹对键：锚点/目标峰值集合键、配对配置、时间量化档位/步长/取整结果。
- 桶键：指纹键、容量配置、桶内全部来源对键和保留对键。
- 查询结论键：查询对键、被命中桶键、投票配置。

因此：

- 改峰值阈值不会使谱图帧键变化，帧全部 `reuse_hit`，只使峰值集合及下游失效。
- 改单帧阈值只改变该帧的 peak-rule 键；只有引用该峰值集合的指纹对和包含这些对的桶失效。
- 改桶容量不改变帧、峰值或指纹对，但所有桶键中的容量域变化。
- 改时间档位会真实改变 `(freq_a, freq_b, gear, delta_q)` 指纹键和桶归属。
- 替换/删除/新增片段只改变相关帧、峰值、对以及实际引用这些对的桶。

`state/<run>/cache/*.json` 是逐条持久化产物；`snapshot.json` 是整体活动快照，`commits/snapshot-XXXXXXXX.json` 保存版本提交。写入使用同目录临时文件和 `os.replace`。

## 6. 候选、投票与版本裁决

候选按 `hit_count desc, track_id asc, version asc, clip_id asc` 排序。`candidate_limit` 截断处的最后一个候选标记为 `boundary_candidate`，并输出下一候选及 `hit_count_delta`、`strength_delta_q`、是否并列。

每个命中的偏移为 `db_anchor_frame - query_anchor_frame`。每条命中投一个 `2^vote_fraction_bits` 的定点票，直方图同时记录计数、`votes_q` 和强度和；主偏移按 `votes_q desc, abs(offset) asc, offset asc` 选择。候选列表仍按总命中数排序，但最终权威按主偏移的对齐命中数/对齐强度裁决，避免离散碰撞以总命中数掩盖连续对齐。同曲目多版本按 `main_hit_count/main_strength_sum_q/version` 决定权威版本，被覆盖版本和原因写入 `version_arbitrations[].covered_versions`。

## 7. 调度、并发与恢复

调度单元按 `(track_id, version, clip_id, frame_index, product_id)` 排序。每个帧单元消耗 `frame_size` 个采样点；分配时先检查批次剩余预算，再累计片段配额。超配额的当前单元完成后，该片段剩余帧写入 `frame_processing` 待办，下一批从最小稳定键继续。

`ThreadPoolExecutor(max_workers=min(config.max_concurrency, selected_count))` 只负责纯计算；结果通过 `executor.map` 保持输入顺序，缓存写入和快照提交在主线程按稳定顺序完成，所以串行/2 线程/4 线程 canonical 指纹一致。

`SIGINT/SIGTERM` 处理器只设置停止标志；调度器仍完成已分配批次、写缓存并原子提交快照，然后不再开始下一批。被 `SIGKILL`/断电等强制终止时，未 rename 的临时文件不会被加载，恢复只读取 `snapshot.json` 或最新有效 `commits/snapshot-*.json`。

## 8. 已验证的一致性

见 `docs/VERIFICATION.md` 和 `artifacts/verification_report.json`。自动化测试还包含 3 组固定种子的 8 步随机编辑序列，以及 `256/768/2048` 采样点预算耗尽恢复；每组均与空白全量重建 deep-diff 为空。
