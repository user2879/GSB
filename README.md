# 本地音频指纹与相似片段检索引擎

这是一个仅依赖 Python 3.10+ 与 `numpy` 的本地实现。它从确定性 WAV 样本生成、参考曲目索引、待检曲目并发指纹提取，到时间偏移估计、边界细化和原子 JSON 发布形成完整闭环。

## 快速开始

```bash
python3 -m pip install --user -r requirements.txt
python3 -m audiofp.demo --workspace .
python3 -m audiofp run task.json
python3 -m unittest discover -s tests -v
```

`audiofp.demo` 会重新生成 `samples/*.wav`、`task.json`，连续运行两轮并比较结果字节，然后写入：

- `output/report.json`：完整技术报告、帧时间轴、逐候选结果、首次超容差位置和异常审计。
- `output/results.json`：供业务系统集成的精简、确定性结果。
- `output/demo_validation.json`：本轮定位误差、异常码、确定性、预算和漂移验证。
- `state/status.json`：运行状态，使用同目录临时文件原子替换。

## 本轮实际验收结果

环境：Linux，Python 3.11.0rc1，NumPy 2.4.6。

| 验收项 | 实际结果 |
|---|---|
| 端到端业务码 | `E_OK`（验证报告）；任务报告因四个预期异常为 `E_PARTIAL` |
| 变速 + 噪声 + 0.36 s 局部插入 | 两个相似片段均命中，偏移随位置向减小方向漂移 |
| 最大时间定位误差 | `147.982 ms` |
| 变速片段测得漂移 | 第 1 段 `-2.044%`，第 2 段 `-1.319%` |
| 帧整数/理论位置最大映射误差 | `0.8` 采样点 |
| 偏移换算误差 | 全部在 `1024` 采样点容差内 |
| 重复运行 | 2 轮结果 SHA-256 相同：`f01a8a9b609b53974a980877d213b01c4429b73356168b26ad78ec009a2f4732` |
| 异常注入 | `E_TRUNC`、`E_SILENT`、`E_RATE`、`E_BAND` 全部按预期区分 |
| 频谱预算 | 容量 42 帧，峰值在途 42 帧，未超预算 |
| 自动化测试 | `12/12` 通过，含 SIGTERM 中断后恢复与原子回滚 |

更详细的算法、约定和结果见 [docs/TECHNICAL.md](docs/TECHNICAL.md)。

## 输入任务格式

`frame_length` 与 `hop_length` 表示 `base_sample_rate` 下的采样点数；其他采样率文件使用精确有理数比换算帧位置。频带单位是 Hz。

```json
{
  "allowed_sample_rates": [16000, 44100, 48000],
  "references": [{"id": "ref", "path": "samples/reference_song.wav"}],
  "queries": [{"id": "query", "path": "samples/query_speed_noise.wav"}],
  "fingerprint": {
    "frame_length": 1024,
    "hop_length": 256,
    "base_sample_rate": 16000,
    "bands": [[80, 600], [600, 1400], [1400, 2600], [2600, 4500], [4500, 7600]],
    "peak_neighborhood": {"frequency_bins": 3, "time_frames": 2},
    "min_peak_db": -42,
    "fanout": 5,
    "max_pair_delta_frames": 24,
    "peaks_per_band": 2,
    "window": "hann"
  },
  "retrieval": {
    "bucket_capacity": 256,
    "voting_window_frames": 16,
    "min_matches": 40,
    "max_candidates": 10,
    "overlap_policy": "keep_highest",
    "boundary_gap_frames": 21,
    "boundary_padding_frames": 2
  },
  "tolerance": {
    "score_scale": 1000,
    "score_tolerance": 1e-12,
    "time_offset_tolerance_samples": 1024
  },
  "concurrency": {"max_workers": 3, "spectrum_buffer_bytes": 524288},
  "state_dir": "state",
  "outputs": {"report": "output/report.json", "results": "output/results.json"}
}
```

完整可运行示例见生成的 `task.json`；其中 `ground_truth` 只用于示例验证，引擎检索不读取它。

## 可集成接口

```python
from audiofp import load_task, run_task
report = run_task(load_task("task.json"))
```

组件级接口：

- `audiofp.wavio.read_wav(...)`：严格解析 16-bit 单声道 PCM WAV。
- `audiofp.fingerprint.fingerprint_audio(audio, config, ...)`：从已解析音频提取帧时间轴、峰值和指纹。
- `audiofp.fingerprint.extract_fingerprint(path, config, ...)`：路径级便捷入口。
- `audiofp.index.ReferenceIndex`：稳定哈希桶索引。
- `audiofp.retrieval.retrieve_similar_fragments(index, query, ...)`：投票、偏移、边界与重叠消解。
- `audiofp.engine.run_task(task_or_path, cancel_token=...)`：并发任务编排和原子报告发布。

`examples/components_example.py` 演示如何绕过任务编排器直接组合这些模块。

## 业务错误码

| 错误码 | 含义 | 处理策略 |
|---|---|---|
| `E_JSON` / `E_SCHEMA` | JSON 或任务结构非法 | 全局失败，不写报告/结果 |
| `E_RATE` | 采样率不允许或声明值不匹配 | 波形级或受影响曲目级 |
| `E_BAND` | 频带非严格递增或越过 Nyquist | 全局或单曲目覆盖参数级 |
| `E_WAV` | 非 16-bit 单声道 PCM | 受影响曲目级 |
| `E_TRUNC` | RIFF/chunk/数据截断 | 受影响曲目级 |
| `E_SILENT` | 整首音频静音 | 受影响曲目级 |
| `E_IO` | 文件读取失败 | 受影响曲目级 |
| `E_WR` | 输出或状态写入失败 | 回滚并保留旧输出 |
| `E_INTERRUPTED` | SIGINT/SIGTERM/取消令牌 | 不发布新输出，可重新运行恢复 |
| `E_INTERNAL` | 未预期内部错误 | 取消在途任务并返回首个致命错误 |

可恢复的曲目数据错误会写入对应片段结果并继续处理其他曲目；内部错误会取消其余任务。详细边界见技术说明。

## 中断与恢复

CLI 捕获 `SIGINT` 和 `SIGTERM`，设置取消令牌。引擎在参考曲目、频谱分块、候选检索和正式发布前检查令牌。中断时旧报告/结果保持不变，`state/status.json` 标记 `interrupted`；同一任务再次运行会从输入确定性重建并发布完整结果。`SIGKILL` 无法被进程处理，但由于正式输出只在全部成功后替换，重新运行同样安全。

