# 技术说明

## 1. 模块边界与数据流

```text
task.json -> task.py 校验 -> wavio.py 解析
                                |
                                v
                     fingerprint.py 逐帧 FFT/峰值/哈希
                                |
                    index.py 参考索引 + retrieval.py 检索
                                |
                 engine.py 有界并发、预算、报告、原子发布
```

公共接口保持分离：WAV 解析、指纹提取、索引、检索和任务编排互不依赖具体 CLI。`examples/components_example.py` 展示了路径级/音频级两种组合方式。

## 2. 任务校验与错误边界

全局结构在写任何输出前完成校验。校验内容包括：

- 采样率在 `allowed_sample_rates` 中；
- WAV 为 PCM、16-bit、单声道；
- `frame_length > 0`、`hop_length > 0`、`hop_length <= frame_length`；
- 频带 `0 <= low < high < base_sample_rate / 2`，且按顺序不重叠；
- `score_scale`、`score_tolerance`、`time_offset_tolerance_samples` 为正；
- 并发数与频谱预算可以容纳至少一帧；
- 输出路径不为输入文件或任务文件。

单曲目的 `declared_sample_rate` 与头部不一致、`fingerprint_override` 频带越界等错误只影响该曲目；全局任务参数错误导致整体失败。CLI 对全局失败仅输出 JSON 错误对象，不创建报告/结果。

## 3. 精确帧时间轴

帧参数定义在 `base_sample_rate` 下。对源采样率 `r_src`，第 `i` 帧的理论位置使用有理数计算：

```text
start_theory(i) = i * hop_length * r_src / base_rate
end_theory(i)   = start_theory(i) + frame_length * r_src / base_rate
start_int(i)    = round_half_up(start_theory(i))
```

每个 `frame_map` 条目记录：

- `source_start_sample` / `source_end_sample`：源文件中的整数采样点；
- `theoretical_start_sample` / `theoretical_end_sample`：理论有理数映射后的浮点值；
- `theoretical_start_seconds` / `theoretical_end_seconds`：理论时间；
- `start_mapping_error_samples` / `end_mapping_error_samples`：整数落点误差。

本机验证最大绝对映射误差为 `0.8` 采样点。不同采样率文件使用同一时间步长，因此哈希中的帧差仍对应一致的时间尺度。

## 4. 频谱与指纹

1. 按帧时间轴切出严格源区间，乘 Hann/Hamming/矩形窗。
2. 对每帧执行 `rfft`，得到幅度谱。
3. 每帧以自身最大幅度为基准计算相对 dB，因此增益偏移不改变峰选择。
4. 在每个频带内做频率邻域局部极大值检测，并按强度选前 `peaks_per_band` 个峰。
5. 做 `time_frames` 邻域的时间局部极大值过滤。
6. 每个锚点与后续 `fanout` 个时间差不超过 `max_pair_delta_frames` 的峰配对。

哈希键包含：

```text
(anchor_band, target_band, frame_delta,
 anchor_pitch_bucket, pitch_interval_bucket,
 anchor_rank, target_rank)
```

频带与帧差是核心时间/频率分布键；量化音高、峰间音程和峰秩用于抑制重复素材造成的桶碰撞。音程在统一变速下近似不变，因而对时间伸缩更稳健。桶容量达到上限后按输入顺序稳定丢弃，保证确定性。

## 5. 得分双累计与排名

每个匹配贡献一个由两侧峰显著度得到的 float64 权重 `w`。候选片段内按稳定匹配键排序后逐步累计：

```text
float_total += w
fixed_total += round(w * score_scale)
```

报告同时给出：

- `score_float_reference`；
- `score_fixed_units`；
- `score = score_fixed_units / score_scale`。

`score_trace` 使用并行数组记录每一步的 float 累计、定点累计与绝对差值；`first_exceed_score_tolerance` 标出首次超过 `score_tolerance` 的步骤和两条时间轴帧号。所有候选过滤和最终排序只使用定点得分及确定的区间键，float 值仅作对照，不参与排名。

## 6. 候选检索与偏移

1. 以 `(query_anchor - reference_anchor)` 计算整数帧偏移。
2. 以 `voting_window_frames / 2` 为半径按权重选择偏移种子，形成不重叠的偏移簇。
3. 在簇内按查询时间连通域拆分，插入造成的大间隙会形成独立片段候选。
4. 对查询轴先做局部密度裁剪，再用 0.1% 分位剔除稀疏离群尾部。
5. 参考轴边界从同一组匹配的参考帧直接得到，避免两个轴独立取边界造成错位。
6. 以帧级匹配密度和匹配数细化并增加声明式 padding。

偏移约定：

```text
query_time ~= rate_ratio * reference_time + intercept
offset_seconds = query_time - reference_time
```

输出整数帧偏移、float 参考采样点偏移、分析域采样点偏移及二者换算误差。本机任务 `time_offset_tolerance_samples=1024`，所有已保留片段均满足该容差。

对每个候选还拟合 `rate_ratio`，并输出：

- `direction`：`increasing`、`decreasing` 或 `stable`；
- `drift_percent`；
- 片段跨度上的漂移秒数与参考采样点数。

本地 1.025 倍速/伸缩样本的两个片段测得漂移方向均为 `decreasing`，幅度分别为 `-2.044%` 和 `-1.319%`。

## 7. 重叠消解

- `keep_highest`：按定点得分降序，在同参考轴与查询轴均重叠时保留高者。
- `merge`：用并查集求二维区间连通分量，合并匹配集合、去重后重新执行边界细化和双累计。

两种策略都只使用稳定排序键；合并结果与候选遍历顺序无关。引擎分别对每首参考曲目消解，避免不同参考曲目互相抑制。

## 8. 并发与频谱预算

- `ThreadPoolExecutor` 的 `max_workers` 由任务给出。
- 频谱计算按块进行，每块申请全局 `FrameBudget` 许可。
- 预算容量按允许的最高采样率和最大 FFT 行宽保守换算：

```text
capacity_frames = spectrum_buffer_bytes // ((fft_size // 2 + 1) * 8)
```

- 等待队列按曲目清单优先级和到达顺序授予许可；预算不足时不会缩短帧长或裁剪频带。
- 每首待检曲目的指纹只依赖自身音频和固定参数，最终候选按固定键排序，因此线程调度不改变结果。
- 可恢复的曲目级业务错误（截断、静音、声明采样率不符、频带越界等）进入错误结果并继续；内部错误立即设置取消令牌并返回首个致命错误。

本机运行中容量为 42 帧，峰值在途为 42 帧，共 64 次预算获取、3 次等待，未超预算。

## 9. 原子输出与恢复

报告和结果先写入各自目标目录的临时文件并 `fsync`。所有临时文件就绪后：

1. 将已有目标移到同目录备份；
2. 逐个用 `os.replace` 发布；
3. 全部成功才删除备份。

任一步失败会删除已发布新文件并恢复全部备份，最后返回 `E_WR`。由于正式发布位于所有计算之后，取消或进程被直接结束时不会留下半个报告。

`state/status.json` 记录 `running`、`interrupted`、`error` 或 `complete`。中断后直接重跑同一任务是确定性恢复路径；测试覆盖 SIGTERM 后无正式输出、状态为 `interrupted`、随后重跑成功。

## 10. 本地验收复现

```bash
python3 -m audiofp.demo --workspace . --runs 2
python3 -m unittest discover -s tests -v
```

样本生成完全由程序决定：

- 2 首 44.1 kHz 参考曲目；
- 48 kHz 变速 1.025、增益与噪声、局部插入变体；
- 16 kHz 增益偏移与噪声变体；
- 44.1 kHz 跨采样率/噪声变体；
- 截断文件、全静音、声明采样率不符、频带越界四类异常。

本轮结果为 `12/12` 测试通过，最大定位误差 `147.982 ms`，帧映射最大误差 `0.8` 采样点，两轮 `results.json` 字节完全一致。

