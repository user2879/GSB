# 技术设计说明

## 1. 总体结构

| 模块 | 职责 |
| --- | --- |
| `loudness_workbench/config.py` | 口径常量、容差、块长边界、压缩参数与格式校验 |
| `loudness_workbench/errors.py` | 六个业务错误码与统一异常 `WorkbenchError` |
| `loudness_workbench/audio_io.py` | 24bit PCM WAV 确定性读写（仅用标准库 `wave`） |
| `loudness_workbench/synth.py` | 固定种子 5.1 素材合成、响度校准、交叉淡化与标签 |
| `loudness_workbench/bs1770.py` | K 加权、流式 400ms 窗、双级门控、LRA、真峰值 |
| `loudness_workbench/compressor.py` | 对数域压缩器、固定 10ms 控制帧、起音/释放平滑 |
| `loudness_workbench/snapshot.py` | 不可变快照与单调版本控制 |
| `loudness_workbench/budget.py` | 交互预算监控与跳帧降级（可注入时钟） |
| `loudness_workbench/engine.py` | 流式引擎，串联表、压缩器、预算与快照 |
| `loudness_workbench/offline.py` | 离线整段扫描、报告装配、流式/离线对照判定 |
| `loudness_workbench/demo.py` | 固定序列演示编排（确定性时钟、固定步骤） |
| `loudness_workbench/ui/main_window.py` | PySide6 界面，只读消费快照 |

## 2. 一致性如何保证

1. **同一条测量代码路径**：实时与离线都使用 `StreamingLoudnessMeter`。
   离线只是一次性把整段信号喂给同一个类，因此不存在两套算法漂移。
2. **滤波状态跨块延续**：K 加权与真峰值 FIR 均使用 `scipy.signal.lfilter`
   的 `zi/zf` 状态传递，块边界不重置状态，切分方式不改变输出样本。
3. **积分窗与块长解耦**：测量窗固定 400ms、步进 100ms，音频块长只决定
   每次喂入样本数；未凑满一窗的样本保存在 `_residual` 中续接。
4. **压缩器控制网格固定**：控制平面恒为 10ms 一跳，与音频块长无关。
   起音/释放为 dB 域一阶平滑，状态可序列化恢复，因此任意块长切分、
   块长切换或重进播放，增益包络逐帧相同（演示中最大差为 0.0 dB）。
5. **状态可往返**：`get_state/set_state` 覆盖滤波器、残差缓冲、门控块
   记录、真峰值状态与压缩器平滑值，重新测量基于快照恢复而非清零。

## 3. BS.1770 口径

- K 加权 = 高搁架（+4dB，fc≈1681.97Hz，Q≈0.7072）级联 RLB 高通
  （fc≈38.135Hz，Q≈0.5003），系数按 RBJ biquad 公式在 48kHz 下计算。
- 声道权重：L/R/C 为 0dB，Ls/Rs 为 +1.5dB，LFE 不参与（权重 0）。
- 400ms 积分窗、100ms 步进；块响度 `LKFS = -0.691 + 10 log10(加权均方)`。
- 双级门限：绝对门限 -70 LKFS；相对门限 = 绝对门限通过块均值 -10 LU。
- LRA 取绝对门限通过块的第 10 与第 95 百分位之差（线性插值）。
- 真峰值：每声道 4 倍零插补 + 41 阶 Hamming 线性相位 FIR 抗镜像，
  取全声道最大幅值转 dBTP；FIR 状态跨块保留。

## 4. 快照与预算语义

- `Snapshot` 为 frozen dataclass，带单调 `version` 与覆盖样本范围。
- `SnapshotStore.commit` 拒绝任何 `version <= 当前版本` 的提交并抛出
  `E_SNAPSHOT_STALE`，界面只从已接受快照读数。
- `BudgetMonitor` 仅控制刷新节奏：超 40ms 预算则该帧跳帧并累计
  `frames_degraded`，连续 3 次超限升级为 `E_BUDGET_EXCEEDED`。降级路径
  不触碰任何测量状态，最终报告不受影响。

## 5. 业务错误码覆盖

| 错误码 | 触发链路 |
| --- | --- |
| `E_AUDIO_FORMAT` | 采样率/声道/位深不符、信号含 NaN/Inf、维度错误 |
| `E_BLOCK_BOUNDS` | 块长超出 10–200ms 或不是 10ms 整数倍 |
| `E_COMP_STATE` | 压缩状态缺失、hop 或参数与活动配置不一致 |
| `E_SNAPSHOT_STALE` | 旧版本快照回退提交 |
| `E_BUDGET_EXCEEDED` | 连续超预算达到声明次数 |
| `E_REPORT_MISMATCH` | 流式与离线响度/真峰值/包络差超容差 |

## 6. 确定性策略

- 全部随机量来自固定种子的 `numpy.random.default_rng(seed)`。
- WAV 使用固定 24bit 量化与标准库写入；JSON 统一 `sort_keys`、固定
  缩进、LF 行尾；报告不含墙钟时间。
- 演示时钟使用 `ScriptedClock` 脚本化时间，预算降级可重复复现。
- 两次运行对 WAV、标签、报告做 SHA-256/逐字节比对。

## 7. 声明容差

| 指标 | 容差 | 演示实测差 |
| --- | --- | --- |
| 整段门控响度（流式 vs 离线） | 0.1 LU | 0.0 LU |
| 真峰值（流式 vs 离线） | 0.05 dBTP | 0.0 dBTP |
| 压缩增益包络最大差 | 1e-9 dB | 0.0 dB |
| 块边界流式/离线增益差 | 1e-9 dB | 0.0 dB |
