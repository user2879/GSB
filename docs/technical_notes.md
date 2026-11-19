# 技术说明

## 1. 确定性测量模型

### 1.1 指数正弦扫频

对持续时间 `T`、起止频率 `f0/f1`，生成：

```text
φ(t) = 2π f0 T / ln(f1/f0) · (exp(t/T · ln(f1/f0)) − 1)
x(t) = sin(φ(t))
```

首尾应用 raised-cosine 淡入淡出并将峰值归一到配置驱动幅度。扫频长度、重复次数、采样率、幅度和种子都进入测量阶段内容键。

### 1.2 通道传播

每条通道依次执行：

1. 频域相位斜坡分数延迟：零填充 FFT 后乘以 `exp(-j2π f d/fs)`。
2. 已知增益。
3. Butterworth 零相位带通，避免额外群延迟破坏延迟真值。
4. PCG64 固定种子高斯噪声。

所有通道由同一源时间线驱动。每段源时间线包含起点同步短脉冲和稍后开始的扫频。同步真值为：

```text
truth[channel, repeat] = segment_start[repeat] + delay_samples[channel]
```

记录同时保存最近整数样本、分数采样真值、源样本位置和扫频起点。

### 1.3 24-bit WAV

浮点样本以同一全通道比例量化为 24-bit little-endian PCM。未做逐通道归一化，因此相对增益信息保留。写盘后立即回读校验格式与形状。WAV 字节哈希、源数组哈希和量化记录哈希共同作为下游依赖。

## 2. 反卷积与冲激响应

对每个扫频段，从 `sweep_start - pre_margin` 提取等长录制窗和源参考窗，并应用同一个 Tukey 窗。对重复段先平均频域估计：

```text
X = FFT(window · source)
Y = FFT(window · recorded)
H = Y · conj(X) / (|X|² + λ · max|X|²) · BandMask
h = IFFT(H)
```

`BandMask` 为有效频带内 1、边缘余弦过渡。IFFT 结果循环移位后截断，并保存：

- `nfft`
- 窗类型和 Tukey `alpha`
- 有效频带和 taper
- 截断样本数
- 原点偏移
- 各重复段窗起点
- 公共归一化因子
- IR 和复传递哈希

峰值位置使用 IR 绝对值最大值，亚采样偏移使用三点抛物线。所有通道共享参考通道峰值作为归一化因子，因此相对增益没有被逐通道归一化抹掉。

单通道复算使用同样的录制、源时间线、配置和公共归一化；输出 `ir_hash` 可与快照逐项比较。

## 3. 亚采样延迟与误差上界

### 3.1 估计

参考/目标 IR 在有效延迟频带内计算互相关：

```text
C(f) = FFT(ir_target) · conj(FFT(ir_reference)) · W(f)
ccf = IFFT(C)
```

`W(f)` 使用平滑带限和按频率平方加权，以增强高频相位信息。随后取解析包络 `|Hilbert(ccf)|` 的峰，并在峰值三点上做抛物线：

```text
δ = 0.5 · (c[-1] − c[+1]) / (c[-1] − 2c[0] + c[+1])
```

该解析包络峰值仍属于“互相关峰值 + 抛物线插值”，但避免带限 IR 符号振荡导致的三点符号偏差。

### 3.2 误差上界

```text
CRB = sqrt(1 / (SNR · Σ(Δx)² / Σx²))
curvature_term = 0.125 · |c'' / c_peak|
model_error = |δ_quartic − δ_parabola|
bound = coverage · sqrt(CRB² + curvature_term²)
        + 2 · model_error + quantization_term
```

其中 SNR 由安静前导区一阶差分稳健噪声估计和扫频段 RMS 得到，`coverage` 默认 3。报告同时给出 CRB、曲率项、五次插值模型差和最终 bound。

若：

- 配置的已知相对延迟绝对值超过 `delay_tolerance_samples`；或
- `|estimate − truth|` 超过偏差容差与误差上界中的较大者；

则阶段状态为 `FAIL` 并报告 `E_DELAY_BIAS`。真值和估计始终同时存在。

## 4. 增益/频率响应失配

在 `[f_low, f_high]` 上生成等距分析频点。对每通道以参考通道复传递为基准：

```text
Δmag_dB(f) = 20 log10(|H_ch(f)| / |H_ref(f)|)
Δphase_deg(f) = angle(H_ch(f) · conj(H_ref(f)))
```

频点权重为 `|H_ref(f)|²` 归一化。分析频带按对数边界划分为可配置数量的子带，每子带复制加权统计，并以参考功率权重汇总到总增益失配。

### 4.1 三类误差贡献

**噪声项。** 反卷积时同时累积：

```text
σ_H(f) = σ_time · sqrt(Σ_r |X_r(f)|² / (|X_r(f)|² + λ)²) / repeats
relative_noise(f) = sqrt((σ_H,ch/|H_ch|)² + (σ_H,ref/|H_ref|)²)
noise_db(f) = 20/ln(10) · relative_noise(f)
```

这使弱谱/带边频点的噪声项自然增大。

**窗/截断泄漏项。** 使用 Tukey 窗均值偏差计算窗泄漏 dB，并加上截断外 IR 能量比例：

```text
leak_db = window_leak_db + 10 log10(1 + E_outside / E_total)
```

**限带响应项。** 对当前频带幅度比做对数频率上的二次平滑，残差作为无法由单一加权增益解释的限带形状项；再加分析带宽相对解卷积带宽的 span 项。

最终每频点半宽：

```text
halfwidth(f) = coverage · sqrt(noise² + leakage² + band_limit²)
```

总区间由随机噪声加权 RMS 和系统项加权均值合成，并乘 `sqrt(span_ratio)`；因此分析频带收窄会同时减少频点边缘误差和带宽归一化因子。区间依赖噪声、正则化、截断、窗、限带响应和带宽配置，不退化为固定值。

## 5. 校准 FIR 与重测

目标默认是参考通道实测复传递。对非参考通道设计：

```text
C_desired(f) = H_ref(f) / H_ch(f)
```

频率域比值采用 Tikhonov 正则化并限制最大提升（默认 12 dB）。校准频带内应用目标修正，带外平滑回落 1。FFT 频域采样加入统一线性相位 `exp(-j2π f D/fs)`，`D=(L-1)/2`，截取长度 `L` 后乘 Blackman-Harris 窗。最多执行 3 次修正迭代以补偿窗化后的实际响应。

重测流程不是直接对估计结果做算术校正，而是：

1. 将每个 FIR 因果卷积到原始激励；
2. 经同一通道延迟、增益、带限和固定噪声重放；
3. 重新量化并反卷积；
4. 重新计算频域失配和延迟。

每个频点的残差必须满足：

```text
|Δmag_residual(f)| ≤ tolerance_factor · halfwidth_mag(f)
|Δphase_residual(f)| ≤ tolerance_factor · halfwidth_phase(f)
```

不满足则报告 `E_CALIB_MISMATCH`，并把相邻违规频点合并为具体频带；错误详情包含该频带幅度/相位超出量。

## 6. 增量依赖图与缓存

阶段依赖：

```text
measurement
  -> sync
  -> deconvolution
       -> delay
       -> mismatch (also depends on delay and analysis band)
            -> calibration (also depends on calibration config)
                 -> remeasurement
                      -> verification
```

每个阶段键为以下内容的规范 SHA-256：

- 该阶段配置；
- 上游内容哈希；
- 相关测量/噪声/分析参数。

缓存为内容寻址文件。读取时重新计算载荷内容哈希，若键、schema 或哈希不符，抛出 `E_SNAPSHOT_STALE`，不会静默复用。报告中的 `reuse` 显示 `computed / memory / disk`。

参数变化影响：

| 变更 | 复用 | 重算 |
|---|---|---|
| 正则化 | measurement, sync | deconvolution 及全部下游 |
| 分析频带 | measurement, sync, deconvolution, delay | mismatch, calibration, remeasurement, verification |
| FIR 长度 | measurement, sync, deconvolution, delay, mismatch | calibration, remeasurement, verification |
| 扫频长度 | 无 | 全部 |

## 7. 快照与版本语义

快照构建时：

```text
content_hash = SHA256(canonical_json(payload_without_hash_or_stamp))
version_stamp = "acoustic-workbench.snapshot/1.0#r<revision>:<hash[:16]>"
```

载荷递归转换为 `MappingProxyType` 和 tuple，调用方无法修改。发布采用临时文件 + `os.replace` 原子替换；若目标快照已存在，必须提供前驱内容哈希，否则 `E_SNAPSHOT_STALE`。加载时重新计算 schema、内容哈希和版本戳。

报告与 GUI 从同一个快照载荷读取数据。GUI 只做像素坐标映射、归一化绘制和格式化，不执行 FFT、估计、误差合成或滤波。

## 8. 可复算性

`measurement/` 目录同时保存 WAV、完整源时间线、通道配置、同步真值、FFT/窗/截断/窗起点和归一化元数据，并与 `state/current_snapshot.json` 对应的最终测量同步。CLI `recompute` 读取这些输入，从解码后的静音段重估噪声标准差，只依赖当前 Python 数值库重算指定通道 IR；结果中的 `ir_hash` 可直接与快照比较。

固定演示通过 `--verify-determinism` 在独立临时目录完整重跑同一序列，并按相对路径逐字节比较全部输出文件（缓存、WAV、快照、报告、PNG、错误链和清单）。
