# 多通道音频时钟同步与漂移补偿诊断工作台

这是一个面向 8 通道、48 kHz、24-bit PCM 采集设备的本地确定性诊断工作台。参考通道与 7 个从通道使用独立晶振采样；从通道具有非整数固定延迟、独立 ppm 偏置和缓慢温漂。工作台完成素材生成、脉冲检测与配对、分段漂移估计、窗函数 sinc 重采样、逐脉冲独立核验、规模自适应策略选择、不可变快照和静态诊断视图。

项目不依赖外部音频资产、GPU、Docker 或网络服务。运行期只依赖 NumPy 与 SciPy。

## 环境

Windows 10/11 PowerShell：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

本项目实际验证环境为 Python 3.11、NumPy 1.26.4、SciPy 1.13.1。Linux/WSL 同样可运行：

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## 快速运行

生成 30 秒低漂移素材：

```powershell
python -m sync_diag synth --config configs/low_drift.json --duration 30 --out data/low_30
```

执行估计、补偿、核验、快照与报告：

```powershell
python -m sync_diag analyze --input data/low_30 --out runs/low_30
```

默认在 120 秒以内保留 `aligned.wav`。长时间素材使用“全流计算 + 流式哈希 + 局部脉冲窗核验”，不会把对齐 WAV 写入磁盘，但保留其 WAV 等价 SHA-256、逐脉冲残差与快照，从而把磁盘峰值限制在约一个源素材。需要保留长档对齐文件时加 `--keep-aligned`，需要关闭双块长复核时加 `--skip-consistency`。

运行完整确定性矩阵：

```powershell
python -m sync_diag demo --configs configs --out artifacts/demo --variants low_drift,high_drift,temp_drift --scales 30,1800
```

运行测试：

```powershell
python -m unittest discover -s tests -v
```

复核正式演示产物（只读，不修改同步状态）：

```powershell
python tools/audit_demo.py --demo artifacts/demo
```

审计会重新计算源 WAV、清单、配置、报告、快照和视图哈希，并从逐脉冲明细重新汇总 RMSE；结果写入 `artifacts/demo/validation.json` 与 `artifacts/demo/validation.sha256`。

## 合成模型

配置文件给出种子、各通道独立时钟参数和测试信号参数。音调相位与显式通道参数一起写入 `config.resolved.json`，因此相同配置与种子会生成逐字节相同的 PCM24 WAV、清单和配置快照。

记标称采样率为 `fs`，从通道样本序号为 `n`，有效 ppm 为：

```text
ppm(n) = base_ppm + temp_ppm_per_c * (T(n) - 25 C)
T(n) = mean_c + amplitude_c * sin(2*pi*n/(fs*period_s) + phase_rad)
```

工作台使用温漂正弦的解析积分计算累积钟差 `D(n)`，从通道源坐标为：

```text
u(n) = n + fixed_delay_samples + D(n)
```

参考通道 `u(n)=n`。参考信号由 4 个多音分量和周期高斯瞬态组成。从通道按照 `u(n)` 对参考连续波形做三次样条采样。每个脉冲在参考时间轴上的整数样本位置写入 `manifest.json`；每个从通道还记录解析求根的观测脉冲位置，仅用于事后独立核验，不参与漂移估计。

为避免从通道时钟较快时末尾缺少插值样本，采集 WAV 自动附带由最大正钟差计算的保护帧。保护帧属于采集文件，但不属于 30 秒/30 分钟分析时长，也不写入参考脉冲真值。

## 算法链路

### 1. 脉冲检测与配对

检测器在参考真值附近使用配置化配对窗搜索。为避免多音把峰顶推偏，检测器先用已知多音频率在脉冲窗边缘拟合音调基线，再在扣除基线后的残差能量上计算质心。30 秒实测单脉冲检测噪声约 `0.01` 样本量级。

配对要求每个参考真值都找到唯一检测脉冲；缺失、越界或重复映射立即返回 `E_PULSE_MISSING`，不返回部分脉冲结果。

### 2. 分段局部线性漂移估计

每个从通道计算观测偏移 `observed - reference`。按配置的脉冲数切段，逐段最小二乘拟合：

```text
offset(t) = a + b * (t - segment_center)
drift_rate_ppm = b * 1e6
```

报告包含每段漂移率、95% 区间（含 Student-t 小样本修正）、段中心偏差置信区间、残差 RMSE/最大值和段起止累积偏差。任一段绝对漂移率超过 `drift_tolerance_ppm` 时抛出 `E_DRIFT_EXCEED`，上下文包含通道、段、时间范围、估计值和区间，不会继续静默补偿。

### 3. 三种策略与预算决策

`decision.json` 对三种策略给出确定性的内存/耗时模型、误差上界与选择依据：

| 策略 | 适用条件 | 曲线构造 | 默认 30 s | 默认 30 min |
|---|---|---|---|---|
| `full_exact` | 短素材、全量锚点可物化 | 每脉冲邻域局部线性预测，全锚点插值 | 选中 | 内存模型超预算 |
| `block_accumulation` | 长素材、分块资源有界 | 每块局部线性拟合，块中心锚点累积 | 可行 | 选中 |
| `sparse_anchor_interpolation` | 更紧内存预算 | 固定步长锚点线性插值 | 可行 | 仅预算收紧时 |

选择顺序固定为：预算/精度可行性过滤，然后短素材优先 `full_exact`，长素材优先 `block_accumulation`，只有分块策略不可行时才使用稀疏锚点。若没有策略同时满足内存、时间与声明的 `accuracy_target_samples`，返回 `E_STRATEGY_BUDGET`。误差上界按温漂曲率与锚点跨度计算并随决策报告固化。

### 4. 连续相位窗函数 sinc 补偿

输出参考时基样本 `m` 的源坐标为：

```text
u(m) = m + estimated_offset(m)
```

插值使用 Kaiser 窗窗函数 sinc，预计算 4096 个多相相位，默认单边 8 taps。每个输出样本都基于绝对源坐标计算，不使用块内相对相位或块内滤波状态，因此块长仅影响 I/O 分块，不改变插值相位。采集保护帧保证末尾 sinc 窗口完整；起始固定延迟区使用边界延拓。

工作台在每个块边界检查坐标步进连续性，并对同一素材使用两种块长重复补偿。两份对齐流的 WAV 等价哈希要求相同，残差差异还必须小于 `consistency_tolerance_samples`；否则返回 `E_RESAMPLE_PHASE`。

### 5. 独立核验

核验只使用清单中的参考脉冲真值和重新检测后的对齐 WAV：

- `before_rmse_samples`：补偿前脉冲偏移 RMSE（包含固定延迟和累积漂移）。
- `after_rmse_samples`：对齐输出相对参考真值的残余偏差 RMSE。
- `fixed_delay_error_samples`：估计固定延迟与清单固定延迟之差。
- `accumulated_drift_error_samples`：逐脉冲估计累积漂移与真值累积漂移之差。
- `measurement_component_after_samples`：实测残余与“固定延迟误差 + 累积漂移误差”模型的差。

每个脉冲均写入 `residual_report.json`。若任一从通道补偿后没有改善或发生劣化，汇总状态明确写为 `DEGRADED_OR_NO_IMPROVEMENT`，不会隐藏。

## 快照与视图

快照包含采集哈希、预算决策、漂移估计、重采样配置、逐脉冲核验。`content_hash` 为规范化 JSON 载荷的 SHA-256。发布规则：

- 新版本号大于当前版本：允许整体替换。
- 新版本号小于当前版本：`E_SNAPSHOT_STALE`。
- 同版本同内容：幂等返回。
- 同版本不同内容：`E_SNAPSHOT_STALE`，版本不可变。

`diagnostic_view.svg` 只读取当前快照并绘制各通道漂移曲线、脉冲时间刻度和补偿后逐脉冲残差，不重新计算或修改同步状态。实测渲染前后快照文件哈希不变。

## 错误码

| 错误码 | 触发条件 |
|---|---|
| `E_CH_FORMAT` | 通道数、位深、采样率、WAV 帧数不符 |
| `E_PULSE_MISSING` | 脉冲缺失、超出配对窗、重复映射或锚点不足 |
| `E_DRIFT_EXCEED` | 分段漂移率绝对值超过声明容差 |
| `E_RESAMPLE_PHASE` | 块边界相位跳变、双块长结果不一致或源窗口越界 |
| `E_STRATEGY_BUDGET` | 无策略满足内存/耗时/精度预算，或显式策略越界 |
| `E_SNAPSHOT_STALE` | 旧版本、同版本不同内容、指针哈希或快照内容哈希失效 |
| `E_MANIFEST_HASH` | 清单侧车哈希、WAV 哈希或配置/清单哈希不匹配 |
| `E_CONFIG` | 配置字段缺失或取值非法 |

所有业务错误返回非零退出码和确定性 JSON 上下文；分析采用 staging 目录，失败不会留下半成品分析目录。

## 输出目录

```text
runs/<name>/
  aligned.wav                  # <=120 s 默认保留；长素材可用 --keep-aligned
  drift_report.json            # 逐通道分段漂移、区间、残差、曲线锚点
  decision.json                # 策略依据、资源模型、误差上界
  residual_report.json         # 逐脉冲补偿前后偏差与误差分解
  pulse_pairing.json           # 检测位置与参考真值配对
  diagnostic_view.svg          # 只读快照视图
  snapshots/
    sync_snapshot.v000001.<content-hash-prefix>.json
    current_snapshot.json
  hashes.json                  # 每步规范化内容哈希与文件哈希
  run_report.json
```

确定性输出不包含时间戳、墙钟耗时、绝对路径或随机 UUID。所有 JSON 使用排序键和固定缩进；哈希使用排序、紧凑、UTF-8 规范化 JSON。相同输入重复运行的分析目录逐文件哈希一致。

## 实际验证结果

以下结果来自本工作区的本地确定性素材与自动化测试：

| 项目 | 结果 |
|---|---|
| 自动化测试 | 14 项通过 |
| 30 秒低漂移素材 | `full_exact`，最大通道 RMSE 0.0782 样本 |
| 30 秒高漂移素材 | `full_exact`，最大通道 RMSE 0.0434 样本 |
| 30 秒温漂素材 | `full_exact`，最大通道 RMSE 0.0847 样本 |
| 30 分钟低漂移素材 | `block_accumulation`，最大通道 RMSE 0.0457 样本 |
| 30 分钟高漂移素材 | `block_accumulation`，最大通道 RMSE 0.0505 样本 |
| 30 分钟温漂素材 | `block_accumulation`，最大通道 RMSE 0.1010 样本 |
| 双块长复核 | 0.25 s / 2.0 s WAV 等价哈希相同，残差差值为 0 |
| 确定性复核 | 同一 30 秒输入两次分析，全部输出文件逐字节一致 |
| 超容差链路 | 注入约 180 ppm 漂移、容差 40 ppm，稳定返回 `E_DRIFT_EXCEED` |
| 正式矩阵 | 6 个运行全部 `target_met=true`，索引见 `artifacts/demo/demo_summary.json` |
| 独立交付审计 | 1320 项检查通过，结果见 `artifacts/demo/validation.json`，SHA-256 为 `c21e64859045ea74ba2bfa240aefa8bfa200b97e52992a70e8adb8f861adf388` |

## 磁盘与规模预算

30 分钟、8 通道、48 kHz、PCM24 源素材约为 `1800 * 48000 * 3 * 8 = 2,073,600,000` 字节（约 1.93 GiB），另加少量保护帧。工作台对该档默认不写对齐 WAV，而是全流计算、流式计算 WAV 等价哈希，并用局部脉冲窗核验。单个 30 分钟运行的实测报告/快照约 43 MB，其中快照约 19.8 MB、逐脉冲残差明细约 16.6 MB；因此单档工作峰值约为一个源素材加 43 MB，约 2.0 GiB。若显式保留对齐 WAV，需要额外约 1.93 GiB。

正式矩阵为便于逐项复核，保留了三份 30 分钟源 WAV，合计约 5.89 GiB；六次运行的报告/快照合计约 224 MiB。若只保留当前单档结果，可在完成哈希归档后仅保留 30 秒源素材和全部运行报告。
