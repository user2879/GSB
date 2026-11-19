# 声学测量与校准本地诊断工作台

这是一个不依赖真实声卡的多通道声学测量、反卷积、延迟/增益失配分析、FIR 校准与误差传播诊断工作台。所有测量样本均由本地确定性模型生成：4 通道、48 kHz、24-bit PCM、20 Hz–20 kHz 指数正弦扫频、固定种子噪声、非整数采样传播延迟和限带滤波。

核心设计原则：

- 同一配置生成的 WAV、源样本、同步真值、快照和报告逐字节一致。
- 诊断视图只读取不可变快照，不在 GUI 内重算 DSP 或统计量。
- 每个阶段使用内容寻址缓存；参数变化只使受影响的下游阶段失效。
- 快照以 schema 版本、内容哈希和版本戳共同校验；缓存/快照不匹配抛出 `E_SNAPSHOT_STALE`。
- 真值、点估计、误差上界、误差区间、校准残余和业务错误都写入同一报告。

## Windows 环境

要求 Windows 10/11 64 位、Python 3.11 64 位。无需声卡或音频设备。

```powershell
cd <project-directory>
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

安装版本与已验证环境一致：`numpy 2.4.6`、`scipy 1.17.1`、`PyQt6 6.12.1`。运行前可关闭占用音频设备的第三方程序，但本工作台本身不打开声卡。

## 快速演示

```powershell
python run_demo.py demo --output output\demo --verify-determinism
```

该命令执行固定序列：

1. 生成含已知延迟、增益和带限差异的四通道样本并完整测量。
2. 进行正则化频域反卷积、同步/延迟估计、增益与频率响应失配分析。
3. 设计线性相位 FIR 校准滤波器，通过同一模拟传播通道重测并检查残余失配。
4. 修改正则化参数，验证仅测量与同步阶段复用。
5. 收窄分析频带，验证反卷积/延迟阶段复用而失配与滤波器阶段重算。
6. 修改 FIR 长度，验证失配结果复用而校准/重测阶段重算。
7. 修改扫频长度，重新生成测量和所有下游阶段。
8. 运行 7 个业务错误码的确定性异常链路。
9. 在临时目录重复完整序列，逐文件比较输出字节。

主要输出：

```text
output/demo/
  measurement/
    measurement.wav                 # 48k / 24-bit / 4ch PCM（与 current_snapshot 同步）
    channel_config.json             # 通道延迟、增益、带限、噪声种子
    sync_markers.json               # 同步标记真值 + 检测值
    manifest.json                   # 音频/扫频/哈希清单
    source_reference.npy            # 完整源时间线，供独立复算
    deconvolution_recipe.json       # FFT 长度、窗、截断、窗起点、归一化
    calibration_filters.npy         # 4 x FIR 系数
  runs/01_initial/ ...              # 每步 snapshot / report / reuse
  error_chain.json                  # 七个错误码的触发结果
  diagnostic_view.png               # 快照渲染视图
  demo_manifest.json                # 各步内容哈希与确定性结果
  state/current_snapshot.json       # 最新不可变快照
  state/cache/                      # 内容寻址阶段缓存
```

`measurement/` 始终绑定 `state/current_snapshot.json` 对应的最终测量，因此可直接用于快照的独立复算。

## GUI

```powershell
python run_gui.py gui --snapshot output\demo\state\current_snapshot.json --output output\gui
```

GUI 提供：

- 频域视图：每通道幅度比、相位差、合成误差区间、校准后残余。
- 时域视图：每通道冲激响应、峰值位置、同步脉冲真值。
- 摘要与错误：延迟真值/估计/上界、增益失配区间、校准残余与错误码。
- 阶段与复用：每个阶段的哈希、缓存来源和是否复用。
- 参数控件：扫频长度、正则化、分析频带、FIR 长度；点击“增量重测”后重新生成快照并刷新视图。

无显示环境可用 Qt `offscreen` 平台渲染 PNG。

## CLI

```powershell
# 从快照生成 JSON + Markdown 报告
python run_demo.py report output\demo\state\current_snapshot.json --output output\report.json --markdown output\report.md

# 用持久化 WAV、源时间线和快照参数独立复算任意通道 IR
python run_demo.py recompute output\demo\state\current_snapshot.json `
  --wav output\demo\measurement\measurement.wav `
  --source output\demo\measurement\source_reference.npy `
  --channel 2 --output-dir output\recompute
```

## 数值方法

- 扫频：`x(t)=sin(2π f0 T/ln(f1/f0)·(exp(t/T·ln(f1/f0))−1))`，首尾 raised-cosine 淡入淡出。
- 传播：频域精确分数延迟、通道增益、4 阶 Butterworth 零相位限带、固定 PCG64 种子高斯噪声。
- 同步：每段起始处短脉冲；真值为 `段起点 + 配置延迟`，记录整数最近样本和分数采样真值。
- 反卷积：同一窗化参考和录制段做 `Y·conj(X)/(|X|²+λ·max|X|²)`，限带后 IFFT；记录 FFT 长度、Tukey 窗、截断点、原点偏移和窗起点。
- 延迟：带限解析互相关包络峰值 + 三点抛物线亚采样插值。误差上界由 Cramér–Rao 项、归一化曲率项、五点四次插值模型差和量化项合成。
- 失配：等距分析频点上的复传递比，参考通道幅度平方加权并分频带汇总；区间由频点噪声传递方差、Tukey/截断泄漏能量和限带响应形状误差合成，并按分析带宽相对解卷积有效带宽归一化。
- 校准：频域目标/实测比值、幅度提升限制、迭代设计、线性相位因果 FIR + Blackman-Harris 窗；将 FIR 加到激励后经同一模拟通道重放、重新反卷积并核对初始误差区间。

完整公式、依赖图和增量语义见 [docs/technical_notes.md](docs/technical_notes.md)。

## 业务错误码

| 错误码 | 触发场景 |
|---|---|
| `E_AUDIO_FORMAT` | 非 48 kHz / 24-bit / 4ch、WAV 损坏、削波或非有限样本 |
| `E_SYNC_MARK` | 同步脉冲缺失、低于阈值或与真值偏差超限 |
| `E_DELAY_BIAS` | 已知延迟超容差，或估计真值偏差超出容差/误差上界 |
| `E_CALIB_MISMATCH` | 校准后残余失配超出预校准误差区间，并给出具体频带 |
| `E_SNAPSHOT_STALE` | 快照 schema/哈希/版本戳不符，缓存条目哈希或键不符，替换快照时前驱不符 |
| `E_BAND_INVALID` | 扫频、解卷积、通道或分析频带非法 |
| `E_FILTER_LEN` | FIR 长度非奇数、超范围或超出 FFT 网格 |

## 测试

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
python -m unittest discover -s tests -v
```

测试覆盖 WAV 字节确定性、同步检测、延迟真值/上界、增益区间随频带收窄、校准残余相容、报告/快照数值一致、GUI 快照只读与 PNG 确定性、增量复用、快照/缓存过期以及固定演示两次运行逐字节一致。

实际验证记录见 [docs/validation_results.md](docs/validation_results.md)。

## 目录

```text
acoustic_workbench/
  models.py            配置与阶段数据模型
  hashing.py           平台稳定规范化哈希
  audio.py             24-bit WAV 编解码与格式校验
  dsp.py               扫频、分数延迟、带限、窗、峰值插值
  simulation.py        确定性多通道模拟与同步真值
  deconvolution.py     正则化反卷积与单通道复算
  analysis.py          延迟、失配、误差传播
  calibration.py       FIR 设计、同通道重测与残余验证
  pipeline.py          增量依赖图与快照发布
  snapshot.py          深层不可变快照、版本戳、持久缓存
  report.py            JSON/Markdown 报告，只读快照
  gui.py               PyQt6 只读快照诊断视图
  demo.py              固定序列与错误码演示
  cli.py / __main__.py 命令行入口
tests/                 单元与集成测试
docs/                  技术说明与实际验证结果
```
