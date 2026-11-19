# 实际验证结果

验证日期：2026-11-19（Asia/Shanghai）

运行环境：

- Python 3.11.0（64 位）
- numpy 2.4.6
- scipy 1.17.1
- PyQt6 6.12.1 / Qt 6.12.0
- Linux 本地确定性样本；GUI 使用 Qt `offscreen` 平台渲染。代码不使用声卡、容器或外部音频资源。

## 单元与集成测试

命令：

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python -m unittest discover -s tests -v
```

结果：

```text
Ran 11 tests in 46.656s

OK
```

覆盖项：

- 48 kHz / 24-bit / 4ch WAV 格式与两次生成逐字节一致。
- 同步脉冲真值/检测。
- 亚采样延迟估计、真值偏差和误差上界。
- 增益真值落在合成区间内。
- 分析频带从 200–12000 Hz 收窄到 300–8000 Hz 后区间宽度严格下降。
- 校准残余全部位于预校准区间内。
- 报告与快照逐项数值一致；快照深层不可变。
- GUI 只读取快照，同快照两次 PNG 渲染逐字节一致。
- 增量复用：正则化/频带/FIR 长度分别只失效对应下游阶段。
- 快照内容篡改和缓存哈希篡改均触发 `E_SNAPSHOT_STALE`。
- 7 个业务错误码均触发正确异常。
- 固定演示独立重跑两次，全部输出文件逐字节一致。
- `measurement/` 落盘素材与最终快照绑定，读取 WAV/源时间线独立复算的 IR 哈希与快照一致。

## 误差区间收缩

同一测量与 IR 下，仅改变分析频带时的总区间宽度（四通道范围）：

| 分析频带 | 区间宽度范围 (dB) |
|---|---:|
| 200–12000 Hz | 0.1577–0.1581 |
| 300–8000 Hz | 0.1267–0.1271 |
| 500–4000 Hz | 0.0851–0.0855 |
| 1000–3000 Hz | 0.0642–0.0646 |

区间由噪声传递方差、窗/截断泄漏、限带响应项和 span 因子生成，不是固定值。

## 固定演示

命令：

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python -m acoustic_workbench demo \
  --output output/demo --verify-determinism
```

确定性比较结果：

```json
{
  "compared_files": 62,
  "content_mismatches": [],
  "identical": true,
  "only_first": [],
  "only_second": []
}
```

固定序列第 5 步（最终配置）摘要：

```text
version_stamp: acoustic-workbench.snapshot/1.0#r5:620b7b0fdd599f54
final_snapshot_hash: 620b7b0fdd599f54b90816934de2d9b158a50e4610afa3872036be58b2b9174d
report_hash: 95a7ca8fc9a1fc0c4f64ddd670612d723dc0fafadb9b53ea2541b6675d28915d
acceptance:
  sync_ok = true
  delay_ok = true
  calibration_ok = true
  mismatch_truth_within_interval = true
```

最终配置：扫频 1.2 s、2 次重复、正则化 2e-6、分析频带 300–8000 Hz、FIR 513 taps。延迟真值/估计/误差上界分别为 CH1 `0.350000 / 0.342001 / 0.074420`、CH2 `-0.220000 / -0.212326 / 0.073673`、CH3 `0.710000 / 0.718528 / 0.075402` 采样。四通道真值均落在合成失配区间内，校准后全部通道残余与区间相容。

错误链：

```text
E_AUDIO_FORMAT      audio-format
E_SYNC_MARK         sync-mark-missing
E_DELAY_BIAS        delay-bias / CH1
E_CALIB_MISMATCH    calibration-mismatch / CH1 + concrete frequency bands
E_SNAPSHOT_STALE    snapshot-stale
E_BAND_INVALID      band-invalid
E_FILTER_LEN        filter-length
```

## 复算

命令：

```bash
.venv/bin/python run_demo.py recompute output/demo/state/current_snapshot.json \
  --wav output/demo/measurement/measurement.wav \
  --source output/demo/measurement/source_reference.npy \
  --channel 2 --output-dir output/recompute
```

结果：WAV 为 `48000 Hz / 24 bit / 4 ch / 131040 frames`；快照与独立复算的通道 2 `ir_hash` 均为 `dd9f9a2d34e0fdbcb1f9693f4dcff57f5034dc80894f769ec1ae10fea0b5bfc0`。噪声标准差由落盘 WAV 的静音段按同一算法重估，避免量化后的展示值引入复算差异。

## GUI 渲染

`diagnostic_view.png` 由最终不可变快照渲染。频域视图显示幅度/相位、误差区间与校准残余；时域视图显示 IR、峰值和同步标记真值。无显示环境下两次渲染字节一致。
