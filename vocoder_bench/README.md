# 声码器波形合成与相位重建诊断工作台

把**已给定的声学特征帧**（基频、谐波幅度、能量增益）还原为 16kHz/16bit 单声道
时域波形，并对相位重建、谐波截断、定点能量缩放引入的**波形级**偏差做诊断。
仅依赖 Python 标准库（`wave`、`struct`、`json`、`threading`、`pathlib`、`signal`
等）与 NumPy 1.26.4，不使用任何外部声码器或音频库。

## 目录结构

```
vocoder_bench/
├── vocoder/
│   ├── hashing.py            # 确定性 FNV1a/splitmix 指纹，无全局随机状态
│   ├── yaml_lite.py          # 扁平 YAML 解析/序列化（无 PyYAML 依赖）
│   ├── config.py             # 配置模型、容差、编辑命令应用
│   ├── wav_io.py             # RIFF/WAVE 校验、PCM16 读写与确定性舍入
│   ├── fixedpoint.py         # 定点整数相位累积 + float64 对照路径
│   ├── dsp.py                # 窗函数、谐波叠加、重叠相加、对齐与波形指标
│   ├── loader.py             # 输入扫描与 E_AUDIO_FORMAT/E_FEATURE_INVALID 校验
│   ├── ledger.py             # 内容指纹缓存键、复用账本（线程安全）
│   ├── snapshot.py           # 临时目录写盘 + 整体替换 + 恢复/版本守卫
│   ├── engine.py             # 相位→谐波→重叠相加→指标 四阶段流水线
│   ├── runner.py             # 线程池并发、稳定排序键、快照提交、报告
│   ├── session.py            # 交互延迟预算、E_INTERACT_BUDGET_PENDING、信号退出
│   ├── generate_samples.py   # 固定种子确定性样本生成器
│   └── cli.py                # generate / run / stream 三个子命令
├── tests/                    # unittest 测试（23 个）
└── samples/                  # 生成的确定性样本（运行后出现）
```

## 快速开始

```bash
pip install numpy==1.26.4
cd vocoder_bench

# 1) 生成确定性样本（参考 WAV、特征 JSONL、配置、命令流、异常注入）
python3 -m vocoder.cli generate samples

# 2) 全量合成 + 波形诊断（输出 work_full/report_full.json 与波形 WAV）
python3 -m vocoder.cli run \
  --input samples --config samples/vocoder_config.yaml --workdir work_full

# 3) 回放编辑命令流（交互预算、增量复用、优雅退出）
python3 -m vocoder.cli stream \
  --input samples --config samples/vocoder_config.yaml \
  --commands samples/edit_commands.json --workdir work_stream

# 4) 测试
python3 -m unittest discover -s tests -v
```

## 关键设计

### 定点相位重建（业务要求 2）
- 相位表示为无符号整数，环长 `2^phase_bits` 线性映射到 `[0, 2π)`。
- 起始相位由 `(utterance ID, 帧内容指纹, 谐波序号)` 经 splitmix64 派生，
  **不读取任何运行时全局随机状态**。固定点与 float64 对照路径共享同一
  48 位虚拟网格种子，因此二者差异只来自相位位宽/量化档位。
- 累积与回绕全部是整数运算（`& (2^bits - 1)`），不使用浮点取模；
  对绝对目标相位按 `phase_quant_step` 网格取整，量化误差每帧独立、不累积。
- 降低 `phase_bits` 或增大 `phase_quant_step` 会真实改变相位矩阵和最终波形。

### 谐波叠加与重叠相加（业务要求 3、4）
- 按谐波序号升序逐个累加正弦分量，float64 确定顺序求和。
- 谐波数下调时精确舍弃最高次谐波，并在 `truncation.dropped_harmonics`
  记录被截断的谐波序号区间与受影响帧。
- 支持 `hann / hamming / rect` 窗与 `cola / none` 归一化，饱和输出 int16。
- 能量缩放支持 `half_up / half_even / trunc` 三种确定性舍入。

### 对齐与波形级指标（业务要求 4、5）
- 在 `align_search_ms` 窗内用归一化互相关找最佳偏移；并列极值取绝对值更小的偏移。
- 每条 utterance 输出幅度谱距离、波形 RMSE、归一化互相关三类指标，
  以及逐帧波形摘要（峰值、RMS、采样数）。
- float64 对照路径重算波形，给出相位位宽引入的波形 RMSE 偏差；
  当幅度谱达标而波形超差时，标注 `phase_dominant=true`，并记录
  超差谐波序号与帧区间，同时给出 `first_dominant_stage` 与按误差排序的阶段。

### 指纹缓存与精确失效（业务要求 6）
- 缓存键由 `(阶段, utterance ID, 特征帧指纹, 配置指纹, 相位档位指纹, 帧号)`
  经 FNV1a 混合派生；修改基频倍率/谐波上限/相位位宽/窗型/舍入模式会改变
  对应阶段的键，仅受影响阶段重算，未命中差异之外的产物直接复用。
- 账本带锁，保证多线程下条目与命中证据不丢失；输出逐阶段复用/重算计数与命中证据。

### 并发一致性与交互预算（业务要求 7）
- 线程池并发执行 utterance 流水线，帧任务携带
  `(utterance ID, frame_index)` 稳定排序键，最终结果按键序装配，
  与单线程顺序执行结果一致（测试在 workers=1/2/4 下比对）。
- 每条编辑命令在 `interactive_budget_ms` 内优先返回最近完成摘要；
  预算内无法完成则返回 `E_INTERACT_BUDGET_PENDING` 中间态而非阻塞。

### 阶段快照与恢复（业务要求 8）
- `load / phase / harmonic / overlap_add / metric` 各阶段产物先写入
  独立临时目录，再 `os.replace` 整体替换；全局清单记录版本号、配置哈希、
  相位档位、已完成帧与命令游标，并拒绝版本回退。
- 强制终止后只识别已完整替换且含 `stage_manifest.json` 的目录，
  临时残留目录不参与恢复。
- 收到 SIGTERM/SIGINT 时完成当前帧阶段原子提交后以退出码 0 退出。

## 输入样本

`generate` 产出：3 条正常 utterance（rising/falling/vibrato 基频轮廓）、
2 条异常注入样本（RIFF 容器损坏 → `E_AUDIO_FORMAT`；帧序号重复 →
`E_FEATURE_INVALID`）、`vocoder_config.yaml`、`edit_commands.json`、
`anomaly_injection.json` 与 `sample_manifest.json`。参考 WAV 由 float64
对照路径在相同正弦/重叠相加模型下渲染，作为波形真值。

## 实测结论（本机）

- 24 位默认配置：float64 对照波形与参考 WAV **逐采样完全相同**，
  固定点路径与 float64 仅差 1 个 int16 量化级，NCC=1.0、对齐偏移 0。
- 粗量化（phase_bits=5, quant_step=4）：幅度谱仍达标而波形 RMSE 超差，
  正确报告 `phase_dominant=true`，并给出超差谐波与帧区间。
- 命令流回到基线参数后，增量运行 0 重算、132 个缓存键全部复用，
  且与冷启动全量合成波形逐采样一致。
- 23 个 unittest 全部通过。
