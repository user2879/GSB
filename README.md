# 广播播控响度与动态范围处理工作台

面向广播播控的 5.1 环绕声响度/动态范围工作台。输入为本地确定性合成的
48kHz / 24bit / 6 声道 PCM 素材（对白、音乐、瞬态冲击段拼接，带分段真值
标签），提供实时监听与离线分析两种模式，并保证两条路径在声明容差内
结论一致、压缩器跨块平滑无跳变、所有界面数值来自不可变快照。

## 运行环境

- Windows 10/11（代码为纯 Python，亦可在 Linux/macOS 复现）
- Python 3.11
- 固定依赖（见 `requirements.txt`）：
  - `numpy==1.26.4`
  - `scipy==1.13.1`
  - `PySide6==6.7.2`

### Windows 复现步骤（PowerShell）

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

若执行策略阻止激活脚本，可改用 `.\.venv\Scripts\python.exe` 直接调用。

## 音频输出设备声明

本工作台默认以**离屏/脚本化缓冲驱动**演示，不打开真实音频设备、不依赖
真实录音，素材全部由本地固定种子合成。需要在无显示环境运行 GUI 时显式
声明离屏平台插件：

```powershell
$env:QT_QPA_PLATFORM="offscreen"
python scripts/run_gui.py
```

有桌面环境时直接 `python scripts/run_gui.py` 即可显示窗口。

## 快速开始

```powershell
# 1. 生成确定性素材并独立复核标签/波形对齐
python scripts/generate_material.py --output-dir artifacts

# 2. 固定序列演示：流式测量、离线分析、压缩、块长/模式切换
python scripts/run_demo.py --output-dir artifacts

# 3. 运行测试套件（21 项）
python -m unittest discover -s tests -v

# 4. 启动 GUI
python scripts/run_gui.py
```

## 演示固定序列

`loudness_workbench/demo.py` 按固定顺序执行：

1. 固定种子渲染 5.1 WAV 与标签清单；
2. 以 50ms 块长流式处理前半段；
3. 块长切换为 100ms 继续流式处理后半段（状态不重置）；
4. 实时→离线模式切换，冻结带版本号与时间范围的快照；
5. 调整第 2 段目标响度并写入新快照（不清空累计状态）；
6. 基于快照重新测量（继承而非清空状态）；
7. 独立离线整段扫描（双级门控、LRA、4x 真峰值）；
8. 离线压缩并对照流式/离线增益包络哈希；
9. 尝试陈旧快照回退（预期被 `E_SNAPSHOT_STALE` 拒绝）；
10. 输出容差判定、降级统计与连续性证据。

## 产物（`artifacts/`）

- `programme_5p1_48k_24bit.wav`：48kHz/24bit/6 声道素材；
- `labels.json`：分段起点/终点、目标响度、目标 LRA、交叉淡化参数；
- `alignment_check.json`：独立标签/波形对齐复核结果；
- `verification_report.json`：完整可复算测量报告（含口径参数、被门控
  切掉的块数、短时响度序列、真峰值、增益包络哈希、容差判定、降级统计）。

## 关键设计

- 实时与离线共用同一个 `StreamingLoudnessMeter`，离线仅一次性喂入整段，
  从根上消除两套算法漂移；K 加权 FIR/IIR 与积分窗状态跨块延续。
- 压缩器控制平面固定 10ms 一跳，与音频块长解耦；dB 域一阶起音/释放
  平滑，状态可序列化，任意块长切分下包络逐帧一致。
- 快照为不可变对象并带单调版本号，旧版本回退一律拒绝；界面只读取
  已接受快照。
- 预算超限时仅跳帧刷新并计数，不改变任何测量状态或最终报告。

详见 `docs/design.md`。

## 业务错误码

| 错误码 | 含义 |
| --- | --- |
| `E_AUDIO_FORMAT` | 非法采样率/声道数/位深或信号数据 |
| `E_BLOCK_BOUNDS` | 块长越界或未对齐 10ms 网格 |
| `E_COMP_STATE` | 压缩器平滑状态缺失或与参数不兼容 |
| `E_SNAPSHOT_STALE` | 陈旧快照版本回退被拒绝 |
| `E_BUDGET_EXCEEDED` | 交互预算持续超限 |
| `E_REPORT_MISMATCH` | 实时与离线结论超出声明容差 |

## 实测验证结果（本机记录）

- 测试套件：21 项全部通过；
- 两次演示运行的 WAV / 标签 / 报告逐字节一致；
- 流式 vs 离线整段响度差：`0.0 LU`（容差 0.1 LU）；
- 流式 vs 离线真峰值差：`0.0 dBTP`（容差 0.05 dBTP）；
- 增益包络最大差：`0.0 dB`（容差 1e-9 dB），269 个块边界增益差均为 0；
- 预算降级 2 帧（脚本化注入 55ms 耗时），测量状态与报告不受影响；
- 陈旧快照回退被 `E_SNAPSHOT_STALE` 显式拒绝。

## 项目结构

```
loudness_workbench/
  audio_io.py  bs1770.py  budget.py  compressor.py  config.py
  demo.py  engine.py  errors.py  offline.py  snapshot.py  synth.py
  ui/main_window.py
scripts/  run_demo.py  run_gui.py  generate_material.py
tests/test_workbench.py
docs/design.md
requirements.txt
```
