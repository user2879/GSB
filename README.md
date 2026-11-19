# codecq — 编解码质量评估可视化与诊断工作台

面向编解码质量评估的领域工作台：本地确定性合成素材 → 多组编码参数编码/解码回放 →
逐帧 PSNR/SSIM 与音频对齐偏差 → 按镜头分段汇总 → 异常注入与检出 → 可复核账本 +
交互式诊断视图。任务意图见 `TASK.md`。

## 环境准备（Windows / Linux 通用）

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -c "import av; print(av.__version__)"   :: PyAV 自带 FFmpeg 运行库，确认可导入
```

Linux 下为 `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`。
不使用 Docker；不需要网络与外部素材（仅安装依赖时需要网络）。

## 一条命令完整演示

```bash
python -m codecq run --config configs/default.yaml --out artifacts/run1
```

依次执行：素材合成（10s 640x360@30fps + 48kHz 双声道 + 真值清单）→ 4 组编码参数
（CRF/GOP/B 帧/4:2:0 与 4:4:4）编码与解码回放 → 逐帧 PSNR/SSIM → 按镜头分段汇总 →
音频互相关对齐 → 参数层异常注入与检出 → 原子提交账本 → 生成 `report.html` 诊断视图。
相同输入两次运行，`ledger/ledger.json` 逐字节一致。

其他入口：

```bash
python -m codecq synth --config configs/default.yaml --out out   # 仅生成素材
python -m codecq view --ledger artifacts/run1/ledger --out report.html  # 仅渲染视图
pytest tests/ -q                                                 # 测试
```

## 输出布局

```
artifacts/run1/
  source/     audio.wav  source.mkv(无损检查副本)  truth.yaml(镜头/色块/音频真值)
  encodes/    <id>.mkv   每组参数一条编码记录
  ledger/     ledger.json  HEAD  snapshots/<version>.json
  report.html 交互式诊断视图（自包含，无外部资源）
```

## 确定性设计（技术说明）

- **素材**：帧画面与音频均为 `(seed, 配置)` 的纯函数（`codecq/synth.py`），逐帧 RNG 用
  `default_rng((seed, frame))`，无挂钟、无全局随机态；视频 300 帧与音频 480000 样本
  时间轴严格对应（1600 样本/帧）。
- **编码**：libx264 `threads=1`，参数快照（CRF/GOP/B 帧/pix_fmt）随记录入账本；音频用
  无损 `pcm_s16le`，避免有损音频 codec 污染对齐评估。
- **账本哈希只覆盖解码内容**（YUV 平面 + PCM + 帧序 + pts），不覆盖容器字节，Matroska
  SegmentUID 等复用层随机性不会渗入账本。
- **指标约定**（随账本声明）：全范围 8bit、BT.601、4:2:0 色度 2x2 box 下采样、通道权重
  Y0.8/U0.1/V0.1、SSIM 11x11 高斯窗 sigma=1.5、PSNR 截顶 99dB；浮点统一保留 6 位小数。
- **分段**：边界直接取真值镜头清单并逐项相等校验，不靠条数相等推断。
- **账本版本**：版本戳 = 内容哈希（`v1-<hash16>`），整体替换时以 HEAD 中版本戳 + 哈希
  判定新旧；写入一律 tmp+fsync+os.replace 原子提交；`*.tmp` 永非有效状态，启动即清理，
  强制终止后恢复只依据已提交账本，篡改内容会导致版本校验失败。

## 异常注入与检出（错误码）

注入只作用于解码流记录与声明参数（输入/参数层），不修改源码；注入配置可序列化并随
账本记录以精确复现（`codecq/inject.py`、`codecq/detect.py`）：

| 注入 | 错误码 | 定位 |
|---|---|---|
| 丢帧 frame_drop | E_FRAME_DROP | 缺失源帧号列表 |
| 时间戳跳变 timestamp_jump | E_TIMESTAMP_JUMP | 首个异常帧号 + 实际/期望 pts |
| 色度采样错标 chroma_mislabel | E_CHROMA_MISMATCH | 编码记录 id，声明 vs 实际 |
| 抽样率谎报 audio_rate_misreport | E_AUDIO_RATE | 编码记录 id，隐含时长对比 |
| 音频偏移 audio_shift | E_AUDIO_DRIFT | 首个越界段号 + 样本偏移 |

单条编码/单帧解码失败被隔离：记录失败原因与重试次数，其余单元照常产出；
失败单元计入 `isolation` 且 `summary.execution_complete=false`，不会被静默计为通过。

## 实际验证结果（本机，Python 3.11 + av 18.1.0 + numpy 2.4.6）

- `pytest tests/ -q`：11 passed（确定性、五类注入检出定位、容差内不误报、分段边界
  与真值一致、失败隔离、版本稳定、tmp 清理、篡改拒绝、视图读取已提交快照）。
- 完整配置连跑两次（`artifacts/run1`、`artifacts/run2`）：`ledger.json` 逐字节一致，
  版本 `v1-7c175f414574a5d5`，sha256
  `13d581dd1a1413d1bdd4bda0d1c7f78f713a34f808e84641670b1d2972ce9c55`；
  `audio.wav`、`truth.yaml` 亦逐字节一致。
- 默认配置检出：E_TIMESTAMP_JUMP(c18, 帧100)、E_FRAME_DROP(c28, 帧12/13/200)、
  E_AUDIO_DRIFT(c28, 段0, 480样本>容差80)、E_CHROMA_MISMATCH(c23)、E_AUDIO_RATE(c35)。
- 干净运行（无注入）零检出；音频偏移 0、增益 1.0、相似度 1.0。
- 单次完整运行约 1.5 分钟（300 帧 × 4 组编码，含 SSIM 与互相关）。
