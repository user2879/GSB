# 音频指纹与相似片段检索工作台

本项目在空白工作区中实现了一个完全离线、确定性可复现的曲库查重工作台。它使用程序化合成音频，不读取真实音乐资产，不访问网络；覆盖多格式解码、频谱峰值指纹、SQLite 倒排索引、片段检索与时间对齐、跨格式归并、算法版本迁移、增量导入/删除、崩溃恢复、PySide6 界面与无图形后端便携模式。

## 一键验收

Windows（离线虚拟环境准备好后）：

```powershell
scripts\run_demo.ps1
```

通用 Python 入口：

```bash
python -m audiofp demo --out artifacts/demo --seed 20261119
```

该命令会清理输出目录，生成固定素材，依次执行提取、跨格式一致性、建库、检索、去重、迁移、增量维护与强制终止恢复测试，并执行第二次独立运行做逐字节比对。输出：

- `artifacts/demo/reports/demo_report.json`
- `artifacts/demo/reports/demo_report.md`
- `artifacts/demo/reports/repeatability.json`
- `artifacts/demo/fixtures/`：合成音频、真值标签、损坏样本
- `artifacts/demo/index/`：当前索引、迁移前/后索引、重建索引、崩溃恢复索引

## Windows 离线准备

要求 Python 3.11（64 位）。将预下载 wheel 放入 `wheelhouse/`，然后执行：

```powershell
scripts\setup_windows_offline.ps1 -Wheelhouse wheelhouse
```

脚本只使用 `pip --no-index --find-links`，不会访问网络。`requirements.txt` 固定了 NumPy、SciPy、soundfile、PySide6、imageio-ffmpeg 与 pytest 版本。

有损编码器的选择顺序：

1. `AUDIOFP_FFMPEG` 指定的本地 ffmpeg；
2. `PATH` 中的 `ffmpeg`；
3. `imageio-ffmpeg` wheel 内置二进制。

```powershell
$env:AUDIOFP_FFMPEG = "C:\tools\ffmpeg\bin\ffmpeg.exe"
python -m audiofp doctor --json
```

演示只读取本地文件。`imageio-ffmpeg` 或外部 ffmpeg 仅作为本地编解码工具使用。

## 常用命令

```bash
python -m audiofp doctor --json
python -m audiofp generate --out artifacts/fixtures --seed 20261119
python -m audiofp build --index artifacts/library.db --audio artifacts/fixtures/audio
python -m audiofp list --index artifacts/library.db --json
python -m audiofp query --index artifacts/library.db \
  --query artifacts/fixtures/queries/aurora_repeat_44k_16b_mono.wav --top-k 5 --json
python -m audiofp delete --index artifacts/library.db --group grp_xxx
python -m audiofp migrate --index artifacts/legacy.db --to fp-v2
python -m audiofp ui --index artifacts/library.db
python -m audiofp ui --index artifacts/library.db --portable
python -m audiofp ui --index artifacts/library.db --smoke
```

`--smoke` 使用 `QT_QPA_PLATFORM=offscreen` 构造窗口并退出。无显示后端、PySide6 缺失或 Qt 初始化失败时，`ui` 会明确打印 `[PORTABLE]` 提示并进入终端模式，支持 `list`、`query <path> [top_k]` 和 `quit`。

## 指纹定义

默认配置 `fp-v2`：

| 参数 | 值 |
| --- | --- |
| 分析采样率 | 11025 Hz |
| 窗长 / 帧移 | 2048 / 256 samples |
| 窗函数 | periodic Hann |
| 对数频带 | 90–5000 Hz，48 bands |
| 峰值密度 | 34 peaks/s（上限） |
| 时频邻域 | 时间 ±2 帧、频带 ±2 |
| 目标配对区 | 30–140 帧 |
| 每峰配对上限 | 5 |
| Δt 量化 | 4 帧 |
| 编码空间 | `logband-pair-dt-v2` |

哈希原像可独立复算：

```text
<encoding_space>|<band1:02d>|<band2:02d>|<round(delta_frames/dt_quantum):04d>
```

摘要为 BLAKE2b-64 十六进制串。所有提取参数与编码空间进入 `config_digest` 并随索引保存；同一输入、同一配置重复提取产生逐条相同且有序的指纹。

## 核心行为

- **严格格式校验**：WAV 优先走 RIFF 结构校验；解码器返回帧数与声明帧数不一致、data chunk 未按帧对齐、文件截断、随机有损容器等均返回 `E_AUDIO_FORMAT`。不静默套用默认参数。
- **融合归一化**：所有来源统一混为单声道并使用确定性多相重采样到 11025 Hz；索引单独保存原始容器、codec、采样率、位深、声道、声明/实际帧数和内容 SHA-256。
- **跨格式一致性**：在共有时间轴上按哈希、锚点容差做一对一匹配，输出每对格式的命中率、Dice、频带偏移与疑似原因。低于声明阈值返回 `E_FORMAT_DRIFT`，不会用扩大检索召回掩盖。
- **检索与对齐**：每个候选得分来自 `dominant_time_delta_votes / query_fingerprint_count`，返回对齐起点、查询时长、最佳来源、全部格式来源和投票直方图。
- **移调/变速**：普通查询得分足够时不启用变换搜索；弱匹配时在受控的音带平移与时间尺度假设上重哈希查询侧指纹，并保存实际使用的变换供复核。
- **同曲去重**：按指纹时间差投票决定归并，不使用文件名或时长；同一逻辑曲目只返回一次并带有全部格式来源。若真值标签显示不同曲目被合并，返回 `E_DEDUP_MERGE` 并附指纹投票证据。
- **版本迁移**：只有 STFT/峰值配对参数完全相同、仅编码空间变化时才允许重哈希迁移；迁移幂等。提取参数变化或无兼容规则时返回 `E_INDEX_VERSION`。
- **增量与原子性**：SQLite 事务只写入受影响曲目/倒排项；删除使用外键级联；强制终止后回滚未提交事务，并回收 `.tmp-*` 孤儿和已恢复的回滚日志。
- **确定性输出**：固定种子合成、稳定排序 JSON、固定 Ogg 流序号并重算 Ogg CRC、稳定 SQLite 写入顺序、报告不含时间戳或绝对路径。

## 实际验证摘要

完整验收报告见 [docs/VALIDATION.md](docs/VALIDATION.md) 和 `artifacts/demo/reports/demo_report.md`。当前固定种子 `20261119` 的关键结果：

- 3 首逻辑曲目、9 个格式来源，归并为 3 组且无误合并。
- 同曲格式对命中率约 `0.72–0.97`，全部高于默认阈值 `0.55`。
- 重复片段查询在 44.1k/16bit/mono、48k/24bit/stereo、Vorbis 三种查询格式下排序一致，真值曲目排名第一，对齐起点约 `7.36s`（真值 `7.35s`）。
- 移调 +3 半音与 1.08x 变速变体均通过显式变换假设找回 Aurora。
- v1→v2 迁移后指纹行集合与全量 v2 重建一致，检索签名一致，二次打开修订号/内容哈希不变。
- 增量导入与删除不改变其他曲目的投票得分；删除组剩余指纹行为 0。
- 子进程在第 42 号退出码处强制终止未提交写入，重启后未提交行不可见、内容哈希回到最近提交状态。
- 两次独立完整运行的 27 个产物逐字节一致。

测试：

```bash
python -m pytest -q
```

设计细节与版本规则见 [docs/DESIGN.md](docs/DESIGN.md)。示例素材、索引和报告均由代码在本地生成；仓库不包含真实音频资产。
