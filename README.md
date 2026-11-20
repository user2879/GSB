# 编解码质量评估工作台

面向本地编解码质量评估的跨格式互操作与版本兼容诊断工作台。项目完全离线运行，不依赖 Docker、ffmpeg、真实账号或外部音视频文件。合成素材、三种承载样本、指标账本和恢复报告均由代码生成。

## 能力概览

- 固定种子生成 10 秒、640x360、30fps 的 Y4M 视频，包含平移色块、动态渐变、滚动像素文字和 4 个硬切镜头。
- 生成 48kHz 双声道 PCM WAV、浮点参考波形和逐帧真值清单：镜头边界、色块位置、文字滚动位置、音频参考采样、帧数/镜头数/样本总数。
- 同一内容生成三种承载：评估工程 JSON v2、紧凑二进制指标帧、缺少镜头分段和指标口径的旧版 JSON。
- 二进制承载校验字节序标记、索引宽度、指标编码版本、字段位图、记录长度与 CRC；解析失败只隔离当前承载。
- 解码 YUV420p，计算逐帧 PSNR/SSIM，并按真值镜头边界分段汇总。
- 音频按固定互相关计算每段样本偏移、增益比与波形相似度，并输出统一时基下的音视频联合偏差。
- 对三承载做逐字段等价性核对，记录首个不一致字段、迁移假设与语义不确定性。
- 依据内容、编码快照、指标口径、镜头映射和逐帧故障签名计算精确失效闭包，只重算闭包内帧和段。
- 注入配置可序列化并写入账本；支持丢帧、时间戳跳变、指标版本错标、音频采样率谎报和非法字段位图等故障。
- 账本分片原子写入，清单最后提交；正常退出写关闭标记；强制终止后按最新已提交快照恢复并回收孤儿产物。

## Windows 准备

要求 Python 3.10 或以上。PowerShell 示例：

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy Pillow
python -m pip install -e .
```

也可以不安装项目，直接在仓库根目录使用 `python -m evalworkbench.cli ...`。

工作区不要放在只有符号链接或不支持同目录原子改名的位置。账本使用同目录临时文件、`fsync` 和 `os.replace` 落盘。完整素材约占 210 MB 临时空间。

## 快速开始

```powershell
python -m evalworkbench.cli demo --workspace C:\work\evalwb
```

该命令依次执行生成、构建承载和基线评估。也可分步执行：

```powershell
python -m evalworkbench.cli generate --workspace C:\work\evalwb --seed 20261120
python -m evalworkbench.cli carriers --workspace C:\work\evalwb
python -m evalworkbench.cli evaluate --workspace C:\work\evalwb
python -m evalworkbench.cli status   --workspace C:\work\evalwb
```

解码回放检查可导出单帧 RGB PPM：

```powershell
python -m evalworkbench.cli preview --workspace C:\work\evalwb --frame 120 --output frame120.ppm
```

## 故障注入与重算

```powershell
python -m evalworkbench.cli evaluate  --workspace C:\work\evalwb --inject-file injections\audio_drift.json
python -m evalworkbench.cli evaluate  --workspace C:\work\evalwb --inject-file injections\binary_field_bitmap.json
python -m evalworkbench.cli recompute --workspace C:\work\evalwb --mutation-file mutations\shots_merge_1_2.json
python -m evalworkbench.cli recover   --workspace C:\work\evalwb
```

内置注入示例位于 `injections/`，闭包变更示例位于 `mutations/`。所有注入只通过输入文件或参数生效，运行后完整配置写入 `ledger/snapshots/<run>/injections.json`。

## 工作区布局

```text
workspace/
  assets/
    source.y4m                  # 无损参考 YUV420p
    encoded.y4m                 # 确定性有损播放 YUV420p
    audio.wav                   # 48kHz stereo PCM16
    audio_reference.f32le       # interleaved float32 reference
    truth.json                  # 逐帧真值、镜头、色块、音频采样、哈希
  carriers/
    project_v2.json             # 自有评估工程 JSON
    metrics_v2.emf              # 二进制指标帧 v2
    project_legacy_v1.json      # 缺 shots/metric_policy 的旧版 JSON
    metrics_v1.emf              # 版本迁移测试用指标帧 v1
    manifest.json               # 承载内容哈希
  ledger/
    snapshots/<run-id>/         # 原子分片 + manifest 提交点
    CURRENT.json                # 最新提交指针
    closed.json                 # 仅正常完整退出时存在
    recovery.json               # 最近一次恢复报告
```

## 业务码

| 业务码 | 含义 |
| --- | --- |
| `E_SCHEMA_VERSION` | schema 或版本声明不支持/不一致 |
| `E_ENDIAN_MISMATCH` | 二进制标记与声明字节序不一致 |
| `E_INDEX_OVERFLOW` | 索引宽度或索引声明无法表示帧/段 |
| `E_METRIC_VERSION` | 指标编码版本不支持或与头部不一致 |
| `E_FIELD_BITMAP` | 字段位图含未知位、缺必需位或与声明冲突 |
| `E_AUDIO_DRIFT` | 音频偏移超过声明容差，记录首个越界段 |
| `E_EQUIV_MISMATCH` | 跨承载或承载声明与解码实际值不等价 |
| `E_CLOSURE_MISMATCH` | 实际重算集合与失效闭包不一致 |
| `E_SNAPSHOT_STALE` | 基于过期版本戳提交账本 |
| `E_AUDIO_SAMPLE_RATE` | 声明采样率与 WAV 头不一致 |
| `E_DROPPED_FRAME` / `E_TIMESTAMP_JUMP` | 注入故障导致的单帧失效 |
| `E_DECODE` / `E_CARRIER_PARSE` | 单元级解码或承载解析失败，已隔离 |

## 测试与验收

```powershell
python -m unittest -v tests.test_workbench
python scripts\validate.py --workspace C:\work\evalwb-validation --report artifacts\validation_report.json
```

实际验证结果已写入 `artifacts/validation_report.json`：29 项检查全部通过，耗时约 45 秒。关键结果：

- 内容哈希：`6c2b2bd21c6a04103cf4982d7c418be58654ec27ccdb45b7ffe061aa2a2d94c4`
- 指标账本摘要：`cac388b6a43161c21402427051fb93796ce805ca4f44ad2afb0307ab1a418695`
- 等价性报告摘要：`26b252542a7b83156d99fd1bf47d94ecde872c21b7d93211dca808683390ffe2`
- 两次生成的所有素材与真值文件 SHA-256 完全一致。
- 重复解码后逐帧、分段、音频与等价性分片字节一致，闭包为零帧零段。
- 编码快照或指标口径变化时闭包精确覆盖全部 300 帧与 4 个段。
- 镜头合并后仅重算第 180–299 帧及段 1/2/3，闭包外帧内容哈希保持不变。
- 注入 `96` 样本音频偏移时检出 `E_AUDIO_DRIFT`，首段偏移 `2.0 ms`，增益比约 `1.05`。
- 强制终止后恢复最近已提交版本，未提交目录及临时文件被登记为孤儿并回收，未伪造关闭标记。

更完整的格式、指标口径、依赖闭包与恢复协议见 [TECHNICAL.md](TECHNICAL.md)。
