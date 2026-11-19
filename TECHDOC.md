# 语音录音修复工作台 — 技术说明

面向本地确定性合成语音素材的专业修复工作台。全部计算离线、确定性、可重复，
不访问网络、不依赖真实录音。核心目标是格式互操作（16/24bit、单/双声道、
本地有损格式）与工程版本兼容（schema v1 → v2）。

## 1. 环境与 Windows 运行

要求 Windows 10/11、Python 3.11（官方安装包自带 Tkinter）。

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt

REM 一键完整验收演示（生成素材 -> 修复 -> 迁移 -> 缓存 -> 恢复 -> 错误码）
python -m voice_repair_workbench demo --work demo_workspace

REM 仅生成确定性素材与本地有损样本
python -m voice_repair_workbench generate --out demo_workspace\materials

REM 图形界面（无显示/无 tkinter 时自动降级并提示使用 CLI）
python -m voice_repair_workbench.gui
```

依赖：`numpy`、`scipy`、`soundfile`（libsndfile），Tkinter 仅 GUI 需要。
当前 Linux 沙箱无系统 tkinter，已验证 GUI 导入失败时优雅降级，核心功能不受影响。

## 2. 目录结构

```
voice_repair_workbench/
  config.py        全局常量、STFT 契约、引擎版本戳、通道/位深策略
  errors.py        7 个业务错误码与可定位异常 RepairError
  hashing.py       确定性 sha256（字节/numpy 数组/规范 JSON）
  synthesis.py     固定种子合成对白/静音/房间底噪，16bit 单声道 + 24bit 双声道
  lossy_codec.py   本地离线有损编解码器 LRX1（WOLA 频域系数量化，.lrx 容器）
  audio_io.py      多格式解析、显式下混、PCM16/24 写出
  pipeline.py      四阶段确定性管线 denoise→gate→normalize→limit
  project.py       schema v1/v2 识别、参数校验、固定迁移表、幂等迁移
  cache.py         阶段缓存、参数指纹、失效闭包
  persistence.py   原子持久化、快照版本识别、孤儿回收、受控退出
  engine.py        编排引擎（闭包重算/缓存复用/快照/恢复）
  metrics.py       逐段指标与跨格式容差口径
  cli.py / gui.py  一键演示命令行 / Tkinter 界面
tests/             23 个 unittest 用例
```

## 3. 关键设计与业务要求映射

### 3.1 确定性多格式素材（要求 1）
- 固定主种子 `MASTER_SEED`，对白=谐波+基频漂移+音节包络，底噪=确定性 IIR 有色噪声。
- 产出 48kHz 单声道 PCM16 与双声道 PCM24 WAV，外加本地有损 `.lrx`。
- 同一配置两次生成的 WAV/JSON/LRX 逐字节一致（测试
  `test_generation_is_byte_deterministic`、`test_lossy_roundtrip_is_lossy_and_deterministic`）。
- 解析识别采样率、位深、声道数、有损标记（`LRX1`），统一为 float64 与
  显式等权平均下混的分析单声道；绝不静默取首通道，声道误配抛
  `E_CHANNEL_MISMATCH`。

### 3.2 四阶段参数化管线（要求 2）
固定顺序 `denoise → gate → normalize → limit`，全部 float64、确定性公式：
- denoise：Hann 窗 STFT（窗长 2048、帧移 512，固化于 `config.STFT_NFFT/HOP`），
  取前 `noise_seconds` 帧平均能量谱做噪声估计，谱减后保留频谱地板。
- gate：分帧 RMS 软门限，帧增益插值到样本级。
- normalize：按目标峰值与最大增益做确定性增益。
- limit：`tanh` 软限幅到 ceiling。
- STFT 窗长/帧移与阶段定义被固化进 `ENGINE_VERSION` 版本戳，任何改变修复
  输出的实现改动都必须提升该戳，缓存与快照据此判定新旧。
- 同参数同素材两次运行产出相同 int16 PCM（`test_deterministic_pcm_and_full_reuse_second_run`）。

### 3.3 schema v1→v2 迁移（要求 3）
- v1 仅含降噪参数；`migrate_v1_to_v2` 按固定迁移表补 gate/normalize/limit
  恒等默认值、通道顺序 `mono_first`、位深策略 force16，并把 v1 的 `alpha`
  重命名为 v2 的 `over_subtraction`（数值不变）。
- 恒等默认值保证迁移前后修复输出一致；迁移幂等（对 v2 再迁移返回等价工程）。
- 不可迁移字段抛 `E_SCHEMA_VERSION` 并指明字段名（如 `denoise.unknown_legacy_knob`）；
  缺字段抛 `E_FIELD_MISSING`；越界抛 `E_PARAM_RANGE`。

### 3.4 跨格式/跨版本一致性（要求 4）
- 双声道承载与单声道相同的主信号，跨格式差异仅来自位深量化；有损样本为
  14bit 系数量化的不可逆压缩，误差被控制在容差内。
- 容差口径 `metrics.TOLERANCE`：每段 RMS(dB) ≤ 0.5、峰值绝对差 ≤ 0.02、
  RMS 绝对差 ≤ 1e-3、int16 PCM 最大样本差 ≤ 2。
- 通道顺序 `mono_first` 与 `per_channel`：denoise/gate 逐声道处理且与顺序
  无关，中间结果哈希不变；normalize 依赖顺序（先下混归一 vs 逐声道归一
  再下混），其结果与下游 limit 改变（`test_channel_order_only_affects_dependent_stages`）。

### 3.5 阶段缓存与失效闭包（要求 5）
- 缓存键 = 引擎版本戳 + 阶段 + 素材内容哈希 + 该阶段参数子集 + 上游阶段键。
- 线性串联下失效闭包 = 最早变化阶段及其全部下游
  （`compute_invalidation_closure`）。
- 实际重算集合与闭包清单完全一致，闭包外阶段标记 `reused` 并记录来源与
  版本戳；禁止整管线重建（`test_recomputed_set_equals_closure`）。

### 3.6 原子持久化与版本识别（要求 6）
- 所有写入先落 `.tmp` 再 `os.replace` 原子替换；写入未完成即未提交。
- `snapshot.json` 记录引擎版本戳、素材哈希、工程指纹、各阶段键/输出哈希。
- 全缓存命中时强制 `verify_snapshot_freshness`，目录整体替换或工程变更导致
  指纹不符即 `E_SNAPSHOT_STALE`，禁止盲目复用旧缓存。
- 重启逐阶段 `cache.get` 校验输出哈希，不符抛 `E_CACHE_HASH`。

### 3.7 受控退出 vs 强制终止（要求 7）
- 受控退出：写完整阶段清单、`CLOSED.flag` 与 `exit_report.json`。
- 强制终止：重启只依据最近一次已提交 `snapshot.json` 恢复；`.tmp` 与无主
- 缓存作为孤儿回收并记录到 `orphans.log`；无关闭标记时绝不伪造退出报告
  （`test_force_termination_recovery_matches_last_commit`），恢复后 PCM 与
  逐段指标与最后一次提交逐项一致。

### 3.8 业务错误码（要求 8）
| 错误码 | 触发场景 | 定位字段 |
| --- | --- | --- |
| E_FORMAT_UNSUPPORTED | 非法扩展名/位深/采样率/声道、损坏 LRX 魔数 | extension/bit_depth/sample_rate/magic |
| E_SCHEMA_VERSION | 未知 schema 版本、v1 字段不可迁移 | schema_version/denoise.<字段> |
| E_FIELD_MISSING | 工程必填字段缺失 | 具体字段名 |
| E_CHANNEL_MISMATCH | 声明声道数与文件不符 | channels |
| E_CACHE_HASH | 缓存内容哈希不符或引擎版本不一致 | 阶段名/engine_version |
| E_SNAPSHOT_STALE | 快照与当前工程/素材/引擎版本不一致 | 陈旧字段列表 |
| E_PARAM_RANGE | 参数越界或位深策略非法 | 参数全名 |

演示第 9 步一次性触发并打印全部 7 个错误码。

## 4. 验证结果

- 单元/集成测试：`python -m unittest discover -s tests` → 23 通过。
- 端到端演示：`python -m voice_repair_workbench demo --work demo_workspace`
  退出码 0，关键结论：
  - 素材生成逐字节一致；迁移幂等；
  - mono16 vs stereo24、mono16 vs lossy 修复结论均在容差内一致；
  - 同参数重跑 int16 PCM 逐样本一致且第二次全缓存复用（闭包为空）；
  - 仅改 gate 参数时重算集合精确等于 `[gate, normalize, limit]`，denoise 复用；
  - 通道顺序变化时 denoise/gate 哈希不变、normalize/limit 变化；
  - 强制终止后孤儿 `.tmp` 被回收、无伪造退出报告，恢复 PCM 与指标逐项一致；
  - 7 个业务错误码全部可触发并带字段定位。
- 产物：`demo_workspace/materials/`（素材与 .lrx）、各 `ws_*/`（工程快照、
  缓存、repaired.wav、退出报告）、`segment_metrics_report.json`、`demo_results.json`。

## 5. 设计取舍说明
- 双声道合成采用与单声道相同的主信号，是为了满足“同一内容跨通道配置修复
  结论一致”；通道顺序差异通过非对称双声道样本单独验证，二者职责分离。
- 有损编码选择 14bit 系数量化：保持不可逆（输出与输入不相等）同时把误差
  控制在声明容差内，模拟真实有损压缩对修复结论的弱影响。
- 24bit 在 numpy 中无原生整型，采用“量化到 24bit 电平→归一浮点→libsndfile
  写 PCM_24”，量化电平本身完全确定。
