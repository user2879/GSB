# HLS 并行转码与自适应缓冲预算工作台

这是一个完全本地、确定性的 HLS 直播分发实验工作台。它从固定种子合成 30 秒 1280x720/30fps 视频与 48kHz 双声道 PCM 音频，按 2 秒切片并行转码为 360p/720p 两档 H.264/AAC MPEG-TS，并在统一逻辑时钟上执行预算调度、故障注入、客户端缓冲仿真和联合档位决策。

无需外部素材、账号、网络服务或容器。默认演示会完整运行两遍相同输入并逐文件比较哈希，再改变带宽估计复用已有切片，验证未受影响产物不变。

## Windows 环境

要求 Python 3.10+、可写工作区，以及约 100MB 以上空闲空间。PowerShell 示例：

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pytest -q
python -m hls_workbench demo --output artifacts/demo --clean
```

`python -m pytest` 和 `python -m hls_workbench` 可直接从仓库根目录运行；如需全局命令，再执行下一节的 editable 安装。

也可安装为命令：

```powershell
pip install -e .
hls-workbench demo --output artifacts/demo --clean
```

离线安装时，在联网机器执行 `pip download -r requirements.txt -d wheelhouse`，把 `wheelhouse` 带到目标机后执行：

```powershell
pip install --no-index --find-links wheelhouse -r requirements.txt
```

核心依赖为 `numpy==2.2.6`、`av==17.1.0`（Windows wheel 内含 FFmpeg）和 `pytest==8.3.5`。本次验证在 WSL2 Ubuntu + CPython 3.10.12 + PyAV 17.1.0/FFmpeg 7 上完成；代码只使用标准库、NumPy 与 PyAV，不依赖 POSIX 专有接口。

## 一条命令

```powershell
python -m hls_workbench demo --output artifacts/demo --clean
```

主要输出：

- `artifacts/demo/run_a/source/`：完整 30 秒 720p H.264 源 TS 与 48kHz PCM WAV。
- `artifacts/demo/run_a/slices/`：15 个源切片视频 TS 与对应 PCM WAV。
- `artifacts/demo/run_a/hls/`：master playlist、两档 media playlist 和 30 个 H.264/AAC 分片。
- `artifacts/demo/run_a/reports/report.json`：分步骤状态、哈希链、预算事件、缓冲摘要和决策。
- `artifacts/demo/run_a/reports/transcode_manifest.json`：每片档位、序号、帧数、PTS、SHA-256、降级/拒绝/重试/隔离/重分配与 30 项时间轴校验。
- `artifacts/demo/run_a/reports/buffer_360p.json`、`buffer_720p.json`：100ms 粒度的缓冲水位、下载时刻、卡顿事件与切换建议。
- `artifacts/demo/repeatability.json`：两次完整运行的逐文件哈希比较。
- `artifacts/demo/variant_decision.json`：改变带宽后的决策与产物复用证明。

## 实现结构

| 模块 | 职责 |
| --- | --- |
| `media.py` | 固定种子渐变、平移色块、滚动自绘文字、帧号条和二进制帧码；PCM 合成；源片与完整源生成；逐字节稳定清单 |
| `transcode.py` | 每任务独立 H.264/AAC 编码；显式覆盖绝对视频帧 PTS 与音频采样 PTS；编码后解码校验帧数、编解码器、分辨率和连续性 |
| `scheduler.py` | 按 `(低档位优先, 序号升序)` 装配预算波次；线程/内存/CPU 时间片联合限制；拒绝、降级、固定指数退避、单元隔离与任务重分配；临时文件原子提交 |
| `buffer.py` | 本地伪下载时延表；单下载队列 + 播放消耗 + 缓冲容量背压；卡顿、最低水位和降档建议 |
| `decision.py` | 带宽预算、缓冲安全阈值和并行度联合决策；按输入签名缓存缓冲模型与决策 |
| `hls.py` | 确定性 master/media playlist |
| `pipeline.py` | 端到端编排、HLS 时间轴校验、报告和哈希链 |
| `demo.py` | 双跑确定性验收、变体决策与复用检查 |

### 调度与预算规则

每个任务带有档位内存成本与 CPU 时间片成本。每轮按固定优先级扫描待办任务，在 `max_workers`、`memory_budget_mb`、`cpu_budget_ms` 三个上限内装入一个波次。放不下的任务记录 `E_BUDGET_DENIED` 并延迟到后续波次，绝不丢弃；若单个任务也放不下，则以记录在案的串行兜底执行，保证最终每档序号完整。

每个降级事件记录 `before_parallelism`、`after_parallelism`、内存/CPU 使用量、预算值与原因。调度结果先按任务键归并，再按档位优先级和序号排序写入清单，不依赖线程完成顺序。

示例预算 `workers=4, memory=300MB, cpu=200ms` 下，360p 单任务为 90MB/60ms，720p 为 180MB/110ms，因此稳定观测到并行度从 4 降为 3；720p 波次进一步因内存和 CPU 降为单任务。

### 重试与隔离

故障按任务和当前转码单元注入。失败先在同一单元按 `base × multiplier^(n-1)` 退避重试；重试耗尽后记录 `E_RETRY_EXHAUSTED`，隔离该单元，将任务重分配到下一健康单元。输出先写入任务专属临时文件，成功后才 `os.replace` 到最终路径；任何失败或重复任务都不会覆盖已成功产物。

### 缓冲与联合决策

调度器为任务生成确定性逻辑就绪时钟；伪下载时延表由固定种子写入 `input/pseudo_download_delays.json`。客户端模型只使用这两个时钟，不使用墙钟时间，因而报告可逐字节复现。缓冲容量限制预取水位，低于阈值时输出建议目标档位与触发时的水位、阈值、已下载量和已消耗量依据。

决策规则固定为：先筛除超过安全带宽预算的档位，再按高码率到低码率选择“无卡顿且最低水位不低于阈值”的最高档位；若都不安全则选最低档并保留风险。并行度取线程、内存、CPU 与切片数限制的最小值。

带宽或阈值变化不会触发转码；决策缓存键与产物内容无关。演示把带宽从 2600 kbps 改为 900 kbps，720p 退出候选，360p 缓冲模型直接命中缓存，30 个切片哈希保持不变。

## 确定性设计

- 所有像素、文字、音频和伪下载时延均由固定种子或纯整数/确定性公式产生，文字使用内置 5x7 位图字体。
- 视频与音频原始内容分别计算 SHA-256；清单只含稳定字段、相对路径和规范 JSON。
- 每片编码参数固定为单线程 x264、无 B 帧、确定性模式；编码包时间戳显式覆盖，不采用墙钟或完成顺序。
- 报告不含主机名、绝对路径、时间戳或真实耗时；调度时间轴为逻辑毫秒。
- HLS 分片、PCM、TS、JSON 与 playlist 均以原子写入或固定顺序生成。

AAC 以标准 1024 采样帧编码。每片首个音频包 PTS 精确等于 `序号 × 2000ms`，视频 PTS 逐帧严格连续；由于 AAC 帧粒度，片尾 AAC 包可能包含不超过 1024 个采样的编码器量化尾样，这不改变下一片首包时间戳或视频帧时序。

## 错误码

| 错误码 | 使用位置 |
| --- | --- |
| `E_BUDGET_DENIED` | 波次准入拒绝、预算降级、强制串行兜底 |
| `E_PTS_GAP` | 少片、缺帧、视频/音频起始或包间 PTS 不连续 |
| `E_TASK_DUPLICATE` | 重复任务键、拒绝覆盖已有成功分片 |
| `E_RETRY_EXHAUSTED` | 重试耗尽、单元隔离与重分配链路 |
| `E_BUFFER_UNDERRUN` | 缓冲耗尽、低水位切换建议与风险决策 |
| `E_MEDIA_FORMAT` | 容器、编解码器、分辨率、声道、采样率或 PCM 格式错误 |
| `E_DETERMINISM_MISMATCH` | 双跑或变体复用出现字节差异 |

## 本次实际验证结果

环境：WSL2 Linux x86_64、CPython 3.10.12、NumPy 2.2.6、PyAV 17.1.0（FFmpeg 7）、pytest 8.3.5。

自动化测试：

```text
11 passed in 8.28s
```

完整 30 秒演示（双跑 + 变体）耗时约 64.1 秒，输出约 61MB：

- 合成：900 帧、1,440,000 个音频采样、15 个 2 秒源切片。
- 转码：30 个 H.264/AAC 分片，30/30 项时间轴校验通过；每片视频 60 帧，首视频 PTS 与首音频 PTS 均精确对齐 0/2/.../28 秒。
- 预算：请求并行度 4，实际最大并行度 3；27 条 `E_BUDGET_DENIED` 准入记录，19 条降级记录。
- 故障：`720p:00003` 在 `unit-02` 注入 3 次失败，退避 15ms、30ms 后记录 `E_RETRY_EXHAUSTED`，隔离 `unit-02` 并重分配到 `unit-03`；因单元隔离共发生 3 次任务重分配，所有任务最终成功。
- 缓冲：720p 最低水位 0ms、2 次卡顿、2 条降档建议；360p 最低水位 2700ms、0 次卡顿，阈值 2600ms。
- 决策：带宽 2600 kbps 时选择 360p，并行度 3；带宽改为 900 kbps 后仍选择 360p，360p 模型缓存命中，30 个转码产物全部复用且哈希不变。
- 双跑：73 个文件（含源、PCM、切片、TS、playlist、清单和报告）逐文件 SHA-256 完全一致；两跑报告 SHA-256 均为 `777629cb5d0e608494bd4a1bb03f234391106f8b10d12a160baf09cff06dbc91`，整树摘要均为 `6958407b756a5bfc1bf673c05605a072833889baa4db9efe98b499f233e22ecc`。

关键哈希链：

```text
synthesis  ffaf59f750d5e6523d25859a1610076b1450a996560bba87f0171dd4adb694e8
transcode  93987c50ff6aeb9e431c2422d8b178fe8162c0037fa3601f1d03a6bbca9f4e55
buffer     81a903f863f3a432b88aa28c9a5bc5ae7bd5effbc1e5bacfb1ba5cfeac203f32
decision   87fc33c0d034c3319576c66c90f7ca65758802d52d27d01f704377fa88af1b93
report     777629cb5d0e608494bd4a1bb03f234391106f8b10d12a160baf09cff06dbc91
```

完整验收产物保留在 `artifacts/demo/`；可在浏览器或播放器中打开 `artifacts/demo/run_a/hls/master.m3u8`。
