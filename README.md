# 音频指纹相似片段检索工作台

这是一个完全本地、确定性、可回退且可崩溃恢复的音频查重工作台。素材由固定种子程序化生成，指纹、倒排索引、检索结果和演示报告均从同一版本化曲库状态派生。实现不依赖网络、GPU、Docker 或外部音频资产。

## 快速开始

要求 Python 3.11/3.12、NumPy、SciPy。

```bash
python3 -m pip install --user -r requirements.txt
python3 -m pytest -q
python3 -m fingerprint_workbench demo --root artifacts/demo
```

演示命令会依次完成：

1. 生成 7 首 44.1 kHz/单声道/16 bit WAV、5 个查询片段及真值清单；
2. 逐首导入前 6 首并输出每个版本的索引哈希；
3. 执行精确重复、移调、轻微变速与同曲重复检索；
4. 修改指纹窗长、帧移、峰值密度与阈值，再回退到旧版本并逐字段比较候选；
5. 在子进程写入分片期间发送 `SIGKILL`，重启后只恢复 `CURRENT.json` 指向的已提交版本；
6. 恢复后继续导入，并与未中断对照库比较状态摘要、索引哈希、版本链和候选结果。

也可以使用模块入口执行单项操作：

```bash
python3 -m fingerprint_workbench generate --out artifacts/materials
python3 -m fingerprint_workbench import --root artifacts/workbench \
  --track-id track_01 --audio artifacts/materials/tracks/track_01.wav
python3 -m fingerprint_workbench search --root artifacts/workbench \
  --query artifacts/materials/queries/q_01_exact_cross.wav
python3 -m fingerprint_workbench config --root artifacts/workbench \
  --fingerprint-config-json '{"window_samples":1024,"hop_samples":256,"peak_density":18,"band_edges_hz":[90,300,900,2200,5200,10000]}' \
  --threshold 0.15
python3 -m fingerprint_workbench undo --root artifacts/workbench
python3 -m fingerprint_workbench revert --root artifacts/workbench --version 6
python3 -m fingerprint_workbench verify-version --root artifacts/workbench --version 6
python3 -m fingerprint_workbench versions --root artifacts/workbench
```

安装为脚本入口（可选）：

```bash
python3 -m pip install --user -e .
fpw demo --root artifacts/demo
```

## 架构

```text
CURRENT.json                原子提交点：version + state_digest + manifest_hash
  └─ ops/vNNNNNN.json       带 parent_version 的不可变操作与完整状态
       └─ indexes/manifests/<hash>.json
            └─ indexes/partitions/<hash>.json    内容寻址分片
audio/<audio_sha>.wav       内容寻址音频
fingerprints/<hash>.json    音频哈希 + 指纹配置 + 指纹记录的内容寻址缓存
```

- `LibraryState` 由成员、成员音频哈希、指纹配置、相似度阈值、逐曲指纹哈希和索引清单哈希组成。
- `state_digest` 对除版本号和清单哈希外的全部业务状态做规范 JSON SHA-256。成员、配置、阈值或指纹任一变化都会改变摘要。
- 索引清单的 `content_hash` 同时是文件名。清单包含每个分片的版本戳、内容哈希、大小和倒排项计数。
- 回退不会扫描或重建整库：回退产生新版本和当前父链，但直接复用目标版本的不可变清单；`shards_rewritten = 0`，索引哈希与目标版本完全相同。
- `verify-version` 是只读审计路径：从历史状态中的指纹重新计算分片与清单哈希，确认历史版本可重建，但不用于冒充状态回退。

### 原子提交协议

一次状态变更按以下顺序执行：

1. 只写受影响的哈希分片：同目录临时文件、`fsync`、原子 `os.replace`；已有内容寻址文件不复写；
2. 写不可变索引清单；
3. 写不可变操作记录 `ops/vNNNNNN.json`；
4. 原子替换 `CURRENT.json`，这是唯一提交点；
5. 正常退出时写 `STOP.json` 关闭标记。

强制终止若发生在第 4 步之前，`CURRENT.json` 仍指向上一个稳定版本。重启时校验 `CURRENT -> op -> manifest -> shards` 全链，删除 `.tmp` 半写文件和未被任一已提交版本引用的分片、清单、指纹及音频，并生成无时间戳的确定性恢复报告。默认会把失败提交留下的完整但未引用的分片也计入孤儿回收。

### 指纹与检索

`FingerprintConfig` 包括窗长、帧移、峰值密度、频带边界、邻域半径、扇出与目标时间区。指纹文档固定记录配置字典、配置摘要、音频内容哈希与规范排序后的记录；任一配置字段变化都会改变配置摘要和指纹内容哈希。

STFT 使用 SciPy。峰值在频带内做时频邻域极大值筛选，再按每秒峰值密度确定性地限制数量。每个锚点与后续目标峰配对，产生四类记录：

- `exact`：绝对频点 + 精确帧差，用于字节/频谱级重复；
- `robust`：半音音程 + 对数时间桶，容忍移调与轻微变速；
- `context`：相邻双目标的相对音程和双时间桶，降低随机碰撞；
- `sequence`：连续三目标音程序列，提供高权重低碰撞证据。

检索按指纹哈希空间分片扫描。对每条命中计算 `track_anchor_frame - query_anchor_frame`，在平滑时间差直方图上取得票峰。得分定义为：

```text
score = histogram_peak_weight / query_weight
weight = exact*1.25 + robust*0.25 + context*3.0 + sequence*5.0
```

候选按得分、票数、对齐帧和曲目 ID 稳定排序。报告给出对齐起点、查询时长、票数分解与完整得分构成。

## 状态演进与恢复

- 每次导入、删除、配置/阈值修改、undo、revert 都创建 `version + 1` 且带 `parent_version` 的新操作。
- 导入只合并新曲目覆盖到的分片；删除只过滤被删曲目出现的分片；仅阈值变化不重写任何分片，只产生新清单。
- 回退/撤销可指向任意历史版本。新操作的状态内容与目标版本相同，因此 `state_digest`、`manifest_hash` 和检索候选精确复原。
- 检索结果不使用进程内旧缓存。指纹缓存按 `audio_sha256 + config` 内容寻址，可安全复用；恢复时只保留已提交版本引用的内容。

## 资源预算与降级

`BudgetConfig` 包括查询线程数、分片缓存 MB、单查询工作预算、索引维护分片 MB 和候选上限。为保证降级决策可复现，`query_time_budget_ms` 映射为固定的候选命中检查数量预算（`1 ms = 1200` 次检查），不使用不可复现的实际墙钟时间作分支条件。

预算不足时严格按以下优先级降级并记录原因：

1. `serial_query` 或 `reduce_query_threads`；
2. `evict_partition_cache` / `bypass_partition_cache`；
3. `skip_low_score_candidates`，记录截断前后数量及被跳过的候选和得分构成。

以本次演示的压低预算运行 `q_03_time_fast`，三种动作全部触发，真值 `track_05` 仍在第 2 名命中；恢复后 `q_05_pitch_down_cross` 也在降级预算下命中 `track_07`。

## 业务错误码

| 错误码 | 触发场景 |
|---|---|
| `E_AUDIO_FORMAT` | WAV 容器损坏、非 44.1 kHz、非单声道、非 16 bit PCM、长度不足 |
| `E_INDEX_VERSION` | 状态、操作、清单或分片 schema 版本不受支持 |
| `E_PARTITION_HASH` | 分片/清单/指纹内容哈希不符、分片归属错误、JSON 损坏 |
| `E_CONFIG_RANGE` | 指纹参数、频带边界或相似度阈值越界 |
| `E_SNAPSHOT_STALE` | `CURRENT` 缺失、操作链断裂、引用文件缺失或状态摘要不符 |
| `E_BUDGET_DENIED` | 线程、内存、时间或候选预算低于硬下限/分片超过维护预算 |
| `E_QUERY_DRIFT` | 查询配置与状态不一致，或回退后候选出现分歧；异常详情包含首个分歧位置与双方得分构成 |

另提供 `E_STATE_CONFLICT`、`E_NOT_FOUND`、`E_LOCKED` 覆盖重复导入、缺失曲目和并发打开。

## 验证结果

本机 Python 3.11.0rc1、NumPy 2.4.6、SciPy 1.17.1 下：

```text
python3 -m pytest -q
..........                                                               [100%]
10 passed in about 24 s
```

两次完整演示的 stdout 逐字节一致，报告与素材清单哈希一致：

```text
material_manifest_sha256 = 2b71ce8196bb277cc0a549e21afa3499154e6e3debb668226709a8dec4b2fcd3
report_sha256            = dd44e603f0065d7daab017b9fe0863804906790b3f18867db33086bd3992bd6d
```

关键版本索引哈希：

```text
v6  baseline index = 59dd6eab2ba6953518edf74ec43f04763dc6ef022e394ea3a9559f7c9f534595
v7  config index   = 0f0600318315021a472b6c134abc6e1adb87bd4c465b4d511085fd2c5a20b6a0
v8  reverted index = 59dd6eab2ba6953518edf74ec43f04763dc6ef022e394ea3a9559f7c9f534595
v9  resumed index  = 017a98ee5bf38c8468ec3f3a0b02715895b80268fc2a6311d1296fed84d010fc
v9  control index  = 017a98ee5bf38c8468ec3f3a0b02715895b80268fc2a6311d1296fed84d010fc
```

- 配置回退 v8 与 v6 索引哈希、状态摘要逐字节一致，所有候选逐字段一致。
- `SIGKILL` 发生在分片临时文件写入后、提交点前；恢复停在 v8，未出现 v9。
- 恢复回收 1 个半写临时文件、2 个未引用完整分片、1 个孤立指纹和 1 个孤立音频。
- 恢复后导入得到 v9，与未中断对照库的状态摘要、索引哈希、版本链和检索候选完全一致。
- 6 个演示真值命中 6 个；对齐误差 240–3428 个采样点（约 5.4–77.7 ms）。

完整报告见 `artifacts/demo/report.json`，逐版本哈希见 `artifacts/demo/version_hashes.json`，崩溃现场见 `artifacts/demo/crash_recovery.json`，素材与真值见 `artifacts/demo/materials/`，简明实测说明见 `artifacts/validation_report.md`。

## 测试覆盖

- 生成器两次运行的所有 WAV 与真值清单逐字节相等，真值区间均落在音频长度内；
- 指纹两次提取逐字节相等，窗长、帧移、峰值密度、频带变化均改变内容哈希；
- 增量导入、阈值更新不重写分片、删除/undo/revert 精确恢复历史清单；
- 历史索引只读重建哈希与当时直接产出一致；
- 分片篡改、schema 篡改、快照缺失、非法音频、非法预算和候选漂移的业务错误链路；
- 子进程强制终止后的稳定版本恢复、孤儿回收和恢复后续操作等价性。
