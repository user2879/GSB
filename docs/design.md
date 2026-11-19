# 设计与可审计性说明

## 1. 确定性素材

`SourceSpec` 的默认配置为 `640x360 / 30fps / 60s / seed=20261119 / GOP=30`，即 1800 帧、60 个 GOP。生成器不读取真实素材、不访问网络，也不写入时间戳或运行路径。

`source.avi` 是带索引的无压缩 BGR24 AVI。文件头、帧块、索引和填充均按固定字节序列写出；`ground_truth.jsonl` 使用排序键、紧凑 JSON 和固定换行。生成清单同时记录：

- `video.sha256`：AVI 文件字节哈希；
- `pixel_content_sha256`：按帧序拼接 BGR 像素后的哈希；
- `truth.sha256`：逐帧真值 JSONL 哈希；
- `manifest_sha256`：清单自哈希，排除该字段本身；
- 帧数、GOP 边界、镜头边界、镜头级水印矩形。

逐帧真值包含镜头索引、镜头边界、GOP 索引、GOP 边界、三个平移色块、局部遮挡和水印矩形。分析管线只用真值做边界标注，不根据真值改变可疑度计算。

## 2. 采样、掩膜与帧特征

采样区域与水印掩膜都是带可选帧范围的矩形列表。对每个帧：

1. 将适用的采样矩形合并为采样位图；
2. 将适用的掩膜矩形合并为排除位图；
3. 有效位图为 `sampling AND NOT mask`；
4. 若有效像素为零，返回 `E_REGION_BOUNDS`；
5. 记录有效像素数、掩膜像素数、与采样区相交的排除像素数；
6. 对有效像素按行优先顺序计算 SHA-256 内容哈希；
7. 将帧划为 8x8 网格，对有效像素灰度均值生成 64 位关键帧签名；
8. 将帧划为 4x4 色块，对每个色块内有效像素计算独立哈希。

帧节点还保存 `selection_digest`。该摘要包含有效位图哈希、实际参与的本帧采样矩形规范摘要和掩膜矩形规范摘要。因此只有在帧的有效采样口径确实变化时才进入失效种子集合；未受影响的帧节点直接复用。

## 3. 指标口径

每个 GOP 以窗口内首帧为关键帧。对帧 `f`：

```
signature_offset = Hamming(signature(f), signature(keyframe)) / 64
tile_mismatch    = differing_tiles(f, keyframe) / 16
frame_score      = 0.65 * signature_offset + 0.35 * tile_mismatch
```

`frame_score` 与阈值 `suspicion_threshold`（默认 0.35）比较。GOP 摘要记录逐帧分数、均值、最大值和高分帧列表。

相邻 GOP 比较项使用两个 GOP 关键帧的签名和 4x4 色块哈希，采用相同归一化公式。最终结论：

```
overall_suspicion = 0.70 * max(gop_suspicion_max)
                  + 0.30 * max(cross_gop_score)
integrity_status  = review  if overall_suspicion >= threshold
                    consistent otherwise
```

所有分数位于 `[0, 1]`。GOP、跨 GOP 和最终节点均把归一化公式和阈值写入 payload，避免脱离节点解释分数。

## 4. DAG 结构

节点类型与 ID：

| 类型 | ID 示例 | 说明 |
|---|---|---|
| source | `source:root` | 素材清单、视频哈希、真值哈希 |
| frame feature | `frame:000123` | 内容哈希、签名、色块哈希、边界标注 |
| GOP summary | `gop:004` | 关键帧、逐帧分数、GOP 分数 |
| cross-GOP item | `xgop:004-005` | 只比较一个相邻 GOP 边界 |
| conclusion | `conclusion` | 当前分析窗口的完整性结论 |

边带 `trigger_keys`，取值来自 `source / sampling_region / watermark_mask / analysis_window / threshold / gop_size / feature_version`。例如：

- `source:root -> frame:*`：源内容、采样或掩膜变化可触发；
- `frame:* -> gop:*`：成员帧失效传播到 GOP 摘要；
- `gop:* -> xgop:*`：失效 GOP 传播到左右相邻跨 GOP 项；
- `gop:* / xgop:* -> conclusion`：结论重算。

每个节点保存内容哈希、输入节点哈希映射、实际影响该节点的参数版本戳和首次产生版本。`graph_hash` 只覆盖参数版本、节点内容哈希和边；审计历史由快照哈希单独保护。

## 5. 精确失效闭包

一次增量运行按以下顺序确定闭包：

1. 比较上一提交图与当前完整参数版本，得到变化参数键；
2. 逐活跃帧计算当前 `selection_digest`，与上一帧节点比较；变化帧成为种子；
3. GOP 成员集合、窗口范围或阈值变化时，相应 GOP 成为种子；
4. 从种子沿目标 DAG 向下游传播：成员帧 -> GOP -> 相邻 xgop -> conclusion；
5. 闭包外节点不读取视频像素，直接复用上一提交节点；
6. 复用前再次校验期望参数版本和输入哈希；任何不一致返回 `E_CLOSURE_MISMATCH`；
7. 每个复用节点记录来源快照版本、来源快照哈希、原产生版本和内容哈希；
8. 默认开启全量对照：对相同目标窗口从源帧重新构建 DAG，并逐节点、逐边比较。

因此单个 GOP 的摘要失效时，只会重算该 GOP 及左右两个独立 `xgop` 比较项，不会沿用其他比较项旧分数。窗口只改变边界 GOP 的成员集合时，帧特征可全部复用，仅边界 GOP、相邻 xgop 和结论重算。

## 6. 稳定提交协议

工作区状态位于 `workspace/state`：

```
state/
  CURRENT.json
  snapshots/snapshot-000001-<hash>.json
  pending/<run-id>.json
  recovery/recovery-000001.json
  orphans/
```

提交步骤：

1. 运行器写入候选 `pending` 摘要；
2. 生成序列化快照并计算 `snapshot_hash`；
3. 在快照目录内写临时文件，`fsync` 后 `os.replace` 为正式快照；
4. 写临时 `CURRENT.json`，`fsync` 后原子替换指针；
5. 删除 `pending` 摘要。

`CURRENT.json` 的原子替换是提交点。快照文件存在但指针未替换时，该快照仍视为未提交。恢复时从当前指针沿 `base.version / base.snapshot_hash` 追溯提交链；链外完整快照、`pending` 文件和原子临时文件均隔离到 `orphans/`。每个孤儿事件写入恢复报告，并标记业务码 `E_SNAPSHOT_VERSION` 与文件哈希、大小、当前版本等上下文。

提交再次读取当前指针，并以 `expected_base_version + expected_base_hash` 做乐观并发校验。版本或哈希不匹配返回 `E_SNAPSHOT_VERSION`，不会把旧图当作新缓存复用。

## 7. 受控退出与强制终止

正常 `execute(..., run_dir=...)` 完成快照提交后写入：

- `exit_manifest.json`：全部节点内容哈希、结论哈希、快照版本与哈希；
- `close.json`：`clean_exit=true` 的关闭标记，引用退出清单哈希。

强制终止演示在“快照文件已写入、`CURRENT` 尚未替换”处调用 `os._exit(17)`。此时运行会话没有关闭标记，存在一个未引用快照和一个候选摘要。重启恢复：

1. 读取最近一次稳定提交，不使用候选快照；
2. 隔离孤儿并写 `state/recovery/*.json`；
3. 对未清洁运行写 `recovery-report.json`，明确 `clean_exit=false`，不补写 `close.json`；
4. 对提交状态逐节点比对；随后以相同参数运行，应得到 0 个重算节点和相同结论哈希。

## 8. 错误码

| 业务码 | 退出码 | 典型链路 |
|---|---:|---|
| `E_FRAME_FORMAT` | 10 | AVI 头、帧块尺寸、帧 dtype/形状、配置结构损坏 |
| `E_REGION_BOUNDS` | 11 | 采样区域越界、分析窗口非法、有效采样为空 |
| `E_DEPENDENCY_STALE` | 12 | 图中缺节点、节点内容哈希不匹配、复用节点缺失 |
| `E_CLOSURE_MISMATCH` | 13 | 增量与全量结果不一致、复用输入哈希过期 |
| `E_SNAPSHOT_VERSION` | 14 | 基准版本变化、快照哈希冲突、孤儿回收事件 |
| `E_MASK_INVALID` | 15 | 掩膜尺寸、原点、帧范围或边界非法 |

所有异常均带 `code / message / context / exit_code`。命令行只把 `ForensicError` 转成结构化 JSON，不以通用异常替代业务异常。

