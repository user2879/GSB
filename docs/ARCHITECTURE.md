# 架构、确定性与快照

## 模块

| 模块 | 职责 |
| --- | --- |
| `generator.py` | 固定种子合成 16 kHz/单声道/16-bit PCM WAV 和三版本样例 |
| `formats.py` | v1/v2/v3 编解码与严格校验 |
| `conversion.py` | 双向转换、残差、旁路元数据和降级项 |
| `scoring.py` | 帧特征、低能量帧噪声底、分档维度分、定点加权 |
| `diagnostics.py` | 偏差、跨版本往返、等级翻转归因、注入项 |
| `cache.py` | 内容寻址缓存、blob 存储、每文件依赖状态 |
| `snapshot.py` | 四阶段清单、原子提交、过期判定、历史清理 |
| `pipeline.py` | 解析 → 帧特征 → 定点打分 → 版本转换四阶段运行时 |
| `edits.py` | 音频替换、配置/等级表编辑、版本切换 |

## 定点评分

1. 以 `frame_length_ms` / `hop_ms` 切完整帧；末段不足一帧时不补零。
2. 每帧计算 RMS 电平、过零率、峰值、最长连续零游程和削波样本数。
3. 低能量帧集合由帧 RMS 的配置分位点确定，只对 `rms_db <= cutoff` 的帧求线性功率均值和噪声底。
4. 只有最长连续零游程达到 `consecutive_zero_samples` 的帧才形成连续性劣化；指标按有效游程占帧长比例分档。
5. 四维原始分档阈值为固定表；恰好等于阈值时使用 `<=` 桶，并生成边界事件。
6. 维度分吸附到配置的 `quantization_levels`（等距 ties 取低档）。
7. 权重以 `Fraction(Decimal(str(weight)))` 表示，加权和 half-up 舍入为整数微分数。
8. 等级按精确 `total_micro` 与 `min_total_permille*1000` 比较；因此 v1/v3 精度损失可能产生可诊断的等级翻转。

## 缓存依赖

| 产物 | 缓存键依赖 |
| --- | --- |
| 帧特征与摘要 | 音频 SHA-256、采样率、帧长、帧移、静噪门限、噪声底分位、削波阈值、连续零阈值、特征算法版本 |
| 四维打分 | 帧特征键、静噪门限、分位、削波阈值、零游程、量化档位 |
| 综合评分 | 四维键、规范化权重 |
| 每文件版本序列化 | 音频指纹、完整语义配置、结果微分数、等级映射条目、目标版本、v3 位宽、帧集合摘要 |

因此：

- 改权重：帧/四维复用，总分与序列化重算。
- 改量化档位：帧复用，四维及下游重算。
- 改静噪门限、噪声底分位、削波阈值、连续零阈值或帧长/帧移：帧摘要及全部下游重算。
- 替换音频：仅该文件及其下游重算；即使分档碰巧相同，序列化也因音频指纹变化而失效。
- 加无关等级：所有产物复用；删除/改名正在使用的等级只会影响相关文件并产生 `E_LEVEL_UNMAPPED`。
- 切换版本：帧、四维、总分复用，仅目标版本序列化失效。

账本顺序固定为 `features`、`dimensions`、`total`、`serialization`，每阶段再按 `file_id`、产物名排序。

## 并发

- 线程池工作数 `min(max_concurrency, concurrency_cap, 16)`。
- 帧任务排序键：`file_id|frame|000123`。
- 文件打分排序键：`file_id|score|dimensions`。
- 所有 future 通过 `as_completed` 收集后按稳定键重排，不依赖调度完成顺序。
- 失败也进入稳定 `(file_id, stage, frame_index, task_key, code)` 排序。
- 配置默认开启串行复算校验；并行与串行帧特征/维度分摘要必须相等。

## 快照

阶段：`parse`、`features`、`scores`、`conversion`。

- 阶段文件先写临时文件，再 `os.replace` 提交。
- 清单原子更新，记录全局版本、配置哈希、内容指纹、版本指纹、各阶段输入指纹和产物版本。
- 启动时全局 job 版本、配置哈希、等级表哈希或活动版本变化会输出 `E_SNAPSHOT_STALE`，清空快照并从阶段零重建；内容寻址缓存仍可用于精确复用。
- 相同配置与内容再次启动可直接恢复阶段；`--stop-after parse` 用于验证中断续跑。
- 完成时清理未被当前清单引用的历史阶段文件。

## 业务码

| 码 | 含义 |
| --- | --- |
| `E_EVAL_FORMAT` | 交换格式、分区、校验和、重复标识等错误 |
| `E_LEVEL_UNMAPPED` | 等级码或名称不在当前映射表 |
| `E_AUDIO_FORMAT` / `E_AUDIO_MISSING` | WAV 格式错误 / 音频缺失 |
| `E_SNAPSHOT_STALE` | 快照版本或配置不一致，已重建 |
| `E_FRAME_FEATURE` / `E_SCORE` / `E_SERIALIZE` | 阶段执行失败 |
| `DEG_V1_TRUNCATION` | v1 千分比精度损失 |
| `DEG_V3_FIXED_WIDTH` | v3 定点位宽损失 |
| `DEG_FRAME_SUMMARY_UNAVAILABLE` | 源数据没有帧摘要 |
| `DEG_FRAME_SUMMARY_DROPPED` | 目标 v1/v2 无法表达帧摘要，数量写入 bypass |
| `DEG_LEVEL_METADATA_LOST` | 目标版本无法表达精度标签或命名等级 |
