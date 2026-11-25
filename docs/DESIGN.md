# 设计说明与可复核契约

## 1. 规范结果与运行报告分离

`result.json` 不含时间戳、绝对状态路径、线程峰值或缓存命中状态，因此可对同一最终音频集合与配置做增量/全量逐字段比较。`report.json` 才包含运行模式、缓存事件与并发观测。

规范结果顶层字段：

```text
schema_version
config
corpus
failures
templates
similarities
confusion_candidates
representatives
clusters
conclusions
confusion_arbitration
confusion_matrix
budget
pending_queue
stage_cache_keys
```

所有集合按稳定键排序：模板键、模板对键、命令词、片段标识、矩阵行列。

## 2. 分阶段缓存键

| 阶段 | 主要键材料 |
|---|---|
| 声学模板 | 模板键/路径、PCM 内容指纹、特征配置指纹、投影配置 |
| 定点模板 | 声学向量指纹、内容指纹、profile、fractional_bits |
| 相似度基值 | 两侧模板键与定点整数指纹 |
| 候选判定 | 相似度基值键、混淆候选阈值 |
| 代表模板 | 同词成员定点指纹、有效帧、代表规则、同词比较覆盖 |
| 命名词簇 | 代表模板键、合并阈值、代表对比较结果 |
| 逐片段结论 | 片段定点指纹、全部代表模板定点指纹与综合分、拒识阈值、候选数、小数位宽 |
| 混淆仲裁 | 候选键、两侧代表综合分、簇归属、合并阈值 |
| 矩阵单元 | 真值/结论、预测簇、逐片段结果键、高一位宽参考 |

缓存命中只减少计算，不改变预算选择。因此带预算的增量运行与空白全量运行仍可逐字段一致。

## 3. 固定点约定

- 定点步长：`1 / 2^fractional_bits`。
- 存储宽度：`profile=qN` 时 N bit，范围 `[-2^(N-1), 2^(N-1)-1]`。
- 舍入：正数 `floor(x+0.5)`，负数 `ceil(x-0.5)`，即半值远离 0。
- 相似度：整数点积与整数范数，`math.isqrt(norm_a)*math.isqrt(norm_b)` 为分母，四舍五入为 permille。
- 综合分：同词模板相似度总和除以比较数，整数半值向上舍入。
- 混淆分歧分：`(1000-sim)*2 + score_gap//2 + max(0, threshold_delta)//2`。

所有“平局”都先以模板键或命令词标识字典序规范化，再写入结果；需要时带 `tie_break_basis` / `tie_reason`。

## 4. 失败顺序

规范失败顺序为：

1. 配置加载与字段校验失败（输入顺序）。
2. 工作区 WAV 扫描失败（路径字典序）。
3. 标注/命令流缺失或非法（JSONL 行序，再按路径检查）。
4. 预算待办失败（稳定任务键字典序）。

错误不从 `run` 抛出为进程异常，而是进入 `failures`；只有调用参数不合法（例如 mode 非 auto/full）会抛 `ValueError`。

## 5. 快照提交

每次运行：

1. 在 `snapshots/.staging-<pid>-<version前缀>/` 写入：
   - `result.json`
   - `run_report.json`
   - `stage-audio_manifest.json`
   - `stage-template_acoustic.json`
   - `stage-quantized_templates.json`
   - `stage-similarities.json`
   - `stage-clusters.json`
   - `stage-conclusions.json`
   - `stage-confusion_arbitration.json`
   - `stage-confusion_matrix.json`
   - `pending-queue.json`
   - `completed-cursors.json`
   - `manifest.json`
2. 整体重命名为 `snapshots/<state_version>/`。
3. 写临时 `current.json` 后使用 `os.replace` 原子替换指针。

内容缓存位于 `state/cache/entries/`，每项也以临时文件写入后替换。结构配置哈希变化时只清理与结构不兼容的内容缓存；阈值、代表规则、预算等运行参数变化仍走内容键增量。

## 6. 已知边界

- 高一位宽参考结论复用低位宽代表模板拓扑；该事实通过 `topology_source` 写入参考比较记录。这样避免把“诊断参考”变成第二套完整重聚类，同时保持参考结论口径明确。
- 预算不足时矩阵仍保持计数不变量，`unavailable_clip_count` 计入尚未执行的逐片段任务；不得把缺失片段伪造成拒识。
- 缓存事件位于运行报告，不进入规范结果，因此“同一最终状态”不因是否发生复用而变化。
