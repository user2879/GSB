# 技术设计说明

## 1. 处理阶段与数据契约

单句流水线分为三个可独立失败的阶段，产物逐句写快照：

1. **分词（segmentation）**：`segpos/segmenter.py` 正向最大匹配。
   - 取词典最大词长为窗口，收集起点处全部词典前缀。
   - 先比长度（最长优先）；同长按词频降序，再按词形字典序升序。
   - 输出每个词的 `start/end/text/match_basis/hit/boundary_confidence`。
   - 未命中前缀的字符产出 `oov_single_char` 片段。
2. **未登录词识别（oov）**：`segpos/tagger.py::segment_with_oov` 对连续单字片段做字符级 HMM 维特比，相邻同词性字符合并为一个 OOV 词，`match_basis=hmm_oov:pos=...`。
3. **词性标注（tagging）**：`segpos/tagger.py::tag_tokens` 对最终词序列做词级一阶 HMM 维特比，回填 `pos` 与 `local_path_score`，整句给出按词数归一化的对数概率 `sentence_logprob`。

维特比实现位于 `segpos/hmm.py`，全程自然对数域、NumPy 矩阵运算；前向变量同时用于计算每位置后验边际，作为边界置信度来源。

## 2. 平滑与命中统计

- 缺失词发射、单字发射、词性转移统一回退到 `config.smoothing`（要求为负的对数值）。
- 每次回退累加 `smoothing_hits`，随句子结果落盘并进入报告。
- 句首先验 `start_logprob` 要求覆盖全部词性，缺失直接判配置非法，避免静默错误。

## 3. 依赖失效判定（核心）

缓存键（`segpos/fingerprints.py::cache_key`）：

```
SHA256( canonical( text_fingerprint, hit_words_fingerprint, prob_table_version ) )
```

`DependencyCache` 同时维护两张结构：

- `records`：句子 ID -> 句子结果（含缓存键）。
- `word_index`：词条 -> 命中该词的句子集合（倒排索引）。

失效规则：

| 编辑 | 判定方式 | 失效范围 |
| --- | --- | --- |
| 新增词条 | 旧索引无该词，对缓存内全部句子文本做子串扫描 | 文本包含该词的句子 |
| 删除词条 / 调整词频 | 倒排索引 `word_index[word]` | 命中该词的句子 |
| 批量编辑 | 合并全部受影响词后一次失效 | 并集句子 |
| 概率表替换 | 版本号递增，清空全部记录 | 全部句子 |
| 单句文本替换 | 仅逐出该句并释放其隔离状态 | 该句 |

即使失效判定出现偏差，缓存键中的“文本 + 命中词集合 + 概率表版本”三重指纹仍会在 `get` 时拒绝过期结果，构成第二道防线。未命中失效集合的句子不会进入待处理队列，因此**不会被重新计算**。

## 4. 版本化快照与恢复

`segpos/snapshot.py`：

- 目录结构：`manifest.json` 与 `results/<sha256(句子ID)>.json`。
- 提交时先在同级临时目录写全部产物，再逐个 `os.replace` 到正式目录，最后原子替换 manifest；manifest 是唯一提交点，强制终止后只可能看到“上一个完整版本”。
- manifest 记录：`global_version`、`dict_hash`、`prob_table_hash`、`prob_table_version`、`completed_sentences`、各阶段 `artifact_versions`、`failures`。
- 启动续跑比对词典哈希、概率表哈希与概率表版本，任一不一致抛 `E_CONFIG_MISMATCH` 并拒绝续跑。
- 同引擎实例内的编辑通过血缘计数 `_lineage_edits` 跳过磁盘复核（输入变更由编辑 API 显式驱动），跨进程新引擎仍严格走哈希校验。

## 5. 并发与串行一致性

- `ThreadPoolExecutor(max_workers=config.workers)` 并发执行无共享写入的纯函数式单句任务。
- 任务按 `stable_sort_key(句子ID, 阶段名)` 排序提交与收集；共享缓存、失败清单、快照提交只在主线程发生。
- 浮点运算全部走同一套 NumPy 路径，`workers=1/2/4/8` 的结果在测试中逐字段相等。

## 6. 异常隔离与低置信度

- 三个阶段分别支持故障注入钩子 `pipeline.fault_injection = {句子ID: 阶段名}`，抛 `StageError(code=E_STAGE_FAILED)`。
- 失败句写入失败清单（句子 ID、阶段名、业务码、异常类型、消息），并进入缓存的持久隔离集合；隔离 ID 随 manifest 保存，恢复后继续生效。
- 边界置信度低于 `low_confidence_threshold` 的词进入 `low_confidence_words` 并在报告汇总计数。

## 7. 优雅退出与报告

- `SIGTERM/SIGINT`（Windows 仅 `SIGINT`）置位停止事件；当前在途句子跑完、当前批次原子提交后退出，退出码为 0，报告在提前停止时不写部分报告。
- 完整运行以临时文件 + `os.replace` 写 JSON 报告，字段覆盖总句数、成功/失败句数、低置信度词数、词典/概率表哈希、快照版本、失效/复用/重算句子清单与失败清单。

## 8. 增量—全量等价性

`examples/run_demo.py` 在同一引擎上依次施加：新增词条、词频调整、词条删除、概率表替换、单句文本替换；随后用最终词典、最终概率表（version=2）、最终句子集在全新工作区一次性全量运行。对比器逐字段比较词序列、词性序列、路径得分、归一化对数概率、平滑命中、低置信度清单、句子顺序与失败清单顺序。
