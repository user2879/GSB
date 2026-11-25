# 实现说明

本项目在本地 WSL/Ubuntu 中实现受控词表、确定性切分、定点词性标注、回执转换、增量复用账本、预算调度和阶段化快照。

## 快速命令

```bash
python3.11 -m pip install -r requirements.txt
python -m lexicon_pipeline generate
python -m lexicon_pipeline validate
python -m pytest -q
python -m lexicon_pipeline run
python -m lexicon_pipeline snapshot-status --snapshot-dir artifacts/snapshot
```

`python -m lexicon_pipeline run` 会依次执行基线运行、编辑命令流、命令后增量运行、空白缓存全量串行重建、串行/并发等价运行，并写出：

- `samples/`：固定整数种子 `3182026` 生成的声明和异常/正常回执样本。
- `artifacts/receipts/final.jv1`：最终定宽回执。
- `artifacts/receipts/final.jv2.json`：最终 JSON 回执。
- `artifacts/receipts/final.sidecar.json`：降级旁路元数据。
- `artifacts/ledger.jsonl`：缓存、失效、复用、调度、降级和边界账本。
- `artifacts/diagnostic-report.json` / `.md`：诊断报告。
- `artifacts/snapshot/current`：当前版本化阶段快照符号链接。

## 模块映射

- `lexicon_pipeline/validation.py`：词表、文档、权重、规则、修订链和命令流校验；修订链使用父引用存在性检查和 DFS 环检测。
- `lexicon_pipeline/formats.py`：jv1 定宽解析/编码、jv2 schema 校验、降级旁路和带旁路恢复的升级。
- `lexicon_pipeline/segmentation.py`：候选生成、FMM 排序、重叠吞并清单、强制切分/合并和切分指纹。
- `lexicon_pipeline/tagging.py`：整数定点 Viterbi、固定递推顺序、稳定并列决断、候选分、路径总分、高位宽参考分和边界片段。
- `lexicon_pipeline/fixedpoint.py`：Decimal 辅助舍入、定点量化和 NumPy 整数累加交叉校验。
- `lexicon_pipeline/cache.py`：内容指纹、分层缓存键、持久化 `cache.json` 和依赖影响规划。
- `lexicon_pipeline/scheduler.py`：稳定任务键、单文档配额、单批预算、待办优先和有界线程池。
- `lexicon_pipeline/snapshot.py`：临时目录写入、版本目录、符号链接原子切换、启动身份判定和中断残留恢复。
- `lexicon_pipeline/commands.py`：编辑命令流、条目版本单调检查、量化/交换版本切换和修订快照回滚。
- `lexicon_pipeline/runner.py`：本地首版交付编排、等价断言和诊断报告。

## jv1 定宽口径

默认字段宽度为：

| 字段 | 宽度 | 说明 |
| --- | ---: | --- |
| `doc_seq` | 4 | 文档序号，零填充整数 |
| `fragment_seq` | 4 | 片段序号，零填充整数 |
| `score_permille` | 5 | 千分比整数得分，`0..1000` |
| `pos_code` | 4 | 整数词性编码 |

单行固定为 17 个字符。宽度不足、超长或数字字段含非法字符均输出 `E_FIELD_TRUNCATED`，问题中带 `line` 和 `field`。jv2 缺字段或类型错误输出 `E_FORMAT_INVALID`，问题中带 JSON Pointer 风格路径。

## 往返恢复

jv2 降 jv1 时：

1. 浮点得分按回执中的命名舍入模式量化为千分比整数。
2. 命名舍入模式按码表映射：`EXACT=0`、`ROUND_HALF_UP=1`、`ROUND_HALF_EVEN=2`、`ROUND_DOWN=3`。
3. 原始浮点得分、残差、精度标签、命名词性、舍入码和被抑制的命名模式写入 `final.sidecar.json`。
4. 降级事件写入账本。

jv1 升 jv2 时，若无旁路则用 `score_permille / 1000.0` 和词性码表做精确有理换算；若有匹配旁路，则恢复原始浮点得分、命名词性、舍入模式和精度标签。正常样本的 22 条回执往返失配数为 0。

## 确定性切分

候选查询顺序为：

1. 词项长度降序；
2. 同长按整数词频降序；
3. 再按词项字典序和稳定条目 ID。

重叠候选裁决顺序为长度降序、同长起始位置升序、再按词项字典序。未中选候选写入 `suppressed`，其中包含 `superseded_by`、吞并条目和区间。规则冲突按规则版本号降序、规则包条目顺序升序、`rule_id` 升序决断。

切分指纹由整数字典修订版本、完整规则包指纹和文档实际生效规则指纹共同派生。

## 定点标注

当前定点尺度为 `scale = 2 ** (fraction_bits + tier_shift)`。样例配置基础小数位宽为 10，`standard` 档位偏移 0；命令流最终切到 `fine`，偏移 +2，因此最终口径为 Q12，`scale=4096`，高一位宽参考为 Q13，`scale=8192`。

递推按片段起始位置升序、词性编码升序、前驱编码升序执行。同分优先保留小编码前驱；最终路径同分时依次比较词性编码和片段起始位置。切换位宽或档位会重新量化权重和发射分，因此中间整数与路径总分真实变化。

边界判定比较当前定点路径前缀与配置阈值 `boundary_threshold`，并使用 Q(n+1) 重算同一路径，输出 `high_reference_total_int`、`delta_high_int` 和 `delta_raw`。候选标签同分也会标记 `candidate_tie`。

## 精确缓存与失效

每个片段生成以下键：

- `segment_candidate`
- `tag_candidate_set`
- `viterbi_path_total`
- `final_annotation`
- `serialization_product:jv1`
- `serialization_product:jv2`

键材料包含引用条目的内容指纹和版本、作用域修订摘要、生效规则指纹、作用域转移权重、量化位宽/档位和交换版本。片段保存 `entry_refs` 与 `weight_refs`。

词频演示修改 `E010` 后，直接引用片段和 Viterbi 下游路径失效；不相关文档和前缀仍命中持久化缓存。报告中的 `precise_reuse_demo` 记录了逐阶段 hit/miss。

## 预算与并发

单文档匹配点数为：

```text
doc.term_count × sum(enabled_candidate_term_lengths)
```

点数超过 `doc_quota_points` 时记录 `E_DOC_QUOTA_EXCEEDED` 并中止该文档本批；当前批剩余预算不足时，从第一个放不下的文档开始保留稳定待办队列并标记 `E_MATCH_BUDGET_PENDING`。下一批将待办 ID 放在稳定顺序最前。

每个任务带有由 `doc_id` 和阶段名派生的 `stable_sort_key` 与各阶段键。线程池大小为 `min(config.concurrency, task_count)`，不会超过配置。

## 快照与强制终止

`SnapshotManager` 先写 `.staging-*` 临时目录，再重命名为不可变版本目录 `releases/snapshot-N-*`，最后通过临时符号链接和 `os.replace` 原子切换 `current`。启动时比较 `config_hash` 和 `lexicon_revision`：一致则续跑，不一致则重建；缺少已提交 manifest 时输出 `E_SNAPSHOT_STALE`。

真实 `SIGKILL` 演练命令：

```bash
python -m lexicon_pipeline crash-demo --snapshot-dir artifacts/crash-demo-run
# shell 观察到 137
python -m lexicon_pipeline snapshot-status --snapshot-dir artifacts/crash-demo-run
# E_SNAPSHOT_STALE，并清理 .staging-kill-demo
```
