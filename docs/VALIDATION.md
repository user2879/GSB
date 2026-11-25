# 实际验证记录

验证时间：2026-11-26 05:37 (+08:00)  
环境：WSL/Ubuntu，Python `3.11.0rc1`，NumPy `1.26.4`  
完整终端记录：`artifacts/verification-results.txt`

## 执行命令

```bash
python -m lexicon_pipeline generate
python -m pytest -q
python -m lexicon_pipeline validate
python -m lexicon_pipeline run
```

结果：

- 固定种子 `3182026` 成功生成 11 个样本文件到 `samples/`。
- `python -m pytest`：18 个测试全部通过（`18 passed`），退出码 0。
- 声明加载校验：`declarations = ok`。
- 完整本地交付：`run_exit=0`，未触发等价性断言。

## 格式校验与往返

异常样本验证到以下定位：

- `samples/jv1/truncated.jv1` 第 2 行、字段 `fragment_seq`：`E_FIELD_TRUNCATED`。
- 同文件第 3 行、字段 `score_permille`：等宽字段中出现非法字符 `0098x`，仍为 `E_FIELD_TRUNCATED`。
- `samples/jv2/invalid.jv2.json` 检出 5 个 `E_FORMAT_INVALID`，包含 `receipts[0].score` 缺字段和 `receipts[1].doc_seq` 类型错误。

最终文档成功处理 6 篇；`D007` 按预算设计触发单文档配额失败。回执数量：

- jv1 行数：22。
- jv2 回执数：22。
- 旁路记录：22。
- 降级条目：22。
- 往返失配数：0。

单测 `test_downgrade_sidecar_restores_float_and_naming` 额外断言 `0.8123719 -> jv1 -> 带旁路 jv2` 后，原始浮点得分、命名词性和命名舍入模式逐字段恢复。

## 定点标注与边界

命令流最终选择 `fine` 量化档位：

- 基础小数位宽：10。
- `fine` 偏移：+2。
- 最终位宽：Q12。
- 当前 scale：4096。
- 高一位宽参考：Q13，scale 8192。

样例 `D006:F0001` 出现 `threshold_equal`、`threshold_half_step` 和 `candidate_tie`，报告包含高一位宽参考总分与差值。测试验证切换 `(bits,tier)` 为 `(8,standard)`、`(10,standard)`、`(10,fine)` 时得到三组不同的 `(scale,path_total_int)`，说明不是仅修改显示。

NumPy 交叉校验测试把选中路径的定点转移分和发射分组成 `int64` 数组累加，结果与 Python 任意精度整数路径总分一致。

## 增量复用

主命令流包含新增、词频调整、权重调整、规则修改、版本降级失败、删除、修订链回滚、交换版本切换和量化档位切换。最终与基线相比存在量化口径变化，因此标注层 22/22 未命中，但切分层有 21 次真实缓存命中、1 次未命中；影响清单含 67 条失效记录和 42 条分阶段复用计划。

专门的词频复用演示只修改真实被 `D004` 引用的 `E010`。测试还分别覆盖新增 `E030` 命中位置、删除 `E010`、修改 `NN>VV` 转移权重后的影响依据和全量等价：

- 缓存事件总计 132 个。
- 命中：122。
- 未命中：10。
- `segment`：21 hit / 1 miss。
- `tag_candidates`：21 hit / 1 miss。
- `path_total`：20 hit / 2 miss。
- `annotation`：20 hit / 2 miss。
- jv1/jv2 序列化各 20 hit / 2 miss。
- 影响清单：3 条失效、20 条复用计划。
- 增量结果与空白缓存全量结果逐字段一致：`true`。

这与“直接引用片段及其 Viterbi 下游失效，其他文档和前缀复用”的设计一致。

## 一致性与调度

`artifacts/diagnostic-report.json` 中：

- `incremental_vs_blank_full_equal = true`。
- `serial_vs_concurrent_equal = true`。
- `serial_vs_concurrent_cache_events_equal = true`。
- `incremental_roundtrip_ok = true`。

预算样例：

- 批次 1：`D001,D002,D003` 执行，消耗 684/700；`D004,D005,D006` 进入待办，剩余 16。
- 批次 2：优先执行待办 `D004,D005,D006`，消耗 627，待办清空。
- `D007` 匹配点 399，超过单文档配额 300，记录 `E_DOC_QUOTA_EXCEEDED`。
- 业务码计数：`E_DOC_QUOTA_EXCEEDED=1`、`E_MATCH_BUDGET_PENDING=3`、`E_VERSION_DECREASED=1`、`E_SNAPSHOT_STALE=1`。

完整流水线中的任务很短，实际观测峰值为 1；调度器单测使用三任务屏障验证线程池可达到 `max_observed_concurrency=3`，且 `execute_pool(..., concurrency=3)` 的 pool size 不超过配置。

## 强制终止恢复

实际执行：

```bash
python -m lexicon_pipeline crash-demo --snapshot-dir artifacts/crash-demo-final
# Killed，shell 退出码 137
python -m lexicon_pipeline snapshot-status --snapshot-dir artifacts/crash-demo-final
```

恢复判定输出 `action=rebuild`、`code=E_SNAPSHOT_STALE`，原因是只有 `.staging-kill-demo` 而没有已提交 manifest；恢复逻辑清理了残留临时目录，状态命令退出码为 2。

## 诊断产物

- 机器报告：`artifacts/diagnostic-report.json`
- Markdown 报告：`artifacts/diagnostic-report.md`
- 账本：`artifacts/ledger.jsonl`
- 最终 jv1：`artifacts/receipts/final.jv1`
- 最终 jv2：`artifacts/receipts/final.jv2.json`
- 旁路：`artifacts/receipts/final.sidecar.json`
- 快照：`artifacts/snapshot/current/manifest.json`
