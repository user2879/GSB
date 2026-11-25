# 实际验证记录

验证日期：2026-11-26（Asia/Shanghai）

## 环境

```text
Python 3.11.0rc1
numpy 2.4.6
pytest 9.2.0
Linux / WSL Ubuntu
虚拟环境：/home/zsl/feedback/uidfc5909d3-default-312/.venv
```

基础解释器缺少 `ensurepip`，首次 `python3.11 -m venv .venv` 失败；随后使用本机 `virtualenv -p python3.11` 在工作目录内创建 `.venv`，并安装 `numpy`、`pytest`。没有使用 Docker，没有访问在线服务或外部数据集。

## 自动化测试

执行：

```bash
.venv/bin/pytest -q
```

结果：

```text
15 passed（约 3 秒）
```

覆盖内容：

1. 文档重复、版本非法、字段非法、词项越界、查询重复/空词、字段偏好越界、坏 JSON、非法编辑等均产生 `E_INDEX_INPUT`。
2. 标题/正文词频、原始位置、停用词排除位置、文档标记排除位置与词项 DF 正确。
3. 串行与线程池索引产物逐字段一致，索引指纹一致。
4. 定点半值舍入、位宽/量化切换会改变真实整数分和饱和中间量。
5. TopK 定点并列按文档 ID/版本决断；边界差在半量化步内进入清单；并列翻转归因正确。
6. 逐级误差归因可分别识别 `term_frequency_saturation_rounding` 与 `field_length_normalization_quantization`，并稳定给出首次超差环节。
7. 零贡献文档的内容变化不会击穿查询-文档得分缓存。
8. 替换单篇文档后，不引用变化词项的查询排序复用，引用变化词项的查询重建；增量投影与全量投影相等。
9. 线程池与串行引擎的比较投影相等，所有观测活动线程数不超过配置上限。
10. 批次预算耗尽会保留查询进度并记 `E_POINT_BUDGET_PENDING`；后续批次继续完成任务。
11. 单查询配额不足时中止并记 `E_QUERY_BUDGET_EXCEEDED`。
12. 单批预算小于单文档点数时调度器停止而非空转，并标记 `scheduler_stalled`。
13. 当前快照损坏时从 `LAST_GOOD` 恢复上一版本。
14. 本地 24 文档/8 查询样本的增量结果与全量结果完全一致，并产生边界文档与次序翻转。
15. 子进程在批次快照提交后调用 `os._exit(137)`，随后 `--resume` 仅依赖持久化快照完成；恢复后的 `scores/ranking/failures/index/term_stats` 与全量重建逐文件一致。

## 最终验收运行

命令：

```bash
.venv/bin/python -m search_proto run \
  --config samples/config.json \
  --vocabulary samples/controlled_vocab.json \
  --stopwords samples/stopwords.json \
  --documents samples/documents.jsonl \
  --queries samples/queries.jsonl \
  --edits samples/edits.jsonl \
  --out-dir artifacts/acceptance-final \
  --snapshot-dir artifacts/acceptance-final-state \
  --mode both
```

核心结果：

```text
comparison.equal = true
incremental projection sha256 = df5f31acd166f16b6c655a4909c74507b4cf0531bd648d10796f7d81944eca37
full projection sha256        = df5f31acd166f16b6c655a4909c74507b4cf0531bd648d10796f7d81944eca37

查询数                 = 8
边界文档数             = 18
定点/浮点次序翻转对    = 22
失败清单条目           = 16（均为 E_POINT_BUDGET_PENDING）

增量缓存：
  active     = 266
  reused     = 9
  rebuilt    = 257
  invalidated= 9

全量缓存：
  active     = 266
  reused     = 0
  rebuilt    = 266
  invalidated= 0

保留快照版本数         = 3（按保留策略压缩，当前版本与最近历史可恢复）
```

翻转归因集合包含：

```text
tie_break_rule
term_frequency_saturation_rounding
accumulation_quantization
```

增量运行先完成基线查询建立缓存，再施加文档追加/替换/删除、字段权重、`k1/b`、停用词、定点位宽/量化、TopK 编辑，并重跑查询。最终查询排序、每文档定点分与浮点参考分、边界文档、失败清单顺序、依赖集合以及活动缓存键/结果指纹与空白全量构建一致。

最终产物位置：

```text
artifacts/acceptance-final/comparison.json
artifacts/acceptance-final/incremental/index.json
artifacts/acceptance-final/incremental/term_stats.json
artifacts/acceptance-final/incremental/scores.json
artifacts/acceptance-final/incremental/ranking.json
artifacts/acceptance-final/incremental/failures.json
artifacts/acceptance-final/incremental/cache_ledger.jsonl
artifacts/acceptance-final/incremental/report.json
artifacts/acceptance-final/full/...
```

## 并发与串行

配置 `max_workers=4`。文档统计和查询打分均在线程池中执行，任务键稳定；合并严格按文档 ID、字段固定顺序和查询词项顺序。串行模式使用 `max_workers_override=1`，其比较投影与线程池模式相等。所有观测 `max_active_observed` 均不超过配置上限。

## 中断恢复

自动化测试先以 `--crash-after-batch 1` 启动真实子进程。进程在完成一次原子快照提交后以退出码 137 终止；随后同一命令加入 `--resume`，运行记录显示 `decision=resume`，并且恢复输出与全量重建输出逐文件相等。
