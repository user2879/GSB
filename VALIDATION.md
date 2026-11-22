# 实际验证记录

验证日期：2026-11-23（Asia/Shanghai）

环境：

```text
Python 3.11.0rc1
numpy 1.26.4
pytest 8.3.3
OS: Linux（实现仅使用 pathlib/os.replace/线程池，示例路径兼容 Windows）
```

执行命令：

```bash
python -m corpus_workbench generate-demo --root .
python examples/run_demo.py
pytest -q
```

测试结果：

```text
.............                                                            [100%]
15 passed
```

## 默认完整运行

配置：`balance.json` / `dedup.json` / `scheduler.json` / `injection.json`。

```text
采样记录                 42
唯一采样文档             36
重复组                   2
去重后保留               34
去除重复文档             2
已消耗 shingle 点数      576
待办                     0
加载失败                 4（空文本、非 UTF-8、未知来源、重复 ID）
快照最终版本             8
状态                     complete
```

两个重复组均由 web 来源的近重复文档对构成；代表文档分别为 `web-0012` 与 `web-0018`，被剔除文档与代表文档的 MinHash 相似度为 `0.78125` 与 `0.734375`。

注入命中：

| injection_id | observed_code |
| --- | --- |
| `inj-empty-text` | `E_ENCODING_INVALID` |
| `inj-non-utf8` | `E_ENCODING_INVALID` |
| `inj-unknown-source` | `E_SOURCE_UNKNOWN` |
| `inj-duplicate-doc-id` | `E_DUP_DOC_ID` |

## 串行/并行一致性

`scheduler_serial.json` 使用 1 个线程、`scheduler_parallel.json` 使用 8 个线程，独立冷缓存运行：

```text
serial   deterministic_result_digest = deb2b102301160fa54172f875b8e58b7acc1a422deea06a00cef28e51ff6cb73
parallel deterministic_result_digest = deb2b102301160fa54172f875b8e58b7acc1a422deea06a00cef28e51ff6cb73
```

测试同时断言两者的 `deterministic_core`、预算扣减序列（`budget.deduction_sequence`）、重复组、失败清单完全相等。

## 预算耗尽与恢复

第一次运行使用 `scheduler_tight.json`：

```text
全局预算              160
已消耗                160
已处理唯一文档        10
待办                  26
状态                  budget_exhausted
业务码                E_SOURCE_QUOTA_EXCEEDED, E_GLOBAL_BUDGET_EXHAUSTED
快照版本              7
```

随后使用 `scheduler_expanded.json` 对同一 `run_id` 执行 `--resume`：

```text
状态                  complete
已消耗                576
待办                  0
最终保留              34
快照版本              13
```

恢复后的最终保留集合与默认完整运行按 `(source, doc_id)` 对照完全一致，`deterministic_result_digest` 也同为 `deb2b102301160fa54172f875b8e58b7acc1a422deea06a00cef28e51ff6cb73`。签名缓存命中不重复扣费，已提交签名由快照直接恢复；历史与新增预算扣减按稳定键规范化。

另验证 `--stop-after-batches 1` 的主动检查点中断：第一次运行保留待办与游标，第二次续跑后待办清空且版本继续增长。

## 快照陈旧拒绝

对已完成的版本 13 快照，改用阈值 `0.90` 的 `dedup_threshold_090.json` 恢复：

```text
status               stale
failure              E_SNAPSHOT_STALE
旧快照版本           13
恢复后 CURRENT 版本  13（未修改）
原因                 dedup configuration hash changed; old signatures/groups are stale
```

另用 `--expected-version 14` 对版本 13 恢复，同样得到 `E_SNAPSHOT_STALE`，旧快照未被覆盖。

## 增量失效与复用

先使用默认阈值 `0.72` 完成一次运行，再以新 `run_id`、相同签名缓存运行阈值 `0.90` 配置：

```text
签名命中       36
签名未命中     0
复用来源       books, papers, web
消耗预算       0
```

随后改为 `shingle_length=4`、`signature_dim=32`：

```text
签名命中              0
签名未命中            36
invalidated_by_params 36
```

内容变更测试修改默认采样的第一篇文档文本，再以相同参数启动新运行：

```text
签名命中              35
签名未命中            1
invalidated_by_content 1
miss_sources          {源: 1}
```

即只有被修改文档的签名失效，其他来源和文档继续复用。

同一场景也可直接 `--resume` 已有完整快照。此时不会整体废弃快照：旧文档点数被回收（`budget.reclaimed_points` 等于该文档原扣减点数），仅该文档出现一次新的签名未命中与新扣减记录，其余已提交签名继续复用，最终运行仍为 `complete`。

## 异常隔离

在 MinHash 阶段对单篇文档注入真实运行时异常后，调度器继续处理其余文档：

```text
成功签名文档           2
失败记录               1
失败阶段               minhash
业务码                 E_MINHASH_FAILED
```

失败文档不进入签名集合与去重结果；其余文档的签名、预算扣减和顺序保持稳定。

## 生成产物位置

```text
output/reports/<run_id>.json             最新报告
output/reports/<run_id>-v000008.json     指定版本报告
output/results/<run_id>.json             结果报告副本
output/snapshots/<run_id>/CURRENT        原子提交指针
output/snapshots/<run_id>/versions/...   各阶段完整快照
output/cache/signatures/...              内容寻址签名缓存
output/cache/group_decisions/...         重复组决断缓存
```

`output/` 已被 `.gitignore` 忽略；运行 `examples/run_demo.py` 可在任意机器上重建相同结果。
