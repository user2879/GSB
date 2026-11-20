# 运行报告（实测结果）

- 运行环境：Python 3.11（要求 >= 3.10）、NumPy 1.26.4、PyYAML；纯本地、无外部服务、无网络访问
- 复现种子：`20261121`（整数 LCG，跨平台一致）
- 复现命令：
  ```bash
  pip install -r requirements.txt
  python scripts/generate_inputs.py --tier small --out data
  python run_pipeline.py --data data --out out/small --serial
  python run_pipeline.py --data data --out out/small --mode resume
  python run_tests.py
  ```

## 1. 单元测试

`python run_tests.py` → **17 passed**（stdlib unittest，无第三方测试框架依赖）

| 测试类 | 覆盖点 |
|---|---|
| FixedPointTests | 半向上量化、负数量化、非法位宽/档位 |
| LoaderTests | 事件 7 类非法条目、依赖环 E_DEF_CYCLE、参数非法 E_DEF_PARAM |
| ComputeTests | 窗口含边界、事件ID字典序累加、count/sum/mean/last、三类空窗来源 |
| BudgetTests | 全局+单特征双层预算、账本 seq 连续、待处理队列、配额上限 |
| CacheTests | LRU 淘汰、同访问序号字典序决断、淘汰后重算 |
| SnapshotTests | E_SNAPSHOT_STALE 版本漂移识别 |
| IncrementalTests | 精确失效集合、增量/全量等价、回填顺序拒绝、定义-only 精确复用 |
| ConcurrencyTests | 并发 1 vs 8 的特征/样本/账本/失败/缓存注册表完全相等 |
| InjectionTests | 非有限上下文/未知行为注入隔离，注入编号命中记录 |
| RecoveryTests | 断点恢复等价、篡改清单被拒绝 |

## 2. small 档位端到端（120 事件，默认验收）

- 增量 vs 全量等价：**true**，`first_diff = null`
- 串行 vs 并发（1/4/8）：features/samples/ledger/failures 全部相等
- 快照恢复：`status = resumed`，恢复点 `resumed_from_seq = 99`，恢复产物与全量逐文件哈希一致
- 快照过期自检：成功识别并输出 `E_SNAPSHOT_STALE`

产物计数（`out/small/full/diagnostics.json`）：

| 指标 | 实测值 |
|---|---|
| 特征行数 | 900 |
| 样本总数 / 就绪 / 待定 | 117 / 114 / 3 |
| 预算账本条目 | 1071 |
| 失败清单条目 | 18 |
| 缓存 命中/未命中/淘汰/重算 | 300 / 300 / 900 / 300 |

业务码命中分布：

| 业务码 | 次数 | 说明 |
|---|---|---|
| E_INJECTION_HIT | 3 | INJ-001/002/003 全部命中 |
| E_EVENT_INVALID | 2 | 注入的非有限上下文、未知行为被隔离 |
| E_BACKFILL_ORDER | 1 | EDIT-000 回填时间戳早于游标，拒绝写入 |
| E_FEAT_BUDGET_EXHAUSTED | 3 | 单特征配额耗尽（含上游级联） |
| E_GLOBAL_BUDGET_EXHAUSTED | 6 | 全局预算耗尽，保留待处理队列 |
| E_SAMPLE_PENDING | 3 | 判定时刻特征不可用，样本待定 |

逐文件哈希等价性（final_full vs incremental，SHA-256 前 16 位相同）：
`features.jsonl`、`samples.jsonl`、`budget_ledger.jsonl`、`failures.jsonl`、`cache.json` 全部 MATCH；
断点恢复 resumed vs full 同样全部 MATCH。

精确失效验证：
- 仅定义变更（窗口修改）场景下，不相关的 `u_count`、`i_count` 进入复用集合、复用块数 > 0
  （见 `test_definition_only_change_preserves_unrelated_features`）。
- 含事件回填的完整编辑序列中，回填会平移同用户/同物品后续窗口，
  相关基础特征及其全部下游被精确失效，失效集合与全量重建差异集一致。

## 3. 其他档位

| 档位 | 事件 | 特征行 | 样本 | 账本 | 重算 | 等价 | 耗时(参考) |
|---|---|---|---|---|---|---|---|
| smoke | 24 | ~200 | 含空窗查询 | 连续 | >0 | true | <1s |
| small | 120 | 900 | 117 | 1071 | 300 | true | ~2s |
| medium | 600 | 4500 | 597 | 5391 | 1500 | true | ~16s |

## 4. 需求覆盖结论

需求 1–8 全部满足并由自动化测试 + 三档端到端运行验证：
加载校验隔离、拓扑增量定点计算、精确失效与复用、标签延迟样本、
双层预算可审计账本、LRU 淘汰重算、增量/全量逐字段等价、
并发与串行一致、快照原子提交/过期识别/断点恢复、异常注入隔离。

## 5. 已知边界（设计内权衡，非缺陷）

- 窗口检索在单实体事件量很大时为 O(n²) 全量重算语义，medium(600) 约 16s，
  侧重确定性与可审计性而非吞吐；可在此基础上引入分块索引优化。
- 缓存驻留按“计算点内共享上游块”建模，块不跨点驻留（滑动窗口随锚点变化），
  LRU 注册表与淘汰/重算遥测完整保留。
- 增量复用在“仅定义变更”时可直接复用无关特征；事件回填因窗口平移需重算受影响实体链，
  这是保证状态一致性的必要精确范围。
