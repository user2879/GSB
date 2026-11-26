# 意图路由复核 — 技术设计说明

## 1. 总体架构

包源码位于 `src/intent_router/`，按职责拆分为无环依赖的模块：

| 模块 | 职责 |
| --- | --- |
| `fixedpoint.py` | 定点整数域、量化档位、round-half-up 舍入、半步判定 |
| `errors.py` | `E_*` 业务码与不中断式失败收集器 |
| `models.py` | 文本、触发词操作、修订、有效条目、配置、修订链等数据模型 |
| `fingerprints.py` | 确定性 SHA-256 内容指纹与缓存键派生 |
| `loader.py` | 语料/修订链/配置加载与全部规则校验 |
| `library.py` | 修订链逐级合成有效词库、最长匹配与抑制清单 |
| `scoring.py` | 定点意图打分、阈值/半步边界标记与唯一决断 |
| `dedup.py` | 子词集合定点 Jaccard、聚类与代表选择 |
| `ledger.py` | 缓存条目、失效依据、复用命中的可复核账本 |
| `scheduler.py` | 全局/通道/批次三重预算与线程池调度、待办保留 |
| `attribution.py` | 翻转归因（权重/量化/阈值/去重）与数值证据 |
| `snapshot.py` | 临时目录暂存、版本与配置哈希门控、原子整体替换 |
| `engine.py` | 指纹失效 + 匹配 + 打分 + 去重的增量流水线 |
| `pipeline.py` | 批次编排、命令流应用、快照提交、断点续跑、全量重建 |
| `statistics.py` | numpy 定点分布与千分比份额统计 |
| `report.py` / `cli.py` | 诊断报告与命令行入口 |

## 2. 定点与量化语义

所有决策仅使用 Python 整数，保证跨平台、跨解释器、跨线程位级一致。

- 定点数：`raw / 2^fraction_bits`，`fraction_bits ∈ {4, 8, 16}`。
- 量化档位：`gear ∈ {1, 2, 4}`，把 `raw` 舍入到 `gear` 的整数倍。
- 舍入规则：确定性 round-half-up（半数恒向正无穷）。
- 评分（按意图标识升序、条目标识升序固定累加）：
  `raw = Σ(weight × hits × SCALE) − length × (SCALE/8 + 1)`。
  奇数偏移 `+1` 保证量化半步在整数域可达。
- 边界标记：参考分恰等于意图阈值，或落在量化半步，或多意图同分时标记。
- 权威结论只取量化后的定点整数；参考分仅用于归因证据展示。
- 同分决断：量化分最高者，平局按意图标识升序；去重代表平局按文本标识字典序。

## 3. 修订链与最长匹配

- 每个修订显式声明父修订；加载时校验父修订存在且从 head 回溯无环。
- 合成从 root 向 head 顺序应用：`upsert` 覆盖词项/权重/意图并记录
  `born_revision`、`last_revision` 与 `overrides`；`delete` 移除条目。
- 匹配优先级：跨度更长优先 → 同长按词项字典序 → 再按起始位置 → 条目号兜底。
- 与已接受候选重叠者进入抑制清单，记录 `swallowed_by` 与原因。

## 4. 近重复去重

- 相似度为子词集合的定点整数 Jaccard，单位千分比：
  `sim = |A∩B| × 1000 // |A∪B|`。
- `sim ≥ 阈值` 即同簇；精确有理相等且恰等于阈值时额外标记边界。
- 簇代表取定点综合分最高者，平局按文本标识字典序；被剔除文本记录
  代表标识与相似度。未参与重复簇的文本意图结论不被改变。

## 5. 指纹、精确失效与复用

- 文本、触发词条目、修订链、量化档位分别产生内容指纹。
- 有效词库、逐文本命中、意图结论、去重归属各自派生缓存键，并记录其引用的
  触发词条目指纹集合与文本指纹。
- 失效判定仅在引用指纹变化时发生：
  文本删除/替换、引用条目增删改、修订链变化、档位/位宽变化。
- 未受影响文本直接命中持久化缓存，并逐条写入复用记录；每条失效记录原因
  与变化指纹。

## 6. 预算调度与并发确定性

- 三重约束：全局候选匹配点数预算、单通道配额、每批重算文本上限。
- 单条文本成本 `match_points = 词项数 + 1`。
- 预算裁决单线程按 `(通道, 文本标识)` 稳定顺序进行，待办项下一批优先；
  超预算以 `E_TEXT_BUDGET_EXCEEDED` / `E_CHANNEL_QUOTA_EXCEEDED` 入待办。
- 仅纯计算进入 `ThreadPoolExecutor`，`max_workers` 严格取配置；结果按稳定
  顺序回收，因此任意并发度输出逐字段相同。

## 7. 翻转归因

对同一文本的前后结论，按优先级输出唯一来源并附高位宽参考分、定点分与差值：

1. `DEDUP_CLUSTER_MERGE`：代表/抑制归属变化；
2. `THRESHOLD_COMPARISON`：参考分不变但阈值资格变化；
3. `FIXED_POINT_ROUNDING`：参考分不变而量化结果变化；
4. `TRIGGER_WEIGHT_CHANGE`：触发词条目变更移动了参考分。

最终结论始终以定点整数结果为准，参考分只作证据。

## 8. 快照与恢复

- 各阶段产物、待办队列与已消耗预算先写临时目录，再 `os.replace` 原子替换。
- 提交前以全局版本号与配置哈希门控，版本冲突直接拒绝。
- `Pipeline.resume_from_snapshot` 仅从最近已提交快照还原账本、结论与待办，
  强制终止后可续跑，且结果与一次完整运行逐字段一致（见 `tests/test_recovery.py`）。

## 9. 验证方式

```bash
# 环境
virtualenv -p python3.11 .venv
.venv/bin/pip install numpy pyyaml

# 确定性样本（固定整数种子）
PYTHONPATH=src .venv/bin/python scripts/generate_samples.py --out data

# 能力演示
PYTHONPATH=src .venv/bin/python scripts/demo.py

# 全量运行 / 增量重放 / 一致性
PYTHONPATH=src .venv/bin/python -m intent_router.cli run --data data
PYTHONPATH=src .venv/bin/python -m intent_router.cli replay --data data
PYTHONPATH=src .venv/bin/python -m intent_router.cli consistency --data data

# 单元与集成测试
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```
