# GSB 定点否定作用域账本设计

## 1. 数据与阶段

样本由固定种子 `383` 生成，入口为 `gsb/samplegen.py`，输出到 `samples/`：

- `controlled_vocabulary.json`：64 个受控词项，含情感词极性、证据分和标点标记。
- `tickets.json`：18 张中文客服工单，含工单 ID、词项数组、渠道、标注员版本和完结状态。
- `negation_triggers.json`：否定触发词及稳定条序。
- `polarity_conflicts.json`：极性冲突词及稳定条序。
- `channel_weights.json`：千分比渠道可信度。
- `config.json`：作用域上限、抵消强度、惩罚系数、阈值、定点位宽/档位、容差、预算和并发数。
- `edits.jsonl`：6 条确定性编辑命令流。

处理阶段为：

1. `schema.py`：加载与聚合校验，失败码 `E_SCHEMA_VIOLATION`。
2. `scanner.py`：字面命中、作用域生成、唯一生效作用域和压制依据。
3. `settlement.py`：定点抵消、饱和轨迹、作用域隔离和长度惩罚。
4. `composition.py`：渠道加权、情感极性、风险等级、边界标记和排序。
5. `reference.py`：NumPy `float64` 无量化参考路径与逐级误差归因。
6. `cache.py` / `incremental.py`：内容指纹缓存键、失效集合和复用账本。
7. `scheduler.py`：点数预算、待办队列、稳定顺序和有限并发。
8. `snapshot.py`：阶段产物、临时目录、原子 symlink 提交和恢复判定。

## 2. 作用域裁决

作用域从否定触发词位置开始，包含触发词本身及后续连续非标点词项，遇到标点或达到 `scope_max_tokens` 后结束。

命中落入多个作用域时，候选键为：

```text
(scope.start ASC, -scope.end DESC, trigger.order ASC)
```

即需求中的“起点升序、终点降序、触发词条序升序”。第一项是唯一生效作用域；其余候选逐命中写入 `suppressions`，包含被压制作用域、选中作用域、双方排序键和规则说明。

样本中：

- `TK-0001` 是嵌套作用域：`S001=[1,7)` 与 `S003=[3,7)`，两个命中均归 `S001`，产生 2 条压制依据。
- `TK-0015` 是交叠作用域：`S000=[0,6)` 与 `S003=[3,9)`，交叠区命中按起点优先归 `S000`，产生 1 条压制依据。

## 3. 定点结算

设定点倍数 `Q=10**decimal_width`。词项证据、抵消强度、下界和阈值都量化为整数 q。每个作用域内命中独立按冲突词 `(conflict.order ASC, position ASC)` 逐步结算：

```text
requested = current_q - cancellation_strength_q
next      = max(lower_bound_q, requested)
saturated = requested < lower_bound_q
```

每次抵消都保留 `before_q`、`requested_after_q`、`after_cancellation_q`、`saturated` 和 `saturation_position`。被抵消到零下界的命中不删除，`canceled_to_zero=true` 且轨迹完整。

长度惩罚使用整数万分比：

```text
retention_bp = max(length_min_retention_bp,
                   10000 - (scope_length - 1) * length_penalty_bp_per_token)
after_penalty_q = round_ratio(after_cancellation_q, retention_bp, 10000)
```

作用域外命中使用空冲突列表和 `10000bp` 保留率，不读取任何作用域内抵消状态，因此天然隔离。验收中的 `TK-ISO` 证明作用域外“延迟”的证据分不变。

## 4. 工单结论与边界规则

每个命中按极性生成带符号证据：

```text
signed_q = polarity_sign * after_length_penalty_q
weighted_q = round_ratio(sum(signed_q), channel_weight_permille, 1000)
risk_score_q = max(0, -weighted_q)
```

极性阈值为正负 `positive/negative_threshold_points`；风险阈值为 `medium < high < critical`。比较使用包含阈值的 `>=` / `<=`。

边界固定规则：

- 量化后恰等阈值：包含该等级。
- 位于报告档位（默认 Q2）半个量化步内但定点比较尚未跨阈值：维持保守判定，同时写 `boundary=true` 和 `boundary_checks`。
- 每个检查记录阈值 q、观测 q、是否恰等、是否在半步内和规则文本。

同分时结论排序键为：

```text
(-risk_rank, -risk_score_q, weighted_sentiment_q, ticket_id ASC)
```

结论内嵌 `referenced_scope_ids` 与完整 `cancellation_trails`。

## 5. float64 真值与误差传播

`reference.py` 用 `numpy.float64` 重放同一套作用域、抵消、惩罚和加权公式，但不做定点量化。四个环节的逐工单差值键为：

- `scope_assignment`
- `cancellation`
- `length_penalty`
- `composition`

每个工单按上述顺序寻找首个 `diff > tolerance` 的环节，写入 `first_exceeded_stage`。

双档位比较运行 Q6 fine 与 Q2 coarse。若风险等级不同：

1. 生效作用域归属指纹不同：`scope_attribution`。
2. 饱和事件数或归零状态不同：`cancellation_saturation`。
3. 否则由边界/阈值比较可解释：`threshold_comparison`。

验收样本中 `TK-0002` 在 Q6 为 `low`、Q2 为 `medium`，归因为 `threshold_comparison`。另外通过受控参数变体分别构造了 `scope_attribution=TK-0002`、`cancellation_saturation=TK-0003` 的真实翻转，三类来源贡献计数各为 1。

## 6. 指纹缓存与账本

缓存键分三层：

- `hit:<ticket_content_fp + relevant_vocab_fp + gear>`
- `settlement:<scope_id + scope_key + relevant_conflict_fp + cancellation_param_fp + penalty_param_fp + gear>`
- `conclusion:<ticket_full_fp + settlement_key_fps + composition_param_fp + channel_fp + metadata_fp + gear>`

否定/冲突依赖指纹只包含该工单或作用域实际出现的词条，因此修改一个触发词只会使引用该 `trigger_id` 或相关词条出现位置的作用域失效。全局 `global_version` 与 `config_hash` 用于快照版本识别，但不替代细粒度依赖指纹。

每次编辑输出：

- `affected_ticket_ids`
- `reused_ticket_ids`
- 逐条 `invalidated` / `recomputed` / `reused` 账本
- 旧/新缓存键和失效依据

验收结果：

- 替换 `TK-0004`：仅 `TK-0004` 失效，其余 17 张直接复用。
- 修改 `NEG_BIE` 条序：仅引用“别”的 `TK-0003` 失效，其余 17 张复用。
- 6 条样例编辑命令后，增量结论、扫描和结算与编辑后空白全量重建逐字段一致。

## 7. 预算与并发一致性

作用域点数：

```text
point_cost = covered_token_count * attributed_hit_count
```

调度队列稳定键为 `(ticket_id ASC, scope_start ASC, scope_id ASC)`。调度器按队首顺序消耗预算；下一项放不下时，该项及其后队列保留为 pending，并写 `E_SCOPE_BUDGET_PENDING`。下一批把旧 pending 放在新工作之前。

`BudgetScheduler` 使用 `ThreadPoolExecutor(max_workers=min(config.max_workers, eligible_count))`，并通过锁记录实际峰值。返回结果再次按稳定键排序，因此线程完成顺序不会改变结论或账本顺序。测试覆盖 `workers=1/2/4/8`、`budget=24/1000`，结论和 `scope_ledger` 均与串行一致。

## 8. 快照与恢复

`SnapshotStore.commit_state` 先在 `.staging-*` 临时目录写：

- `01_validation.json`
- `02_hit_scan.json`
- `03_scope_settlement.json`
- `04_composition.json`
- `05_error_report.json`
- `queue.json`
- `manifest.json`

所有文件 fsync 后，临时目录 `os.replace` 为不可变快照目录，再用原子 symlink 替换 `current`。快照目录名和 manifest 均包含 `global_version` 与 `config_hash`；配置哈希不一致时抛 `E_SNAPSHOT_VERSION_MISMATCH`。

恢复只读取 `current/manifest.json` 指向的已提交快照。若 `current` 不存在、manifest 损坏或状态不是 committed，抛出 `E_SNAPSHOT_UNAVAILABLE`，CLI 返回码 4，不承诺恢复。验收分别从 `hit_scan` 与 `composition` 快照续跑，结果与空白全量重建 JSON 逐字段一致。

## 9. 常用命令

```bash
# 若 Debian/Ubuntu 缺少 ensurepip，可使用机器已装的 virtualenv；标准机器优先：
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

python -m compileall -q main.py gsb tests
python -m unittest discover -s tests -v
python -m gsb.cli verify
python -m gsb.cli run --budget 1000 --workers 4
python -m gsb.cli resume-snapshot --snapshot-dir artifacts/snapshot-demo
```

最新实际输出保存在 `artifacts/validation-results.txt`，结构化验收报告保存在 `artifacts/acceptance-report.json`。
